"""Phase C, 6a: per-cell morphology, with 3DMorph's definitions and not its arithmetic.

The definitions are inherited **exactly**, because they are peer-reviewed and citable, and
because they are unusual enough that a library's same-named statistic would silently
redefine them while the column headings stayed put (`DESIGN_RATIONALE.md` §4.12):

- **branch length** is the arc length of the shortest skeleton path from an *endpoint back to
  the soma* — so **branch count = endpoint count**, and the shared trunk is counted in every
  branch. It is *not* inter-branchpoint segment length.
- **branch hierarchy** is the per-voxel count of overlapping endpoint paths, capped at 4.
- **ramification index** is convex territorial volume over cell volume.

Two 3DMorph defects are fixed here rather than ported (`DESIGN_RATIONALE.md` §5.1–5.2), and
both are invisible on near-isotropic data, which is why they survived peer review:

1. **`NearestPixel` ignores `zscale`** — its `dist` expression multiplies the z term by `1`
   while converting x and y to microns, so the soma attachment point is biased toward staying
   in-plane. Every branch length is measured from that point. Fixed by a `cKDTree` query in
   physical µm.
2. **`ConnectPointsAlongPath` minimises hops, not distance** — it grows wavefronts with
   `convn(mask, ones(3,3,3))`, so all 26 neighbours cost one step even though a z-hop covers
   5.5× more physical distance at this anisotropy. Paths route preferentially through z and
   lengths are overestimated. Fixed by weighting every edge with its true µm length and
   running one Dijkstra per cell.

The second fix also makes the code shorter and faster than the loop it replaces.

**A third, found 2026-08-31 and not inherited from anywhere — the thinning itself ran in
voxel space.** Both fixes above weight a graph correctly in microns, but a correctly weighted
graph over the wrong skeleton is still wrong: `skimage.skeletonize` has no `spacing`, so at
5.5:1 it found the medial axis of a distorted cell. One process of true length 9.00 µm read
between **5.40 and 11.95 µm depending on its orientation**. The mask is now resampled to an
isotropic grid before thinning (`SKELETON_TARGET_UM`, `resample_isotropic`), which brings the
orientation spread to 5.7% — that of an ideal isotropic measurement. 3DMorph has the same
defect and it is invisible on near-isotropic data, like the other two.

**Nothing here returns a silent zero.** 3DMorph wraps its per-cell loop in a `try...catch`
that does nothing, over pre-zeroed arrays, so a cell that failed reports 0 endpoints
indistinguishably from a real measurement (`Script_3DMorph.m:643`, `832-835`). Every measure
here is `NaN` with a `status` explaining why.

Built without `skan` — see `docs/COMPONENT_06_PLAN.md` §2. It could only have supplied
skeleton-to-graph extraction and spur pruning, both short to write, and its segment
statistics measure something different under the same name.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull

from .skeleton import (LOOP_PRUNE_DEFAULT, NODE_ROOT, SKELETON_TARGET_UM, CellSkeletons,
                       SkeletonArtefact, build_skeletons, resample_isotropic)

__all__ = [
    "SkeletonGraph",
    "CellMorphology",
    "SHOLL_STEP_UM",
    "SKELETON_TARGET_UM",
    "build_skeleton_graph",
    "prune_spurs",
    "resample_isotropic",
    "sholl_profile",
    "measure_cell",
]

#: Cap on the branch hierarchy, from 3DMorph: primary, secondary, tertiary, quaternary.
MAX_HIERARCHY = 4

#: Shell spacing for Sholl analysis, in microns. **A documented choice, not a silent
#: constant** (`COMPONENT_06_PLAN.md` §8.2). The literature uses 1-5 µm; our voxel is 0.108 µm
#: in xy against 0.6 µm in z, so anything >= 1 µm sits safely above the sampling in every
#: direction. 2.0 is the middle of the published range.
SHOLL_STEP_UM = 2.0


def sholl_profile(radii: np.ndarray, edges: tuple[np.ndarray, np.ndarray],
                  step_um: float = SHOLL_STEP_UM) -> dict[float, int]:
    """Intersections per shell, counted as **edges that cross the shell**.

    An intersection at radius *r* is a skeleton edge with one end inside *r* and one end
    outside. Counting edges rather than voxels is what makes the measure **topological rather
    than metric**, and that is what lets it survive 5.5:1 anisotropy: a process running along
    z is expressed in ~1/5.5 as many voxels as the same physical length running across xy, but
    it still crosses each shell exactly once.

    ``radii`` is one radius per skeleton node — Euclidean from the soma centroid, or geodesic
    along the skeleton. ``edges`` is ``(u, v)`` node-index arrays, each undirected edge once.

    Shells run ``step, 2*step, ...`` out to the furthest node, so the profile ends where the
    arbour does. Non-finite radii (nodes in a component disconnected from the origin) sit
    outside every shell and therefore contribute no crossings, which is the correct reading:
    an unreachable fragment is not an intersection.
    """
    u, v = edges
    finite = radii[np.isfinite(radii)]
    if finite.size == 0 or u.size == 0:
        return {}
    step = float(step_um)
    n_shells = max(1, int(np.ceil(float(finite.max()) / step)))
    out: dict[float, int] = {}
    ru, rv = radii[u], radii[v]
    for k in range(1, n_shells + 1):
        r = step * k
        out[round(r, 6)] = int(((ru <= r) != (rv <= r)).sum())
    return out


def _offsets() -> list[tuple[int, int, int]]:
    """The 13 forward half of the 26-neighbourhood — each edge visited once."""
    all_off = [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
    return [o for o in all_off if o > (0, 0, 0)]


def _paired_slices(shape: Sequence[int], off: Sequence[int]):
    """Slice pair addressing every voxel and its neighbour at ``off``."""
    src, dst = [], []
    for extent, delta in zip(shape, off):
        if delta >= 0:
            src.append(slice(0, extent - delta))
            dst.append(slice(delta, extent))
        else:
            src.append(slice(-delta, extent))
            dst.append(slice(0, extent + delta))
    return tuple(src), tuple(dst)


def neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    """How many of the 26 neighbours of each skeleton voxel are also skeleton."""
    kernel = np.ones((3, 3, 3), np.uint8)
    counts = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    return (counts - skeleton.astype(np.uint8)) * skeleton


@dataclass
class SkeletonGraph:
    """Skeleton voxels as a graph whose edge weights are true physical distances."""

    coords: np.ndarray                  # (n, 3) voxel coordinates, (z, y, x)
    graph: coo_matrix                   # (n, n) symmetric, weights in µm
    index: np.ndarray                   # volume-shaped; voxel -> node id, -1 elsewhere

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])


def build_skeleton_graph(skeleton: np.ndarray,
                         voxel_size_um: Sequence[float]) -> SkeletonGraph:
    """Graph over skeleton voxels, 26-connected, weighted in microns.

    The weighting is the whole point. 3DMorph's wavefront growth charges one step for every
    neighbour, so at 5.5:1 anisotropy a z-hop buys 5.5× the physical distance of an xy-hop
    at the same price — paths route through z and branch lengths inflate. Here a diagonal
    edge costs exactly ``sqrt((dz*z_um)^2 + (dy*y_um)^2 + (dx*x_um)^2)``.
    """
    z_um, y_um, x_um = (float(v) for v in voxel_size_um)
    coords = np.argwhere(skeleton)
    n = coords.shape[0]
    index = np.full(skeleton.shape, -1, dtype=np.int64)
    if n:
        index[tuple(coords.T)] = np.arange(n)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for off in _offsets():
        src, dst = _paired_slices(skeleton.shape, off)
        a, b = index[src], index[dst]
        both = (a >= 0) & (b >= 0)
        if not both.any():
            continue
        weight = float(np.sqrt((off[0] * z_um) ** 2 + (off[1] * y_um) ** 2
                               + (off[2] * x_um) ** 2))
        u, v = a[both], b[both]
        rows.append(u)
        cols.append(v)
        data.append(np.full(u.size, weight))

    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        d = np.concatenate(data)
        # symmetric: dijkstra is run undirected, but an explicit pair keeps the matrix
        # meaningful to anything else that reads it
        graph = coo_matrix((np.concatenate([d, d]),
                            (np.concatenate([r, c]), np.concatenate([c, r]))),
                           shape=(n, n))
    else:
        graph = coo_matrix((n, n))
    return SkeletonGraph(coords=coords, graph=graph, index=index)


def prune_spurs(skeleton: np.ndarray, voxel_size_um: Sequence[float],
                min_spur_um: float) -> np.ndarray:
    """Iteratively remove short dead-end branches, as `SlimSkel3D` does.

    3DMorph's `THR = 100` is **in voxels** and must not be copied as a number: its sample
    data is near-isotropic while ours is 5.5:1, so the same count means a very different
    physical length along z than across xy. The rule is ported; the length is in microns.

    Iterated to convergence because removing one spur can expose another — a branch point
    that loses a limb may become an ordinary path voxel, turning its neighbour into a new
    dead end.

    **A one-voxel stub survives each pruned spur.** The voxel where a spur meets the trunk
    is 26-adjacent to three trunk voxels, so it has degree 3 and is classified as a junction
    rather than as spur. It is left in place deliberately rather than special-cased: it is
    no longer an *endpoint*, so it contributes no branch and cannot affect branch count or
    any branch length. The only quantity it touches is ``total_skeleton_um``, by one voxel
    edge per pruned spur.
    """
    if min_spur_um <= 0:
        return skeleton.copy()

    current = skeleton.copy()
    while True:
        counts = neighbour_count(current)
        endpoints = current & (counts == 1)
        branch = current & (counts >= 3)
        if not endpoints.any() or not branch.any():
            return current

        # Segments are what remains once branch points are removed; a spur is a segment
        # holding an endpoint.
        segments, n_seg = ndimage.label(current & ~branch,
                                        structure=np.ones((3, 3, 3), bool))
        if n_seg == 0:
            return current

        graph = build_skeleton_graph(current, voxel_size_um)
        doomed = np.zeros_like(current)
        for seg_id in range(1, n_seg + 1):
            seg = segments == seg_id
            ends = seg & endpoints
            if not ends.any():
                continue                       # an internal link, not a dead end
            nodes = graph.index[seg]
            nodes = nodes[nodes >= 0]
            if nodes.size < 2:
                length = 0.0
            else:
                start = graph.index[tuple(np.argwhere(ends)[0])]
                reach = dijkstra(graph.graph.tocsr(), directed=False, indices=start)
                finite = reach[nodes]
                length = float(finite[np.isfinite(finite)].max())
            if length < min_spur_um:
                doomed |= seg

        if not doomed.any():
            return current
        current = current & ~doomed


#: Below this a mesh is dominated by discretisation rather than shape, so surface area and
#: sphericity are reported `NaN`. A **numerical-validity guard, not a scientific threshold**
#: — it is not one of the parameters deliberately left for the user.
MIN_MESH_VOXELS = 27

# The isotropic resampling, the thinning guard and the target constant all MOVED to
# `skeleton.py` on 2026-09-01 when Phase C was split into skeletonisation, measurement and
# export. They are imported above rather than duplicated: two definitions of
# `SKELETON_TARGET_UM` would be two answers to the same question.


def surface_mesh(mask: np.ndarray, voxel_size_um: Sequence[float]):
    """Closed triangular surface of ``mask``, in **physical microns**.

    ``spacing=voxel_size_um`` is what makes this comparable to Imaris and correct under
    anisotropy. The alternative — counting exposed voxel faces — overestimates a smooth
    object's area by roughly half (the staircase problem), and here that error would be
    *direction-dependent*: at 5.5:1 the z faces and the xy faces of one object inflate by
    different amounts. Since sphericity is an area-over-volume ratio, that error propagates
    straight into it.

    The mask is **padded by one voxel** first. Without it an object touching the array border
    meshes as an open surface, and the missing face silently understates the area — worst for
    exactly the large cells that reach the crop edge.

    Returns ``(verts, faces)`` in microns relative to the unpadded frame, or ``None`` when the
    object is too small for a mesh to mean anything.
    """
    from skimage.measure import marching_cubes

    if mask is None or int(mask.sum()) < MIN_MESH_VOXELS:
        return None
    spacing = tuple(float(v) for v in voxel_size_um)
    padded = np.pad(mask.astype(np.uint8), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    except (RuntimeError, ValueError):
        return None
    return verts - np.asarray(spacing, float), faces


def _triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def surface_area_um2(mask: np.ndarray, voxel_size_um: Sequence[float]) -> float:
    """Surface area in µm², meshed per :func:`surface_mesh`. `NaN` when too small."""
    mesh = surface_mesh(mask, voxel_size_um)
    if mesh is None:
        return float("nan")
    return float(_triangle_areas(*mesh).sum())


def _ratio(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, `NaN` on anything that is not a real quotient.

    A zero denominator is not an infinity here — a cell whose soma erosion left nothing has an
    unmeasured soma, not an infinitely large cell — and `inf` in a CSV column poisons every mean
    taken over it while `NaN` is filtered by any tool that reads it.
    """
    n, d = float(numerator), float(denominator)
    if not np.isfinite(n) or not np.isfinite(d) or d <= 0:
        return float("nan")
    return n / d


#: **`span_ratio` and `span_ratio_points` were removed 2026-09-02** with `hull_span_ratio`,
#: their only caller. They took the longest ÷ shortest principal axis from the eigenvalues of
#: the coordinate covariance, scaled to microns first so that 5.5:1 anisotropy could not
#: masquerade as elongation. Recorded here rather than deleted silently, because the microns-
#: first detail is the part worth having if the measure is ever wanted again.
#:
#: `_ratio` **stays** — `cell_to_soma_ratio` and `soma_to_cell_fraction` still call it.


def z_span_um(mask: np.ndarray, voxel_size_um: Sequence[float]) -> float:
    """Extent of the mask along z, in µm, inclusive of both end planes."""
    z = np.flatnonzero(np.asarray(mask).astype(bool).any(axis=(1, 2)))
    if z.size == 0:
        return float("nan")
    return float((int(z[-1]) - int(z[0]) + 1) * float(voxel_size_um[0]))


def distance_to_xy_border_um(mask: np.ndarray, bbox, image_shape,
                             voxel_size_um: Sequence[float]) -> float:
    """Shortest distance from the cell to the frame edge in x or y, in µm.

    **Measured in the whole image, not in the crop**, which is the only reason `bbox` and
    `image_shape` are arguments: a cell mask is stored crop-relative, so a crop-local answer
    would say every cell touches its own edge. 0.0 means the cell reaches the frame, which is
    the same condition `touches_xy_border` reports as a boolean — the two agree by construction
    and this one says by how much.

    z is deliberately excluded. The stack's top and bottom are an optical limit rather than a
    field-of-view edge, and `touches_z_border` already carries that separately.
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return float("nan")
    (z0, z1), (y0, y1), (x0, x1) = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5])) \
        if len(bbox) == 6 else ((0, 0), (bbox[1][0], bbox[1][1]), (bbox[2][0], bbox[2][1]))
    ys = np.flatnonzero(m.any(axis=(0, 2)))
    xs = np.flatnonzero(m.any(axis=(0, 1)))
    vy, vx = float(voxel_size_um[1]), float(voxel_size_um[2])
    top = (int(y0) + int(ys[0])) * vy
    bottom = (int(image_shape[1]) - 1 - (int(y0) + int(ys[-1]))) * vy
    left = (int(x0) + int(xs[0])) * vx
    right = (int(image_shape[2]) - 1 - (int(x0) + int(xs[-1]))) * vx
    return float(max(0.0, min(top, bottom, left, right)))


def sphericity(volume_um3: float, area_um2: float) -> float:
    """Imaris's definition: ``pi**(1/3) * (6V)**(2/3) / A``.

    Reported for plaques (§5.2 of the literature review, where it is a compaction proxy) and
    available for cells. Meaningless without a meshed area — a voxel-face area would depress
    it further still, and anisotropically.

    > **A perfect sphere reads ~0.93 here, not 1.0, and that is not an error.** Marching cubes
    > cannot recover the smooth surface a binary mask samples; it interpolates a faceted one
    > that is systematically larger — measured at **+7.7%** on a 12-voxel-radius ball — and
    > area sits in the denominator. So the attainable ceiling is ~0.93 at this sampling, and
    > it falls further for smaller objects, where discretisation is a larger share of the
    > surface. **Compare objects with each other, never against 1.0**, and never compare
    > sphericity between objects of very different size without saying so.
    > `test_sphere_sphericity_is_the_discretisation_ceiling_not_one` pins this.
    """
    if not np.isfinite(volume_um3) or not np.isfinite(area_um2) or area_um2 <= 0:
        return float("nan")
    return float(np.pi ** (1 / 3) * (6.0 * volume_um3) ** (2 / 3) / area_um2)


@dataclass
class CellMorphology:
    """Every per-cell measure. `NaN` and a status on failure, never a zero.

    A zero endpoint count is a real, meaningful measurement — a cell with no processes.
    Using it to mean "this cell failed" is the defect in §5.3, and it is not repeated.
    """

    status: str = "ok"
    cell_volume_um3: float = float("nan")
    surface_area_um2: float = float("nan")
    sphericity: float = float("nan")
    soma_volume_um3: float = float("nan")
    territorial_volume_um3: float = float("nan")
    ramification_index: float = float("nan")

    #: ---- added 2026-09-01 for `cells_mask.csv` -------------------------------------------
    #: All measurable from material already in hand — the cell mask, the soma mask the collapse
    #: already builds, and the convex hull `territorial_volume_um3` is already computed from.
    #: None needs a new pass over the image.
    soma_surface_area_um2: float = float("nan")
    soma_sphericity: float = float("nan")
    #: The two ratios are kept as **columns rather than left to the analyst** because a ratio
    #: computed in a spreadsheet from two columns on two different sheets is where sign and
    #: units errors live. `soma_to_cell_fraction` is the reciprocal of `cell_to_soma_ratio` and
    #: both are carried at the user's request: the fraction is the form the activation
    #: literature reports, the ratio the form that stays finite as a soma shrinks.
    #:
    #: **There were four until 2026-09-02.** `cell_to_nucleus_ratio` and `soma_to_nucleus_ratio`
    #: went with `nucleus_volume_um3`, their denominator — three columns the user had not
    #: planned, which arrived in a previous session's table. `nucleus_count` on
    #: `cells_review.csv` is a different, older column and is untouched.
    cell_to_soma_ratio: float = float("nan")
    soma_to_cell_fraction: float = float("nan")
    #: Hull shape beyond its volume.
    #:
    #: **`hull_density` was specified and dropped 2026-09-01, `hull_span_ratio` 2026-09-02.**
    #: Cell volume over territorial volume is the reciprocal of `ramification_index`, so the
    #: density would have been a derived column beside its own source; the user then dropped the
    #: span ratio with it, the set having lost its other member. `morphometrics.span_ratio` and
    #: `span_ratio_points` went too — nothing else called them. See `EXPORT_SPEC.md` §5
    #: `cells_mask.csv`.
    hull_surface_area_um2: float = float("nan")
    #: Extent in z, and how close the cell comes to the frame edge in xy. The two border
    #: booleans say *whether* the frame cut a cell; these say *by how much* it might have, which
    #: is what lets truncation be a covariate rather than only an exclusion.
    z_span_um: float = float("nan")
    distance_to_xy_border_um: float = float("nan")
    n_endpoints: int = -1                     # = branch count, by 3DMorph's definition
    n_branch_points: int = -1
    branch_lengths_um: list[float] = field(default_factory=list)
    min_branch_um: float = float("nan")
    max_branch_um: float = float("nan")
    mean_branch_um: float = float("nan")
    total_skeleton_um: float = float("nan")

    #: Skeleton length in **microns** per hierarchy order, keyed 1 (quaternary) to 4
    #: (primary). Length, not the voxel count this used to hold: at 5.5:1 anisotropy a voxel
    #: is not a unit of length, so a z-running primary and an xy-running one of the same
    #: physical extent would report wildly different counts (`EXPORT_SPEC.md` §10.4). The
    #: four orders sum to ``total_skeleton_um``, which is the assertion that catches both a
    #: units error and a double-counting one.
    hierarchy_um: dict[int, float] = field(default_factory=dict)

    #: Sholl profiles: intersections keyed by shell radius in µm. Two metrics, because they
    #: are **different measures rather than one measure in two units** (`EXPORT_SPEC.md`
    #: §10.3). Euclidean is Sholl (1953) and what the literature reports; a tortuous process
    #: can cross one shell repeatedly. Geodesic is monotone along the skeleton, so each
    #: process crosses each shell exactly once and the count *is* the number of paths at
    #: least that long. Their ratio is a tortuosity measure, which is why both are carried.
    sholl_profile: dict[float, int] = field(default_factory=dict)
    sholl_profile_geodesic: dict[float, int] = field(default_factory=dict)

    #: The per-skeleton columns, keyed by their exported names — added 2026-09-01. Dicts rather
    #: than fifty-odd attributes because nearly every name appears on **both** skeleton sheets
    #: and would need an `anchored_`/`free_` prefix declared twice. The prefix belongs where
    #: §1.1's collision rule actually bites, which is a join in R, so the export applies it when
    #: it writes the header. Empty when the artefact does not exist, which is how a cell with no
    #: soma reports no anchored measures without inventing zeros for them.
    skeleton_anchored: dict[str, Any] = field(default_factory=dict)
    skeleton_free: dict[str, Any] = field(default_factory=dict)
    sholl_step_um: float = SHOLL_STEP_UM
    sholl_max_intersections: float = float("nan")
    sholl_critical_radius_um: float = float("nan")
    sholl_geodesic_max_intersections: float = float("nan")
    sholl_geodesic_critical_radius_um: float = float("nan")

    #: Mean depth of the cell's voxels **in the source stack, in µm** — not in its own crop,
    #: see ``measure_cell``'s ``z_offset``. This is the export's only covariate and its whole
    #: job is depth: CD68 and 6E10 attenuate through the 24 µm stack, so intensity carries a
    #: depth bias an analysis has to be able to control for. A crop-relative value would put
    #: every cell at half its own thickness and carry no depth information at all — which is
    #: what it did until 2026-08-12 (`EXPORT_AUDIT.md` §3.1). `NaN` for an empty mask.
    #:
    #: **Microns since 2026-08-12, and named for it (NEW-a).** It was ``mean_z``, in voxel
    #: indices, in an export where every other length is µm — so the one column an analysis is
    #: meant to regress on was the one column in different units, and nothing in its name said
    #: so. At the 0.6 µm z-spacing a bare index is out by **1.67×** and looks entirely
    #: plausible.
    #:
    #: **The conversion is exact and that is why it was approved.** ``mean_z_um`` is a
    #: **single-axis** measure, so multiplying by the z voxel size introduces no anisotropy
    #: inflation. That risk is real only where a measure combines axes; it does not apply here.
    #:
    #: **This is the only mean depth in the codebase.** A second one lived on
    #: `markers.Cd68Measures`, computed correctly and never exported, and that pair — one
    #: name, two meanings, the wrong one wired to the CSV — is how the defect survived.
    mean_z_um: float = float("nan")

    #: The two artefacts this measurement was read off. **Carried, not serialised** — the
    #: geometry goes to its own `.npz` beside the pool, and only the scalar summaries reach the
    #: measurement payload. Excluded from `repr` and comparison because two cells are the same
    #: measurement when their numbers match, not when their node arrays do.
    skeletons: object | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class _CentroidOnly:
    """A stand-in soma for a caller that has a centroid and no region.

    Four scripts call `measure_cell` with ``soma_centroid=`` alone. The collapse needs a
    *region*, so those callers get the nearest-node fallback — and `CellSkeletons` records that
    as ``collapse_status="nearest_node"`` rather than producing a differently-rooted tree that
    nothing distinguishes from a properly collapsed one.
    """

    region = None

    def __init__(self, centroid) -> None:
        self.centroid = centroid


def measure_cell(
    mask: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    soma: object | None = None,
    soma_centroid: Sequence[float] | None = None,
    skeletons: CellSkeletons | None = None,
    min_spur_um: float = 1.0,
    loop_prune: str = LOOP_PRUNE_DEFAULT,
    z_offset: float = 0.0,
    #: ``nucleus_mask`` was **removed 2026-09-02**. It existed only for `nucleus_volume_um3`
    #: and the two ratios that divided by it, and all three columns went.
    bbox: Sequence[int] | None = None,
    image_shape: Sequence[int] | None = None,
) -> CellMorphology:
    """Measure one cell from its mask.

    **Nothing is thinned here since 2026-09-01.** Pass ``skeletons`` from
    `skeleton.build_skeletons` and this reads them; pass nothing and it builds them once, which
    is what keeps the four tuning scripts working unchanged. ``soma`` is a `soma.SomaResult` and
    is what the collapse needs — ``soma_centroid`` remains accepted for those scripts and routes
    to the recorded nearest-node fallback.

    ``soma_centroid`` is in voxel coordinates within ``mask``, as `soma.find_soma` returns
    it. Without one there is no origin for the endpoint→soma paths, so branch measures are
    `NaN` while the volume measures — which need no origin — still report.

    ``z_offset`` is the crop's z origin in the source stack — ``cell.bbox[0]``, **in voxels**,
    which `phasec.measure_one` passes. Every other measure here is translation-invariant and
    does not care where the crop came from; ``mean_z_um`` is the one that does, because it is a
    **depth** and a depth only means something in the stack's own coordinates. The default of
    0.0 keeps a bare `measure_cell(mask)` honest for a caller measuring a whole image, where
    the crop origin genuinely is 0.

    Note the units on each side: ``z_offset`` is a **voxel index**, because that is what a bbox
    holds, and it is added to the crop-local voxel mean **before** the whole sum is multiplied
    by the z voxel size. ``mean_z_um`` is the only µm-valued output that depends on it.

    ``min_spur_um`` became **1.0 on 2026-08-09** and was **re-derived on the corrected
    skeleton on 2026-09-01**, `tuning_runs/spur_sweep_isotropic.txt`, 148 cells. It survives.
    Medians per cell, threshold-2 coefficient 0.05:

    ==========  ============  =========  ===============  ==============  ===============
    min_spur_um  5xFAD ends    WT ends    **WT/5xFAD**     5xFAD skeleton  5xFAD max branch
    ==========  ============  =========  ===============  ==============  ===============
    0.0          56            260        4.64             266.0 um        29.9 um
    0.5          44            220        5.00             262.4           29.9
    **1.0**      **35**        **183**    **5.23**         253.9           **29.9**
    1.5          29            131        4.52             242.2           29.9
    2.0          24            99         4.13             235.3           29.6
    3.0          19            65         3.42             214.7           29.5
    ==========  ============  =========  ===============  ==============  ===============

    The three tests that made 1.0 a choice rather than a point on a curve all still hold, and
    one is now stronger: **the genotype separation peaks at 1.0** (4.64 -> 5.23 from no pruning,
    then falling away), **max branch is untouched to 1.5** so pruning provably reaches only dead
    ends, and **2.0 starts eating real branches** — max branch moves and the ratio drops to
    4.13. At coefficient 0.18 the peak sits at 0.5 (5.26) with 1.0 adjacent at 5.09, so 1.0 is
    within noise of the best on both. **No parameter change.**

    .. warning::

       **What did move is the genotype contrast itself, and it is not yet explained.** On the
       same 23 WT cells the median endpoint count went **79 -> 183** while 5xFAD went 40 -> 35,
       taking WT/5xFAD from ~2.0 to ~5.2. 5xFAD max branch fell 37.0 -> 29.9 µm, which is the
       expected direction — voxel space over-measured shallow processes by up to 38% — and WT
       max branch barely moved (57.7 -> 56.4). But **an endpoint count that more than doubles in
       one arm and not the other is a large change in this study's phenotype measure**, and this
       sweep cannot say whether the isotropic grid is resolving fine WT processes that were
       previously merged, or generating spurious branches on the more convoluted WT masks.
       Pruning does not settle it: at 1.0 the 5xFAD count falls by a *larger* fraction (37%)
       than WT (30%), so the extra WT endpoints are not simply short spurs. **Do not cite the
       endpoint ratio until this is understood** (`PROJECT_LOG.md` §55).

    The superseded voxel-space sweep, over 152 cells, `tuning_runs/spur_sweep_all.txt` — kept
    because the reasoning above is a comparison with it:

    ==========  ============  =========  ===============  ==============  ===============
    min_spur_um  5xFAD ends    WT ends    **WT/5xFAD**     5xFAD skeleton  5xFAD max branch
    ==========  ============  =========  ===============  ==============  ===============
    0.0          58            116        2.00             307.4 um        37.0 um
    0.5          49            96         1.96             305.1           37.0
    **1.0**      **40**        **79**     **1.98**         300.2           **37.0**
    2.0          32            60         1.88             278.7           36.3
    ==========  ============  =========  ===============  ==============  ===============

    Its three tests, in the form they were originally written — the 2026-09-01 re-derivation
    above applies the same three to the corrected numbers and reaches the same value:

    * **The genotype effect is untouched** from 0 to 1.0 — the WT/5xFAD endpoint ratio holds
      at ~2.0 while 31% of branches are removed. An artefact filter that discards a third of
      the data without moving the effect size is removing noise, not signal, and that is not
      guaranteed in advance.
    * **Max branch length is identical** (37.0 -> 37.0; WT 57.7 -> 57.7), so pruning provably
      touches only dead ends, never the endpoint->soma path the measurement is built on.
    * **At 2.0 it starts eating real branches** — the ratio falls to 1.88 and skeleton length
      drops 9% — so 1.0 sits inside a plateau with a visible far edge.

    There is no branch-level ground truth, so this rests on internal consistency. But 0.0 is
    *demonstrably* wrong: it produced the genotype-dependent endpoint optimum that was later
    retracted as a spur artefact. 3DMorph's own `SlimSkel3D THR=100` cannot be inherited —
    it is in **voxels**, and parity showed that at the same 1.0 um it yields 24 endpoints
    isotropically against 128 anisotropically, so no single micron value reproduces it.
    """
    z_um, y_um, x_um = (float(v) for v in voxel_size_um)
    voxel_volume = z_um * y_um * x_um
    out = CellMorphology()

    if not mask.any():
        out.status = "empty_mask"
        return out

    filled = np.argwhere(mask)
    out.cell_volume_um3 = float(mask.sum()) * voxel_volume
    # Stack coordinates, never crop coordinates. The `+ z_offset` is the whole measure: the
    # crop-local mean is ~half the cell's own z-extent for every cell in the pool, top of the
    # stack or bottom, and a covariate that identical carries nothing to control for.
    #
    # **Converted here, at the producer, not at the flattener** (NEW-a, 2026-08-12) — the same
    # decision as the `z_offset` fix above and for the same reason: one place computes it, so
    # the stored payload and every future reader agree without anyone having to remember.
    out.mean_z_um = ((float(filled[:, 0].mean()) + float(z_offset))
                     * float(voxel_size_um[0]))

    # ---- surface and sphericity ------------------------------------------
    # Meshed, never voxel-counted — see `surface_mesh`. Both are `NaN` for an object below
    # `MIN_MESH_VOXELS`, which is a validity guard rather than a measurement choice.
    out.surface_area_um2 = surface_area_um2(mask, voxel_size_um)
    out.sphericity = sphericity(out.cell_volume_um3, out.surface_area_um2)

    # ---- territorial volume and ramification -----------------------------
    # A hull needs four non-coplanar points; a flat or near-linear cell has none, which is
    # a real condition worth naming rather than a crash to swallow.
    physical = filled * np.array([z_um, y_um, x_um])
    try:
        hull = ConvexHull(physical)
        out.territorial_volume_um3 = float(hull.volume)
        if out.cell_volume_um3 > 0:
            out.ramification_index = out.territorial_volume_um3 / out.cell_volume_um3
        #: `ConvexHull.area` is the hull's **surface** area in 3D, not a planar area, and it
        #: comes free with the hull the territory measure already builds.
        out.hull_surface_area_um2 = float(hull.area)
    except Exception:
        out.status = "degenerate_hull"

    # ---- the rest of the mask block, added 2026-09-01 ---------------------
    out.z_span_um = z_span_um(mask, voxel_size_um)
    if bbox is not None and image_shape is not None:
        out.distance_to_xy_border_um = distance_to_xy_border_um(
            mask, bbox, image_shape, voxel_size_um)

    #: The soma region is the collapse's own mask, so this measures **exactly** the object
    #: `soma_volume_um3` is the volume of — no second definition and no second erosion.
    #: **`soma_volume_um3` is read here as well as in `phasec.measure_one`.** It was only ever
    #: set after this function returned, which was harmless while nothing here depended on it —
    #: the four ratios below do, so it has to be in hand before they are computed. The two
    #: assignments are the same value from the same object; `phasec`'s is left in place so a
    #: caller that builds the soma itself is unaffected.
    if soma is not None and np.isfinite(float(getattr(soma, "volume_um3", float("nan")) or
                                              float("nan"))):
        out.soma_volume_um3 = float(soma.volume_um3)

    region = getattr(soma, "region", None) if soma is not None else None
    if region is not None:
        region = np.asarray(region).astype(bool)
        if region.shape == mask.shape and region.any():
            #: Sphericity's denominator must be the volume of **this** mask, not
            #: `soma_volume_um3`, which is the morphological opening and a different object.
            #: Mixing them would give a shape ratio for two different solids.
            region_vol = float(region.sum()) * voxel_volume
            out.soma_surface_area_um2 = surface_area_um2(region, voxel_size_um)
            out.soma_sphericity = sphericity(region_vol, out.soma_surface_area_um2)

    #: Ratios last, so every numerator and denominator above is final. `_ratio` returns `NaN`
    #: rather than `inf` on a zero denominator — an unmeasured soma is not an infinite cell.
    #: The two nucleus ratios were removed 2026-09-02 with their denominator.
    out.cell_to_soma_ratio = _ratio(out.cell_volume_um3, out.soma_volume_um3)
    out.soma_to_cell_fraction = _ratio(out.soma_volume_um3, out.cell_volume_um3)

    # ---- the skeleton measures, read off the ARTEFACTS -------------------
    # Nothing is thinned here any more. `skeleton.build_skeletons` produces both artefacts and
    # this reads them, which is the Phase C stage split of 2026-09-01: skeletonisation, then
    # measurement, then export.
    #
    # **Which artefact feeds which column matters, and it is not the obvious one.** Every field
    # below is 3DMorph-lineage -- branch length is the path from a tip back to the soma, and
    # `branch count = endpoint count` is its defining identity. That identity only holds on the
    # rooted tree, where unreachable material cannot enter. Taking `n_endpoints` from the
    # soma-free artefact instead would reinstate agenda item A2 exactly: a detached fragment
    # adding an endpoint that contributes no branch. The soma-free numbers are carried
    # separately, under their own names, by `phasec`.
    if soma is None and soma_centroid is not None:
        soma = _CentroidOnly(soma_centroid)
    if skeletons is None:
        skeletons = build_skeletons(mask, voxel_size_um=voxel_size_um, soma=soma,
                                    min_spur_um=min_spur_um, loop_prune=loop_prune)
    out.skeletons = skeletons

    tree = skeletons.soma_anchored
    if tree is None or tree.n_nodes == 0:
        if out.status == "ok":
            out.status = skeletons.soma_free.status if skeletons.soma_free.n_nodes == 0 \
                else "no_soma"
        #: **Still fill the free measures on this branch.** A cell with no soma has no anchored
        #: tree and no branch measures, but its soma-free skeleton is exactly as measurable as
        #: any other — that is the entire reason for keeping a second artefact. Returning here
        #: without this call was the difference between "no soma" and "no skeleton at all".
        _fill_graph_measures(out, skeletons, soma=soma, voxel_size_um=voxel_size_um,
                             mask_shape=mask.shape)
        return out

    n = tree.n_nodes
    edge, elen = tree.edge, np.asarray(tree.edge_length_um, dtype=float)
    root = int(np.flatnonzero(tree.node_role == NODE_ROOT)[0]) if \
        (tree.node_role == NODE_ROOT).any() else 0

    # children, from the parent pointers the tree already carries
    children: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(np.asarray(tree.parent, dtype=int)):
        if p >= 0:
            children[p].append(i)

    # geodesic distance from the root, walked outward once
    geodesic = np.full(n, np.inf)
    geodesic[root] = 0.0
    length_to = {int(b): float(w) for (a, b), w in zip(edge, elen)}
    order: list[int] = [root]
    for node in order:
        for kid in children[node]:
            geodesic[kid] = geodesic[node] + length_to.get(kid, 0.0)
            order.append(kid)

    leaves = [i for i in range(n) if not children[i] and i != root]
    out.n_endpoints = len(leaves)
    #: **Junctions are forks of the TREE**, not voxels that happened to reach degree 3. A node
    #: with two or more children is a real branch point; the merged-junction rule that used to
    #: be applied to voxel clumps is now structural.
    out.n_branch_points = int(sum(1 for i in range(n) if i != root and len(children[i]) >= 2))
    out.total_skeleton_um = float(elen.sum())

    out.branch_lengths_um = sorted(float(geodesic[i]) for i in leaves
                                   if np.isfinite(geodesic[i]))
    if out.branch_lengths_um:
        arr = np.asarray(out.branch_lengths_um)
        out.min_branch_um = float(arr.min())
        out.max_branch_um = float(arr.max())
        out.mean_branch_um = float(arr.mean())

    # ---- hierarchy: overlapping endpoint paths, capped at 4 --------------
    # 3DMorph's definition, computed on the tree instead of by re-walking every path: the
    # number of endpoint paths crossing a node is the number of LEAVES IN ITS SUBTREE, which
    # one reverse sweep gives for every node at once.
    traversals = np.zeros(n, dtype=np.int64)
    for i in leaves:
        traversals[i] = 1
    for node in reversed(order):
        p = int(tree.parent[node])
        if p >= 0:
            traversals[p] += traversals[node]
    node_level = np.clip(traversals, 1, MAX_HIERARCHY)
    #: An edge carries as many endpoint paths as the *lesser* of its two ends, so that is its
    #: order. Clipping to at least 1 keeps every edge in some order, which is what makes the
    #: four sum exactly to `total_skeleton_um` -- an identity that catches both a units error
    #: and a double count.
    edge_level = (np.minimum(node_level[edge[:, 0]], node_level[edge[:, 1]])
                  if edge.size else np.zeros(0, dtype=np.int64))
    out.hierarchy_um = {level: float(elen[edge_level == level].sum())
                        for level in range(1, MAX_HIERARCHY + 1)}

    # ---- Sholl, both metrics ---------------------------------------------
    # Near-free: the geodesic radius is the array walked above, and the Euclidean radius is a
    # norm over coordinates the artefact already stores. No new traversal and no distance
    # transform (`EXPORT_SPEC.md` §10.3).
    root_um = np.asarray(tree.node_um[root], dtype=float)
    euclidean = np.linalg.norm(np.asarray(tree.node_um, dtype=float) - root_um, axis=1)
    edges_uv = ((edge[:, 0], edge[:, 1]) if edge.size
                else (np.zeros(0, np.int64), np.zeros(0, np.int64)))
    out.sholl_step_um = float(SHOLL_STEP_UM)
    out.sholl_profile = sholl_profile(euclidean, edges_uv, SHOLL_STEP_UM)
    out.sholl_profile_geodesic = sholl_profile(geodesic, edges_uv, SHOLL_STEP_UM)
    for profile, peak, radius in (
        (out.sholl_profile, "sholl_max_intersections", "sholl_critical_radius_um"),
        (out.sholl_profile_geodesic, "sholl_geodesic_max_intersections",
         "sholl_geodesic_critical_radius_um"),
    ):
        if profile:
            best = max(profile.values())
            setattr(out, peak, float(best))
            # The innermost radius attaining the peak — the convention, and the one that
            # does not drift outward when a plateau is flat.
            setattr(out, radius, float(min(r for r, n in profile.items() if n == best)))

    _fill_graph_measures(out, skeletons, soma=soma, voxel_size_um=voxel_size_um,
                         mask_shape=mask.shape)
    return out


def detached_skeleton_um(free, soma, voxel_size_um, mask_shape=None) -> float:
    """Length of the soma-free skeleton in components that **do not reach the soma**.

    > ### ⇒ MEASURED DIRECTLY SINCE 2026-09-02. IT WAS A SUBTRACTION, AND IT OVERSTATED.
    >
    > It used to be `free.total_length_um - anchored.edge_length_um.sum()`, floored at 0. The
    > two artefacts do not differ only by detached material: the collapse **absorbs the whole
    > intra-soma path**, so every primary process contributes roughly a soma radius to that
    > difference on **every cell**, whether or not anything is detached at all. The overstatement
    > is therefore systematic rather than noisy, and it is **larger on 5xFAD**, whose somata are
    > about twice WT volume — a bias pointing the same way as the effect under study.
    >
    > Now: label the free skeleton's connected components, find the ones holding a node inside
    > the soma region, and sum the edge length of every component that does not. Nothing is
    > inferred from the anchored tree, so the intra-soma path cannot leak in.

    **`NaN` when there is no soma, and when no node of the skeleton lies inside it.** With
    nothing to call "attached", every component would count as detached and the column would
    report a cell's entire arbour as debris — the loudest possible wrong answer. The rule is the
    user's, 2026-09-02, and the second case follows from the same argument as the first.

    ``free.node_um`` is in physical microns in the **crop** frame. The soma region is resampled
    onto the **isotropic skeleton grid** before it is indexed, so that "inside the soma" is
    decided the same way here as it is when the collapse picks its root nodes — see the block
    at the conversion below.
    """
    from . import graph_measures as gm

    region = getattr(soma, "region", None) if soma is not None else None
    if free is None or free.n_nodes == 0 or region is None:
        return float("nan")
    region = np.asarray(region).astype(bool)
    if not region.any() or (mask_shape is not None and region.shape != tuple(mask_shape)):
        return float("nan")

    #: > ### ⇒ TESTED ON THE ISOTROPIC GRID, THE ONE THE COLLAPSE USES — 2026-09-02
    #: >
    #: > This asked "is this node inside the soma?" on the **native** grid while
    #: > `skeleton.build_skeletons` asks the identical question on the **isotropic** one when it
    #: > picks the collapse's root nodes. Two roundings of one question, which could disagree.
    #: >
    #: > **The justification is the resolution the answer is entitled to, not a tighter bound.**
    #: > Measured, the two roundings differ per axis rather than uniformly: the native test
    #: > admits a node up to 0.3 µm outside in z against the isotropic 0.1, and 0.054 µm
    #: > in-plane against 0.1. Neither is strictly tighter, and on a well-separated fragment
    #: > they agree. What settles it is that **node positions were derived on the isotropic
    #: > grid** and are only meaningful to half a voxel of it, so testing them against the finer
    #: > in-plane native pitch claims a precision the coordinates do not have — and asking the
    #: > same question two ways invites the two to drift.
    #: >
    #: > The property this buys is testable and is asserted in `tests/test_skeleton.py`: a cell
    #: > whose `collapse_status` is `collapsed` gets a detachment figure, and one that fell back
    #: > to `nearest_node` gets `NaN`. Both statements are now decided by one computation.
    #: >
    #: > **What this does not change** is the rule itself: a fragment truly inside the soma
    #: > region still counts as attached. That is the specified behaviour — material inside the
    #: > cell body is not detached process.
    from .skeleton import resample_isotropic

    region_grid, grid_spacing = resample_isotropic(region, voxel_size_um)
    idx = np.rint(np.asarray(free.node_um, float)
                  / np.asarray(grid_spacing, float)).astype(int)
    for axis in range(3):
        np.clip(idx[:, axis], 0, region_grid.shape[axis] - 1, out=idx[:, axis])
    component = gm._components(free)
    attached = np.unique(component[region_grid[idx[:, 0], idx[:, 1], idx[:, 2]]])
    if attached.size == 0:
        return float("nan")

    edge = np.asarray(free.edge).reshape(-1, 2)
    if not edge.size:
        return 0.0
    #: Both endpoints of an edge are in the same component by construction, so either one
    #: identifies it.
    length = np.asarray(free.edge_length_um, float)
    detached = ~np.isin(component[edge[:, 0]], attached)
    return float(length[detached & np.isfinite(length)].sum())


def _fill_graph_measures(out: CellMorphology, skeletons: CellSkeletons, *,
                         soma=None, voxel_size_um=None, mask_shape=None) -> None:
    """Populate the two per-skeleton dicts from the artefacts — added 2026-09-01.

    **Two dicts rather than twenty-eight dataclass fields.** Almost every name here appears on
    *both* skeleton sheets and would need an `anchored_`/`free_` prefix as an attribute, which
    is a lot of near-duplicate declarations for values the artefacts already summarise. The
    export prefixes them when it writes the header, which is where §1.1's collision rule
    actually applies — a join in R — rather than in the measurement object. `hierarchy_um` and
    the two Sholl profiles are already carried this way.

    `detached_skeleton_um` lives on the free sheet because the free artefact is the one that
    keeps unreachable material. **It is measured on that artefact alone** since 2026-09-02 —
    see `detached_skeleton_um` for what it was and why the subtraction it replaced overstated
    detachment on every cell. It needs the **soma region**, which is why this takes `soma`.
    """
    from . import graph_measures as gm

    free = getattr(skeletons, "soma_free", None)
    anchored = getattr(skeletons, "soma_anchored", None)

    if free is not None:
        out.skeleton_free = {
            "free_n_endpoints": int(free.n_endpoints),
            #: **`free_n_branch_points` since 2026-09-02.** The artefact's attribute keeps the
            #: name `n_junctions` — it is the graph's own vocabulary — but the exported column
            #: is prefixed, because the anchored sheet counts the same thing and §1.1's rule
            #: prefixes a measure computable on both artefacts so a join cannot collide.
            "free_n_branch_points": int(free.n_junctions),
            "total_length_um": float(free.total_length_um),
            "n_skeleton_components": int(free.n_components),
            "free_n_loops": int(free.n_loops),
            "free_n_triple_points": int(free.n_triple_points),
            "free_n_quadruple_points": int(free.n_quadruple_points),
            "free_n_slab_voxels": int(free.n_slab_voxels),
            **gm.summarise_free(free),
        }
        #: **Written on this branch too**, so a cell with no anchored tree gets the column's
        #: honest `NaN` rather than no key at all. It used to be filled only inside the
        #: anchored block below, because it was computed from both artefacts.
        out.skeleton_free["detached_skeleton_um"] = detached_skeleton_um(
            free, soma, voxel_size_um, mask_shape)

    if anchored is not None and anchored.n_nodes:
        out.skeleton_anchored = {
            "anchored_n_endpoints": int(anchored.n_endpoints),
            "anchored_n_loops": int(anchored.n_loops),
            "anchored_n_triple_points": int(anchored.n_triple_points),
            "anchored_n_quadruple_points": int(anchored.n_quadruple_points),
            "anchored_n_slab_voxels": int(anchored.n_slab_voxels),
            **gm.summarise_anchored(anchored),
        }
