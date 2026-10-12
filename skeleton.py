"""Phase C, stage 1: the mask becomes **two skeletons**, and neither is thrown away.

Until 2026-09-01 the skeleton lived about forty lines inside `morphometrics.measure_cell` and
was discarded the moment its numbers were read. That is why nothing could show a detached
fragment, why an external comparison meant re-deriving it, and why every new feature meant
thinning again. It is now an artefact.

**Two artefacts, from one set of voxels.** The thinning never needs a soma, so both start from
the same medial axis and differ only in what is built on top:

* **soma-free** — the undirected graph over everything the mask contains, **including material
  not connected to the cell body**. Branches are the runs between junctions, which is
  AnalyzeSkeleton's convention (Arganda-Carreras et al. 2010) and needs no soma at all. This is
  what a PrestoCell-style mask with no identified soma still measures, and it is the substrate
  for the root-free measures of agenda item A8.
* **soma-anchored** — a **rooted tree**. The soma region is collapsed to a single root and a
  shortest-path sweep gives every node one parent. That is loop-free and detached-free *by
  construction* rather than by filtering: unreachable material has infinite distance and never
  enters, which is agenda item A2 solved structurally, and the resulting endpoint count is a
  count of real branches, restoring 3DMorph's identity ``branch count = endpoint count``.

**The two disagree on purpose, and the difference is the finding.** Soma-free counts detached
material because it has no notion of reachability; soma-anchored cannot. Differencing them is
how much of a cell arrived in pieces.

**Anisotropy.** Thinning happens on an isotropic grid (`SKELETON_TARGET_UM`), because
`skimage.skeletonize` has no `spacing` and at 5.5:1 it finds the medial axis of a distorted
cell — one process read between 5.40 and 11.95 µm depending on its orientation
(`PROJECT_LOG.md` §54). The soma region is resampled onto the *same* grid before the collapse,
so the containment test is physical too.

**Order of operations differs between the two, deliberately** (`PROJECT_LOG.md` §57):

* soma-free is **loops then spurs** — cutting a cycle turns one arm into a dead end, and only a
  spur pass that runs afterwards can see it;
* soma-anchored is **collapse then spurs** — so ``min_spur_um`` is applied to processes as they
  emerge from the cell body, which is what the number means to a reader, rather than to raw
  segments that still include the soma's own medial tangle.

Spurs are pruned once per artefact, never on the shared skeleton, so when loop pruning is
``none`` the two are treated identically.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from skimage.morphology import skeletonize

__all__ = [
    "SKELETON_TARGET_UM",
    "LOOP_PRUNE_METHODS",
    "LOOP_PRUNE_DEFAULT",
    "SkeletonArtefact",
    "CellSkeletons",
    "resample_isotropic",
    "neighbour_count",
    "build_skeletons",
]

#: The isotropic grid the mask is resampled onto **before thinning**, µm. Full derivation in
#: `morphometrics.SKELETON_TARGET_UM`'s original home and `PROJECT_LOG.md` §54: 0.20 measured
#: better than the 0.1083 lateral pitch (2.6% worst orientation error against 16.2%) and uses
#: 0.88x the native voxel count. A **numerical-validity constant, not a user parameter** — the
#: user settled that on 2026-08-31, with the sub-target-feature cost stated.
SKELETON_TARGET_UM = 0.20

#: How a cycle in the **soma-free** skeleton is opened, if at all. AnalyzeSkeleton exposes the
#: same family and defaults to leaving them alone; so do we, at the user's instruction.
#:
#: **The soma-anchored tree never sees any of this.** It is acyclic by construction, so applying
#: a pruning rule there would be meaningless — and applying it to the *shared* skeleton instead
#: would let this setting silently move the tree as well.
LOOP_PRUNE_METHODS = ("none", "shortest_branch", "lowest_intensity_branch",
                      "lowest_intensity_voxel")
LOOP_PRUNE_DEFAULT = "none"

#: Node roles on a stored artefact.
NODE_ROOT, NODE_ENDPOINT, NODE_JUNCTION = 0, 1, 2


# --------------------------------------------------------------------------
# The stored artefact
# --------------------------------------------------------------------------

@dataclass
class SkeletonArtefact:
    """One skeleton, **simplified and in microns** — never a voxel image.

    A node is a point on the centre-line. Only the ones that mean something are kept: the root,
    every fork, and every tip. Each unbroken run between two of them is one edge carrying its
    true physical length. The user chose this over full resolution on 2026-09-01: a cell carries
    ~4,000 skeleton voxels at 0.20 µm, so storing every one would be 0.5–2 GB across the study
    against roughly 220 MB for this, and it is the representation every tracing tool already
    reads.

    **What that costs:** the exact path geometry. A rendered arbour is straight lines between
    junctions rather than following each process. Phase C runs after rich review so no screen is
    affected; it is visible only to someone opening an exported file.

    ``total_length_um`` is measured on the **full-resolution** graph before simplification, so
    it is the true skeleton length. The simplified ``edge_length_um`` sum comes out slightly
    lower, because merged junction clumps absorb their own interiors — the two are different
    quantities and both are kept rather than reconciled.
    """

    kind: str                                   # "soma_free" | "soma_anchored"
    node_um: np.ndarray                         # (n, 3) float32, physical (z, y, x)
    node_role: np.ndarray                       # (n,) uint8, NODE_* above
    node_radius_um: np.ndarray                  # (n,) float32, largest sphere that fits
    edge: np.ndarray                            # (m, 2) int32, node indices
    edge_length_um: np.ndarray                  # (m,) float32
    #: Radius **along the segment's own path**, mean and maximum, sampled in `_simplify` while
    #: the full-resolution chain is still in hand — added 2026-09-01 for `mean_radius_um` on
    #: `branches_anchored.csv` and `segments_free.csv`.
    #:
    #: **They are not derivable from `node_radius_um`.** This artefact keeps nodes only where
    #: the arbour ends or forks, so the node radii are a segment's two **ends**: averaging them
    #: samples exactly the extremes of a taper and throws away everything between. Since a
    #: process thinning distally is precisely what these columns exist to detect, the two-ended
    #: mean would be biased against the measurement rather than merely imprecise.
    #:
    #: Default to empty so an artefact built before this field — or by a caller that does not
    #: supply it — is still constructible; the measures then read `NaN`, not a wrong number.
    edge_radius_um: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    edge_radius_max_um: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    #: Parent per node, `-1` at the root. **Soma-anchored only**; `None` on the soma-free
    #: artefact, which is a graph and has no parents.
    parent: np.ndarray | None = None
    n_endpoints: int = 0
    n_junctions: int = 0
    n_triple_points: int = 0
    n_quadruple_points: int = 0
    n_slab_voxels: int = 0
    n_components: int = 1
    #: Independent cycles, `E - V + C`, counted on the **shared** skeleton before any pruning.
    #: Spurs cannot be part of a cycle, so pruning never moves this and there is no ordering
    #: ambiguity to document.
    n_loops: int = 0
    total_length_um: float = float("nan")
    status: str = "ok"

    @property
    def n_nodes(self) -> int:
        return int(self.node_um.shape[0])

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Flat arrays for `np.savez_compressed`. Scalars ride along as 0-d arrays."""
        out = {
            f"{self.kind}__node_um": self.node_um.astype(np.float32, copy=False),
            f"{self.kind}__node_role": self.node_role.astype(np.uint8, copy=False),
            f"{self.kind}__node_radius_um": self.node_radius_um.astype(np.float32, copy=False),
            f"{self.kind}__edge": self.edge.astype(np.int32, copy=False),
            f"{self.kind}__edge_length_um": self.edge_length_um.astype(np.float32, copy=False),
            f"{self.kind}__edge_radius_um":
                self.edge_radius_um.astype(np.float32, copy=False),
            f"{self.kind}__edge_radius_max_um":
                self.edge_radius_max_um.astype(np.float32, copy=False),
        }
        if self.parent is not None:
            out[f"{self.kind}__parent"] = self.parent.astype(np.int32, copy=False)
        for name in ("n_endpoints", "n_junctions", "n_triple_points", "n_quadruple_points",
                     "n_slab_voxels", "n_components", "n_loops"):
            out[f"{self.kind}__{name}"] = np.asarray(getattr(self, name), dtype=np.int64)
        out[f"{self.kind}__total_length_um"] = np.asarray(self.total_length_um, np.float64)
        out[f"{self.kind}__status"] = np.asarray(self.status)
        return out

    def summary(self) -> dict[str, Any]:
        """The scalars, for the measurement payload. Geometry stays in the `.npz`."""
        return {
            "n_nodes": self.n_nodes,
            "n_endpoints": self.n_endpoints,
            "n_junctions": self.n_junctions,
            "n_triple_points": self.n_triple_points,
            "n_quadruple_points": self.n_quadruple_points,
            "n_slab_voxels": self.n_slab_voxels,
            "n_components": self.n_components,
            "n_loops": self.n_loops,
            "total_length_um": self.total_length_um,
            "simplified_length_um": float(self.edge_length_um.sum())
            if self.edge_length_um.size else 0.0,
            "status": self.status,
        }


@dataclass
class CellSkeletons:
    """Both artefacts for one cell, plus what the shared stage measured once."""

    soma_free: SkeletonArtefact
    soma_anchored: SkeletonArtefact | None = None
    spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
    status: str = "ok"
    #: Set when the soma region held no skeleton node and the collapse could not run. **Visible
    #: rather than silent**: a cell measured a different way must not be indistinguishable in a
    #: column from one measured the usual way.
    collapse_status: str = "collapsed"

    def to_arrays(self) -> dict[str, np.ndarray]:
        out = dict(self.soma_free.to_arrays())
        if self.soma_anchored is not None:
            out.update(self.soma_anchored.to_arrays())
        out["spacing_um"] = np.asarray(self.spacing_um, np.float64)
        out["status"] = np.asarray(self.status)
        out["collapse_status"] = np.asarray(self.collapse_status)
        return out


# --------------------------------------------------------------------------
# Resampling and thinning — the A1 fix
# --------------------------------------------------------------------------

def resample_isotropic(mask: np.ndarray, voxel_size_um: Sequence[float],
                       target_um: float | None = None
                       ) -> tuple[np.ndarray, tuple[float, float, float]]:
    """``mask`` onto an isotropic grid, via a **signed distance field in true microns**.

    **The distance field is the whole point.** Zooming the 0/1 array and thresholding at 0.5
    interpolates a field carrying no information about how far inside or outside a voxel sits,
    and on a process two or three voxels across it dissolves or bloats: measured, that lost a
    real endpoint on 2 of 6 fixture geometries where voxel space lost none. Both transforms take
    ``sampling=``, so what is resampled is in physical microns.

    **The spacing rule is paired to the grid convention.** ``ndimage.zoom`` defaults to
    ``grid_mode=False``, which maps first and last voxel *centres* onto each other, so the grid
    spans ``n - 1`` intervals and the achieved spacing is ``v * (n_old - 1) / (n_new - 1)``.
    Pairing that with ``n_old / n_new`` leaves ~1.5% error in z against ~0.3% in plane — a
    miniature of the anisotropy being fixed. ``grid_mode=True`` was measured *worse* (10.9%
    orientation spread against 5.7%) and is deliberately not used.
    """
    target_um = SKELETON_TARGET_UM if target_um is None else float(target_um)
    inside = ndimage.distance_transform_edt(mask, sampling=voxel_size_um)
    outside = ndimage.distance_transform_edt(~mask, sampling=voxel_size_um)
    zoomed = ndimage.zoom((inside - outside).astype(np.float32),
                          tuple(float(v) / target_um for v in voxel_size_um), order=1)
    resampled = zoomed > 0.0
    achieved = tuple(float(v) * (a - 1) / (b - 1) if b > 1 else float(v)
                     for v, a, b in zip(voxel_size_um, mask.shape, resampled.shape))
    return resampled, achieved


def _skeletonise(mask: np.ndarray) -> np.ndarray:
    """`skeletonize(method="lee")`, guarded against its **silent empty return**.

    ``skimage.skeletonize`` can return an entirely empty skeleton for a perfectly ordinary
    solid, with no error. Measured on skimage 0.25.2: a solid capsule of 4088 voxels over a
    9x11x60 extent — one component, no holes, not touching the border — thinned to **nothing**,
    while a hand-built ellipsoidal cylinder of nearly the same size thinned correctly. Flat bars
    hit it by parity: a 50x10 bar works at 3 voxels thick and returns nothing at 1, 2 and 4.

    **An empty skeleton for a non-empty solid is provably wrong**, so the retry needs no
    threshold. **Transposing fixes it** — the same capsule gave 46 voxels with z and y swapped —
    which places the cause in axis-order handling inside the thinning.

    > ### ⇒ THE GUARD WAS BLIND TO A SINGLE LOST COMPONENT — fixed 2026-09-02
    >
    > The retry above fires only when the skeleton is empty **everywhere**. The same parity
    > defect can take **one** component out of an array holding several, and nothing noticed:
    > measured on `tests/_shapes.py`'s branched cell with a 0.7 µm fragment beside it, the
    > fragment thins to **23 voxels alone and 0 in company**. At 0.5, 0.6, 0.9 and 1.1 µm it
    > survives both ways, so it is thickness parity rather than size.
    >
    > **The direction of the error is what made this worth fixing before any data exists.** The
    > lost piece is detached material, and the soma-free artefact exists precisely to keep it
    > (agenda item A2). The soma-anchored tree is untouched, because a detached fragment was
    > never reachable from the soma anyway — so the **free-minus-anchored comparison collapses
    > toward zero**, and that comparison is the instrument `PROJECT_LOG.md` §55 built to decide
    > whether detached fragments explain the WT endpoint jump. A diagnostic that can silently
    > return "fragments are not the cause" is worse than a noisy measurement.
    >
    > **The fix is a second pass over the components that came back empty, not a change to how
    > anything else is thinned.** Every component that produced skeleton voxels keeps exactly
    > the skeleton it had, so no cell that was not losing a piece moves at all — which matters,
    > because every skeleton figure in the export is downstream of this function. Re-running an
    > isolated component also lets the transpose retry above do its work, because that
    > component's own array *is* empty.
    """
    skeleton = skeletonize(mask, method="lee").astype(bool)
    if not mask.any():
        return skeleton
    if not skeleton.any():
        for axes in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
            alt = skeletonize(np.ascontiguousarray(mask.transpose(axes)), method="lee")
            if alt.any():
                return np.ascontiguousarray(alt.transpose(np.argsort(axes))).astype(bool)
        return skeleton
    return _recover_lost_components(mask, skeleton)


def _recover_lost_components(mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """Re-thin any input component whose skeleton came back empty, and union the result.

    **Single-component masks are the overwhelming majority and cost one labelling.** A cell crop
    normally holds one object, `ndimage.label` returns 1, and the function returns immediately
    without a second thinning. The expensive path is taken only where something was actually
    lost.

    **A component that stays empty even alone is left empty, deliberately.** That is the
    genuinely-unthinnable case — a solid too small or too flat to carry a medial axis — and
    inventing voxels for it would be worse than reporting none. The transpose retry inside
    `_skeletonise` has already had its chance by then.
    """
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    if n <= 1:
        return skeleton

    #: Which components carry no skeleton voxel at all. `np.bincount` over the labels of the
    #: skeleton's own voxels is one pass, rather than one comparison per component.
    hit = np.bincount(labels[skeleton], minlength=n + 1)
    lost = [i for i in range(1, n + 1) if hit[i] == 0]
    if not lost:
        return skeleton

    out = skeleton.copy()
    for i in lost:
        component = labels == i
        #: Recursion, not a bare `skeletonize` call: the isolated component is a single-object
        #: array, so `_skeletonise`'s transpose retry applies to it — which is the mechanism
        #: that recovers the parity case in the first place.
        out |= _skeletonise(component)
    return out


def _offsets() -> list[tuple[int, int, int]]:
    """The 13 forward half of the 26-neighbourhood — each edge visited once."""
    return [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o) and o > (0, 0, 0)]


def _paired_slices(shape: Sequence[int], off: Sequence[int]):
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
    """How many of the 26 neighbours of each skeleton voxel are also skeleton.

    The classification rule is AnalyzeSkeleton's exactly: fewer than two neighbours is an
    **endpoint**, exactly two a **slab** voxel along a branch, three or more a **junction**.
    """
    kernel = np.ones((3, 3, 3), np.uint8)
    counts = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    return (counts - skeleton.astype(np.uint8)) * skeleton


def _voxel_graph(skeleton: np.ndarray, spacing: Sequence[float]):
    """26-connected graph over skeleton voxels, **weighted in microns**.

    A diagonal edge costs exactly ``sqrt((dz*z)^2 + (dy*y)^2 + (dx*x)^2)``. 3DMorph's wavefront
    charges one step per neighbour, so at 5.5:1 a z-hop buys 5.5x the distance at the same
    price and paths route through depth; this is that defect not inherited.
    """
    z_um, y_um, x_um = (float(v) for v in spacing)
    coords = np.argwhere(skeleton)
    n = int(coords.shape[0])
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
        rows.append(a[both])
        cols.append(b[both])
        data.append(np.full(int(both.sum()), weight))

    if rows:
        r, c, d = (np.concatenate(rows), np.concatenate(cols), np.concatenate(data))
        graph = coo_matrix((np.concatenate([d, d]),
                            (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))
    else:
        graph = coo_matrix((n, n))
    return coords, index, graph.tocsr()


# --------------------------------------------------------------------------
# Voxel skeleton -> simplified node/edge form
# --------------------------------------------------------------------------

def _simplify(skeleton, coords, index, graph, spacing, radius_field):
    """Contract every run of degree-2 voxels into one edge between two real nodes.

    Nodes are the tips and the forks; **adjacent junction voxels are merged into one actual
    junction** first, which is AnalyzeSkeleton's rule and matters because a single fork routinely
    spans several voxels under 26-connectivity. A raw count over-counts, and not uniformly — it
    scales with arbour complexity, so branchier cells inflate more.

    An edge's length is the sum of the physical steps along its chain plus the one step at each
    end that reaches its anchor. The interior of a merged junction clump is therefore **not**
    attributed to any edge, which is why the simplified lengths sum slightly below
    ``total_length_um``.
    """
    counts = neighbour_count(skeleton)
    junction_vox = skeleton & (counts >= 3)
    endpoint_vox = skeleton & (counts <= 1)
    structure = np.ones((3, 3, 3), bool)
    jlab, n_j = ndimage.label(junction_vox, structure=structure)

    n_vox = int(coords.shape[0])
    node_of = np.full(n_vox, -1, dtype=np.int64)
    node_um: list[np.ndarray] = []
    node_role: list[int] = []
    node_radius: list[float] = []

    if n_j:
        com = np.array(ndimage.center_of_mass(junction_vox, jlab, range(1, n_j + 1)))
        rmax = ndimage.maximum(radius_field, jlab, range(1, n_j + 1))
        rmax = np.atleast_1d(np.asarray(rmax, dtype=float))
        for j in range(n_j):
            node_um.append(np.asarray(com[j]) * np.asarray(spacing))
            node_role.append(NODE_JUNCTION)
            node_radius.append(float(rmax[j]))
        clump = jlab[tuple(coords.T)]
        inside = clump > 0
        node_of[inside] = clump[inside] - 1

    ends = np.flatnonzero(endpoint_vox[tuple(coords.T)])
    for v in ends:
        node_of[v] = len(node_um)
        node_um.append(coords[v] * np.asarray(spacing))
        node_role.append(NODE_ENDPOINT)
        node_radius.append(float(radius_field[tuple(coords[v])]))

    indptr, indices, data = graph.indptr, graph.indices, graph.data
    edges: list[tuple[int, int]] = []
    lengths: list[float] = []
    #: **Radius sampled along the segment's own path, gathered here because here is the only
    #: place it exists.** The simplified artefact keeps nodes at endpoints and junctions only
    #: (`NODE_ROOT` / `NODE_ENDPOINT` / `NODE_JUNCTION`), so `node_radius_um` holds the two
    #: **ends** of a process and nothing between them. A per-segment mean taken from those two
    #: numbers averages exactly the extremes of a taper and discards its interior — biased, not
    #: merely noisy, and the taper is the thing the measure exists to see. The chain walk below
    #: already touches every voxel of the segment, so sampling `radius_field` as it goes costs
    #: one array append per voxel and makes `mean_radius_um` a genuine path mean.
    radii_mean: list[float] = []
    radii_max: list[float] = []
    visited_chain = np.zeros(n_vox, bool)

    def _r(voxel: int) -> float:
        return float(radius_field[tuple(coords[voxel])])

    anchors = np.flatnonzero(node_of >= 0)
    for v in anchors:
        for k in range(indptr[v], indptr[v + 1]):
            u, w = int(indices[k]), float(data[k])
            if node_of[u] >= 0:
                # two anchors touching directly; emit once, from the lower voxel id
                if v < u and node_of[u] != node_of[v]:
                    edges.append((int(node_of[v]), int(node_of[u])))
                    lengths.append(w)
                    pair = (_r(int(v)), _r(u))
                    radii_mean.append(float(np.mean(pair)))
                    radii_max.append(float(np.max(pair)))
                continue
            if visited_chain[u]:
                continue
            # walk the chain of degree-2 voxels away from v
            total, prev, cur = w, int(v), u
            seen = [_r(int(v)), _r(u)]
            while True:
                visited_chain[cur] = True
                nxt, step = -1, 0.0
                for kk in range(indptr[cur], indptr[cur + 1]):
                    cand = int(indices[kk])
                    if cand != prev:
                        nxt, step = cand, float(data[kk])
                        break
                if nxt < 0:
                    break                       # a chain that dead-ends without an endpoint
                total += step
                seen.append(_r(nxt))
                if node_of[nxt] >= 0:
                    edges.append((int(node_of[v]), int(node_of[nxt])))
                    lengths.append(total)
                    radii_mean.append(float(np.mean(seen)))
                    radii_max.append(float(np.max(seen)))
                    break
                prev, cur = cur, nxt

    #: **A closed ring has no tips and no forks, so it has no nodes** — every voxel on it is
    #: degree 2 and the walk above never starts. Left alone it would vanish from the simplified
    #: form entirely while still being real material and a real loop. Each gets one synthetic
    #: node at its centre of mass and one self-edge carrying its circumference, which is also
    #: what makes the cycle count come out right: a self-edge is E=1, V=1, C=1, so E-V+C=1.
    orphan = (node_of < 0) & ~visited_chain
    if orphan.any():
        ring_vol = np.zeros(skeleton.shape, bool)
        ring_vol[tuple(coords[orphan].T)] = True
        rlab, n_r = ndimage.label(ring_vol, structure=structure)
        per_voxel = rlab[tuple(coords.T)]
        for r in range(1, n_r + 1):
            ids = np.flatnonzero(per_voxel == r)
            if ids.size == 0:
                continue
            nid = len(node_um)
            sub = graph[ids][:, ids]
            ring_r = radius_field[tuple(coords[ids].T)]
            edges.append((nid, nid))
            lengths.append(float(sub.sum() / 2.0))
            #: A ring's every voxel is its own interior, so the path mean is over all of them.
            radii_mean.append(float(np.mean(ring_r)))
            radii_max.append(float(np.max(ring_r)))
            node_um.append(coords[ids].mean(0) * np.asarray(spacing))
            node_role.append(NODE_JUNCTION)
            node_radius.append(float(np.max(ring_r)))

    node_um_arr = (np.asarray(node_um, dtype=np.float32) if node_um
                   else np.zeros((0, 3), np.float32))
    return {
        "node_um": node_um_arr,
        "node_role": np.asarray(node_role, dtype=np.uint8),
        "node_radius_um": np.asarray(node_radius, dtype=np.float32),
        "edge": (np.asarray(edges, dtype=np.int32) if edges
                 else np.zeros((0, 2), np.int32)),
        "edge_length_um": np.asarray(lengths, dtype=np.float32),
        "edge_radius_um": np.asarray(radii_mean, dtype=np.float32),
        "edge_radius_max_um": np.asarray(radii_max, dtype=np.float32),
        "n_junctions": int(n_j),
        "n_slab_voxels": int((skeleton & (counts == 2)).sum()),
        "n_endpoints_voxel": int(endpoint_vox.sum()),
    }


def _junction_orders(edge: np.ndarray, node_role: np.ndarray) -> tuple[int, int]:
    """Triple and quadruple points, the way AnalyzeSkeleton labels a merged junction.

    Kim et al. (2024) single out quadruple points as the most topologically complex feature and
    the one that degrades first when 3D information is discarded, which is why they are worth
    separating rather than folding into one junction count.
    """
    if edge.size == 0:
        return 0, 0
    degree = np.bincount(edge.ravel(), minlength=node_role.size)
    junctions = node_role == NODE_JUNCTION
    return (int(((degree == 3) & junctions).sum()), int(((degree == 4) & junctions).sum()))


# --------------------------------------------------------------------------
# Loops, spurs, and the collapse
# --------------------------------------------------------------------------

class _DisjointSet:
    """Union-find. **Not a sparse matrix**, because a loop is often a multi-edge.

    Two junctions joined by two separate chains is exactly what a cycle looks like here, and a
    sparse adjacency matrix cannot hold parallel edges — it sums them and the second one
    disappears. `scipy`'s spanning-tree routines inherit that, so the forest is grown by hand.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, a: int) -> int:
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[rb] = ra
        return True


def _independent_cycles(n_nodes: int, edge: np.ndarray) -> int:
    """``E - V + C`` on the **simplified** graph, which is the only place it means anything.

    Counted here rather than on the voxel graph, and that is not a preference. A one-voxel-wide
    skeleton under 26-connectivity is **full of triangles** — three mutually adjacent voxels
    along any diagonal run — so ``E - V + C`` over the voxel graph counts the connectivity rule
    rather than the cell. Measured: a plain four-armed star with no loop at all reported 4, and
    an eight-armed one reported 12. On the simplified graph, where a node is a fork or a tip and
    an edge is the run between them, a cycle is a cycle.
    """
    if n_nodes == 0 or edge.size == 0:
        return 0
    ds = _DisjointSet(n_nodes)
    closing = 0
    for a, b in np.asarray(edge, dtype=np.int64):
        if int(a) == int(b) or not ds.union(int(a), int(b)):
            closing += 1
    return int(closing)


def _cycle_edges(n_nodes: int, edge: np.ndarray, preference: np.ndarray) -> np.ndarray:
    """Indices of the edges that close a cycle, keeping the most *preferred* ones.

    A maximum spanning forest by preference: edges are offered in descending order and one that
    joins two already-connected nodes is the one that closes a loop. Exactly ``E - V + C`` edges
    come back, which is the independent-cycle count.
    """
    if edge.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-np.asarray(preference, dtype=float), kind="stable")
    ds = _DisjointSet(n_nodes)
    closing: list[int] = []
    for e in order:
        a, b = int(edge[e, 0]), int(edge[e, 1])
        if a == b or not ds.union(a, b):
            closing.append(int(e))
    return np.asarray(sorted(closing), dtype=np.int64)


def _prune_leaf_edges(n_nodes: int, edge: np.ndarray, length: np.ndarray,
                      min_spur_um: float, protect: int = -1):
    """Iteratively drop dead-end edges shorter than ``min_spur_um``.

    This is `prune_spurs`' rule expressed on the simplified graph: a spur is a segment holding
    an endpoint, and its length is the run from that endpoint to its junction — which is exactly
    a leaf edge here. Iterated because removing one spur can expose another, when a junction
    that loses a limb becomes an ordinary path node.

    ``protect`` is the root, which must never be pruned away even when the whole cell is small.
    """
    keep = np.ones(edge.shape[0], bool)
    if min_spur_um <= 0 or edge.size == 0:
        return keep
    while True:
        degree = np.bincount(edge[keep].ravel(), minlength=n_nodes)
        doomed = False
        for e in np.flatnonzero(keep):
            a, b = int(edge[e, 0]), int(edge[e, 1])
            leaf = (degree[a] == 1 and a != protect) or (degree[b] == 1 and b != protect)
            if leaf and float(length[e]) < min_spur_um:
                keep[e] = False
                doomed = True
        if not doomed:
            return keep


def _compact(node_um, node_role, node_radius, edge, length, keep_edge, extra_keep=()):
    """Drop unreferenced nodes after an edge removal and renumber what is left."""
    edge, length = edge[keep_edge], length[keep_edge]
    used = np.zeros(node_um.shape[0], bool)
    if edge.size:
        used[edge.ravel()] = True
    for n in extra_keep:
        if 0 <= n < used.size:
            used[n] = True
    remap = np.full(used.size, -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    edge = remap[edge] if edge.size else edge
    return (node_um[used], node_role[used], node_radius[used],
            edge.astype(np.int32), length, remap)


def _node_intensity(node_um, intensity, native_voxel_um):
    """Sample an intensity volume at each node. `None` when no volume was supplied.

    **A two-point sample per edge, not a walk along it.** The simplified artefact no longer
    holds the chain of voxels between two nodes, so an edge is scored by its two ends rather
    than by every voxel along it. That is a heuristic for choosing which arm of a loop to open,
    not a measurement, and it is only ever consulted when the user asks for an intensity-based
    loop rule.
    """
    if intensity is None:
        return None
    idx = np.rint(node_um / np.asarray(native_voxel_um, dtype=float)).astype(np.int64)
    for a in range(3):
        idx[:, a] = np.clip(idx[:, a], 0, intensity.shape[a] - 1)
    return np.asarray(intensity)[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float)


def _loop_preference(method, edge, length, node_intensity):
    """Per-edge keep-preference for the spanning forest, and whether it can be honoured."""
    if edge.size == 0:
        return None, "ok"
    if method == "shortest_branch":
        #: keep the long arms, so the *shortest* arm of each loop is the one opened
        return np.asarray(length, dtype=float), "ok"
    if method in ("lowest_intensity_branch", "lowest_intensity_voxel"):
        if node_intensity is None:
            return None, "loop_prune_needs_intensity"
        ends = node_intensity[edge]
        return (ends.mean(1) if method == "lowest_intensity_branch" else ends.min(1)), "ok"
    return None, "ok"


def _artefact(kind, simp, node_um, node_role, node_radius, edge, length, *,
              edge_radius=None, edge_radius_max=None,
              parent=None, n_components=1, n_loops=0, total_length_um=float("nan"),
              status="ok"):
    degree = (np.bincount(edge.ravel(), minlength=node_um.shape[0])
              if edge.size else np.zeros(node_um.shape[0], np.int64))
    leaves = int(((degree == 1) & (node_role != NODE_ROOT)).sum())
    triple, quad = _junction_orders(edge, node_role)
    #: **A fork of THIS artefact, counted by degree — not by what a voxel was classified as.**
    #: `node_role` records how a node arose, which survives the collapse and the pruning; a
    #: node that was a junction voxel can end up with one or two edges once the soma has been
    #: absorbed around it. Counting roles reported 1 junction for an eight-armed star whose
    #: arms are straight and unbranched, where the true answer is 0. The root is excluded
    #: because its degree is the number of primary processes, not a branch point.
    forks = int(((degree >= 3) & (node_role != NODE_ROOT)).sum())
    n_e = int(edge.shape[0])
    #: Empty rather than wrong when a caller supplies nothing, and length-checked rather than
    #: trusted: an array that does not match the edge list would silently mis-attribute every
    #: radius to the wrong segment, which is worse than reporting none.
    def _per_edge(a):
        a = np.zeros(0, np.float32) if a is None else np.asarray(a, np.float32)
        return a if a.shape[0] == n_e else np.full(n_e, np.nan, np.float32)

    return SkeletonArtefact(
        kind=kind, node_um=node_um, node_role=node_role, node_radius_um=node_radius,
        edge=edge, edge_length_um=length,
        edge_radius_um=_per_edge(edge_radius),
        edge_radius_max_um=_per_edge(edge_radius_max), parent=parent,
        n_endpoints=leaves, n_junctions=forks,
        n_triple_points=triple, n_quadruple_points=quad,
        n_slab_voxels=int(simp["n_slab_voxels"]), n_components=int(n_components),
        n_loops=int(n_loops), total_length_um=float(total_length_um), status=status)


def _empty(kind, status):
    return SkeletonArtefact(
        kind=kind, node_um=np.zeros((0, 3), np.float32),
        node_role=np.zeros(0, np.uint8), node_radius_um=np.zeros(0, np.float32),
        edge=np.zeros((0, 2), np.int32), edge_length_um=np.zeros(0, np.float32),
        total_length_um=0.0, status=status)


def _root_the_tree(node_um, node_role, node_radius, edge, length, root_nodes, root_um,
                   root_radius, min_spur_um, edge_radius=None, edge_radius_max=None):
    """Collapse ``root_nodes`` into one root and grow the shortest-path tree out from it.

    **Loop-free and detached-free by construction, not by filtering.** Every reachable node gets
    exactly one parent, so N nodes give N-1 edges, and a graph with N-1 edges cannot hold a
    cycle: the arm that would close one simply is not on any shortest path from the root. And
    anything the sweep cannot reach has infinite distance and never enters -- which is agenda
    item A2 handled structurally, so an endpoint count taken here counts real branches.

    **Then spurs, not before** (`PROJECT_LOG.md` section 57). Pruning after the collapse applies
    ``min_spur_um`` to processes *as they emerge from the cell body*, which is what the number
    means to a reader. Pruning first would apply it to raw segments that still include the
    medial tangle inside the cell body, and would leave sub-threshold stubs hanging off the root
    that nothing ever removes.
    """
    n_old = node_um.shape[0]
    is_root = np.zeros(n_old, bool)
    is_root[np.asarray(root_nodes, dtype=np.int64)] = True

    remap = np.zeros(n_old, dtype=np.int64)
    others = np.flatnonzero(~is_root)
    remap[others] = np.arange(1, others.size + 1)
    n_new = others.size + 1

    new_um = np.vstack([np.asarray(root_um, np.float32)[None, :], node_um[others]])
    new_role = np.concatenate([[NODE_ROOT], node_role[others]]).astype(np.uint8)
    new_rad = np.concatenate([[root_radius], node_radius[others]]).astype(np.float32)

    a, b = remap[edge[:, 0]], remap[edge[:, 1]]
    live = a != b                                   # intra-soma edges vanish into the root
    a, b, w = a[live], b[live], np.asarray(length, float)[live]

    #: **Parallel edges are reduced by MINIMUM, never summed.** A sparse matrix adds duplicate
    #: entries, which for a shortest path is simply wrong: two arms of a loop between the same
    #: pair would become one arm of their combined length.
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * n_new + hi
    order = np.lexsort((w, key))
    key_s, w_s, lo_s, hi_s = key[order], w[order], lo[order], hi[order]
    first = np.ones(key_s.size, bool)
    first[1:] = key_s[1:] != key_s[:-1]
    #: **The per-edge radii follow the weights through exactly the same three steps** — the
    #: `live` mask, the lexsort, and this de-duplication — so the radius that survives belongs
    #: to *the same* parallel edge whose length survived. Averaging duplicates instead would
    #: describe an arm the shortest path does not take, which is the mirror of the summing
    #: mistake the comment above forbids.
    have_r = edge_radius is not None and np.asarray(edge_radius).size == np.asarray(length).size
    if have_r:
        r_s = np.asarray(edge_radius, float)[live][order][first]
        rmax_s = np.asarray(edge_radius_max, float)[live][order][first]
    lo_s, hi_s, w_s = lo_s[first], hi_s[first], w_s[first]
    #: Keys of the surviving edges, ascending — `key` was the lexsort's primary term, so this
    #: is already sorted and `searchsorted` can read it without another sort.
    key_kept = lo_s * n_new + hi_s

    graph = coo_matrix((np.concatenate([w_s, w_s]),
                        (np.concatenate([lo_s, hi_s]), np.concatenate([hi_s, lo_s]))),
                       shape=(n_new, n_new)).tocsr()
    dist, pred = dijkstra(graph, directed=False, indices=0, return_predecessors=True)

    reach = np.isfinite(dist)
    reach[0] = True
    tree_edges, tree_len = [], []
    for i in np.flatnonzero(reach):
        if i == 0:
            continue
        p = int(pred[i])
        if p < 0:
            continue
        tree_edges.append((p, int(i)))
        tree_len.append(float(dist[i] - dist[p]))

    edge_t = (np.asarray(tree_edges, np.int32) if tree_edges else np.zeros((0, 2), np.int32))
    len_t = np.asarray(tree_len, np.float32)

    #: **Every predecessor-tree edge is a graph edge**, so each one inherits the radius of the
    #: single original it came from. Its *length* cannot be inherited the same way — `tree_len`
    #: is a difference of shortest-path distances — which is exactly why the radius has to be
    #: looked up rather than carried alongside the weight.
    if have_r and edge_t.size:
        want = (np.minimum(edge_t[:, 0], edge_t[:, 1]).astype(np.int64) * n_new
                + np.maximum(edge_t[:, 0], edge_t[:, 1]))
        pos = np.searchsorted(key_kept, want)
        pos = np.clip(pos, 0, max(key_kept.size - 1, 0))
        hit = key_kept.size > 0
        rad_t = np.where(hit & (key_kept[pos] == want), r_s[pos], np.nan).astype(np.float32)
        radmax_t = np.where(hit & (key_kept[pos] == want), rmax_s[pos], np.nan).astype(np.float32)
    else:
        rad_t = np.full(edge_t.shape[0], np.nan, np.float32)
        radmax_t = np.full(edge_t.shape[0], np.nan, np.float32)

    keep = _prune_leaf_edges(n_new, edge_t, len_t, float(min_spur_um), protect=0)
    #: `_compact` filters edges by `keep` and preserves their order, so the same mask applied
    #: here keeps these two arrays aligned with `edge_t` without widening its signature.
    rad_t, radmax_t = rad_t[keep], radmax_t[keep]
    new_um, new_role, new_rad, edge_t, len_t, _ = _compact(
        new_um, new_role, new_rad, edge_t, len_t, keep, extra_keep=(0,))

    parent = np.full(new_um.shape[0], -1, dtype=np.int32)
    if edge_t.size:
        parent[edge_t[:, 1]] = edge_t[:, 0]
    return new_um, new_role, new_rad, edge_t, len_t, parent, rad_t, radmax_t


def build_skeletons(mask: np.ndarray, *, voxel_size_um: Sequence[float],
                    soma: Any | None = None, min_spur_um: float = 1.0,
                    loop_prune: str = LOOP_PRUNE_DEFAULT,
                    intensity: np.ndarray | None = None,
                    target_um: float | None = None) -> CellSkeletons:
    """One mask in, **two skeletons out**. Never raises on a degenerate cell.

    ``soma`` is a `soma.SomaResult`; its ``region`` drives the collapse and its ``centroid``
    places the root. Without one, only the soma-free artefact is produced -- which is the point
    of having it, and is what a PrestoCell-style mask with no soma still measures.
    """
    if loop_prune not in LOOP_PRUNE_METHODS:
        raise ValueError(f"unknown loop_prune {loop_prune!r}; expected {LOOP_PRUNE_METHODS}")

    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return CellSkeletons(soma_free=_empty("soma_free", "empty_mask"),
                             soma_anchored=_empty("soma_anchored", "empty_mask"),
                             status="empty_mask", collapse_status="no_mask")

    resampled, spacing = resample_isotropic(mask, voxel_size_um, target_um)
    skeleton = _skeletonise(resampled)
    if not skeleton.any():
        return CellSkeletons(soma_free=_empty("soma_free", "no_skeleton"),
                             soma_anchored=_empty("soma_anchored", "no_skeleton"),
                             spacing_um=spacing, status="no_skeleton",
                             collapse_status="no_skeleton")

    radius_field = ndimage.distance_transform_edt(resampled, sampling=spacing)
    coords, index, graph = _voxel_graph(skeleton, spacing)
    total_length_um = float(graph.sum() / 2.0)
    n_comp = int(connected_components(graph, directed=False, return_labels=False))

    simp = _simplify(skeleton, coords, index, graph, spacing, radius_field)
    node_um, node_role = simp["node_um"], simp["node_role"]
    node_rad, edge, length = simp["node_radius_um"], simp["edge"], simp["edge_length_um"]
    e_rad, e_radmax = simp["edge_radius_um"], simp["edge_radius_max_um"]
    n_loops = _independent_cycles(node_um.shape[0], edge)

    # ---- soma-free: LOOPS, then spurs ------------------------------------
    free_status = "ok"
    keep = np.ones(edge.shape[0], bool)
    if loop_prune != "none" and edge.size:
        pref, free_status = _loop_preference(
            loop_prune, edge, length, _node_intensity(node_um, intensity, voxel_size_um))
        if pref is not None:
            keep[_cycle_edges(node_um.shape[0], edge, pref)] = False
    f_um, f_role, f_rad, f_edge, f_len, _ = _compact(
        node_um, node_role, node_rad, edge, length, keep)
    #: `_compact` preserves edge order, so the per-edge radii ride the same two masks.
    f_erad, f_eradmax = e_rad[keep], e_radmax[keep]
    keep2 = _prune_leaf_edges(f_um.shape[0], f_edge, f_len, float(min_spur_um))
    f_um, f_role, f_rad, f_edge, f_len, _ = _compact(f_um, f_role, f_rad, f_edge, f_len, keep2)
    f_erad, f_eradmax = f_erad[keep2], f_eradmax[keep2]
    soma_free = _artefact("soma_free", simp, f_um, f_role, f_rad, f_edge, f_len,
                          edge_radius=f_erad, edge_radius_max=f_eradmax,
                          n_components=n_comp, n_loops=n_loops,
                          total_length_um=total_length_um, status=free_status)

    # ---- soma-anchored: COLLAPSE, then spurs -----------------------------
    region = getattr(soma, "region", None) if soma is not None else None
    centroid = getattr(soma, "centroid", None) if soma is not None else None
    if centroid is None or node_um.shape[0] == 0:
        return CellSkeletons(soma_free=soma_free,
                             soma_anchored=_empty("soma_anchored", "no_soma"),
                             spacing_um=spacing, collapse_status="no_soma")

    root_um = np.asarray(centroid, dtype=float) * np.asarray(voxel_size_um, dtype=float)
    collapse_status = "collapsed"
    root_nodes = np.zeros(0, dtype=np.int64)
    if region is not None:
        region_res, _ = resample_isotropic(np.asarray(region).astype(bool),
                                           voxel_size_um, target_um)
        if region_res.shape == skeleton.shape:
            idx = np.rint(node_um / np.asarray(spacing)).astype(np.int64)
            for a in range(3):
                idx[:, a] = np.clip(idx[:, a], 0, skeleton.shape[a] - 1)
            root_nodes = np.flatnonzero(region_res[idx[:, 0], idx[:, 1], idx[:, 2]])
    if root_nodes.size == 0:
        #: **Visible, not silent.** A cell rooted a different way must not be
        #: indistinguishable in a column from one rooted the usual way.
        collapse_status = "nearest_node"
        root_nodes = np.asarray([int(np.argmin(np.linalg.norm(node_um - root_um, axis=1)))])

    root_radius = float(np.max(node_rad[root_nodes])) if node_rad.size else 0.0
    a_um, a_role, a_rad, a_edge, a_len, parent, a_erad, a_eradmax = _root_the_tree(
        node_um, node_role, node_rad, edge, length, root_nodes, root_um, root_radius,
        float(min_spur_um), edge_radius=e_rad, edge_radius_max=e_radmax)
    anchored = _artefact("soma_anchored", simp, a_um, a_role, a_rad, a_edge, a_len,
                         edge_radius=a_erad, edge_radius_max=a_eradmax,
                         parent=parent, n_components=1, n_loops=n_loops,
                         total_length_um=float(a_len.sum()), status="ok")
    return CellSkeletons(soma_free=soma_free, soma_anchored=anchored, spacing_um=spacing,
                         collapse_status=collapse_status)
