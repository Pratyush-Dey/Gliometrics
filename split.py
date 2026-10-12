"""Dividing clumped cells, using nuclei as the only authority.

This is what the whole design exists for. On the marked test crop, **9 of 13 cells sit
inside just 2 connected objects** — one holding 4 somata, one holding 5. Without
splitting, 13 cells would be reported as 6, and the two largest "cells" would be
chimeras of four and five.

Three rules, and no others:

* an object overlapping **two or more** nuclei is split into that many parts;
* an object overlapping **one** nucleus is left whole;
* an object overlapping **no** nucleus is left whole — never force-split.

3DMorph instead splits whenever a blob exceeds a hand-picked size, and forces the count
to at least two (``if nuc == 1, nuc = 2``), so a genuinely large single cell is always
divided. It has no independent evidence of how many cells are present; here the nuclei
channel supplies exactly that.

**Why an intensity landscape, not a distance transform.** Marker-controlled watershed on
a distance transform is the classic way to separate touching round cells, and it is
wrong here: microglial processes are thin everywhere, so the distance map is nearly flat
along them and boundaries land arbitrarily mid-process. Flooding smoothed *intensity*
instead puts boundaries where the signal genuinely dips.

Smoothing is sized in **microns**, not voxels. With 5.5:1 anisotropy an isotropic
gaussian would blur ~5.5x further in z than in plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima, local_maxima
from skimage.segmentation import watershed

from .membership import nuclei_per_object, overlap_table
from .objects import Objects

__all__ = [
    "SplitOutcome",
    "split_by_watershed",
    "split_by_pbc",
    "split_by_tomato",
    "decompose_by_persistence",
    "decompose_by_persistence_levels",
    "TOMATO_RADIUS",
    "split_objects",
    "merge_objects_sharing_a_nucleus",
]


@dataclass
class SplitOutcome:
    """Final per-cell labels, with a record of how each cell was produced."""

    labels: np.ndarray                          # (z, y, x) int32; 0 = background
    n_cells: int
    method_by_cell: dict[int, str] = field(default_factory=dict)
    nuclei_by_cell: dict[int, list[int]] = field(default_factory=dict)
    n_objects_in: int = 0
    n_split: int = 0
    n_merged: int = 0

    def summary(self) -> str:
        return (f"{self.n_objects_in} objects -> {self.n_cells} cells "
                f"({self.n_split} split, {self.n_merged} merged)")

    @property
    def cells_without_nuclei(self) -> list[int]:
        """Kept whole by design; they still become cells, just unsplittable ones."""
        return [c for c, n in self.nuclei_by_cell.items() if not n]


def merge_objects_sharing_a_nucleus(
    object_labels: np.ndarray,
    nucleus_labels: np.ndarray,
    *,
    min_overlap_fraction: float = 0.10,
) -> tuple[np.ndarray, int]:
    """Fuse objects that a single nucleus spans.

    One cell can be broken into two connected components by a gap in the threshold —
    a dim patch across the soma is enough. Both fragments then contain part of the same
    nucleus. Left alone, that cell is reported twice, and each half carries a partial
    volume and a truncated skeleton.

    PrestoCell does the equivalent (``main.py:1534-1542``); 3DMorph has no such step,
    because with one channel it has no way to tell a broken cell from two cells.
    """
    a_ids, b_ids, counts = overlap_table(nucleus_labels, object_labels)
    if a_ids.size == 0:
        return object_labels, 0

    n_nuc = int(nucleus_labels.max())
    sizes = np.bincount(nucleus_labels.ravel(), minlength=n_nuc + 1)

    groups: dict[int, list[int]] = {}
    for nucleus, obj, count in zip(a_ids, b_ids, counts):
        if count / max(sizes[int(nucleus)], 1) >= min_overlap_fraction:
            groups.setdefault(int(nucleus), []).append(int(obj))

    # union-find over objects that share a nucleus
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    merges = 0
    for objs in groups.values():
        for other in objs[1:]:
            if find(objs[0]) != find(other):
                merges += 1
            union(objs[0], other)

    if merges == 0:
        return object_labels, 0

    lookup = np.arange(int(object_labels.max()) + 1, dtype=np.int32)
    for obj in list(parent):
        lookup[obj] = find(obj)
    merged = lookup[object_labels]

    present = np.unique(merged)
    present = present[present != 0]
    compact = np.zeros(int(merged.max()) + 1, dtype=np.int32)
    compact[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return compact[merged], merges


def split_by_watershed(
    intensity: np.ndarray,
    object_mask: np.ndarray,
    markers: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    smooth_um: float = 0.4,
) -> np.ndarray:
    """Flood smoothed intensity outward from each nucleus, confined to the object.

    Every voxel of the object is assigned to exactly one marker, so a split never
    discards material — the parts always reconstitute the original object.
    """
    if not (intensity.shape == object_mask.shape == markers.shape):
        raise ValueError(
            f"shapes must match: intensity {intensity.shape}, "
            f"mask {object_mask.shape}, markers {markers.shape}"
        )

    sigma = [float(smooth_um) / float(v) for v in voxel_size_um]
    smoothed = ndimage.gaussian_filter(intensity.astype(np.float32), sigma=sigma)
    # watershed climbs, so invert: bright regions become basins to fill from.
    return watershed(-smoothed, markers=markers, mask=object_mask).astype(np.int32)


def split_by_pbc(
    intensity: np.ndarray,
    object_mask: np.ndarray,
    markers: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    smooth_um: float = 0.4,
    tau_fraction: float = 0.05,
) -> np.ndarray:
    """PrestoCell's fragment propagation: re-cluster finely, then assign whole fragments.

    The difference from watershed is *what* gets assigned. Watershed decides voxel by
    voxel, flooding outward from each nucleus, so a boundary can fall anywhere the
    intensity landscape dips. This clusters the clump into persistence-stable fragments
    first and then assigns each fragment **whole**, by adjacency. A thin process therefore
    travels to its nucleus intact rather than being cut wherever it happens to dim — which
    is the case that matters here, since microglial processes are thin everywhere and dim
    at their tips.

    **An approximation of PrestoCell's PBC, kept for comparison — not the real thing.**
    Written with `h_maxima` while `gudhi` was unavailable. `split_by_tomato` below is the
    faithful port; use that one to represent PBC.

    The persistence filter itself is a fair substitution: ToMATo merges clusters whose
    separating saddle lies within τ of the lower peak, which is what `h_maxima(field, tau)`
    applies, and it does produce real over-segmentation (median 54 fragments per clump on
    the marked images). **The τ handling is not fair.** An earlier note here claimed the
    recursive halving loop did not affect the assignment rule; that was wrong. PrestoCell
    *searches* τ per clump, halving until every nucleus owns an exclusive cluster
    (`main.py:611`), while this function uses a fixed fraction. Measurement showed the
    entire apparent advantage of this back end over watershed lives in that constant and
    disappears at ``tau_fraction >= 0.10``.

    ``tau_fraction`` is a fraction of the object's own intensity span rather than an
    absolute level: these are 32-bit images with peaks near 155,000 and no per-image
    rescaling (§4.13), so any absolute τ would mean something different in every file.
    """
    if not (intensity.shape == object_mask.shape == markers.shape):
        raise ValueError(
            f"shapes must match: intensity {intensity.shape}, "
            f"mask {object_mask.shape}, markers {markers.shape}"
        )

    sigma = [float(smooth_um) / float(v) for v in voxel_size_um]
    smoothed = ndimage.gaussian_filter(intensity.astype(np.float32), sigma=sigma)

    inside = smoothed[object_mask]
    if inside.size == 0:
        return np.zeros(markers.shape, dtype=np.int32)

    # Confine the field so maxima cannot be found outside the clump, and so the
    # persistence filter sees the object's own dynamic range rather than the crop's.
    floor = float(inside.min())
    field = np.where(object_mask, smoothed, floor)
    tau = (float(inside.max()) - floor) * float(tau_fraction)

    peaks = (h_maxima(field, tau) > 0) if tau > 0 else local_maxima(field)
    peaks &= object_mask
    seeds, n_seeds = ndimage.label(peaks)
    if n_seeds == 0:                       # a perfectly flat clump: one fragment
        seeds = object_mask.astype(np.int32)
    fragments = watershed(-smoothed, markers=seeds, mask=object_mask).astype(np.int32)

    n_frag = int(fragments.max())
    n_mark = int(markers.max())
    if n_frag == 0:
        return np.zeros(markers.shape, dtype=np.int32)

    # ---- which fragments already contain a nucleus -------------------------
    flat_f, flat_m = fragments.ravel(), markers.ravel()
    both = (flat_f > 0) & (flat_m > 0)
    owner = np.zeros(n_frag + 1, dtype=np.int32)      # 0 unowned, -1 contested
    if both.any():
        pair = flat_f[both].astype(np.int64) * (n_mark + 1) + flat_m[both]
        tally = np.bincount(pair, minlength=(n_frag + 1) * (n_mark + 1))
        tally = tally.reshape(n_frag + 1, n_mark + 1)[:, 1:]
        present = (tally > 0).sum(axis=1)
        owner[present == 1] = tally[present == 1].argmax(axis=1) + 1
        owner[present > 1] = -1

    # ---- fragment adjacency, weighted by contact area ----------------------
    contacts: dict[tuple[int, int], int] = {}
    for axis in range(3):
        a = np.take(fragments, np.arange(fragments.shape[axis] - 1), axis=axis)
        b = np.take(fragments, np.arange(1, fragments.shape[axis]), axis=axis)
        touching = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[touching].ravel(), b[touching].ravel()):
            key = (int(min(u, v)), int(max(u, v)))
            contacts[key] = contacts.get(key, 0) + 1

    neighbours: dict[int, list[tuple[int, int]]] = {}
    for (u, v), n in contacts.items():
        neighbours.setdefault(u, []).append((v, n))
        neighbours.setdefault(v, []).append((u, n))

    # ---- propagate ownership outward along the strongest contact -----------
    # Repeated sweeps rather than one pass: a fragment two hops from any nucleus can only
    # be reached after its neighbour has been claimed.
    changed = True
    while changed:
        changed = False
        for f in range(1, n_frag + 1):
            if owner[f] != 0:
                continue
            claims: dict[int, int] = {}
            for nb, weight in neighbours.get(f, []):
                if owner[nb] > 0:
                    claims[int(owner[nb])] = claims.get(int(owner[nb]), 0) + weight
            if claims:
                owner[f] = max(claims, key=lambda k: claims[k])
                changed = True

    out = np.zeros(markers.shape, dtype=np.int32)
    settled = owner.copy()
    settled[settled < 0] = 0
    out[object_mask] = settled[fragments[object_mask]]

    # ---- resolve fragments holding two nuclei, and anything left over ------
    # A split must never discard material: every object voxel ends up owned, so the parts
    # always reconstitute the original object.
    unresolved = object_mask & (out == 0)
    if unresolved.any():
        filled = watershed(-smoothed, markers=markers, mask=object_mask).astype(np.int32)
        out[unresolved] = filled[unresolved]

    return out


def split_by_tomato(
    intensity: np.ndarray,
    object_mask: np.ndarray,
    markers: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    radius: float = 1.8,
    merge_threshold: float | None = None,
    min_merge_threshold: float = 0.01,
    physical_radius: bool = False,
) -> np.ndarray:
    """PrestoCell's actual PBC — gudhi ToMATo with its adaptive threshold search.

    Ported from `main.py:556-680`. The mechanism, which `split_by_pbc` above approximates
    and gets wrong in one decisive respect:

    1. ToMATo clusters the clump: a **radius graph over foreground voxel coordinates**,
       weighted by raw intensity as the density (`main.py:613-619`).
    2. A cluster belongs to a nucleus if it overlaps that nucleus **and no other**
       (`main.py:600`). Clusters touching several go to their best-overlapping nucleus.
    3. **If any nucleus has no exclusive cluster, `merge_threshold` is halved and the whole
       clump re-clustered** (`main.py:611`), until every nucleus is separated or the
       threshold falls below 0.01.
    4. Remaining clusters are claimed round by round from those seeds, by adjacency
       (`main.py:664-678`).

    Step 3 is what `DESIGN_RATIONALE.md` §4.3 said "drops out", and it does not — it is what
    makes PBC work. `split_by_pbc` uses a fixed τ instead, and measurement showed its entire
    apparent advantage over watershed lived in that constant, vanishing at τ ≥ 0.10. Here τ
    is *searched per clump*, so there is no constant left to get wrong.

    **On `metric='euclidean'` over voxel coordinates — this was first written up here as an
    anisotropy defect, and that was wrong.** The correction matters, because "port the rule,
    fix the bug" is the project's habit and this one does not qualify:

    - 3DMorph's two defects (§5.1-5.2) are arithmetic errors: they compute a branch length
      *in microns* and get the number wrong. ToMATo's radius graph computes no physical
      quantity at all. It fixes which voxels are neighbours, and the clustering then runs on
      the level-set topology of the density over that graph. Physical distance never enters
      the persistence calculation, so there is no wrong µm to correct.
    - `r=1.8` in voxel space is almost exactly the 26-neighbourhood — `(1,1,1)` is 1.73,
      `(2,0,0)` is 2.0 — i.e. plain voxel-lattice adjacency, the standard choice for
      grid-sampled data.
    - At 5.5:1 there is no physical radius that behaves. Connecting one z step needs
      r >= 0.6 µm, which reaches 5.5 voxels in xy (~285 neighbours per point against 26) and
      smears the clustering in-plane; anything smaller disconnects z entirely.

    ``physical_radius=True`` scales coordinates to microns so the alternative can be
    *measured* rather than argued. The default stays on lattice adjacency, which is a
    deliberate choice here and not an inherited defect.
    """
    from gudhi.clustering.tomato import Tomato

    if not (intensity.shape == object_mask.shape == markers.shape):
        raise ValueError(
            f"shapes must match: intensity {intensity.shape}, "
            f"mask {object_mask.shape}, markers {markers.shape}"
        )

    coords = np.argwhere(object_mask)
    if coords.size == 0:
        return np.zeros(markers.shape, dtype=np.int32)
    density = intensity[object_mask].astype(np.float64)

    points = coords.astype(np.float64)
    if physical_radius:
        points = points * np.asarray([float(v) for v in voxel_size_um])

    n_mark = int(markers.max())
    span = float(density.max() - density.min())
    tau = float(merge_threshold) if merge_threshold is not None else max(span, 1.0)

    def overlap_table_of(clusters: np.ndarray) -> np.ndarray:
        n_clu = int(clusters.max())
        both = (clusters > 0) & (markers > 0)
        if not both.any():
            return np.zeros((n_clu + 1, n_mark + 1), dtype=np.int64)
        pair = (clusters[both].astype(np.int64) * (n_mark + 1)
                + markers[both].astype(np.int64))
        return np.bincount(pair, minlength=(n_clu + 1) * (n_mark + 1)
                           ).reshape(n_clu + 1, n_mark + 1)

    clusters = np.zeros(markers.shape, dtype=np.int32)
    while True:
        model = Tomato(graph_type="radius", r=float(radius), metric="euclidean",
                       density_type="manual", merge_threshold=float(tau))
        model.fit(X=points, weights=density)
        clusters = np.zeros(markers.shape, dtype=np.int32)
        clusters[object_mask] = model.labels_ + 1

        touched = overlap_table_of(clusters)[:, 1:] > 0
        exclusive = touched.sum(axis=1) == 1
        served = np.zeros(n_mark, dtype=bool)
        for clu in np.flatnonzero(exclusive):
            if clu:
                served[int(touched[clu].argmax())] = True
        if served.all() or tau <= min_merge_threshold:
            break
        tau /= 2.0

    overlap = overlap_table_of(clusters)
    n_clu = int(clusters.max())
    owner = np.zeros(n_clu + 1, dtype=np.int32)
    for clu in range(1, n_clu + 1):
        row = overlap[clu, 1:]
        if row.any():
            owner[clu] = int(row.argmax()) + 1        # best overlap wins, as in main.py:596

    neighbours: dict[int, set[int]] = {}
    for axis in range(3):
        a = np.take(clusters, np.arange(clusters.shape[axis] - 1), axis=axis)
        b = np.take(clusters, np.arange(1, clusters.shape[axis]), axis=axis)
        live = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[live].ravel(), b[live].ravel()):
            neighbours.setdefault(int(u), set()).add(int(v))
            neighbours.setdefault(int(v), set()).add(int(u))

    while True:                       # round-by-round claiming; first claimant wins
        grew = False
        for clu in range(1, n_clu + 1):
            if owner[clu]:
                continue
            for nb in neighbours.get(clu, ()):
                if owner[nb] > 0:
                    owner[clu] = owner[nb]
                    grew = True
                    break
        if not grew:
            break

    out = np.zeros(markers.shape, dtype=np.int32)
    out[object_mask] = owner[clusters[object_mask]]
    stranded = object_mask & (out == 0)
    if stranded.any():
        filled = watershed(-intensity, markers=markers, mask=object_mask).astype(np.int32)
        out[stranded] = filled[stranded]
    return out


TOMATO_RADIUS = 1.8


def decompose_by_persistence(
    intensity: np.ndarray,
    region_mask: np.ndarray,
    *,
    idx_coef: float = 0.96,
    radius: float = TOMATO_RADIUS,
) -> np.ndarray:
    """Partition ONE region into persistence-stable parts. PrestoCell's `pbc_mid_tau`.

    Ported from `main.py:405-434`. **This is not splitting and does not belong to the
    pipeline's output** — no cell boundary moves, nothing is measured, and no column depends
    on it. It exists so Component 5's rich review can colour a cell's soma and each of its
    processes separately, and so a reviewer can point at one of them.

    It lives here rather than in `ui/` because **this module is where ToMATo is configured**,
    and the configuration is the part that must not be duplicated: `radius=1.8` on the voxel
    lattice, `metric='euclidean'`, `density_type='manual'` with raw intensity as the density.
    `split_by_tomato` above records at length why that radius is lattice adjacency rather than
    an anisotropy defect, and a second call site in another module would be free to drift from
    that reasoning with nothing to say so.

    **τ is a percentile of this region's own persistence spans, not a constant.** The first
    fit runs at a merge threshold above the whole density range, so nothing merges and the
    diagram carries every span; ``idx_coef`` then indexes the sorted spans and the second fit
    merges at that τ. So granularity adapts to the structure of the cell in hand rather than
    resting on a number tuned on somebody else's image.

    **0.96 for processes and 0.90 for parts, and PrestoCell's shipped 0.75 is far too fine**
    — 50 to 154 processes on real cells here. Measured across five cells of 324–1114 µm³:
    0.96 gives 11 · 13 · 17 · 25 · 29 processes, 0.90 gives 9 · 4 · 7 · 7 · 7 parts inside
    one process. The user chose 0.96 over B's recommended 0.99; the objection is recorded at
    `COMPONENT_05_PLAN.md` §3b and the consequence is `COMPONENT_03_PLAN.md` §6d's seventh
    constraint.

    Returns labels in ``region_mask``'s frame, numbered from 1, with 0 outside the mask.
    """
    from gudhi.clustering.tomato import Tomato

    if intensity.shape != region_mask.shape:
        raise ValueError(f"shapes must match: intensity {intensity.shape}, "
                         f"mask {region_mask.shape}")
    if not 0.0 <= idx_coef < 1.0:
        raise ValueError(f"idx_coef must be in [0, 1), got {idx_coef}")

    solid = np.asarray(region_mask).astype(bool, copy=False)
    out = np.zeros(solid.shape, dtype=np.int32)
    coords = np.argwhere(solid)
    if coords.shape[0] < 2:
        out[solid] = 1
        return out

    points = coords.astype(np.float64)
    density = np.asarray(intensity)[solid].astype(np.float64)

    def fit(tau: float) -> np.ndarray:
        model = Tomato(graph_type="radius", r=float(radius), metric="euclidean",
                       density_type="manual", merge_threshold=float(tau))
        model.fit(X=points, weights=density)
        return model

    #: **Above the whole range, so the first fit merges nothing** and the diagram holds every
    #: span there is. `main.py:416` writes `np.max(density) + 1`; the +1 matters when a region
    #: is flat, where the range is 0.
    first = fit(float(density.max()) + 1.0)
    diagram = np.asarray(first.diagram_, dtype=np.float64)
    if diagram.size == 0:
        out[solid] = np.asarray(first.labels_, dtype=np.int32) + 1
        return out

    spans = np.sort(diagram[:, 0] - diagram[:, 1])
    #: `int(len * coef)` can land one past the end at coef → 1; PrestoCell would raise there.
    tau = float(spans[min(int(len(spans) * idx_coef), len(spans) - 1)])

    out[solid] = np.asarray(fit(tau).labels_, dtype=np.int32) + 1
    return out


def decompose_by_persistence_levels(
    intensity: np.ndarray,
    region_mask: np.ndarray,
    *,
    idx_coefs: Sequence[float],
    radius: float = TOMATO_RADIUS,
) -> tuple[np.ndarray, ...]:
    """ONE ToMATo fit, cut at several τ. **The cuts are GUARANTEED NESTED.**

    `decompose_by_persistence` above fits per call, so two granularities meant two fits over two
    *different* diagrams — the fine one computed over a sub-region in isolation. Nothing made the
    fine partition a refinement of the coarse one, and the fine boundaries depended on which
    region had been entered.

    **ToMATo is hierarchical, so one fit answers every level.** Raising the merge threshold only
    ever merges; it never redraws a boundary. gudhi exposes that directly — `merge_threshold_`
    re-cuts a fitted model and `labels_` re-derives **with no refit** — so cutting one diagram at
    three τ gives three partitions where each finer one is a strict refinement of the coarser.

    Three consequences, all of them the reason this exists:

    * **Granularity is conserved by construction.** Entering a coarse region and seeing its finer
      pieces yields exactly the pieces those finer cuts contain — not approximately, and not
      subject to measurement.
    * **It is faster.** One fit per cell, rather than one per fit plus one per click.
    * **The pieces are stable.** A part is a property of the cell, not of the route taken to it.

    ``idx_coefs`` are percentiles into **this region's own** sorted persistence spans, coarsest
    first. Returns one label array per coefficient, in that order, numbered from 1 with 0 outside
    the mask.

    > ### MEASURED ON crop2, 2026-08-27 — WHY THE COEFFICIENT IS A WEAK LEVER
    >
    > Cluster counts saturate, because **ToMATo merges a cluster only into an ADJACENT higher
    > peak**: clusters in disconnected pieces of the region can never merge at any τ. On one
    > 552 µm³ cell the count ran 306 · 232 · 202 · 187 · 183 across coefficients 0.75 · 0.90 ·
    > 0.96 · 0.99 · 0.999, against **183 connected components** — so τ accounted for 4 to 123
    > subdivisions and connectivity for all the rest.
    >
    > **`COMPONENT_05_PLAN.md` §3b's 11–29 regions counts only components worth looking at.**
    > 54–61% of the components on these cells are **≤5 voxels** — a broken rind left where the
    > soma opening's boundary meets the mask's — and they are under 11% of the non-soma volume.
    > A minimum component size, not a coefficient, is the lever for those. **Deferred by the user
    > until after the soma mechanism changes**, since that subtraction is what generates them.
    """
    coefs = [float(c) for c in idx_coefs]
    if not coefs:
        raise ValueError("at least one idx_coef is required")
    for coef in coefs:
        if not 0.0 <= coef < 1.0:
            raise ValueError(f"idx_coef must be in [0, 1), got {coef}")
    if intensity.shape != region_mask.shape:
        raise ValueError(f"shapes must match: intensity {intensity.shape}, "
                         f"mask {region_mask.shape}")

    solid = np.asarray(region_mask).astype(bool, copy=False)
    blank = np.zeros(solid.shape, dtype=np.int32)
    coords = np.argwhere(solid)
    if coords.shape[0] < 2:
        one = blank.copy()
        one[solid] = 1
        return tuple(one.copy() for _ in coefs)

    from gudhi.clustering.tomato import Tomato

    points = coords.astype(np.float64)
    density = np.asarray(intensity)[solid].astype(np.float64)

    #: **Above the whole range, so this first fit merges nothing** and the diagram holds every
    #: span there is — the same construction `decompose_by_persistence` uses, kept identical so
    #: the two cannot disagree about what a span is.
    model = Tomato(graph_type="radius", r=float(radius), metric="euclidean",
                   density_type="manual", merge_threshold=float(density.max()) + 1.0)
    model.fit(X=points, weights=density)

    diagram = np.asarray(model.diagram_, dtype=np.float64)
    if diagram.size == 0:
        one = blank.copy()
        one[solid] = np.asarray(model.labels_, dtype=np.int32) + 1
        return tuple(one.copy() for _ in coefs)

    spans = np.sort(diagram[:, 0] - diagram[:, 1])
    out: list[np.ndarray] = []
    for coef in coefs:
        #: `int(len * coef)` can land one past the end as coef -> 1; PrestoCell would raise.
        tau = float(spans[min(int(len(spans) * coef), len(spans) - 1)])
        model.merge_threshold_ = tau                      # re-cut, no refit
        level = blank.copy()
        level[solid] = np.asarray(model.labels_, dtype=np.int32) + 1
        out.append(level)
    return tuple(out)


def split_objects(
    objects: Objects,
    nucleus_labels: np.ndarray,
    intensity: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    method: str = "watershed",
    smooth_um: float = 0.4,
    min_overlap_fraction: float = 0.10,
    crop_pad_um: float = 1.0,
    tau_fraction: float = 0.05,
    physical_radius: bool = False,
) -> SplitOutcome:
    """Apply the three rules across every object, returning final cell labels.

    Splitting runs inside each clump's own bounding box rather than across the whole
    volume. At ~53 M voxels after downsampling, and with only a handful of objects
    holding two or more nuclei, that is the difference between seconds and minutes.
    """
    if method not in ("watershed", "pbc", "pbc_gudhi"):
        raise ValueError(
            f"unknown split method {method!r}; expected 'watershed', 'pbc' "
            "(h_maxima approximation) or 'pbc_gudhi' (PrestoCell's ToMATo)."
        )

    object_labels, n_merged = merge_objects_sharing_a_nucleus(
        objects.labels, nucleus_labels, min_overlap_fraction=min_overlap_fraction
    )
    grouped = nuclei_per_object(
        nucleus_labels, object_labels, min_overlap_fraction=min_overlap_fraction
    )

    final = np.zeros_like(object_labels, dtype=np.int32)
    method_by_cell: dict[int, str] = {}
    nuclei_by_cell: dict[int, list[int]] = {}
    next_id = 1
    n_split = 0

    boxes = ndimage.find_objects(object_labels)
    for obj_id in range(1, int(object_labels.max()) + 1):
        box = boxes[obj_id - 1]
        if box is None:
            continue
        nuclei_here = grouped.get(obj_id, [])

        if len(nuclei_here) < 2:
            final[box][object_labels[box] == obj_id] = next_id
            method_by_cell[next_id] = "unsplit"
            nuclei_by_cell[next_id] = list(nuclei_here)
            next_id += 1
            continue

        padded = tuple(
            slice(max(0, sl.start - int(np.ceil(crop_pad_um / float(v)))),
                  min(dim, sl.stop + int(np.ceil(crop_pad_um / float(v)))))
            for sl, v, dim in zip(box, voxel_size_um, object_labels.shape)
        )
        mask = object_labels[padded] == obj_id
        markers = np.zeros(mask.shape, dtype=np.int32)
        for i, nucleus in enumerate(nuclei_here, start=1):
            markers[(nucleus_labels[padded] == nucleus) & mask] = i

        surviving = np.unique(markers)
        surviving = surviving[surviving != 0]
        if surviving.size < 2:
            # every nucleus fell outside the object once cropped and masked
            final[padded][mask] = next_id
            method_by_cell[next_id] = "unsplit"
            nuclei_by_cell[next_id] = list(nuclei_here)
            next_id += 1
            continue

        if method == "pbc_gudhi":
            parts = split_by_tomato(
                intensity[padded], mask, markers,
                voxel_size_um=voxel_size_um,
                physical_radius=physical_radius,
            )
        elif method == "pbc":
            parts = split_by_pbc(
                intensity[padded], mask, markers,
                voxel_size_um=voxel_size_um, smooth_um=smooth_um,
                tau_fraction=tau_fraction,
            )
        else:
            parts = split_by_watershed(
                intensity[padded], mask, markers,
                voxel_size_um=voxel_size_um, smooth_um=smooth_um,
            )
        n_split += 1
        for i, nucleus in enumerate(nuclei_here, start=1):
            piece = parts == i
            if not piece.any():
                continue
            final[padded][piece] = next_id
            method_by_cell[next_id] = method
            nuclei_by_cell[next_id] = [nucleus]
            next_id += 1

    return SplitOutcome(
        labels=final,
        n_cells=next_id - 1,
        method_by_cell=method_by_cell,
        nuclei_by_cell=nuclei_by_cell,
        n_objects_in=int(object_labels.max()),
        n_split=n_split,
        n_merged=n_merged,
    )
