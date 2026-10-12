"""Measures read off a built `SkeletonArtefact` — the graph, not the mask.

**Why a module of its own.** `skeleton.py` *builds* the artefacts and `morphometrics.py`
measures the cell; everything here is a pure function of an artefact that was already built, so
it can be tested against a hand-written graph with no image, no soma and no thinning. That
matters because these are the measures added in the 2026-09-01 export patch and several of them
— Strahler order, centrifugal order, tortuosity — have textbook definitions worth checking
against a fixture whose answer can be worked out by hand.

**Two skeletons, two vocabularies.** The soma-anchored artefact is a rooted tree and can answer
anything defined from a root: branch order, Strahler order, trunk radius, primary process count.
The soma-free artefact is a general graph with no root, possibly several components and possibly
cycles, so it answers none of those and instead reports *segments* — the inter-junction runs
`EXPORT_SPEC.md` calls `segments_free.csv`. Functions here are named for whichever they take and
raise rather than guess if handed the wrong one.

**Aggregation rule, set by the user 2026-09-01: means over segments are LENGTH-WEIGHTED.** A
2 µm stub and a 40 µm process are not two equal observations of "a process", and an unweighted
mean says they are. The weighted form answers "what is the radius (or tortuosity) at a randomly
chosen point along the arbour", which is the quantity that survives comparison between cells of
different size.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .skeleton import NODE_ROOT, SkeletonArtefact

__all__ = [
    "edge_straight_um",
    "edge_tortuosity",
    "centrifugal_order",
    "strahler_order",
    "weighted_mean",
    "polarity_index",
    "branch_rows",
    "segment_rows",
    "summarise_anchored",
    "summarise_free",
]

#: `segment_type` on `segments_free.csv`. The root-free substitute for branch order: it says
#: what a segment connects, which is the only topological statement available without a root.
SEGMENT_TYPES = ("endpoint-endpoint", "endpoint-junction", "junction-junction")


def _root_of(art: SkeletonArtefact) -> int:
    roots = np.flatnonzero(np.asarray(art.node_role) == NODE_ROOT)
    return int(roots[0]) if roots.size else 0


def _degree(art: SkeletonArtefact) -> np.ndarray:
    n = art.n_nodes
    if not art.edge.size:
        return np.zeros(n, np.int64)
    return np.bincount(np.asarray(art.edge).ravel(), minlength=n).astype(np.int64)


def edge_straight_um(art: SkeletonArtefact) -> np.ndarray:
    """Straight-line distance between each edge's two nodes, in µm."""
    if not art.edge.size:
        return np.zeros(0, float)
    p = np.asarray(art.node_um, float)
    a, b = np.asarray(art.edge)[:, 0], np.asarray(art.edge)[:, 1]
    return np.linalg.norm(p[a] - p[b], axis=1)


def edge_tortuosity(art: SkeletonArtefact) -> np.ndarray:
    """Path length ÷ straight-line distance, per edge. **Exact, and stored nowhere.**

    `edge_length_um` is measured on the full-resolution chain *before* simplification, so it is
    the true arc length; the straight-line distance is a norm over two stored coordinates. The
    ratio therefore needs no extra storage and no second traversal — unlike the per-edge radius,
    which had to be sampled at build time because the artefact keeps no interior nodes.

    A ring's self-edge has zero straight-line distance and would divide by zero. It is `NaN`
    here rather than `inf`: a closed loop has no meaningful tortuosity, and `inf` would poison
    any mean taken over it while `NaN` is filtered by the aggregators below.
    """
    length = np.asarray(art.edge_length_um, float)
    straight = edge_straight_um(art)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(straight > 0, length / straight, np.nan)
    return np.asarray(out, float)


def weighted_mean(values: Sequence[float] | np.ndarray,
                  weights: Sequence[float] | np.ndarray) -> float:
    """Length-weighted mean, ignoring non-finite values and non-positive weights.

    `NaN` when nothing survives, never 0.0 — the export's standing rule that a blank must not be
    readable as a measurement.
    """
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if v.size == 0 or v.shape != w.shape:
        return float("nan")
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(v[ok], weights=w[ok]))


def _children(art: SkeletonArtefact) -> tuple[int, list[list[int]], list[int]]:
    """Root, children per node, and a parents-before-children ordering."""
    if art.parent is None:
        raise ValueError("centrifugal/Strahler order need the soma-anchored artefact, "
                         "which is the only one carrying parent pointers")
    n = art.n_nodes
    parent = np.asarray(art.parent, int)
    kids: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n and p != i:
            kids[p].append(i)
    root = _root_of(art)
    order = [root]
    for node in order:
        order.extend(kids[node])
    return root, kids, order


def centrifugal_order(art: SkeletonArtefact) -> np.ndarray:
    """Per node: how many **branch points** lie between it and the root.

    The root is 0, and the order rises by one every time the path passes a node with two or more
    children. It is deliberately *not* the count of edges from the root — that grows with mere
    distance, so a long unbranched process would score high while having no hierarchy at all.
    """
    root, kids, order = _children(art)
    out = np.zeros(art.n_nodes, np.int64)
    for node in order:
        branching = 1 if len(kids[node]) >= 2 else 0
        for kid in kids[node]:
            out[kid] = out[node] + branching
    return out


def strahler_order(art: SkeletonArtefact) -> np.ndarray:
    """Per node, the Horton–Strahler order. The cell's value is the **root's**.

    Every leaf is 1. Where two or more children share the highest order *n*, the parent is
    *n + 1*; where one child alone holds the highest, the parent takes *n* unchanged. So the
    order rises only when two equally substantial subtrees meet, which is what makes it a
    measure of **hierarchical depth** rather than of size: adding a twig to a large arbour leaves
    it untouched, while losing a whole balanced subtree lowers it.

    That insensitivity is why it is worth a column here specifically. `min_spur_um` now ships at
    0.0, so every short spur is retained; a spur is an order-1 leaf and propagates no change
    upward, making this the branch-complexity measure least disturbed by that choice.
    """
    _, kids, order = _children(art)
    out = np.ones(art.n_nodes, np.int64)
    for node in reversed(order):
        if not kids[node]:
            continue
        child = np.asarray([out[k] for k in kids[node]], np.int64)
        top = int(child.max())
        out[node] = top + 1 if int((child == top).sum()) >= 2 else top
    return out


def branch_rows(art: SkeletonArtefact) -> list[dict[str, Any]]:
    """One row per edge of the **soma-anchored** tree — `branches_anchored.csv`.

    A row is a *branch*: the run between one node and its parent, carrying its own length,
    order, thickness and tortuosity. That is the shape the wide `branch_lengths.csv` could not
    express, where a branch was a column position and could carry nothing but a length.
    """
    if art.parent is None:
        raise ValueError("branch_rows needs the soma-anchored artefact")
    cent = centrifugal_order(art)
    stra = strahler_order(art)
    tort = edge_tortuosity(art)
    length = np.asarray(art.edge_length_um, float)
    rad = np.asarray(art.edge_radius_um, float)
    out: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(np.asarray(art.edge)):
        #: The branch is named for its FAR end — the child — so its order is the child's.
        #: Taking the parent's would give every branch its origin's depth and make the first
        #: branch of every cell order 0.
        far = int(b) if int(art.parent[int(b)]) == int(a) else int(a)
        out.append({
            "branch_id": i + 1,
            "length_um": float(length[i]) if i < length.size else float("nan"),
            "centrifugal_order": int(cent[far]),
            "strahler_order": int(stra[far]),
            "mean_radius_um": float(rad[i]) if i < rad.size else float("nan"),
            "tortuosity": float(tort[i]) if i < tort.size else float("nan"),
        })
    return out


def segment_rows(art: SkeletonArtefact) -> list[dict[str, Any]]:
    """One row per edge of the **soma-free** graph — `segments_free.csv`.

    No order columns: this artefact has no root, so centrifugal and Strahler order are not
    defined on it. `segment_type` stands in their place, and `component_id` exists only here
    because only this artefact keeps detached material.
    """
    role = np.asarray(art.node_role)
    deg = _degree(art)
    comp = _components(art)
    tort = edge_tortuosity(art)
    length = np.asarray(art.edge_length_um, float)
    rad = np.asarray(art.edge_radius_um, float)
    out: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(np.asarray(art.edge)):
        a, b = int(a), int(b)
        ends = sorted(("endpoint" if (deg[n] <= 1 or role[n] == 1) else "junction")
                      for n in (a, b))
        out.append({
            "segment_id": i + 1,
            "component_id": int(comp[a]) + 1 if comp.size else 1,
            "segment_type": f"{ends[0]}-{ends[1]}",
            "length_um": float(length[i]) if i < length.size else float("nan"),
            "mean_radius_um": float(rad[i]) if i < rad.size else float("nan"),
            "tortuosity": float(tort[i]) if i < tort.size else float("nan"),
        })
    return out


def _components(art: SkeletonArtefact) -> np.ndarray:
    """Connected-component label per node. Union-find, so no scipy round trip for a small graph."""
    n = art.n_nodes
    if n == 0:
        return np.zeros(0, np.int64)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    for a, b in np.asarray(art.edge).reshape(-1, 2):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    roots = np.asarray([find(i) for i in range(n)])
    _, out = np.unique(roots, return_inverse=True)
    return out.astype(np.int64)


def polarity_index(art: SkeletonArtefact, origin_um: Sequence[float],
                   target_um: Sequence[float]) -> float:
    """Fraction of the arbour's **length** lying in the hemisphere that faces `target_um`.

    The user's definition, 2026-09-01, chosen over the alternative of an angle between the
    soma→plaque vector and a mean process direction. Length is the right weight because a
    single long process carries more of the cell toward a plaque than several short ones, and
    because a mean direction can be near-zero for a symmetric cell while telling you nothing
    about how much material lies on each side.

    **The reference point is the soma, not the cell's centre of mass.** `origin_um` is the soma
    centroid and `target_um` the plaque's; a hemisphere is defined by the plane through the soma
    perpendicular to the line joining them, and an edge counts by its midpoint.

    **0.5 is the no-preference value, not 0.** A cell with no orientation toward the plaque has
    half its length on each side, so the neutral reading sits in the middle of the range and
    both polarisation *toward* and *away* are visible. Returns `NaN` — never 0.5 — when there is
    no length to weigh or the two points coincide, so "no answer" cannot be read as "no
    preference".
    """
    length = np.asarray(art.edge_length_um, float)
    if not art.edge.size or not np.isfinite(length).any():
        return float("nan")
    o = np.asarray(origin_um, float)
    t = np.asarray(target_um, float)
    direction = t - o
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        return float("nan")
    direction = direction / norm

    p = np.asarray(art.node_um, float)
    edge = np.asarray(art.edge)
    midpoints = 0.5 * (p[edge[:, 0]] + p[edge[:, 1]])
    facing = (midpoints - o) @ direction > 0

    ok = np.isfinite(length)
    total = float(length[ok].sum())
    if total <= 0:
        return float("nan")
    return float(length[ok & facing].sum() / total)


def summarise_anchored(art: SkeletonArtefact) -> dict[str, Any]:
    """The per-cell columns of `cells_skeleton_anchored.csv` this module owns.

    `trunk_radius_um` is the radius of the **first segment leaving the soma** (the user,
    2026-09-01). Where a cell has several primary processes there are several such segments, and
    this reports their plain mean — *not* length-weighted, because trunk radius is a statement
    about thickness at the soma and a long thin trunk should not outvote a short thick one.
    **ASSUMPTION**: the user's answer settled the single-process reading; the multi-process
    aggregation is this module's choice and is flagged in `docs/BUILD_STATE_EXPORTS.md`.
    """
    if art is None or art.n_nodes == 0:
        return {}
    root, kids, _ = _children(art)
    length = np.asarray(art.edge_length_um, float)
    rad = np.asarray(art.edge_radius_um, float)
    radmax = np.asarray(art.edge_radius_max_um, float)
    cent, stra = centrifugal_order(art), strahler_order(art)
    edge = np.asarray(art.edge)

    incident = (np.flatnonzero((edge[:, 0] == root) | (edge[:, 1] == root))
                if edge.size else np.zeros(0, np.int64))
    trunk = rad[incident] if incident.size and rad.size == edge.shape[0] else np.zeros(0)
    trunk = trunk[np.isfinite(trunk)]

    return {
        "n_primary_processes": int(len(kids[root])),
        "max_centrifugal_order": int(cent.max()) if cent.size else 0,
        "mean_centrifugal_order": float(cent.mean()) if cent.size else float("nan"),
        "strahler_order": int(stra[root]) if stra.size else 0,
        "anchored_mean_process_radius_um": weighted_mean(rad, length),
        "anchored_max_process_radius_um": (float(np.nanmax(radmax))
                                           if radmax.size and np.isfinite(radmax).any()
                                           else float("nan")),
        "trunk_radius_um": float(trunk.mean()) if trunk.size else float("nan"),
        "anchored_mean_tortuosity": weighted_mean(edge_tortuosity(art), length),
    }


def summarise_free(art: SkeletonArtefact) -> dict[str, Any]:
    """The per-cell columns of `cells_skeleton_free.csv` this module owns.

    `n_segments` is a row count of `segments_free.csv` and is carried anyway: the two files are
    read separately, and a per-cell sheet that cannot say how many segments a cell had would
    force a join to answer the simplest possible question about it.
    """
    if art is None or art.n_nodes == 0:
        return {}
    length = np.asarray(art.edge_length_um, float)
    rad = np.asarray(art.edge_radius_um, float)
    radmax = np.asarray(art.edge_radius_max_um, float)
    finite = length[np.isfinite(length)]
    return {
        "n_segments": int(length.size),
        "mean_segment_um": float(finite.mean()) if finite.size else float("nan"),
        "median_segment_um": float(np.median(finite)) if finite.size else float("nan"),
        "free_mean_process_radius_um": weighted_mean(rad, length),
        "free_max_process_radius_um": (float(np.nanmax(radmax))
                                       if radmax.size and np.isfinite(radmax).any()
                                       else float("nan")),
        "free_mean_tortuosity": weighted_mean(edge_tortuosity(art), length),
    }
