"""Skeleton artefacts as SWC, plus the sidecar that makes the pair lossless.

**SWC is a rooted tree and these skeletons are not.** Each row is
``id type x y z radius parent``: every node has exactly one parent and one root has parent
``-1``. Two things in these artefacts do not fit that.

* **Loops.** ``n_loops`` is an exported column on *both* skeleton sheets and
  ``loop_prune_method`` ships at ``none``, so cycles are retained by design. A cycle has no
  valid parent assignment, so one edge of it cannot be written.
* **The soma-free artefact has no root and may have several components**, which is what
  ``n_skeleton_components`` and ``detached_skeleton_um`` are columns for.

Everything else survives the format exactly. Coordinates and radius are native to SWC; **node
role is recoverable from degree** (1 = endpoint, 2 = slab, >=3 = junction) so it needs no
column; parent pointers are native and the anchored artefact already carries them.

So the loss is **exactly one thing**, and the sidecar carries exactly that: every edge of the
graph that is not in the spanning tree, as ``node_a,node_b,length_um``. SWC plus sidecar
reconstructs the artefact. The user chose this over a node/edge CSV pair or GraphML because it
is the only option a tracing tool opens — SNT, Vaa3D, neuTube all read SWC — and the sidecar
costs one small file per skeleton, usually empty.

**Nothing here is a fallback for the `.npz`.** The artefacts stay in the pool; this is a
handoff format, so a reader who needs the exact graph has one and a reader who wants to open a
cell in Fiji has the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .skeleton import NODE_ROOT, SkeletonArtefact

__all__ = ["artefact_from_arrays", "spanning_tree", "swc_rows", "write_swc",
           "write_extra_edges"]


def artefact_from_arrays(arrays, kind: str) -> SkeletonArtefact | None:
    """Rebuild one artefact from the flat arrays `Pool.load_skeletons` returns.

    The inverse of `SkeletonArtefact.to_arrays`, which prefixes every key with the artefact's
    kind so both ride in one `.npz`. `None` when this run wrote no artefact of that kind — a
    cell with no soma has no anchored tree, and that is an ordinary outcome rather than a fault.

    **The two radius arrays are optional**, so an artefact written before 2026-09-01 still
    loads; its per-segment radii are simply absent rather than wrong.
    """
    prefix = f"{kind}__"
    if f"{prefix}node_um" not in arrays:
        return None

    def get(name, default=None):
        value = arrays.get(prefix + name)
        return default if value is None else value

    def scalar(name, default=0):
        value = arrays.get(prefix + name)
        return default if value is None else value.item()

    return SkeletonArtefact(
        kind=kind,
        node_um=np.asarray(get("node_um")),
        node_role=np.asarray(get("node_role")),
        node_radius_um=np.asarray(get("node_radius_um")),
        edge=np.asarray(get("edge")),
        edge_length_um=np.asarray(get("edge_length_um")),
        edge_radius_um=np.asarray(get("edge_radius_um", np.zeros(0, np.float32))),
        edge_radius_max_um=np.asarray(get("edge_radius_max_um", np.zeros(0, np.float32))),
        parent=None if f"{prefix}parent" not in arrays else np.asarray(get("parent")),
        n_endpoints=int(scalar("n_endpoints")),
        n_junctions=int(scalar("n_junctions")),
        n_loops=int(scalar("n_loops")),
        n_components=int(scalar("n_components", 1)),
        status=str(scalar("status", "ok")),
    )

#: SWC's structure identifier. 1 is soma and 3 is (basal) dendrite by the format's convention;
#: microglial processes are neither, but 3 is what every viewer renders as a neurite and the
#: alternative — 0, "undefined" — makes some tools refuse to draw the arbour at all.
SWC_SOMA, SWC_PROCESS = 1, 3


def spanning_tree(art: SkeletonArtefact,
                  origin_um: Sequence[float] | None = None
                  ) -> tuple[list[int], list[int]]:
    """Parent per node and the indices of the edges the tree could not use.

    Uses the artefact's own ``parent`` array when it has one — the soma-anchored tree is already
    rooted, and re-deriving a different tree would export something the measurements were not
    taken on. The soma-free graph gets a breadth-first tree per component, rooted at the node
    nearest ``origin_um`` (the soma centroid) so the arbour is drawn outward from the cell body
    rather than from an arbitrary corner of it.

    Returns ``(parent, extra)``: ``parent[i]`` is the node index of *i*'s parent or ``-1`` for a
    root, and ``extra`` holds the edge indices that close a cycle or duplicate a tree edge.
    """
    n = art.n_nodes
    edge = np.asarray(art.edge).reshape(-1, 2)
    if n == 0:
        return [], []

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for index, (a, b) in enumerate(edge):
        a, b = int(a), int(b)
        if a == b:
            continue                     # a ring's self-edge is never a tree edge
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))

    parent = [-1] * n
    used: set[int] = set()
    seen = [False] * n

    if art.parent is not None:
        stored = np.asarray(art.parent, int)
        for i in range(n):
            p = int(stored[i])
            parent[i] = p if 0 <= p < n and p != i else -1
            seen[i] = True
        for index, (a, b) in enumerate(edge):
            a, b = int(a), int(b)
            if parent[b] == a or parent[a] == b:
                used.add(index)
    else:
        order = _roots_first(art, origin_um)
        for start in order:
            if seen[start]:
                continue
            seen[start] = True
            queue = [start]
            while queue:
                node = queue.pop(0)
                for neighbour, index in adjacency[node]:
                    if not seen[neighbour]:
                        seen[neighbour] = True
                        parent[neighbour] = node
                        used.add(index)
                        queue.append(neighbour)

    extra = [i for i in range(edge.shape[0]) if i not in used]
    return parent, extra


def _roots_first(art: SkeletonArtefact,
                 origin_um: Sequence[float] | None) -> list[int]:
    """Node indices, nearest to ``origin_um`` first, so each component roots near the soma."""
    n = art.n_nodes
    roots = np.flatnonzero(np.asarray(art.node_role) == NODE_ROOT)
    if roots.size:
        head = [int(roots[0])]
        return head + [i for i in range(n) if i != head[0]]
    if origin_um is None:
        return list(range(n))
    points = np.asarray(art.node_um, float)
    distance = np.linalg.norm(points - np.asarray(origin_um, float), axis=1)
    return [int(i) for i in np.argsort(distance)]


def swc_rows(art: SkeletonArtefact,
             origin_um: Sequence[float] | None = None
             ) -> tuple[list[tuple[int, int, float, float, float, float, int]], list[int]]:
    """SWC rows and the edge indices the format could not carry.

    **Coordinates are reordered.** The artefact stores `(z, y, x)` in µm, as everything in this
    codebase does; SWC's columns are `x y z`. Writing them in storage order would put every
    skeleton on its side in any viewer, silently.
    """
    parent, extra = spanning_tree(art, origin_um)
    if not parent:
        return [], extra
    role = np.asarray(art.node_role)
    points = np.asarray(art.node_um, float)
    radius = np.asarray(art.node_radius_um, float)
    rows = []
    for i in range(art.n_nodes):
        z, y, x = points[i]
        r = float(radius[i]) if i < radius.size and np.isfinite(radius[i]) else 0.0
        kind = SWC_SOMA if role[i] == NODE_ROOT else SWC_PROCESS
        rows.append((i + 1, kind, float(x), float(y), float(z), r,
                     parent[i] + 1 if parent[i] >= 0 else -1))
    return rows, extra


def write_swc(path: str | Path, art: SkeletonArtefact, *,
              cell_uid: str, kind: str,
              origin_um: Sequence[float] | None = None) -> Path:
    """One skeleton as SWC. Returns the path written.

    The header names the cell and the artefact, because an SWC on its own says nothing about
    which of the two skeletons it is — and the anchored and free forms of one cell differ in
    exactly the ways a reader needs to know about.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, extra = swc_rows(art, origin_um)
    lines = [
        f"# {cell_uid} — {kind} skeleton, prestomorph",
        "# coordinates and radius in microns; columns: id type x y z radius parent",
    ]
    if extra:
        lines.append(f"# {len(extra)} edge(s) omitted: SWC is a tree and this graph has "
                     f"cycles — see the _extra_edges.csv beside this file")
    for node_id, kind_id, x, y, z, r, parent in rows:
        lines.append(f"{node_id} {kind_id} {x:.4f} {y:.4f} {z:.4f} {r:.4f} {parent}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_extra_edges(path: str | Path, art: SkeletonArtefact,
                      extra: Iterable[int]) -> Path:
    """The edges SWC could not carry. **Written even when empty**, and that is deliberate.

    An absent file is ambiguous — it could mean no cycles, or that nobody wrote one — while a
    file with a header and no rows says *this skeleton is a tree* as a positive statement.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edge = np.asarray(art.edge).reshape(-1, 2)
    length = np.asarray(art.edge_length_um, float)
    lines = ["node_a,node_b,length_um"]
    for index in extra:
        a, b = edge[index]
        um = float(length[index]) if index < length.size else float("nan")
        lines.append(f"{int(a) + 1},{int(b) + 1},{um:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
