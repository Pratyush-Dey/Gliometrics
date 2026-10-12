"""The cell editor's model — decomposition, ownership, editing, splitting, resolution.

`COMPONENT_05_PLAN.md` §3b *RICH REVIEW*, §3d.2–§3d.6, §4, §8, §9.2, §10.2.

> ### SHARED, AND BUILT ONCE — that is the whole reason this module exists
>
> **Two routes reach rich review**: item 7, inside a semi-auto session, reading from **memory**;
> and `review --pool P`, after a batch merge, reading from the **pool**. §3b settles that both
> use the same two screens from different data, so everything that is not Qt lives here and
> neither route owns it.
>
> **No Qt and no napari in this module.** The same rule `semiauto.py` follows: which region a
> click lands in, what a double-click does to material this cell does not own, whether an
> addition needs bridging and how far the bridge reaches are all decidable — and therefore
> testable — with no display anywhere.

**The soma comes from `soma.find_soma`, the decomposition from `split.decompose_by_persistence`
and a manual split from `split.split_by_watershed`.** Nothing here reimplements a computation the
package already owns; what it adds is the *review semantics* on top of them — which region you
are inside, what mutes, what a click means, and what one undo step covers.

> ### THE ONE INVARIANT WORTH STATING AT THE TOP
>
> **An addition is BRIDGED, never merely added.** §3b: a detached addition is invisible three
> ways at once — `status` stays `ok`, `n_endpoints` gains one while `branch_lengths_um` does
> not, and `ramification_index` moves by an amount that **scales with distance rather than
> size**. Bridging converts an invisible failure into a visible one: a wrong click draws a line
> across the screen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from ..soma import (SOMA_ANCHOR_UM_DEFAULT, SOMA_METHOD_DEFAULT, SOMA_MIN_SHEET_UM3_DEFAULT,
                    SOMA_SHEET_AREA_UM2_DEFAULT, SOMA_THICKNESS_UM_DEFAULT,
                    SOMA_WAIST_UM_DEFAULT, find_soma)
from ..split import (decompose_by_persistence, decompose_by_persistence_levels,
                     split_by_watershed)
from . import palette

__all__ = [
    "LEVEL_IDX_COEFS", "MAX_DEPTH", "PROCESS_IDX_COEF", "SOMA_LABEL", "FIRST_PROCESS_LABEL",
    "NO_SOMA_NOTE", "BRIDGE_WIDTH_UM", "SEVER_ANCHORS", "touching",
    "soma_region", "mask_stamp", "decompose_cell", "decompose_cell_levels", "sever_to_anchor",
    "region_adjacency", "region_colours", "region_alphas",
    "first_hit_along_ray", "OWNERSHIP", "double_click_action", "bridge_path", "bridged",
    "Operation", "EditRecord", "CellEditor",
    "REJECT_REASONS", "Resolution", "reject_question", "count_question",
    "split_progress", "GESTURE_LEGEND", "walk_ray", "bridge_length_um", "bridge_question",
    "unavailable_note", "REMOVAL_REFUSALS",
]

#: **THREE levels, coarsest first — the user's decision 2026-08-27.** Two levels were `0.96`
#: for processes and `0.90` for parts; the reviewer now descends twice.
#:
#: **They are cuts of ONE persistence diagram**, not three independent clusterings — see
#: `split.decompose_by_persistence_levels`. ToMATo is hierarchical, so raising τ only merges and
#: each finer cut is a strict refinement of the coarser. That is what makes the user's
#: requirement exact rather than hoped for: *entering a merged region shows precisely the finer
#: pieces it contains*, so nothing is lost by coarsening the top.
#:
#: **Chosen from the subdivisions each coefficient adds ABOVE the connectivity floor**, measured
#: on crop2 2026-08-27 over three cells of 552–1806 µm³:
#:
#: | | 0.99 | 0.97 | 0.96 | 0.90 | 0.75 |
#: |---|---|---|---|---|---|
#: | subdivisions above floor | +4 · +6 · +16 | *interpolated* | +19 · +26 · +70 | +49 · +67 · +176 | +123 · +168 · +442 |
#:
#: The steps are consistent across cells — **0.99→0.96 is ~4.5x, 0.96→0.90 is ~2.6x** — so the
#: shipped pair was an uneven ladder. **0.97 is the value that evens it** at ~3.3x per step, and
#: it interpolated to 0.969 on all three cells independently. **0.85 rather than 0.90 because
#: the low end does NOT saturate**: 0.75 still yields ~2.5x more subdivision than 0.90, so there
#: is real headroom below the old floor, without reaching PrestoCell's shipped 0.75 — rejected
#: as far too fine at 50–154 regions.
#:
#: > ### THE COEFFICIENT IS A WEAK LEVER, AND §3b's NUMBERS DO NOT REPRODUCE HERE
#: >
#: > `COMPONENT_05_PLAN.md` §3b records 11–29 regions at 0.96 and 5–11 at 0.99. **Measured on
#: > crop2 the totals are 202 and 187** on one 552 µm³ cell. The difference is **not** the
#: > coefficient: 183 of those are connected components, which no τ can merge because ToMATo
#: > merges only into an *adjacent* higher peak. §3b counts components worth looking at; 54–61%
#: > of the real ones are ≤5 voxels and under 11% of the non-soma volume.
#: >
#: > **A minimum component size is the lever for those, not a coefficient. Deferred by the user
#: > 2026-08-27 until after the soma mechanism changes**, because `rest = mask & ~soma` is what
#: > generates the rind — so the threshold would only need re-tuning afterwards.
LEVEL_IDX_COEFS: tuple[float, ...] = (0.99, 0.97, 0.85)

#: How deep `enter` can descend. Two entries from the top level, because there are three cuts.
MAX_DEPTH = len(LEVEL_IDX_COEFS) - 1

#: The top level, kept as a name because `decompose_cell` takes one coefficient and the tests
#: and the plan both talk about "the process level".
PROCESS_IDX_COEF = LEVEL_IDX_COEFS[0]

#: **The soma is always label 1 and processes start at 2**, so `labels == SOMA_LABEL` is the
#: white region on every screen and in every test without a lookup.
SOMA_LABEL = 1
FIRST_PROCESS_LABEL = 2

#: §3b: *"A cell with no soma reads as one process with no soma colour, and the screen says
#: why."* This is the sentence.
NO_SOMA_NOTE = "no soma — erosion consumed the whole cell"

#: Thickness of a drawn bridge, **in microns, not voxels** — the user's value, 2026-09-01.
#:
#: > ### IT WAS 3 VOXELS, WHICH IS 1.8 µm DEEP AND 0.33 µm ACROSS
#: >
#: > `BRIDGE_WIDTH_VOXELS = 3` dilated in index space, so the fabricated material came out
#: > **5.5x thicker through depth than in plane** at this study's geometry — and that material
#: > enters the mask, so it is then skeletonised, volume-counted and measured for CD68 as if it
#: > were cytoplasm. A voxel is not a unit of length here.
#:
#: 0.5 µm is deliberately **thinner than a real process** (~0.6–1.5 µm at this resolution):
#: a bridge is fabricated material, so it should maintain connectivity while adding as little
#: as possible. What is actually drawn is **0.6 µm in z** — one slice, the thinnest thing the
#: sampling can express — and **~0.54 µm in plane**, so it is very nearly isotropic rather than
#: a slab. A 3 µm bridge adds ~0.6 µm³ against a cell of several hundred.
BRIDGE_WIDTH_UM = 0.5


# --------------------------------------------------------------------------
# The soma, and the decomposition
# --------------------------------------------------------------------------

def soma_region(cell_mask: np.ndarray, nucleus_mask: np.ndarray | None = None, *,
                voxel_size_um: Sequence[float],
                thickness_um: float = SOMA_THICKNESS_UM_DEFAULT,
                anchor_um: float = SOMA_ANCHOR_UM_DEFAULT,
                method: str = SOMA_METHOD_DEFAULT,
                sheet_area_um2: float = SOMA_SHEET_AREA_UM2_DEFAULT,
                waist_um: float = SOMA_WAIST_UM_DEFAULT,
                min_sheet_um3: float = SOMA_MIN_SHEET_UM3_DEFAULT) -> np.ndarray | None:
    """The soma as a boolean array, **from `find_soma` rather than from a second opening.**

    §3b: *"what a reviewer sees coloured as soma is what the export measures."* That holds only
    if one construction serves both, which is why `SomaResult` carries the region rather than
    this module rebuilding it.

    > ### ⇒ WHAT THAT REGION IS HAS CHANGED, AND THIS FUNCTION DELIBERATELY DOES NOT SAY
    >
    > It used to read *"the opening of the cell mask at `soma_thickness_um`, the definition the
    > user chose from five options on 2026-08-12"*. **That is now one of two**, and not the
    > default: `soma.SOMA_METHODS` carries `opening` and `anchor_sheet`, and the default moved
    > to the second on 2026-08-30. **Corrected 2026-08-31** — this said *three*, naming
    > `bounded_occupancy`, which was deleted with both its parameters on 2026-08-30.
    >
    > ### ⇒ THE PLUMBING GAP WAS CLOSED ON 2026-08-30, AND IT WAS WIDER THAN RECORDED
    >
    > This read *"this screen takes `find_soma`'s own defaults rather than being handed the
    > parameters object"*, and scoped the damage to *"`review --pool` with a non-default
    > parameters file"*. **The semi-automatic path had it too**, and worse: `app.py` never passed
    > `thickness_um` either, so 7b drew the soma at built-in defaults whatever the session had
    > chosen, while `Derivation.payloads` measured it at the chosen values. They agreed only
    > because nothing had ever moved off a default.
    >
    > **§3b is the rule that breaks** — *"what a reviewer sees coloured as soma is what the
    > export measures"* — and it broke silently, in the direction that looks fine. Building item
    > 6.5 on top of it would have shipped a screen whose own "?" points at a picture its numbers
    > could not move.
    >
    > Every parameter is now passed from `Derivation`, which is the same object `payloads`
    > measures from, so the two cannot disagree by construction rather than by discipline.

    `None` when nothing survived. **Not an empty array**: empty would say the soma is empty,
    and the honest statement is that there is not one.
    """
    result = find_soma(np.asarray(cell_mask).astype(bool), nucleus_mask,
                       voxel_size_um=voxel_size_um, thickness_um=thickness_um,
                       anchor_um=anchor_um,
                       method=method, sheet_area_um2=sheet_area_um2,
                       waist_um=waist_um, min_sheet_um3=min_sheet_um3)
    if result.region is None or not np.any(result.region):
        return None
    return np.asarray(result.region).astype(bool)


def mask_stamp(mask, bbox=None) -> str:
    """A fingerprint of the exact voxels of one cell's mask. **Cheap, and exact.**

    > ### ⇒ TWO THINGS IN THE REVIEW PATCH NEED THE SAME STAMP, SO THERE IS ONE
    >
    > `HANDOFF_PATCH_CPU.md` §4. **B7** caches a neighbour's soma, which is only valid while
    > that neighbour's mask has not changed. **B5c** rebuilds a prefetched `CellEditor` whose
    > mask no longer matches. Both are the same question — *is this still the array the work
    > was done from?* — and a second answer to it is a second thing to keep in step.
    >
    > **Why a hash and not a voxel count.** `apply_cell_edit`'s own note: *"a voxel carries one
    > label, so painting a neighbour's process into this cell removes it from the neighbour by
    > construction."* A transfer usually moves the count, but a stroke that adds one voxel and
    > takes another away does not — and the failure that would cause is silent, an editor
    > opening on the pre-transfer shape with nothing to say so. A digest cannot miss it.
    >
    > **The cost is nothing beside what it guards.** `np.packbits` gives one bit per voxel and
    > blake2b runs at hundreds of MB/s, so a few-MB crop stamps in about a millisecond — against
    > `soma_region` at **~0.4 s** and a `CellEditor` at **~1.8 s**.
    >
    > `bbox` is folded in because a mask that is identical but sits somewhere else is a
    > different cell region, and the stamp has to say so.
    """
    array = np.ascontiguousarray(np.asarray(mask).astype(bool))
    digest = hashlib.blake2b(np.packbits(array).tobytes(), digest_size=16)
    digest.update(repr((array.shape, tuple(bbox) if bbox is not None else None)).encode())
    return digest.hexdigest()


def decompose_cell(intensity: np.ndarray, cell_mask: np.ndarray,
                   soma: np.ndarray | None = None, *,
                   idx_coef: float = PROCESS_IDX_COEF) -> np.ndarray:
    """Label the cell: **soma as `SOMA_LABEL`, each process from `FIRST_PROCESS_LABEL` up.**

    The soma is carved out *before* clustering rather than being one cluster among many, because
    it is not a persistence cluster at all — it is whatever `find_soma` returns, and it is not
    editable. Clustering the whole cell and then asking which cluster was the soma would make a
    settled region depend on a τ search.

    **A cell with no soma is ONE process**, per §3b's *"a thin fragment with no soma reads as one
    process with no soma colour"*. Taken literally: such a cell is a fragment, and decomposing a
    fragment into parts labelled as processes would claim a structure the cell does not have.
    *The alternative — decompose as usual and simply draw no white region — is one line, and is
    what to change if a real fragment turns out to want it.*
    """
    intensity = np.asarray(intensity)
    solid = np.asarray(cell_mask).astype(bool)
    labels = np.zeros(solid.shape, dtype=np.int32)
    if not solid.any():
        return labels

    if soma is None or not np.any(soma):
        labels[solid] = FIRST_PROCESS_LABEL
        return labels

    body = np.asarray(soma).astype(bool) & solid
    rest = solid & ~body
    labels[body] = SOMA_LABEL
    if rest.any():
        parts = decompose_by_persistence(intensity, rest, idx_coef=idx_coef)
        labels[rest] = parts[rest] + (FIRST_PROCESS_LABEL - 1)
    return labels


def decompose_cell_levels(intensity: np.ndarray, cell_mask: np.ndarray,
                          soma: np.ndarray | None = None, *,
                          idx_coefs: Sequence[float] = LEVEL_IDX_COEFS,
                          ) -> tuple[np.ndarray, ...]:
    """`decompose_cell` at every level at once, **from ONE persistence fit.**

    Returns one label array per coefficient, coarsest first, each with the soma at `SOMA_LABEL`
    and regions from `FIRST_PROCESS_LABEL` up. **The soma is identical in all of them** — it
    comes from `find_soma` rather than from the persistence fit, so it does not subdivide.

    **The finer arrays are strict refinements of the coarser**, inherited from
    `split.decompose_by_persistence_levels`. That is the property the three-level navigation
    rests on: entering a region and seeing its pieces is a lookup, not a re-clustering, so what
    the reviewer reaches is the same however they got there.

    A cell with no soma is **one region at every level**, per §3b's *"a thin fragment with no
    soma reads as one process with no soma colour"* — decomposing a fragment would claim a
    structure it does not have, and that argument does not change with depth.
    """
    coefs = tuple(float(c) for c in idx_coefs)
    intensity = np.asarray(intensity)
    solid = np.asarray(cell_mask).astype(bool)
    blank = np.zeros(solid.shape, dtype=np.int32)

    if not solid.any():
        return tuple(blank.copy() for _ in coefs)

    if soma is None or not np.any(soma):
        one = blank.copy()
        one[solid] = FIRST_PROCESS_LABEL
        return tuple(one.copy() for _ in coefs)

    body = np.asarray(soma).astype(bool) & solid
    rest = solid & ~body
    if not rest.any():
        one = blank.copy()
        one[body] = SOMA_LABEL
        return tuple(one.copy() for _ in coefs)

    levels = decompose_by_persistence_levels(intensity, rest, idx_coefs=coefs)
    out = []
    for level in levels:
        labels = blank.copy()
        labels[body] = SOMA_LABEL
        labels[rest] = level[rest] + (FIRST_PROCESS_LABEL - 1)
        out.append(labels)
    return tuple(out)


#: The order `sever_to_anchor` tries, and the user's decision 2026-08-27: *"soma, and yes nucleus
#: is fallback."* `largest` is the third, added because a cell can have neither — `find_soma`
#: returns `centroid_fallback` with **no region at all** when erosion consumes the cell, and a
#: two-channel study has no nuclei. Leaving the anchor undefined there would sever arbitrarily.
SEVER_ANCHORS: tuple[str, str, str] = ("soma", "nucleus", "largest")


def sever_to_anchor(mask: np.ndarray, soma: np.ndarray | None = None,
                    nucleus: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the piece of ``mask`` still attached to the cell body. Returns `(kept, severed)`.

    **The user's decision, 2026-08-27:** deleting a region from the middle of a process should
    cut the mask there, *"so the rest of the process becomes debris"* rather than travelling with
    a cell it is no longer part of.

    **Severed material becomes debris for free, with no new storage and no schema change.**
    Debris is *measurement mask − (confirmed + rejected-real)*, recomputed in Phase C — so a
    fragment dropped from the cell's label volume is still inside the measurement mask and
    claimed by nobody, which is the definition. Nothing here writes a debris record.

    **26-connectivity**, matching `region_adjacency` and the pipeline's own object labelling: two
    pieces meeting only along a diagonal are attached, and severing them would be a surprise.

    The anchor is `SEVER_ANCHORS` in order — the component holding the most soma voxels, else the
    most nucleus voxels, else the largest. **A mask already in one piece is returned untouched**,
    so this is safe to call unconditionally.

    > ### THIS ALSO EXISTS BEFORE ANY EDIT — measured on crop2, 2026-08-27
    >
    > Two of five cells arrive from Phase A **already in two components**. That is out of scope
    > here and is recorded rather than fixed: an orphan inflates `morphometrics`' `n_endpoints`,
    > which counts skeleton endpoints *before* the reachability filter that drops them from every
    > branch length. Fixing it changes an exported column and is the user's call.
    """
    solid = np.asarray(mask).astype(bool)
    empty = np.zeros(solid.shape, dtype=bool)
    if not solid.any():
        return solid.copy(), empty

    components, n = ndimage.label(solid, structure=np.ones((3,) * solid.ndim))
    if n <= 1:
        return solid.copy(), empty

    chosen = 0
    for probe in (soma, nucleus):
        if probe is None:
            continue
        marker = np.asarray(probe).astype(bool)
        if marker.shape != solid.shape or not marker.any():
            continue
        counts = np.bincount(components[marker & solid].ravel(), minlength=n + 1)
        counts[0] = 0
        if counts.max() > 0:
            chosen = int(np.argmax(counts))
            break

    if not chosen:
        sizes = np.bincount(components.ravel(), minlength=n + 1)
        sizes[0] = 0
        chosen = int(np.argmax(sizes))

    kept = components == chosen
    return kept, solid & ~kept


#: The 13 unique offsets of the 26-neighbourhood. **26, not 6**, and the reason is what the
#: colouring is for: two regions meeting only along a diagonal still look adjacent on screen,
#: and giving them the same hue would read as one region. The pipeline's own object labelling
#: uses 26-connectivity for the same reason.
_NEIGHBOUR_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    if (dz, dy, dx) > (0, 0, 0)
)


def region_adjacency(labels: np.ndarray) -> dict[int, set[int]]:
    """Which regions touch which, at 26-connectivity. Background is not a region.

    > ### THE PAIRS ARE DEDUPLICATED BEFORE PYTHON SEES THEM — 2026-09-11
    >
    > This walked every adjacent voxel pair with differing labels, across all 13 offsets, in a
    > Python `for` loop: **0.094 s per cell entry** measured on the study's own pool, 27% of
    > everything `draw` does outside napari. A cell decomposes into 183-306 regions and the
    > *distinct* pairs among them number in the hundreds, while the voxel pairs that witness
    > them number in the millions — so the loop re-added the same edge thousands of times.
    >
    > `np.unique` over the stacked pairs collapses that to the distinct edges first, and the
    > Python loop then runs once per edge. **The graph produced is identical**, which is what
    > `test_touching_regions_are_adjacent` and the colouring tests check.
    """
    labels = np.asarray(labels)
    graph: dict[int, set[int]] = {int(i): set() for i in np.unique(labels) if i}
    for dz, dy, dx in _NEIGHBOUR_OFFSETS:
        a = labels[max(dz, 0):labels.shape[0] + min(dz, 0),
                   max(dy, 0):labels.shape[1] + min(dy, 0),
                   max(dx, 0):labels.shape[2] + min(dx, 0)]
        b = labels[max(-dz, 0):labels.shape[0] + min(-dz, 0),
                   max(-dy, 0):labels.shape[1] + min(-dy, 0),
                   max(-dx, 0):labels.shape[2] + min(-dx, 0)]
        live = (a > 0) & (b > 0) & (a != b)
        if not live.any():
            continue
        #: One row per distinct touching pair, rather than one per touching voxel.
        pairs = np.unique(np.stack((a[live].ravel(), b[live].ravel()), axis=1), axis=0)
        for u, v in pairs.tolist():
            graph[int(u)].add(int(v))
            graph[int(v)].add(int(u))
    return graph


def region_colours(labels: np.ndarray, ring: Sequence[str] = palette.WARM_SIX) -> dict[int, str]:
    """**Soma white, processes graph-coloured from the warm six so touching regions differ.**

    §3d.2 splits `LABEL_RING` in half and gives the decomposition the warm end; §3d.3 gives the
    soma white, *"the one region a dichromat can always find"* — it differs in **lightness**,
    which is the channel dichromacy leaves intact, and it is also the only named region.

    **Six hues against up to ~29 regions is enough because of the SHAPE of the graph, not
    because of its size.** Processes radiate from a soma and mostly touch only the soma and
    their own neighbours, so the adjacency graph is close to a star and two or three colours
    already suffice. That is what §6d constraint 7's *"cannot be solved by hue alone"* is
    answered by: not more hues, but never reusing one where it would show.

    > ### AMONG THE LEGAL HUES, THE LEAST-USED ONE WINS — and that is not a tidiness rule
    >
    > **The graph constraint alone is satisfied by giving every non-adjacent process the same
    > hue**, and a microglion's processes touch only the soma, so *"first free hue"* paints a
    > whole cell in one pink. Measured on a six-process cell: 6 processes, 1 hue, zero adjacent
    > pairs sharing — **legal, and useless.**
    >
    > §3b asks for something stronger than the constraint: *"soma one colour, **each process its
    > own**."* Spreading across the ring delivers both — no touching pair shares, and the six
    > hues are actually spent. The constraint is still what is guaranteed; the spread is what
    > makes it worth having.

    **Where a proper colouring does not exist the fallback is stated rather than hidden**: the
    hue conflicting with fewest neighbours wins, and it is deterministic. Colouring in
    descending degree order is what makes that rare — the most constrained region is served
    first.
    """
    graph = region_adjacency(labels)
    chosen: dict[int, str] = {}
    if SOMA_LABEL in graph:
        chosen[SOMA_LABEL] = palette.WHITE.hex

    used = {hue: 0 for hue in ring}
    order = sorted((r for r in graph if r != SOMA_LABEL),
                   key=lambda r: (-len(graph[r]), r))
    for region in order:
        taken = {chosen[n] for n in graph[region] if n in chosen}
        free = [hue for hue in ring if hue not in taken]
        if free:
            pick = min(free, key=lambda hue: (used[hue], ring.index(hue)))
        else:
            counts = {hue: sum(1 for n in graph[region] if chosen.get(n) == hue)
                      for hue in ring}
            pick = min(ring, key=lambda hue: (counts[hue], used[hue], ring.index(hue)))
        chosen[region] = pick
        used[pick] += 1
    return chosen


def region_alphas(regions: Sequence[int], entered: int | None = None,
                  master: float = 1.0) -> dict[int, float]:
    """§3d.4. **Entered 1.0, every other PROCESS 0.30 — and the soma does not mute.**

    A 3.3× separation, and stronger than item 4's 0.6 deliberately: item 4 separates two
    classes, here up to twenty-nine regions compete and only one is being worked on.

    **The soma keeps full strength while you are inside a process**, and that is a reading of
    §3d.4 rather than an exception to it — the rule says *"every other **process** mutes"*, and
    §3b is explicit that the soma is not a process and is not editable. It is also the
    positional anchor the whole screen rests on: *"this cell's nucleus is the one inside the
    white soma"*, which stops being true if the white recedes. *The alternative — mute the soma
    too — is one condition, and it is what to change if the entered process gets lost against
    it.*

    **`master` MULTIPLIES rather than replaces — 2026-08-25.** `this cell` was excluded from the
    translucency slider because its alpha carries meaning, and the user asked for the slider
    back. Letting it *set* the alpha would erase the entered/not-entered cue the moment it
    moved, which is the one thing on this layer alpha is for. Scaling keeps the 3.3× ratio at
    every setting, so the slider changes how much of the crop shows through and changes nothing
    about which process you are inside.
    """
    master = max(0.0, min(1.0, float(master)))
    return {int(r): master * (palette.THIS_CELL_ALPHA
                              if entered is None or int(r) == int(entered)
                              or int(r) == SOMA_LABEL
                              else palette.OTHER_PROCESS_ALPHA)
            for r in regions}


# --------------------------------------------------------------------------
# Clicking — the ray, and what a click means
# --------------------------------------------------------------------------

def first_hit_along_ray(volume: np.ndarray, start: Sequence[float], end: Sequence[float],
                        step: float = 1.0) -> int:
    """Walk the ray and return the **first non-zero voxel's value**, or 0 for a miss.

    PrestoCell's mechanism, `main.py:1826-1850`: the ray from `get_ray_intersections` is
    normalised by its **largest component** — so one unit of `step` is one voxel along the
    dominant axis — and the first non-zero value wins. A click therefore takes whatever is in
    front **from the current angle**, and you rotate to reach something occluded.

    > ### THE STEP IS 1 HERE AND PRESTOCELL'S IS 2, AND THAT IS A FIX RATHER THAN A DIFFERENCE
    >
    > `for j in np.arange(1, u_max - 2, 2)` **skips every other voxel**, so a click can pass
    > straight through a process one voxel thick — and microglial processes are thin everywhere.
    > Halving the samples saves nothing worth having: the walk is a few hundred lookups against
    > a 0.19–0.48 s decomposition on the very next line. **Asserted in both directions**, so the
    > 2 cannot come back by accident.

    **The ray tests only what is drawn**, which is what makes hiding `confirmed` cells safe
    rather than merely tidy — a hidden cell is not in the volume this is called on, so it cannot
    be clicked and needs no refusal popup. `COMPONENT_05_PLAN.md` §3c: *"the hiding rule removed
    a whole dialog rather than adding one."*
    """
    volume = np.asarray(volume)
    for index in walk_ray(start, end, step):
        if any(c < 0 or c >= dim for c, dim in zip(index, volume.shape)):
            continue
        value = int(volume[index])
        if value:
            return value
    return 0


def walk_ray(start: Sequence[float], end: Sequence[float], step: float = 1.0):
    """The voxel indices along the ray, **nearest end first**. No volume, no bounds.

    Split out of `first_hit_along_ray` on 2026-08-26 so that more than one volume can be tested
    at each sample. **That is what "nearest wins" needs**: with one volume per walk, whichever
    volume is walked first wins regardless of what is physically in front — a neighbour's
    process floating between the camera and this cell would lose to the cell behind it.

    The normalisation is unchanged and is PrestoCell's: the direction is divided by its largest
    component, so one unit of `step` is one voxel along the dominant axis.
    """
    origin = np.asarray(start, dtype=float)
    finish = np.asarray(end, dtype=float)
    direction = finish - origin
    dominant = float(np.max(np.abs(direction)))
    if dominant == 0.0:
        yield tuple(int(round(c)) for c in origin)
        return
    unit = direction / dominant
    for offset in np.arange(0.0, dominant + float(step), float(step)):
        point = origin + unit * offset
        yield tuple(int(round(c)) for c in point)


#: §3b's ownership table, and it is the data model's answer as much as the interface's.
#:
#: **`debris` adds with no popup**, because §9.4e has already deleted its arrays — its voxels
#: *"cannot be identified as owned even in principle"*. **`confirmed` cannot be clicked at all**,
#: because it is not drawn.
#: **Reachable from the canvas since 2026-08-25.** Every entry below was written when the
#: screen was, and none of it ran: `_on_canvas_double_click` passed no `status` and no
#: `material`, so every double-click defaulted to `this cell` and resolved to `remove`. It could
#: not have been otherwise — nothing but this cell was drawn until the neighbours and the debris
#: arrived on the screen in the same session as the fix.
#:
#: **`debris -> add` is live now, and the comment beside it used to say it could not be.** §9.4e
#: reasoned that debris *"has no arrays, so its voxels cannot be identified as owned even in
#: principle"* — true while debris was discarded at segmentation. Debris has a mask since the
#: patch of 2026-08-24, and the user's decision of 2026-08-25 is that it is takeable: a fragment
#: beside a cell may be a severed process, and reclaiming it is exactly the reviewer's job.
#: The screen passes `unowned` rather than `debris` for it, which is the honest word — `debris`
#: as a REJECT REASON is a verdict about a whole cell wearing the same string.
OWNERSHIP: dict[str, str] = {
    "confirmed": "hidden",
    "real_cell": "transfer",
    "debris": "add",
    "candidate": "transfer",
    "this cell": "remove",
    "unowned": "add",
}


def double_click_action(status: str) -> str:
    """`add` · `remove` · `transfer` (a popup first) · `hidden` (impossible — it is not drawn)."""
    try:
        return OWNERSHIP[status]
    except KeyError:
        raise KeyError(f"{status!r} is not one of {sorted(OWNERSHIP)}") from None


# --------------------------------------------------------------------------
# Adding — and the bridge that makes a wrong click visible
# --------------------------------------------------------------------------

def touching(cell_mask: np.ndarray, addition: np.ndarray) -> bool:
    """Whether the two are **already one connected object** under 26-connectivity.

    **Asked directly, because touching is not a distance.** This used to be expressed as a
    distance threshold — ``edt(~cell).min() <= sqrt(3)`` — which was correct only while the
    transform ran in *index* space, where sqrt(3) is the corner-neighbour separation. The
    moment that transform is given `sampling` it returns microns, and sqrt(3) becomes 1.73 µm.

    **No micron threshold can replace it either**, which is the point worth keeping. At this
    study's geometry an in-plane corner neighbour is **0.153 µm** away and a z-diagonal corner
    neighbour is **0.619 µm** away, and both are 26-adjacent — already one object. Any single
    threshold splits that class: below 0.619 the program draws a bridge between voxels that are
    already connected, above it genuinely separated material half a micron apart is called
    touching and the mask is left in two pieces. Adjacency has no such failure mode and needs
    no number.
    """
    cell = np.asarray(cell_mask).astype(bool)
    extra = np.asarray(addition).astype(bool)
    if not cell.any() or not extra.any():
        return False
    grown = ndimage.binary_dilation(cell, structure=np.ones((3, 3, 3), dtype=bool))
    return bool((grown & extra).any())


def bridge_path(cell_mask: np.ndarray, addition: np.ndarray, *,
                voxel_size_um: Sequence[float],
                width_um: float = BRIDGE_WIDTH_UM) -> np.ndarray:
    """The voxels joining `addition` to `cell_mask`. **Empty when they already touch.**

    §3b: *"There is no connectivity check, and added material is BRIDGED."* A strict touch
    requirement would make adding unowned material impossible rather than rare — `split_objects`
    assigns every voxel of an object holding a nucleus to exactly one marker, so unowned
    material is a **separate connected object by construction**, and a severed process is
    detached by definition.

    The line runs between the closest pair of voxels, which is the shortest thing that can be
    drawn and the least it can distort. **It is deliberately visible**: an accidental addition
    across the crop draws a line across the crop.

    > ### CLOSEST IN MICRONS, NOT IN VOXELS — fixed 2026-09-01 (agenda A7)
    >
    > The distance transform ran without `sampling`, so "closest" meant closest in **index**
    > space. At 5.5:1 that under-weights z by the full anisotropy: a pair two slices apart
    > (**1.20 µm**) scored 2.0 while a pair ten voxels apart in plane (**1.08 µm**, genuinely
    > nearer) scored 10.0 — so the bridge systematically preferred to reach through depth and
    > could join a pair much further apart than the one it should have chosen. The old
    > docstring's claim that the two rules *"disagree only about which of several near-equal
    > pairs is chosen"* was wrong here; they can differ by 5.54x.
    >
    > **It also made the popup understate the bridge.** `bridge_length_um` always passed
    > `sampling`, so it reported the distance to the *physically* nearest pair while the line
    > was drawn to a different one. Both now choose the same pair, so the number describes the
    > line.
    >
    > **The line is still rasterised in index space, and that is correct** — a straight line
    > stays straight under a diagonal scaling, so the drawn path is the same physical segment.
    > Only the choice of its two ends was ever wrong.
    """
    cell = np.asarray(cell_mask).astype(bool)
    extra = np.asarray(addition).astype(bool)
    path = np.zeros(cell.shape, dtype=bool)
    if not cell.any() or not extra.any():
        return path
    if touching(cell, extra):
        return path

    spacing = tuple(float(v) for v in voxel_size_um)
    distance, indices = ndimage.distance_transform_edt(
        ~cell, sampling=spacing, return_indices=True)

    coords = np.argwhere(extra)
    nearest = int(np.argmin(distance[extra]))
    tip = coords[nearest]
    anchor = np.array([axis[tuple(tip)] for axis in indices], dtype=float)

    span = anchor - tip
    n = int(max(abs(span).max(), 1)) + 1
    for i in range(n + 1):
        point = tip + span * (i / n)
        path[tuple(int(round(c)) for c in point)] = True

    #: **Thickened in microns, by a distance transform** — the same replacement `soma.py` made
    #: on 2026-08-29 when its voxelised structuring element went. `binary_dilation` iterations
    #: grow one voxel per step on every axis, which at this geometry is 0.6 µm through depth
    #: against 0.108 µm across. Here every voxel within `width_um / 2` of the line joins it, so
    #: the bridge is as near isotropic as the sampling allows.
    if width_um > 0 and path.any():
        spread = ndimage.distance_transform_edt(~path, sampling=spacing)
        path = spread <= (float(width_um) / 2.0)
    return path & ~cell


def bridge_length_um(cell_mask: np.ndarray, addition: np.ndarray,
                     voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """How far the material is from the cell, **in microns**. `0.0` when they already touch.

    This is the number the warning popup carries, and it is the honest measure of how
    far-fetched a take is: the bridge becomes part of the cell's mask, so a 20 um bridge is a
    fabricated primary process that the skeleton, the Sholl profile and the branch lengths will
    all measure as real.

    > ### THIS NUMBER AND THE DRAWN LINE NOW DESCRIBE THE SAME PAIR — 2026-09-01
    >
    > This function always passed `sampling`, and until agenda item A7 was fixed `bridge_path`
    > did not — so the two chose **different pairs**, and the number shown to the reviewer could
    > be shorter than the bridge actually drawn. It was recorded as *"a small distortion of
    > where the bridge is drawn"* on the reasoning that the two disagree only about near-equal
    > pairs; at 5.5:1 they could disagree by the full anisotropy, which is why it was fixed
    > rather than left. Both now minimise the same physical distance.
    """
    cell = np.asarray(cell_mask).astype(bool)
    extra = np.asarray(addition).astype(bool)
    if not cell.any() or not extra.any():
        return 0.0
    #: **The same `touching` rule `bridge_path` uses, so the two cannot disagree.** This was a
    #: physical threshold — one voxel's body diagonal, 0.619 µm at this geometry — chosen to
    #: express 26-adjacency in microns. It does not: in-plane material **0.5 µm away is not
    #: adjacent** but sits under that threshold, so the popup would report no gap on material
    #: `bridge_path` would draw a bridge to. Adjacency is topological and is now asked as such.
    if touching(cell, extra):
        return 0.0
    spacing = tuple(float(v) for v in voxel_size_um)
    distance = ndimage.distance_transform_edt(~cell, sampling=spacing)
    return float(distance[extra].min())


def bridge_question(gap_um: float) -> str:
    """The warning on a detached take. **It states the cost, not just the fact.**

    The reviewer is allowed through — the user's decision of 2026-08-26, *"they should be warned
    but they should be allowed to if they want"* — so the popup's job is to make the consequence
    visible rather than to argue. What it must say is that the bridge is not a drawing: it joins
    the mask, and everything measured from the mask measures it too.
    """
    return (f"This material is not touching the cell.\n\n"
            f"Taking it draws a bridge {gap_um:.1f} um long. The bridge becomes part of this "
            f"cell's mask, so it is measured as a process — in the branch lengths, the Sholl "
            f"profile and the surface area.\n\nTake it anyway?")


#: The refusals that answer a **removal** rather than a take, so the popup can say which.
#:
#: > `report_unavailable` titled every refusal *"Cannot take that"*, which was true while
#: > every refusal was a take. Item B2 gave it two that are not: a double-click on the soma,
#: > and anything else `remove` declines. A reviewer who tried to delete something was told
#: > they could not *take* it, which names the wrong gesture.
REMOVAL_REFUSALS: frozenset[str] = frozenset({"soma", "not_removable"})


def unavailable_note(status: str) -> str:
    """Why a take did not happen. **Nothing declines in silence** — the user's rule.

    Unreachable by design since the editable frame became the margin's maximum: the context box
    cannot exceed it, so every visible piece fits. It stays because *"a deliberate refusal that
    looks identical to a dead gesture is indistinguishable from a bug"*, which is the position
    this screen was in before 2026-08-26.
    """
    if status == "soma":
        #: **The user's wording, verbatim** — item B2, 2026-09-03. The second sentence is the
        #: one that matters: it names what to do instead, so the refusal is a redirection
        #: rather than a wall.
        return ("The cell body cannot be deleted. Nothing was changed. A cell with no body is "
                "not a repair — reject the cell instead if it should not be measured.")
    if status == "not_removable":
        #: **The catch-all's own sentence, the user's wording** - 2026-09-03. It used to pass
        #: the caller's take-status through, so a refused *removal* was explained as a refused
        #: *take* and the note described a boundary the reviewer had not crossed. Short on
        #: purpose: this answers a refusal the screen did not anticipate, so it must not claim
        #: to know which one.
        return "That cannot be removed. Nothing was changed."
    if status == "whole_cell":
        return ("That is the whole of the neighbouring cell, not one of its processes.\n\n"
                "Nothing was changed. Taking every part of another cell would merge the two, "
                "which is a different decision from moving a process between them.")
    which = "debris" if status == "unowned" else "material"
    return (f"That {which} reaches outside the area this cell can be edited in, so it cannot be "
            f"taken whole.\n\nNothing was changed. Taking part of it would give the cell a "
            f"process that stops at a boundary you cannot see.")


def bridged(cell_mask: np.ndarray, addition: np.ndarray, *,
            voxel_size_um: Sequence[float],
            width_um: float = BRIDGE_WIDTH_UM) -> np.ndarray:
    """`cell_mask | addition | the bridge` — the mask stays **connected**, which is the point."""
    cell = np.asarray(cell_mask).astype(bool)
    extra = np.asarray(addition).astype(bool)
    out = cell | extra
    if not cell.any() or not extra.any():
        return out

    #: **Every disconnected piece is bridged, not merely the nearest one.** Added material can
    #: arrive in more than one component — a debris clump entered and taken in parts is the
    #: ordinary case — and one bridge would leave the rest floating, which is the failure this
    #: whole function exists to prevent. Bounded by the component count, so it terminates.
    structure = np.ones((3, 3, 3), dtype=bool)
    for _ in range(int(extra.sum()) + 1):
        labels, n = ndimage.label(out, structure=structure)
        if n <= 1:
            break
        home = set(int(i) for i in np.unique(labels[cell]) if i)
        anchored = np.isin(labels, list(home))
        floating = out & ~anchored
        if not floating.any():
            break
        out = out | bridge_path(anchored, floating, voxel_size_um=voxel_size_um,
                                width_um=width_um)
    return out


# --------------------------------------------------------------------------
# The undo record — one structure, exactly as item 4's
# --------------------------------------------------------------------------

@dataclass
class Operation:
    """One undoable edit. `kind` is `remove` · `add` · `split`.

    **The voxels are held as flat indices, not as a mask**, so an episode of forty edits on a
    crop costs the voxels touched rather than forty crop-sized arrays.
    """

    kind: str
    voxels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    region: int | None = None
    n_children: int = 0

    #: **Voxels the edit DETACHED, carried in the same operation** — the user's decision
    #: 2026-08-27: *"undo restores everything to pre deletion."* Severing is a consequence of the
    #: removal, not a second act, so two operations would let `u` restore a cell to a state it
    #: was never in — the middle of a process gone but its far end back.
    severed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)


class EditRecord:
    """**The record IS the undo stack**, one structure — item 4's decision, and for its reason.

    Two structures could disagree, and the one that would be believed is the persisted one. `u`
    pops one operation per press; undo on an empty record returns `None` rather than raising,
    because pressing `u` once too often is not an error.

    **It is held in memory until the cell resolves.** §3b: *"`u` unwinds anything until
    finalisation — process edits and splits alike — because the episode is held in memory until
    it resolves, so the parent of a split has not been deleted yet."*
    """

    def __init__(self) -> None:
        self.operations: list[Operation] = []

    def __len__(self) -> int:
        return len(self.operations)

    def append(self, operation: Operation) -> Operation:
        self.operations.append(operation)
        return operation

    def has_edits(self) -> bool:
        return bool(self.operations)

    def undo(self) -> Operation | None:
        return self.operations.pop() if self.operations else None

    def kinds(self) -> list[str]:
        return [op.kind for op in self.operations]


# --------------------------------------------------------------------------
# The editor
# --------------------------------------------------------------------------

class CellEditor:
    """One cell under edit: its mask, its decomposition, what you are inside, and the record.

    **This is the whole of rich review that is not Qt.** The screen owns the layers, the pickers
    and the popups; everything about *what an edit does* is here, so `review --pool` gets it
    without a second implementation and the suite tests it without a display.
    """

    def __init__(self, intensity: np.ndarray, cell_mask: np.ndarray,
                 nucleus_mask: np.ndarray | None = None, *,
                 voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0),
                 thickness_um: float = SOMA_THICKNESS_UM_DEFAULT,
                 anchor_um: float = SOMA_ANCHOR_UM_DEFAULT,
                 method: str = SOMA_METHOD_DEFAULT,
                 sheet_area_um2: float = SOMA_SHEET_AREA_UM2_DEFAULT,
                 waist_um: float = SOMA_WAIST_UM_DEFAULT,
                 min_sheet_um3: float = SOMA_MIN_SHEET_UM3_DEFAULT,
                 idx_coefs: Sequence[float] = LEVEL_IDX_COEFS):
        self.intensity = np.asarray(intensity)
        self.mask = np.asarray(cell_mask).astype(bool).copy()
        self.nucleus_mask = nucleus_mask
        self.voxel_size_um = tuple(float(v) for v in voxel_size_um)
        self.thickness_um = float(thickness_um)
        self.anchor_um = float(anchor_um)
        #: **All four travel together**, because `_recompute` runs after every edit and a soma
        #: drawn from a different set than the export measures is §3b's failure with extra steps.
        self.soma_method = str(method)
        self.sheet_area_um2 = float(sheet_area_um2)
        self.waist_um = float(waist_um)
        self.min_sheet_um3 = float(min_sheet_um3)
        self.idx_coefs = tuple(float(c) for c in idx_coefs)

        self.record = EditRecord()
        #: **The chain of regions entered, outermost first.** Replaces the single `entered` id:
        #: with three cuts a reviewer descends twice, and one id cannot say where they are.
        self.path: list[int] = []
        self.soma: np.ndarray | None = None
        self.levels: tuple[np.ndarray, ...] = ()
        self.labels = np.zeros(self.mask.shape, dtype=np.int32)
        self.decompose()

    # -- decomposition --------------------------------------------------

    @property
    def entered(self) -> int | None:
        """The deepest region entered, or `None`. **Read-only** — `enter`/`leave` own the path."""
        return self.path[-1] if self.path else None

    @property
    def depth(self) -> int:
        """How many levels down the reviewer is. `0` at the top, `MAX_DEPTH` at the finest."""
        return len(self.path)

    def decompose(self) -> np.ndarray:
        """Re-derive the soma and **every level**. The soma is always recomputed, never edited.

        §3b: *"On a split the soma is not edited either. Each child's soma is re-derived from
        that child's own mask by the same opening. The soma is always computed, never drawn."*
        Every edit that changes the mask therefore comes back through here.

        **All three levels come from one persistence fit**, so they cost one fit per edit rather
        than one per level, and they are nested by construction.
        """
        self.soma = soma_region(self.mask, self.nucleus_mask,
                                voxel_size_um=self.voxel_size_um,
                                thickness_um=self.thickness_um,
                                anchor_um=self.anchor_um,
                                method=self.soma_method,
                                sheet_area_um2=self.sheet_area_um2,
                                waist_um=self.waist_um,
                                min_sheet_um3=self.min_sheet_um3)
        self.levels = decompose_cell_levels(self.intensity, self.mask, self.soma,
                                            idx_coefs=self.idx_coefs)
        self.labels = self.levels[0]
        #: **An edit returns you to the top**, and that is a decision rather than an omission.
        #: Re-deriving renumbers every level, so holding the path would leave the screen pointing
        #: at regions that may no longer be the same ones. Leaving is visible — the cell
        #: un-mutes — where a stale path would not be. *The alternative, re-entering whichever
        #: new region holds the old one's voxels, is worth building if deleting several pieces of
        #: one process turns out to be the common act.*
        self.path = []
        return self.labels

    def has_soma(self) -> bool:
        return self.soma is not None

    def note(self) -> str:
        """What the screen says when there is no soma, and nothing when there is."""
        return "" if self.has_soma() else NO_SOMA_NOTE

    def regions(self) -> list[int]:
        """Every region **currently on screen**, which is what a click can land in."""
        return [int(r) for r in np.unique(self.displayed_labels()) if r]

    def processes(self) -> list[int]:
        return [r for r in self.regions() if r != SOMA_LABEL]

    # -- entering -------------------------------------------------------

    def _displayed(self) -> tuple[np.ndarray, set[int]]:
        """`(labels, entered_ids)` — the drawn labelling and which ids are the innermost region.

        Walks the path, replacing the region at each step with the **next finer cut restricted to
        it**. Because the cuts are nested, that restriction is exactly the pieces the region
        contains — no re-clustering, and no dependence on the route taken.
        """
        out = self.levels[0].copy()
        entered: set[int] = set()
        for depth, region in enumerate(self.path, start=1):
            if depth >= len(self.levels):
                break
            inside = out == region
            if not inside.any():
                break
            finer = self.levels[depth]
            offset = int(out.max())
            entered = set()
            for step, value in enumerate(
                    [int(v) for v in np.unique(finer[inside]) if v], start=1):
                new = offset + step
                out[inside & (finer == value)] = new
                entered.add(new)
        return out, entered

    def enter(self, region: int) -> np.ndarray:
        """Single click. **Descends one level; the soma cannot be entered.**

        Returns the labels now drawn. At the finest level there is nothing further to descend
        into, so a click there is a no-op rather than an error — pressing on past the bottom is
        not a mistake worth raising on.
        """
        region = int(region)
        if region == SOMA_LABEL:
            raise ValueError("the soma is not a persistence cluster and cannot be entered")
        if region not in self.regions():
            raise KeyError(f"no region {region}; have {self.regions()}")
        if self.depth >= MAX_DEPTH:
            return self.displayed_labels()
        self.path.append(region)
        return self.displayed_labels()

    def leave(self) -> None:
        """Clicking the background. **Rises ONE level**, rather than all the way out.

        With two levels those were the same act. With three they are not, and climbing out one
        step keeps the reviewer's place — the alternative, clearing the whole path, throws away
        the descent they just made to reach a neighbouring piece.

        **CONFIRMED by the user 2026-08-27**, who asked for exactly this after it was offered as
        the reversible choice: *"make it so that background click takes u one level up from
        wherever you are."* No longer an open question.
        """
        if self.path:
            self.path.pop()

    def displayed(self) -> tuple[np.ndarray, dict[int, float]]:
        """`(labels, alphas)` from **one** walk of the path. §3d.4's pair, computed together.

        `draw` needs both on every redraw and used to ask for them separately, which walked
        `_displayed` — and copied the whole editable frame — twice per frame for one answer.
        The two accessors below are kept because the click handlers and `regions()` want the
        labels alone, and they are the same walk.
        """
        labels, entered = self._displayed()
        return labels, self._alphas(labels, entered)

    def _alphas(self, labels: np.ndarray, entered: set[int]) -> dict[int, float]:
        """§3d.4. **The innermost region full strength, every other region muted, soma never.**"""
        regions = [int(r) for r in np.unique(labels) if r]
        if not entered:
            return region_alphas(regions)
        return {r: (palette.THIS_CELL_ALPHA
                    if r in entered or r == SOMA_LABEL
                    else palette.OTHER_PROCESS_ALPHA)
                for r in regions}

    def displayed_labels(self) -> np.ndarray:
        """What the `this cell` layer draws: the cell, with the entered region replaced by its
        finer pieces. **Muted rather than hidden**, which is §3c's whole argument — the user
        removed a `STATE` block because *"a reviewer can see the process has split into parts"*,
        and that is only true if the rest of the cell visibly recedes.
        """
        return self._displayed()[0]

    def displayed_alphas(self) -> dict[int, float]:
        """§3d.4, for a caller that wants the alphas alone. `displayed()` is the pair."""
        labels, entered = self._displayed()
        return self._alphas(labels, entered)

    # -- editing --------------------------------------------------------

    def _sever(self) -> np.ndarray:
        """Drop whatever the mask no longer connects to the cell body. Returns severed indices.

        Called after an edit that can detach material. **`add` is exempt** — it goes through
        `bridged`, so it cannot leave anything disconnected — which is why only `remove` and
        `split` reach here, the user's decision 2026-08-27.
        """
        kept, severed = sever_to_anchor(self.mask, self.soma, self.nucleus_mask)
        if not severed.any():
            return np.zeros(0, dtype=np.int64)
        self.mask = kept
        return np.flatnonzero(severed.ravel())

    def remove(self, region: int) -> Operation:
        """Double click. **Removes the region as drawn, at whatever level you are inside.**

        The soma is refused: *"a cell with no body is not a repair."*

        **Anything the removal detaches goes with it** — cutting a piece out of the middle of a
        process leaves the far end attached to nothing, and a fragment that travels with a cell
        it is no longer part of would be counted into `cell_volume_um3` and would add an endpoint
        that contributes no branch length. It becomes debris by the ordinary definition, with
        nothing written here. See `sever_to_anchor`.
        """
        region = int(region)
        if region == SOMA_LABEL:
            raise ValueError("the soma cannot be deleted — a cell with no body is not a repair")
        #: **Ids are the DISPLAYED ones**, which is what a ray cast returns — so a double-click
        #: on a muted neighbouring region removes that region, and one inside the entered region
        #: removes just that piece. The rule is per click, not per screen state.
        target = self.displayed_labels() == region
        if not target.any():
            raise KeyError(f"nothing to remove for region {region}")

        voxels = np.flatnonzero(target.ravel() & self.mask.ravel())
        self.mask.ravel()[voxels] = False
        severed = self._sever()
        operation = self.record.append(
            Operation("remove", voxels, region=region, severed=severed))
        self.decompose()
        return operation

    def add(self, material: np.ndarray) -> Operation:
        """Double click on material this cell does not own. **Bridged, always.**"""
        extra = np.asarray(material).astype(bool)
        new_mask = bridged(self.mask, extra, voxel_size_um=self.voxel_size_um)
        voxels = np.flatnonzero(new_mask.ravel() & ~self.mask.ravel())
        self.mask = new_mask
        operation = self.record.append(Operation("add", voxels))
        self.decompose()
        return operation

    def split(self, markers: np.ndarray) -> list[np.ndarray]:
        """Shift-click a marker in each cell body, release shift. **Route B of §9.2.**

        `split_by_watershed` is the program-wide splitter, settled 2026-08-18, and it *"takes
        markers as an array and does not care where they came from"* — a reviewer's clicks are a
        valid marker array.

        Fewer than two markers does nothing, exactly as item 4's split does — one marker is a
        misfire, not an instruction.

        **Every voxel is assigned to exactly one marker, so a split never discards material**:
        the children always reconstitute the parent.

        > ### SEVERING WAS AGREED FOR SPLIT TOO, BUILT, AND THEN BACKED OUT — 2026-08-27
        >
        > The user scoped severing to *"remove and split only"*, and the split half was written.
        > **It breaks the invariant in the paragraph above**, which is documented here and
        > asserted by `test_a_split_never_discards_material` — a test whose fixture happens not
        > to produce a disconnected child, so it stayed green and the conflict was found by
        > reading rather than by failing.
        >
        > **Backed out because discarding is the wrong answer here, not merely a surprising
        > one.** Watershed can hand child A a piece reachable only through child B's territory.
        > That piece is not orphaned material — **it belongs to the sibling**, and severing it
        > would turn a misattribution into debris while the correct fix is a reassignment.
        >
        > **The fix belongs with the parked split work**, which already owes three answers
        > (child uids, a `MANUAL` member on `SplitMethod`, and `_finish`'s stranded children).
        > Give a disconnected piece to whichever sibling it touches, and only genuinely orphaned
        > material to debris. Severing on `remove` is unaffected and stands.
        """
        markers = np.asarray(markers).astype(np.int32)
        present = [int(m) for m in np.unique(markers) if m]
        if len(present) < 2:
            raise ValueError("a split needs a marker in each cell — one marker does nothing")

        parts = split_by_watershed(self.intensity, self.mask, markers,
                                   voxel_size_um=self.voxel_size_um)
        children = [(parts == m) & self.mask for m in present]
        self.record.append(Operation("split", np.flatnonzero(self.mask.ravel()),
                                     n_children=len(children)))
        return children

    def undo(self) -> Operation | None:
        """`u`. **One step per press**, and it reaches splits as well as edits.

        A removal restores what it took **and what it detached**, in one press — the user's
        decision 2026-08-27. Restoring only the removal would leave the cell in a state it was
        never in.
        """
        operation = self.record.undo()
        if operation is None:
            return None
        if operation.kind == "remove":
            self.mask.ravel()[operation.voxels] = True
            if operation.severed.size:
                self.mask.ravel()[operation.severed] = True
        elif operation.kind == "add":
            self.mask.ravel()[operation.voxels] = False
        elif operation.kind == "split":
            flat = np.zeros(self.mask.size, dtype=bool)
            flat[operation.voxels] = True
            self.mask = flat.reshape(self.mask.shape)
        self.decompose()
        return operation


# --------------------------------------------------------------------------
# Resolution — §3b, §8, §10.2
# --------------------------------------------------------------------------

#: `schema.RejectReason`'s two values, in the order the popup offers them. **Exactly two**, and
#: the enum's own docstring says why a third is worse than absent.
REJECT_REASONS: tuple[str, str] = ("real_cell", "debris")


@dataclass
class Resolution:
    """How the cell left rich review. **There is no undecided outcome** — §3a decision 4.

    That is what makes triage's *"a single cell is resolved only by entering rich review"* true.
    """

    kind: str                       # "accept" | "reject"
    reason: str | None = None       # RejectReason, on a rejection
    n_cells: int = 1                # §4.4's NEW-d, on a `real_cell` rejection
    note: str = ""                  # optional, and never required — §8

    def __post_init__(self) -> None:
        if self.kind not in ("accept", "reject"):
            raise ValueError(f"kind must be accept or reject, got {self.kind!r}")
        if self.kind == "reject" and self.reason not in REJECT_REASONS:
            raise ValueError(f"a rejection needs a reason from {REJECT_REASONS}")
        if self.n_cells < 1:
            raise ValueError("a rejection covers at least one cell")


def reject_question() -> str:
    """The first popup. **The asymmetry it draws is about COUNTING**, not permanence.

    > ### ⇒ IT CLAIMED THE IMAGE DATA WAS DELETED. THAT WAS FALSE — 2026-08-30
    >
    > The text read *"Debris — not a cell. Its image data is deleted permanently and cannot be
    > recovered"*, and this docstring justified it: §9.4e deletes the arrays, *"once the `.npz`
    > is gone, reopening a cleared file cannot restore the cell."* **§9.4e was never
    > implemented.** Nothing in `store.py` deletes a cell's arrays — the only `unlink` there is
    > temp-file cleanup in the atomic-write helper. The popup was telling reviewers that an
    > irreversible thing had happened, every time, and it had not.
    >
    > **The sentence is gone and the popup stays**, on the user's instruction: it is how the
    > reason is chosen, and §10.2's rule still holds — *"this is the one place the interface
    > must not imply a symmetry it does not have."*
    >
    > **So the real asymmetry is named instead.** It is not that one choice destroys data and
    > the other does not; today neither does. It is that a `real_cell` rejection is **counted**
    > — in `n_real_cells`, at the reviewer's own multiplicity, with a partial measurement row, a
    > territory hull and a contribution to every plaque it engages — while `debris` is left out
    > of every cell count and stores a blank rather than a number.
    >
    > **It promises no reversibility either**, deliberately. The data survives, but the
    > interface offers no way back: the tile leaves the grid on `mark_resolved`. Saying *nothing
    > is deleted* would be a new false claim pointing the other way.

    The second line prepares the reviewer for the count popup, so a second modal is never a
    surprise — `EXPORT_SPEC.md` §42 for what the number reaches.
    """
    return ("Is this a real cell, or debris?\n\n"
            "Real cell — counted as a microglia you could not use. You will be asked how many "
            "cells it is.\n\n"
            "Debris — not a cell. It is left out of every cell count.")


def count_question(default: int = 1) -> str:
    """The second popup, on `real_cell` only. **The count is authoritative, not an estimate.**

    §4.4's NEW-d: *"this is 3 cells I cannot afford to split"*. This project already treats
    human marks as ground truth, so calling the same person's cell count an estimate would be
    incoherent — there is no separate estimated column and there is not going to be one.
    """
    #: **The default is not echoed in the text.** The spin box beside it already shows the
    #: number and is where the number is changed; printing it twice made the sentence longer
    #: and told the reviewer nothing the control was not already saying.
    return ("How many real cells does this rejection cover?\n\n"
            "The number is recorded as given. It is what stops a clumped rejection reading as "
            "one lost cell when it was three.")


def split_progress(seconds: float) -> str:
    """§9.2: a manual split costs 0.7–2.5 s, *"so shift-release cannot silently produce
    children"*. One sentence, and the screen shows it while the watershed runs.
    """
    return f"Splitting — about {seconds:.1f} s"


#: §3c, and it is the largest legend in the project — seven gestures behind **one** mark.
#: *"A screen whose controls are mostly invisible gestures is better served by one honest mark
#: than by a wall of text nobody reads twice."*
GESTURE_LEGEND = """CLICK a process              enter it — it re-colours into its parts,
                             and the rest of the cell mutes

CLICK the background         leave the process you are in

DOUBLE-CLICK                 remove — a part when you are inside a process,
                             the whole process when you are not

DOUBLE-CLICK other material  add it to this cell. Material that is not owned is
                             taken directly; a candidate's or a rejected real
                             cell's asks first

SHIFT + CLICK each cell,     split this cell into that many. Markers show while
then RELEASE SHIFT           SHIFT is held. One marker does nothing

u                            undo, one step per press, until the cell is resolved

An addition is always joined to the cell by a bridge, so the mask stays connected and \
morphometrics stays coherent. A wrong click draws a line across the crop."""
