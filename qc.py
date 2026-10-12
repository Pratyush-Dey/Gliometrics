"""Segmentation quality signals, and the ranking that turns them into a review order.

**This module answers one question: which cells should a reviewer look at first?** It never
rejects a cell, never hides one and never edits one. Every number here changes the *order* the
contact sheet is built in and nothing else — the rule item 7 has carried since it was written,
because there is no correct automatic response to *"this looks wrong"*.

> ### ⇒ WHY THE NUCLEUS IS ALMOST ABSENT FROM THIS FILE — the user's decision, 2026-09-03
>
> Triage used to rank on `agreement_score` and `size_outlier_score`, and **both were
> nucleus-derived**: soma/nucleus co-location, and volume *per nucleus* against the image
> median. So the ordering was really ranking how well nucleus detection went.
>
> The nucleus channel exists in this program to drive segmentation and nothing else — it has no
> measurement threshold of its own, no equivalent of item 5, and was never built to be
> reconstructed faithfully in 3D. Ranking segmentation quality by it was ranking the wrong
> thing, and it is why a bare nucleus with a few stubs sorted *last*: one nucleus, cleanly
> matched, small volume. Nothing was wrong from the nucleus's point of view.
>
> **What survives is one binary fact** — no nucleus was assigned to this cell at all — which is
> unambiguous whatever you think of nucleus quality, and which forces the hard band rather than
> entering the ranking. `nucleus_in_soma` is the one graded nucleus signal kept, and it is a
> *containment* measure rather than a co-location one.

**Two categories, and they must measure different things.** `soma/cell` and `(cell-soma)/cell`
are the same number one minus the other, so putting one in each category would manufacture
*bad in both* out of a single observation. The soma category therefore asks *is this a
plausible soma?* using the raw volume, and the process category asks *does this cell have a
plausible amount of process?* using the fraction. The two can disagree, which is the point.

**There are no weights anywhere.** Every signal contributes by its rank, ranks are on a common
scale by construction, and each category is re-bracketed so that holding more signals cannot
buy it more influence. Nothing in this file needs a constant defended in a methods section.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

__all__ = [
    "N_BRACKETS", "CONNECTIVITY",
    "nucleus_in_soma", "soma_volume_um3", "soma_bbox_fill",
    "process_fraction", "detached_fraction",
    "triage_signals",
    "log_badness", "low_badness", "brackets", "category", "order_key",
]

#: **Twenty, against images of 100-250 cells** — the user's number, 2026-09-03. That is 5 to 12
#: cells a bracket: fine enough to separate, coarse enough that a hair's difference in a
#: heuristic does not carry real weight in the sum. Five would be coarse; a hundred would be raw
#: ranks with extra steps.
N_BRACKETS = 20

#: 26-connectivity, matching `objects.label_objects` and `markers`. A process joined to the soma
#: only at a corner is joined.
CONNECTIVITY = np.ones((3, 3, 3), bool)


def _bool(array) -> np.ndarray | None:
    """`None` for absent **and for empty**, which are the same thing to every caller here."""
    if array is None:
        return None
    out = np.asarray(array).astype(bool)
    return out if out.any() else None


# --------------------------------------------------------------------------
# The soma category — three signals
# --------------------------------------------------------------------------

def nucleus_in_soma(nucleus_mask, soma_region) -> float | None:
    """Fraction of the nucleus lying inside the soma. **One-sided: low is bad.**

    There is no such thing as too much of the nucleus being inside the soma, so unlike the two
    volume signals this one is not scored on distance from typical.

    > ### ⇒ IT IS NOT 1.0 BY CONSTRUCTION, WHICH IS THE ONLY REASON IT IS WORTH HAVING
    >
    > Checked before it was built, because the user made the decision conditional on it. The
    > soma grows from an *anchor* — the dilated nucleus — but the soma itself is thick eroded
    > material connected to that anchor, dilated back and **clipped to the cell mask**. The
    > nucleus is never copied into the output. And nuclei are upsampled nearest-neighbour and
    > deliberately **not** clipped to cell boundaries, so a nucleus can lie partly outside the
    > Iba1 mask entirely, at which point this cannot reach 1.
    >
    > **It is not a pure soma measure and is not sold as one.** A low value means the soma was
    > placed badly *or* the nucleus was segmented badly, and it cannot say which. A nucleus half
    > outside its own cell is worth a look either way, which is what earns it a place.

    `None` when there is no nucleus or no soma. A cell with no nucleus is already forced into the
    hard band, so nothing is lost by declining to score it here.
    """
    nucleus, soma = _bool(nucleus_mask), _bool(soma_region)
    if nucleus is None or soma is None or nucleus.shape != soma.shape:
        return None
    return float((nucleus & soma).sum()) / float(nucleus.sum())


def soma_volume_um3(soma_region, voxel_size_um) -> float | None:
    """The soma's volume in µm³. **Two-sided against the image median.**

    **Raw volume, not a share of the cell.** The share is one minus `process_fraction` and would
    put the same observation in both categories — see the module docstring. This asks whether the
    soma is a plausible size *for a soma*, which a cell that is nearly all soma and a cell that is
    nearly all process can answer differently from the process category.
    """
    soma = _bool(soma_region)
    if soma is None:
        return None
    return float(soma.sum()) * float(np.prod([float(v) for v in voxel_size_um]))


def soma_bbox_fill(soma_region) -> float | None:
    """How much of its own bounding box the soma fills. **One-sided: low is bad.**

    A soma is a roundish body; a sphere fills about 0.52 of its box and a compact blob rather
    more. A thin sheet or a sprawling shape fills very little, and that is the failure this
    catches — allocation that called a flat spread of membrane a cell body.

    **Bounding-box fill rather than sphericity**, for two reasons: it needs no surface extraction,
    and it is stable on objects a few voxels across, where a surface-area estimate is dominated by
    voxelisation. The measure is deliberately crude; it orders a contact sheet, it does not report
    a shape.

    Computed on the **voxel** box rather than a physical one. The anisotropy is 5.5:1 and
    identical for every cell in an image, so it shifts every score by the same factor and cannot
    change a ranking.
    """
    soma = _bool(soma_region)
    if soma is None:
        return None
    box = ndimage.find_objects(soma.astype(np.uint8))[0]
    extent = int(np.prod([sl.stop - sl.start for sl in box]))
    return float(soma.sum()) / float(extent) if extent else None


# --------------------------------------------------------------------------
# The process category — two signals
# --------------------------------------------------------------------------

def process_fraction(mask, soma_region) -> float | None:
    """The share of the cell that is not soma. **Two-sided against the image median.**

    Near zero is a bare soma with a few stubs — the failure that started this work, and which the
    old nucleus-derived ordering pushed to the *back* of the queue, because a small clean cell
    scored well on every signal triage then had.

    Near one is a cell with essentially no soma, which is equally wrong and equally worth seeing.
    Hence two-sided: half and double are the same distance in log space.

    **Exactly zero cannot be logged, and takes the worst bracket** — which is the right answer
    rather than a fallback. A cell with no process at all is the extreme case of the failure this
    signal exists to find.
    """
    cell, soma = _bool(mask), _bool(soma_region)
    if cell is None or soma is None or cell.shape != soma.shape:
        return None
    total = float(cell.sum())
    inside = float((cell & soma).sum())
    return (total - inside) / total


def detached_fraction(mask, soma_region) -> float | None:
    """Volume of the cell not connected to its own soma, as a share of the cell.

    **A fraction rather than a count of fragments, and that removes a parameter.** Counting
    fragments needs a floor below which a speck is not a process — and thresholding leaves plenty
    of two-voxel specks, so without one the signal fires on every cell. Any floor would have been
    invented: `min_object_voxels` sits far below a real process, and a new parameter would enter
    `params_hash` for a number that never touches a measurement.

    A fraction needs no floor at all. A three-voxel speck is a hundredth of a percent and sorts at
    the bottom; a process taken from a neighbour is fifteen percent and sorts at the top.

    **The mask-level counterpart of `morphometrics.detached_skeleton_um`**, which sums the edge
    length of every free-skeleton component holding no node inside the soma. That measure runs in
    Phase C, after review, so it cannot order a contact sheet — but the two should agree on a
    finished run, and disagreeing is worth knowing about.

    > ### ⇒ NO SOMA MEANS UNDEFINED, NOT MAXIMAL — mirroring the Phase C rule
    >
    > `detached_skeleton_um` returns `NaN` when there is no soma, on the user's reasoning that
    > with nothing to call *attached* every component counts as detached and the column reports
    > the cell's whole arbour as debris: *"the loudest possible wrong answer"*. Exactly the same
    > applies here, and such cells are already forced into the hard band by `centroid_fallback`.
    """
    cell, soma = _bool(mask), _bool(soma_region)
    if cell is None or soma is None or cell.shape != soma.shape:
        return None
    labels, n = ndimage.label(cell, structure=CONNECTIVITY)
    attached = [int(v) for v in np.unique(labels[cell & soma]) if v]
    if not attached:
        #: **The anchor is checked before the component count, and the order matters.** A cell in
        #: one piece whose soma lies somewhere else entirely has nothing to be attached TO, so
        #: `0.0` would report it as perfectly connected. That is the Phase C no-anchor case in a
        #: different shape and it gets the same answer: undefined. Found by the test.
        return None
    if n <= 1:
        return 0.0
    detached = float(np.isin(labels, attached, invert=True)[cell].sum())
    return detached / float(cell.sum())


# --------------------------------------------------------------------------
# Badness — turning each signal into "bigger is worse"
# --------------------------------------------------------------------------

def log_badness(values):
    """Two-sided distance from the population median, in log space.

    Half and double are the same distance from typical, which is what makes *too small* and *too
    big* rank equally badly.

    > ### ⇒ THE ONE-SIDED SCORE IS WHY BARE NUCLEI SORTED LAST
    >
    > `size_outlier_score` was a bare ratio sorted descending, so a cell at four times the median
    > scored 4.0 and led the queue while a cell at a quarter scored 0.25 and sat at the very back.
    > A nucleus with almost no processes is exactly a quarter-median cell. The ranking was
    > actively burying the thing it was meant to surface.

    `None` in, `None` out, and **a non-positive value is `None` too** — the log is undefined there
    and `brackets` turns it into the worst bracket, which is the correct reading for every signal
    that reaches this function.

    **The median is per image**, self-calibrating and needing no constant — the rule §3b already
    states for the band fractions, applied to the signals as well.
    """
    finite = [float(v) for v in values
              if v is not None and math.isfinite(float(v)) and float(v) > 0]
    if not finite:
        return [None] * len(values)
    median = float(np.median(finite))
    if median <= 0:
        return [None] * len(values)
    out = []
    for value in values:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            out.append(None)
        else:
            out.append(abs(math.log(float(value) / median)))
    return out


def low_badness(values):
    """One-sided: **a low value is bad**, so badness is its negation.

    For `nucleus_in_soma` and `soma_bbox_fill`, where there is no such thing as too much. No
    median is involved — these are already fractions of a fixed thing, so they mean the same
    across images and need no per-image calibration.
    """
    return [None if value is None or not math.isfinite(float(value)) else -float(value)
            for value in values]


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def brackets(badness, n_brackets: int = N_BRACKETS, zero_is_best: bool = False):
    """Rank a badness list into `1..n_brackets`, **1 being the worst.**

    **Ties never straddle a boundary.** Where a bracket edge falls inside a run of equal values,
    the whole run takes the worse bracket — the user's rule, and it is what makes the scheme safe
    on signals that are coarse or heavily tied. It costs a slightly larger worse bracket and buys
    the guarantee that two cells with identical evidence are never separated.

    **A missing value takes the worst bracket**, consistent with how *no nucleus matched* has
    always behaved. Giving it a middle value would hide it, which is the opposite of triage.

    `zero_is_best` reserves the best bracket for cells scoring exactly zero and ranks only the
    rest into `1..n_brackets-1`. Detachment needs this: most cells have none at all, so without it
    a tied block of perhaps 90% of the image lands wherever the first boundary falls and the
    bracketing collapses to two values. With it, the ranking discriminates within the population
    it was built for, and **a cell with any detached material can never reach the best bracket** —
    which is the *"a detached process forces the top band"* rule obtained by construction rather
    than as a special case.
    """
    n = len(badness)
    if n == 0:
        return []

    missing = [i for i, value in enumerate(badness) if value is None]
    if zero_is_best:
        clean = [i for i, value in enumerate(badness)
                 if value is not None and float(value) == 0.0]
        scored = [i for i, value in enumerate(badness)
                  if value is not None and float(value) != 0.0]
        top = n_brackets - 1
    else:
        clean, scored, top = [], [i for i, v in enumerate(badness) if v is not None], n_brackets

    out = [n_brackets] * n
    for index in missing:
        out[index] = 1

    if scored and top >= 1:
        order = sorted(scored, key=lambda i: -float(badness[i]))
        first_bracket: dict[float, int] = {}
        for position, index in enumerate(order):
            value = float(badness[index])
            if value not in first_bracket:
                #: A run of equal values takes the bracket its FIRST member falls in, which is
                #: the worse one — the tie rule, expressed as a lookup rather than a scan.
                first_bracket[value] = min(top, 1 + int(position * top / len(order)))
            out[index] = first_bracket[value]

    for index in clean:
        out[index] = n_brackets
    return out


def category(badness_lists, zero_is_best=()):
    """One category's `(bracket, raw)` per cell, from however many signals it holds.

    Each signal is bracketed on its own, the brackets are summed, and the sums are bracketed
    again — so a category arrives at the top level on the same scale whatever number of signals it
    contains. **A category holding four signals cannot outweigh one holding two simply by having
    more of them.** That self-normalisation is why this is a rank aggregation and not a weighted
    score.

    The returned `raw` is the un-bracketed sum, kept only so `order_key` can break ties with the
    detail the bracketing deliberately discarded.
    """
    if not badness_lists:
        return []
    zero_flags = set(zero_is_best)
    per_signal = [brackets(values, zero_is_best=(index in zero_flags))
                  for index, values in enumerate(badness_lists)]
    raw = [float(sum(column)) for column in zip(*per_signal)]
    #: Summed brackets are a badness in which LOW is worse, so they are flipped into the
    #: *high is worse* convention `brackets` expects. The offset keeps every value positive.
    ceiling = max(raw) + 1.0
    return list(zip(brackets([ceiling - value for value in raw]), raw))


def order_key(soma, process):
    """The final per-cell sort key: **bracket sum first, raw sum as the tiebreak.**

    A cell fourth-worst in both categories beats a cell worst in one and near-clean in the other,
    which is the property the whole scheme exists for — evidence from two independent categories
    is worth more than a single loud alarm. Rank-sum gives it without being asked for it.

    The raw sum breaks ties without ever overriding the brackets, so the fine detail the
    bracketing discarded is demoted rather than destroyed.
    """
    return [(s_bracket + p_bracket, s_raw + p_raw)
            for (s_bracket, s_raw), (p_bracket, p_raw) in zip(soma, process)]


def triage_signals(mask, nucleus_mask, soma, voxel_size_um) -> dict:
    """Every per-cell field triage ranks on, as `ReviewCell` keyword arguments.

    **The seam both review routes cross.** The semi-automatic flow has a `Derivation`; `review
    --pool` has stored arrays and a parameters file. They arrive with the same three arrays and
    a `SomaResult`, and they leave with the same dictionary — which is what makes *one ordering
    rule serves both contexts* a structural fact rather than a promise.

    **Nothing here is stored.** The user's decision of 2026-09-03: the pool keeps no QC columns
    and both routes recompute. It costs one `find_soma` per cell at review time, which the pool
    route was already paying for edited masks and which the semi-automatic route pays on a
    derivation it is holding anyway.
    """
    region = getattr(soma, "region", None)
    return {
        "has_nucleus": nucleus_mask is not None and bool(np.asarray(nucleus_mask).any()),
        "soma_method": getattr(soma, "method", "") or "",
        "nucleus_in_soma": nucleus_in_soma(nucleus_mask, region),
        "soma_volume_um3": soma_volume_um3(region, voxel_size_um),
        "soma_bbox_fill": soma_bbox_fill(region),
        "process_fraction": process_fraction(mask, region),
        "detached_fraction": detached_fraction(mask, region),
    }
