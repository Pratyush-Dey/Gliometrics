"""Locating the soma, and scoring how well the two channels agree about it.

3DMorph finds the soma by eroding the cell until only thick material survives, then
taking the **largest** remnant (`Functions/SomaCentroid.m`). Its own comment says this
is done "to find the true nucleus" — it is approximating, from one channel, something
we can measure directly from another. The approximation fails when a thick proximal
process survives erosion and happens to outweigh the soma.

Here the nucleus **anchors** the soma rather than replacing it. That distinction matters: the
anchor stays in the cell channel, where the skeleton lives. Using the nucleus centroid directly
would import a cross-channel registration offset into a point that is later snapped onto the
skeleton, and a snap that lands on a process instead of the soma corrupts every branch length
for that cell.

**Two mechanisms, and `soma_method` chooses the second.** Mechanism 1 is `soma_anchor` — the
nucleus and the cytoplasm around it, guaranteed, which no threshold can veto. Mechanism 2 decides
what *else* is soma, and it has two implementations:

* ``opening`` — the morphological opening at ``soma_thickness_um``. The original, and **the only
  one with validation behind it** (`EXPORT_SPEC.md` §18, §28, §33, §34).
* ``anchor_sheet`` — **the default since 2026-08-30**. Material whose 2D cross-section clears
  ``soma_sheet_area_um2`` in *every* plane, plus a closing that joins the soma across a waist.

*The word "selects" was accurate until 2026-08-29 and is left here as a pointer: the old rule
could only choose among the components erosion happened to leave, and had no way to split one.*

Cells with no nucleus fall back to 3DMorph's rule unchanged — they exist by design, so
the fallback is a normal path rather than an error case.

**The agreement score** is a free quality signal neither parent program has: the two
channels are independent measurements of nearly the same location, so their
disagreement is informative. It runs from 1 (co-located) down to 0, and drives the
worst-first ordering in review — which is what makes ~20,000 cells reviewable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "soma_for","SomaResult", "find_soma", "fill_nucleus_lumen", "soma_anchor",
           "min_plane_area", "SOMA_METHODS",
           "SOMA_THICKNESS_UM_DEFAULT", "SOMA_ANCHOR_UM_DEFAULT"]

#: The mechanism-2 criteria. **Both are live, so switching is a parameter value, not a revert.**
#:
#: > ### ⇒ `bounded_occupancy` WAS DELETED — the user's decision, 2026-08-30
#: >
#: > It was a trial from the day it landed and it lost the comparison it was built for. The
#: > criterion that shipped is **`min_plane_area`, AUC 0.908**, against occupancy's 0.821 — and
#: > occupancy's two parameters were the only two that `anchor_sheet` never read, so they sat on
#: > item 8 and in `params_hash` as controls that could not change anything. **Deleting the
#: > parameters deletes the mechanism**, because `soma_bound_um` and `soma_occupancy_theta`
#: > *were* the criterion; there is nothing left of it once they go.
#: >
#: > **What it was, kept because the numbers are worth having:** occupied material within
#: > `soma_bound_um` of the nucleus surface, at a local occupancy of `soma_occupancy_theta` over
#: > a 1.5 µm Gaussian neighbourhood. It was defaulted on at the user's request on 2026-08-29
#: > and superseded within the day. The measured separation of process from cell body, AUC where
#: > 0.5 is useless: **thickness 0.618**, local shape ~0.60, **occupancy 0.821**, **min plane
#: > area 0.908**.
#:
#: **`opening` stays, and it is not a courtesy.** It is the mechanism validated in
#: `EXPORT_SPEC.md` §18, §28 and §34, and §32's four soma-volume validation steps are still
#: open — re-running them needs the mechanism they were measured against. It reads
#: `soma_thickness_um` and `soma_anchor_um`, and nothing else.
#:
#: > ### ⇒ BOTH ARE OFFERED ON ITEM 6.5 — REVERSED BY THE USER, 2026-08-30
#: >
#: > This read *"a switch to it would leave the other three parameters on item 6.5 inert, and
#: > that is why `soma_method` is not offered there"*. The user reversed it hours later: the
#: > program had **no way to ask for the soma to be the nucleus and the cytoplasm around it**,
#: > which is what `opening` amounts to on these masks (§37: only 2.2% of a cell clears the
#: > thickness floor, and `soma_volume_um3` came out **83% anchor**).
#: >
#: > **The inert-box objection was answered rather than overruled.** Item 6.5's boxes follow the
#: > dropdown — `ui.screen_soma.METHOD_FIELDS` is this table in code — so choosing `opening`
#: > removes the three rather than stranding them. **A method added here must be added there**,
#: > and a test fails if it is not.
SOMA_METHODS = ("opening", "anchor_sheet")

#: **Defaults live here rather than only on `SegmentationParams`**, so that `ui.editing.soma_region`
#: — which takes its own keyword defaults and is not handed the parameters object — draws the soma
#: the export measures. The existing `soma_thickness_um` plumbing gap (`COMPONENT_05_PLAN.md`) is
#: unchanged by this and still bites only on `review --pool` with a non-default parameters file.
#: **`anchor_sheet`, and it is what item 6.5's dropdown opens on** — the default the study has
#: been walked under. It stopped being *fixed* on 2026-08-30, when the user put the switch back
#: on that screen; `SOMA_METHODS` carries what changed and why. The default itself did not move.
#:
#: > ### ⇒ `sheet` IS IMPLEMENTED AND IS NOT THE DEFAULT — measured 2026-08-29
#: >
#: > The user is right that a radius from the nucleus *"is likely trying the wrong thing"*,
#: > and `sheet` is the better-shaped idea: on a synthetic cell it reads **2.70 µm at the
#: > body against 0.60 µm along a process** and excludes the process cleanly.
#: >
#: > **It finds nothing on this study's masks.** Filled, in the best of three planes, even
#: > *perinuclear* cross-sections admit only a **0.74 µm** disc against 0.43 µm distal —
#: > AUC 0.728, below occupancy's 0.821. At a defensible 2.0 µm it returns less soma than
#: > the opening does. **There is no sheet in the mask to find.**
#: >
#: > Kept, because the criterion is sound and the input is not: on a mask that held a solid
#: > cell body this is the best of the three, and it is one parameter value away.
SOMA_METHOD_DEFAULT = "anchor_sheet"

#: How far the guaranteed soma reaches out from the nucleus, µm. **Mechanism 1's radius,
#: and nothing else's** — read under BOTH methods, because the anchor is identical under both.
#:
#: > ### ⇒ SPLIT OUT OF `soma_thickness_um` — 2026-08-31, the user's decision
#: >
#: > One number sized both mechanisms until this build, and **they pull opposite ways**:
#: > raising it dilates the anchor further (bigger soma) while making the thickness floor
#: > stricter (smaller soma). The soma/cell median therefore stops responding while what it
#: > is made of changes completely underneath — measured on crop2, the anchor goes from
#: > **44% of the soma at 0.9 to 91% at 2.0**, so two runs reporting the same volume are not
#: > measuring the same object. At the top of that range the measure has degenerated into a
#: > nucleus measurement scaled by a constant, which would track nucleus size rather than
#: > soma swelling — the study's activation readout. `SHAKEOUT_FINDINGS.md`, *THE SWEEP'S
#: > PLATEAU IS THE ANCHOR TAKING OVER*.
#: >
#: > **The coupling was not even one-to-one.** The shared number bought `r` µm of anchor
#: > reach while demanding `2r` µm of material thickness — two physical scales out of one
#: > field, locked at an arbitrary 2:1.
#: >
#: > **It defaults to the old shared value, so nothing moves until someone moves it.** What
#: > it buys is that each radius can now be swept alone, which no sweep could do before.
#:
#: **This is a floor, not a ceiling.** `find_soma` unions the anchor in *after* the criterion,
#: so nothing can take the cell body away — and nothing caps how far beyond it the soma may
#: reach. The mechanism that *did* bound the soma by a radius was `bounded_occupancy`, deleted
#: 2026-08-30 on the user's ruling that *"if our program is bounding the soma to any radius
#: from the nucleus it's likely trying the wrong thing."*
SOMA_ANCHOR_UM_DEFAULT = 0.9

#: The local half-thickness material must clear to count as soma under ``opening``, µm — so
#: **material must be 2x this across**, on every axis. Mechanism 2's radius under that method
#: and unread under `anchor_sheet`, which has its own criterion.
#:
#: **Promoted here from `ui.screen_soma` on 2026-08-31**, where it had been the odd one out:
#: the other three soma defaults already lived in this module, beside their evidence, and the
#: UI layer is the wrong home for a number the pipeline reads.
SOMA_THICKNESS_UM_DEFAULT = 0.9

#: The cross-sectional area at which material counts as a sheet rather than a line, µm²,
#: required in **every** plane. **5 µm² is a 2.5 µm disc.**
#:
#: > ### 16 -> 8 ON THE USER'S LOOK — 2026-08-29
#: >
#: > They proposed 4 µm². It shipped at 16, because at 4 the measurement put 27-38% of what
#: > it flagged more than 5 µm from the nucleus. **16 was too strict**: the user reported the
#: > soma still white only around the nucleus, which is what the numbers said would happen —
#: > at 16 the only cross-sections that qualify are perinuclear, where the anchor already
#: > reaches, so mechanism 2 contributed nothing visible.
#: >
#: > At 8, material added **beyond the anchor** sits at a median of 2.73, 1.51 and 1.12 µm
#: > from the nucleus on three of four cells — a nucleus-adjacent lobe, which is what was
#: > asked for. **The fourth cell is the exception and it is predicted:** cell 3 gains
#: > 60 µm³ at a median of 12.5 µm, a corridor running down a process to a 33 µm³ component
#: > that clears the sheet floor by three. That cell has been anomalous in every probe
#: > this session. If it shows on screen, the lever is the volume floor, not this.
#: > ### 8 -> 6 ON THE SECOND LOOK — 2026-08-30
#: >
#: > At 8 the user reported the nucleus rim still good, proximal processes still good, and
#: > *"a lot of the nucleus adjacent soma extensions have been caught"* — so the mechanism is
#: > doing its job and they asked for one more small step.
#: >
#: > **Cell 3 degrades further and it is the same cause**: its soma goes 24.9% to 39.0% and
#: > its reach 14.8 to 19.5 µm, because two more distant components clear the volume floor
#: > (55 µm³ at 18.6 µm and 39 µm³ at 14.0 µm). The other three gain at a median of 2.9, 1.5
#: > and 1.1 µm from the nucleus — still nucleus-adjacent. **The lever for cell 3 remains the
#: > volume floor or a corridor cap, not this.**
#: > ### 6 -> 4 ON THE THIRD LOOK — 2026-08-30
#: >
#: > 6 improved several cells and worsened one. Walked down to the user's original figure to
#: > see it before deciding on the volume floor.
#: >
#: > **At 4 the volume floor stops being a usable lever, and that is the thing to know.** On
#: > cell 3 the spurious component is **135 µm³ at 19.4 µm** while its real body component is
#: > **131 µm³ at 0.5 µm** — the junk is *larger than the body*, so no size test can separate
#: > them. At 6 they were 55 and 70 and a floor at 60 did separate them cleanly.
#: > Cell 2 also gains a 36 µm³ component at 9.8 µm that it does not have at 6.
SOMA_SHEET_AREA_UM2_DEFAULT = 5.0

#: **How wide a WAIST inside the soma may be and still be soma**, µm. A closing of this
#: radius joins two parts of the soma separated by up to twice it.
#:
#: > ### THE GAP THIS CLOSES — reported by the user 2026-08-30
#: >
#: > They saw **two white regions with a process-coloured band between them**. The band
#: > is material that fails the area test, with unambiguous soma on both sides, and
#: > nothing in the mechanism claimed it: the corridor bridges a *detached* sheet back
#: > to the nucleus, and has no rule for a constriction *inside* the body.
#: >
#: > **This is the bottleneck fix, in the regime where it finally means something.** It
#: > measured as recovering 1-4 µm³ on 2026-08-29 and was dropped — but that was when
#: > nothing passed the floor, so there was no soma on either side of a neck to join.
#: >
#: > It closes the **soma region**, clipped to the cell, so it cannot reach
#: > `cell_volume_um3` or the measurement mask. Set to 0 to switch it off.
#: > ### ⇒ 0 -> 3.0 ON 2026-08-30, AND THE CONTACT-COUNTING RULE WAS REMOVED
#: >
#: > A rule taking material that touches the soma **twice** was built and reverted the same
#: > day: *"that did not help at all, made things worse."* It was also **slow enough to be
#: > felt on the screen** — it labelled *cell minus soma*, which fragments into hundreds of
#: > pieces on a mask this porous, and ran a full-array dilation per piece.
#: >
#: > **3.0 µm joins soma across up to 6 µm.** That is a wide closing and it is blind by
#: > nature — it takes any cell material within reach, including a process lying alongside
#: > the body, not only material with soma on both sides.
SOMA_WAIST_UM_DEFAULT = 3.0


#: **The size a DETACHED sheet must reach to be claimed as part of this soma**, µm³.
#:
#: > ### IT APPLIES ONLY TO DETACHED PIECES — changed 2026-08-30
#: >
#: > It used to apply to every sheet component including the one containing the cell body,
#: > which meant a small cell could lose its own soma to a size test. A piece the anchor can
#: > already walk to **is** the body and is never dropped; only a piece needing a corridor
#: > has to earn its place.
#: >
#: > **The value went 30 -> 15 -> 30.** 15 was tried and made the tail worse, which is what
#: > a lower floor does: at the same 6 µm² threshold, cell 3's soma went 39.0% to 46.0% and
#: > its reach 19.5 to 20.8 µm purely from the floor change. The structural half — the floor
#: > applying only to detached pieces — was kept. The value has been walked 30 -> 15 -> 30
#: > -> 60 -> 30 against the screen; **30 µm³ is a 3.9 µm sphere.**
#:
#: > ### ⇒ IT BECAME A PARAMETER ON 2026-08-30, at the user's request
#: >
#: > It was a module constant read at one line, and the record above is a user walking it five
#: > times **by hand-editing the source**. It is named twice in these notes as *the* lever when
#: > a cell drags a spurious component down a process — *"the lever for cell 3 remains the
#: > volume floor"* — so the one number the tuning kept reaching for was the one number no
#: > screen could reach. It is now `soma_min_sheet_um3` on `SegmentationParams`, offered on item
#: > 6.5, and in `params_hash` like every other Phase A field.
SOMA_MIN_SHEET_UM3_DEFAULT = 30.0
def soma_for(settings, mask, nucleus_mask, voxel_size_um):
    """`find_soma` at a run's own settings, from anything carrying the six `soma_*` fields.

    **`SegmentationParams` and `Derivation` both carry them under identical names**, which is why
    this is duck-typed rather than taking one or the other: `Derivation.payloads` has the values
    inline, the semi-automatic review path reads them off the derivation, and `review --pool`
    reads them off the parameters file. Three call sites, one argument list.

    Written 2026-09-03, when triage started needing the soma **region** rather than only its
    centroid. The pool stores the centroid and not the region, so the region is recomputed where
    it is needed — which `review --pool` already had to do, and already demands `--params` for.
    """
    return find_soma(mask, nucleus_mask, voxel_size_um=voxel_size_um,
                     thickness_um=settings.soma_thickness_um,
                     anchor_um=settings.soma_anchor_um,
                     method=settings.soma_method,
                     sheet_area_um2=settings.soma_sheet_area_um2,
                     waist_um=settings.soma_waist_um,
                     min_sheet_um3=settings.soma_min_sheet_um3)


@dataclass
class SomaResult:
    """Where the soma is, and how much the two channels agree about it."""

    centroid: tuple[float, float, float] | None     # (z, y, x), voxel coordinates
    remnant_found: bool
    agreement_score: float | None                   # 1 = co-located, 0 = far apart
    #: Which rule produced this soma: ``nucleus_anchored`` (the opening), ``nucleus_occupancy``,
    #: ``nucleus_sheet``, or the two no-nucleus paths ``largest_remnant`` and
    #: ``centroid_fallback``. **The last two are a different definition, not a worse instance
    #: of the same one** — 3DMorph's largest-blob rule, and no soma at all. Not exported;
    #: recommended and undecided.
    method: str
    #: Volume of the soma in µm³ — **the voxel count of `region` below, whatever mechanism
    #: produced it.** `NaN` when nothing survived: never a zero, and never the cell's own
    #: volume.
    #:
    #: > ### ⇒ IT IS THE OPENING ONLY UNDER ``soma_method="opening"`` — since 2026-08-29
    #: >
    #: > This field used to define itself as *the morphological opening of the cell mask at
    #: > `thickness_um`*, which was the whole definition when there was one mechanism. There
    #: > are now three, and the default is not that one. **The paragraphs below are the record
    #: > of the opening**, which is still selectable and still the only validated choice;
    #: > `EXPORT_SPEC.md` §40 carries what actually runs.
    #:
    #: **Added 2026-08-10 because the column existed and the measure did not.** `cells.csv`
    #: has carried `soma_volume_um3` since Component 7 was written and it was blank in every
    #: row: `CellMorphology` declared the field and nothing ever assigned it. That is exactly
    #: the class of defect the export rebuild exists to close, found by the rebuild.
    #:
    #: **The definition is the opening, and the user chose it** — one of five options put to
    #: them, recorded 2026-08-12 in `EXPORT_SPEC.md` §18 and `PROJECT_LOG.md` §25. Reporting
    #: the eroded remnant directly would understate the soma by one erosion and by a
    #: *shape-dependent* amount; restoring it makes the measure "the part of the cell that can
    #: contain a ball of radius ``thickness_um``", which is a standard construction and is
    #: precisely the region this function already uses to *locate* the soma. It is not a new
    #: parameter — ``soma_thickness_um`` already existed and already decided where the soma is.
    #:
    #: **What that inherits is the accepted cost**, not an oversight: ``soma_thickness_um`` was
    #: tuned to *locate* somata, so **if it ever changes, every soma volume changes with it**
    #: and volumes from two runs at different values are not comparable. A separate radius for
    #: measurement was considered and rejected — see §18.4. Do not re-open it here.
    volume_um3: float = float("nan")

    #: **The soma itself, as a boolean array in the cell mask's own frame** — the very array
    #: whose voxel count becomes ``volume_um3`` above, kept rather than recomputed.
    #:
    #: Added 2026-08-19 for Component 5's rich review, which paints the soma white and must
    #: paint *exactly* what the export measures. Recomputing the opening at the screen would
    #: be a second construction of one region, free to drift from this one with nothing to
    #: say so — the `EXPORT_SPEC.md` §5.1 failure in the form this project keeps meeting.
    #:
    #: `None` when no remnant survived: `centroid_fallback` is *"a thin fragment with no
    #: soma"*, and an empty array would say the soma is empty rather than absent.
    #:
    #: **Excluded from equality and from `repr`.** An array field would otherwise make
    #: `SomaResult == SomaResult` raise on the ambiguous truth value of an array.
    region: np.ndarray | None = field(default=None, repr=False, compare=False)

    def summary(self) -> str:
        if self.centroid is None:
            return "no soma found"
        score = ("n/a" if self.agreement_score is None
                 else f"{self.agreement_score:.2f}")
        return f"soma via {self.method}, agreement {score}"


def _enclosed_cavities(cell_mask: np.ndarray) -> np.ndarray:
    """Voxels the mask encloses but does not contain. **Two passes, and one is not enough.**

    > ### THE FIRST VERSION USED ONLY THE 3D FILL, AND MISSED MOST LUMENS — 2026-08-27
    >
    > `threshold.fill_soma_holes` has said why since it was written: *"A 3D fill closes cavities
    > fully enclosed by signal; thresholding almost always leaves gaps in the shell, so a
    > per-slice fill then closes rings that are enclosed within their own plane even where the 3D
    > shell leaks."* The lumen fill shipped with the 3D pass alone and inherited exactly that gap.
    >
    > **Measured on crop2 at the user's settings, after the user reported the soma still visibly
    > wrong:**
    >
    > | cell | 3D pass alone | both passes | missed |
    > |---|---|---|---|
    > | 2 | 32.1 µm³ | 42.9 | 10.8 |
    > | 3 | 34.4 | 42.3 | 7.9 |
    > | **1** | **0.0** | **28.8** | **28.8** |
    > | **4** | **0.0** | **64.9** | **64.9** — holding 53.1 µm³ of its own nucleus |
    >
    > **Two cells read as having no lumen at all** and were wrongly written up as such — cell 4's
    > 34% coverage was attributed to an upstream mask ceiling when it was this. A leaking shell is
    > the *common* case, not the exceptional one.

    **The nucleus gate still does all the guarding.** A per-slice fill is more permissive than a
    3D one — a ring closed within its own plane now counts — so it could in principle reach a ring
    formed by wrapping processes. It does not, because `fill_nucleus_lumen` keeps only cavities
    adjacent to this cell's own nucleus, and a process ring encloses none.
    """
    filled = ndimage.binary_fill_holes(cell_mask)
    for index in range(cell_mask.shape[0]):
        plane = cell_mask[index]
        if plane.any():
            filled[index] |= ndimage.binary_fill_holes(plane)
    return filled & ~cell_mask


def fill_nucleus_lumen(
    cell_mask: np.ndarray, nucleus_mask: np.ndarray | None
) -> np.ndarray:
    """Close the nucleus-shaped cavity **so the erosion sees a solid soma, not a shell.**

    **Added 2026-08-27, after the soma was found sitting in the wrong place on real cells.**
    Iba1 stains cytoplasm, so a soma is a *shell* around an unstained nucleus. `fill_soma_holes`
    exists for exactly that and its docstring names *soma erosion* as a consumer needing the soma
    solid — but **patch B set `fill_holes=False` for the measurement mask**, and the cell labels
    `find_soma` receives are built from it. Its comment says *"nothing downstream of here needs
    its lumens closed"*; `find_soma` is downstream, and that sentence was wrong.

    Eroding a thin cytoplasmic rim by 0.325 µm in plane destroys most of it, so what survived was
    whichever spot happened to be thickest — off-centre, or out along a proximal process. The
    chosen remnant then *is* the soma, and its centre of mass is the Dijkstra root for every
    branch length, the hierarchy and both Sholl profiles.

    > ### MEASURED ON crop2 AT THE USER'S OWN SETTINGS — 2026-08-27
    >
    > Coverage of the cell's nucleus by its soma, before and after:
    >
    > | cell | before | after |
    > |---|---|---|
    > | 2 | 74% | **100%** |
    > | 3 | 51% | **100%** |
    > | 1 | 86% | 86% — no lumen |
    > | 4 | 34% | 34% — no lumen; its mask holds only 29% of its nucleus |
    >
    > One cell's soma centroid sat **11.17 µm** from its nucleus before the fix.

    **Why adjacency and not containment.** The nucleus this is handed is **clipped to the cell
    mask** (`pipeline.payloads`), so it is the *rim around* the lumen and has **no voxel inside
    it** — a containment test finds nothing. Dilating the rim by one voxel makes the cavity it
    encloses reachable, which is what the clipped nucleus can actually attest to. *An unclipped
    nucleus would allow containment, and was measured as exactly equivalent — it was not adopted
    because it needs a wider array at every call site and buys nothing.*

    **It cannot touch a process, and that is structural rather than tuned.** Only voxels that are
    (a) inside a cavity **fully enclosed** by this cell's own mask and (b) in a cavity adjacent to
    this cell's own nucleus are ever added. A process hole is not adjacent to the nucleus; a ring
    closed by wrapping processes encloses no nucleus. **This is the whole reason the fill is
    nucleus-marked rather than a plain `binary_fill_holes`** — the user removed hole filling from
    the measurement mask precisely because a blanket fill could close process rings and could do
    so at different rates in 5xFAD and WT, putting a **genotype-differential bias** into cell
    volume. Nothing here is decided by hole topology; the marker is the DAPI channel.

    **The unfilled mask is still what `cell_volume_um3` measures.** This returns a separate array
    for the soma computation and does not change the cell.

    Returns ``cell_mask`` unchanged when there is no nucleus, no cavity, or no cavity the nucleus
    can vouch for — so it is safe to call unconditionally.
    """
    if nucleus_mask is None:
        return cell_mask
    nucleus = np.asarray(nucleus_mask).astype(bool)
    if not nucleus.any() or nucleus.shape != cell_mask.shape:
        return cell_mask

    cavities = _enclosed_cavities(cell_mask)
    if not cavities.any():
        return cell_mask

    element = np.ones((3,) * cell_mask.ndim, dtype=bool)
    labels, _ = ndimage.label(cavities, structure=element)
    reach = ndimage.binary_dilation(nucleus, structure=element)
    keep = [int(v) for v in np.unique(labels[reach & cavities]) if v]
    if not keep:
        return cell_mask
    return cell_mask | np.isin(labels, keep)


def _distance_to_background_um(
    mask: np.ndarray, voxel_size_um: Sequence[float]
) -> np.ndarray:
    """Each voxel's distance to the nearest non-mask voxel, **in microns on every axis.**

    **The array is padded with one layer of background and cropped back**, so material touching
    a crop face is eroded from it. That is what `ndimage.binary_erosion` did with its default
    ``border_value=0``, and it is the right convention here: a cell mask is cropped to its own
    bounding box, so the cell really does end at the face. Without the pad, a cell would be
    treated as continuing past its own crop and the soma would bulge at the edges.

    One layer is enough. The nearest point of the half-space beyond a face is the padded voxel
    directly perpendicular to it, so a one-voxel shell gives the same answer an infinite one
    would.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    distance = ndimage.distance_transform_edt(padded, sampling=voxel_size_um)
    return distance[(slice(1, -1),) * mask.ndim]


def _erode_physically(
    mask: np.ndarray, radius_um: float, voxel_size_um: Sequence[float]
) -> np.ndarray:
    """Keep material at least ``radius_um`` from the outside — **2x that across, on every axis.**

    > ### ⇒ THIS REPLACED A VOXELISED STRUCTURING ELEMENT — 2026-08-29, and it is a bias fix
    >
    > Until now the erosion used an ellipsoid built out of **whole voxels**: each radius was
    > ``round(radius_um / spacing)``, clamped to at least one. Rounding a physical radius onto a
    > coarse axis does not approximate it, it **biases** it, and the bias was large.
    >
    > At the production geometry — 0.6 µm z, 0.108 µm in plane — asking for 0.9 µm gave a z
    > radius of 2 voxels and an xy radius of 8. The minimum thickness that survived was then:
    >
    > | axis | voxelised element | **this** |
    > |---|---|---|
    > | in plane | 17 voxels = 1.84 µm | 17 voxels = **1.84 µm** |
    > | **z** | **5 voxels = 3.00 µm** | 3 voxels = **1.80 µm** |
    >
    > **The two axes now agree at ~2 x radius, which is what the parameter always claimed to
    > mean.** Before, a lobe had to be 3.0 µm deep to survive but only 1.84 µm wide — so a broad,
    > shallow part of a cell body was deleted for its depth while the identical shape lying in
    > plane was kept. **The user reported exactly that**: a soma lobe lateral to the nucleus and
    > thin in z, drawn as process.
    >
    > **It also made the parameter move in steps.** 0.7 and 0.8 round to the same z radius, then
    > 0.9 steps to two — which is where the 58% → 25% cliff in the 2026-08-27 threshold sweep
    > sits. That cliff was read as a structural boundary in the tissue and used as one of three
    > arguments for 0.9. **It was the rounding.** The sweep has to be re-run on this ruler; see
    > `scripts/sweep_soma_thickness.py`.
    >
    > **`soma_volume_um3` moves again**, with everything rooted at the centroid. `EXPORT_SPEC.md`
    > §28's validation predates this as it already predated two other changes.

    **The convention is ``>=`` against the distance to the nearest background *voxel centre*.**
    That is the standard EDT opening and it is what makes the axes agree above; it sits half a
    voxel outside the true surface, equally on both sides, so it does not reintroduce a
    direction-dependent bias.
    """
    return _distance_to_background_um(mask, voxel_size_um) >= radius_um


def _dilate_physically(
    mask: np.ndarray, radius_um: float, voxel_size_um: Sequence[float]
) -> np.ndarray:
    """Everything within ``radius_um`` of ``mask``, in microns on every axis.

    The other half of `_erode_physically`'s fix: a restore built from a voxelised element carries
    the same rounding bias as the erosion that preceded it. **No pad here** — this measures
    distance *to* a set inside the array, and voxels beyond the array are not being asked about.

    Where an axis cannot represent the radius, this rounds **down**: at 0.6 µm z spacing a 0.9 µm
    dilation reaches one voxel, because two would be 1.2 µm and that is further than was asked.
    The old element rounded up to 1.2. Rounding down is the honest direction — it never claims
    reach the geometry cannot support.
    """
    solid = np.asarray(mask).astype(bool)
    if not solid.any():
        return np.zeros(solid.shape, dtype=bool)
    return ndimage.distance_transform_edt(
        ~solid, sampling=voxel_size_um) <= radius_um


def min_plane_area(
    solid_mask: np.ndarray, voxel_size_um: Sequence[float]
) -> np.ndarray:
    """**The smallest cross-sectional area this voxel's material has**, over the three planes, µm².

    Mechanism 2. The user's criterion, 2026-08-29, and it is the first thing tried on this data
    that separates a cell body from a process cleanly.

    **Two properties are needed and every earlier attempt had only one.**

    | | destroyed by one hole | only diluted by holes |
    |---|---|---|
    | cannot tell a line from a blob | local thickness — AUC **0.618** | local occupancy — **0.821** |
    | can tell a line from a blob | inscribed disc — **0.728** | **this — 0.908** |

    *Area is an aggregate*, so the tunnels that riddle these masks reduce it proportionally
    instead of destroying it — which is why the inscribed disc measured the cell body at 0.74 µm
    and concluded nothing was there, while this measures the same voxels at **21.6 µm²**, the area
    of a 5 µm disc.

    **The MINIMUM over the three planes, not the maximum**, and that is what makes it refuse a
    process. A process 10 µm long and 1 µm thick reads ~10 µm² seen from above and from the side,
    and ~0.8 µm² cut across:

    | | xy | xz | yz | min |
    |---|---|---|---|---|
    | process along x | long stripe | long stripe | **small dot** | small |
    | soma, 5 µm across | ~20 µm² | ~20 | ~20 | **large** |

    A tube is long in one direction and thin in two, so **at least one of the three axis planes
    always cuts it close to transversally** — and cutting across a tube gives its cross-section,
    which is small by definition. The worst case is a process running diagonally, where no plane
    cuts it square; even then the area grows only as 1/cos, so a 1 µm process along (1,1,1) reads
    ~1.4 µm² rather than 0.8. **The limit is a process genuinely thicker than the threshold's own
    disc** — at 16 µm² that is 4.5 µm, well clear of this tissue's 0.5-1 µm processes.

    **Anisotropy is the whole measurement.** Each plane carries its own pixel area — xy is
    0.0117 µm² here while xz and yz are 0.0650, **5.5 times larger**. Counting pixels instead of
    microns would make the two coarse planes look 5.5x bigger and the minimum would come from the
    wrong plane on every voxel.

    Slices are **filled** before labelling, which closes the z-running tunnels that appear in an
    xy slice as enclosed holes. That is measurement only: nothing filled is ever added to the soma.
    """
    voxel = [float(v) for v in voxel_size_um]
    best = np.full(np.asarray(solid_mask).shape, np.inf, np.float32)
    for axis in range(len(voxel)):
        pixel_area = float(np.prod([v for a, v in enumerate(voxel) if a != axis]))
        moved = np.moveaxis(np.asarray(solid_mask).astype(bool), axis, 0)
        out = np.zeros(moved.shape, np.float32)
        for index in range(moved.shape[0]):
            plane = moved[index]
            if not plane.any():
                continue
            labels, n = ndimage.label(ndimage.binary_fill_holes(plane))
            if not n:
                continue
            counts = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float32)
            counts[0] = 0.0
            out[index] = counts[labels] * pixel_area
        best = np.minimum(best, np.moveaxis(out, 0, axis))
    return np.where(np.isfinite(best), best, 0.0).astype(np.float32)


def _fatten_to_local_thickness(
    solid: np.ndarray, path: np.ndarray, voxel_size_um: Sequence[float]
) -> np.ndarray:
    """Widen a one-voxel path to **whatever the cell is locally** — the user's instruction.

    A shortest path is a thread, and a soma joined to a sheet by a thread would render as two
    blobs on a wire and would measure as one too. Each voxel is taken if its distance to the path
    is within the cell's own local half-thickness **at the nearest path voxel**, so the corridor
    is exactly as thick as the cell is where it passes — no wider, and no new parameter.

    > ### IT IS USUALLY A SMALL ADDITION, AND THAT IS MEASURED — 2026-08-29
    >
    > Removing the *call* to this survives mutation testing. Not an assertion gap: the
    > anchor and the sheet are both broad regions, so almost every voxel of a short
    > connector already lies on *some* shortest path between them. Measured on a 3.2 µm
    > wide link, the bare corridor covers **93.3%** of its cross-section against this
    > function's **100%**.
    >
    > Kept because it is correct, because it is what the user asked for, and because it is
    > load-bearing wherever the path squeezes through a constriction and the cell widens
    > after it — which a fixture can show and this tissue's short links do not. The helper
    > has its own test; the call site's contribution is small by construction.
    """
    half = _distance_to_background_um(solid, voxel_size_um)
    distance, indices = ndimage.distance_transform_edt(
        ~path, sampling=[float(v) for v in voxel_size_um], return_indices=True)
    return np.asarray(solid).astype(bool) & (distance <= half[tuple(indices)])


def _shortest_path_corridor(
    passable: np.ndarray, start: np.ndarray, goal: np.ndarray,
    voxel_size_um: Sequence[float],
) -> np.ndarray:
    """Every voxel lying on a **shortest path** through ``passable`` from ``start`` to ``goal``.

    The user's *"directly connected line from the nucleus up to that point"*. Computed as the set
    where the geodesic distance from the start plus the geodesic distance from the goal equals the
    shortest total — which yields all shortest paths at once and needs no traceback.

    **Geodesic, in microns.** A Euclidean line would cut across background and claim material the
    cell does not have; `MCP_Geometric` with ``sampling`` walks inside the mask and pays the right
    price per axis.
    """
    from skimage import graph                                   # local: heavy, rarely needed

    if not start.any() or not goal.any():
        return np.zeros(passable.shape, dtype=bool)
    costs = np.where(passable, 1.0, np.inf)
    spacing = tuple(float(v) for v in voxel_size_um)

    def field(seed: np.ndarray) -> np.ndarray:
        mcp = graph.MCP_Geometric(costs, sampling=spacing)
        got, _ = mcp.find_costs([tuple(int(c) for c in idx) for idx in np.argwhere(seed)])
        return np.asarray(got)

    total = field(start) + field(goal)
    finite = total[np.isfinite(total)]
    if finite.size == 0:
        return np.zeros(passable.shape, dtype=bool)
    #: A half-voxel of slack, so that ties broken differently by the two sweeps still join up.
    return passable & (total <= finite.min() + 0.5 * min(spacing))


def _close_soma_waists(
    soma: np.ndarray, solid: np.ndarray, radius_um: float, voxel_size_um: Sequence[float]
) -> np.ndarray:
    """Material with **soma on both sides** across a short span becomes soma.

    A morphological closing at ``radius_um``, in physical microns, joining two parts of the soma
    separated by up to twice it — and **clipped to the cell**, so nothing outside the mask is ever
    added and `cell_volume_um3` cannot move.

    Padded before the erosion half, because `_erode_physically` treats the array edge as
    background: without the pad a soma touching its own crop face would be eaten back from it.
    """
    if radius_um <= 0:
        return soma
    solid = np.asarray(solid).astype(bool)
    if not soma.any():
        return soma
    pad = [int(np.ceil(float(radius_um) / float(v))) + 2 for v in voxel_size_um]
    wide = np.pad(soma, [(p, p) for p in pad])
    shut = _erode_physically(
        _dilate_physically(wide, radius_um, voxel_size_um), radius_um, voxel_size_um)
    inner = shut[tuple(slice(p, -p) for p in pad)]
    return (soma | inner) & solid


def _sheet_candidate(
    solid: np.ndarray, anchor: np.ndarray, voxel_size_um: Sequence[float],
    sheet_area_um2: float, min_sheet_um3: float,
) -> np.ndarray:
    """Sheet-like material, and the corridor joining it to the nucleus. **Mechanism 2, in full.**

    The user's design, 2026-08-29, in three rules:

    1. **sheet-like material is soma wherever it sits** — no radius from the nucleus, because a
       proximity cap limits damage rather than defining a cell body;
    2. **a sheet must be body-sized** (``min_sheet_um3``), or it is a per-slice fill closing a
       ring of crossing processes — the one way `min_plane_area` can be fooled, and the user's own
       standing objection to filling anything;
    3. **there must be a direct path back to the nucleus**, and that path is soma too, fattened to
       the cell's local thickness. A sheet with no path is **dropped** — the user's explicit rule,
       and it also removes any chance of a spurious distant sheet dragging a corridor down a
       process.

    Branches leaving the result stay processes, exactly as primary processes leaving the body
    already did.
    """
    area = min_plane_area(solid, voxel_size_um)
    sheets = np.asarray(solid).astype(bool) & (area >= float(sheet_area_um2))
    if not sheets.any():
        return sheets

    element = np.ones((3,) * np.asarray(solid).ndim, dtype=bool)
    voxel_volume = float(np.prod([float(v) for v in voxel_size_um]))
    labels, n = ndimage.label(sheets, structure=element)
    joined, _ = ndimage.label(sheets | anchor, structure=element)
    reachable = {int(v) for v in np.unique(joined[anchor]) if v}

    out = np.zeros(sheets.shape, dtype=bool)
    for index in range(1, n + 1):
        piece = labels == index
        #: **A piece the anchor can already walk to is part of the cell body and is never
        #: dropped for its size.** The floor exists to make a *detached* piece earn its place;
        #: applying it to the body as well would let a small cell lose its own soma to it.
        if int(joined[piece].flat[0]) in reachable:
            out |= piece
            continue
        if float(piece.sum()) * voxel_volume < float(min_sheet_um3):
            continue
        corridor = _shortest_path_corridor(solid, anchor, piece, voxel_size_um)
        if not corridor.any():
            continue                    # no path to the nucleus, so by rule 3 it is not soma
        out |= piece | _fatten_to_local_thickness(solid, corridor, voxel_size_um)
    return out


def soma_anchor(
    solid_mask: np.ndarray,
    nucleus_mask: np.ndarray | None,
    anchor_um: float,
    voxel_size_um: Sequence[float],
) -> np.ndarray | None:
    """The nucleus and the cytoplasm wrapping it. **Soma unconditionally — mechanism 1 of two.**

    **The user's design, 2026-08-29.** A thickness test alone can pick the wrong region: if a cell
    has one thick proximal process, the cell's thickest material is **in the process**, so a rule
    keyed only to thickness selects the process and drops the real cell body.

    > ### THE ANCHOR IS NARROWER THAN IT WAS FIRST ARGUED — corrected 2026-08-29
    >
    > The original claim was that a thickness floor can never find a *compact* soma, its rim being
    > thinner than any useful threshold. **With the lumen filled that is usually false.** Filling
    > makes the nucleus region solid, so a 4 µm nucleus presents ~2 µm of local thickness and
    > clears a 0.9 µm floor easily — `fill_nucleus_lumen` had already done half the job this was
    > proposed for. A test fixture built on the original claim failed its own precondition, which
    > is how it was caught.
    >
    > **Two jobs remain, and they are real.** It covers cells whose nucleus is too small to clear
    > the floor even filled — which this study's own DAPI background filter produces, having
    > shrunk nuclei 2–4.4× at strength 4 — and it guarantees the cytoplasm *around* the nucleus
    > rather than only the part thick enough to qualify on its own.

    So the anchor is guaranteed and cannot be vetoed. `find_soma` unions it into the soma after
    the thickness test, never before, and it is what makes an aggressive ``thickness_um`` safe:
    the threshold may be set hard enough to reject any process, because the cell body is already
    covered.

    > ### ⇒ SIZED BY ITS OWN ``anchor_um`` SINCE 2026-08-31 — it shared ``thickness_um`` before
    >
    > The original argument was that *"the rim is on the order of the soma/process boundary
    > itself, so the same distance serves; a second radius here would be the invented parameter
    > `EXPORT_SPEC.md` §18.4 refused."* **§18.4's refusal was about a second *measurement*
    > radius** — a scale at which to report the volume. This is the reach of a mechanism whose
    > job is stated, and sharing the number meant **the floor could not be tuned at all**: every
    > move changed the criterion and the guarantee together, so no sweep could isolate either.
    >
    > See `SOMA_ANCHOR_UM_DEFAULT` for the measurement that settled it.

    It reaches ``anchor_um`` into the base of each process too — bounded, symmetric, and
    arguably right, since a process's base is part of the cell body.

    ``solid_mask`` must already have its lumen closed (`fill_nucleus_lumen`), so a hollow shell
    still yields a solid anchor. Returns `None` when there is no nucleus to anchor to, which is
    the `largest_remnant` path.
    """
    if nucleus_mask is None:
        return None
    nucleus = np.asarray(nucleus_mask).astype(bool)
    if not nucleus.any() or nucleus.shape != solid_mask.shape:
        return None
    core = nucleus & solid_mask
    if not core.any():
        return None
    anchor = _dilate_physically(core, anchor_um, voxel_size_um) & solid_mask
    return anchor if anchor.any() else None


def find_soma(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray | None = None,
    *,
    voxel_size_um: Sequence[float],
    thickness_um: float = SOMA_THICKNESS_UM_DEFAULT,
    anchor_um: float = SOMA_ANCHOR_UM_DEFAULT,
    method: str = SOMA_METHOD_DEFAULT,
    sheet_area_um2: float = SOMA_SHEET_AREA_UM2_DEFAULT,
    waist_um: float = SOMA_WAIST_UM_DEFAULT,
    min_sheet_um3: float = SOMA_MIN_SHEET_UM3_DEFAULT,
) -> SomaResult:
    """Find the soma. **Mechanism 1 is the anchor; `method` chooses mechanism 2.**

    ``cell_mask`` and ``nucleus_mask`` must share a coordinate frame — normally both
    cropped to the same bounding box. The returned centroid is in that frame; callers
    add the crop offset to place it in the full image.

    **The anchor is identical under both methods** and so is everything before it: the lumen fill,
    the guarantee, the fallbacks. Only the criterion that decides what *else* is soma differs.

    | | `opening` | `anchor_sheet` |
    |---|---|---|
    | anchor radius | ``anchor_um`` | ``anchor_um`` — **identical, and read under both** |
    | criterion | local thickness ≥ 2 x ``thickness_um`` | ``min_plane_area`` ≥ ``sheet_area_um2`` in **every** plane, plus a corridor back to the anchor |
    | size floor | none | ``min_sheet_um3``, on **detached** pieces only |
    | restore step | dilate the eroded core back | none — the candidate is already at full extent |
    | waists closed | no | yes, at ``waist_um`` |
    | validated by | `EXPORT_SPEC.md` §18, §28, §34 | the screen, and AUC 0.908 against 0.618 |

    **Both methods read ``anchor_um``; beyond that they share nothing.** ``opening`` reads
    ``thickness_um`` and `anchor_sheet` reads the other three — so ``thickness_um`` is genuinely
    unread under `anchor_sheet`, which is why item 6.5 hides it there. `ui.screen_soma`'s
    `METHOD_FIELDS` is this paragraph in code and a test fails if the two disagree.

    **Without an anchor there is no sheet growth**, so `anchor_sheet` falls back to the thickness
    remnant and 3DMorph's largest-remnant rule, exactly as `opening` does.
    """
    if method not in SOMA_METHODS:
        raise ValueError(f"unknown soma method {method!r}; expected one of {SOMA_METHODS}")
    if not cell_mask.any():
        return SomaResult(None, False, None, "empty")

    # **Erode the cell with its nucleus lumen closed, never the bare shell** — see
    # `fill_nucleus_lumen`. `solid` replaces `cell_mask` everywhere below, so the chosen
    # remnant, its centroid and the restored region are all derived from the same array.
    # `cell_volume_um3` is measured elsewhere and still sees the unfilled mask.
    solid = fill_nucleus_lumen(cell_mask, nucleus_mask)

    # **MECHANISM 1** — the anchor, guaranteed. See `soma_anchor`. It is unioned in *after*
    # the thickness test, never before, so no threshold can take the cell body away.
    anchor = soma_anchor(solid, nucleus_mask, anchor_um, voxel_size_um)

    has_nucleus = nucleus_mask is not None and nucleus_mask.any()

    # **MECHANISM 2's candidate set.** The bound needs a nucleus to measure from, so a cell
    # without one takes the thickness route whatever `method` says — the same cells that already
    # fall through to `largest_remnant` below.
    by_sheet = method == "anchor_sheet" and anchor is not None
    if by_sheet:
        remnant = _sheet_candidate(solid, anchor, voxel_size_um,
                                   sheet_area_um2, min_sheet_um3)
    else:
        remnant = _erode_physically(solid, thickness_um, voxel_size_um)

    if not remnant.any() and anchor is None:
        # Erosion consumed the whole cell and there is no nucleus to anchor to: a thin
        # fragment with no soma. Fall back to the centroid of the cell itself.
        centroid = ndimage.center_of_mass(solid)
        return SomaResult(tuple(float(c) for c in centroid), False,
                          None, "centroid_fallback")

    if anchor is not None:
        # **MECHANISM 2, seeded by mechanism 1.** Keep thick material *connected to the
        # anchor*, rather than picking whichever eroded blob has the most nucleus in it.
        #
        # **This replaces selection with growth, and that is the substantive change.** The
        # old rule could only choose among components the erosion happened to leave; it had
        # no way to *split* one, so a thick proximal process fused to the body was taken
        # whole. Growing from the anchor through material above the thickness floor stops at
        # the neck instead, because a process falls below the floor and a soma does not.
        seeded = remnant | anchor
        labels, _ = ndimage.label(seeded, structure=np.ones((3, 3, 3)))
        keep = [int(v) for v in np.unique(labels[anchor]) if v]
        core = np.isin(labels, keep) & remnant
        #: **The export says which criterion produced the soma**, so a pool holding rows from a
        #: trial and rows from the validated mechanism can be told apart by reading a column
        #: rather than by remembering what was set when.
        chosen = "nucleus_sheet" if by_sheet else "nucleus_anchored"
    else:
        # No nucleus: 3DMorph's rule, unchanged. These cells exist by design.
        labels, n = ndimage.label(remnant, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        sizes[0] = 0
        core = labels == int(np.argmax(sizes))
        chosen = "largest_remnant"

    # Soma volume: the opening, i.e. the chosen remnant restored to its pre-erosion extent
    # and clipped to the cell. Physical microns, so it means the same thing at 5.5:1 as it
    # would isotropically.
    # **Clipped to `solid`, so the soma INCLUDES the nucleus it envelops** — the user's
    # decision, 2026-08-27. That is what Imaris manual surface tracing and the nucleator both
    # measure: one traces the soma's outline, the other casts rays to its boundary, and neither
    # subtracts the nucleus. With no ground truth of our own, being comparable to the two
    # reference methods is worth more than staying close to the previous number.
    #
    # **It raises `soma_volume_um3` by the lumen** — ~32 µm³ on the affected crop2 cells. The
    # alternative, clipping back to the unfilled `cell_mask`, keeps the old magnitude while still
    # fixing the position, and adds no parameter; it is what to change if comparability stops
    # being the priority.
    # **Only the eroded core is restored.** The anchor is already at full extent — it was
    # built by dilating the nucleus — so dilating it again would grow the soma by twice the
    # radius and undo the point of the threshold.
    #: **`anchor_sheet` has nothing to restore.** The opening dilates its core back because the
    #: core was *eroded* to find it; the sheet candidate was never shrunk, so dilating it would
    #: grow the soma by a radius nobody asked for.
    if by_sheet:
        restored = core
    else:
        restored = (_dilate_physically(core, thickness_um, voxel_size_um) & solid
                    if core.any() else np.zeros_like(solid))
    restored = restored | anchor if anchor is not None else restored
    #: **Only `anchor_sheet` closes its waists.** `opening` is the validated mechanism and
    #: must reproduce its own numbers exactly, so nothing may be added to what it returns.
    if by_sheet:
        restored = _close_soma_waists(restored, solid, waist_um, voxel_size_um)

    # **The centroid is the SOMA's, not the eroded core's** — changed 2026-08-29 with the
    # anchor. It is the Dijkstra root for every branch length, the hierarchy and both Sholl
    # profiles, so it should be the centre of the region a reader would call the cell body.
    centroid_t = tuple(float(c) for c in ndimage.center_of_mass(restored))

    voxel_volume = float(np.prod([float(v) for v in voxel_size_um]))
    volume_um3 = float(restored.sum()) * voxel_volume

    score = None
    if has_nucleus:
        nucleus_centre = np.array(ndimage.center_of_mass(nucleus_mask), dtype=float)
        scale = np.array([float(v) for v in voxel_size_um])
        distance = float(np.linalg.norm((np.array(centroid_t) - nucleus_centre) * scale))

        # Normalised by the *nucleus* radius, not the eroded remnant's. The remnant has
        # already been shrunk by erosion, so its radius understates the true soma and
        # pushes the ratio above 1 for ordinary cells.
        nucleus_volume = float(nucleus_mask.astype(bool).sum()) * float(np.prod(scale))
        reference_um = max((3.0 * nucleus_volume / (4.0 * np.pi)) ** (1.0 / 3.0), 1e-6)

        # A saturating form (1 - d/r clamped at zero) collapses a third of real cells to
        # exactly 0.00, which destroys the ordering this score exists to provide. This
        # decays smoothly instead: 1.0 when co-located, 0.5 at one nucleus radius, and
        # always discriminating however far apart the two channels put the soma.
        score = float(1.0 / (1.0 + distance / reference_um))

    return SomaResult(centroid_t, True, score, chosen, volume_um3=volume_um3,
                      region=restored)
