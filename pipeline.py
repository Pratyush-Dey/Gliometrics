"""Phase A: one image in, candidate cells in the pool.

Wires the segmentation modules into a single unattended call. No GUI, no prompts, no
blocking — that is what lets this run as a SLURM array task on a GPU node, and it is
precisely what PrestoCell cannot do, because its pipeline functions call `napari.run()`
and wait for a human (`main.py:1351`).

The order matters in two places:

**Filtering happens before the return to native resolution.** On the test crop, 429
objects become 448 cells, of which 439 are debris under 10 µm³. Upsampling all 448 and
then discarding 427 of them would be wasted work; filtering first leaves ~21 to promote.

**Native resolution is restored by region assignment, not by upsampling labels.** The
coarse labels only decide *which* cell a voxel belongs to; the shape comes from the
native threshold mask. Upsampling the label volume alone would quantise every boundary
to the downsample factor and blunt exactly the thin processes the study measures.

Every parameter is an argument. Where those values come from — a slider, a saved file,
a test — is deliberately not this module's concern; that separation is what allows one
implementation to serve both the interactive and the batch mode.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from . import cpu
from .filters import filter_cells
from .io import ChannelRoles, ImageSource
from .markers import DebrisLoad, measure_debris
from .membership import filter_nuclei_by_membership
from .nuclei import Detector, detect_nuclei, diameter_in_pixels
from .objects import find_cell_objects
from .resolution import (
    MeasurementMask,
    assign_labels_by_connectivity,
    downsample_mask_xy,
    downsample_xy,
    upsample_xy,
    voxel_size_after_downsample,
)
from .skeleton import LOOP_PRUNE_DEFAULT
from .soma import (SOMA_ANCHOR_UM_DEFAULT, SOMA_METHOD_DEFAULT, SOMA_MIN_SHEET_UM3_DEFAULT,
                   SOMA_SHEET_AREA_UM2_DEFAULT, SOMA_THICKNESS_UM_DEFAULT,
                   SOMA_WAIST_UM_DEFAULT, find_soma, soma_for)
from .split import split_objects
from .store import PoolWriter
from .threshold import build_cell_mask, threshold_channel

__all__ = ["SegmentationParams", "PipelineResult", "Derivation", "segment_image",
           "apply_cell_size_cutoff", "apply_cell_edit", "PHASE_A_FIELDS", "PHASE_C_ONLY_FIELDS"]


@dataclass(frozen=True)
class SegmentationParams:
    """Every knob Phase A exposes, and the Phase C ones that must travel with it.

    Not purely Phase A, and saying so here rather than leaving the docstring one field behind:
    `plaque_threshold`, `cd68_threshold`, `min_plaque_volume_um3`, `soma_thickness_um` and
    `min_spur_um` are read during **measurement**. They live here because `hash()` is what
    makes a result traceable to its settings, and a measurement knob outside the hash is a
    knob two runs can differ on while claiming to be the same run — which is exactly what
    `min_spur_um` was until 2026-08-12 (NEW-g).

    Defaults are **measured against 13 hand-marked somata in a single image** and should
    be treated as provisional until the wider marked set exists. See `PROJECT_LOG.md`.

    **`nuclei_coefficient` was removed on 2026-08-15 and is not coming back by accident.**
    It was declared here and read nowhere — an Otsu multiplier for a DAPI-thresholding design
    that was never built, since DAPI goes to Cellpose and `detect_nuclei` takes no threshold
    of any kind. Inside `params_hash` while affecting nothing, it made two identical runs hash
    apart and re-queue Phase C for no reason: the exact mirror of the `min_spur_um` defect
    (that one was read but was not a field; this was a field but was never read). It is
    recorded here rather than deleted silently because its *name* invites a nucleus-threshold
    control to be wired to it (`COMPONENT_03_PLAN.md` §2c), so a reader who goes looking for
    it should find out that its absence is a decision.
    """

    #: Which channel index carries which stain. **Real parameters since 2026-08-15**, on
    #: Workstream B's request: the roles reached Phase C through `FileRecord.channel_roles`
    #: but were on no dataclass, so they were in neither the parameters file nor
    #: `params_hash` — **two runs that segmented completely different channels hashed
    #: identically.** That is the `min_spur_um` shape with a worse consequence: measuring
    #: microglia off the CD68 channel is wrong in every column and plausible in every one.
    #:
    #: **Never positional.** PrestoCell hardcodes ``data[:, 0]`` and ``data[:, 1]`` and
    #: applies its one configurable swap twice so that it cancels out; `io.ChannelRoles`
    #: exists because of that, and these four fields mirror it exactly.
    #:
    #: **`SegmentationParams` says which channels to segment with; `FileRecord.channel_roles`
    #: records which channels an image actually was segmented with.** Two homes, two jobs —
    #: written down because the same field having two homes and neither naming its meaning is
    #: precisely how `soma_centroid` came to be added twice (`EXPORT_SPEC.md` §21).
    nuclei_channel: int = 0

    #: Channel index carrying the Iba1 (microglia) stain — threshold 1 and the cell mask are
    #: built from it. Required, like `nuclei_channel`: a run without it has nothing to segment.
    cell_channel: int = 2

    #: **`None`, not 1 and 3 — and that is load-bearing rather than a missing default.**
    #: `None` means *this image has no such channel*: `ChannelRoles.index` raises on it, which
    #: is how the pipeline learns a file carries no CD68, and every measure behind it exports
    #: blank with a status (`image_cd68_status`, `burden_status`) rather than a plausible zero.
    #: A default of 1 would make an image genuinely without CD68 **assert that channel 1
    #: carries it**, and every CD68 column would then be measured off whatever that channel
    #: holds. Same rule as `cd68_threshold` and `plaque_threshold` above: unset is a state,
    #: not an omission.
    cd68_channel: int | None = None

    #: Channel index carrying the 6E10 (amyloid) stain, or `None` when the image has none —
    #: on exactly the same terms as `cd68_channel` above. Absent means the plaque veto does
    #: not run and every burden and compartment measure exports blank with `burden_status`.
    plaque_channel: int | None = None

    # thresholding — cell_coefficient is threshold 1: it decides how many cells there are
    # and where they divide. It is tuned for soma recall and reliable separation.
    cell_coefficient: float = 0.18
    #: The speck floor for the **cell** mask, in voxels — 3DMorph's ``noise`` parameter, applied
    #: in 3D after thresholding. Item 5's `measurement_min_object_voxels` is its counterpart on
    #: the measurement mask; the two are separate because the two masks are tuned separately.
    min_object_voxels: int = 20
    #: Bridges small gaps in the cell shell before the fill, µm. A shell with a hole in it fills
    #: as background however bright it is, so this is what stops a cell being hollow.
    closing_radius_um: float = 0.8

    #: 6E10 threshold for the **plaque veto only** — the mask that suppresses DAPI inside
    #: amyloid before Cellpose runs (§4.13.1). Split from the measurement coefficient on
    #: 2026-08-09 because the two uses want opposite settings and were sharing one number.
    #:
    #: **1.0 is evidence-backed here**, unusually for a default: the A/B measured recall
    #: *higher* with the veto on (97% vs 94%) and false positives lower, and it is inert on
    #: WT (0.0-1.7% of DAPI, against up to 25.4% on 5xFAD). The veto wants to be permissive —
    #: over-vetoing costs almost nothing, while a plaque read as a nucleus manufactures a
    #: spurious split whose fragments fall under the volume cutoff and take a real cell with
    #: them. That asymmetry is the opposite of the measurement one below.
    plaque_veto_coefficient: float = 1.0

    #: 6E10 **absolute intensity** for measuring amyloid burden, picked in the semi-auto UI
    #: with live preview. **No default, deliberately** — `None` means unchosen, and every
    #: measure behind it reports `NaN` with a status rather than a plausible zero.
    #:
    #: Replaces `plaque_measure_coefficient`, an Otsu multiplier, on 2026-08-10. That
    #: parameter had no default and could not acquire one: the sweep across 5 WT + 10 5xFAD
    #: images (`tuning_runs/plaque_sweep.txt`) has **no plateau** — 5xFAD/WT separation rises
    #: monotonically 16x -> 59x with no turning point to read a value off. An absolute
    #: threshold dissolves the problem rather than solving it. Otsu is computed per slice per
    #: image, so one multiplier lands on a different intensity in every file, which defeats
    #: choosing once and applying to a batch; acquisition here was **held constant across
    #: images**, which makes a single absolute value both defensible and standard practice.
    plaque_threshold: float | None = None

    #: CD68 **absolute intensity**, on the same terms and for the same reasons. **No default.**
    #: A background-referenced alternative was tried and reverted the same day — CD68 is
    #: punctate, so a background multiple separates "any signal" from "exactly zero"
    #: (`markers.measure_cd68`).
    cd68_threshold: float | None = None

    #: The one plaque size floor, µm³. **25.0 replaces both `MIN_PLAQUE_UM3 = 5.0` and the
    #: five-value `PLAQUE_SIZE_FLOORS_UM3` curve**, which could disagree about how many
    #: plaques an image contains. Unlike the two thresholds above this **does** have a value:
    #: it is calibrated against a negative control rather than picked by eye. See
    #: `markers.MIN_PLAQUE_VOLUME_UM3` for the calibration and the 73 µm³ ceiling.
    min_plaque_volume_um3: float = 25.0

    # threshold 2 — the measurement mask, deciding how far each cell *extends*.
    #
    # **No default, deliberately.** `None` means the user has not chosen one, and the
    # pipeline then behaves exactly as it did before two-mask support existed. A
    # threshold tuned to make somata come out cleanly drops the dimmest, thinnest
    # processes — and ramified cells carry more of those than activated ones, so a
    # conservative mask compresses the very difference the ramification index exists to
    # measure. That is a bias toward the null, so the value must be a decision, never a
    # default inherited from a parameter fitted for a different job.
    mask_coefficient: float | None = None
    #: The same gap-bridging for the **measurement** mask, µm — item 5 shows it as its own
    #: control, because the outline that decides how far a cell extends is a different judgement
    #: from the one that decides where cells divide.
    mask_closing_radius_um: float = 0.8

    #: The speck floor for the **measurement** mask, in voxels — item 5's own, and the
    #: measurement threshold's counterpart to `min_object_voxels` at item 3.
    #:
    #: **It had no box until the debris patch and shared item 3's value silently.**
    #: `ui/app.py`'s `measurement_mask` recorded that as deliberate — *"item 5 has no box
    #: for it and inventing one here would make the screen's mask differ from the one batch
    #: rebuilds"* — which was the right call while item 3's mask decided the cell set. It is
    #: not any more: since this patch the measurement mask is the one that gets split, so its
    #: floor decides what can become a cell at all, and a threshold that decides that with no
    #: control is the one thing item 5 cannot leave to item 3.
    #:
    #: **20 is item 3's value, chosen by the user as the default so the patch changes no
    #: number on its own.** Everything it removes is removed *before* debris exists and is
    #: therefore counted nowhere — which is the same objection that retired `speck_floor_um3`
    #: and is accepted here for a different reason: this floor is asked for, visible on its
    #: own screen, and recorded in the parameters file, so what it removes is a stated choice
    #: rather than a constant nobody was shown. `docs/SHAKEOUT_FINDINGS.md` carries the
    #: argument and the rejected alternative of counting sub-floor objects as debris.
    measurement_min_object_voxels: int = 20

    #: How much of the broad-scale DAPI background estimate is subtracted before detection.
    #: Useful values run about 0 to 4; the screen refuses only a negative one.
    #:
    #: **0.0 is off, and that is the default** — this removes real signal as well as background,
    #: so it is a decision rather than something inherited. 1.0 subtracts the whole estimate and
    #: **nothing in the arithmetic bounds it** — the screen takes this in a number box rather
    #: than a slider, so the useful range is advice. Above ~0.4 on the demo crop it is trimming
    #: nucleus edges rather than removing background; see `ui.screen_background` for the sweep.
    #:
    #: **Chosen on its own screen, between item 1 and item 2**, and the position is forced:
    #: `nuclei.subtract_background` must run **before the plaque veto**, which is the only step
    #: that writes zeros into DAPI, because a Gaussian estimate rings around a zero region
    #: (+189 against a background 95th percentile of 290, where a top-hat moved +0.11).
    #:
    #: **Strength rather than scale is the control, and that was measured.** Sweeping the scale
    #: gives a cliff and then a plateau — below one nucleus diameter it destroys the nuclei,
    #: above it the background is already entirely gone — so the scale is a box that is rarely
    #: touched and this is the slider. At scale 12 µm the nucleus-to-background contrast runs
    #: 5.8x at 0, 14.6x at 0.4, **58x at 0.6**, and past 0.8 only the nuclei are still being
    #: eaten. Full reasoning and the sweep at `nuclei.subtract_background`.
    #:
    #: **Not tuned.** No sweep has been run across the nineteen ground-truth crops, and the
    #: acceptance test when one is is **not the nucleus count** — it is that the four confirmed
    #: somata in `program training crop2.tif` keep their nuclei. A count that came out right for
    #: the wrong reason has already cost this project twice.
    dapi_background_strength: float = 0.0

    #: The scale of that background estimate, µm — the Gaussian's sigma, in plane only.
    #: **12.0 is right for 7.6 µm nuclei and is not a universal constant**: the rule is that it
    #: must comfortably exceed one nucleus diameter, and `nucleus_diameter_um` is chosen three
    #: screens later, so it cannot be expressed relative to it. The sweep says its exact value
    #: barely matters above that floor, which is why it is a number box rather than a slider.
    #:
    #: Inert while `dapi_background_strength` is 0. See `nuclei.HALF_RES_SIGMA_PX` for what this
    #: value costs: at this study's 0.1083 µm pixels, 12 µm is a 111 px kernel.
    dapi_background_scale_um: float = 12.0

    # nucleus detection
    nucleus_diameter_um: float = 7.6
    #: IoU for joining a nucleus across z slices. **Detection runs per slice and stitches**,
    #: rather than max-projecting blocks as PrestoCell does — that fuses anything stacked in
    #: depth and silently loses splits. This trades it for the opposite risk, which is why the
    #: value is one to validate rather than a default to accept.
    stitch_threshold: float = 0.25
    #: Drop detections smaller than this, in **downsampled** voxels. Deliberately NOT applied on
    #: the nuclei screen — see `screen_nuclei.py` for why the preview and the run differ here.
    nucleus_min_volume_voxels: int = 200
    #: How far the thresholded 6E10 mask is expanded before the plaque veto subtracts plaque
    #: DAPI, µm. **The margin exists because the two boundaries do not coincide**: the 6E10
    #: threshold edge and the DAPI-in-plaque edge sit in different places, so without it a rim
    #: of plaque DAPI survives the veto and can still seed a false nucleus — which manufactures
    #: a split whose fragments fall under `min_cell_volume_um3` and take a real cell with them.
    #: `pipeline.py` → `nuclei.detect_nuclei(plaque_dilate_um=…)`.
    #:
    #: **The failure mode is asymmetric, and that is why the value matters.** Too small and the
    #: veto leaks; too large and it eats **genuine nuclei near plaques** — exactly where
    #: plaque-associated microglia are, the population this study exists to characterise.
    #:
    #: **Documented and swept for the first time on 2026-08-12 (NEW-i)**, having appeared in no
    #: `docs/` file at all until then. `tuning_runs/plaque_dilate_sweep.txt` carries the sweep.
    #:
    #: **0.3 → 0.0, decided by the user 2026-08-13.** The sweep is what changed it, and its own
    #: conclusion is the reason: *"nothing here earns it. 0.00 scores identically on recall and
    #: vetoes 5.5% of DAPI against 8.4%. On this data the margin is **unpaid-for exposure**, and
    #: the exposure lands **entirely on 5xFAD**."* Recall is 163/172 at both 0.00 and 0.30, and
    #: what would have justified a margin — a case where the veto **leaks** at a small dilation
    #: and costs a cell — **did not appear in 172 confident somata across 15 images**.
    #:
    #: **The margin's rationale above is not retracted, only unpaid for.** The two boundaries do
    #: genuinely differ: the mask is 6E10 at Otsu, which captures the bright core and loses the
    #: dim periphery, so the deposit extends past it. What the sweep shows is that on this data
    #: the gap is small enough to cost nothing. **Raise it again if a leak is ever observed** —
    #: `COMPONENT_03_PLAN.md` §4 item 2 is the screen where it would be seen, with raw 6E10, the
    #: dilated mask and the vetoed DAPI on one view.
    #:
    #: **Read the veto's 97%/94% A/B (`tuning_runs/veto_ab.txt`) with care: it was measured at
    #: 0.3**, not at this value. The sweep's own note applies — that A/B ran on 5 images and 62
    #: somata against the sweep's 15 and 172, so the two are not directly comparable, and what
    #: the sweep establishes is that recall does not move between 0.00 and 0.30.
    plaque_dilate_um: float = 0.0
    #: Intensity conditioning before nucleus **detection** only — `'log'` compression, so a
    #: bright nucleus does not set a scale that hides a dim one. Detection sees this; every
    #: measurement reads the raw channel.
    normalize: str = "log"

    # membership — 0.25, not 0.50: Iba1 is cytoplasmic, so even a filled soma only
    # partly encloses its nucleus. At 0.50, 3 of 13 confirmed microglia were lost.
    #
    # 0.30 -> 0.25 on 2026-08-09, and the argument is **margin, not score**: recall is
    # unchanged at 60/62 and the sweep buys one under-split for one false positive. What
    # decides it is where the boundary sits in a bimodal distribution — confirmed microglial
    # nuclei score 0.35-0.98, non-microglial ones ~0.02, and nothing lives between. A cutoff
    # belongs in the middle of that empty gap; 0.30 sat at its top edge, and a confirmed
    # microglial nucleus scored **0.2996**, clearing by four ten-thousandths.
    #
    # There is a floor, so this is not "lower is better": at 0.15 recall drops to 57/62,
    # because admitting non-microglial nuclei manufactures spurious splits whose fragments
    # fall under `min_cell_volume_um3` and take real cells with them.
    membership_threshold: float = 0.25

    # splitting
    split_method: str = "watershed"
    #: Gaussian sigma applied to the intensity before the watershed, µm — converted per axis, so
    #: it is a physical distance rather than a voxel count on anisotropic data.
    smooth_um: float = 0.4
    #: How much of a nucleus must fall inside an object before it counts as belonging to it.
    #: **Ignores incidental grazing contact**, so a nucleus brushing a neighbour cannot make the
    #: object look clumped and force a split that is not there.
    min_overlap_fraction: float = 0.10

    # filtering. `min_cell_volume_um3` is the user's cell-size cutoff, chosen visually
    # against the volume distribution. Its 200.0 was measured against *threshold-1*
    # volumes (13/13 marked cells kept, smallest at 213 um3); measurement-mask volumes are
    # larger, so it should be re-chosen once `mask_coefficient` is in use. What it removes
    # becomes DEBRIS rather than nothing, and it only began removing anything on
    # 2026-08-25 -- until then item 6 wrote it to the parameters file and deleted no cell.
    min_cell_volume_um3: float = 200.0

    #: **Deprecated — always False.** Removing xy-border cells at segmentation time
    #: destroys them irreversibly to answer a question better answered later. Every cell
    #: already records `touches_xy_border`, so the flag travels with the data: use it as a
    #: display filter during review and an inclusion choice at analysis. 3DMorph forces
    #: this decision up front (`RemoveXY`); there is no reason to inherit that.
    remove_xy_border: bool = False

    # geometry
    downsample: int = 2

    #: **The local half-thickness material must clear to count as soma on its own**, µm — so
    #: material must be **2x this across**, on every axis. Mechanism 2's radius under ``opening``,
    #: and **unread under ``anchor_sheet``**, which has its own criterion. Offered on item 6.5
    #: since 2026-08-30. It is not a distance from the nucleus: that is `soma_anchor_um`.
    #:
    #: > ### ⇒ IT IS A THICKNESS FLOOR AGAIN — the 2026-08-31 split reversed the note below
    #: >
    #: > This comment read *"How far the GUARANTEED soma reaches out from the nucleus"* until the
    #: > radii were split, and **that was correct while it lasted**: the field sized the anchor
    #: > under both methods, so the anchor was the honest description of it. `soma_anchor_um` now
    #: > holds that job and this field went back to being what its name says.
    #: >
    #: > **The correction it replaced is kept because the trap is live and points both ways:**
    #:
    #: > ### IT IS NOT A THICKNESS FLOOR ANY MORE, AND THIS COMMENT SAID IT WAS — 2026-08-30
    #: >
    #: > It opened *"where the soma stops and a process begins, as a floor on local
    #: > half-thickness"*, which is what it does under ``opening`` — the mechanism that has not
    #: > been the default since 2026-08-30. Under ``anchor_sheet`` it sizes `soma.soma_anchor`
    #: > and nothing else; the soma/process boundary is `soma_sheet_area_um2`'s job. The old
    #: > sentence reached `PARAMETERS.md` and would have reached item 6.5's own "?" next.
    #: >
    #: > **The scale argument still holds**: a microglial process is 0.5-1 µm whatever the cell's
    #: > state and a soma is 4-8 µm, so both radii belong in that gap — which is why they could
    #: > share a number for as long as they did without anyone noticing they should not.
    #: >
    #: > **Twice now this field's one-line summary has described a mechanism it did not
    #: > implement, and both times it reached `PARAMETERS.md` before anyone read it.** The second
    #: > time was caught by regenerating the document and seeing two fields claim one job.
    #:
    #: **It also sets an exported measure.** ``soma_volume_um3`` is the opening of the cell mask
    #: at this radius (`EXPORT_SPEC.md` §18), so changing it changes that column and everything
    #: rooted at the soma centroid — branch lengths, hierarchy, both Sholl profiles. Volumes from
    #: two runs at different values are not comparable.
    #:
    #: **It cannot take the cell body away**, however aggressive it is set: `soma.soma_anchor`
    #: guarantees the nucleus and the cytoplasm within this distance of it, unconditionally, and
    #: `find_soma` unions that in after the floor has been applied.
    #:
    #: > ### ⇒ RE-DERIVED ON THE FIXED RULER, AND THE PLATEAU IS A CANCELLATION — 2026-08-31
    #: >
    #: > **0.9 was chosen against an erosion that rounded a physical radius onto whole voxels**,
    #: > demanding 3.00 µm of depth against 1.84 µm of width; the 58% → 25% cliff that justified
    #: > it was the rounding step, not the tissue. The ruler was fixed on 2026-08-29 and the
    #: > sweep re-run on 2026-08-31 — `tuning_runs/soma_thickness_rederive_opening.txt` and its
    #: > `_low` companion, on the user's own crop at their own settings.
    #: >
    #: > | this field | soma median | anchor median | **anchor as a share of the soma** |
    #: > |---|---|---|---|
    #: > | 0.40 | 46.6% | 17.0% | **36%** |
    #: > | 0.60 | 38.7% | 22.0% | 57% |
    #: > | 0.70 | 31.4% | 24.4% | 78% |
    #: > | **0.90** | **32.2%** | **26.5%** | **82%** |
    #: > | 1.20 | 30.7% | 29.0% | 94% |
    #: > | 1.50-2.00 | 31.3 → 34.7% | 31.2 → 34.7% | **100%** |
    #: >
    #: > **The 0.7-1.35 plateau is not a boundary. It is the anchor taking over** — the soma
    #: > median sits in a two-point band while the anchor climbs 78% → 99% underneath it, so two
    #: > runs in that range report almost the same volume and are not measuring the same object.
    #: > From 1.5 up the anchor **is** the soma, and the spread that grows there is nucleus-size
    #: > variation wearing soma clothing. **This is the second plateau in this project to mean
    #: > the measurement had gone deaf**, after the 2026-08-27 one that was the z rounding.
    #: >
    #: > **The floor only does real work below 0.7.** At 0.4 the soma is 46.6% with the anchor
    #: > at 36%; by 0.9 the floor contributes about a fifth and the anchor the rest.
    #:
    #: **0.9 stands, and since 2026-08-31 it is a chosen value rather than an inherited one.**
    #: The user walked the flow under ``opening`` and confirmed the soma it draws. That the
    #: anchor supplies ~82% of it is the mechanism working as asked, not a defect: ``opening``
    #: was reinstated precisely to mean *the nucleus and the cytoplasm around it* (§37).
    #:
    #: **Still not tuned against ground truth by any route** — `EXPORT_SPEC.md` §32's four
    #: validation steps remain open, and no sweep can close them: a sweep shows how the number
    #: responds, never which response is right.
    #:
    #: **It no longer sizes the anchor.** Until 2026-08-31 this one number did both jobs; see
    #: `soma_anchor_um` below for the split and the measurement behind it. **Read only under
    #: ``opening``** — `anchor_sheet` has its own criterion and does not consult it.
    soma_thickness_um: float = SOMA_THICKNESS_UM_DEFAULT

    #: **How far the guaranteed soma reaches out from the nucleus, µm — mechanism 1's radius.**
    #: Read under **both** methods, because the anchor is identical under both.
    #:
    #: Split out of `soma_thickness_um` on 2026-08-31 and **defaulted to the same value, so no
    #: number moves until someone moves it.** The two mechanisms pull opposite ways, so sharing
    #: one field made both untunable: see `soma.SOMA_ANCHOR_UM_DEFAULT` for the crop2
    #: measurement — the anchor rising from 44% to 91% of the soma across a range whose total
    #: looked flat.
    #:
    #: **Phase A**, like every other soma field: it changes `soma_centroid` on every stored
    #: cell record, which is the Dijkstra root for every branch length and both Sholl profiles.
    #:
    #: **0.9 is a chosen value as of 2026-08-31**, not merely the one it inherited: the user
    #: walked the flow under ``opening``, looked at the soma on item 7b and confirmed it. That
    #: matters here more than anywhere else — on this study's lace masks only 2.2% of a cell
    #: clears the thickness floor (§37), so under ``opening`` **this field is most of the soma
    #: definition**, and a radius nobody had judged would have been carrying the measure.
    soma_anchor_um: float = SOMA_ANCHOR_UM_DEFAULT

    #: **Which criterion decides what is soma BESIDES the anchor.** Defaults to ``anchor_sheet``.
    #: ``opening`` is the morphological opening validated in `EXPORT_SPEC.md` §18, §28 and §34.
    #: Switching changes `soma_volume_um3` and everything rooted at the soma centroid, and
    #: nothing else. **Setting it to ``opening`` restores the previous numbers exactly** — both
    #: paths are live and both are tested.
    #:
    #: **Asked for on item 6.5, as a dropdown, since 2026-08-30** — the first control on that
    #: screen, and the boxes below it follow it. This field spent that morning hidden and fixed;
    #: the user reversed it, because the program otherwise had no way to ask for the soma to be
    #: the nucleus and the cytoplasm around it. `soma.SOMA_METHODS` carries the argument and
    #: `ui.screen_soma.METHOD_FIELDS` carries which boxes each method reads.
    #:
    #: `opening` reads **only** `soma_thickness_um` (besides `soma_anchor_um`, which both read).
    #: The other three keep their defaults under it
    #: rather than whatever was typed under `anchor_sheet`, so a run cannot carry an inert value
    #: into `params_hash` — `ui.screen_soma.SomaParametersScreen._apply_method`.
    soma_method: str = SOMA_METHOD_DEFAULT

    #: Cross-sectional area at which material counts as a sheet rather than a line, µm²,
    #: required in **every** plane. Read by ``anchor_sheet``. **5 µm² is a 2.5 µm disc.** See
    #: `soma.min_plane_area` for why the minimum over the three planes is what refuses a long
    #: process, and why counting pixels instead of microns would take the minimum from the wrong
    #: plane on every voxel. Walked 16 -> 8 -> 6 -> 4 against the screen; the record is in
    #: `soma.SOMA_SHEET_AREA_UM2_DEFAULT`.
    #:
    #: **Offered on item 6.5 since 2026-08-30**, with `soma_thickness_um`, `soma_waist_um` and
    #: `soma_min_sheet_um3`. `soma_bound_um` and `soma_occupancy_theta` were **deleted** the same
    #: day with `bounded_occupancy`, the mechanism they were the whole of — see `soma.SOMA_METHODS`.
    soma_sheet_area_um2: float = SOMA_SHEET_AREA_UM2_DEFAULT

    #: How wide a waist inside the soma may be and still count as soma, µm — a closing of
    #: this radius on the **soma region**, clipped to the cell. Read by ``anchor_sheet``.
    #: It joins two parts of one cell body separated by a constriction that fails the area
    #: test, which the corridor cannot: that bridges a *detached* sheet back to the nucleus
    #: and has no rule for a narrowing inside the body. **0 switches it off.**
    soma_waist_um: float = SOMA_WAIST_UM_DEFAULT

    #: The volume a **detached** sheet must reach before a corridor is run to it, µm³. Read by
    #: ``anchor_sheet``. A piece the anchor can already walk to is the cell body and is never
    #: dropped for its size; only a piece needing a corridor has to earn its place. **30 µm³ is
    #: a 3.9 µm sphere.**
    #:
    #: **A module constant until 2026-08-30**, when the user asked for it on item 6.5. It is the
    #: lever `soma.SOMA_MIN_SHEET_UM3_DEFAULT` names twice for a cell that drags a spurious
    #: component down a process, and it had been walked 30 -> 15 -> 30 -> 60 -> 30 by hand-editing
    #: the source, because no screen could reach it.
    soma_min_sheet_um3: float = SOMA_MIN_SHEET_UM3_DEFAULT

    #: Longest edge of the stored per-cell review thumbnail, px. Written at export time rather
    #: than rebuilt on demand: the batch reviewer opens thousands of them.
    thumbnail_max_px: int = 256

    #: How a cycle in the **soma-free** skeleton is opened, if at all. One of
    #: `skeleton.LOOP_PRUNE_METHODS`. A **Phase C** knob like `min_spur_um` below.
    #:
    #: **`none` by default, at the user's instruction 2026-09-01**, and AnalyzeSkeleton
    #: defaults the same way: a loop may be two processes of one cell genuinely touching,
    #: which is a real topological feature rather than noise. Loops are always *counted*
    #: whatever this says, because detection is lossless and pruning is not.
    #:
    #: **The soma-anchored tree never reads this.** It is acyclic by construction, so the
    #: setting would mean nothing there — and applying it to the shared skeleton instead
    #: would let this silently move the tree as well.
    #:
    #: It is a real field now rather than when the item 8 dropdown lands, because it
    #: changes measurement output and so must be in `hash()`. Adding it later would move
    #: the hash a second time and orphan any run made in between.
    loop_prune_method: str = LOOP_PRUNE_DEFAULT

    #: Skeleton spurs shorter than this are pruned before any branch measure is taken, µm.
    #: A **Phase C** knob, like `soma_thickness_um` above and the two measurement thresholds.
    #:
    #: **A real field since 2026-08-12 (NEW-g).** It was read as
    #: `getattr(params, "min_spur_um", 1.0)`, and `hash()` digests `asdict(self)`, so it
    #: reached neither the parameters file nor `params_hash`: **two runs pruned at different
    #: lengths hashed identically**, and the defensive read made it look settable when nothing
    #: could set it. Adding it changes the hash for otherwise-identical settings — done while
    #: no pool existed anywhere, which is the only time that is free.
    #:
    #: **1.0 is measured, not chosen by eye** — 152 real cells, `tuning_runs/spur_sweep_all.txt`,
    #: table and full argument at `morphometrics.measure_cell`. It removes **31% of all
    #: branches** while the WT/5xFAD endpoint ratio holds at ~1.98, and it sets `n_endpoints`
    #: directly: **5xFAD 40, WT 79**. Max branch length is unchanged (37.0 → 37.0), so it
    #: provably touches dead ends only.
    #:
    #: **What would justify moving it:** more data shifting the plateau's far edge. That edge
    #: is visible today — at 2.0 the ratio falls to 1.88 and skeleton length drops 9% — so 1.0
    #: sits inside a plateau rather than on a monotone curve.
    #:
    #: **User-settable, and — like `min_plaque_volume_um3` — deliberately not a UI picker.**
    #: There is no visual anchor for a pruning length: nobody can look at a field and see
    #: 1 µm of spur. The sweep is the instrument here, not the eye.
    #: > ### ⇒ THE DEFAULT IS 0.0 SINCE 2026-09-01, AND THAT IS NOT A CLAIM THAT 0.0 IS BEST
    #: >
    #: > 1.0 was derived twice and both derivations are superseded. The first ran over the
    #: > **voxel-space** skeleton, which read one process between 5.40 and 11.95 µm by
    #: > orientation alone (§54). The re-derivation of §55 fixed that but still pruned
    #: > *before* the soma collapse, and pruning now happens **after** it (§57) — so the
    #: > threshold applies to processes as they leave the cell body rather than to raw
    #: > segments, and A2 has changed what `n_endpoints` counts underneath it.
    #: >
    #: > **0.0 applies no filter at all**, which is the honest default while the value is
    #: > unvalidated: this project does not ship invented numbers, and a threshold carried
    #: > forward from a measurement that no longer exists is an invented number wearing an
    #: > old sweep's clothes. The user sets it on item 8, and the table below is kept as the
    #: > record of how 1.0 was reached rather than as a live recommendation.
    min_spur_um: float = 0.0

    def hash(self) -> str:
        """Stable digest, recorded on every cell so results trace to their settings."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]

    def phase_a_hash(self) -> str:
        """The **pool** key — Phase A fields only. `COMPONENT_03_PLAN.md` §4a.9.

        `hash()` digests every field and is the right key for tracing a *result* to its
        settings. It is the wrong key for deciding which pool a *cell* belongs in, because a
        measurement knob would strand the cell set:

        * change `min_spur_um` — a skeleton-pruning length touching no mask — and you get a new,
          empty pool, repopulated only by re-running Cellpose at 200-500 s an image;
        * and the case that lands on Component 3's own flow: **`cd68_threshold` and
          `plaque_threshold` are `None` until item 9**, so picking the two study thresholds — the
          last act before saving — would move the whole session's cells into a *different*, empty
          pool. **The pickers would invalidate the pool they were picked from.**

        Two questions, two keys. See `PHASE_A_FIELDS` for what is in it and why
        `soma_thickness_um` is, despite being documented as a measurement knob.
        """
        subset = {name: getattr(self, name) for name in PHASE_A_FIELDS}
        payload = json.dumps(subset, sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:8]


#: **Read during measurement and nowhere else**, so none of them changes which cells exist or
#: what their stored arrays contain. `COMPONENT_03_PLAN.md` §4a.9.
PHASE_C_ONLY_FIELDS: tuple[str, ...] = (
    "cd68_threshold",
    "loop_prune_method",
    "min_plaque_volume_um3",
    "min_spur_um",
    "plaque_threshold",
)

#: **Every field that affects which cells exist or what their stored arrays contain** — the pool
#: key. **37 fields since 2026-09-01**, when `loop_prune_method` was added — and it is Phase C
#: only, because it reshapes a skeleton at measurement time and changes no cell's existence or
#: stored arrays. It was 37 from 2026-08-31, when `soma_anchor_um` split off
#: `soma_thickness_um`; that one is Phase A for the same reason every other soma field is, and
#: landed there without a hand edit because this tuple is the complement of
#: `PHASE_C_ONLY_FIELDS`.
#:
#: **`soma_thickness_um` is in the subset despite being documented as a measurement knob**, and the
#: exception is the point: `segment_image` calls `find_soma(..., thickness_um=params.soma_thickness_um)`
#: to compute `soma_centroid` on **every stored cell record**, so it is read in both phases and
#: changing it changes stored Phase A output. The dataclass's own grouping is *incomplete, not
#: wrong* — and it is the second place that same trap has appeared, after §4 item 8's read-only
#: classification.
#:
#: **Derived, not transcribed.** A hand-listed subset rots silently: omit a field and changing it
#: does not move the pool key, so cells made differently land in the same pool and nothing errors.
#: `PHASE_C_ONLY_FIELDS` is the only list maintained by hand, and `test_pipeline` asserts the two
#: partition the dataclass exactly — a new parameter belongs to neither until a human classifies
#: it, and the suite stays red until one does.
PHASE_A_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(SegmentationParams) if f.name not in PHASE_C_ONLY_FIELDS
)


@dataclass
class Derivation:
    """The finished cell set **before it is written** — arrays, not crops.

    `COMPONENT_03_PLAN.md` §4a.5 as amended: semi-auto holds steps 1-7 entirely in memory and
    the first write happens at item 8, so the computation and the write had to stop being one
    act. **This is the seam, and it is deliberately not a second code path** — `payloads()` is
    what both routes iterate, so batch and the interactive flow cannot drift into writing
    different cells from the same parameters. That is §6.1's reproducibility gate held by
    construction rather than by discipline.

    > **The memory cost is one array, not sixty**, and that is why this holds the label volume
    > rather than a list of crops. `payloads()` is a **generator**: batch consumes it one cell at
    > a time straight into `PoolWriter`, and item 8 does the same at save time. Materialising it
    > would put ~60 cell crops beside the channels the session is already holding.
    """

    #: The final label volume at native resolution — what item 7 reviews and item 8 writes.
    labels: np.ndarray
    nuclei_labels: np.ndarray
    segmentation_filled: np.ndarray
    voxel_size_um: tuple[float, float, float]
    n_cells: int
    nuclei_by_cell: dict[int, list[int]] = field(default_factory=dict)
    split_methods: dict[int, str] = field(default_factory=dict)
    soma_thickness_um: float = SOMA_THICKNESS_UM_DEFAULT
    soma_anchor_um: float = SOMA_ANCHOR_UM_DEFAULT
    #: Carried so `payloads` measures the soma the parameters asked for. See
    #: `SegmentationParams` for what each means and `soma.find_soma` for what they do.
    soma_method: str = SOMA_METHOD_DEFAULT
    soma_sheet_area_um2: float = SOMA_SHEET_AREA_UM2_DEFAULT
    soma_waist_um: float = SOMA_WAIST_UM_DEFAULT
    soma_min_sheet_um3: float = SOMA_MIN_SHEET_UM3_DEFAULT
    thumbnail_max_px: int = 256

    #: **A separate array from `labels`, and that is what keeps items 6 and 7a unchanged.**
    #: Both screens read `derivation.labels`; debris lives here instead, so it cannot reach a
    #: size ramp, a thumbnail or a triage tile without a screen asking for it by name. Item 7b
    #: asks, because there debris is scene rather than something to review.
    debris_mask: np.ndarray | None = None

    #: Labels a reviewer changed by hand in item 7b, so `payloads()` can stamp `human_edited`.
    #:
    #: > ### THE COLUMN EXISTED EVERYWHERE EXCEPT WHERE IT IS SET — 2026-08-26
    #: >
    #: > `human_edited` is a field on `CellRecord`, a column in the `cells` table, an argument
    #: > to `add_cell` and a member of `CELL_QC` — and **nothing in the program ever wrote it
    #: > `True`**. It is the one column that tells an analyst a cell's numbers are not
    #: > reproducible from the parameters file, which is exactly what an edited cell is.
    #:
    #: A set of label ids rather than a flag per payload, because the edit arrives from item 7
    #: long after `payloads()` last ran and the labels are what the two halves share.
    human_edited: set[int] = field(default_factory=set)

    #: **A segment-time diagnostic, not an exported figure — schema 3, 2026-08-25.** It counts
    #: what no DERIVED cell claimed; the export needs what no SURVIVING cell claims, which is a
    #: different number as soon as a reviewer rejects one as debris. Phase C computes that from
    #: the measurement mask it rebuilds. This exists so the batch log can say how much material
    #: went unclaimed before anyone looked, and `cd68_integrated` is deliberately NaN here.
    debris: DebrisLoad | None = None

    def payloads(self, *, threads: int | None = None):
        """One `PoolWriter.add_cell` keyword-argument dict per cell, in label order.

        Every quantity here is derived from the arrays rather than accepted from a caller, so a
        mistake at one call site cannot produce a record whose invariants disagree — the same
        rule `add_cell` itself follows for `n_voxels` and `bbox_diag_um`.

        > ### ⇒ COMPUTED A CHUNK AT A TIME, YIELDED IN LABEL ORDER — CPU patch, §3 A6
        >
        > The per-cell work is strictly per-cell: each payload is derived from that cell's own
        > crops and touches nothing shared. It is dominated by `soma_for`, measured at **~0.4 s
        > per call on real cells** — 14-24 s an image at 35-60 cells — and per-cell soma work
        > threads at ~2.1x.
        >
        > **It stays a generator, and the chunking is why.** This class's own docstring says
        > *"the memory cost is one array, not sixty"* and that materialising every payload would
        > put ~60 cell crops beside the channels the session already holds. A plain
        > `map_ordered` over every label would do exactly that. So a chunk of `threads` labels
        > is computed in parallel and yielded in order before the next chunk starts, which
        > bounds the extra crops at the thread count rather than the cell count — 16, not 60 —
        > and leaves `writer.add_cell` consuming one at a time exactly as before.
        >
        > **In label order, and that is not cosmetic.** `add_cell` assigns cell ids in
        > iteration order, and item 7's `ReviewStep` addresses cells *by position in
        > `payloads()`*. Reordering them would silently attach every review decision to a
        > different cell.
        """
        boxes = ndimage.find_objects(self.labels)
        wanted = [label for label in range(1, self.n_cells + 1)
                  if boxes[label - 1] is not None]
        n_threads = cpu.count(threads)
        for start in range(0, len(wanted), n_threads):
            chunk = wanted[start:start + n_threads]
            yield from cpu.map_ordered(
                lambda label: self._payload(label, boxes[label - 1]), chunk, n_threads)

    def _payload(self, label: int, box) -> dict:
        """One cell's `add_cell` arguments. **Reads only; shares nothing between calls.**

        Split out of `payloads` so the per-cell work can be handed to a worker. Everything it
        touches on `self` is read — the label volume, the nuclei labels, the segmentation mask
        and the six soma settings — and everything it allocates is its own crop.
        """
        labels = self.labels
        mask = labels[box] == label
        nucleus_crop = np.where(mask, self.nuclei_labels[box], 0).astype(np.uint16)
        own = np.isin(nucleus_crop, list(self.nuclei_by_cell.get(label, [])) or [0])
        nucleus_crop = np.where(own, nucleus_crop, 0).astype(np.uint16)

        #: > ### THESE FIELDS ARE WHY ITEM 6.5 NEEDS NO RE-DERIVATION
        #: >
        #: > The soma is measured **here**, lazily, from the values the `Derivation` carries —
        #: > not baked in when the derivation was built. Item 6.5 sits *after* the derive
        #: > (item 6 triggers it), so a screen that could only set `session.chosen` would have
        #: > arrived too late and the export would have measured the defaults. Writing the
        #: > four values onto the cached derivation is enough, and it costs nothing.
        #: **`soma_for` rather than seven arguments spelled out here**, since 2026-09-03.
        #: Triage needs the soma region and the pool does not store it, so both review
        #: routes recompute it — three call sites, and one of them must not drift.
        soma = soma_for(self, mask, nucleus_crop, self.voxel_size_um)
        offset = np.array([sl.start for sl in box], dtype=float)
        centroid = (tuple(float(c) for c in (np.array(soma.centroid) + offset))
                    if soma.centroid is not None else None)

        # Which nuclei belong to this cell comes from the split assignment, not
        # from whatever labels happen to fall inside the mask. Nuclei are upsampled
        # nearest-neighbour and are not clipped to cell boundaries, so a neighbour's
        # nucleus can graze this crop; counting those would overstate the count and
        # imply clumps that were in fact correctly separated.
        nucleus_ids = [int(n) for n in self.nuclei_by_cell.get(label, [])]
        centroids = []
        for nid in nucleus_ids:
            voxels = nucleus_crop == nid
            if voxels.any():
                centre = ndimage.center_of_mass(voxels)
                centroids.append(tuple(float(c) for c in (np.array(centre) + offset)))
        nucleus_ids = [n for n in nucleus_ids if (nucleus_crop == n).any()]

        return dict(
            mask=mask,
            bbox=(box[0].start, box[0].stop, box[1].start, box[1].stop,
                  box[2].start, box[2].stop),
            nucleus_mask=nucleus_crop,
            thumbnail=_thumbnail(mask, nucleus_crop, self.thumbnail_max_px),
            nucleus_ids=nucleus_ids,
            nucleus_centroids=centroids,
            split_method=self.split_methods.get(label, "unsplit"),
            human_edited=label in self.human_edited,
            n_voxels_segmentation=int((mask & self.segmentation_filled[box]).sum()),
            soma_centroid=centroid,
            soma_remnant_found=soma.remnant_found,
            touches_xy_border=bool(
                box[1].start == 0 or box[2].start == 0
                or box[1].stop == labels.shape[1] or box[2].stop == labels.shape[2]
            ),
            touches_z_border=bool(
                box[0].start == 0 or box[0].stop == labels.shape[0]
            ),
        )


def apply_cell_edit(derivation: Derivation, label: int, box, mask) -> dict:
    """Fold one reviewer-edited cell back into the derivation. **In place.**

    > ### ITEM 7b's EDITS REACHED NOTHING, AND THIS IS WHERE THEY LAND — 2026-08-26
    >
    > `_leave` recorded `(status, reason, note)` and stopped. The edited mask lived on
    > `CellEditor` and was discarded; `session.cells = derivation.payloads()` then rebuilt every
    > cell from a label volume nobody had touched, so **every add and every remove was thrown
    > away** and the pool received the Phase A cell.
    >
    > **Painting the label volume rather than overriding the payload** is the same shape item 6
    > already uses — `apply_cell_size_cutoff` mutates the derivation in place and lets everything
    > re-derive. It is also what `payloads()`'s own rule requires: *"every quantity here is
    > derived from the arrays rather than accepted from a caller, so a mistake at one call site
    > cannot produce a record whose invariants disagree."* An override would be that call site,
    > and after an edit the bbox, thumbnail, soma, nucleus assignment, voxel counts and border
    > flags all move together — re-deriving gets every one of them right for free.
    >
    > **The transfer's paired write falls out of it.** A voxel carries one label, so painting a
    > neighbour's process into this cell removes it from the neighbour by construction. There is
    > no second write and no way for the two to disagree.

    `box` is the half-open editable frame in image coordinates and `mask` is the cell's final
    shape within it. **The frame must contain the cell's whole original footprint** — it is the
    bounding box grown by the margin's maximum, so it does — and that is asserted rather than
    assumed, because a box that clipped the old footprint would leave orphaned voxels wearing
    this cell's label outside it.

    Returns what moved, so a caller can log or assert on it.
    """
    labels = derivation.labels
    label = int(label)
    z0, z1, y0, y1, x0, x1 = (int(v) for v in box)
    region = labels[z0:z1, y0:y1, x0:x1]
    mask = np.asarray(mask).astype(bool)
    if mask.shape != region.shape:
        raise ValueError(f"the edited mask is {mask.shape}, the frame is {region.shape}")

    was = region == label
    #: **The whole cell, or the arithmetic below is wrong.** Voxels of this label outside the
    #: frame would keep it after the paint and read as a second component of the same cell.
    if int(was.sum()) != int((labels == label).sum()):
        raise ValueError(f"cell {label} reaches outside the frame it was edited in")

    gained = mask & ~was
    lost = was & ~mask
    donors: dict[int, int] = {}
    for other in np.unique(region[gained]):
        other = int(other)
        if other and other != label:
            donors[other] = int((region[gained] == other).sum())

    region[was] = 0
    region[mask] = label

    #: **Removed material becomes debris and taken debris stops being debris**, which is the
    #: same bookkeeping `apply_cell_size_cutoff` does for a whole cell. Voxels taken from a
    #: neighbour were never in `debris_mask`, so clearing them there is a no-op rather than a
    #: special case.
    if derivation.debris_mask is not None:
        debris = derivation.debris_mask[z0:z1, y0:y1, x0:x1]
        debris |= lost
        debris &= ~gained

    #: > ### ⇒ `size_ratios` AND `median_ratio` WERE MAINTAINED HERE UNTIL 2026-09-03
    #: >
    #: > An edit moved one cell's volume-per-nucleus and the population median with it, so both
    #: > were recomputed on every edit and again on every cutoff and every renumbering. All of
    #: > that existed to produce **one** field, `size_outlier_score`, and triage stopped ranking
    #: > on it — see `prestomorph.qc`. Nothing reads a ratio now, so nothing maintains one.
    #: >
    #: > The signals that replaced it are recomputed from the mask and the soma at review time,
    #: > which is why an edit needs no bookkeeping here at all: the next ranking measures the
    #: > mask as it now is.

    derivation.human_edited.add(label)
    return {"gained": int(gained.sum()), "lost": int(lost.sum()), "donors": donors}


def apply_cell_size_cutoff(derivation: Derivation, min_volume_um3: float) -> int:
    """Drop cells under `min_volume_um3` and fold them into debris. **In place.**

    > ### ITEM 6's BOX CHANGED THE PICTURE AND NOTHING ELSE — the defect this closes
    >
    > `min_cell_volume_um3` is read in exactly one place, `segment_image`. The interactive
    > flow derives on **arrival** at item 6 (`app._enter`, `Step.SIZE`) and caches the result
    > behind `if session.derivation is None`, which nothing invalidates. The screen's own
    > `__init__` calls `_sync()` first, writing its `DEFAULT_CUTOFF = "0"` into
    > `session.chosen` — so the derive runs at **0.0** and the number the user then types
    > removes nothing at all.
    >
    > **Measured, not inferred.** One synthetic cell of 256 µm³, cutoff typed at 2560:
    > the screen read `cells kept: 0 of 1`, `session.chosen` held `2560.0`, and item 7a still
    > received the cell. The parameters file recorded 2560 — so a batch re-run from the file a
    > session wrote produced a **different cell set from the session that wrote it**.

    **`filter_cells` does the work, not a second implementation of it.** That is what holds
    §6.1's reproducibility gate: `filter_cells` renumbers survivors in ascending old-label
    order, so filtering at 0 and then at *X* lands on exactly the labels, ids and masks that
    filtering at *X* once would have — which is what `test_the_cutoff_composes_with_the_derive`
    asserts rather than assumes.

    **No debris re-measure, and that is a consequence of a later decision.** Debris is
    recomputed in Phase C as *measurement mask − (confirmed + rejected-real)*, so a cell
    removed here never reaches the pool and is debris there by subtraction. All this has to
    keep true is `debris_mask`, which item 7b draws. Nothing needs the CD68 channel, which
    `Compute.derive` has already released.

    **Monotonic, not reversible.** Re-entering item 6 and raising the cutoff removes more;
    lowering it restores nothing, because the labels are gone. The user's decision of
    2026-08-25, taken with the alternative — a non-destructive view — described.

    Returns how many cells were removed, so the caller can log or assert on it.
    """
    labels = derivation.labels
    if labels is None or derivation.n_cells == 0:
        return 0
    cutoff = float(min_volume_um3)
    if cutoff <= 0.0:
        return 0

    trimmed = filter_cells(labels, min_volume_um3=cutoff,
                           voxel_size_um=derivation.voxel_size_um)
    if trimmed.n_after == trimmed.n_before:
        return 0

    #: Computed **before** the labels are replaced, and it is the only extra array this
    #: function allocates — one boolean at native resolution, transient.
    doomed = (labels > 0) & (trimmed.labels == 0)
    derivation.labels = trimmed.labels
    derivation.debris_mask = (doomed if derivation.debris_mask is None
                              else (derivation.debris_mask | doomed))

    #: New id -> old id, exactly as `segment_image` builds it. Every per-cell record is keyed
    #: by label, and `filter_cells` renumbers, so indexing these with a new id before the
    #: remap silently returns another cell's provenance.
    origin = {int(new): int(old) for old, new in trimmed.relabel_map.items()}
    derivation.nuclei_by_cell = {new: derivation.nuclei_by_cell.get(old, [])
                                 for new, old in origin.items()}
    derivation.split_methods = {new: derivation.split_methods.get(old, "unsplit")
                                for new, old in origin.items()}
    derivation.n_cells = int(trimmed.n_after)

    return int(trimmed.n_before - trimmed.n_after)


@dataclass
class PipelineResult:
    """What happened, in enough detail to diagnose a batch run from its log."""

    source_path: str
    file_uid: str
    fragment_path: Path | None
    n_cells: int
    stages: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    #: Set only when `segment_image` was asked to derive without writing — `pool_root=None`.
    #: **`fragment_path` is then `None`**, which is the honest statement that nothing is on
    #: disk, rather than a path to a file that was never created.
    derivation: "Derivation | None" = field(default=None, repr=False)

    def summary(self) -> str:
        total = sum(self.timings.values())
        return f"{Path(self.source_path).name}: {self.n_cells} cells in {total:.0f}s"

    def report(self) -> str:
        lines = [self.summary()]
        for stage, text in self.stages.items():
            lines.append(f"  {stage:<12} {text}  ({self.timings.get(stage, 0):.1f}s)")
        return "\n".join(lines)


def _thumbnail(cell: np.ndarray, nucleus: np.ndarray, max_px: int) -> np.ndarray:
    """Two-channel max projection for the review contact sheet.

    Stored at write time rather than generated on demand: the batch reviewer opens
    thousands of these, and rebuilding each from a source stack would make it unusable.
    """
    panes = [cell.max(axis=0).astype(np.uint8) * 255,
             (nucleus > 0).max(axis=0).astype(np.uint8) * 255]
    stacked = np.stack(panes)
    longest = max(stacked.shape[1:])
    if longest > max_px:
        step = int(np.ceil(longest / max_px))
        stacked = stacked[:, ::step, ::step]
    return stacked


def segment_image(
    source_path: str | Path,
    pool_root: str | Path | None,
    *,
    roles: ChannelRoles | Mapping[str, int],
    params: SegmentationParams | None = None,
    run_id: str = "run",
    expected_voxel_size_um: Sequence[float] | None = None,
    detector: Detector | None = None,
    gpu: bool | None = None,
    verify_roles: bool = True,
    progress: bool = False,
    channels: Mapping[str, np.ndarray] | None = None,
    cell_mask=None,
    measurement_mask: np.ndarray | None = None,
    plaque_mask: np.ndarray | None = None,
    nuclei_result=None,
) -> PipelineResult:
    """Segment one image and write its candidate cells to the pool.

    Raises rather than guessing when the file contradicts what was declared — a wrong
    channel assignment or voxel size produces plausible-looking output rather than an
    obvious failure, so neither is allowed to pass silently.

    > ### ⇒ THE LAST FIVE ARGUMENTS ARE *"I ALREADY HAVE THIS"* — patch item A10, 2026-09-03
    >
    > **The batch route passes none of them and is byte-for-byte unchanged.** They exist for the
    > interactive route, which arrives at item 6 holding four products and five channels that
    > this function would otherwise rebuild from disk — including a Cellpose run of **200-500 s**
    > that item 4's COMMIT paid minutes earlier at the same parameters.
    >
    > **Every one of them falls back to computing.** A caller that passes nothing, or passes
    > `None` for one of them, gets exactly what it always got — which is what keeps §6.1's
    > reproducibility gate honest: there is one implementation, and reuse can only ever skip
    > recomputing an answer, never produce a different one. The tests assert each reused product
    > equals what this function would have built.
    """
    params = params or SegmentationParams()
    source_path = Path(source_path)
    stages: dict[str, str] = {}
    timings: dict[str, float] = {}

    def mark(stage: str, started: float, text: str) -> None:
        timings[stage] = time.perf_counter() - started
        stages[stage] = text
        if progress:
            print(f"  {stage:<12} {text}")

    # ---- load ------------------------------------------------------------
    t = time.perf_counter()
    src = ImageSource(source_path, roles,
                      expected_voxel_size_um=expected_voxel_size_um)
    if verify_roles:
        check = src.verify_roles()
        if check.suspect_swap:
            raise ValueError(
                f"{source_path.name}: {check.summary()}. Refusing to proceed — "
                "segmenting the wrong channel yields plausible but meaningless output."
            )
    voxel = src.voxel_size_um
    voxel_small = voxel_size_after_downsample(voxel, params.downsample)
    mark("load", t, src.meta.describe())

    #: The cell channel, once it has been read. See `channel` for why this one and no other.
    held_cell: np.ndarray | None = None

    def channel(role: str) -> np.ndarray:
        """The role's voxels — **from the caller when it already holds them**, else from disk.

        The interactive session has all four channels in memory when it reaches item 6, and this
        function read five of them back off the file: `cell` three times, plus `nuclei` and
        `plaque`. Sharing the arrays it already has is not a second copy — it is one copy
        instead of two, which is the direction §2's memory row points.

        > ### ⇒ AND ON THE BATCH ROUTE THE CELL CHANNEL IS HELD, BECAUSE NOBODY HANDS IT OVER
        >
        > `ImageSource.channel` reads from disk every time and **deliberately does not cache** —
        > its own docstring says so, because one full-field channel is ~850 MB and the design
        > note is that the caller decides what to keep. This is the caller deciding.
        >
        > With no `channels` mapping there are three reads of the same role: item 3's mask, item
        > 5's mask, and the array `split_objects` is given. On Lustre, ~573 times over. So the
        > hold is here rather than in `ImageSource`, where it would change memory behaviour for
        > every caller including the UI.
        >
        > **Only `cell`.** `nuclei` and `plaque` are read once each, so holding them would add
        > ~850 MB apiece to the peak and save nothing.
        """
        held = None if channels is None else channels.get(role)
        if held is not None:
            return held
        if role != "cell":
            return src.channel(role)
        nonlocal held_cell
        if held_cell is None:
            held_cell = src.channel(role)
        return held_cell

    # ---- masks -----------------------------------------------------------
    t = time.perf_counter()
    #: **Item 3's committed mask when there is one.** Accepted as either a `CellMask` or the
    #: filled array itself, because `products[Step.THRESHOLD_1]` has been both.
    cell = cell_mask if cell_mask is not None else build_cell_mask(
        channel("cell"),
        coefficient=params.cell_coefficient,
        min_object_voxels=params.min_object_voxels,
        voxel_size_um=voxel,
        closing_radius_um=params.closing_radius_um,
    )
    cell_filled = getattr(cell, "filled", cell)
    #: **Item 2's committed veto, which is the UNDILATED mask** — `detect_nuclei` applies
    #: `plaque_dilate_um` itself, and that is the contract `products[Step.VETO]` already carries.
    plaque = plaque_mask
    if plaque is None and src.has_role("plaque"):
        plaque = threshold_channel(
            channel("plaque"),
            coefficient=params.plaque_veto_coefficient,
            min_object_voxels=50,
        ).mask
    mark("threshold", t, f"{100 * cell_filled.mean():.1f}% of volume is cell")

    # ---- nuclei ----------------------------------------------------------
    t = time.perf_counter()
    nuclei = nuclei_result if nuclei_result is not None else detect_nuclei(
        channel("nuclei"),
        diameter_px=diameter_in_pixels(params.nucleus_diameter_um, voxel,
                                       params.downsample),
        detector=detector,
        #: **The device, and it is deliberately not a parameter.** `None` means *use a GPU if
        #: there is one*; the value never reaches `SegmentationParams`, because that dataclass
        #: is hashed and the hash names the pool — a CPU run and a GPU run of the same study
        #: must land in the same pool. See `nuclei.resolve_device`.
        gpu=gpu,
        stitch_threshold=params.stitch_threshold,
        downsample=params.downsample,
        plaque_mask=plaque,
        plaque_dilate_um=params.plaque_dilate_um,
        voxel_size_um=voxel,
        min_volume_voxels=params.nucleus_min_volume_voxels,
        normalize=params.normalize,
        #: **The batch route filters here, from the parameters file.** The interactive route
        #: hands over the volume its own screen already filtered and passes a zero strength, so
        #: the subtraction lands exactly once on either path — the same contract
        #: `products[Step.VETO]` carries for the undilated plaque mask.
        background_sigma_um=params.dapi_background_scale_um,
        background_strength=params.dapi_background_strength,
    )
    mark("nuclei", t, nuclei.summary())

    # ---- membership ------------------------------------------------------
    t = time.perf_counter()
    cell_small = downsample_mask_xy(cell_filled, params.downsample)
    member = filter_nuclei_by_membership(
        nuclei.labels, cell_small, threshold=params.membership_threshold
    )
    mark("membership", t, member.summary())

    # ---- ONE STAGE: the measurement mask, split by the microglial nuclei --
    #
    # **Threshold 2 is REQUIRED.** The user's decision, 2026-08-23: *"no if threshold 2 is
    # unset program cant work. any semi auto user will set threshold 2 while creating the
    # parameters file for batch anyways."* There is deliberately no one-mask fallback — a
    # silent second architecture is how two routes drift into answering the same parameters
    # differently, and this one would differ in what a cell *is*.
    t = time.perf_counter()
    if params.mask_coefficient is None:
        raise ValueError(
            "mask_coefficient is unset. The measurement threshold is the mask that gets "
            "split into cells, so there is nothing to segment without it — choose it on "
            "item 5, or set it in the parameters file before running a batch."
        )

    # **`fill_holes=False` — patch B.** Hole filling exists so `membership_fractions` works:
    # Iba1 is cytoplasmic, so an unfilled mask leaves each nucleus sitting in a hole and every
    # fraction collapses toward zero. Membership ran against the *segmentation* mask above,
    # before this, so nothing downstream of here needs its lumens closed — and closing them
    # inflates every exported volume by the lumen of each soma.
    #
    # **The closing still runs.** `mask_closing_radius_um` bridges gaps in the outline, which is
    # a different operation answering a different objection, and it is a control item 5 shows.
    # Reading `.raw` instead of passing this flag would have silently made that control a no-op.
    #: **Item 5's committed mask when there is one.** It is what `commit_result` stored, which
    #: is what this call returned at that commit — the same array, not an equal one.
    measure_filled = measurement_mask
    if measure_filled is None:
        measure_filled = build_cell_mask(
            channel("cell"),
            coefficient=float(params.mask_coefficient),
            min_object_voxels=params.measurement_min_object_voxels,
            voxel_size_um=voxel,
            closing_radius_um=params.mask_closing_radius_um,
            fill_holes=False,
        ).filled
    measure_small = downsample_mask_xy(measure_filled, params.downsample)

    # **The measurement mask is what gets split now — the segmentation mask no longer
    # decides how many cells there are or where they divide.** Item 3 survives for exactly
    # one job: `filter_nuclei_by_membership` above, which is what makes a nucleus microglial.
    # One stage replaces the old segment-then-grow pair, so a cell's shape is never assembled
    # from two thresholds and the filled lumens can never be restored from the other side.
    objects = find_cell_objects(measure_small)
    split = split_objects(
        objects, member.labels,
        downsample_xy(channel("cell"), params.downsample),
        voxel_size_um=voxel_small,
        method=params.split_method,
        smooth_um=params.smooth_um,
        min_overlap_fraction=params.min_overlap_fraction,
    )
    mark("split", t, split.summary())

    # ---- cells and debris, at native resolution --------------------------
    #
    # **Every component no microglial nucleus claims is DEBRIS, not a discard.** The user's
    # argument for keeping it, 2026-08-23: *"anything genuinely not worth keeping is deleted
    # by minimum voxels filter in the iba1 thresholding step."* Whatever survives that floor
    # was surveyed, so counting it as unsurveyed — which is what discarding it amounted to —
    # was a false statement about the field rather than a missing feature.
    t = time.perf_counter()
    claimed = np.zeros(int(split.labels.max()) + 1, dtype=split.labels.dtype)
    for label, nuclei_here in split.nuclei_by_cell.items():
        if nuclei_here:
            claimed[int(label)] = int(label)
    cells_small = claimed[split.labels]

    # **`assign_labels_by_connectivity` is the mechanism here, not a leftover of the old
    # architecture.** Upsampling on its own quantises every boundary to `downsample` and
    # blunts exactly the thin processes the analysis measures, while
    # `assign_labels_to_native`'s unbounded nearest-label fill would attach a detached blob
    # of debris to whichever cell happened to be closest — silently, and into that cell's
    # skeleton and its CD68 territory. Its three rules are what this stage needs: a component
    # holding one cell becomes that cell whole, a component holding several is arbitrated by
    # nearest label, and **a component holding none is left alone** — which under this patch
    # is no longer "counted and thrown away" but the definition of debris itself.
    seed = np.where(measure_filled,
                    upsample_xy(cells_small, params.downsample, measure_filled.shape[1:]), 0)
    measurement = assign_labels_by_connectivity(seed, measure_filled, voxel_size_um=voxel)

    # The user's cell-size cutoff, applied to the measurement-mask volumes it was chosen
    # against. **It no longer deletes**: what falls under it stops being a cell and becomes
    # debris, the second half of the definition agreed 2026-08-23. Debris is then whatever
    # the measurement mask holds that no surviving cell holds, so both halves fall out of one
    # expression instead of being accumulated separately and kept in step by hand.
    trimmed = filter_cells(
        measurement.labels,
        min_volume_um3=params.min_cell_volume_um3,
        voxel_size_um=voxel,
    )
    native = trimmed.labels
    debris_mask = measure_filled & (native == 0)

    # Nuclei are upsampled, *not* region-assigned. `assign_labels_to_native` gives every
    # foreground voxel the nearest label, which is right for cells — each cell voxel
    # must belong to some cell — and wrong for nuclei, which occupy a small part of the
    # cell. Region-assigning them smears each nucleus across the whole cell mask,
    # inflating the recorded nucleus count and corrupting the soma-nucleus distance.
    nuclei_native = upsample_xy(member.labels, params.downsample, cell_filled.shape[1:])
    #: **`cd68=None` on purpose, since schema 3.** The exported debris figures are Phase C's,
    #: recomputed against the cell set review produced; this call survives only to put a line in
    #: the batch log. Passing the channel here would read a whole channel — the one read
    #: `segment_image` would otherwise not make — for a number nothing consumes.
    debris = measure_debris(debris_mask, voxel_size_um=voxel, cd68=None)
    mark("filter", t, f"{trimmed.summary()}; {debris.summary()}")

    # ---- derive, then write ----------------------------------------------
    t = time.perf_counter()
    n_cells = int(native.max())

    # Final cell id back to the split label it came from. Built explicitly because the
    # ids are renumbered at every filtering step, and per-cell provenance — which nuclei
    # a cell owns, how it was split — is keyed by the *original* label. Indexing those
    # dictionaries with a renumbered id silently returns another cell's record.
    #
    # **One hop rather than two since the debris patch.** `assign_labels_by_connectivity`
    # carries the split ids through unchanged, so `filter_cells` is the only renumbering
    # between splitting and the final set and its `relabel_map` is the whole provenance
    # chain. The old two-step composition existed because the speck floor renumbered first.
    origin: dict[int, int] = {int(new): int(old)
                              for old, new in trimmed.relabel_map.items()}
    nuclei_by_cell = {new: split.nuclei_by_cell.get(old, [])
                      for new, old in origin.items()}

    derivation = Derivation(
        labels=native,
        nuclei_labels=nuclei_native,
        segmentation_filled=cell_filled,
        voxel_size_um=voxel,
        debris_mask=debris_mask,
        debris=debris,
        n_cells=n_cells,
        nuclei_by_cell=nuclei_by_cell,
        split_methods={label: split.method_by_cell.get(origin.get(label, -1), "unsplit")
                       for label in range(1, n_cells + 1)},
        soma_thickness_um=params.soma_thickness_um,
        soma_anchor_um=params.soma_anchor_um,
        soma_method=params.soma_method,
        soma_sheet_area_um2=params.soma_sheet_area_um2,
        soma_waist_um=params.soma_waist_um,
        soma_min_sheet_um3=params.soma_min_sheet_um3,
        thumbnail_max_px=params.thumbnail_max_px,
    )

    #: **`pool_root=None` derives without writing**, which is what `COMPONENT_03_PLAN.md` §4a.5
    #: needs for semi-auto: steps 1-7 are entirely in memory and item 8 is the first write. The
    #: cells are identical either way because both routes iterate `Derivation.payloads()` — *"the
    #: same code, which is what the reproducibility gate requires. No second write path to
    #: diverge."*
    if pool_root is None:
        mark("write", t, f"{n_cells} cells derived, not written")
        return PipelineResult(
            source_path=str(source_path),
            file_uid="",
            fragment_path=None,
            n_cells=n_cells,
            stages=stages,
            timings=timings,
            derivation=derivation,
        )

    with PoolWriter(
        pool_root,
        source_path=str(source_path),
        voxel_size_um=voxel,
        image_shape=(src.meta.n_z, *src.meta.yx_shape),
        channel_roles=src.roles.as_dict(),
        run_id=run_id,
        params_hash=params.hash(),
    ) as writer:
        for payload in derivation.payloads():
            writer.add_cell(**payload)
        #: > ### ⇒ THE WHOLE-IMAGE NUCLEI, STORED BEFORE THEY ARE UPSAMPLED — 2026-09-02
        #: >
        #: > `member.labels` and not `nuclei_native`. The native version is the one this
        #: > function already built four lines above, and storing it would cost `downsample`
        #: > squared in plane — ~850 MB of int32 per production field instead of ~210 —
        #: > for an array `upsample_xy` restores exactly and cheaply on demand.
        #: >
        #: > **Batch review cannot draw a neighbour's nucleus without this.** Each cell's own
        #: > `nucleus_mask` is clipped to that cell's mask, so reassembling the volume from
        #: > the stored crops would give the right nuclei eroded at every cell boundary. See
        #: > `PoolWriter.write_nucleus_labels`.
        writer.write_nucleus_labels(member.labels, downsample=params.downsample)
        fragment = writer.fragment_path
        file_uid = writer.file_record.file_uid
    mark("write", t, f"{n_cells} cells written")

    return PipelineResult(
        source_path=str(source_path),
        file_uid=file_uid,
        fragment_path=fragment,
        n_cells=n_cells,
        stages=stages,
        timings=timings,
        derivation=derivation,
    )
