"""The semi-automatic application — what `prestomorph interactive` runs.

**This is the only module that knows the whole flow.** Every screen was built knowing nothing
about the ones on either side of it: it takes arrays through a `set_*` method and emits
`continue_requested`. This joins them, and it is the one place where the direction of §2's first
constraint is visible in code — **the UI calls the pipeline, and the pipeline never calls the
UI.**

> ### THE DERIVATION IS `segment_image`'s OWN, RUN WITHOUT WRITING
>
> §4a.5: the cell set comes from *"`PoolWriter` → fragment → merge, **exactly as batch does** —
> the same code, which is what the reproducibility gate requires. No second write path to
> diverge."* And its 2026-08-16 amendment moves the write to item 8, so steps 1-7 hold the label
> volume in memory.
>
> **Both halves are satisfied by `segment_image(..., pool_root=None)`**, which computes exactly
> what batch computes and returns a `Derivation` instead of writing it. Item 8 then streams
> `Derivation.payloads()` into `PoolWriter`. **There is no second implementation of segmentation
> anywhere in `ui/`**, which is what makes §6.1's *"an UNEDITED semi-auto run must replay
> identically in batch"* hold by construction.

> ### THE HEAVY STEPS GO THROUGH `Compute`, AND THAT IS WHAT MAKES THIS TESTABLE
>
> Cellpose takes 200-500 s and a real image is 3.17 GiB, so a test that drove the eleven screens
> through the real functions would be neither fast nor headless. `Compute` is a Protocol with one
> implementation that calls the pipeline and one that a test substitutes — the same arrangement
> `viewer.ViewerPort` uses, and for the same recorded reason: *"no test of which screen follows
> which should have to construct a Qt window or find a display."*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit, QStackedWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

from ..cache import DETECTION_FIELDS, NucleusCache, detection_key
from . import theme
from .semiauto import FLOW_ORDER, Decision, Flow, Session, Step
from .viewer import ViewerPort

__all__ = [
    "Compute", "PipelineCompute", "ReviewStep", "SemiAutoApp", "build_screen",
    "review_cells", "fit_to_display", "main", "AFTER_SIZE",
]

#: Every step that runs after item 6, and therefore every step that must see the cell size
#: cutoff applied. **Derived from `FLOW_ORDER` rather than listed**, so a screen inserted after
#: item 6 later cannot silently miss it — which is how item 6's box came to be inert in the
#: first place: a value with exactly one reader and nothing routing to it.
AFTER_SIZE: frozenset[Step] = frozenset(FLOW_ORDER[FLOW_ORDER.index(Step.SIZE) + 1:])

#: Item 7b's scene categories. **A cell you have ACCEPTED is erased rather than recoloured** —
#: `OWNERSHIP` maps `confirmed` to `hidden`, documented as *"impossible — it is not drawn"*, so
#: not drawing it is what makes its material untakeable by construction rather than by
#: discipline. Rejecting as debris moves a cell to `SCENE_DEBRIS`; rejecting as a real cell
#: leaves it a cell, because it is one. The user's rules, 2026-08-25.
SCENE_NOTHING = 0
SCENE_DEBRIS = 1
SCENE_CELL = 2


# --------------------------------------------------------------------------
# The pipeline seam
# --------------------------------------------------------------------------

@runtime_checkable
class Compute(Protocol):
    """What the flow needs from the pipeline, and nothing more.

    Every method here is something a screen in `COMPONENT_03_PLAN.md` §4 actually waits for.
    Anything that would pull cellpose or a 3 GiB read stays behind it.
    """

    def load(self, session: Session) -> None:
        """Read the image into `session.channels`, and set the shape and voxel size."""

    def filtered_nuclei(self, session: Session) -> np.ndarray:
        """The background screen's commit: DAPI with the broad-scale background subtracted.

        It comes **before** the veto, because the veto is the only step that writes zeros into
        DAPI and a Gaussian estimate rings around a hole. See `PipelineCompute.filtered_nuclei`.
        """

    def veto(self, session: Session) -> tuple[np.ndarray, np.ndarray]:
        """Item 2's commit: `(plaque mask, the veto as applied)`.

        The first goes downstream to detection, which dilates it itself; the second is what
        the screen draws. See `PipelineCompute.veto`.
        """

    def segmentation_mask(self, session: Session) -> np.ndarray:
        """Item 3's commit: the threshold-1 mask over the whole volume."""

    def measurement_mask(self, session: Session) -> np.ndarray:
        """Item 5's commit: the threshold-2 mask, at `mask_coefficient`. ~59 s."""

    def detect(self, session: Session) -> tuple[np.ndarray, np.ndarray]:
        """Item 4's commit — the 200-500 s one: `(nucleus labels, cell mask)`."""

    def derive(self, session: Session):
        """The final cell set, **not written**. Returns a `pipeline.Derivation`."""


@dataclass
class PipelineCompute:
    """The real one. **Every import is inside a method**, as `nuclei.py` does for cellpose.

    Nothing here is new segmentation code: each method calls the function batch calls, so the
    two routes cannot answer the same parameters differently.
    """

    #: **The last plaque threshold, kept so arrival and COMMIT do not each pay for it.**
    #: `(channel, coefficient, mask)`, compared by identity on the channel — see `veto`.
    _veto_threshold: tuple | None = field(default=None, repr=False)

    #: **Cellpose's answer, for the length of the session** — patch item A10. In memory only:
    #: nothing is written to `output_root`, so there is no folder, no cleanup and no lifetime
    #: question. See `cache.NucleusCache` and `_detection_key`.
    _nuclei: NucleusCache = field(default_factory=NucleusCache, repr=False)

    def _detection_params(self, session: Session):
        """The `SegmentationParams` this session's detection runs at.

        > ### ⇒ ONE SOURCE, AND FINDING OUT WHY COST A TEST — patch item A10, 2026-09-03
        >
        > `detect` read its detection values straight off `session.chosen`, while the key is
        > built from `session_params` — which reads `chosen` through a **display whitelist**.
        > `downsample` and `normalize` are `screen_save.HIDDEN_FIELDS`, so they are dropped
        > there and kept here, and `test_detect_returns_the_cell_mask_at_the_resolution_the_
        > labels_came_back_at` pins that `detect` honours `chosen["downsample"]`.
        >
        > **A key and a computation reading two different places is the exact shape of a cache
        > that serves the wrong answer**: item 4 would detect at one resolution, the key would
        > name another, and `derive` would hand `segment_image` labels at a resolution it is not
        > expecting. So this one object carries the session's choice for **every field in the
        > key**, and item 4, the key and item 6 all run at it.
        >
        > **It changes nothing today**, which is what makes it safe: no screen writes either
        > field, so the overrides are empty and this is `session_params` exactly.
        >
        > ⇒ **What is NOT fixed here, and is worth knowing:** the parameters *file* is still
        > written from plain `session_params`, so a session that somehow held a hidden field
        > would record a value it did not run at. That is a `screen_save` question, outside this
        > patch, and unreachable while nothing writes those fields.
        """
        from dataclasses import replace

        from .screen_save import session_params

        params = session_params(session)
        chosen = {name: session.chosen[name]
                  for name in DETECTION_FIELDS if name in session.chosen}
        return replace(params, **chosen) if chosen else params

    def _detection_key(self, session: Session, params=None) -> tuple:
        """The key both `detect` and `derive` look the detection up under.

        **One expression, called from both**, which is the whole defence against the trap
        `cache.detection_key` records: the two call sites reach `detect_nuclei` with different
        arguments that compute the same answer, so anything derived from the arguments would
        miss every time and miss silently.
        """
        return detection_key(params if params is not None else self._detection_params(session),
                             image=session.image_path,
                             voxel_size_um=session.voxel_size_um, roles=session.roles)

    def load(self, session: Session) -> None:
        from ..io import ImageSource

        #: **The voxel size item 1 collected is passed in, not ignored.** `ImageSource` refuses a
        #: file that records none unless it is given one — and item 1 asks for it *"and confirms
        #: it either way"* precisely because files can lack it. Reading without it made a
        #: perfectly usable image unopenable through the flow while `segment --params` opened it
        #: fine. Found by running one; §2's *"the one place the file can be wrong invisibly"* is
        #: the other half of the same rule.
        src = ImageSource(session.image_path, dict(session.roles),
                          expected_voxel_size_um=session.voxel_size_um)
        for role in session.roles:
            session.channels[role] = src.channel(role)
        session.image_shape = (src.meta.n_z, *src.meta.yx_shape)
        if session.voxel_size_um is None:
            session.voxel_size_um = tuple(float(v) for v in src.voxel_size_um)

    def filtered_nuclei(self, session: Session) -> np.ndarray:
        """The DAPI channel with the broad-scale background subtracted, at **native** resolution.

        > ### ONE SUBTRACTION, AND THE TWO ROUTES REACH IT DIFFERENTLY
        >
        > **Batch filters inside `detect_nuclei`**, from `dapi_background_scale_um` and
        > `dapi_background_strength` in the parameters file. **The interactive route filters
        > here, once, at the background screen's COMMIT**, stores the result on the session and
        > hands *that* to detection with a zero strength — so the subtraction lands exactly once
        > on either path and the two compute the same function of the same numbers.
        >
        > **The alternative was to let `detect_nuclei` filter on both routes** and keep no
        > product at all, which is what item 2's veto does — *"suppressing the channel ourselves
        > would be a second implementation."* It does not transfer: the veto is applied by one
        > call either way, while this would run the ~67 s filter a second time on top of item
        > 4's 200-500 s, for an array the session is already holding. Task 24 settled the same
        > trade the same way for item 3's committed mask.
        """
        from ..nuclei import subtract_background

        return subtract_background(
            session.channels["nuclei"],
            sigma_um=float(session.chosen.get("dapi_background_scale_um", 12.0)),
            voxel_size_um=session.voxel_size_um or (1.0, 1.0, 1.0),
            strength=float(session.chosen.get("dapi_background_strength", 0.0)))

    def veto(self, session: Session):
        """`(the raw threshold mask, the veto as applied)`.

        > ### BOTH, BECAUSE THEY HAVE DIFFERENT CONSUMERS — task 26
        >
        > **What goes downstream must be undilated.** `detect_nuclei` applies
        > `plaque_dilate_um` itself, so handing it a mask that has already been widened would
        > dilate twice.
        >
        > **What item 2 draws must be the dilated one**, or the outline, the removed-DAPI
        > overlay and the readout all sit still while the user moves the value the screen
        > exists to choose. This returned `(mask, vetoed DAPI)` and both call sites discarded
        > the second element, so the dilation reached nothing the user could see.

        **The threshold is memoised on `plaque_veto_coefficient` since A2**, so moving only the
        margin costs the dilation and not the ~17 s Otsu underneath it.
        """
        from ..nuclei import veto_mask_at_margin
        from ..threshold import threshold_channel

        plaque = session.channels["plaque"]
        coefficient = float(session.chosen.get("plaque_veto_coefficient", 1.0))
        #: ⇒ **THE THRESHOLD IS MEMOISED ON ITS COEFFICIENT — patch item A2, 2026-09-03**
        #:
        #: `prepare` computes the veto on arrival at item 2 and `run_commit` computes it again
        #: on COMMIT, and the Otsu threshold behind both is **~17 s on a real field** — paid
        #: twice for the same answer whenever only `plaque_dilate_um` moved, which is the
        #: common case on this screen. The margin below is cheap and is never cached.
        #:
        #: **One entry, and identity on the channel.** A session holds one plaque channel, so a
        #: second entry could only ever be the previous image's; `threshold_channel` is a pure
        #: function of `(channel, coefficient)` at a fixed `min_object_voxels`, and the mask is
        #: treated as immutable everywhere it goes — `veto_mask_at_margin` dilates into a new
        #: array and `detect_nuclei` re-applies its own margin rather than editing this one.
        held = self._veto_threshold
        if held is not None and held[0] is plaque and held[1] == coefficient:
            mask = held[2]
        else:
            mask = threshold_channel(
                plaque,
                coefficient=coefficient,
                min_object_voxels=50).mask
            self._veto_threshold = (plaque, coefficient, mask)
        applied = veto_mask_at_margin(
            mask,
            dilate_um=float(session.chosen.get("plaque_dilate_um", 0.0)),
            voxel_size_um=session.voxel_size_um)
        return mask, applied

    def segmentation_mask(self, session: Session) -> np.ndarray:
        from ..threshold import build_cell_mask

        return build_cell_mask(
            session.channels["cell"],
            coefficient=float(session.chosen.get("cell_coefficient", 0.18)),
            min_object_voxels=int(session.chosen.get("min_object_voxels", 20)),
            voxel_size_um=session.voxel_size_um,
            closing_radius_um=float(session.chosen.get("closing_radius_um", 0.8)),
        ).filled

    def measurement_mask(self, session: Session) -> np.ndarray:
        """Item 5's mask, **the same call `segment_image`'s two-mask branch makes.**

        `pipeline.py:601` builds it as `build_cell_mask(cell, coefficient=mask_coefficient,
        min_object_voxels=..., closing_radius_um=mask_closing_radius_um)` — so this is
        `segmentation_mask` at a different coefficient and closing radius, and writing it as its
        own method rather than a flag is what keeps the two field names visible at the call site.

        > ### TWO CHANGES FROM THE DEBRIS PATCH, 2026-08-24, AND BOTH ARE CONTRACT
        >
        > **`fill_holes=False`.** Hole filling served `membership_fractions`, which runs
        > against the *segmentation* mask and never against this one; closing the lumens here
        > inflated every exported volume by the lumen of each soma. The screen has to show the
        > unfilled mask because that is now the mask cells are made of.
        >
        > **`measurement_min_object_voxels`, not item 3's `min_object_voxels`.** This method
        > used to record sharing item 3's value as deliberate — *"item 5 has no box for it and
        > inventing one here would make the screen's mask differ from the one batch rebuilds."*
        > That reasoning held while item 3's mask decided the cell set. Item 5 now has its own
        > box, so the two masks no longer share a floor and this reads the field that box sets.
        """
        from ..threshold import build_cell_mask

        return build_cell_mask(
            session.channels["cell"],
            coefficient=float(session.chosen.get("mask_coefficient", 0.18)),
            min_object_voxels=int(
                session.chosen.get("measurement_min_object_voxels", 20)),
            voxel_size_um=session.voxel_size_um,
            closing_radius_um=float(session.chosen.get("mask_closing_radius_um", 0.8)),
            fill_holes=False,
        ).filled

    def detect(self, session: Session):
        """**The expensive one**, and the reason item 4 carries a progress line and a gate.

        > ### THIS WOULD HAVE RAISED ON ITS FIRST CALL — task 24
        >
        > It was never called from anywhere, and it had drifted: `detect_nuclei` takes
        > **`diameter_px`**, not `diameter_um`, so the old body was a `TypeError` waiting for a
        > caller — unexpected keyword *and* a missing required argument. It also returned the
        > `NucleiResult` where the caller wanted `.labels`. Dead code does not stay correct.
        >
        > **The veto is applied INSIDE `detect_nuclei`, not before it.** `pipeline.py:542` hands
        > it the raw DAPI plus `plaque_mask` and `plaque_dilate_um`; suppressing the channel
        > ourselves and passing no mask would be a second implementation of the veto, and the
        > two would diverge the first time either changed. Item 2's committed mask is what goes
        > in — which is also what makes item 2 *"sit outside the freely re-enterable range"*.

        **Everything is at segmentation resolution, exactly as the pipeline computes it.**
        `nuclei.labels` come back downsampled by `params.downsample`, and `pipeline.py:559`
        immediately downsamples the cell mask to match before taking membership. Returning a
        native-resolution mask beside downsampled labels would make item 4's membership
        fractions disagree with the ones `segment_image` will compute at item 8 — and would
        cost an 849 MB upsample of the label volume on a production field to boot.
        """
        from ..nuclei import detect_nuclei, diameter_in_pixels
        from ..resolution import downsample_mask_xy

        #: **Every detection value comes off this one object** — see `_detection_params`. It is
        #: also what the cache key below is built from, so the two cannot name different runs.
        params = self._detection_params(session)
        downsample = int(params.downsample)
        voxel = session.voxel_size_um or (1.0, 1.0, 1.0)
        #: > ### THE BACKGROUND SCREEN'S COMMITTED CHANNEL, NOT A SECOND FILTERING OF IT
        #: >
        #: > `subtract_background` at the default 12 µm scale is **~67 s on a full field**, and
        #: > the screen before item 2 has already paid it — its COMMIT stores the filtered
        #: > channel on the session. Re-filtering here would put that on top of this call's
        #: > 200-500 s for an array the session is already holding, and would open the gap the
        #: > commit gates exist to close: a channel refiltered from `chosen` can differ from the
        #: > one the user looked at and approved. **Item 3's committed mask is reused for the
        #: > same reason, decided at task 24.**
        #: >
        #: > `background_strength=0` is therefore not *the filter is off* — it is *the filter
        #: > has already run*. The fallback below covers the route with no commit behind it,
        #: > which the flow's gate forbids and the tests take.
        #: > ### ⇒ THE DETECTION IS CACHED FOR THE SESSION — patch item A10, 2026-09-03
        #: >
        #: > Keyed on the **parameters** `detect_nuclei` reads, never on the arrays it is handed
        #: > — see `cache.detection_key` for why that distinction is the whole item. Coming back
        #: > to item 3, changing `cell_coefficient` and re-committing here is then seconds
        #: > instead of 200-500 s, which is the *"steps 4-6 are freely re-enterable"* promise
        #: > `HANDOFF_WORKSTREAM_B.md` made and nothing implemented until now.
        #: >
        #: > **The DAPI is resolved inside the miss**, because `filtered_nuclei` is ~67 s on a
        #: > full field and a cache hit must not pay it.
        def _detect():
            dapi = session.products.get(Step.BACKGROUND)
            if dapi is None:
                dapi = self.filtered_nuclei(session)
            return detect_nuclei(
                dapi,
                background_strength=0.0,
                diameter_px=diameter_in_pixels(
                    float(params.nucleus_diameter_um), voxel, downsample),
                stitch_threshold=float(params.stitch_threshold),
                downsample=downsample,
                plaque_mask=session.products.get(Step.VETO),
                plaque_dilate_um=float(params.plaque_dilate_um),
                voxel_size_um=voxel,
                min_volume_voxels=int(params.nucleus_min_volume_voxels),
                normalize=params.normalize,
            )

        result = self._nuclei.get(self._detection_key(session, params), _detect)
        #: > ### ITEM 3's COMMITTED MASK, NOT A SECOND COMPUTATION OF IT
        #: >
        #: > `build_cell_mask` over the whole stack is **~59 s at 40 x 2304 x 2304**, and item 3
        #: > has already paid it — its COMMIT stores the `CellMask` on the session. Rebuilding it
        #: > here put that 59 s on top of item 4's 200-500 s for an array the session was already
        #: > holding, and opened the one gap `run_commit`'s item 5 branch exists to close: a
        #: > mask recomputed from `chosen` can answer differently from the mask the user looked
        #: > at and approved. **Item 5 reuses the commit; this now does the same thing.**
        #: >
        #: > The fallback stays for the route with no commit behind it — a driver that reaches
        #: > detection without item 3, which the flow's gate forbids and the tests do.
        segmentation = session.products.get(Step.THRESHOLD_1)
        segmentation = getattr(segmentation, "filled", segmentation)
        if segmentation is None:
            segmentation = self.segmentation_mask(session)
        return result.labels, downsample_mask_xy(segmentation, downsample)

    def derive(self, session: Session):
        """`segment_image` with **no pool**, and with everything the session already holds.

        > ### ⇒ IT RAN THE WHOLE PIPELINE AGAIN — patch item A10, 2026-09-03
        >
        > Arriving at item 6 re-opened the file, read `cell` **three times** plus `nuclei` and
        > `plaque`, rebuilt both threshold masks, and **re-ran Cellpose from scratch** — 200-500 s
        > that item 4's COMMIT had paid minutes earlier at the same detection parameters. Item 5
        > to item 6 was the entire pipeline, twice.
        >
        > Four products and the channels now go in as arguments, **each with `segment_image`'s
        > own fallback behind it**: a session missing any of them gets exactly what it got
        > before. The batch route passes none of them and is untouched, which is §6.1's
        > reproducibility gate — one implementation, and reuse that can only skip work.
        >
        > ### ⇒ AND THE CHANNELS ARE NO LONGER RELEASED FIRST
        >
        > That was §2's second row — *"save memory by holding FEWER channels, not narrower
        > ones"* — and it was right while `segment_image` read its own: for the length of the
        > call the session's four channels were **pure duplication**, 740 MB on a 40 × 1068 ×
        > 1084 field and four times that at production width, at the moment memory peaks.
        >
        > **Sharing them is not a second copy — it is one copy instead of two**, so the peak is
        > no higher than releasing and re-reading, and five disk reads and the `_ensure_loaded`
        > re-read afterwards go with it. The row's rule is honoured, not overturned: what it
        > forbids is holding two copies, which is what this used to do.
        """
        from ..pipeline import segment_image

        from .screen_save import session_params

        #: **The same object the key is built from and the same object `segment_image` runs
        #: at** — see `_detection_params`. `session_params` is imported above because the rest
        #: of this module reaches it that way; this call goes through the helper so item 4 and
        #: item 6 cannot run at different values.
        params = self._detection_params(session)
        measurement = session.products.get(Step.THRESHOLD_2)
        result = segment_image(
            session.image_path, None,
            roles=dict(session.roles),
            params=params,
            expected_voxel_size_um=session.voxel_size_um,
            channels=dict(session.channels) or None,
            #: Item 3's commit is a `CellMask`; `segment_image` takes either it or the array.
            cell_mask=session.products.get(Step.THRESHOLD_1),
            measurement_mask=getattr(measurement, "filled", measurement),
            #: **Undilated**, which is the contract this product already carries.
            plaque_mask=session.products.get(Step.VETO),
            #: `peek`, not `get`: a miss here means *let `segment_image` detect for itself*,
            #: which is the fallback, not a reason to run Cellpose twice.
            nuclei_result=self._nuclei.peek(self._detection_key(session, params)),
        )
        return result.derivation


# --------------------------------------------------------------------------
# The screens, in flow order
# --------------------------------------------------------------------------

def _soma_params(derivation) -> dict:
    """The five soma values a `Derivation` carries, as `set_cell` keyword arguments.

    **Empty when there is no derivation**, so `CellEditor` keeps its own defaults — the case
    `review --pool` and every screen test arrive in. Named here rather than inlined because item
    6.5 writes these onto the derivation and item 7b reads them off it, and the two ends of that
    should refer to one list.
    """
    if derivation is None:
        return {}
    return {"thickness_um": float(derivation.soma_thickness_um),
            "anchor_um": float(derivation.soma_anchor_um),
            "method": str(derivation.soma_method),
            "sheet_area_um2": float(derivation.soma_sheet_area_um2),
            "waist_um": float(derivation.soma_waist_um),
            "min_sheet_um3": float(derivation.soma_min_sheet_um3)}


def review_cells(derivation):
    """`(tiles, crops)` for item 7, from the derivation. **The uid is the label id.**

    Triage needs a thumbnail, two scores and the border flag per cell; rich review needs the
    masks. Both are already in `Derivation.payloads()`, which is the point of it being the one
    shape both routes use — item 7 does not invent a per-cell record and then have it disagree
    with what item 8 writes.

    > **The generator is consumed here and again at item 8, and that is the cheaper of the two
    > options.** Keeping every payload would hold ~60 cell crops beside the channels for the
    > rest of the session; re-running it costs a second pass of `find_soma` over ~60 small crops
    > at save time. §4a.5's *"the memory cost is one array, not sixty"* is the constraint that
    > decides it.
    """
    from .. import qc
    from ..soma import soma_for
    from .screen_triage import ReviewCell

    tiles, crops = [], {}
    for index, payload in enumerate(derivation.payloads(), start=1):
        uid = str(index)
        crops[uid] = payload
        #: **The soma is measured again here, and that is the price of storing no QC columns.**
        #: `payloads` measures it to write `soma_centroid`, but the *region* is what triage
        #: ranks on and no payload carries an array the pool will not store. `review --pool`
        #: pays the same cost for the same reason, which is what keeps the two routes identical
        #: — see `qc.triage_signals`.
        soma = soma_for(derivation, payload["mask"], payload.get("nucleus_mask"),
                        derivation.voxel_size_um)
        tiles.append(ReviewCell(
            uid=uid,
            thumbnail=payload.get("thumbnail"),
            touches_xy_border=bool(payload.get("touches_xy_border", False)),
            **qc.triage_signals(payload["mask"], payload.get("nucleus_mask"),
                                soma, derivation.voxel_size_um),
        ))
    return tiles, crops


class ReviewStep(QWidget):
    """Item 7 is **two screens**, and the flow sees one step. `COMPONENT_05_PLAN.md` §3a.

    Triage is the contact sheet and rich review is where a cell is actually resolved; a cell
    entered from the grid opens in the editor and returns to it resolved. **Neither screen knows
    the other exists** — triage emits `cell_entered`, the editor emits `resolved`, and this is
    the only place those two facts meet. Composed here rather than inside either screen so
    Component 5's route can reuse both against a pool without inheriting Component 3's flow.

    **`continue_requested` fires on `file_accepted`**, which is triage's *"this file is done"* —
    the same signal the batch route will use to move to the next file.
    """

    continue_requested = Signal()
    #: > ### BACK LEAVES REVIEW ALTOGETHER, FROM WHEREVER YOU ARE IN IT — 2026-08-30
    #: >
    #: > The user's rule: *"for item 7a regardless of which triage screen they are on, if they
    #: > press back then it takes the user back to the item 6.5 screen."* Triage pages within
    #: > itself and rich review is entered from it, and **none of that is a step** — the flow
    #: > sees one `REVIEW`, so back from anywhere inside it is back one step, to item 6.5.
    #: >
    #: > **Rich review has no Back of its own**, by the same decision. It is left by resolving a
    #: > cell, which returns to triage, and the Back is there.
    #: >
    #: > What that costs is stated where the user can read it, in the popup: every triage
    #: > decision and every rich-review edit goes, and item 7 starts again with every cell shown.
    back_requested = Signal()
    #: The inner screen changed, so the canvas may have to move. See `canvas_host`.
    canvas_moved = Signal()

    def __init__(self, viewer: ViewerPort | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        from .screen_edit_cell import EditCellScreen
        from .screen_triage import TriageScreen

        self.viewer = viewer
        self.cells: dict[str, dict] = {}
        #: > ### ⇒ B7 — THE NEIGHBOUR SOMATA, CACHED BY uid AND MASK STAMP
        #: >
        #: > `HANDOFF_PATCH_CPU.md` §4 B7, and it is **60% of a cell entry**. Headless, averaged
        #: > over five real cells: `_other_cells_around` **7.95 s (81.3%)**, `CellEditor` 1.83 s,
        #: > every other helper 0.00 s. It calls `soma_region` for every overlapping neighbour —
        #: > roughly 20 `find_soma` calls — on **every** cell entry, and `_rewiden` calls it
        #: > again on every margin move.
        #: >
        #: > **A neighbour's soma only changes if that neighbour is edited**, and
        #: > `editing.mask_stamp` is what notices. Keyed by uid *and* stamp rather than uid
        #: > alone, so the cache is self-checking: an edit changes the voxels, the stamp moves,
        #: > and the entry is simply missed rather than having to be found and invalidated.
        #: >
        #: > `None` is cached too — a neighbour with no soma is an answer, and recomputing it
        #: > costs the same `find_soma` as one with.
        self._soma_cache: dict[str, tuple[str, "np.ndarray | None"]] = {}
        #: **B6's seam, and `None` is the semi-automatic route.** A callable `(uid, stamp) ->
        #: CellEditor | None` set by the pool driver, which builds editors ahead on the triage
        #: grid. Answering `None` for any reason — nothing built yet, the reviewer outran the
        #: prefetch, or the mask has moved since — falls through to building one on arrival,
        #: which is what this screen has always done. See `_enter`.
        self.editor_source = None
        #: **What item 8 writes onto the pool**, keyed the same way the tiles are — see
        #: `screen_save.apply_decisions` for why an empty one makes the whole of Phase C empty.
        self.decisions: dict[str, Decision] = {}
        self.entered: str | None = None
        self.voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
        #: The whole measurement mask — cells and debris — sliced per cell in `_enter`.
        self.scene: np.ndarray | None = None
        #: **Full-volume, held by reference for the length of item 7.** These are the session's
        #: own channels and the derivation's own labels, not copies, so holding them costs
        #: nothing beyond the pointer — and `_enter` slices a crop per cell rather than the
        #: screen re-reading anything.
        self.iba1: np.ndarray | None = None
        self.dapi: np.ndarray | None = None
        self.nuclei_labels: np.ndarray | None = None
        #: Nuclei belonging to a cell that still exists. **Orphans stay out**: a nucleus whose
        #: cell fell under item 6's cutoff is now debris, and drawing its nucleus would imply a
        #: neighbouring cell that is not there. The user's decision, 2026-08-25.
        self.owned_nuclei: frozenset[int] = frozenset()
        #: **The derivation itself, by reference, so an edit has somewhere to land.**
        #: `_derive` caches it on the session, so this is the same object item 8 reads — which
        #: is what lets `_leave` paint into `labels` and item 8's `payloads()` re-derive every
        #: field of the edited cell from it. `None` on the pool-backed route, where there is no
        #: derivation and nothing to paint.
        self.derivation = None

        self.stack = QStackedWidget(self)
        self.triage = TriageScreen(self)
        self.editor = EditCellScreen(viewer, self)
        self.stack.addWidget(self.triage)
        self.stack.addWidget(self.editor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

        self.triage.cell_entered.connect(self._enter)
        #: Triage's Back is item 7's Back — see `back_requested` above.
        self.triage.back_requested.connect(self.back_requested)
        #: **The canvas has to follow the inner switch too.** Item 7 is two screens behind one
        #: step, and only one of them shows a picture — so the flow driver cannot place the
        #: canvas from `Step.REVIEW` alone. This is the seam that tells it the inner screen
        #: changed, and it is emitted after the switch so `canvas_host()` already answers with
        #: the new one.
        self.stack.currentChanged.connect(lambda _index: self.canvas_moved.emit())
        #: **Order matters, and Qt honours it.** The decisions must be recorded before
        #: `continue_requested` propagates, because item 8 copies them off this widget the moment
        #: the flow arrives there.
        self.triage.screen_accepted.connect(self._record_resolved)
        self.triage.file_accepted.connect(self._record_resolved)
        #: **The other direction, and it did not exist until B5.** `undo` restored the tiles and
        #: nothing removed what the accept had recorded — see `_undo_decisions`.
        self.triage.undone.connect(self._undo_decisions)
        #: **The `×` on a tile** — patch item B5. The popups are asked by the grid, because
        #: that is where the click happened; the decision is recorded here, because that is
        #: where every other decision is.
        self.triage.cell_rejected.connect(self._reject_from_grid)
        self.triage.file_accepted.connect(self.continue_requested)
        self.editor.resolved.connect(self._leave)
        #: **Widening is re-cutting.** The editor holds crops; the scene lives here, so it is
        #: here that a wider box can be taken from it. Without this the margin slider widens the
        #: context once at entry and is inert for the rest of the cell.
        self.editor.margin_changed.connect(self._rewiden)
        #: **Switching a deferred layer on is the other kind of re-cut.** Same seam, same
        #: reason: the screen holds crops and the scene lives here. See `_cut_context`.
        self.editor.context_wanted.connect(self._on_context_wanted)
        #: **A take is the one edit that changes a cell the reviewer is not looking at.** The
        #: donor's payload is what its own `this cell` layer is built from and what
        #: `_repaint_scene` marks its footprint with, so both go stale the moment material
        #: moves. This is where they are kept honest.
        self.editor.material_taken.connect(self._on_material_taken)
        #: **Answered here because only here knows.** The screen holds one cell's crops; the
        #: payloads that say how big a neighbour is belong to this step. Assigned rather than
        #: connected because the screen needs the ANSWER before it acts, and a signal returns
        #: nothing - the same shape `confirm_transfer` and `confirm_bridge` already take.
        self.editor.would_empty_the_donor = self.would_empty_a_donor

    @property
    def display_range(self) -> dict:
        """Forwarded to the inner screen. **`ReviewStep` shows no channel of its own** — rich
        review is what carries `Iba1 signal` and `raw DAPI`, and `prepare` hands the ranges to
        whichever screen is arriving without knowing that item 7 is two screens behind one step.
        """
        return self.editor.display_range

    @display_range.setter
    def display_range(self, value) -> None:
        self.editor.display_range = dict(value)

    def canvas_host(self):
        """The host of whichever inner screen is showing, or `None` for triage.

        **Triage genuinely has no canvas**, and that is not an omission: the contact sheet draws
        its own thumbnails as Qt tiles, because five across scrolling vertically is a grid of
        pictures rather than one picture. The canvas simply stays where it was until a cell is
        entered.
        """
        current = self.stack.currentWidget()
        return getattr(current, "canvas", None)

    def canvas_layout(self):
        """The arrangement the inner screen wants, or `None` when it shows no canvas."""
        current = self.stack.currentWidget()
        return getattr(current, "CANVAS", None)

    def bind_gestures(self) -> None:
        """Forward to rich review, which is the half of item 7 that has any.

        > ### EVERY GESTURE ON 7b WAS BOUND TO NOTHING - found by walking the screen, 2026-08-26
        >
        > `show_step` binds by asking **the step's screen** for `bind_gestures`, and item 7 is
        > *two screens behind one step*: the step's screen is this widget, and the gestures
        > live on `EditCellScreen` inside it. `getattr(screen, "bind_gestures", None)` found
        > nothing, so the call was skipped and **not one of the five gestures was ever wired**
        > in the running program - click, double-click, `u`, and SHIFT press/release alike.
        >
        > **Every handler was already correct and every one is directly tested**, which is
        > exactly why this survived: `test_ui_screen_edit_cell.py` calls `bind_gestures` on the
        > screen itself, where it works. Nothing tested the one seam that has to call it.
        > **It is the same shape as the tile that never left the grid** - a complete mechanism
        > with no one holding its end - and it is the third forwarding this class needs after
        > `canvas_host` and `display_range`, both of which were written and this one was not.
        >
        > **Bound once, on arrival at item 7, and it lasts the whole episode.** Only `show_step`
        > unbinds, and it does not run on the inner triage/editor switch. Binding while triage
        > is showing is harmless: every handler returns immediately while `CellEditor` is
        > `None`, which it is until a cell is entered.
        """
        self.editor.bind_gestures()

    def set_cells(self, review_cells, payloads: dict, path: str = "",
                  voxel_size_um=(1.0, 1.0, 1.0), scene=None, iba1=None, dapi=None,
                  nuclei_labels=None, owned_nuclei=None, derivation=None) -> None:
        """``scene`` is the whole measurement mask — every cell **and the debris**.

        > ### DEBRIS REACHES RICH REVIEW AS ITS OWN LAYER — and `everything else` is gone
        >
        > `COMPONENT_05_PLAN.md` §393 defined layer 4 as *"the file's threshold mask minus this
        > cell — the rest of the above-threshold Iba1, whoever owns it."* **That one layer
        > became two on 2026-08-25**: one flat yellow could not tell a neighbouring cell from
        > unclaimed material, so `other cells` and `debris` replaced it and answer separately.
        >
        > **`everything else` survived that split as dead work for seventeen days.** Its name
        > left `LAYERS`, so nothing could ever draw it, but `_enter` went on calling
        > `_scene_around` and passing the result to `set_cell` on every cell entry. Removed
        > 2026-09-11 — the keyword, the attribute and the method — as part of cutting the 7a ->
        > 7b transition down. Nothing on screen changes, because nothing on screen used it.
        """
        self.cells = dict(payloads)
        self.decisions = {}
        #: **Cleared with the cells, because uids repeat across files.** The pool route calls
        #: this once per file and the uids start again at `"1"`, so an entry left behind would
        #: be keyed to a cell in a file that is no longer open. The stamp would almost always
        #: miss — but *almost* is not the property this is allowed to have.
        self._soma_cache.clear()
        self.voxel_size_um = tuple(float(v) for v in voxel_size_um)
        self.scene = None if scene is None else np.asarray(scene)
        self.iba1 = None if iba1 is None else np.asarray(iba1)
        self.dapi = None if dapi is None else np.asarray(dapi)
        self.nuclei_labels = None if nuclei_labels is None else np.asarray(nuclei_labels)
        self.owned_nuclei = frozenset(int(n) for n in (owned_nuclei or ()))
        self.derivation = derivation
        self.triage.set_file(review_cells, path=path)
        self.stack.setCurrentWidget(self.triage)

    def _record_resolved(self) -> None:
        """**Triage's own `resolved` flags are the source of truth**, swept after every accept.

        `accept_this_screen` and `accept_rest_of_file` are *"a pure status flip"* — they set
        `resolved` on the cells they clear and emit their signal. **Accepting is what makes a cell
        measurable**: `PoolWriter` writes `CANDIDATE` and `measure` takes only `CONFIRMED`, so a
        reviewer who accepts three screens and then the file, and whose accepts were recorded
        nowhere, gets empty CSVs from the exact commands the SAVED popup hands them.

        **`setdefault`, so a rich-review rejection is never overwritten by a later sweep.** The
        editor decides a cell explicitly and triage's accept must not undo that.
        """
        for band in self.triage.screens:
            for cell in band:
                if cell.resolved:
                    self.decisions.setdefault(cell.uid, Decision("confirmed"))

    def _reject_from_grid(self, uid: str, resolution) -> None:
        """A cell rejected from the contact sheet, without being opened — patch item B5.

        The same three outcomes as rich review, recorded the same way and repainted into the
        scene the same way: a `debris` rejection stops being takeable, a `real_cell` rejection
        stays takeable, and neither is an accept. **The only thing missing beside `_leave` is
        `_persist_edits`, because a cell rejected from the grid was never edited.**
        """
        self.decisions[uid] = self._decision_from(resolution)
        reason = getattr(resolution, "reason", None)
        self._repaint_scene(uid, "rejected" if reason else "confirmed", reason)

    @staticmethod
    def _decision_from(resolution) -> Decision:
        """One `Resolution` as one `Decision`. **Both callers read the same names.**

        > The names are the trap this shares with `_leave`: `reject_reason` and `reject_note`
        > are the *storage* names — fields on `schema.CellRecord` — while what the editor emits
        > calls them **`reason`** and **`note`**. Reading the wrong pair recorded a debris
        > rejection as a confirmed cell, and it went unnoticed because every fixture carried the
        > same misunderstanding.
        """
        reason = getattr(resolution, "reason", None)
        note = getattr(resolution, "note", None) or None
        n_cells = int(getattr(resolution, "n_cells", 1) or 1)
        return Decision("rejected" if reason else "confirmed", reason, note, n_cells)

    def _undo_decisions(self, uids) -> None:
        """Take back what `undo` restored: **the decisions, and the scene with them.**

        > ### ⇒ THE UNDO WAS COSMETIC — patch item B5, 2026-09-03
        >
        > `TriageScreen.undo` set `resolved = False` and rebuilt the grid; `decisions` is
        > written here from those flags and **nothing removed it**. Accept a screen, press `u`,
        > and the twenty tiles came back while all twenty stayed recorded `confirmed` — which is
        > what item 8 then wrote. The gesture whose whole purpose is taking a decision back was
        > the one gesture that did not.
        """
        for uid in uids:
            self.decisions.pop(uid, None)
            self._restore_scene(uid)

    def _restore_scene(self, uid: str) -> None:
        """Put an undone cell's material back where a rejection took it from.

        **Exact, and it needs no snapshot.** A cell's own footprint is always `SCENE_CELL`
        before it is resolved, and a voxel has exactly one owner — so writing `SCENE_CELL` over
        the payload's own mask, inside the payload's own bounding box, cannot overwrite a
        neighbour's value. That is what makes the inverse of `_repaint_scene` a single line
        rather than a history.
        """
        if self.scene is None:
            return
        payload = self.cells.get(uid)
        if payload is None:
            return
        z0, z1, y0, y1, x0, x1 = payload["bbox"]
        mask = np.asarray(payload["mask"]).astype(bool)
        self.scene[z0:z1, y0:y1, x0:x1][mask] = SCENE_CELL

    def accept_all(self) -> None:
        """Confirm every cell that has no decision yet — triage's *accept the rest*, from the
        outside. Kept as a named entry point because the flow driver and a future pool-backed
        route both need one that does not depend on triage's internal state."""
        for uid in self.cells:
            self.decisions.setdefault(uid, Decision("confirmed"))


    def editor_inputs(self, uid: str):
        """The three arrays and the settings `CellEditor` is built from, plus their stamp.

        > ### ⇒ ONE CONSTRUCTION, TWO CALLERS — `HANDOFF_PATCH_CPU.md` §4 B6 and B5c
        >
        > `_enter` builds a `CellEditor` on arrival; the pool route's prefetcher builds the same
        > one ahead of time, on a worker. A second expression of *what a `CellEditor` is made
        > of* is exactly how a prefetched editor and an entered one would come to differ, so
        > there is one, here, and it is not Qt — a worker thread may call it.
        >
        > **The mask is snapshotted, and the stamp is taken from the snapshot.** A worker can be
        > building cell 9 while the reviewer's transfer rewrites cell 9's mask on the Qt thread.
        > Stamping the copy the editor was actually built from is what makes that safe: if the
        > read was torn, or merely stale, the stamp will not match the mask on entry and B5c
        > rebuilds. A stamp taken from the live array afterwards could agree with a shape the
        > editor does not have.

        **The Iba1 CHANNEL, not the cell mask.** `set_cell`'s first argument is `intensity`,
        and until 2026-08-25 the call passed `payload["mask"].astype(float)` — so the layer
        named `Iba1 signal` drew the binary silhouette of the very cell being reviewed. The
        judgement the screen exists for is whether adjacent material belongs to this cell, and
        a silhouette is precisely the picture that cannot answer it.

        **The editable frame, not the bounding box** — see `editable_box`. The intensity is
        re-cut from the whole volume; the two masks are placed into it, because neither has
        anything outside the bounding box to re-cut.
        """
        from . import editing

        payload = self.cells.get(uid)
        if payload is None:
            return None
        mask = np.array(np.asarray(payload["mask"]).astype(bool), copy=True)
        stamp = editing.mask_stamp(mask, payload.get("bbox"))
        snapshot = dict(payload)
        snapshot["mask"] = mask
        intensity = (self._context_crop(self.iba1, snapshot, self._editable_margin())
                     if self.scene is not None
                     else self._crop(self.iba1, snapshot, mask.astype(float)))
        cell = (self._in_editable_frame(mask, snapshot) if self.scene is not None else mask)
        nucleus = (self._in_editable_frame(snapshot.get("nucleus_mask"), snapshot)
                   if self.scene is not None else snapshot.get("nucleus_mask"))
        return intensity, cell, nucleus, _soma_params(self.derivation), stamp

    def build_editor(self, uid: str):
        """``(stamp, CellEditor)`` for one cell. **No Qt; safe on a worker thread.**

        Measured at **1.83 s** per cell. Returns `None` when the cell is gone.
        """
        from . import editing

        made = self.editor_inputs(uid)
        if made is None:
            return None
        intensity, cell, nucleus, soma_params, stamp = made
        return stamp, editing.CellEditor(np.asarray(intensity), cell, nucleus,
                                         voxel_size_um=self.voxel_size_um, **soma_params)

    def _enter(self, uid: str) -> None:
        """**Rich review always resolves the cell** — the user's own restatement, §7 of the merge
        handoff. There is no way back to the grid that leaves the cell undecided."""
        payload = self.cells.get(uid)
        if payload is None:
            return
        self.entered = uid
        #: The context layers are cut wider than the cell by the margin the screen is showing,
        #: which is the first time that control has changed anything at all.
        margin = float(getattr(self.editor, "margin_um", 0.0) or 0.0)
        #: The three arrays come from `editor_inputs`, which is also what the pool route's
        #: prefetcher builds from — see there for why the Iba1 CHANNEL rather than the cell
        #: mask, and why the editable frame rather than the bounding box.

        #: > ### ⇒ B5c — A PREFETCHED EDITOR IS USED ONLY IF ITS MASK STILL MATCHES
        #: >
        #: > `HANDOFF_PATCH_CPU.md` §4 B5c, and it is **not optional**. Transfers move material
        #: > *between* neighbours — `apply_cell_edit`'s own note: *"a voxel carries one label,
        #: > so painting a neighbour's process into this cell removes it from the neighbour by
        #: > construction."* A stale prefetched editor would open showing the pre-transfer
        #: > shape, **silently**.
        #: >
        #: > Stamp-and-rebuild was chosen over invalidating a computed list of affected
        #: > neighbours precisely because it is self-checking: the question asked here is *was
        #: > this built from the mask this cell has now*, and there is no list to get wrong.
        #: >
        #: > **B3 lives in the same line.** `editor_source` answering `None` — nothing
        #: > prefetched, the reviewer outran it, or the stamp moved — falls through to
        #: > `set_cell` building one, which is exactly today's blocking behaviour. No new
        #: > progress UI.
        intensity, cell_mask, nucleus, _soma, stamp = self.editor_inputs(uid)
        prefetched = None if self.editor_source is None else self.editor_source(uid, stamp)
        #: **Only the context layers that are switched on are cut.** See `_cut_context`.
        context = self._cut_context(payload, margin)
        self.editor.set_cell(intensity, cell_mask, nucleus,
                             voxel_size_um=self.voxel_size_um,
                             editor=prefetched,
                             #: **Item 6.5's four, off the derivation itself** — the same object
                             #: `payloads()` measures the soma from, so the white region here and
                             #: the exported `soma_volume_um3` cannot disagree. §3b, closed.
                             soma_params=_soma_params(self.derivation),
                             iba1_context=self._context_crop(self.iba1, payload, margin),
                             dapi=(self._context_crop(self.dapi, payload, margin)
                                   if self.scene is not None
                                   else self._crop(self.dapi, payload)),
                             other_nuclei=context["other_nuclei"],
                             other_cells=context["other_cells"],
                             debris=context["debris"],
                             context_origin=self._context_origin(payload, margin))
        self.stack.setCurrentWidget(self.editor)

    def _rewiden(self, margin_um: float) -> None:
        """Re-cut the entered cell's context layers at a new margin.

        **Not `set_cell`** — that rebuilds `CellEditor` and would discard every edit made so far.
        Moving a view control must never cost the reviewer their work, so only the context
        arrays are replaced and the editor is left alone. The screen draws immediately after.

        > ### IT RE-CUT TWO LAYERS AND THE OTHER THREE STAYED AT THE BOUNDING BOX - 2026-08-26
        >
        > `other cells` and `debris` grew with the margin while `Iba1 signal`, `other nuclei`
        > and `DAPI as detected` did not, **so pulling the margin out grew the neighbours into
        > a black field**: the channel they are cut from ended at the old boundary. Reported by
        > the user as the margin *"only expanding other cells and debris"*, which is exactly
        > what it was doing.
        >
        > **`self.iba1` is not re-cut, and that distinction is what makes this safe.**
        > `CellEditor` decomposes it, so the editing frame must not move under the reviewer's
        > edits. `iba1_context` is the same channel cut wider and it feeds the layer alone.
        """
        payload = self.cells.get(self.entered or "")
        if payload is None or self.scene is None:
            return
        context = self._cut_context(payload, margin_um)
        self.editor.set_context(
            other_cells=context["other_cells"],
            debris=context["debris"],
            context_origin=self._context_origin(payload, margin_um),
            iba1_context=self._context_crop(self.iba1, payload, margin_um),
            dapi=self._context_crop(self.dapi, payload, margin_um),
            other_nuclei=context["other_nuclei"])

    def _label_of(self, uid: str) -> int | None:
        """The label id wearing this cell's original footprint.

        **Read from the volume rather than parsed from the uid.** `review_cells` numbers tiles
        by position in `payloads()`, and `payloads()` skips a label whose bounding box came back
        empty — so position and label can diverge, and a paint keyed on the wrong one would
        rewrite another cell. Taking it from the array cannot be wrong.
        """
        payload = self.cells.get(uid)
        if payload is None or self.derivation is None:
            return None
        z0, z1, y0, y1, x0, x1 = payload["bbox"]
        under = self.derivation.labels[z0:z1, y0:y1, x0:x1][np.asarray(payload["mask"]).astype(bool)]
        under = under[under > 0]
        return int(under[0]) if under.size else None

    def _donor_of(self, taken_image_mask):
        """Which cell those voxels belonged to. **At most one** — a voxel has one owner."""
        for uid, payload in self.cells.items():
            if uid == self.entered:
                continue
            z0, z1, y0, y1, x0, x1 = payload["bbox"]
            overlap = taken_image_mask[z0:z1, y0:y1, x0:x1] & np.asarray(payload["mask"])
            if overlap.any():
                return uid, overlap
        return None, None

    def _taken_in_image_frame(self, material):
        """One take, moved from the editable frame into whole-image coordinates."""
        payload = self.cells.get(self.entered or "")
        if payload is None or self.derivation is None:
            return None
        z0, z1, y0, y1, x0, x1 = self.editable_box(payload)
        out = np.zeros(self.derivation.labels.shape, bool)
        out[z0:z1, y0:y1, x0:x1] = np.asarray(material).astype(bool)
        return out

    def _on_material_taken(self, status: str, material) -> None:
        """A take landed, so the cell it came from stops claiming it — **immediately.**

        > ### THE DONOR WOULD HAVE RECLAIMED IT — the defect this exists to prevent
        >
        > `ReviewStep.cells` is a snapshot of `payloads()` taken once on arrival at item 7 and
        > held for the whole step. Nothing re-derives it. So without this, a neighbour whose
        > process had been taken would still be holding its original mask, and:
        > >
        > > * entering it would draw the taken process in its `this cell` layer, and
        > > * finalising it would paint that mask back over the label volume, **silently giving
        > >   the process back and undoing the transfer**, and
        > > * `_repaint_scene` would mark its old footprint, erasing the taker's new material
        > >   from the scene every later cell is judged against.
        >
        > Only a transfer needs this. An add takes debris, which no payload claims, and a remove
        > touches nothing but the entered cell.
        """
        if status != "candidate" or self.derivation is None:
            return
        taken = self._taken_in_image_frame(material)
        if taken is None:
            return
        uid, overlap = self._donor_of(taken)
        if uid is None:
            return
        #: In place on the payload's own array, which is a fresh crop from `payloads()` rather
        #: than a view of the label volume — so this cannot reach `labels` behind `_leave`'s back.
        np.asarray(self.cells[uid]["mask"])[overlap] = False

    def would_empty_a_donor(self, material) -> bool:
        """Whether a take would leave the cell it comes from with nothing at all.

        **A transfer may not empty its donor**, and the refusal is the loud one. Taking the
        entirety of another cell is a merge, which §3d.7's gesture does not offer — and a cell
        that drops to zero voxels vanishes from `payloads()`, which would slide every decision
        after it onto the wrong cell, because `apply_decisions` joins by position.

        Reachable only for a neighbour with no soma: `_neighbour_piece` offers processes alone,
        and `NO_SOMA_NOTE` records that a cell whose erosion consumed it reads as one process.
        """
        taken = self._taken_in_image_frame(material)
        if taken is None:
            return False
        uid, overlap = self._donor_of(taken)
        if uid is None:
            return False
        return int(overlap.sum()) >= int(np.asarray(self.cells[uid]["mask"]).sum())

    def _persist_edits(self, uid: str) -> dict | None:
        """Fold this cell's edits back into the derivation. **The one durable act of item 7b.**

        > ### EVERY EDIT WAS DISCARDED, AND THIS IS THE LINE THAT WAS MISSING — 2026-08-26
        >
        > `_leave` recorded `(status, reason, note)` and stopped. The edited mask lived on
        > `CellEditor`; `session.cells = derivation.payloads()` then rebuilt every cell from a
        > label volume nobody had touched. **So adds, removes and transfers were all thrown
        > away** and the pool received the Phase A cell with `human_edited` false.
        >
        > **At finalisation, once, rather than on every edit.** The episode's own `u` unwinds
        > everything until the cell resolves, so painting per edit would write a state the
        > reviewer can still take back — and §9.4b already asks for one transaction per episode.
        > It also makes the paint the same shape as item 6's cutoff: monotonic, and irreversible
        > once done, which is what the two screens that change a cell now have in common.

        Everything else follows from the paint: `payloads()` re-derives the bbox, the thumbnail,
        the soma, the nucleus assignment, the voxel counts and the border flags from the volume
        it just wrote, so no field can disagree with another.
        """
        payload = self.cells.get(uid)
        if payload is None or self.derivation is None or self.editor.editor is None:
            return None
        label = self._label_of(uid)
        if label is None:
            return None
        edited = np.asarray(self.editor.editor.mask).astype(bool)
        #: **Only when it actually changed.** An accepted cell nobody touched must not be
        #: stamped `human_edited`, and re-painting an identical mask would still add it to the
        #: set. The comparison is against what the volume currently says, not against the
        #: payload, because a transfer may already have moved voxels off this cell.
        z0, z1, y0, y1, x0, x1 = self.editable_box(payload)
        if np.array_equal(self.derivation.labels[z0:z1, y0:y1, x0:x1] == label, edited):
            return None
        from ..pipeline import apply_cell_edit

        moved = apply_cell_edit(self.derivation, label,
                                (z0, z1, y0, y1, x0, x1), edited)
        #: The payload follows the volume, so anything still reading it — `_repaint_scene` on
        #: the very next line, and `_other_cells_around` for every cell after this one — sees
        #: the cell that was actually written.
        bz0, bz1, by0, by1, bx0, bx1 = payload["bbox"]
        payload["mask"] = (self.derivation.labels[bz0:bz1, by0:by1, bx0:bx1] == label)
        return moved

    def _repaint_scene(self, uid: str, status: str, reason) -> None:
        """Fold a resolved cell into the scene, **live, for every cell reviewed after it.**

        Three outcomes, and they are the user's rules of 2026-08-25 rather than three shades of
        the same thing:

        * **confirmed** -> erased. `OWNERSHIP` maps it to `hidden`, documented as *"impossible —
          it is not drawn"*, so erasing it is what makes an accepted cell's material untakeable
          **by construction**. Its territory is settled and cannot be given to a later cell.
        * **rejected as debris** -> `SCENE_DEBRIS`. It stops being a cell because you said so.
        * **rejected as a real cell** -> unchanged. It is still a cell; you simply had no time
          for it, and `OWNERSHIP` keeps it takeable behind the transfer popup.

        In place on the array already held, so this costs one boolean write over one crop.
        """
        if self.scene is None:
            return
        payload = self.cells.get(uid)
        if payload is None:
            return
        reason = getattr(reason, "value", reason)
        if status == "rejected" and reason == "real_cell":
            return
        z0, z1, y0, y1, x0, x1 = payload["bbox"]
        mask = np.asarray(payload["mask"]).astype(bool)
        region = self.scene[z0:z1, y0:y1, x0:x1]
        region[mask] = SCENE_DEBRIS if status == "rejected" else SCENE_NOTHING

    def _crop(self, volume, payload: dict, fallback=None):
        """One full-volume array cut to this cell's bounding box, or the fallback when absent."""
        if volume is None:
            return fallback
        z0, z1, y0, y1, x0, x1 = payload["bbox"]
        return volume[z0:z1, y0:y1, x0:x1]

    def _context_crop(self, volume, payload: dict, margin_um: float = 0.0):
        """One full-volume array cut at the **context** box - the bbox grown by the margin.

        The counterpart of `_crop`, which cuts at the bounding box itself. Both exist because
        the screen draws in two frames: the cell's own, which the editor works in, and the
        wider one the context layers are registered into by `translate`.
        """
        if volume is None or self.scene is None:
            return None
        z0, z1, y0, y1, x0, x1 = self.context_box(payload, margin_um)
        return volume[z0:z1, y0:y1, x0:x1]

    def _other_nuclei(self, payload: dict, margin_um: float = 0.0):
        """The **neighbours'** microglial nuclei in this crop — violet, and never this cell's.

        Three exclusions, each asked for: this cell's own nuclei, which have their own blue
        layer; nuclei belonging to no surviving cell, which would draw a neighbour that is not
        there; and everything outside the crop - which is now the **context** box rather than
        the bounding box, so a neighbour's nucleus just outside the cell appears when the margin
        is pulled out. Before 2026-08-26 it could not: this layer stopped at the bbox while
        `other cells` grew past it, so a neighbour arrived with no nucleus in it.
        """
        if self.nuclei_labels is None:
            return None
        crop = (self._context_crop(self.nuclei_labels, payload, margin_um)
                if self.scene is not None else self._crop(self.nuclei_labels, payload))
        mine = {int(n) for n in (payload.get("nucleus_ids") or ())}
        keep = self.owned_nuclei - mine
        return np.isin(crop, list(keep)) if keep else np.zeros(crop.shape, bool)

    def context_box(self, payload: dict, margin_um: float = 0.0):
        """This cell's bbox grown by `margin_um`, **clipped to the whole volume**.

        > ### THE MARGIN CONTROL HAS NEVER DONE ANYTHING, AND THIS IS WHY
        >
        > `screen_edit_cell.draw()` calls `crop_box(self.editor.mask, margin, ...)`, and
        > `crop_box` grows the bbox and then clips to **the array it was handed** — which is
        > already that same bbox. So the margin could not widen anything, and debris or a
        > neighbour sitting *beside* a cell was never drawn. Recorded as parked in
        > `SHAKEOUT_FINDINGS.md`; this is the half of it the context layers need.
        >
        > Growing here instead works because `self.scene` is the **whole volume**, so there is
        > something outside the bbox to clip against. The offset goes to the layer as
        > `translate`, so the wider layers stay registered against the four that are not.
        """
        z0, z1, y0, y1, x0, x1 = payload["bbox"]
        pad = [int(round(float(margin_um) / float(v))) if float(v) > 0 else 0
               for v in self.voxel_size_um]
        shape = self.scene.shape
        return (max(0, z0 - pad[0]), min(shape[0], z1 + pad[0]),
                max(0, y0 - pad[1]), min(shape[1], y1 + pad[1]),
                max(0, x0 - pad[2]), min(shape[2], x1 + pad[2]))

    def editable_box(self, payload: dict):
        """The frame rich review can EDIT in: the bounding box grown by the margin's **maximum**.

        > ### EVERYTHING YOU CAN EVER SEE IS TAKEABLE, AND THAT IS THE WHOLE RULE
        >
        > `CellEditor` used to work in the cell's own bounding box while the context layers grew
        > past it, so a piece you could **see** could lie outside the piece you could **take** —
        > and `_into_cell_frame` declined it, silently. Widening the editable frame to the
        > slider's maximum makes the refusal unreachable by construction rather than by rule:
        > the context box can never exceed a box padded by a margin it can never exceed.
        >
        > **Cut once, at entry, and never again.** The margin is a view control and §3b's rule
        > is that moving one must never cost the reviewer their work — re-basing the editable
        > frame mid-episode would mean remapping every edit already recorded in it. Pinning it
        > to the maximum instead of the current value is what buys that.
        >
        > **⇒ THE COST WAS NOT MEASURED.** `MARGIN_RANGE` tops out at 20 µm, which on crop2's
        > (0.6, 0.1083, 0.1083) voxels is ±33 z and ±185 xy — on a 2304² field a ~90² bounding
        > box becomes ~460², so `decompose_cell` runs over roughly 18M voxels per cell entered
        > rather than a few hundred thousand. The user's decision of 2026-08-26 was to build at
        > 20 and measure only if it bites. **If entering a cell in rich review becomes slow, or
        > memory climbs, time `decompose_cell` at 6, 10, 15 and 20 µm on a real field before
        > changing anything else** — this is the first thing to suspect.
        """
        return self.context_box(payload, self._editable_margin())

    @staticmethod
    def _editable_margin() -> float:
        """The margin the editable frame is cut at: **the slider's maximum**, not its value."""
        from .screen_edit_cell import MARGIN_RANGE

        return float(MARGIN_RANGE[1])

    def _in_editable_frame(self, array, payload: dict):
        """A bounding-box-shaped array placed into the editable frame, zero-padded around it.

        The cell's own mask and its nucleus exist only inside the bounding box — there is
        nothing of them outside it — so widening them is a placement rather than a re-cut. The
        intensity is cut from the whole volume instead, because there *is* signal out there.
        """
        if array is None:
            return None
        array = np.asarray(array)
        z0, z1, y0, y1, x0, x1 = self.editable_box(payload)
        bz0, bz1, by0, by1, bx0, bx1 = payload["bbox"]
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), array.dtype)
        out[bz0 - z0:bz1 - z0, by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = array
        return out

    def _context_origin(self, payload: dict, margin_um: float = 0.0):
        """Where the widened context sits relative to the cell's own crop, in voxels.

        Negative or zero on every axis, because the context box only ever grows outward. The
        screen hands it to the viewer as `translate`, which is what keeps the wide layers
        registered against the four cut to the bounding box.
        """
        if self.scene is None:
            return None
        z0, _, y0, _, x0, _ = self.context_box(payload, margin_um)
        #: **From the EDITABLE frame's corner, not the bounding box's.** The editable frame is
        #: the widest box on the screen since 2026-08-26, so this offset is now zero or positive
        #: where it used to be zero or negative. `_into_cell_frame` and `_to_context` both take
        #: either sign; `_world_offset` turns it back into a translate against the drawn crop.
        ez0, _, ey0, _, ex0, _ = self.editable_box(payload)
        return (float(z0 - ez0), float(y0 - ey0), float(x0 - ex0))

    def _without_this_cell(self, payload: dict, box):
        """`scene[box]` with the entered cell's own voxels cleared, whatever the box's size."""
        z0, z1, y0, y1, x0, x1 = box
        crop = self.scene[z0:z1, y0:y1, x0:x1].copy()
        bz0, bz1, by0, by1, bx0, bx1 = payload["bbox"]
        mask = np.asarray(payload["mask"]).astype(bool)
        crop[bz0 - z0:bz1 - z0, by0 - y0:by1 - y0, bx0 - x0:bx1 - x0][mask] = SCENE_NOTHING
        return crop

    def _cut_context(self, payload: dict, margin_um: float):
        """The three deferred context arrays, **cutting only the layers that are switched on.**

        > ### ⇒ THE THREE THAT ARRIVE OFF ARE NO LONGER BUILT — 2026-09-11
        >
        > `other nuclei`, `other cells` and `debris` arrive off (`screen_edit_cell.LAYERS`) and
        > used to be cut, handed over and uploaded to the GPU on every entry anyway, to be
        > hidden the moment they landed. `other cells` is the expensive one: it is what the
        > neighbour somata are computed for, and it was **the single largest item in the
        > compute half of an entry**.
        >
        > **The toggle is the question, not the default.** A reviewer who switches `debris` on
        > and moves to the next cell still wants it, so this reads the live checkbox rather
        > than `LAYERS` — which is also what makes the first switch-on work: by the time
        > `context_wanted` reaches `_on_context_wanted` the box is already ticked.
        """
        on = self.editor.toggles.is_on
        neighbours = on("other cells") or on("debris")
        #: `other cells` and `debris` are two readings of one crop — see `_neighbourhood`.
        neighbourhood = self._neighbourhood(payload, margin_um) if neighbours else None
        return {
            "other_nuclei": (self._other_nuclei(payload, margin_um)
                             if on("other nuclei") else None),
            "other_cells": (self._other_cells_around(payload, margin_um,
                                                     neighbourhood=neighbourhood)
                            if on("other cells") else None),
            "debris": (self._debris_around(payload, margin_um, neighbourhood=neighbourhood)
                       if on("debris") else None),
        }

    def _on_context_wanted(self, name: str) -> None:
        """A deferred layer was switched on for the first time. **Cut it, here.**

        The screen holds crops; the scene lives here, which is the same division
        `margin_changed` and `_rewiden` already rest on. The screen draws immediately after
        this returns — the signal is a direct connection, so `set_context` has landed by then.
        """
        payload = self.cells.get(self.entered or "")
        if payload is None or self.scene is None:
            return
        margin = float(getattr(self.editor, "margin_um", 0.0) or 0.0)
        context = self._cut_context(payload, margin)
        self.editor.set_context(
            other_cells=context["other_cells"], debris=context["debris"],
            context_origin=self._context_origin(payload, margin),
            other_nuclei=context["other_nuclei"])

    def _neighbourhood(self, payload: dict, margin_um: float = 0.0):
        """`scene` around this cell with its own voxels cleared. **Cut once, read twice.**

        `other cells` and `debris` are two readings of one crop — `== SCENE_CELL` and
        `== SCENE_DEBRIS` — and each used to cut its own copy of the scene box. Both callers
        take it from here instead, so `_enter` and `_rewiden` copy the neighbourhood once.
        """
        if self.scene is None:
            return None
        return self._without_this_cell(payload, self.context_box(payload, margin_um))

    def _debris_around(self, payload: dict, margin_um: float = 0.0, *, neighbourhood=None):
        """Unclaimed material near this cell. Yellow, flat — debris has no parts."""
        if self.scene is None:
            return None
        if neighbourhood is None:
            neighbourhood = self._neighbourhood(payload, margin_um)
        return neighbourhood == SCENE_DEBRIS

    def _other_cells_around(self, payload: dict, margin_um: float = 0.0, *,
                            neighbourhood=None):
        """The neighbours in this cell's box, as **1 soma / 2 process** for the two treatments.

        The soma comes from `editing.soma_region`, the same opening the export measures, so what
        reads as a neighbour's cell body here is the body every other screen and every column
        agrees on. One erosion per visible neighbour, on crops already held.
        """
        if self.scene is None:
            return None
        z0, z1, y0, y1, x0, x1 = self.context_box(payload, margin_um)
        if neighbourhood is None:
            neighbourhood = self._without_this_cell(payload, (z0, z1, y0, y1, x0, x1))
        neighbours = neighbourhood == SCENE_CELL
        out = np.zeros(neighbours.shape, np.uint8)
        if not neighbours.any():
            return out
        out[neighbours] = 2
        from . import editing

        #: Every OTHER cell's mask, cropped to this box, so the somata can be found without
        #: re-reading anything — the payloads are already held for the whole of item 7.
        for uid, other in self.cells.items():
            if uid == self.entered:
                continue
            oz0, oz1, oy0, oy1, ox0, ox1 = other["bbox"]
            if oz1 <= z0 or oz0 >= z1 or oy1 <= y0 or oy0 >= y1 or ox1 <= x0 or ox0 >= x1:
                continue
            soma = self._neighbour_soma(uid, other)
            if soma is None or not soma.any():
                continue
            #: Intersect the neighbour's own crop with this cell's box, in image coordinates.
            zs, ze = max(z0, oz0), min(z1, oz1)
            ys, ye = max(y0, oy0), min(y1, oy1)
            xs, xe = max(x0, ox0), min(x1, ox1)
            here = out[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0]
            there = soma[zs - oz0:ze - oz0, ys - oy0:ye - oy0, xs - ox0:xe - ox0]
            here[there & (here == 2)] = 1
        return out

    def _neighbour_soma(self, uid: str, payload: dict):
        """One neighbour's soma, computed once per version of that neighbour's mask.

        **B7, and it is the largest single cost in a cell entry.** See `_soma_cache` in
        `__init__` for the measurement and for why the key carries a stamp rather than only a
        uid.

        The stamp is taken from the mask *and* the bbox, so a neighbour that was edited, split,
        or moved is a different key and recomputes. Nothing invalidates anything: a stale entry
        is unreachable rather than stale.

        > ### ⇒ THE PARAMETERS WERE NOT PASSED, AND THIS WAS THE LAST SITE — closed 2026-09-11
        >
        > `soma_region`'s own docstring records the plumbing gap being closed on 2026-08-30,
        > *"Every parameter is now passed from `Derivation`"*. **This call site was missed.**
        > It took `find_soma`'s built-in defaults — `method="anchor_sheet"` (`soma.py:107`) and
        > `thickness_um=0.9` (`soma.py:145`) — while the entered cell's own soma, two lines
        > away in `_enter`, was handed `_soma_params(self.derivation)`. So the green *other
        > cells* layer drew every neighbour's body **by a different rule than the export
        > measures and than this cell's own white soma**, which is precisely the §3b violation
        > the 2026-08-30 pass existed to end.
        >
        > **What it changes on screen.** Measured on this study's own pool (`opening` at
        > 3.5 µm against `anchor_sheet` at 0.9 µm), across 14 neighbours: the corrected soma
        > is a **strict subset** of the old one every time — not one voxel is gained — and it
        > is between 0.1% and 65% smaller. Voxels that used to render as the neighbour's pale
        > body now render as its saturated process, which also makes them **takeable**, since
        > `_context_status` offers value 2 and refuses value 1. That is the correct behaviour
        > arriving, not new behaviour: what is takeable is now what the export calls a process.
        >
        > **And it is the largest single cost in a cell entry.** `anchor_sheet` is the
        > expensive branch — per-plane hole filling, two distance transforms, three-axis plane
        > area scans and waist closing on a padded array — against `opening`'s single
        > morphological opening. Measured 2-3x per call on the same neighbours.
        """
        from . import editing

        mask = np.asarray(payload["mask"]).astype(bool)
        stamp = editing.mask_stamp(mask, payload.get("bbox"))
        hit = self._soma_cache.get(uid)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        soma = editing.soma_region(mask, payload.get("nucleus_mask"),
                                   voxel_size_um=self.voxel_size_um,
                                   **_soma_params(self.derivation))
        self._soma_cache[uid] = (stamp, soma)
        return soma

    def _leave(self, resolution) -> None:
        """The editor resolved a cell, so the decision is recorded before the grid returns.

        A `Resolution` carrying a `reason` is a rejection; anything else is an accept. **Both
        are decisions** — §6.4's rule that a rejected cell is recorded with its reason rather
        than quietly dropped, so `counts_by_reject_reason` can answer for it later.

        > ### EVERY OUTCOME WAS RECORDED AS `confirmed` — found by walking the screen,
        > 2026-08-26
        >
        > This read `resolution.reject_reason` and `resolution.reject_note`. **Those are the
        > STORAGE names** — fields on `schema.CellRecord`, columns in the `cells` table,
        > headers in `cells.csv`. What the editor emits is `editing.Resolution`, which calls
        > the same two things **`reason` and `note`**. Both lookups missed, and both were
        > written as `getattr(..., None)`, so they answered *no rejection here* rather than
        > raising — and `status` is derived from that answer. **A debris rejection was stored
        > as a confirmed cell**, measured and exported as a real one; a `real_cell` rejection
        > was erased from the scene by `_repaint_scene` instead of staying takeable.
        >
        > **The four tests that exercise this method all pass a hand-written stand-in** built
        > with `reject_reason` on it, or a bare `object()`. The fixtures carried the same
        > misunderstanding as the code, so they agreed with it. `test_ui_app.py` now passes
        > real `Resolution`s — §3's lesson 2 again, and the third time this project has been
        > bitten by a fixture that could not fail.
        >
        > **The attributes are read directly rather than through `getattr` defaults.** This is
        > a known object arriving on a known signal; a `Resolution` without a `reason` is not
        > an accept by omission, it is a bug, and it should say so here rather than three
        > screens downstream in a CSV.

        **The tile leaves the grid here too.** `mark_resolved` is what makes §3d.7's gap real
        for a cell that went through rich review; without it the thumbnail stayed on the
        contact sheet after being decided.
        """
        if self.entered is not None:
            reason = getattr(resolution, "reason", None)
            status = "rejected" if reason else "confirmed"
            #: **One builder, shared with the grid's own rejection** — see `_decision_from`,
            #: which also records why the two attribute names are read the way they are.
            self.decisions[self.entered] = self._decision_from(resolution)
            self._persist_edits(self.entered)
            self._repaint_scene(self.entered, status, reason)
            self.triage.mark_resolved(self.entered)
            self.entered = None
        self.stack.setCurrentWidget(self.triage)


def build_screen(step: Step, flow: Flow, viewer: ViewerPort | None) -> QWidget:
    """One screen, constructed late.

    **Built on arrival rather than all at once**, because five of the ten want data that does
    not exist until the screens before them have run — item 5's slider ceiling is item 3's
    coefficient, and item 9a's slider range is two percentiles of a volume.
    """
    from .screen_background import BackgroundFilterScreen
    from .screen_cd68 import Cd68ThresholdScreen
    from .screen_intake import IntakeScreen
    from .screen_nuclei import NucleiScreen
    from .screen_plaque import PlaqueThresholdScreen
    from .screen_soma import SomaParametersScreen
    from .screen_save import SaveParametersScreen
    from .screen_size import CellSizeCutoffScreen
    from .screen_threshold1 import SegmentationThresholdScreen
    from .screen_threshold2 import MeasurementThresholdScreen
    from .screen_veto import VetoScreen

    if step is Step.INTAKE:
        return IntakeScreen(flow)
    if step is Step.SAVE:
        return SaveParametersScreen(flow)
    #: **Item 6.5 has no viewer**, so it joins items 1 and 8 above the table rather than in it
    #: — the table's entries all take `(flow, viewer)`.
    if step is Step.SOMA:
        return SomaParametersScreen(flow)
    if step is Step.REVIEW:
        return ReviewStep(viewer)

    plain = {
        Step.BACKGROUND: BackgroundFilterScreen,
        Step.VETO: VetoScreen,
        Step.THRESHOLD_1: SegmentationThresholdScreen,
        Step.NUCLEI: NucleiScreen,
        Step.THRESHOLD_2: MeasurementThresholdScreen,
        Step.SIZE: CellSizeCutoffScreen,
        Step.CD68: Cd68ThresholdScreen,
        Step.PLAQUE: PlaqueThresholdScreen,
    }
    return plain[step](flow, viewer)



#: Widgets where LEFT and RIGHT mean **move the caret**, and must never be taken for a z step.
#: Everything else on these screens is a button, a swatch or a toggle, where Qt's own answer to
#: an arrow key is to move focus — which is the behaviour being replaced.
TEXT_ENTRY = (QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit)

#: LEFT steps back one slice, RIGHT steps forward. **Modifiers are deliberately ignored**, so
#: `Shift-Left` steps z exactly like `Left` — item 4's split markers are placed with SHIFT held
#: and a clump's daughters sit on different slices, so z has to move while SHIFT is down.
Z_STEP_KEYS = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}


class SemiAutoApp(QStackedWidget):
    """The eleven screens in one window, advanced by the gates in `Flow`.

    **The window never decides whether Continue is allowed** — `Flow` does, and every screen
    already asks it. This only reacts to `continue_requested`, which a screen emits *after* its
    own gate has passed. That keeps the gate logic in the one module with no Qt in it.
    """

    def __init__(self, session: Session | None = None, viewer: ViewerPort | None = None,
                 compute: Compute | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = Flow(session if session is not None else Session())
        #: **Captured before any screen is built.** `IntakeScreen.__init__` runs its own
        #: `_sync`, which writes its empty boxes back onto the session — so a path handed
        #: in on the command line is gone by the time `prepare` could read it off the
        #: session. Found by driving the flow; there is no test that a screen's constructor
        #: leaves its session alone, and item 1's does not.
        self.requested_image = self.flow.session.image_path
        self.viewer = viewer
        self.compute = compute if compute is not None else PipelineCompute()
        self.screens: dict[Step, QWidget] = {}
        #: Where the one canvas currently sits — a `frames.CanvasHost`, or `None` before the
        #: first viewer screen. Held so the previous host can release its claim.
        self.canvas_home = None
        #: The completion popup, once item 8 has been saved. **Held, not local** — a modeless
        #: `QDialog` whose only reference was a local would be collected the moment `_show_saved`
        #: returned, and the popup would flicker and vanish instead of staying on the glass.
        self.saved_popup = None
        self.setStyleSheet(theme.stylesheet())
        #: **The arrow keys are taken application-wide rather than bound to the canvas.** See
        #: `eventFilter`. Installed before the first screen is built, so item 1 is covered too.
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        self.show_step(self.flow.step)


    def eventFilter(self, watched, event):        # noqa: N802 - Qt's name
        """LEFT and RIGHT step z, **wherever the focus happens to be.**

        > ### FOCUS WAS THE WRONG THING TO DEPEND ON — task 26, second attempt
        >
        > napari dispatches its keymap from the canvas widget's Qt key events, so the bindings
        > in `gestures.bind_z_stepping` only fire while that widget holds focus. Task 26's first
        > fix called `setFocus()` on arrival and **it did not take**: `place_canvas` runs during
        > the stack switch, before the arriving widget is shown, and Qt ignores a focus request
        > to a widget that is not visible yet.
        >
        > The reviewer's description named it exactly — *"pressing the keys behaves as tab
        > would, moves the selector around"*. That is Qt's own focus navigation, which is what
        > an arrow key does when nothing else claims it.
        >
        > **So this claims it, application-wide, and never asks who has focus** — except to step
        > aside for a text box, where an arrow key means the caret. Consuming the event is half
        > the fix: it is also what stops the focus ring wandering while the user steps through z.
        """
        if event.type() != QEvent.Type.KeyPress or event.key() not in Z_STEP_KEYS:
            return False
        #: **`watched` first, `focusWidget()` only as a second opinion.** Qt delivers a key
        #: event to the focused widget, so the object being watched *is* that widget — while
        #: `focusWidget()` answers `None` whenever no window is active, which is every run
        #: under `QT_QPA_PLATFORM=offscreen`. Trusting it alone would make the headless suite
        #: unable to exercise this branch at all, and would step z on a keystroke meant for a
        #: number box on any platform that answered the same way.
        application = QApplication.instance()
        focused = application.focusWidget() if application is not None else None
        if isinstance(watched, TEXT_ENTRY) or isinstance(focused, TEXT_ENTRY):
            return False
        step = getattr(self.currentWidget(), "step_z", None)
        if not callable(step):
            return False
        step(Z_STEP_KEYS[event.key()])
        return True

    # -- navigation -----------------------------------------------------

    def show_step(self, step: Step) -> QWidget:
        """Display a step **and move the flow to it**.

        The two cannot be separated: every screen holds the same `Flow` and asks it
        `is_stale()` and `can_continue()`, and those are answered **for `flow.step`**. A
        driver that swapped the widget without moving the flow would show item 2 while its
        gate reported on item 1 — the exclamation and the disabled Continue would both be
        answering a question about a different screen.

        **The gate is not here.** `advance()` is the only way forward and it goes through
        `Flow.advance()`, which refuses a stale view.
        """
        self.flow.step = step
        screen = self.screens.get(step)
        if screen is None:
            screen = build_screen(step, self.flow, self.viewer)
            self.screens[step] = screen
            self.addWidget(screen)
            #: **Item 8 has no Continue, and that is not an omission** — it is the last screen in
            #: `FLOW_ORDER`. *"SAVE is the only control that goes forward, and the window's close
            #: button is the only one that goes anywhere else."*
            forward = getattr(screen, "continue_requested", None)
            if forward is not None:
                forward.connect(self._on_continue)
            back = getattr(screen, "back_requested", None)
            if back is not None:
                back.connect(self._on_back)
            #: Item 7's two inner screens, only one of which shows a picture.
            moved = getattr(screen, "canvas_moved", None)
            if moved is not None:
                moved.connect(self.place_canvas)
            #: **The three commits whose work lives behind `Compute`** — task 24. Items 3 and 9b
            #: run theirs inline, because every function they call is pure and in their own
            #: module; these three need the pipeline, and a screen that imported it would be the
            #: one thing §2's first constraint forbids.
            commit = getattr(screen, "commit_requested", None)
            if commit is not None:
                commit.connect(lambda s=step: self.run_commit(s))
            #: > ### ⇒ THE END OF THE FLOW HAD NO CALLER — wired 2026-08-31
            #: >
            #: > `SavedPopup` was built, styled and tested, and **nothing ever showed it**:
            #: > `show_saved()` had no caller outside the suite, and `saved` and
            #: > `close_program_requested` were emitted into nothing. A session that reached the
            #: > end wrote everything correctly and then sat there — with no `measure` and no
            #: > `export` command, which is the whole of the bridge to Phase C that §4's
            #: > decision 5 closed without.
            #: >
            #: > **The same shape as item 6's inert cutoff and item 9b's dead `Step by batch`:
            #: > correct, tested code behind no caller.** The tests passed because they called
            #: > `show_saved()` themselves — which is why the guard for this lives in
            #: > `test_ui_app.py`, where the driver is, and not beside the widget.
            finished = getattr(screen, "saved", None)
            if finished is not None:
                finished.connect(lambda _result, s=screen: self._show_saved(s))
            #: **`self.close()`, because `SemiAutoApp` IS the window** — `main` calls `show()` on
            #: it directly. Closing the last window ends `app.exec_()`, so the button does what
            #: its label says rather than merely dismissing the dialog.
            closing = getattr(screen, "close_program_requested", None)
            if closing is not None:
                closing.connect(self.close)
        #: > ### THE WINDOW CACHES ARRIVED WITH THE SCREENS AND NEVER LEFT - follow-up to A9
        #: >
        #: > Each z-batch screen holds its own `WindowCache` and `self.screens` keeps every
        #: > screen for the session, so visiting all five at depth 10 accumulates ~1.3 GB of
        #: > cached planes at full width - and `WindowCache.release` was written and called by
        #: > nothing, which is the shape A9 existed to fix, one level up.
        #: >
        #: > **Releasing the screen being LEFT costs nothing, and that is why it is safe.** You
        #: > never return to a screen object you have left: going forward you never revisit
        #: > one, and going back `_discard_screens_from` destroys it and it is rebuilt fresh
        #: > with an empty cache. A departed screen's cache is dead weight on every path.
        #: >
        #: > **Guarded on `is not screen`** because `show_step` is also called to re-show the
        #: > screen already up - `__init__` does it for item 1, and so does the first step of
        #: > any walk of `FLOW_ORDER`.
        outgoing = self.currentWidget()
        if outgoing is not None and outgoing is not screen:
            windows = getattr(outgoing, "windows", None)
            if windows is not None:
                windows.release()
        self.setCurrentWidget(screen)
        #: **The arriving screen gets an empty viewer.** There is one viewer for the session and
        #: nothing else removes a layer, so without this every screen inherits every layer the
        #: screens before it added — see `NapariViewer.clear` for the two things that breaks.
        #: Safe unconditionally: `prepare` hands each viewer screen its arrays and every one of
        #: them draws on receipt, so a cleared viewer is refilled in the same call.
        if self.viewer is not None:
            #: > ### THE GESTURES COME OFF WITH THE LAYERS - task 25
            #: >
            #: > One viewer serves eleven screens, so a handler left attached answers a click on
            #: > behalf of a screen the user is no longer looking at: item 7b's ray walk would
            #: > still be bound while item 9a is showing, and it would walk item 9a's layers.
            #: > **Exactly the failure `clear()` fixes for layers, one level up** - and found
            #: > the same way, by asking what a shared object is still holding.
            self.viewer.unbind_all()
            self.viewer.clear()
        self.prepare(step)
        #: > ### AN ARRIVING SCREEN RE-READS ITS OWN GATE - found by walking the Back path
        #: >
        #: > Every screen keeps Continue and the red `!` in step with `Flow` through its own
        #: > `_sync`, and `_sync` runs when one of **that screen's** controls moves. Nothing ran
        #: > it on arrival - so a screen whose gate changed while it was off-stage came back
        #: > showing the answer it left with.
        #: >
        #: > **Item 4 is the only screen with a Back button, and that is the path:** commit item
        #: > 4, go Back, re-commit item 3, Continue. `Flow.commit` now invalidates item 4, and
        #: > without this line item 4 would still arrive with Continue enabled and no
        #: > exclamation - the gate correct in the model and wrong on the glass.
        #: >
        #: > It also writes the arriving screen's parameters onto the session, so a screen's
        #: > defaults are on `session.chosen` from the moment it is seen rather than from the
        #: > first time a control is touched.
        #:
        #: **After `prepare`**, which is what fills the boxes the gate reads, and **never for
        #: item 1**, whose `_sync` is a *write* rather than a read.
        #:
        #: > Item 1 is the authority on intake: its `_sync` puts its own boxes onto the session,
        #: > so running it against a session carrying a path from the command line or roles from
        #: > a driver **clears them** — the hazard `__init__` records above, arriving through a
        #: > second door. Every other screen's `_sync` reads its own controls and writes only its
        #: > own parameters, which is why the exception is item 1's alone.
        #: >
        #: > It is not hypothetical: `show_step(Step.INTAKE)` runs once in `__init__` and again
        #: > on the first step of any walk of `FLOW_ORDER` — which is exactly what
        #: > `scripts/gui_shakeout.py` does, and what task 26 will do against a real file.
        #: > **This was two guards and one of them was dead.** It also skipped screens being
        #: > shown for the first time, on the argument that their constructors had just synced -
        #: > true, and worth nothing: a mutation flipping that guard on could not be made to
        #: > fail a test, because running `_sync` twice on a fresh screen changes nothing. One
        #: > live guard is better than a live one and a dead one kept in step by hand.
        refresh = getattr(screen, "_sync", None) if step is not Step.INTAKE else None
        if callable(refresh):
            refresh()
        #: **After `prepare`, not before.** `prepare` is what hands a screen its arrays, and
        #: several screens draw as soon as they receive them — item 6 pushes three layers inside
        #: `set_cells`. Placing the canvas first would apply a grid to a viewer that has no
        #: layers in it yet, and napari assigns panels by layer index at the moment the layer
        #: arrives.
        self.place_canvas()
        #: **After the canvas is placed**, because a screen binds against the viewer it is
        #: about to be shown in and item 7b's binding asks the viewer for its `ndisplay`.
        #: Screens with no gestures do not have the method; that is most of them.
        gestures = getattr(screen, "bind_gestures", None)
        if callable(gestures) and self.viewer is not None:
            gestures()
        return screen

    # -- the commits, which are the work each button names --------------

    def run_commit(self, step: Step) -> None:
        """Run the computation the COMMIT on `step` names, and hand the result to the screen.

        > ### EVERY COMMIT USED TO ONLY FLIP THE GATE — task 24, §3's second structural gap
        >
        > Five screens carry a COMMIT and the button is what admits the cost: item 2 is ~17 s,
        > items 3 and 5 ~59 s, **item 4 is 200-500 s**, item 9b 9-11 s. Items 3 and 9b ran their
        > work; the other three called `flow.commit(None)` and nothing else. **`Compute.detect`
        > was never called from anywhere in the program**, so item 4 — the screen the whole
        > 200-500 s budget exists for — received no nuclei at all and drew an empty label layer
        > over the DAPI.
        >
        > **Why here and not in the screens.** Items 3 and 9b compute with pure functions from
        > their own modules, so they own their commits. These three need `ImageSource`,
        > `detect_nuclei` and `build_cell_mask`; a screen importing those would break §2's first
        > constraint — *the UI calls the pipeline, and the pipeline never calls the UI* — in the
        > one direction that matters, and would put a cellpose import inside a widget.

        **The screen has already flipped its own gate** before this runs, and that ordering is
        deliberate: the gate is about *whether the parameters moved since the last commit*, which
        is answered by the button being pressed, not by the work succeeding. What the work
        produces is the picture and the readout, and those arrive here.
        """
        session, screen = self.flow.session, self.screens[step]
        if step is Step.BACKGROUND:
            #: **The product is the channel every later screen and the detector read.** It is a
            #: second full-resolution float32 copy of DAPI — 849 MB at production width — held
            #: alongside the raw one, and that is deliberate rather than overlooked: the screen
            #: needs the raw channel to preview against and to fall back to when a control
            #: moves, and §2's memory rule is *hold fewer channels, not narrower ones*, which
            #: this does not violate — it holds one extra, for one channel, knowingly.
            filtered = self.compute.filtered_nuclei(session)
            session.products[Step.BACKGROUND] = filtered
            screen.commit_result(filtered)

        elif step is Step.VETO:
            #: Both products, and they are **not** the same array: `products[Step.VETO]` is the
            #: undilated mask because detection dilates it itself, and `applied` is what the
            #: screen draws so the outline and readout move with `plaque_dilate_um`.
            mask, applied = self.compute.veto(session)
            session.products[Step.VETO] = mask
            #: > ### ⇒ THIS REDREW THE RAW CHANNEL, AND ONLY AFTER COMMIT — fixed 2026-08-31
            #: >
            #: > `prepare` hands this screen `self._dapi(session)`, the **filtered** channel item
            #: > 1.5 committed, and says why: *"the dapi that goes through to the next part will
            #: > be the surviving dapi signal only."* **This branch read
            #: > `session.channels["nuclei"]` instead** — the raw channel — so item 2 showed the
            #: > right DAPI on arrival and swapped to the unfiltered one the moment COMMIT was
            #: > pressed, which is exactly when it is being looked at hardest.
            #: >
            #: > **Both the `DAPI` layer and the `log DAPI` toggle were affected**, because
            #: > `LOG_DAPI` is a display transform over this same array rather than a layer of
            #: > its own. Found by the user, 2026-08-31: *"item 2 is not showing what I want it
            #: > to for the dapi and log dapi."*
            #: >
            #: > The rule was already written **four lines above the branch that broke it** —
            #: > which is the argument for `_dapi` existing at all rather than the two-line
            #: > fallback being repeated per call site.
            dapi, plaque = self._dapi(session), session.channels.get("plaque")
            if dapi is not None and plaque is not None:
                #: `draw` sets the readout itself from the mask it is given, so redrawing is
                #: what makes `DAPI removed by the veto` report **this** commit's veto rather
                #: than the one computed at arrival from the defaults.
                #: **No z or depth**, because item 2 keeps neither — `draw` takes them as
                #: arguments and `prepare` passes the defaults. The z-batch box on this
                #: screen is not yet connected to anything; that is task 25.
                screen.draw(dapi, applied, plaque)

        elif step is Step.NUCLEI:
            labels, cell_mask = self.compute.detect(session)
            session.products[Step.NUCLEI] = labels
            screen.set_detection(labels, cell_mask)

        elif step is Step.THRESHOLD_2:
            #: Item 5 compares **two** masks, and the left one is item 3's own commit rather
            #: than a second computation of it — `commit_mask` returns a `CellMask` and
            #: `filled` is the hole-filled array the pipeline carries forward.
            segmentation = session.products.get(Step.THRESHOLD_1)
            segmentation = getattr(segmentation, "filled", segmentation)
            if segmentation is None:
                segmentation = self.compute.segmentation_mask(session)
            measurement = self.compute.measurement_mask(session)
            screen.commit_result(segmentation, measurement, session.channels.get("cd68"))

    # -- the canvas, which is one object for the whole session ----------

    def place_canvas(self) -> None:
        """Move the single canvas into the screen now showing, and arrange it for that screen.

        > ### THE CANVAS TRAVELS; IT IS NOT REBUILT — task 23
        >
        > Before this, all seven viewer screens held a black `QLabel` and `main()` opened a
        > **separate** napari window beside the control window. The picture and the controls for
        > the picture were in two different windows, and neither knew where the other was.
        >
        > One viewer, moved. A viewer per screen would be seven OpenGL contexts beside a 3.17 GiB
        > image, and the flow already guarantees exactly one screen is visible.

        **A screen with no canvas leaves it where it is** rather than re-parenting it out.
        `setParent(None)` would make the canvas a top-level window — a stray napari window, which
        is precisely the arrangement being removed. Parked inside the previous screen it is
        hidden by the stack along with that screen, at no cost.

        **The arrangement is applied even when there is no widget to move**, which is the
        headless case: `RecordingViewer` records the grid and `ndisplay` a screen asked for, so
        the arrangement decisions stay assertable in a suite with no display. What no test can
        assert is whether an image appeared — that is what a person in front of the screen is
        for, and §3's new rule.
        """
        screen = self.currentWidget()
        host = getattr(screen, "canvas_host", None)
        host = host() if callable(host) else getattr(screen, "canvas", None)
        layout = getattr(screen, "canvas_layout", None)
        layout = layout() if callable(layout) else getattr(screen, "CANVAS", None)
        if host is None:
            return
        if self.canvas_home is not None and self.canvas_home is not host:
            self.canvas_home.give_up()
        widget = self.viewer.widget() if hasattr(self.viewer, "widget") else None
        host.take(widget)
        self.canvas_home = host
        if layout is not None and self.viewer is not None:
            self.viewer.set_ndisplay(layout.ndisplay)
            self.viewer.set_grid(layout.shape, layout.stride)
            #: **Last, and that order is the whole of it.** The camera is fitted to the viewbox
            #: it is looking through, and both lines above change that viewbox — the grid halves
            #: it and `ndisplay` changes what "fitted" means. Refitting first would fit the
            #: previous screen's arrangement. See `NapariViewer.reset_view`.
            self.viewer.reset_view()

        #: **The canvas is given keyboard focus every time it lands.** napari dispatches its
        #: keymap from the canvas widget's Qt key events, so LEFT and RIGHT were bound and
        #: unreachable until something focused it — see `NapariViewer.focus_canvas`. It goes
        #: here rather than in `show_step` so it follows the canvas, including item 7's inner
        #: switch from triage into rich review.
        if self.viewer is not None and hasattr(self.viewer, "focus_canvas"):
            self.viewer.focus_canvas()

    def _on_continue(self) -> None:
        self.advance()

    def _show_saved(self, screen) -> "QWidget":
        """Show the completion popup. **The last thing the flow does.**

        `screen.show_saved()` builds it and connects its `Close Program` through to the screen's
        own `close_program_requested`; the driver's job is only to put it on the glass and to
        keep a reference so Qt does not collect it.

        **`setModal(True)` and `show()`, never `exec()`.** Both are application-modal, but
        `exec()` spins its own event loop and does not return until the dialog closes — which
        would hang the headless suite on the one test that proves this popup appears at all.
        The popup is returned for the same reason `show_saved` returns it: so a caller can
        assert on what was shown rather than on what was built.
        """
        popup = screen.show_saved()
        self.saved_popup = popup
        popup.setModal(True)
        popup.show()
        return popup

    #: **One popup, in the driver, not one per screen.** Ten screens each raising their own
    #: *are you sure* is ten chances to word it differently and one chance to forget it
    #: entirely — the shape of every count this project has had to go back and fix.
    BACK_CONFIRM_TITLE = "Go back"

    def _back_question(self) -> str:
        """What is about to be lost, named. The user asked for *are you sure*; naming the screen
        is what makes the answer informed rather than reflexive."""
        #: **Asked of the flow, not of the module map.** `BACK_FROM` is the four-channel
        #: answer; a session that skipped item 9a must not be offered a screen it never
        #: saw, and `_on_back` has already refused the move where there is no target.
        target = self.flow.back_target()
        return (f"Go back to {target.value}?\n\n"
                f"Everything you set on {self.flow.step.value} will be discarded, and "
                f"{target.value} starts again at its defaults.")

    def confirm_back(self, question: str) -> bool:
        """The *are you sure* modal. **A named method so a test can answer it** — the pattern
        item 7b's `confirm_transfer` established, and the reason nothing here has to construct a
        dialog to prove that declining changes nothing."""
        from .widgets import ConfirmPopup

        return ConfirmPopup(question, "Yes", "No",
                            title=self.BACK_CONFIRM_TITLE, parent=self).ask()

    def _on_back(self) -> None:
        """Ask, then go. **Refusing changes nothing at all** — no commit is dropped, no widget
        rebuilt, and the screen is exactly where it was."""
        if not self.flow.can_go_back():
            return
        if not self.confirm_back(self._back_question()):
            return
        target = self.flow.back()
        #: **The widgets are dropped here, and this is the other half of the rule.** `Flow`
        #: cleared the commits and the products; a screen kept in `self.screens` would come back
        #: holding the text its boxes had when it was left, so *the screen is default whether you
        #: go forward or back to it* would be true of the model and false of the glass.
        self._discard_screens_from(target)
        self.show_step(target)

    def _discard_screens_from(self, target: Step) -> None:
        """Forget every built screen at `target` or later, so each is rebuilt fresh.

        **Forward re-entry still keeps its screen** — `test_the_screens_are_built_late_and_kept`
        pins that, and it is why this is called from `_on_back` rather than from `show_step`.

        > ### ⇒ THIS TOOK THE NAPARI CANVAS WITH IT, AND THAT IS THE Back CRASH — item B1
        >
        > The screen being discarded is often **the one currently showing**, and the single
        > canvas is parented *inside* it: `place_canvas` does `host.take(widget)`, and
        > `CanvasHost.give_up` explicitly *"leaves the widget parented here"*. So
        > `screen.setParent(None)` made the screen a top-level widget with no Python reference,
        > it was collected, and **its C++ destructor took every child with it — including
        > napari's `_qt_viewer`.** The next `place_canvas` touched a deleted C++ object, which
        > is an abort rather than a traceback. That is the crash from every Back button.
        >
        > **Why no test caught it.** `place_canvas` is guarded by `hasattr(self.viewer,
        > "widget")` and `widget()` exists only on `NapariViewer`; the headless suite uses
        > `RecordingViewer`, which has none — so `host.take(None)` re-parents nothing and there
        > is nothing to destroy. The suite runs the exact code path and cannot reach the bug.
        > `test_ui_canvas.py` now hands the driver a stub with a real `QWidget` to close that.
        >
        > **Nothing else changes.** `Flow.discard_from` still drops every commit, product and
        > derivation from the target forward and the screens are still rebuilt fresh — the
        > user's rule of 2026-08-30, *"the screen is default whether you go forward or back to
        > it"*. Only the canvas stops being collateral.
        """
        doomed = [step for step in FLOW_ORDER[FLOW_ORDER.index(target):] if step in self.screens]
        self._evacuate_canvas([self.screens[step] for step in doomed])
        for step in doomed:
            screen = self.screens.pop(step, None)
            if screen is not None:
                self.removeWidget(screen)
                screen.setParent(None)

    def _evacuate_canvas(self, doomed) -> None:
        """Take the canvas out of any screen about to be destroyed, before it is.

        Re-parented to the app and hidden, which is where a canvas between screens belongs:
        `setParent(None)` would make it a top-level window — the stray napari window this whole
        arrangement exists to remove — and leaving it where it is destroys it.

        **`canvas_home` is cleared too**, or the next `place_canvas` would call `give_up()` on a
        host whose C++ object is gone.
        """
        host = self.canvas_home
        if host is None:
            return
        if not any(screen is host or screen.isAncestorOf(host) for screen in doomed):
            return
        widget = host.give_up()
        if widget is not None:
            widget.setParent(self)
            widget.hide()
        self.canvas_home = None

    def advance(self) -> Step:
        """Leave the current screen. **`Flow.advance` refuses a stale view**, so a screen that
        emits `continue_requested` while gated raises rather than moving on."""
        following = self.flow.advance()
        self.show_step(following)
        return following

    # -- the data hand-offs ---------------------------------------------

    def prepare(self, step: Step) -> None:
        """Give a screen what it needs. **The one place the flow's data dependencies live.**

        Each branch names what it hands over and where it came from, because that is the
        information a screen deliberately does not have: item 5 shows threshold-1's mask beside
        its own, item 9a shows the cells item 7 confirmed, and neither knows which step produced
        them.
        """
        session, screen = self.flow.session, self.screens[step]
        if step is not Step.INTAKE:
            self._ensure_loaded(session)

        #: **Item 6's cutoff is applied on the way OUT of item 6, not on the way in**, which is
        #: what lets the box mean anything: the screen needs the whole nucleated population to
        #: choose against, and every screen after it needs the choice. Placed here rather than on
        #: the Continue button because `advance()` is not the only way forward — the driver can be
        #: sent to a step directly, and a cutoff that depended on which route you took would be
        #: the same class of defect this closes.
        if step in AFTER_SIZE:
            self._apply_cutoff(session)

        #: **Item 1's per-channel display range, pushed to whichever screen is arriving.**
        #: Set before the `set_*` hand-off below, because those methods draw immediately and a
        #: range that arrived after the first draw would not be applied until the next one.
        #: Screens that hold no raw channel simply do not carry the attribute.
        if hasattr(screen, "display_range"):
            screen.display_range = dict(session.display_range)

        if step is Step.INTAKE:
            #: **The screen is the authority on intake, so anything already on the session has
            #: to be pushed INTO it.** `IntakeScreen._sync` runs at construction and writes the
            #: boxes back onto the session — empty roots, no file, no roles — so a session built
            #: with `--image` would have been silently cleared. Found by driving the flow.
            #:
            #: **Nothing else is pre-filled**, per §4 item 1: *nothing is remembered between
            #: runs*. This carries one value the user typed on the command line one second ago,
            #: which is not cross-run state.
            if self.requested_image and not screen.file_path.text():
                screen.file_path.setText(self.requested_image)

        elif step is Step.BACKGROUND:
            screen.set_volume(session.channels.get("nuclei"),
                              voxel_size_um=session.voxel_size_um)

        elif step is Step.VETO:
            mask, applied = self.compute.veto(session)
            session.products[Step.VETO] = mask
            #: **The filtered channel, not the raw one** — the user's rule, 2026-08-23: *"the
            #: dapi that goes through to the next part will be the surviving dapi signal
            #: only."* Item 2 is where the veto is judged, so it has to be judged against the
            #: DAPI the veto will actually be applied to. `_dapi` falls back to the raw channel
            #: when the screen before has not committed, which is the same shape as every other
            #: product reuse here.
            screen.draw(self._dapi(session), applied,
                        session.channels["plaque"])

        elif step is Step.THRESHOLD_1:
            screen.set_volume(session.channels["cell"], voxel_size_um=session.voxel_size_um)

        elif step is Step.NUCLEI:
            #: **At segmentation resolution, because the labels arrive there.** `detect` returns
            #: what `pipeline.py` computes — labels downsampled by `params.downsample` and the
            #: cell mask downsampled to match — and a label layer half the size of the image
            #: layers beneath it would sit in the corner of the canvas. Averaged rather than
            #: subsampled: `downsample_xy` preserves signal-to-noise on the faint processes.
            #:
            #: > ### THE `DAPI -veto` LAYER WAS SHOWING THE FULL DAPI CHANNEL — fixed 2026-08-20
            #: >
            #: > It was handed `session.channels["nuclei"]` raw, so **the layer named for a
            #: > subtraction never performed one.** Detection was never affected — the veto is
            #: > applied inside `detect_nuclei`, which re-applies it with its own
            #: > `plaque_dilate_um` — so this was a display defect, not a measurement one.
            #: >
            #: > **But it is the screen where the veto is judged**, and showing full DAPI there
            #: > makes a working veto look like a broken one. The user found it while chasing
            #: > false-positive nuclei at a plaque; the real cause was the xy-only margin
            #: > (`veto_mask_at_margin`), and this defect hid the evidence.
            #: >
            #: > **Zeroed at FULL resolution and downsampled afterwards, not the reverse.**
            #: > `downsample_xy` block-averages, so averaging first would smear vetoed intensity
            #: > into its surviving neighbours and leave a ghost exactly where the veto bit.
            #: >
            #: > **Recomputed from `products[Step.VETO]` rather than re-running the threshold.**
            #: > That product is the *undilated* mask by contract — the margin is applied by
            #: > `_veto_as_applied`, which is one distance transform instead of a ~17 s
            #: > re-threshold of the 6E10 channel on every entry to this screen.
            screen.set_channels(self._vetoed_nuclei(session),
                                self._at_segmentation_resolution(session, "cell"))

        elif step is Step.THRESHOLD_2:
            #: **Item 3's committed coefficient, pushed before the volumes.** The ceiling of
            #: this screen's slider is that value — `slider_ceiling` calls it *"set by ANOTHER
            #: SCREEN"* — and it was a constructor argument this mapping never passed, so every
            #: session got the 0.18 default no matter what the user chose. It goes first
            #: because `set_volumes` draws, and a draw wants the slider it will read.
            chosen = session.chosen.get("cell_coefficient")
            if chosen:
                screen.set_cell_coefficient(float(chosen))
            screen.set_volumes(session.channels.get("cell"), session.channels.get("cd68"))

        elif step is Step.SIZE:
            derivation = self._derive(session)
            screen.set_cells(derivation.labels, self._volumes(derivation, session))

        elif step is Step.REVIEW:
            derivation = self._derive(session)
            cells, payloads = review_cells(derivation)
            #: **A CATEGORY volume since 2026-08-25**, not a mask: `0` nothing, `1` debris,
            #: `2` another cell. One flat layer could not tell a neighbour from unclaimed
            #: material, so the question item 7b exists to answer — *does this belong to me or
            #: to the cell next to it* — had no visible answer.
            #:
            #: `uint8` costs the same as the bool it replaces, and its union is still exactly
            #: `measure.filled`: moving material between cell and debris never changes the total,
            #: which is what makes the picture invariant under review.
            scene = np.zeros(derivation.labels.shape, np.uint8)
            if derivation.debris_mask is not None:
                scene[derivation.debris_mask] = SCENE_DEBRIS
            scene[derivation.labels > 0] = SCENE_CELL
            #: Every nucleus a surviving cell owns, flattened once here rather than per entered
            #: cell — `_other_nuclei` subtracts the entered cell's own from it.
            owned = {int(n) for ids in derivation.nuclei_by_cell.values() for n in ids}
            screen.set_cells(cells, payloads, path=session.image_path,
                             voxel_size_um=session.voxel_size_um or (1.0, 1.0, 1.0),
                             scene=scene,
                             iba1=session.channels.get("cell"),
                             dapi=self._detector_dapi(session),
                             nuclei_labels=derivation.nuclei_labels,
                             owned_nuclei=owned,
                             #: **The same object item 8 reads** — `_derive` caches it on the
                             #: session — which is what lets an edit painted here be re-derived
                             #: into a record there without a second channel between them.
                             derivation=derivation)

        elif step is Step.CD68:
            #: **Iba1, not the cell set.** Item 9a became a two-panel screen on 2026-08-30 —
            #: Iba1 in both panels, CD68 raw on the left and above-threshold on the right —
            #: and the `cells` outline was removed, so no derivation is needed here at all.
            #: **The Iba1 MEASUREMENT mask, not the cell set** — the user's correction,
            #: 2026-08-30. It is what item 5 chose and what every CD68 volume is measured
            #: inside, so it is the outline to judge a CD68 threshold against.
            #: ⇒ **ITEM 5's COMMIT ALREADY BUILT THIS — patch item A7, 2026-09-03.**
            #:
            #: `products[Step.THRESHOLD_2]` is written by `commit_result` — it is exactly what
            #: `measurement_mask` returned at that commit — and until now it was **read by
            #: nothing**, so arriving here rebuilt a ~59 s mask the session was already holding.
            #:
            #: **The fallback is the shape `Compute.detect` already uses** for items 2 and 3: a
            #: driver that reaches this screen without walking through item 5 still gets the
            #: mask the parameters describe, rather than `None` and a blank outline.
            measurement = session.products.get(Step.THRESHOLD_2)
            if measurement is None:
                measurement = self.compute.measurement_mask(session)
            screen.set_channels(session.channels.get("cd68"),
                                session.channels.get("cell"),
                                measurement)

        elif step is Step.SOMA:
            #: **Nothing to hand it.** Item 6.5 is a form with no viewer and no volume — the one
            #: thing it needs is the cached `Derivation`, which it reaches through the session
            #: itself so that its own `_sync` can write the four values onto it. See
            #: `screen_soma.SomaParametersScreen._sync`.
            pass

        elif step is Step.PLAQUE:
            #: **The veto AS APPLIED, not `products[Step.VETO]`** — the user's correction,
            #: 2026-08-30: *"the seed mask shown here should be the dilated one from the item 2
            #: screen."* The product is undilated by contract, so passing it drew a narrower
            #: mask than item 2 showed and classified seeded from unseeded against it.
            screen.set_channels(session.channels.get("plaque"),
                                self._veto_as_applied(session),
                                session.voxel_size_um or (1.0, 1.0, 1.0))

        elif step is Step.SAVE:
            derivation = self._derive(session)
            session.cells = derivation.payloads()
            #: **Item 7's decisions travel to item 8 here, and nowhere else could carry them.**
            #: Without this the pool holds only `CANDIDATE` cells and `measure` reports
            #: *"measured 0 cell(s) across 0 image(s)"* — see `screen_save.apply_decisions`.
            review = self.screens.get(Step.REVIEW)
            if review is not None:
                session.decisions = dict(review.decisions)
            screen.refresh()

    @staticmethod
    def _at_segmentation_resolution(session: Session, role: str):
        """One channel, block-averaged in xy by `downsample`. `None` stays `None`."""
        from ..resolution import downsample_xy

        volume = session.channels.get(role)
        if volume is None:
            return None
        factor = int(session.chosen.get("downsample", 2))
        return downsample_xy(volume, factor) if factor > 1 else volume

    def _dapi(self, session: Session):
        """**The DAPI every screen after the background filter sees**, at native resolution.

        The committed filtered channel when there is one, the raw channel when there is not —
        which is the state before that screen commits, and the state of any driver that reaches
        a later screen without walking through it. One accessor rather than the same two-line
        fallback at three call sites, for `widgets.py`'s reason: three copies drift three ways.
        """
        filtered = session.products.get(Step.BACKGROUND)
        return session.channels.get("nuclei") if filtered is None else filtered

    def _vetoed_nuclei(self, session: Session):
        """The DAPI channel with the veto removed, at segmentation resolution.

        Returns the plain channel when there is no plaque role or no committed veto, so a
        two-channel study still gets its nuclei layer.

        **It starts from `_dapi`, so the background filter is already in it.** Item 4's layer is
        named for what the detector receives, and the detector receives the filtered channel —
        showing the unfiltered one there would repeat 2026-08-20's `DAPI -veto` defect in the
        other half of the same expression.
        """
        from ..resolution import downsample_xy

        dapi = self._vetoed_nuclei_native(session)
        factor = int(session.chosen.get("downsample", 2))
        if dapi is not None and factor > 1:
            dapi = downsample_xy(dapi, factor)
        return dapi

    @staticmethod
    def _veto_as_applied(session: Session):
        """**The veto with `plaque_dilate_um` on it** — what item 2 draws, and what removes DAPI.

        `products[Step.VETO]` is the *undilated* mask by contract, because `detect_nuclei`
        applies the margin itself and a pre-widened mask would dilate twice. Every screen that
        needs the mask a **user** can see comes through here instead of reading the product.

        **One helper, two screens.** Item 4 needs it for `DAPI -veto` and item 9b needs it as
        6E10's seed mask; two copies of *the veto as applied* differing by a default argument is
        precisely how the 2026-08-20 defect got into one resolution and not the other.

        **It costs nothing at `plaque_dilate_um = 0`** — `veto_mask_at_margin` returns its input
        unchanged there — and one distance transform above it, which is what item 4 has paid per
        entry since task 26. `None` in, `None` out, so a two-channel study is quiet rather than
        broken.
        """
        from ..nuclei import veto_mask_at_margin

        raw = session.products.get(Step.VETO)
        if raw is None:
            return None
        return veto_mask_at_margin(
            raw,
            dilate_um=float(session.chosen.get("plaque_dilate_um", 0.0)),
            voxel_size_um=session.voxel_size_um)

    def _vetoed_nuclei_native(self, session: Session):
        """The same channel **at native resolution** — item 7b draws native crops.

        Split out of `_vetoed_nuclei` rather than copied, because two versions of "the DAPI the
        detector sees" differing only in resolution is exactly how the `DAPI -veto` defect of
        2026-08-20 got into one of them and not the other.
        """
        import numpy as np

        dapi = self._dapi(session)
        applied = self._veto_as_applied(session)
        if dapi is not None and applied is not None:
            dapi = np.where(applied, 0, dapi)
        return dapi

    def _detector_dapi(self, session: Session):
        """**What Cellpose actually sees**, at native resolution: filtered, vetoed, log-compressed.

        Item 7b's fifth layer was called `raw DAPI` and was passed nothing at all. Raw is the
        wrong thing to show even once it is passed: the detector runs on
        `normalize_for_detection`'s log compression, and the gap between the two is not
        cosmetic — the measured result recorded in `nuclei.py` is that an aggressive percentile
        clip found **5** nuclei where log compression found **24**. A reviewer judging whether a
        nucleus was missed has to be looking at the volume the detector judged.
        """
        from ..nuclei import normalize_for_detection

        dapi = self._vetoed_nuclei_native(session)
        return None if dapi is None else normalize_for_detection(dapi, method="log")

    def _ensure_loaded(self, session: Session) -> None:
        """Read the image **once**, on first need rather than at a named step.

        Item 2 is the first screen that wants a channel, so tying the read to it would be
        correct for the normal path and wrong for every other entry — and a screen reached
        with `session.channels` empty does not fail loudly, it hands `None` to a slider and
        raises somewhere unrecognisable. Found by driving the flow out of order.
        """
        if not session.channels and session.image_path and session.roles:
            self.compute.load(session)

    def _derive(self, session: Session):
        """The cell set, computed **once** and held on the session.

        Recomputing it per screen would re-run segmentation between item 6 and item 8 and could
        return a different answer if any parameter moved — which is precisely the stale-view
        problem the commit gates exist to prevent, one level up.
        """
        if session.derivation is None:
            session.derivation = self.compute.derive(session)
        return session.derivation

    def _apply_cutoff(self, session: Session) -> int:
        """Fold cells under item 6's cutoff out of the derivation and into debris.

        **Idempotent at a given value, monotonic across values.** Applying 50 twice removes
        nothing the second time — `filter_cells` has no sub-50 cells left to find — so the guard
        is an optimisation rather than a correctness fix. Raising the cutoff on a re-entry
        removes more; lowering it restores nothing, which is the user's decision of 2026-08-25
        taken with the non-destructive alternative described.

        **Silent when there is nothing to do**, including before the derivation exists — the
        driver can be sent to a late step directly, and refusing there would break entry paths
        that have nothing to do with this.
        """
        from ..pipeline import apply_cell_size_cutoff

        derivation = session.derivation
        if derivation is None:
            return 0
        cutoff = session.chosen.get("min_cell_volume_um3")
        if cutoff is None:
            return 0
        cutoff = float(cutoff)
        if (session.cutoff_applied_um3 is not None
                and cutoff <= session.cutoff_applied_um3):
            return 0
        removed = apply_cell_size_cutoff(derivation, cutoff)
        session.cutoff_applied_um3 = cutoff
        return removed

    @staticmethod
    def _volumes(derivation, session: Session):
        """µm³ per cell id, from the label volume — the same arithmetic `filter_cells` uses."""
        voxel = float(np.prod([float(v) for v in (session.voxel_size_um or (1.0, 1.0, 1.0))]))
        counts = np.bincount(np.asarray(derivation.labels).ravel())
        return [float(counts[i]) * voxel for i in range(1, len(counts))]


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------

def main(image: str | None = None) -> int:
    """Open the window and run it. **The only caller of `napari.run()`'s neighbourhood.**

    Everything GUI is imported here rather than at module scope, so importing `ui.app` costs
    nothing on a cluster node — the same discipline `nuclei.py` applies to cellpose.
    """
    from qtpy.QtWidgets import QApplication

    from .viewer import NapariViewer

    app = QApplication.instance() or QApplication([])
    theme.load_bundled_font()

    session = Session(image_path=image or "")
    #: **`show=False`, and that is task 23's whole point.** The canvas is moved into the screen
    #: that is showing, so napari's own window must never appear — it would be a second window
    #: holding the picture while the screens' centre panels stayed black.
    window = SemiAutoApp(session, viewer=NapariViewer(show=False))
    window.setWindowTitle("prestomorph")
    fit_to_display(window, app)
    window.show()
    return int(app.exec_())


def fit_to_display(window, app, *, preferred=(1400, 900)) -> tuple[int, int]:
    """Open at `preferred`, **but never larger than the display can show.**

    > ### A WINDOW THAT SIZES ITSELF TO ITS CONTENT HAS NO WAY BACK — found by running item 6
    >
    > `SelectionList` grew one row per cell with nothing to scroll in, and one measured field
    > holds 537 of them: the window was photographed at **2470 px tall** on a `resize(1400, 900)`.
    > That widget now scrolls, which is the actual fix — this is the guard that keeps the class
    > of defect from reaching a user again, because any child with an unbounded size hint does
    > the same thing and there are five columns of them.
    >
    > **Bunya is the reason it is worth a function.** The run happens in an Open OnDemand desktop
    > session inside a browser window, so the usable desktop is smaller than a real one and
    > unknown until the session opens. A window taller than it cannot be dragged back.

    Returns the size actually used, so a caller can say what it did.
    """
    screen = app.primaryScreen()
    if screen is None:
        window.resize(*preferred)
        return preferred
    available = screen.availableGeometry()
    #: Leave room for the window frame and whatever the desktop keeps for itself.
    width = min(preferred[0], max(800, available.width() - 40))
    height = min(preferred[1], max(600, available.height() - 80))
    window.setMaximumSize(available.width(), available.height())
    window.resize(width, height)
    return (width, height)
