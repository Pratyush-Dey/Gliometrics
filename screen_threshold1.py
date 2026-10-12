"""Screen 3 — Iba1 Segmentation Threshold. `COMPONENT_03_PLAN.md` §6g, §4 item 3.

**Frame A, and the only screen that uses it.** No `Colours` block, no `Layers` block, no z-batch
box and no projection mode — which is *why* the controls take the left column. §4's note that
they sit left here and right at item 2 is a consequence of the left column being free, not a
stylistic alternation.

Owns `cell_coefficient`, `min_object_voxels` and `closing_radius_um`. It decides **how many
cells there are and where they divide**, and it must precede detection because membership needs
it to say which nuclei are microglial.

> ### THE PREVIEW IS `threshold_channel` ON ONE SLICE — AND IT NEEDS THE WHOLE STACK'S CONTRAST
>
> `threshold_channel`'s empty-slice guard is **volume-relative**: it suppresses any plane whose
> contrast falls below `empty_slice_ratio` of the whole stack's. Hand it a single slice and that
> slice becomes its own reference, so the test reads `x < x * 0.05` and **can never fire** —
> the preview would show a mask the commit then blanks. `volume_contrast` is computed **once**
> in `set_volume` and passed to every preview. This is task 6's fix, and it is the one thing on
> this screen that cannot be got wrong quietly.

**Nothing else about the preview is faithful by accident either.** `min_object_voxels` is a 3D
connected-component filter and `binary_fill_holes` runs in 3D, so neither can be reproduced on
one plane — both are commit-only, and that is what keeps the right-hand panel honest. The
committed mask will look meaningfully **fuller** than the preview; §4 considered a line saying so
and dropped it, which is why the panel is headed `Threshold preview` rather than `binary mask`.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ..cache import SliceCache, cache_key
from ..objects import find_cell_objects
from ..threshold import CellMask, build_cell_mask, slice_contrast, threshold_channel
from .palette import ring_colour
from . import gestures
from .frames import CanvasHost, FrameA
from .preview import middle_slice, preview_mask, volume_contrast_limits, z_position
from .semiauto import Flow, Session, Step, parsed_numbers
from .viewer import CanvasLayout, ViewerPort
from .widgets import (
    CommitButton, HeaderStrip, HelpButton, NavigationRow, ParameterSlider,
)

#: **The four preview helpers moved to `ui/preview.py` at task 17** and are re-exported
#: here, unchanged, because this screen still uses all four and every existing caller —
#: item 5, and this screen's own tests — names them here. The definitions are in one
#: place; the import path is two.
__all__ = [
    "SegmentationThresholdScreen", "LAYERS", "PANELS", "COEFFICIENT_RANGE",
    "COMMITTED_LAYERS", "COMMITTED_PANELS", "OBJECT_LAYER", "cell_objects",
    "middle_slice", "z_position", "volume_contrast_limits",
    "preview_mask", "commit_mask", "object_count",
]

#: §6g.1's table. **Two layers, both white, both at alpha 1.0** — and the 1.0 is a decision, not
#: an oversight: `palette.CHANNEL_ALPHA["Iba1"]` is 0.40, but that number is a **weight in an
#: additive sum**, and there is no sum here. The two layers sit in separate panels of one grid
#: and never overlap, so each is entitled to all of the light.
#:
#: **Greyscale IS the white colormap**, so §6f.4's *Iba1 is white* and §4's *"black background,
#: white surviving signal, following both parents"* are the same instruction, arrived at
#: independently — 3DMorph's `ThreshImg` and PrestoCell's binarised `x_img` on one side, the
#: channel convention on the other. **This screen has no colour picker and should not have one.**
#: The labels layer's name, and the readout's word. `n objects: 5` and a panel called
#: `Objects` are the same noun deliberately — the panel is what the number counts.
OBJECT_LAYER = "Objects"

LAYERS: tuple[tuple[str, str, float, bool], ...] = (
    ("Raw Iba1", "white", 1.0, True),
    ("Threshold preview", "white", 1.0, True),
)

#: **The panels change what they hold when the commit lands.** The user's decision, 2026-08-21:
#: *"after commit the image on the right changes to the new labels layer and the image on the
#: left becomes the one on the right."*
#:
#: > ### THE COMMITTED MASK WAS NEVER ON SCREEN, AND `n objects` COUNTED IT — task 26
#: >
#: > `draw` pushed `self.preview()` unconditionally, so both panels showed a **single-slice**
#: > `threshold_channel` no matter what had been committed. Meanwhile the readout came from
#: > `build_cell_mask` over the whole stack — hole-filled, closed and size-filtered. The number
#: > described something the screen could not show, and the difference between the two is
#: > exactly what those three operations do.
#: >
#: > **After the commit the pair is raw-threshold against finished-objects**, so that difference
#: > is what the two panels now carry.
#:
#: **No layer changes type**, which is what makes this safe on the one screen where layer order
#: is load-bearing: `Threshold preview` is an image in both modes, `Objects` is only ever
#: labels, and `Raw Iba1` only ever an image.
COMMITTED_LAYERS: tuple[tuple[str, str, float, bool], ...] = (
    ("Threshold preview", "white", 1.0, True),
    (OBJECT_LAYER, None, 1.0, True),
)

#: The header strip, and it is also the panel order. **Raw left, threshold right** — both parents
#: do it and left-to-right reads as before-to-after.
#:
#: > **This is the one screen where layer ORDER is load-bearing.** Everywhere else addition
#: > commutes, so *which layer is on top* changes nothing and the layer list is a reading order.
#: > Here napari's 1x2 grid assigns a panel **by layer index**, so pushing the preview first
#: > would silently swap the panels while every colour, alpha and header stayed correct.
PANELS: tuple[str, ...] = tuple(name for name, _, _, _ in LAYERS)
COMMITTED_PANELS: tuple[str, ...] = tuple(name for name, _, _, _ in COMMITTED_LAYERS)

#: §4: **0 to 3, and it is a judgement rather than a measurement.** PrestoCell's equivalent runs
#: 0 to 5 against a shipped `cell_coefficient` of 0.18 — 96% of the travel above the value in
#: use, which is §2b's own definition of a control dead across most of its range.
#: `cell_coefficient`'s usable range has never been swept; a sweep could narrow this further.
COEFFICIENT_RANGE = (0.0, 3.0)


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

def commit_mask(volume: np.ndarray, *, coefficient: float, min_object_voxels: int,
                voxel_size_um: tuple[float, float, float] | None,
                closing_radius_um: float) -> CellMask:
    """The expensive step — `build_cell_mask` over the whole stack, ~59 s at 40 x 2304 x 2304.

    **No `volume_contrast` passthrough, and that is correct**: this runs on the whole volume, so
    the value `threshold_channel` derives for itself is the value the preview was handed.
    """
    return build_cell_mask(volume, coefficient=coefficient,
                           min_object_voxels=min_object_voxels,
                           voxel_size_um=voxel_size_um,
                           closing_radius_um=closing_radius_um)


def cell_objects(filled_mask: np.ndarray):
    """The pipeline's own labelling of the committed mask — **labels, sizes and count.**

    `object_count` discarded everything but the number, and the labels are what the screen now
    draws. Same call, same connectivity, same hole-filled input: the picture and the readout
    cannot disagree because they are one result.

    **Cast to the narrowest integer that holds the labels.** `ndimage.label` returns int32, and
    a measured field is 46 M voxels — 185 MB against 93 MB for int16, on a session already
    holding four channels. Falls back to int32 above 32767 objects, which no real field of this
    study approaches.
    """
    objects = find_cell_objects(filled_mask)
    if objects.n_objects and objects.n_objects <= np.iinfo(np.int16).max:
        objects.labels = objects.labels.astype(np.int16, copy=False)
    return objects


def object_count(filled_mask: np.ndarray) -> int:
    """The screen's one readout, through the pipeline's **own** labelling.

    `find_cell_objects` is what `pipeline.py:418` counts with, on the **hole-filled** mask, at
    26-connectivity. Reusing it rather than counting here is the point: the number on screen is
    the number the pipeline will find, not a second opinion about it.

    **It is the diagnostic this screen needs.** A *falling* count as the coefficient drops is
    the signature of cells fusing — one measured field ran *"489 objects -> 537 cells"*, so a
    single object held roughly 49 microglia.
    """
    return find_cell_objects(filled_mask).n_objects


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

class SegmentationThresholdScreen(QWidget):
    """Frame A: one slider, two number boxes, a gated Commit, two panels, Continue below."""

    continue_requested = Signal()
    #: **Every screen but items 1, 9a and 7b, since 2026-08-30.**
    back_requested = Signal()

    #: Raw left, threshold right, at stride 1 — which is what makes PANELS' order
    #: load-bearing. The grid assigns a panel by layer INDEX.
    CANVAS = CanvasLayout(shape=(1, 2))

    #: `SegmentationParams`' own defaults, restated as the values the controls open with.
    #: `tests/test_ui_screen_threshold1.py` asserts each against the dataclass, so a pipeline
    #: default that changes without this changing goes red rather than drifting quietly.
    DEFAULT_COEFFICIENT = 0.18
    DEFAULT_MIN_OBJECT_VOXELS = "20"
    DEFAULT_CLOSING_RADIUS_UM = "0.8"

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.THRESHOLD_1)
        self.viewer = viewer

        self.volume: np.ndarray | None = None
        self.voxel_size_um: tuple[float, float, float] | None = None
        self.volume_contrast: float = 0.0
        self.contrast_limits: tuple[float, float] | None = None
        #: The committed labelling, or `None` while uncommitted. **What the right panel draws
        #: once a commit lands, and what `n objects` counts.** See `COMMITTED_LAYERS`.
        self.objects = None
        #: Which layer set is on the canvas, so a mode change can be told from a redraw.
        self._drawn_committed = False
        self.z = 0
        self.cache = SliceCache()

        self.frame = FrameA("Iba1 segmentation threshold", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: the three controls, Commit, the z indicator ----------
        self.coefficient = ParameterSlider(
            "cell_coefficient", COEFFICIENT_RANGE[0], COEFFICIENT_RANGE[1],
            self.DEFAULT_COEFFICIENT,
            help_button=HelpButton(
                "cell_coefficient",
                "How much Iba1 signal counts as cell, as a multiple of each slice's automatic "
                "threshold.\n\n"
                "LOWER  -> more signal included; cells reach further and may touch\n"
                "HIGHER -> less included; cells shrink and may break apart\n\n"
                "LIVE: the Threshold preview panel follows it on this slice. The object count "
                "and the finished mask arrive on COMMIT (about 59 s)."),
            parent=self)
        self.coefficient.value_changed.connect(self._on_coefficient_moved)
        #: **Recalculating on RELEASE, not on every step.** Dragging through twenty positions at
        #: 0.25 s each would queue five seconds of work behind the user's hand. The live value
        #: above the track still follows the handle, so the number is readable throughout.
        self.coefficient.slider.sliderReleased.connect(self.refresh_preview)
        self.frame.add_left(self.coefficient)

        self.min_object_voxels = self._number(
            "min_object_voxels", self.DEFAULT_MIN_OBJECT_VOXELS,
            "Drops connected blobs smaller than this many voxels, measured in 3D across the "
            "whole stack.\n\n"
            "Not previewed: a blob that looks small on one slice may be a long process in 3D."
            "\n\nTakes effect on COMMIT (about 59 s).")
        self.closing_radius_um = self._number(
            "closing_radius_um", self.DEFAULT_CLOSING_RADIUS_UM,
            "Bridges small gaps in the cell outline, then fills the hole left by the nucleus. "
            "Iba1 stains cytoplasm, so a cell body appears as a ring around an unstained "
            "nucleus; this makes it solid.\n\n"
            "Not previewed — part of it works across slices.\n\n"
            "Takes effect on COMMIT (about 59 s).")

        self.commit = CommitButton("Commit", self)
        self.commit.clicked.connect(self._on_commit)
        self.frame.add_left(self.commit)

        self._z_heading = self._heading(
            "z slice",
            "Which slice you are looking at, out of the whole stack. LEFT and RIGHT arrow keys "
            "move through z.")
        self.z_indicator = QLabel("", self)
        self.z_indicator.setProperty("role", "helper")
        self.frame.add_left(self.z_indicator)
        self.frame.left.addStretch(1)

        # -- centre: the header strip, then one canvas split 50/50 ------
        self.header = HeaderStrip(
            PANELS, "n objects: —",
            HelpButton("n objects",
                       "How many separate objects the finished mask contains, across the whole "
                       "stack.\n\nUpdates on COMMIT."),
            self)
        self.frame.add_centre(self.header)

        #: **One canvas, black, split 50/50, and the gap between the panels is invisible** —
        #: `GridCanvas` keeps both viewboxes inside a single camera and its `spacing` defaults
        #: to 0.0, so any gap shows the same black as the panels. §6f.9's *"the canvas separates
        #: itself from the chrome"* does not hold **between** two panels. The header strip above
        #: is what marks the boundary; no divider is drawn.
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        # -- the bottom strip, which is how bottom-right exists here -----
        self.navigation = NavigationRow(back=True, parent=self)
        self.navigation.back_clicked.connect(self.back_requested)
        self.navigation.continue_clicked.connect(self._on_continue)
        self.frame.add_to_strip(self.navigation)
        #: **The row owns the layout; `continue_button` stays as an alias**, so every `_sync`
        #: and every test that reads it keeps working. `NavigationRow` exists to stop ten copies
        #: of a two-button row drifting ten ways, not to rename what they already had.
        self.continue_button = self.navigation.continue_button

        self._sync()

    # -- construction ---------------------------------------------------

    def _heading(self, label: str, help_text: str) -> QLabel:
        """A control's name and its "?", on one row. §4: every adjustable control carries one."""
        name = QLabel(label, self)
        row = QHBoxLayout()
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(HelpButton(label, help_text, self))
        host = QWidget(self)
        host.setLayout(row)
        self.frame.add_left(host)
        return name

    def _number(self, label: str, value: str, help_text: str) -> QLineEdit:
        self._heading(label, help_text)
        edit = QLineEdit(value, self)
        edit.textChanged.connect(self._on_parameter_changed)
        self.frame.add_left(edit)
        return edit

    # -- loading --------------------------------------------------------

    def set_volume(self, volume: np.ndarray,
                   voxel_size_um: tuple[float, float, float] | None = None) -> None:
        """Take the Iba1 channel and open at the middle slice.

        **`volume_contrast` and the contrast limits are computed HERE, once**, and never again.
        The first is what keeps the single-slice preview agreeing with the commit; the second is
        §6g.4's fixed stretch. Deriving either per slice reintroduces exactly the divergence the
        other one exists to close.
        """
        if volume.ndim != 3:
            raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
        self.volume = volume
        self.voxel_size_um = voxel_size_um
        self.volume_contrast = slice_contrast(volume)
        self.contrast_limits = volume_contrast_limits(volume)
        self.cache.invalidate()
        self.set_z(middle_slice(volume.shape[0]))

    # -- z --------------------------------------------------------------

    def set_z(self, z: int) -> None:
        if self.volume is None:
            return
        self.z = max(0, min(int(z), self.volume.shape[0] - 1))
        self.z_indicator.setText(z_position(self.z, self.volume.shape[0]))
        self.draw()

    def bind_gestures(self) -> None:
        gestures.bind_z_stepping(self.viewer, self)

    def step_z(self, delta: int) -> None:
        """LEFT and RIGHT. **The user's decision, and the reason is the hardware**: this runs on
        a laptop, and a laptop user cannot reliably scroll.

        napari binds `increment_dims_left`/`right` to the arrow keys by default, so the canvas
        needs nothing built — but the indicator in the control column is ours and has to follow,
        which is what this method is for.
        """
        self.set_z(self.z + delta)

    def keyPressEvent(self, event):     # noqa: N802 - Qt's name
        if event.key() == Qt.Key.Key_Left:
            self.step_z(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.step_z(1)
        else:
            super().keyPressEvent(event)

    # -- the preview ----------------------------------------------------

    def preview(self) -> np.ndarray:
        """This slice's mask at this coefficient, cached per `(z, coefficient)`.

        **LRU rather than FIFO**, because the reason a preview is cached at all is that stepping
        *back* should be free — which is exactly what FIFO evicts first.
        """
        if self.volume is None:
            raise RuntimeError("no volume loaded; call set_volume first")
        coefficient = self.coefficient.value()
        key = cache_key(self.z, coefficient)
        return self.cache.get(key, lambda: preview_mask(
            self.volume, self.z, coefficient=coefficient,
            volume_contrast=self.volume_contrast))

    def refresh_preview(self) -> None:
        self.draw()

    def committed_view(self) -> bool:
        """Whether the canvas is showing the finished objects rather than the live preview."""
        return self.objects is not None

    def draw(self) -> None:
        """Push the two layers **in panel order**, for whichever pair the screen is showing.

        > ### THE LAYER LIST IS REBUILT ON A MODE CHANGE, NOT PATCHED — task 26
        >
        > napari assigns a grid panel **by layer index**, which this module's `PANELS` note
        > already calls the one place order is load-bearing. Entering the committed view is
        > safe either way: dropping `Raw Iba1` leaves `Threshold preview` at index 0 and
        > appending `Objects` lands at 1.
        >
        > **Leaving it is not.** Re-adding `Raw Iba1` appends it at index 1, which silently
        > swaps the panels while every colour, alpha and header stays correct — the exact
        > failure the note warns about. Clearing and repushing costs two 2D slices and removes
        > the ordering question rather than answering it once.
        """
        if self.viewer is None or self.volume is None:
            return

        committed = self.committed_view()
        if committed != self._drawn_committed:
            self.viewer.clear()
            self._drawn_committed = committed
            self.header.set_panels(COMMITTED_PANELS if committed else PANELS)

        #: The mask is binary, so its limits are its own range — not the raw channel's, which
        #: would render a 0/1 array as flat black.
        preview = self.preview().astype(np.float32)
        if not committed:
            data = {"Raw Iba1": self.volume[self.z], "Threshold preview": preview}
            limits = {"Raw Iba1": self.contrast_limits, "Threshold preview": (0.0, 1.0)}
            for name, colour, alpha, visible in LAYERS:
                self.viewer.show_image(name, data[name], colour=colour, alpha=alpha,
                                       visible=visible, contrast_limits=limits[name])
            return

        self.viewer.show_image("Threshold preview", preview, colour="white", alpha=1.0,
                               visible=True, contrast_limits=(0.0, 1.0))
        #: **One slice of the labels, in the same frame as everything else on this screen.**
        #: Every layer here is a 2D plane at `self.z`, so there is no napari slider to keep in
        #: step and the arrow keys are the whole z control.
        plane = np.asarray(self.objects.labels[self.z])
        ids = np.unique(plane)
        ids = ids[ids > 0]
        #: **Full strength for every object, and no muted class.** The user's decision,
        #: 2026-08-21 — item 4's two tiers separate microglial from not, and there is no such
        #: division here: everything drawn has been identified.
        self.viewer.show_labels(
            OBJECT_LAYER, plane,
            colours={int(i): ring_colour(int(i)) for i in ids},
            alphas={int(i): 1.0 for i in ids},
            visible=True)

    # -- parameters and the commit gate ---------------------------------

    def _on_coefficient_moved(self, value: float) -> None:
        """Moving the slider invalidates the commit **immediately**, at every step of the drag.

        The recompute waits for the release; the gate does not. A user who drags and never lets
        go still has a view that no longer matches what Continue would carry forward.
        """
        self._invalidate_view()

    def _on_parameter_changed(self) -> None:
        self._invalidate_view()

    def _invalidate_view(self) -> None:
        """Drop the commit, **and the committed panels with it.**

        > ### THE SLIDER HAS ITS OWN HANDLER, AND IT IS THE ONE MOST USED — task 26
        >
        > This was written into `_on_parameter_changed` alone, which serves the two text boxes.
        > `cell_coefficient` is a slider on `value_changed`, so **the control the user actually
        > drags reverted the gate and left the objects on screen** — a coloured panel claiming
        > to be the committed answer to parameters that had since moved. Found by driving it.
        >
        > One method, both callers, so there is no longer a live path and a forgotten one.

        The user's rule, 2026-08-21: *"if anything changes after commit is pressed it should
        drop back to being the type of screen it was."*

        **The readout deliberately does not revert with them** — see `set_object_count`. The
        count is a record of the last commit, and watching it fall as the coefficient drops is
        the fusion signature the readout exists for; the panels are a view of current state. The
        `!` beside Commit is what says the two describe different moments.
        """
        #: **Redraw only to swap the panels back, never merely because the gate moved.**
        #: `draw` calls `preview`, which is 0.25 s, and the slider invalidates on *every step of
        #: a drag* — §4's whole reason for recomputing on release rather than on movement. An
        #: unconditional redraw here put that cost back on every step, and
        #: `test_dragging_the_slider_does_not_recompute_but_releasing_does` caught it.
        was_committed = self.objects is not None
        self.objects = None
        self.flow.invalidate()
        self.commit.set_working("")
        if was_committed:
            self.draw()
        self._sync()

    def parameter_error(self) -> str | None:
        """What is wrong with the two number boxes, or `None`.

        Refusing the commit is the only honest option: both values reach `build_cell_mask`
        directly, and silently substituting a default would commit at settings the screen is not
        showing.
        """
        try:
            if int(self.min_object_voxels.text().strip()) < 0:
                return "min_object_voxels cannot be negative"
        except ValueError:
            return "min_object_voxels must be a whole number"
        try:
            if float(self.closing_radius_um.text().strip()) < 0:
                return "closing_radius_um cannot be negative"
        except ValueError:
            return "closing_radius_um must be a number"
        return None

    def _on_commit(self) -> None:
        problem = self.parameter_error()
        if problem is not None:
            self.commit.set_working(problem)
            return
        if self.volume is None:
            self.flow.commit(None)
            self._sync()
            return

        self.commit.set_working("Working…")
        try:
            mask = commit_mask(
                self.volume,
                coefficient=self.coefficient.value(),
                min_object_voxels=int(self.min_object_voxels.text().strip()),
                voxel_size_um=self.voxel_size_um,
                closing_radius_um=float(self.closing_radius_um.text().strip()))
            objects = cell_objects(mask.filled)
        finally:
            self.commit.set_working("")
        self.objects = objects
        self.flow.commit(mask)
        self.set_object_count(objects.n_objects)
        self.draw()
        self._sync()

    def set_object_count(self, count: int) -> None:
        """`n objects: 489`, right-aligned in the strip **above** the canvas.

        §6f.8, program-wide: no text is ever drawn over image data. §4 drew this number inside
        the right-hand panel; moving it out is the same correction item 2's `4.7%` got.

        **It survives a parameter change rather than blanking.** The number then describes the
        last commit and not the controls — which is what the exclamation beside Commit says, and
        keeping it is what lets a user watch the count *fall* as the coefficient drops. That
        fall is the fusion signature the readout exists for, and it cannot be seen one number
        at a time.
        """
        self.header.set_readout(f"n objects: {count}")

    def parameters(self) -> dict:
        """The `SegmentationParams` fields this screen owns, under **those** names.

        > ### NOTHING THE USER TYPED REACHED THE PIPELINE — task 24
        >
        > `session.chosen` is what `screen_save.session_params` serialises and what
        > `PipelineCompute` reads, and before this task **only items 9a and 9b ever wrote to
        > it.** Items 2, 3, 4, 5 and 6 held their values in `QLineEdit`s and nowhere else, so
        > every commit ran at `SegmentationParams`' defaults and the parameters file recorded
        > defaults — for a screen whose entire purpose is choosing the value.
        >
        > **A half-typed box contributes nothing rather than a zero.** `float("")` raises, and a
        > box mid-edit is not a choice; the commit gate refuses on the same values through
        > `parameter_error`, so a skipped field can never reach a commit either.

        Written by `_sync`, which every control change and every commit already calls.
        """
        chosen = {"cell_coefficient": float(self.coefficient.value())}
        chosen.update(parsed_numbers({
            "min_object_voxels": (self.min_object_voxels.text(), int),
            "closing_radius_um": (self.closing_radius_um.text(), float),
        }))
        return chosen

    def _sync(self) -> None:
        self.flow.session.chosen.update(self.parameters())
        self.commit.set_stale(self.flow.is_stale())
        self.continue_button.setEnabled(self.flow.can_continue())

    def _on_continue(self) -> None:
        if self.flow.can_continue():
            self.continue_requested.emit()
