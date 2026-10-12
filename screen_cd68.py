"""Screen 9a — CD68 Threshold. `COMPONENT_03_PLAN.md` §4 item 9, §6k.

**Frame C, and the screen is PURELY VISUAL.** No readout, no commit, no gate, no exclamation, no
header strip. *"You slide until the lit voxels look like CD68 puncta inside cells, and press
Continue."*

> ### STUDY-LEVEL, AND IT RUNS AFTER ITEM 7 RATHER THAN WITH THE PER-IMAGE SCREENS
>
> `cd68_threshold` is picked **once for the whole study** and applied unchanged to every image.
> Flow order and item numbering are decoupled on purpose: 9a and 9b precede item 8 because both
> pickers are `None` until item 9 runs, and writing the parameters file first would make it
> incomplete by design.

> ### NO COMMIT, AND THAT IS MEASURED RATHER THAN CHOSEN
>
> **The whole above-threshold volume is 460 ms** — 273 ms on a re-run — against 11 ms for one
> slice. Both are under Q2's line, so there is no reason to preview a single plane and no stale
> view for a gate to guard. `Step.CD68` is in `LIVE_SCREENS`, not `COMMIT_SCREENS`.
>
> **What that buys:** free z stepping and projection available at all times rather than only
> after a commit. **What it costs:** the slider recomputes on RELEASE, as item 3's does — 460 ms
> behind a dragging hand would queue.

**The judgement is CONTAINMENT** — is the orange sitting inside the white outlines? CD68 is
lysosomal, so its signal must lie within Iba1⁺ cytoplasm. §6k.3's rule, which item 9a is what
revealed: **the CONTAINING thing is an outline and the CONTAINED thing is a fill.** On items 2, 4
and 5 the derived mask happened to be the container; here it is the contained thing.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ..markers import measure_cd68
from . import gestures, palette
from .frames import CanvasHost, FrameC
from .preview import middle_slice, volume_contrast_limits, z_position
from .semiauto import Flow, Session, Step
from ..cache import WindowCache
from .viewer import (
    BATCHED, ROLLING, CanvasLayout, ViewerPort, cached_plane, outline_of_batch, step_window,
    window_bounds,
    z_window_range, z_window_word,
)
from .widgets import ColourRow, HelpButton, LayerToggles, ParameterSlider, batch_step_box

__all__ = [
    "Cd68ThresholdScreen", "LAYERS", "CONCEPTS", "PANELS", "LAYER_ORDER", "GRID_STRIDE",
    "TOGGLES", "POSITIVE_LAYER",
    "cd68_positive", "slider_range", "batch_projection", "z_label", "SLIDER_STEPS",
]

#: The thresholded preview's layer name. **`>` and not `≥`, and that is a correction to the
#: drawing** — see `cd68_positive` below.
POSITIVE_LAYER = "CD68 > T"

#: **Item 5's Iba1 measurement mask** — the user's correction, 2026-08-30: not the final
#: cell set. It is the mask every CD68 volume is measured inside, so it is the outline a
#: CD68 threshold should be judged against.
MASK_LAYER = "measurement mask"

#: > ### ⇒ REMOVED AND THEN PUT BACK, SAME DAY — 2026-08-30
#: >
#: > Dropped at the user's request — *"I only need 2 layers on this screen"* — and restored
#: > once the two-panel screen was in front of them: *"add the cell mask outline back as a
#: > layer as well. Make the outline colour yellow."*
#: >
#: > **Yellow, not the white it had.** White is Iba1's hue and Iba1 is now a layer on this
#: > screen in its own right, so the outline needed a colour that is not another layer's.
#: > §6k.3 is unchanged: *the containing thing is an outline and the contained thing is a fill.*

#: § — **TWO PANELS, on item 5's pattern.** Rebuilt 2026-08-30 at the user's request: *"I want it
#: to be like the measurement threshold screen... one layer (same on both images) Iba1 raw,
#: another layer is the CD68 — on the left image it's raw, on the right image it's the CD68 post
#: threshold."*
#:
#: **The comparison is the screen.** A single panel asked the user to remember what the raw CD68
#: looked like while judging the thresholded version on top of it; side by side, the threshold's
#: effect is a difference between two pictures rather than a memory.
PANELS: tuple[str, str] = ("CD68 raw", "CD68 above threshold")

#: The concepts the user picks colour and translucency for — **one row driving BOTH panels**,
#: exactly as item 5 does.
#:
#: > ### `CD68 > T` LOST ITS OWN COLOUR, AND THE REASON IT HAD ONE IS GONE
#: >
#: > It was orange against CD68's green because both lay in **one** panel and a state of a
#: > channel takes its own colour — §6k.4, recorded as the weakest colour pair of that session at
#: > 20.4 against green. **Panel position now carries that distinction**, so the two are one
#: > concept in one colour, and the marginal pair is retired rather than defended.
CONCEPTS: tuple[tuple[str, str, float, bool], ...] = (
    ("Iba1 raw", "white", palette.CHANNEL_ALPHA["Iba1"], True),
    ("CD68", "green", palette.CHANNEL_ALPHA["CD68"], True),
    (MASK_LAYER, "yellow", palette.DERIVED_MASK_ALPHA, True),
)

#: Kept as the name the toggle bar and the colour rows are built from.
LAYERS = CONCEPTS
TOGGLES: tuple[tuple[str, bool], ...] = tuple((name, on) for name, _, _, on in CONCEPTS)

#: **Six napari layers, and the ORDER assigns the panels** — 0-2 left, 3-5 right, at `stride=3`.
#: Panel assignment is computed from `layers.index(layer)` and **not** from visibility, so hiding
#: a layer empties its half rather than reshuffling the panels; that is what lets one toggle drive
#: both copies without the geometry moving.
#:
#: **`CD68 raw` and `CD68 > T` keep their old names** because they are the same two things they
#: always were, now in separate panels rather than stacked.
LAYER_ORDER: tuple[str, ...] = (
    "Iba1 raw — raw", "CD68 raw", "measurement mask — raw",
    "Iba1 raw — thresholded", POSITIVE_LAYER, "measurement mask — thresholded",
)

GRID_STRIDE = 3

#: 1000 positions across three orders of magnitude is ~0.7% per step, which is finer than the
#: number box's own integer resolution anywhere above ~150.
SLIDER_STEPS = 1000


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

def cd68_positive(volume: np.ndarray, threshold: float) -> np.ndarray:
    """Which voxels are CD68-positive. **`>` , because that is what `measure_cd68` computes.**

    > ### ⇒ THE DRAWING SAYS `CD68 ≥ T` AND THE PIPELINE SAYS `>`. THE PIPELINE WINS.
    >
    > `markers.measure_cd68` is `positive = values > float(threshold)`, and every CD68 column in
    > `cells.csv` comes from that line. §6k and §4 both draw the layer as `CD68 ≥ T`.
    >
    > **On this channel the difference is real rather than pedantic.** CD68 arrives as integers
    > from a `uint16` source, and the threshold is *an absolute integer the user types* — so
    > `T = 2677` puts every voxel of value exactly 2677 on one side under the label and the
    > other side under the measurement. A screen that lit voxels the export will not count is
    > the *complete-looking but wrong* failure this project keeps rejecting.
    >
    > **So the layer is named `CD68 > T`.** The alternative — keep the drawn `≥` and change
    > `measure_cd68` — is one character too, but it would move a shipped measurement to suit a
    > label, and the label is the thing that was written casually. **Recorded rather than
    > asked**, under the standing rule that the number on screen is the number the pipeline will
    > find. A test asserts the two agree on a voxel sitting exactly at T.
    """
    return np.asarray(volume) > float(threshold)


def slider_range(volume: np.ndarray, high_percentile: float = 99.9) -> tuple[float, float]:
    """**The log slider's ends: this image's own non-zero intensity range.**

    §4: *"the slider is log-scaled over this image's non-zero intensity range, with a number box
    carrying the exact absolute value."* The smallest non-zero value is the floor because
    `log(0)` has no answer and because zero is not a candidate threshold — most of the tissue
    *and* most of the inside of a cell has ~zero CD68.

    **The top is p99.9, not the max**, for §6g.4's reason in another form: one hot voxel would
    push the entire useful range into the bottom of a slider that exists to escape exactly that.

    A volume with no non-zero voxels returns `(1.0, 1.0)` rather than raising — an all-zero CD68
    channel is a real thing to be shown, and refusing to draw the screen would be a worse answer
    than drawing it with nothing lit.
    """
    values = np.asarray(volume)
    positive = values[values > 0]
    if positive.size == 0:
        return (1.0, 1.0)
    low = float(positive.min())
    high = float(np.percentile(positive, high_percentile))
    return (low, max(high, low))


def batch_projection(volume: np.ndarray, z: int, depth: int,
                     mode: str = ROLLING) -> np.ndarray:
    """The brightest value across a group of slices. **Depth 1 IS the single-slice view.**

    §4: the batch box at depth 1 has 40 positions, which is napari's native z slider, *"so the
    two controls collapse into one box and nothing is lost."* **There is no projection toggle
    and no single-slice mode on this screen.**

    **CD68 is the channel where this matters most** — it is punctate, which is why item 1
    max-projects its intake thumbnails, so a single plane is the least informative view of
    exactly this channel. And item 4's reason for making projection view-only does not reach
    here: **9a has no clicks at all**, so there is no missing z for a click to need.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
    start, stop = window_bounds(z, volume.shape[0], depth, mode)
    return volume[start:stop].max(axis=0)


def z_label(z: int, n_slices: int, depth: int, mode: str = ROLLING) -> str:
    """`z slice (23/40)` at depth 1, `z batch (23/40)` above it.

    The popup already says so — *"at z-batch depth above 1 this counts groups of slices instead
    and reads z batch"* — and a label that kept saying *slice* while showing ten would be the
    kind of quiet contradiction §6e.4 exists to prevent.
    """
    return f"{z_window_word(depth)} {z_window_range(z, n_slices, depth, mode)}"


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

class Cd68ThresholdScreen(QWidget):
    """Frame C: three pickers and three toggles left, one overlaid view, controls right."""

    continue_requested = Signal()

    #: One view, and no header strip above it — there is nothing to name.
    CANVAS = CanvasLayout(shape=(1, 2), stride=GRID_STRIDE)

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.CD68)
        self.viewer = viewer
        self.cd68: np.ndarray | None = None
        self.iba1: np.ndarray | None = None
        self.cells: np.ndarray | None = None
        self.contrast: tuple[float, float] | None = None
        self.z = 0
        self.depth = 1
        #: Rolling until the user ticks `Step by batch`. The two modes agree at depth 1, so
        #: the toggle can never move the picture on its own.
        self.batch_mode = ROLLING
        #: **The window cache — item A9.** Three planes a position: CD68, Iba1 and the cell
        #: outline. The thresholded layer is derived from the cached CD68 plane, so it costs a
        #: comparison over 2D rather than an entry of its own.
        self.windows = WindowCache()
        self.threshold = 0.0

        self.frame = FrameC("CD68 Threshold", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: three pickers, three toggles ---------------------------
        self.frame.section("Colours", HelpButton(
            "Colours",
            "Sets how each layer is drawn. Colour and translucency only.\n\n"
            "It changes nothing that is computed or saved.", self))

        self.rows: dict[str, ColourRow] = {}
        for name, colour, alpha, _ in LAYERS:
            row = ColourRow(name, colour, alpha, self)
            row.colour_changed.connect(lambda c, n=name: self._recolour(n, c))
            row.alpha_changed.connect(lambda a, n=name: self._reweight(n, a))
            self.frame.add_left(row)
            self.rows[name] = row

        self.frame.section("Layers", HelpButton(
            "Layers",
            "CD68 raw — the channel as imaged.\n\n"
            "CD68 > T — the voxels above the value you are choosing.\n\n"
            "cells — the cells you kept, after review and the size cutoff.", self))
        self.toggles = LayerToggles(TOGGLES, self)
        self.toggles.toggled.connect(self._toggle)
        self.frame.add_left(self.toggles)
        self.frame.left.addStretch(1)

        # -- centre: the canvas, and NO header strip ----------------------
        #: **The first screen with no header strip.** Items 2, 3, 4 and 6 put a readout there and
        #: item 5 carries panel names; 9a has one panel, so there is nothing to name, and §4
        #: removed its readouts entirely once they were measured at 39 ms and found unwanted.
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        # -- right: the log slider, the number box, z, and Continue -------
        self.slider = ParameterSlider(
            "cd68_threshold", 1.0, 2.0, 1.0, decimals=0, steps=SLIDER_STEPS, log=True,
            help_button=HelpButton(
                "cd68_threshold",
                "The intensity above which a voxel counts as CD68-positive.\n\n"
                "An ABSOLUTE value, not a multiplier. You pick it once here and it is applied "
                "unchanged to every image in the study.\n\n"
                "The slider is logarithmic: CD68 is punctate, so the useful range spans a wide "
                "span of low intensities.\n\n"
                "LIVE: the whole volume updates when you release the slider.", self),
            parent=self)
        self._wire_slider(self.slider)
        self.frame.add_right(self.slider)

        self.log_note = QLabel("LOG scale", self)
        self.log_note.setProperty("role", "helper")
        self.frame.add_right(self.log_note)

        #: > ### ⇒ THE ABSOLUTE VALUE BOX WAS REMOVED — 2026-08-30, at the user's request
        #: >
        #: > It read *"the BOX is authoritative, not the slider"*: 1000 log positions over
        #: > three orders of magnitude quantise to ~0.7%, so dragging to 2677 landed on 2681,
        #: > and typing set the value exactly. **With the box gone the slider is the only
        #: > writer, so `cd68_threshold` now carries whatever position the slider quantises
        #: > to** — and that number goes into a parameters file that is hashed and applied to
        #: > every image in the study.
        #: >
        #: > **CLOSED 2026-08-30 by storing what is displayed**, at the user's direction: *"if
        #: > the slider says whole numbers then the recorded value should be a whole number as
        #: > well."* `set_threshold` rounds through the slider's own formatter, so the
        #: > quantisation now sits on the glass rather than in the file. **It needed no control
        #: > to close** — which is why neither remedy offered at the time was the right one.

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("z-batch depth"))
        depth_row.addStretch(1)
        depth_row.addWidget(HelpButton(
            "z-batch depth",
            "How many z slices are combined into one view.\n\n"
            "  1  -> one slice at a time\n"
            "  10 -> the brightest value across each group of 10 slices\n\n"
            "Available at any time on this screen, as the whole volume is already computed.\n\n"
            "Changes only what you see. Immediate. LEFT and RIGHT move through z.", self))
        depth_host = QWidget(self)
        depth_host.setLayout(depth_row)
        self.frame.add_right(depth_host)

        self.depth_box = QLineEdit("1", self)
        self.depth_box.setFixedWidth(96)
        self.depth_box.editingFinished.connect(self._on_depth_edited)
        self.frame.add_right(self.depth_box)
        self.batch_step = batch_step_box(self)
        self.batch_step.toggled.connect(self._on_batch_mode_changed)
        self.frame.add_right(self.batch_step)

        z_row = QHBoxLayout()
        self.z_indicator = QLabel(z_label(0, 1, 1), self)
        self.z_indicator.setProperty("role", "helper")
        z_row.addWidget(self.z_indicator)
        z_row.addStretch(1)
        z_row.addWidget(HelpButton(
            "z slice",
            "Which slice you are looking at, out of the stack. LEFT and RIGHT arrow keys move "
            "through z.\n\n"
            "At z-batch depth above 1 this counts groups of slices instead and reads "
            "\"z batch\".", self))
        z_host = QWidget(self)
        z_host.setLayout(z_row)
        self.frame.add_right(z_host)

        self.frame.right.addStretch(1)

        #: **No commit, no `(!)`, no gate** — and no Back, consistent with items 5, 6 and 9b.
        self.continue_button = QPushButton("Continue", self)
        self.continue_button.clicked.connect(self._on_continue)
        self.frame.add_right(self.continue_button)
        self._sync()

    # -- loading --------------------------------------------------------

    def set_channels(self, cd68: np.ndarray, iba1: np.ndarray | None = None,
                     cells: np.ndarray | None = None,
                     threshold: float | None = None) -> None:
        """Take the CD68 channel and Iba1, and open at the middle slice.

        **The screen HOLDS its volumes** rather than taking them per `draw()` call — item 4's
        reason: `set_z` calls `draw`, and with the volumes passed in per call a z step silently
        drops a layer the first time a caller forgets one.

        ``iba1`` defaults to `None` so a caller that predates the two-panel rebuild still works.
        Its two layers are then drawn **empty rather than skipped** — see `draw`, where skipping
        one collapses both panels into the left half.
        """
        self.cd68 = np.asarray(cd68)
        self.iba1 = None if iba1 is None else np.asarray(iba1)
        self.cells = None if cells is None else np.asarray(cells).astype(bool)
        self.contrast = volume_contrast_limits(self.cd68)
        self.iba1_contrast = (None if self.iba1 is None
                              else volume_contrast_limits(self.iba1))
        self.z = middle_slice(self.cd68.shape[0])
        self.depth = 1
        self.depth_box.setText("1")

        low, high = slider_range(self.cd68)
        opening = float(threshold) if threshold is not None else low
        self.slider = self._rebuild_slider(low, high, opening)
        self.set_threshold(opening)

    def _rebuild_slider(self, low: float, high: float, value: float) -> ParameterSlider:
        """The ends come from the image, so the slider is rebuilt when the image arrives.

        **A slider whose range is data cannot be constructed before the data**, and item 5 met
        the same thing — its ceiling is a constructor argument for exactly this reason. Here the
        range is not merely a number the caller knows, it is two percentiles of the volume.
        """
        old = self.slider
        fresh = ParameterSlider("cd68_threshold", low, max(high, low * 1.000001), value,
                                decimals=0, steps=SLIDER_STEPS, log=True,
                                help_button=old.help_button, parent=self)
        self._wire_slider(fresh)
        index = self.frame.right.indexOf(old)
        self.frame.right.removeWidget(old)
        old.setParent(None)
        self.frame.right.insertWidget(max(0, index), fresh)
        return fresh

    def _wire_slider(self, slider: ParameterSlider) -> None:
        """**One place, because there are two sliders and they must behave identically.**

        The constructor builds a placeholder — the range is two percentiles of a volume that has
        not arrived — and `set_channels` replaces it. Wiring each site separately means the live
        one can quietly differ from the one the constructor tested, which is exactly what a
        mutation pass found: flipping the constructor's connection changed nothing observable
        because that slider is discarded before a user ever touches it.

        **RELEASE only.** §4: 460 ms behind a dragging hand would queue. There is no gate to
        invalidate at the first step of the drag, unlike item 3 — this screen has no commit.
        """
        slider.slider.sliderReleased.connect(self._on_slider_released)

    # -- the controls ---------------------------------------------------

    def set_threshold(self, value: float) -> None:
        """**The one number this screen exists to choose.** Live: it redraws immediately.

        **Stored as the slider displays it** — `ParameterSlider.as_displayed`. With the absolute
        box gone the slider is the only writer, and a label reading `2681` beside a parameters
        file holding `2681.4372` is a screen saying one thing while the hashed file records
        another.
        """
        self.threshold = self.slider.as_displayed(value)
        self.flow.session.chosen["cd68_threshold"] = self.threshold
        self.draw()

    def _on_slider_released(self) -> None:
        """§4: *"the slider recalculates on release"* — 460 ms behind a dragging hand would
        queue. **There is no gate to invalidate at the first step of the drag**, unlike item 3:
        this screen has no commit, so a drag simply does nothing until it ends.
        """
        self.set_threshold(self.slider.value())

    def _on_depth_edited(self) -> None:
        try:
            depth = int(float(self.depth_box.text()))
        except ValueError:
            self.depth_box.setText(str(self.depth))
            return
        self.set_depth(depth)

    def set_depth(self, depth: int) -> None:
        n = 1 if self.cd68 is None else self.cd68.shape[0]
        self.depth = max(1, min(int(depth), n))
        self.depth_box.setText(str(self.depth))
        self.batch_step.setEnabled(self.depth > 1)
        self.draw()

    def set_z(self, z: int) -> None:
        """**Absolute, because napari's z slider reports a slice rather than a step.**

        Items 3, 4 and 5 have carried this pair since they were built; 9a and 9b had only the
        step, so the shared binding in `gestures.py` could move them with an arrow and the
        slider under the canvas had nothing to call. One contract, seven screens.
        """
        if self.cd68 is None:
            return
        self.z = max(0, min(int(z), self.cd68.shape[0] - 1))
        self.draw()

    def step_z(self, delta: int) -> None:
        """LEFT and RIGHT. **The canvas is napari's own binding; the indicator is ours**, so
        this is the seam the canvas wiring connects to — item 3's arrangement exactly.
        """
        if self.cd68 is None:
            return
        self.set_z(step_window(self.z, delta, int(self.cd68.shape[0]),
                               self.depth, self.batch_mode))

    def _on_batch_mode_changed(self, batched: bool) -> None:
        self.batch_mode = BATCHED if batched else ROLLING
        self.set_z(self.z)

    def bind_gestures(self) -> None:
        gestures.bind_z_stepping(self.viewer, self)

    def keyPressEvent(self, event):         # noqa: N802 - Qt's name
        if event.key() == Qt.Key.Key_Left:
            self.step_z(-1)
            return
        if event.key() == Qt.Key.Key_Right:
            self.step_z(1)
            return
        super().keyPressEvent(event)

    # -- drawing --------------------------------------------------------

    def draw(self) -> None:
        """Push all four layers **in grid order**: 0-1 left, 2-3 right, at `stride=2`.

        **Iba1 is identical in both panels; only the CD68 differs** — raw on the left,
        above-threshold on the right. That is the whole point of the rebuild: the slider's
        effect is a difference between two pictures rather than something to be remembered.
        """
        if self.viewer is None or self.cd68 is None:
            return

        #: ⇒ **PROJECT, THEN THRESHOLD — patch item A7, 2026-09-03.** Item 9b's rule, and the
        #: same identity it rests on: `>` is monotone, so `max(w) > T` is the same set as
        #: `max(w > T)`. Thresholding the **whole volume** and projecting one window out of the
        #: result was **1.01 s**; thresholding the projected plane is **0.106 s**.
        #:
        #: At depth 1 `batch_projection` hands back the volume itself, so this is the same work
        #: it always was there — the saving is the whole point of the z-batch box being on.
        raw = cached_plane(
            self.windows, "cd68", self.cd68, self.z, self.depth, self.batch_mode,
            lambda: batch_projection(self.cd68, self.z, self.depth, self.batch_mode),
            projection=True)
        positive = cd68_positive(raw, self.threshold).astype(np.uint8)
        #: **A missing Iba1 becomes an empty plane rather than a skipped layer.**
        #:
        #: > ### ⇒ SKIPPING IT COLLAPSED THE PANELS — 2026-08-30
        #: >
        #: > napari assigns a panel from `layers.index(layer) // stride`, over the layers that
        #: > **actually exist**. Push only two of the four and both land in panel 0, leaving the
        #: > right half empty. An earlier version of this docstring claimed the panels line up
        #: > regardless because assignment is by index over `LAYER_ORDER`; that was wrong, and it
        #: > is the same trap §6i already records for visibility — except that hiding a layer
        #: > keeps it in the list and never creating it does not.
        plane_iba1 = (np.zeros_like(raw) if self.iba1 is None else cached_plane(
            self.windows, "iba1", self.iba1, self.z, self.depth, self.batch_mode,
            lambda: batch_projection(self.iba1, self.z, self.depth, self.batch_mode),
            projection=True))

        outline = (np.zeros_like(positive) if self.cells is None else cached_plane(
            self.windows, "cells", self.cells, self.z, self.depth, self.batch_mode,
            lambda: outline_of_batch(self.cells, self.z, self.depth, mode=self.batch_mode)))
        data = {
            LAYER_ORDER[0]: plane_iba1, LAYER_ORDER[1]: raw, LAYER_ORDER[2]: outline,
            LAYER_ORDER[3]: plane_iba1, LAYER_ORDER[4]: positive, LAYER_ORDER[5]: outline,
        }
        #: **Every layer carries explicit limits, and the binary one most of all.**
        #:
        #: > ### ⇒ THE THRESHOLDED LAYER WENT BLANK WITHOUT THIS — 2026-08-30
        #: >
        #: > napari **recomputes a layer's contrast limits whenever its data is replaced**, which
        #: > is every slider release and every z step. On a binary layer that is fine while both
        #: > 0 and 1 are present — and it collapses to `[0, 0]` the moment the threshold admits
        #: > nothing. With `contrast_limits=None` nothing ever sets them again, so the layer stays
        #: > blank after that even though the data underneath keeps changing correctly.
        #: >
        #: > **That is the reported defect in both its forms**: *"changing the threshold slider
        #: > doesn't change anything on the CD68 threshold layer"*, and then *"the right side of
        #: > the screen is not showing any CD68 signal."* Item 5 pins `(0.0, 1.0)` on its binary
        #: > layers for this reason; this screen never did.
        limits = {
            LAYER_ORDER[0]: self.iba1_contrast, LAYER_ORDER[3]: self.iba1_contrast,
            LAYER_ORDER[1]: self.contrast, LAYER_ORDER[4]: (0.0, 1.0),
            #: Binary, pinned for the same reason as the thresholded layer.
            LAYER_ORDER[2]: (0.0, 1.0), LAYER_ORDER[5]: (0.0, 1.0),
        }
        for index, name in enumerate(LAYER_ORDER):
            plane = data[name]
            if plane is None:
                continue
            concept = CONCEPTS[index % GRID_STRIDE][0]
            row = self.rows[concept]
            self.viewer.show_image(name, plane, colour=row.colour().name, alpha=row.alpha(),
                                   visible=self.toggles.is_on(concept),
                                   contrast_limits=limits.get(name))

        self.z_indicator.setText(
            z_label(self.z, self.cd68.shape[0], self.depth, self.batch_mode))

    def _layers_for(self, concept: str) -> tuple[str, str]:
        """The two napari layers one picker row drives — **the pair, not one copy.**"""
        index = [name for name, _, _, _ in CONCEPTS].index(concept)
        return LAYER_ORDER[index], LAYER_ORDER[index + GRID_STRIDE]

    def _recolour(self, concept: str, colour: str) -> None:
        for layer in self._layers_for(concept):
            if self.viewer is not None and layer in self.viewer.layer_names():
                self.viewer.set_colour(layer, colour)

    def _reweight(self, concept: str, alpha: float) -> None:
        for layer in self._layers_for(concept):
            if self.viewer is not None and layer in self.viewer.layer_names():
                self.viewer.set_alpha(layer, alpha)

    def _toggle(self, concept: str, on: bool) -> None:
        """Hiding a layer **empties its half rather than reshuffling the panels** — panel
        assignment is computed from the layer's index, not from its visibility."""
        for layer in self._layers_for(concept):
            if self.viewer is not None and layer in self.viewer.layer_names():
                self.viewer.set_visible(layer, on)

    # -- continue -------------------------------------------------------

    def _sync(self) -> None:
        """**No commit gate**, so Continue is live from arrival — item 6's arrangement."""
        self.continue_button.setEnabled(self.flow.can_continue())

    def _on_continue(self) -> None:
        if self.flow.can_continue():
            self.continue_requested.emit()
