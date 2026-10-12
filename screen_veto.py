"""Screen 2 — Plaque DAPI Core Veto. `COMPONENT_03_PLAN.md` §6f.10, §4 item 2.

Frame C. Owns `plaque_veto_coefficient` and **`plaque_dilate_um`** — the name in the code today,
by the user's decision of 2026-08-19, *"so the exports work as intended for the demo"*. §6f.10
draws it as `plaque_veto_margin_um`, which is request 17's rename and rides the v5 -> v6 bump.

**The veto mask is drawn as its 2 px BOUNDARY, not as a fill**, and that is what makes the screen
fit: no four of napari's usable colormaps are simultaneously visible on black and separable by a
dichromat — best 4-set ΔE 7.8 against 95.4 for the best 3. Drawing the boundary takes the
simultaneous-hue count from four to three.

It is also the better encoding regardless. The judgement is **containment**, and the veto is
`DAPI ∧ mask` (`nuclei.py:157`, `out[veto] = 0`), so **inside the line is removed and outside is
kept, by construction** — the redundant channel §6e.4 demands was already in the design.

    LEAK        blue inside the outline                      hue + position
    OVER-REACH  red with nucleus shape hard against the      hue + shape + position
                outline's edge
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from . import gestures, palette
from .frames import CanvasHost, FrameC
from .preview import middle_slice, z_position
from .semiauto import Flow, Session, Step, parsed_numbers
from .preview import contrast_limits, log_limits, log_view
from ..cache import WindowCache
from .viewer import (
    BATCHED, ROLLING, CanvasLayout, ViewerPort, cached_plane, outline_2d, step_window,
    window_bounds, z_window_range, z_window_word,
)
from .widgets import (
    ColourRow, CommitButton, HeaderStrip, HelpButton, LayerToggles, NavigationRow,
    batch_step_box,
)

__all__ = ["VetoScreen", "LAYERS", "TOGGLES", "LOG_DAPI", "removed_fraction"]

#: **A display transform, not a layer** — item 4's arrangement exactly. It redraws both DAPI
#: layers the way the detector sees them and adds nothing to the picker.
LOG_DAPI = "log DAPI"

#: §6f.6's table. Five picker rows against four toggles — `DAPI kept` and `DAPI removed` are two
#: halves of one toggle, because they are one channel under two states.
#:
#: **`vetoed DAPI` is OFF at arrival, and it is the same decision as additive blending.** Layer 4
#: is not *similar* to the removed half of layer 3 — it is the **identical voxels**. Two layers of
#: the same voxels in the same colour sum to double brightness, and a reviewer reads *brighter* as
#: *a different kind of thing*. It is the isolating view: untick `DAPI`, tick this, and look at
#: what is being removed with nothing competing.
LAYERS: tuple[tuple[str, str, float, bool], ...] = (
    ("6E10 raw", "magenta", palette.CHANNEL_ALPHA["6E10"], True),
    ("6E10 mask", "magenta", palette.DERIVED_MASK_ALPHA, True),
    ("DAPI", "blue", palette.CHANNEL_ALPHA["DAPI"], True),
    ("vetoed DAPI", "red", palette.CHANNEL_ALPHA["DAPI"], False),
)

#: Five toggles for four layers — the picker recolours, the toggle shows or hides, and the
#: fifth is a **display transform** rather than a layer of its own.
#:
#: > ### ⇒ `log DAPI` CAME TO THIS SCREEN LAST, AND IT WAS THE ONE THAT NEEDED IT MOST
#: >
#: > The user asked on 2026-08-30 whether item 2's DAPI was *"the log version, the one that the
#: > program sees"*. **It was not.** The background filter's right panel is log and item 4 has
#: > had this toggle since `m6`; item 2 was the only one of the three screens that show DAPI
#: > drawing it linear.
#: >
#: > **And it is the screen where the difference matters most.** What is being judged here is
#: > how much DAPI the veto takes at a plaque margin — dim, plaque-adjacent material, which is
#: > exactly what `log1p` lifts and a linear stretch buries.
#: >
#: > **Off at arrival**, as item 4's is: the veto has been judged against the linear channel up
#: > to now, and a screen that changed what it showed on opening would be answering a question
#: > nobody asked.
TOGGLES: tuple[tuple[str, bool], ...] = (
    ("6E10 raw", True), ("6E10 mask", True), ("DAPI", True), ("vetoed DAPI", False),
    (LOG_DAPI, False),
)


def removed_fraction(dapi: np.ndarray, veto: np.ndarray) -> float:
    """The readout: what share of total DAPI **intensity** the veto removes.

    Intensity rather than voxel count, matching `nuclei.veto_plaque_signal`'s own second return
    value — which is where this number comes from in the real flow rather than being recomputed.
    """
    total = float(dapi.sum())
    return 0.0 if total <= 0 else float(dapi[veto.astype(bool)].sum()) / total


#: Q2's rule is *progress shown, no cancel*. ~17 s measured on a real field.
COMMIT_PROGRESS = "Applying the veto…"


class VetoScreen(QWidget):
    """Frame C, five pickers, four toggles, two number boxes and a gated Commit."""

    continue_requested = Signal()
    #: **Every screen but items 1, 9a and 7b, since 2026-08-30.**
    back_requested = Signal()
    #: **The COMMIT's work, which this screen cannot run itself.** It needs the pipeline,
    #: and a screen importing `detect_nuclei` or `ImageSource` would break §2's first
    #: constraint. `app.SemiAutoApp.run_commit` is the one listener. Emitted **after** the
    #: gate flips, because the gate answers *have the parameters moved since the last
    #: commit*, which the press itself settles.
    commit_requested = Signal()

    #: One overlaid view. Five layers, all in the same panel.
    CANVAS = CanvasLayout()

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.VETO)
        self.viewer = viewer

        #: > ### THE ONE VIEWER SCREEN WITH NO Z - closed at task 25
        #: >
        #: > `draw` has taken `z` and `depth` since task 10 and **the driver always passed the
        #: > defaults**, so the z-batch box changed nothing and the `z slice` label read the
        #: > words `z slice` with no number after them. Every other screen with a canvas holds
        #: > its own slice; this one drew slice 0 of the stack, forever.
        self.z = 0
        self.depth = 1
        #: Rolling until the user ticks `Step by batch`. The two modes agree at depth 1, so the
        #: toggle can never move the picture on its own.
        self.batch_mode = ROLLING
        #: **Per-channel display range from item 1**, keyed by role; empty means the
        #: percentile stretch. Set by the flow driver on arrival — see `app.prepare`.
        self.display_range: dict[str, tuple[float, float]] = {}
        #: Whether a first draw has chosen the middle slice yet.
        self._z_started = False
        #: The three volumes, held rather than passed to every `draw` - items 4, 5 and 9b all
        #: keep theirs for the same reason: a z step must redraw without being handed the data
        #: again.
        self.dapi = None
        self.veto = None
        self.plaque = None
        #: **The readout is a whole-field number, so it is computed per VETO, not per draw.**
        #: `removed_fraction` is 1.95 s on a production field and a z step cannot move it — the
        #: share is of the total DAPI intensity in the volume, not in the window. These hold the
        #: two arrays it was last computed from, compared by identity; they are the same objects
        #: the screen already keeps above, so nothing extra is kept alive.
        self._readout_dapi = None
        self._readout_veto = None
        self._readout_share = 0.0
        #: **The window cache — item A9.** Four layers a position, so revisiting one is free
        #: while a fresh one still costs its projection. Entries are dropped per layer when the
        #: array behind them is replaced, so a COMMIT's new veto cannot be served the old one's.
        self.windows = WindowCache()

        self.frame = FrameC("plaque DAPI core veto", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: Colours, then Layers ---------------------------------
        self.frame.section("Colours", HelpButton(
            "Colours",
            "Every picker offers the same ten colours on every screen, so a colour means one "
            "thing throughout. The slider beneath each row is a weight: layers add rather than "
            "cover each other, so it sets how much of the sum that layer contributes. Takes "
            "effect immediately."))

        self.rows: dict[str, ColourRow] = {}
        for name, colour, alpha, _ in LAYERS:
            row = ColourRow(name, colour, alpha, self)
            row.colour_changed.connect(lambda c, n=name: self._recolour(n, c))
            row.alpha_changed.connect(lambda a, n=name: self._reweight(n, a))
            self.frame.add_left(row)
            self.rows[name] = row

        self.frame.section("Layers", HelpButton(
            "Layers",
            "DAPI is the whole channel. vetoed DAPI is just the part the veto removes, and it "
            "starts off — turn it on to see exactly what is being taken. Turn DAPI off at the "
            "same time to see it with nothing competing.\n\n"
            "log DAPI redraws both DAPI layers the way the detector sees them. Faint nuclear "
            "signal at a plaque margin is easy to miss on the linear view and is exactly what "
            "this screen is judging.\n\n"
            "It changes the picture only. The readout below is measured on the linear channel "
            "either way."))
        self.toggles = LayerToggles(TOGGLES, self)
        self.toggles.toggled.connect(self._toggle)
        self.frame.add_left(self.toggles)
        self.frame.left.addStretch(1)

        # -- centre: readout strip above the canvas ---------------------
        self.header = HeaderStrip(
            [], "DAPI removed by the veto  —",
            HelpButton("DAPI removed by the veto",
                       "What share of the total DAPI signal the veto takes out, measured as "
                       "intensity rather than as a count of voxels. Updates on COMMIT."),
            self)
        self.frame.add_centre(self.header)
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        # -- right: the two parameters, commit, z, continue -------------
        self.coefficient = self._number(
            "plaque_veto_coefficient", "1.0",
            "Multiplies each slice's Otsu level on 6E10 to find plaque. Evidence-backed at 1.0: "
            "recall 97% with the veto against 94% without. Takes effect on COMMIT.")
        self.margin = self._number(
            "plaque_dilate_um", "0.0",
            "A signed margin on the veto, in micrometres, applied in xy only.\n\n"
            "POSITIVE widens it beyond the antibody signal, because plaque-associated DAPI "
            "often extends a little past the 6E10.\n\n"
            "NEGATIVE erodes it inward, toward the dense core — which is where the "
            "DAPI-positive material actually sits, and it spares the halo around the core "
            "where microglia cluster. Measured on a study field, mean DAPI rises 4.7x from the "
            "rim of the mask to 1-2 um inside it, while the 6E10 signal itself stays flat.\n\n"
            "Eroding far enough removes small objects from the veto entirely. That is the "
            "control working: something too small to survive erosion has no dense core, so "
            "there is nothing there to veto.\n\n"
            "Takes effect on COMMIT.")

        self.commit = CommitButton("Commit", self)
        self.commit.clicked.connect(self._on_commit)
        self.frame.add_right(self.commit)

        self.z_depth = self._number(
            "z-batch depth", "1",
            "How many slices the view combines. Above 1 the mask is projected and then outlined "
            "— never the reverse, which would stack boundaries into nested rings. View only.")
        #: **On ENTER or when focus leaves, never per keystroke.** The user's
        #: decision, 2026-08-21. A z-batch is a whole re-projection of every layer,
        #: and applying one to each character of `10` means projecting at 1 on the
        #: way past — which is a dimensionality change of every layer, twice.
        self.z_depth.editingFinished.connect(self._on_depth_changed)
        self.batch_step = batch_step_box(self)
        self.batch_step.toggled.connect(self._on_batch_mode_changed)
        self.frame.add_right(self.batch_step)
        self.z_slice = QLabel("z slice", self)
        self.z_slice.setProperty("role", "helper")
        self.frame.add_right(self.z_slice)
        self.z_indicator = QLabel("", self)
        self.z_indicator.setProperty("role", "helper")
        self.frame.add_right(self.z_indicator)

        self.frame.right.addStretch(1)
        self.navigation = NavigationRow(back=True, parent=self)
        self.navigation.back_clicked.connect(self.back_requested)
        self.navigation.continue_clicked.connect(self._on_continue)
        self.frame.add_right(self.navigation)
        #: **The row owns the layout; `continue_button` stays as an alias**, so every `_sync`
        #: and every test that reads it keeps working. `NavigationRow` exists to stop ten copies
        #: of a two-button row drifting ten ways, not to rename what they already had.
        self.continue_button = self.navigation.continue_button

        self._sync()

    # -- construction ---------------------------------------------------

    def _number(self, label: str, value: str, help_text: str) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addStretch(1)
        row.addWidget(HelpButton(label, help_text))
        host = QWidget(self)
        host.setLayout(row)
        self.frame.add_right(host)

        edit = QLineEdit(value, self)
        edit.textChanged.connect(self._on_parameter_changed)
        self.frame.add_right(edit)
        return edit

    # -- behaviour ------------------------------------------------------

    def _recolour(self, layer: str, colour: str) -> None:
        if self.viewer is not None and layer in self.viewer.layer_names():
            self.viewer.set_colour(layer, colour)

    def _reweight(self, layer: str, alpha: float) -> None:
        if self.viewer is not None and layer in self.viewer.layer_names():
            self.viewer.set_alpha(layer, alpha)

    def _toggle(self, name: str, on: bool) -> None:
        """One toggle, one layer — **except `log DAPI`, which is a redraw.**

        **`DAPI` used to drive two**, back when the channel was drawn as a kept half and a
        removed half; it is one layer now and the mapping is the identity. `log DAPI` is item
        4's arrangement: it changes both DAPI layers' data *and* their limits together, which
        only a redraw can do — `set_visible` on a layer that does not exist would be silent.
        """
        if name == LOG_DAPI:
            self.redraw()
            return
        if self.viewer is not None and name in self.viewer.layer_names():
            self.viewer.set_visible(name, on)

    def _on_parameter_changed(self) -> None:
        """Both parameters take effect on commit, so moving one makes the view stale."""
        self.flow.invalidate()
        self._sync()

    def _on_commit(self, product=None) -> None:
        """Flip the gate, then ask for the ~17 s the button names.

        `product` stays in the signature because a caller can hand the mask straight
        in — the tests do, and so would a route that had already computed it.
        """
        self.flow.commit(product)
        self._sync()
        if product is None:
            self.commit.set_working(COMMIT_PROGRESS)
            try:
                self.commit_requested.emit()
            finally:
                self.commit.set_working("")

    def set_readout(self, fraction: float) -> None:
        self.header.set_readout(f"DAPI removed by the veto  {fraction:.1%}")

    def set_z(self, z: int) -> None:
        """**Absolute**, because napari's own z slider reports a slice rather than a step."""
        if self.dapi is None:
            return
        n = int(self.dapi.shape[0])
        self.z = max(0, min(int(z), n - 1))
        self.z_slice.setText(z_window_word(self.depth))
        self.z_indicator.setText(z_window_range(self.z, n, self.depth, self.batch_mode))
        #: **No `viewer.set_z` here any more.** That existed because the channels were whole
        #: volumes and napari picked the slice; every layer is now a plane computed at
        #: `self.z`, so there is no second place holding a z to keep in step. Item 4 still
        #: pushes volumes and still needs it.
        self.redraw()

    def step_z(self, delta: int) -> None:
        """**The stride is the toggle**: one slice rolling, a whole batch when `Step by batch`
        is ticked. `step_window` also stops where the window last fills, so rolling cannot walk
        into the tail where the projection thins while the label still claims a full batch."""
        if self.dapi is None:
            return
        self.set_z(step_window(self.z, delta, int(self.dapi.shape[0]),
                               self.depth, self.batch_mode))

    def _on_batch_mode_changed(self, batched: bool) -> None:
        """Re-entering `set_z` is what snaps the window: batched rounds the current slice down
        to its batch, and the indicator follows."""
        self.batch_mode = BATCHED if batched else ROLLING
        self.set_z(self.z)

    def bind_gestures(self) -> None:
        gestures.bind_z_stepping(self.viewer, self)

    def keyPressEvent(self, event):     # noqa: N802 - Qt's name
        """LEFT and RIGHT, for when the **controls** have focus. Once the user clicks the image
        napari's canvas owns the keyboard, and `bind_gestures` is what answers there."""
        if event.key() == Qt.Key.Key_Left:
            self.step_z(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.step_z(1)
        else:
            super().keyPressEvent(event)

    def _on_depth_changed(self) -> None:
        """**Above 1 the mask is projected and then outlined** — never the reverse, which would
        stack boundaries into nested rings. `outline_of_batch` does it in that order."""
        try:
            self.depth = max(1, int(self.z_depth.text().strip()))
        except ValueError:
            return
        #: **Meaningless at depth 1** — both modes show the same single slice — so the control
        #: says so rather than sitting there doing nothing.
        self.batch_step.setEnabled(self.depth > 1)
        self.set_z(self.z)

    def redraw(self) -> None:
        """Redraw at the current z from what was last drawn. **Nothing whole-field recomputes.**

        > ### THIS SAID *"NOTHING RECOMPUTES"* AND IT WAS FALSE — patch item A2, 2026-09-03
        >
        > A z step is a view change, and the veto behind it is whatever the last COMMIT
        > produced — but `draw` was rebuilding the whole field on every arrow key to show ten
        > slices of it: `veto.astype(bool)` **0.24 s**, `np.where(solid, dapi, 0)` **2.74 s**,
        > two `log_view` calls at **9.72 s each** with `log DAPI` on, and the readout at
        > **1.95 s** — roughly **3-4 GB of transient arrays per keypress**.
        >
        > **The veto itself still never recomputes**, which is what the sentence was reaching
        > for. What is left per step is the window: `depth` slices of `np.where` and a max.
        """
        if self.dapi is not None and self.veto is not None and self.plaque is not None:
            self.draw(self.dapi, self.veto, self.plaque, self.z, self.depth)

    def draw(self, dapi: np.ndarray, veto: np.ndarray, plaque: np.ndarray,
             z: int | None = None, depth: int | None = None) -> None:
        """Push the five layers, with the mask as an outline.

        **Project then outline** at depth above 1 — §6f.6's rule, and outlining first would
        stack ten slices' boundaries into nested rings. Identical at depth 1, so there is one
        path rather than two.
        """
        self.dapi, self.veto, self.plaque = dapi, veto, plaque
        if z is None:
            z = self.z
        if depth is None:
            depth = self.depth
        #: **The middle slice on arrival**, as items 3, 5, 9a and 9b all open. Slice 0 of a
        #: 40-slice stack is the edge of the tissue and is routinely empty.
        if not self._z_started:
            self._z_started = True
            self.z = middle_slice(int(np.asarray(dapi).shape[0]))
            z = self.z
        self.z = max(0, min(int(z), int(np.asarray(dapi).shape[0]) - 1))
        self.depth = max(1, int(depth))
        #: **`draw` writes the indicator too, because `app.py` calls it directly** without
        #: going through `set_z`. It must use the same window-aware pair, or a direct draw
        #: silently reverts the label to a single index — which is exactly what it did until a
        #: live check caught it, with all 839 tests passing either way.
        n_slices = int(np.asarray(dapi).shape[0])
        self.z_slice.setText(z_window_word(self.depth))
        self.z_indicator.setText(z_window_range(self.z, n_slices, self.depth, self.batch_mode))
        if self.viewer is None:
            return
        #: ### ⇒ THE WINDOW IS SLICED FIRST, AND EVERYTHING ELSE HAPPENS ON IT — item A2
        #:
        #: Item 4's rule, *"after the projection, not before … 0.4 MB against 850 MB"*, with
        #: item 9b as the reference implementation. This screen ignored it and paid for the
        #: whole field on every arrow key to draw ten slices of it.
        #:
        #: **Every step here is an identity, not an approximation**, which is what makes it a
        #: cost change and not a picture change:
        #:
        #:     np.where is elementwise            windowing before it cannot alter a voxel
        #:     log1p and clip are monotone        log(max(w)) is max(log(w)), exactly
        #:     any() over a numeric window        is astype(bool) then any(), exactly
        #:
        #: The contrast limits below are deliberately **not** windowed: they come from the whole
        #: volume, so stepping z does not restretch the picture underneath the judgement.
        start, stop = window_bounds(self.z, n_slices, self.depth, self.batch_mode)
        window = slice(start, stop)
        dapi_window = np.asarray(dapi)[window]
        #: **DAPI is one layer, not two halves** — the user's decision, 2026-08-20: *"keep the
        #: DAPI layer as DAPI, no need to split kept or not. DAPI veto layer can be turned on
        #: to get the exact effect."* `DAPI removed` and `vetoed DAPI` were the same array
        #: drawn twice under different toggles, so the split cost a layer and a colour to say
        #: what one toggle says.
        #: **Each of the four goes through the window cache — item A9.** Stepping back to a
        #: position already visited is then free; a fresh one still costs its ~80 ms. The two
        #: raw channels are plain max projections, so `projection=True` lets **batched** mode
        #: serve them from `ProjectionCache`, which precomputes every batch on a depth change.
        #: `vetoed DAPI` names **both** arrays it derives from, so a new veto drops it alone.
        def _plane(name, source, build, *, projection=False):
            return cached_plane(self.windows, name, source, self.z, self.depth,
                                self.batch_mode, build, projection=projection)

        removed_plane = _plane(
            "vetoed DAPI", (dapi, veto),
            lambda: np.where(np.asarray(veto)[window].astype(bool),
                             np.asarray(dapi)[window], 0).max(axis=0))
        dapi_plane = _plane("DAPI", dapi, lambda: dapi_window.max(axis=0), projection=True)
        plaque_plane = _plane("6E10 raw", plaque,
                              lambda: np.asarray(plaque)[window].max(axis=0), projection=True)
        line = _plane("6E10 mask", veto,
                      lambda: outline_2d(np.asarray(veto)[window].any(axis=0)))

        #: **Computed from the source volumes, once per draw, and shared by both DAPI
        #: layers.** `vetoed DAPI` is `dapi` with everything outside the veto zeroed, so
        #: stretching it on its own percentiles would give the two layers different ramps and
        #: make the overlay incomparable with what it sits on.
        #: > ### THE LOG VIEW KEEPS THE SHARED RAMP, AND THAT IS THE WHOLE POINT OF IT HERE
        #: >
        #: > The two DAPI layers already shared one set of limits, so the overlay stayed
        #: > comparable with what it sits on. **`log1p` is monotone and the limits come from the
        #: > raw percentiles**, so the pair is still one ramp in log space — and `log1p(0)` is
        #: > `0`, so `vetoed DAPI`'s zeroing outside the veto survives the transform exactly.
        #: >
        #: > The user's display range is **not** applied in log space: it is typed in raw
        #: > intensity units and means nothing there. Item 4's `log DAPI` made the same call for
        #: > the same reason.
        #: **The transform lands on the PLANE, the limits still come from the VOLUME.**
        #: `log1p` is monotone, so logging after the projection is the same picture for 0.4 MB
        #: instead of 850 MB; the limits are a property of the channel and are memoised per
        #: volume in `preview` since A1, so neither of these lines costs a percentile twice.
        logged = self.toggles.is_on(LOG_DAPI)
        dapi_view = log_view(dapi_plane) if logged else dapi_plane
        removed_view = log_view(removed_plane) if logged else removed_plane
        dapi_limits = (log_limits(dapi) if logged
                       else contrast_limits(dapi, self.display_range.get("nuclei")))

        limits = {
            "6E10 raw": contrast_limits(plaque, self.display_range.get("plaque")),
            "6E10 mask": (0.0, 1.0),
            "DAPI": dapi_limits,
        }
        limits["vetoed DAPI"] = limits["DAPI"]

        #: **Every layer that is not a label projects over the batch.** `6E10 mask` already
        #: did, through `outline_of_batch`; the three channels under it did not, so the box
        #: moved an outline across a picture frozen on one slice.
        #: > ### PLANES, NOT VOLUMES — SO THERE IS NO napari SLIDER. The user's rule, 2026-08-21
        #: >
        #: > *"dont provide a slider for z step, let left and right key be the only way to do it,
        #: > like the iba1 threshold screens."* Items 3 and 5 push a 2D plane per layer and have
        #: > never had a slider; this screen pushed whole volumes and let napari slice them, so
        #: > it grew one. `batch_plane` is `batch_projection` without the depth-1 exception — a
        #: > plane at every depth — and with no 3D layer left napari has no third axis to offer.
        #: >
        #: > **The arrow keys are now the only z control here**, which is why they had to be made
        #: > to work first: they are bound application-wide in `app.SemiAutoApp.eventFilter`
        #: > rather than depending on canvas focus.
        for name, data in (("6E10 raw", plaque_plane),
                           ("6E10 mask", line),
                           ("DAPI", dapi_view),
                           ("vetoed DAPI", removed_view)):
            row = self.rows[name]
            visible = self.toggles.is_on(name)
            self.viewer.show_image(name, data, colour=row.colour().name,
                                   alpha=row.alpha(), visible=visible,
                                   contrast_limits=limits[name])
        #: **Measured on the LINEAR channel, whatever is displayed.** The background filter's
        #: own note says why: *"log intensities do not add up to a total that would mean
        #: anything."* A share of total intensity is only a share in the units it was measured in.
        self.set_readout(self._removed_share(dapi, veto))

    def _removed_share(self, dapi, veto) -> float:
        """`removed_fraction` **once per veto**, never once per arrow key.

        A whole-field 1.95 s on a production field, and a z step cannot move it: the readout is
        a share of the total DAPI intensity in the *volume*. Compared by identity rather than by
        value — a new veto is a new array, and both arrays are ones the screen already holds, so
        remembering them keeps nothing alive that was not alive anyway.
        """
        if dapi is not self._readout_dapi or veto is not self._readout_veto:
            self._readout_share = removed_fraction(dapi, veto)
            self._readout_dapi, self._readout_veto = dapi, veto
        return self._readout_share

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
        return parsed_numbers({
            "plaque_veto_coefficient": (self.coefficient.text(), float),
            "plaque_dilate_um": (self.margin.text(), float),
        })

    def _sync(self) -> None:
        self.flow.session.chosen.update(self.parameters())
        self.commit.set_stale(self.flow.is_stale())
        self.continue_button.setEnabled(self.flow.can_continue())

    def _on_continue(self) -> None:
        if self.flow.can_continue():
            self.continue_requested.emit()
