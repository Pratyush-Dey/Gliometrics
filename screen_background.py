"""The DAPI Background Filter screen — between items 1 and 2. `SHAKEOUT_FINDINGS.md` M3.

**Frame C with a two-panel canvas, and item 5's committed/uncommitted rule.** It owns
`dapi_background_strength` and `dapi_background_scale_um`, and it decides **how much diffuse
DAPI reaches Cellpose** — never anything about nuclei, cells or plaques.

> ### THE SCREEN EXISTS BECAUSE THE PICTURE AND THE DETECTOR DISAGREE
>
> Detection runs on `nuclei.normalize_for_detection`'s log compression, which lifts the dim end
> hard. Measured on `program training crop2.tif`: DAPI at its 75th percentile renders at **0.1%
> grey** on a linear screen and is **26.9%** to the detector; at its 90th percentile, 2.9%
> against **70.8%**. **92.6% of the volume renders below 5% grey.** So diffuse background is
> nearly invisible to the person judging the result and bright to the thing being judged, and
> *"I cannot see DAPI there"* was never evidence about whether DAPI was there.

> ### THE TWO PANELS DIFFER BY THE TRANSFORM AND BY NOTHING ELSE — the user's rule, 2026-08-23
>
> Left linear, right log, **both carrying all three layers and both changing together**. That
> is item 5's structure taken literally — *"the panels must be in the same state or the screen
> flatters threshold 2"* — and here it is what makes the comparison say one thing: any
> difference between the halves is the transform, because nothing else differs.
>
> **The right panel is what Cellpose sees, and that is measured rather than asserted.**
> `log_view` is `log1p` and the limits are the raw 1st/98th percentiles, which `log1p`'s
> monotonicity carries into log space exactly — and napari's `contrast_limits` is the same
> affine rescale `normalize_for_detection` finishes with. Compared against what `detect_nuclei`
> actually hands its detector: **max difference 0.012 of the grey range, mean 0.003, nothing
> above 0.02.** The one real gap is position in the flow, not arithmetic — this screen sits
> *before* the plaque veto, so the 2.5% of voxels the veto will later zero (8.7% of DAPI
> intensity on the demo crop) are still lit here.

> ### THREE LAYERS, AND TWO OF THEM DO NOT EXIST UNTIL COMMIT — the user's rule, 2026-08-23
>
> `DAPI` is the channel as it arrived and is always there. **`DAPI kept` is what goes
> forward to the rest of the program and `DAPI removed` is what does not**, and neither
> means anything before the button is pressed — so both are greyed and carry nothing until
> then. Committing switches the picture to `DAPI kept`, which is *"what will go forward"*
> in the user's own words; typing in either box withdraws all three back to the channel.
>
> **The three are exhaustive by construction.** The filter is subtractive and clipped at
> zero, so `removed = min(v, k*blur)` and **`kept + removed = DAPI` at every voxel**. There
> is no binary mask anywhere on this screen, which is why the readout is a share of
> **intensity** — the same convention as item 2's `removed_fraction`, and the only one
> available here. It is computed on the **linear** channel; the log panel is display only.
>
> **Every effect here is commit-gated, which is why both parameters are boxes.** The
> strength was a slider while a live preview existed to justify one; with nothing to
> preview, a slider is a less exact way to type a number that reaches a hashed parameters
> file — item 9a's *"the number box is AUTHORITATIVE"*, reached from the other direction.

**No z-batch depth box, by the user's decision.** Every extra slice in the window is another
background estimate — the subtraction and a max-projection do not commute, so a batch cannot be
filtered as one plane — and at ~1.7 s a plane on a full field a depth of 10 would put fifteen
seconds behind every arrow key. LEFT and RIGHT step one slice, as on every other z screen.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ..pipeline import SegmentationParams
from . import gestures, palette
from .frames import CanvasHost, FrameC
from .preview import contrast_limits, log_limits, log_view, middle_slice
from .semiauto import Flow, Session, Step, parsed_numbers
from .viewer import CanvasLayout, ViewerPort
from .widgets import (
    ColourRow, CommitButton, HeaderStrip, HelpButton, LayerToggles, NavigationRow,
)

__all__ = [
    "BackgroundFilterScreen", "PANELS", "CONCEPTS", "LAYER_ORDER", "GRID_STRIDE",
    "COMMITTED_ONLY", "DEFAULT_SCALE_UM", "DEFAULT_STRENGTH",
    "log_view", "log_limits", "removed_fraction",
]

#: Both boxes open on **the dataclass's own defaults, read rather than retyped.** A screen
#: opening on a number `SegmentationParams` no longer holds is the drift this avoids.
#:
#: **0.0 is off**, and that is what a session which walks past this screen without touching it
#: computes — exactly what the program computed before the screen existed.
DEFAULT_STRENGTH: float = SegmentationParams().dapi_background_strength

#: The box opens on **the dataclass's own default, read rather than retyped.** A screen opening
#: on a number `SegmentationParams` no longer holds is the drift this avoids — item 6 records
#: the opposite case, where a box deliberately opens somewhere else and a test pins both values
#: side by side so the contrast is on the record.
DEFAULT_SCALE_UM: float = SegmentationParams().dapi_background_scale_um

#: §6i.5's rule — **the headers name what the panel is, not its order.** Here the panels differ
#: only by the display transform, so the transform is the honest name for each.
PANELS: tuple[str, str] = ("DAPI", "log DAPI")

#: The three the user thinks in — one picker row and one toggle each, **each driving both
#: panels**. `DAPI` is the channel as it arrived; `DAPI kept` is what goes forward; `DAPI
#: removed` is what does not. The three are exhaustive by construction: **kept + removed = DAPI
#: at every voxel**, because the filter subtracts an amount and clips at zero.
#:
#: **Blue and red are item 2's, not new ones.** §6f.4's rule is one hue one meaning everywhere:
#: DAPI is blue, and *removed* is red on the one other screen that removes DAPI. **Yellow for
#: `DAPI kept` is the user's choice, 2026-08-23** — it is a third *state* of one channel, and
#: item 2's precedent is that a state takes its own colour rather than the channel's.
CONCEPTS: tuple[tuple[str, str, float, bool], ...] = (
    ("DAPI", "blue", palette.CHANNEL_ALPHA["DAPI"], True),
    ("DAPI kept", "yellow", palette.CHANNEL_ALPHA["DAPI"], False),
    ("DAPI removed", "red", palette.CHANNEL_ALPHA["DAPI"], False),
)

#: **`DAPI kept` and `DAPI removed` do not exist until COMMIT** — the user's rule, 2026-08-23.
#: Before it there is no answer to *what survives*, only the channel, so both toggles are greyed
#: and the layers carry nothing. This replaces the live preview the screen opened with: every
#: effect on this screen is now commit-gated, which is also why the strength is a box.
COMMITTED_ONLY: tuple[str, ...] = ("DAPI kept", "DAPI removed")

#: **Six napari layers, and the ORDER is what assigns the panels.** `grid.shape=(1, 2)` with
#: `stride=3` places layers 0-2 in the left panel and 3-5 in the right; panel assignment is
#: computed from `layers.index(layer)` and **not** from visibility, so hiding a layer empties
#: part of a panel rather than reshuffling the panels — which is what lets one toggle drive both
#: copies without the geometry moving. Item 5's arrangement exactly.
#:
#: > ### THE COMMITTED PAIR IS PUSHED EMPTY RATHER THAN NOT PUSHED
#: >
#: > **Because the index is the panel.** Withholding two layers before the commit and adding
#: > them after would renumber every layer and move the split between the panels — the trap
#: > `LAYER_ORDER` exists to name. An all-zero layer contributes nothing under additive
#: > blending, so it is invisible without being absent, and the geometry never moves.
LAYER_ORDER: tuple[str, ...] = (
    "DAPI — linear", "DAPI kept — linear", "DAPI removed — linear",
    "DAPI — log", "DAPI kept — log", "DAPI removed — log",
)

GRID_STRIDE = 3

#: Q2's rule is *progress shown, no cancel*. ~1 s on the demo crop, **~67 s on a full field** —
#: see `nuclei.HALF_RES_SIGMA_PX` for where that number comes from and what it replaced.
COMMIT_PROGRESS = "Filtering the DAPI channel…"

#: The strength's useful span, quoted in its `?` popup. **Advisory, not enforced** — it is a
#: box, so the only value it refuses is a negative one.
#:
#: > ### IT WAS A SLIDER FOR ONE AFTERNOON, AND THE USER MADE IT A BOX — 2026-08-23
#: >
#: > The slider was justified by a live preview: the blur is computed once per plane, so the
#: > multiplier costs nothing to re-apply. **The user then made the two derived layers appear
#: > only after COMMIT**, which removes the thing the liveness was for — with nothing previewing,
#: > a slider is a less exact way to type a number. *"the effect is commit gated regardless."*
#: >
#: > **It also ends the ceiling question.** The slider went 0-1, then 0-2, then 0-4 in one
#: > sitting, each time because the top was in the way; a box has no top. Item 9a already
#: > records the same conclusion from the other direction — *"the number box is AUTHORITATIVE"*,
#: > because 1000 slider positions quantise a value that reaches a hashed parameters file.
#:
#: > ### ABOVE ~0.4 THIS STOPS REMOVING BACKGROUND AND STARTS TRIMMING NUCLEI
#: >
#: > Swept on `program training crop2.tif` at scale 12 µm. `core` is the top 0.1% of raw DAPI,
#: > `dim` the 50th-90th percentile band, `background` everything below the median:
#: >
#: > | k | removed | core kept | dim kept | background kept |
#: > |---|---|---|---|---|
#: > | 0.4 | 11.7% | 99.0% | 37.1% | **0.00%** |
#: > | 0.6 | 15.9% | 98.5% | 24.4% | 0.00% |
#: > | 1.0 | 23.2% | 97.4% | 11.4% | 0.00% |
#: > | 1.5 | 30.8% | 96.2% | 5.1% | 0.00% |
#: > | 2.0 | 37.3% | 94.9% | 2.7% | 0.00% |
#: > | **4.0** | **55.6%** | **89.8%** | **0.5%** | 0.00% |
#: >
#: > **The background is entirely gone by 0.4.** Past that the control is eating the dim band,
#: > which is where genuine nucleus *periphery* lives — 35.9% of voxels inside a confident
#: > nucleus core render below 5% grey, and DAPI at its 90th percentile is 70.8% to the detector.
#: > **That is a use rather than a fault**: `SHAKEOUT_FINDINGS.md` O3 is that nucleus labels
#: > reach past visible DAPI, and tightening them toward their cores is one of the two things
#: > measured as fixing it. There is no cliff anywhere up to 5 — the cores hold 87% even there,
#: > so any ceiling is a judgement about the control rather than about the data.
#:


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

#: > ### ⇒ `log_view` AND `log_limits` MOVED TO `preview.py` ON 2026-08-30
#: >
#: > This module's copy said the duplication with `screen_nuclei` was deliberate and named the
#: > condition for ending it: *"task 13's rule is that a third caller earns a move to
#: > `preview.py`; there are two."* **Item 2's `log DAPI` toggle is the third**, so the rule
#: > fired on its own terms and both copies went at once.
#: >
#: > Re-exported here rather than repointed at every call site, because this screen's own tests
#: > and its panel code name them here, and the definitions are one import away.


def removed_fraction(raw: np.ndarray, surviving: np.ndarray) -> float:
    """The readout: what share of total DAPI **intensity** the filter takes out.

    Intensity rather than voxels, and here that is forced rather than chosen — the filter is
    subtractive, so there is no set of voxels it removes. It matches `screen_veto`'s own
    `removed_fraction`, which is the same number about the other thing that removes DAPI.
    """
    total = float(np.asarray(raw, dtype=np.float64).sum())
    if total <= 0:
        return 0.0
    kept = float(np.asarray(surviving, dtype=np.float64).sum())
    return max(0.0, min(1.0, (total - kept) / total))


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

class BackgroundFilterScreen(QWidget):
    """Frame C, two picker rows, two toggles, one live slider, one box and a gated Commit."""

    continue_requested = Signal()
    #: **Every screen but items 1, 9a and 7b, since 2026-08-30.**
    back_requested = Signal()
    #: **The COMMIT's work, which this screen cannot run itself.** The filter is cheap enough
    #: that the screen could call it — but the *product* is a whole channel that later screens
    #: and the detector read off the session, and only the driver holds that. `run_commit` is
    #: the one listener. Emitted **after** the gate flips, as every other commit screen does.
    commit_requested = Signal()

    #: Two panels, two layers each. §6g.2: one canvas, split 50/50, and the gap between the
    #: panels is invisible because both sides of it are the same black — the header strip is
    #: what marks the boundary.
    CANVAS = CanvasLayout(shape=(1, 2), stride=GRID_STRIDE)

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.BACKGROUND)
        self.viewer = viewer

        self.dapi: np.ndarray | None = None
        self.voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.z = 0
        #: Whether a first draw has chosen the middle slice yet.
        self._z_started = False
        #: **Per-channel display range from item 1**, keyed by role. It reaches the LINEAR panel
        #: only: the log panel's limits are in log intensity units and a range typed in raw ones
        #: means nothing there — item 4's `log DAPI` made the same call for the same reason.
        self.display_range: dict[str, tuple[float, float]] = {}

        #: > ### THE COMMITTED STATE, AND IT IS ITEM 5's FLAG RATHER THAN THE FLOW's
        #: >
        #: > The flow's gate answers *may the user continue*; this answers *what are the panels
        #: > showing*. **Before the commit both panels show the whole channel** with the removed
        #: > part previewed beside it; after it, the first layer is the surviving signal only.
        #: > Moving either control drops both panels back, in step with the exclamation
        #: > appearing beside Commit — because a surviving-signal picture computed at a strength
        #: > the slider has since left is the stale view the gate exists to refuse.
        self.is_committed = False
        #: The whole-volume filtered channel, as the last COMMIT produced it. `None` until then.
        self.filtered: np.ndarray | None = None

        self.frame = FrameC("DAPI background filter", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: Colours, then Layers ---------------------------------
        self.frame.section("Colours", HelpButton(
            "Colours",
            "Every picker offers the same ten colours on every screen, so a colour means one "
            "thing throughout. The slider beneath each row is a weight: layers add rather than "
            "cover each other, so it sets how much of the sum that layer contributes. Each row "
            "drives its layer in BOTH panels. Takes effect immediately."))

        self.rows: dict[str, ColourRow] = {}
        for name, colour, alpha, _ in CONCEPTS:
            row = ColourRow(name, colour, alpha, self)
            row.colour_changed.connect(lambda c, n=name: self._recolour(n, c))
            row.alpha_changed.connect(lambda a, n=name: self._reweight(n, a))
            self.frame.add_left(row)
            self.rows[name] = row

        self.frame.section("Layers", HelpButton(
            "Layers",
            "DAPI is the channel as it arrived, and it is always available.\n\n"
            "DAPI kept is what goes forward to the rest of the program. DAPI removed is what "
            "does not. Together they are exactly DAPI — every voxel is in both of them in "
            "some proportion, because the filter takes an amount off each one rather than "
            "deleting a set of them.\n\n"
            "Both are greyed out until you COMMIT, because until then there is no answer to "
            "what survives — only the channel."))
        self.toggles = LayerToggles(
            tuple((name, on) for name, _, _, on in CONCEPTS), self)
        self.toggles.toggled.connect(self._toggle)
        self.frame.add_left(self.toggles)

        #: **The disabled state carried twice** — the greyed boxes for everyone, and this line
        #: for anyone who cannot separate the two greys. §6e.4, and item 5's z-batch box set the
        #: precedent including hiding the sentence once it stops being true.
        self.layers_note = QLabel("(kept and removed appear after Commit)", self)
        self.layers_note.setProperty("role", "helper")
        self.layers_note.setWordWrap(True)
        self.frame.add_left(self.layers_note)
        self.frame.left.addStretch(1)

        # -- centre: the two panel names and the readout -----------------
        self.header = HeaderStrip(
            PANELS, "DAPI removed  —",
            HelpButton("DAPI removed",
                       "What share of the total DAPI signal the filter takes out.\n\n"
                       "Measured on the LINEAR channel — the raw intensities, the numbers in "
                       "the file. The log panel is a display transform and nothing else; no "
                       "figure on this screen is computed from it, and one that was would not "
                       "mean anything, because log intensities do not add up to a total that "
                       "a share can be taken of.\n\n"
                       "It is a share of intensity rather than a count of voxels, because the "
                       "filter subtracts an amount from every voxel rather than deleting a set "
                       "of them — there is no count to give.\n\n"
                       "Whole-field, over the entire stack rather than the slice on screen, so "
                       "it updates on COMMIT."),
            self)
        self.frame.add_centre(self.header)
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        # -- right: the two boxes, commit, z, continue -------------------
        self.strength = self._number(
            "dapi_background_strength",
            f"{DEFAULT_STRENGTH:g}",
            "How much of the estimated background is subtracted.\n\n"
            "0 is off — nothing is removed and the channel goes forward untouched. At 1 the "
            "whole estimate is taken away. Useful values run about 0 to 4, but the box does "
            "not stop there; only a negative value is refused.\n\n"
            "Measured on a study crop at the default scale, the background is entirely gone "
            "by about 0.4. Past that this is no longer removing background — it is trimming "
            "the dim edges of the nuclei themselves, which can be what you want, because "
            "nucleus labels tend to reach past the DAPI you can see. The bright cores are "
            "hard to hurt: they keep 97% of their signal at 1 and 90% at 4.\n\n"
            "Takes effect on COMMIT.")

        self.scale = self._number(
            "dapi_background_scale_um",
            f"{DEFAULT_SCALE_UM:g}",
            "How broad the background is assumed to be, in micrometres — the width of the blur "
            "used to estimate it.\n\n"
            "It has to comfortably exceed one nucleus, or the estimate contains the nuclei and "
            "the subtraction eats them: at 2 um a nucleus core keeps 7% of its signal. Above "
            "one nucleus diameter the background is already entirely removed and the value "
            "stops mattering much — 8, 16 and 80 um differ by almost nothing.\n\n"
            "12 um is right for the 7.6 um nuclei in this study and is not a universal "
            "constant, which is the only reason it is exposed at all.\n\n"
            "Takes effect on COMMIT.")

        self.commit = CommitButton("Commit", self)
        self.commit.clicked.connect(self._on_commit)
        self.frame.add_right(self.commit)

        #: **No z-batch depth box and no `Step by batch`** — see the module docstring. The
        #: indicator stays, because the arrow keys still move and a picture with no idea which
        #: slice it is showing is what m2 was about.
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

        #: **Greyed from arrival, not only after a parameter moves.** A screen that opened with
        #: two live checkboxes over empty layers would be offering a view of a commit that has
        #: not happened.
        self._set_derived_available(False)
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

    # -- the data -------------------------------------------------------

    def set_volume(self, dapi, voxel_size_um=None) -> None:
        """Hand the screen its channel. **The screen holds it** rather than taking it per draw.

        Items 3, 4, 5 and 9b all do the same, for the reason item 4's first smoke test found: a
        z step has to redraw without being handed the data again, and passing the volumes per
        call silently dropped layers on the second draw.
        """
        self.dapi = None if dapi is None else np.asarray(dapi)
        if voxel_size_um is not None:
            self.voxel_size_um = tuple(float(v) for v in voxel_size_um)
        #: A new volume is a new session's worth of everything: the commit that produced
        #: `filtered` was about the old one. **The derived pair goes back to greyed with it** —
        #: they describe a commit, and there is no longer one.
        self.is_committed = False
        self.filtered = None
        self._z_started = False
        self._set_derived_available(False)
        self.toggles.set_on("DAPI", True)
        self.set_readout(None)
        self.draw()

    def commit_result(self, filtered: np.ndarray) -> None:
        """The whole-volume filtered channel, from `app.run_commit`.

        **The readout is computed here, from the two volumes**, rather than being passed in as a
        number — so the figure on screen is taken from the array that actually goes forward.
        """
        self.filtered = np.asarray(filtered)
        self.is_committed = True
        if self.dapi is not None:
            self.set_readout(removed_fraction(self.dapi, self.filtered))
        #: **The two derived layers come into existence here**, and the screen switches to
        #: `DAPI kept` — *"after commit the user should see what will go forward"*, the user's
        #: rule of 2026-08-23. `DAPI` goes off rather than staying under it: kept is a subset of
        #: it, and under additive blending leaving both on would draw the surviving signal at
        #: double weight and the removed part at single, which reads as the opposite of what it
        #: is. Both remain one tick away.
        self._set_derived_available(True)
        self.toggles.set_on("DAPI", False)
        self.toggles.set_on("DAPI kept", True)
        self.draw()

    def _set_derived_available(self, available: bool) -> None:
        """Grey or ungrey `DAPI kept` and `DAPI removed`, and the sentence that repeats it."""
        for name in COMMITTED_ONLY:
            self.toggles.set_enabled(name, available)
            if not available:
                self.toggles.set_on(name, False)
        self.layers_note.setVisible(not available)

    # -- behaviour ------------------------------------------------------

    def _recolour(self, layer: str, colour: str) -> None:
        """One picker row, **both panels.** `_layers_for` is where that pairing lives."""
        for name in self._layers_for(layer):
            if self.viewer is not None and name in self.viewer.layer_names():
                self.viewer.set_colour(name, colour)

    def _reweight(self, layer: str, alpha: float) -> None:
        for name in self._layers_for(layer):
            if self.viewer is not None and name in self.viewer.layer_names():
                self.viewer.set_alpha(name, alpha)

    def _toggle(self, name: str, on: bool) -> None:
        for layer in self._layers_for(name):
            if self.viewer is not None and layer in self.viewer.layer_names():
                self.viewer.set_visible(layer, on)

    @staticmethod
    def _layers_for(concept: str) -> tuple[str, ...]:
        """The two napari layers one picker row drives — the linear copy and the log copy."""
        return tuple(name for name in LAYER_ORDER if name.rsplit(" — ", 1)[0] == concept)

    def _on_parameter_changed(self) -> None:
        """**Either box, and both take effect on COMMIT.**

        Typing does not filter anything — it withdraws the last answer. `DAPI kept` and `DAPI
        removed` describe a commit, so a keystroke that changed the parameters without changing
        them would leave two layers claiming to show the consequence of a value the screen is no
        longer holding. **That is the stale view the gate exists to refuse**, and here it is
        expressed by the layers disappearing rather than by a mark alone.
        """
        self._drop_to_uncommitted()

    def _drop_to_uncommitted(self) -> None:
        """Back to the channel alone, the derived pair greyed, and the gate closed.

        > ### ⇒ IT ENDED IN `draw()`, ON `textChanged` — patch item A3, 2026-09-03
        >
        > Both boxes are wired to `textChanged`, so this ran **per character**, and `draw()`
        > took the whole-channel percentiles with it: **~9.4 s a keystroke** before A1 memoised
        > them, and a pointless full redraw after it. This was the only screen in the flow whose
        > commit-gated boxes redrew — items 2, 3, 4 and 5 only invalidate.
        >
        > **Withdrawing the derived pair is a visibility change, not a picture change.**
        > `_set_derived_available(False)` unticks `DAPI kept` and `DAPI removed`, and
        > `LayerToggles.toggled` is wired to `_toggle`, which calls `viewer.set_visible` — so
        > the layers leave the canvas without a byte being recomputed. Nothing else here touches
        > what is drawn: the gate, the readout reset and the DAPI toggle all stay.
        """
        self.is_committed = False
        self.filtered = None
        self.set_readout(None)
        self._set_derived_available(False)
        self.toggles.set_on("DAPI", True)
        self.flow.invalidate()
        self._sync()

    def _on_commit(self, product=None) -> None:
        """Refuse a bad scale, then flip the gate and ask for the work the button names.

        **The refusal comes first and leaves the gate alone**, which is item 3's arrangement —
        the value reaches `subtract_background` directly, so silently substituting a default
        would filter at a setting the screen is not showing.
        """
        problem = self.parameter_error()
        if problem is not None:
            self.commit.set_working(problem)
            return
        self.flow.commit(product)
        self._sync()
        if product is None:
            self.commit.set_working(COMMIT_PROGRESS)
            try:
                self.commit_requested.emit()
            finally:
                self.commit.set_working("")

    def set_readout(self, fraction: float | None) -> None:
        """**`—` rather than `0.0%` before the first commit.** A zero here would be a claim that
        the filter removes nothing, which is a different statement from *not yet measured* — and
        at strength 0 it happens to be true, so the two would be indistinguishable at exactly
        the value the screen opens on."""
        self.header.set_readout(
            "DAPI removed  —" if fraction is None else f"DAPI removed  {fraction:.1%}")

    # -- z --------------------------------------------------------------

    def set_z(self, z: int) -> None:
        """**Absolute**, because napari's own z slider reports a slice rather than a step."""
        if self.dapi is None:
            return
        n = int(self.dapi.shape[0])
        self.z = max(0, min(int(z), n - 1))
        self.z_indicator.setText(f"({self.z + 1}/{n})")
        self.draw()

    def step_z(self, delta: int) -> None:
        if self.dapi is not None:
            self.set_z(self.z + int(delta))

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

    # -- drawing --------------------------------------------------------

    def panel_planes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(DAPI, DAPI kept, DAPI removed)` at the current z.

        **The single place the committed/uncommitted decision lives**, which is item 5's
        `panel_masks()` in its own shape. The first is always the channel as it arrived. The
        other two are **zeros until COMMIT**, because until then there is no answer to what
        survives — and zeros, rather than absent, so the layer indices that assign the panels
        never move. See `LAYER_ORDER`.

        After a commit `kept + removed == DAPI` at every voxel, by construction: the filter
        clips at zero, so `removed` is `min(v, k * blur)` and nothing is invented or lost.
        """
        raw = np.asarray(self.dapi)[self.z]
        if not self.is_committed or self.filtered is None:
            empty = np.zeros_like(raw)
            return raw, empty, empty
        kept = np.asarray(self.filtered)[self.z]
        return raw, kept, np.clip(raw - kept, 0.0, None)

    def draw(self) -> None:
        """Push the six layers in panel order: the linear three, then the log three."""
        if self.dapi is None:
            return
        volume = np.asarray(self.dapi)
        #: **The middle slice on arrival**, as items 2, 3, 5, 9a and 9b all open. Slice 0 of a
        #: 40-slice stack is the edge of the tissue and is routinely empty.
        if not self._z_started:
            self._z_started = True
            self.z = middle_slice(int(volume.shape[0]))
        self.z = max(0, min(int(self.z), int(volume.shape[0]) - 1))
        self.z_indicator.setText(f"({self.z + 1}/{int(volume.shape[0])})")
        if self.viewer is None:
            return

        planes = dict(zip((name for name, *_ in CONCEPTS), self.panel_planes()))
        #: **All three DAPI layers share ONE pair of limits, computed from the raw volume.**
        #: Kept and removed are parts of the same channel, so stretching either on its own
        #: percentiles would give them different ramps and make them incomparable with the
        #: channel they came from — `screen_veto` records the identical decision for the
        #: identical reason. They are also taken from the **volume**, not the plane, so stepping
        #: z does not restretch the picture underneath the judgement being made.
        limits = {"linear": contrast_limits(volume, self.display_range.get("nuclei")),
                  "log": log_limits(volume)}

        for name in LAYER_ORDER:
            concept, panel = name.rsplit(" — ", 1)
            plane = planes[concept]
            row = self.rows[concept]
            self.viewer.show_image(
                name, plane if panel == "linear" else log_view(plane),
                colour=row.colour().name, alpha=row.alpha(),
                visible=self.toggles.is_on(concept),
                contrast_limits=limits[panel])

    # -- the parameters, and the gate -----------------------------------

    def parameters(self) -> dict:
        """The `SegmentationParams` fields this screen owns, under **those** names.

        A half-typed box contributes nothing rather than a zero — `float("")` raises and a box
        mid-edit is not a choice. **Both are boxes now**, so neither is guaranteed present, and
        `parameter_error` refuses the commit on exactly the values this skips.
        """
        return parsed_numbers({
            "dapi_background_strength": (self.strength.text(), float),
            "dapi_background_scale_um": (self.scale.text(), float),
        })

    def parameter_error(self) -> str | None:
        """Why the commit is refused, or `None`. Item 3's shape, and its reasoning — both values
        reach `subtract_background` directly, so silently substituting a default would filter at
        a setting the screen is not showing.

        **The strength may be zero and the scale may not.** Zero strength is what *off* means and
        has to stay expressible; zero scale is a second, worse way to spell the same thing, and a
        negative sigma has no meaning at all. **No upper bound on the strength** — it is a box
        precisely so that the useful range is advice rather than a limit.
        """
        try:
            strength = float(self.strength.text().strip())
        except ValueError:
            return "dapi_background_strength must be a number"
        if strength < 0:
            return "dapi_background_strength cannot be negative"
        try:
            scale = float(self.scale.text().strip())
        except ValueError:
            return "dapi_background_scale_um must be a number"
        if scale <= 0:
            return "dapi_background_scale_um must be greater than zero"
        return None

    def _sync(self) -> None:
        self.flow.session.chosen.update(self.parameters())
        self.commit.set_stale(self.flow.is_stale())
        self.continue_button.setEnabled(self.flow.can_continue())

    def _on_continue(self) -> None:
        if self.flow.can_continue():
            self.continue_requested.emit()
