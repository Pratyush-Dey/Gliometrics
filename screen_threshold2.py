"""Screen 5 — Iba1 Measurement Threshold. `COMPONENT_03_PLAN.md` §6i, §4 item 5.

**Frame B, and the only screen that uses it.** It decides **how far each cell extends**, never how
many cells there are — that was fixed at item 3 and by splitting at item 4.

> ### THE STRUCTURE IS §6i.1'S CORRECTION, AND IT IS THE WHOLE SCREEN
>
> **Four layers, three toggles, and each panel shows three of the four.** Both panels carry
> `Iba1 raw` and `CD68` **identically**; only the threshold mask differs — segmentation left,
> measurement right. One `threshold` picker drives a different layer in each panel, which is the
> one thing napari's own layer list cannot express: it would show **six** rows grouped by panel
> rather than the three concepts the user is thinking in. That is why the left column is ours.
>
> **CD68 on one side only is the failure `DESIGN_RATIONALE_B.md` §5 names outright** — *"the
> panels must be in the same state or the screen flatters threshold 2."* §4's own layer table
> said exactly that and was corrected in place on 2026-08-19.

> ### RAW-VS-RAW, THEN CELLS-VS-CELLS. NEVER ONE OF EACH.
>
> With threshold 1's *finished* mask facing threshold 2's *raw preview*, most of the visible
> difference would be hole-filling and 3D closing rather than the coefficient. **So moving the
> slider after a commit drops BOTH panels back to raw**, in step with the exclamation appearing
> beside Commit. `is_committed` is that flag, and it is the same rule the projection toggle obeys.

**There is no value to suggest and the slider does not pretend otherwise.** The median marginal
return runs 5.68 → 3.41 → 2.57 → 2.21 across 0.10 → 0.02 — diminishing, **no turnover** — and
`mask_coefficient_sweep.txt` ends *"No value is adopted here."* The slider opens at the **top**,
which is the no-op, not a suggestion: every move down is then visibly additive.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ..cache import SliceCache, cache_key
from ..threshold import slice_contrast
from . import gestures, palette
from .frames import CanvasHost, FrameB
from .preview import middle_slice, preview_mask, volume_contrast_limits, z_position
from .semiauto import Flow, Session, Step, parsed_numbers
from ..cache import WindowCache
from .viewer import (
    BATCHED, ROLLING, CanvasLayout, ViewerPort, batch_plane, cached_plane, outline_of_batch,
    step_window,
    z_window_range, z_window_word,
)
from .widgets import (
    AccumulatingTable, ColourRow, CommitButton, HeaderStrip, HelpButton, LayerToggles,
    NavigationRow, ParameterSlider, batch_step_box,
)

__all__ = [
    "MeasurementThresholdScreen", "LAYERS", "CONCEPTS", "PANELS", "LAYER_ORDER",
    "GRID_STRIDE", "TABLE_COLUMNS", "cd68_inside_fraction", "mask_volume_fraction",
    "table_row", "slider_ceiling", "COEFFICIENT_RANGE",
]

#: §6i.5. **The headers name the JOB, not the order** — not `Threshold 1` and `Threshold 2`.
#: §4's title table renamed these two screens *"by what they decide — how many cells there are
#: against how far each one reaches"*, and the panel headers use the same vocabulary so the
#: screen and the two screen titles say one thing rather than two.
PANELS: tuple[str, str] = ("Segmentation threshold", "Measurement threshold")

#: The three the user thinks in — one picker and one toggle each, **each driving BOTH panels**.
#:
#: `threshold` takes a colour here although item 3's mask does not, and the reason is the
#: overlay: item 3's mask sits **alone in its own panel**, while here it lies on top of Iba1 raw
#: and CD68. §4 gave it a picker because a filled black-and-white layer *"would occlude both"* —
#: correct then, and now moot, because §6i.2 draws it as an **outline** and an outline occludes
#: nothing. The picker stays; only its default was settled.
CONCEPTS: tuple[tuple[str, str, float, bool], ...] = (
    ("Iba1 raw", "white", palette.CHANNEL_ALPHA["Iba1"], True),
    ("CD68", "green", palette.CHANNEL_ALPHA["CD68"], True),
    ("threshold", "white", palette.DERIVED_MASK_ALPHA, True),
)

#: **All three ON at arrival.** Unlike item 4, all three layers are the content here — and CD68
#: must be visible because the headline number is about it.
LAYERS = CONCEPTS

#: **Six napari layers, and the ORDER is what assigns the panels.** Verified against napari
#: 0.7.1: `grid.shape=(1, 2)` with **`stride=3`** places layers 0-2 in the left panel and 3-5 in
#: the right. Panel assignment is computed from `layers.index(layer)`, **not from visibility**,
#: so hiding a layer empties its half rather than reshuffling the panels — which is exactly what
#: makes one toggle able to drive both copies without the geometry moving.
LAYER_ORDER: tuple[str, ...] = (
    "Iba1 raw — segmentation", "CD68 — segmentation", PANELS[0],
    "Iba1 raw — measurement", "CD68 — measurement", PANELS[1],
)

GRID_STRIDE = 3

#: §6i.6. **Marginal CD68 per unit volume was offered and declined** — it is the delta between
#: adjacent rows and the sweep script calls it *"the one to read"*, so it stays a free derived
#: column if it is ever wanted. The user's answer was no, and the accumulating table is what
#: makes that cheap: the trade is legible by eye down the two right columns.
TABLE_COLUMNS: tuple[str, ...] = ("coef", "closing", "CD68 in", "mask vol")


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

def cd68_inside_fraction(cd68: np.ndarray, mask: np.ndarray) -> float:
    """The headline number, and it is **threshold-free** — `cd68[mask].sum() / cd68.sum()`.

    No `cd68_threshold` anywhere in it, which is precisely why the two study-level pickers can
    sit later at item 9 and this screen still gets its key figure.

    **A faithful figure is a whole-field statistic**, so it arrives with the commit rather than
    live. A per-slice version would be computable and noisy.
    """
    total = float(cd68.sum())
    return 0.0 if total <= 0 else float(cd68[mask.astype(bool)].sum()) / total


def mask_volume_fraction(mask: np.ndarray) -> float:
    """The cost. *"A mask that captures all the CD68 by swallowing the whole field has not solved
    anything, and this column is what makes that visible."*
    """
    return float(np.asarray(mask).astype(bool).mean())


def table_row(coefficient: float, closing_radius_um: float,
              cd68_inside: float | None, mask_volume: float) -> tuple[str, str, str, str]:
    """One commit, formatted for §6i.6's table.

    **`None` prints an em dash, and it is explicitly not `0.0`.** A study acquired without a
    CD68 channel has no containment fraction to show, and *"CD68 in"* is the number this whole
    screen exists to let a user judge — a plausible zero there would be read as a mask that
    captures none of it. **Four columns either way**, so the table keeps a fixed width rather
    than one that follows the session.
    """
    return (f"{coefficient:.2f}", f"{closing_radius_um:g}",
            "—" if cd68_inside is None else f"{cd68_inside:.1%}", f"{mask_volume:.2%}")


#: **The same travel as item 3's, since 2026-08-25.** Both screens threshold the same channel
#: the same way, so a reader moving between them meets one scale rather than two.
COEFFICIENT_RANGE: tuple[float, float] = (0.0, 3.0)


def slider_ceiling(cell_coefficient: float) -> float:
    """Item 3's committed coefficient — **no longer the ceiling, but still worth showing.**

    > ### THE CEILING'S JUSTIFICATION DIED WITH THE DEBRIS PATCH — removed 2026-08-25
    >
    > It was capped here because above `cell_coefficient` the parameter was *provably* a no-op:
    > `assign_labels_by_connectivity` unions the measurement mask with the existing cells before
    > assigning, *"so a cell can never shrink"* — confirmed identical volume, skeleton and
    > endpoints at 0.18 through 1.20.
    >
    > **That union does nothing any more.** Since 2026-08-24 the cells are split out of the
    > measurement mask itself, and `pipeline.segment_image` builds the seed as
    > `np.where(measure.filled, ..., 0)` — so the seed is a **subset** of the measurement mask
    > and `measure_mask | occupied` is just `measure_mask`. Nothing is added, so nothing stops a
    > cell shrinking, and the region above item 3's value is live.
    >
    > **Measured on crop2**, mask volume against `mask_coefficient`: 4803 µm³ at 0.18 — the old
    > ceiling — falling monotonically to 3873 at 0.30, 1578 at 1.00 and **152.6 at 3.00**. A 31x
    > reduction across travel the user could not reach. The cap was not protecting anyone from a
    > dead control; it was hiding a live one.
    >
    > **There is no floor either.** 0.02 is only where the original sweep stopped; the mask grows
    > smoothly below it — 22.9% foreground at 0.02, 25.3% at 0.01, 27.5% at 0.005 — and the
    > empty-slice guard does not begin discarding slices until 0.002.

    Kept as a function because the value is still **shown** on the screen: item 3's threshold is
    the reference point this one is judged against, even now that it does not bound it.
    """
    if cell_coefficient <= 0:
        raise ValueError(f"the reference comes from cell_coefficient, got {cell_coefficient}")
    return float(cell_coefficient)


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

#: Q2's rule is *progress shown, no cancel*. ~59 s measured on a real field.
COMMIT_PROGRESS = "Building both masks…"


class MeasurementThresholdScreen(QWidget):
    """Frame B: three pickers, three toggles, one live slider, one commit-only number box."""

    continue_requested = Signal()
    #: **Every screen but items 1, 9a and 7b, since 2026-08-30.**
    back_requested = Signal()
    #: **The COMMIT's work, which this screen cannot run itself.** It needs the pipeline,
    #: and a screen importing `detect_nuclei` or `ImageSource` would break §2's first
    #: constraint. `app.SemiAutoApp.run_commit` is the one listener. Emitted **after** the
    #: gate flips, because the gate answers *have the parameters moved since the last
    #: commit*, which the press itself settles.
    commit_requested = Signal()

    #: Six layers over two panels, so the stride is 3 — LAYER_ORDER's own
    #: constant rather than a second number that could drift from it.
    CANVAS = CanvasLayout(shape=(1, 2), stride=GRID_STRIDE)

    DEFAULT_CLOSING_RADIUS_UM = "0.8"

    #: **Item 3's value, deliberately** — the debris patch gives item 5 its own floor and
    #: defaults it to the number the two masks used to share, so the patch changes no result
    #: on its own. `pipeline.SegmentationParams.measurement_min_object_voxels`.
    DEFAULT_MIN_OBJECT_VOXELS = "20"

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 cell_coefficient: float = 0.18, parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.THRESHOLD_2)
        self.viewer = viewer

        self.cell_coefficient = slider_ceiling(cell_coefficient)
        self.iba1: np.ndarray | None = None
        self.cd68: np.ndarray | None = None
        self.segmentation_cells: np.ndarray | None = None
        self.measurement_cells: np.ndarray | None = None
        self.volume_contrast: float = 0.0
        self.contrast_limits: tuple[float, float] | None = None
        self.z = 0
        self.depth = 1
        #: Rolling until the user ticks `Step by batch`. The two modes agree at depth 1, so
        #: the toggle can never move the picture on its own.
        self.batch_mode = ROLLING
        #: **The window cache — item A9.** Four planes a position: two channels and the two
        #: panel outlines. The live preview keeps its own `SliceCache`, which is keyed on the
        #: coefficient rather than on the window and answers a different question.
        self.windows = WindowCache()
        self.is_committed = False
        self.cache = SliceCache()

        self.frame = FrameB("Iba1 measurement threshold", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: three pickers, three toggles -------------------------
        self.frame.section("Colours", HelpButton(
            "Colours",
            "Every picker offers the same ten colours on every screen, so a colour means one "
            "thing throughout. The slider beneath each row is a weight: layers add rather than "
            "cover each other.\n\n"
            "Each row drives BOTH panels. They have to stay in the same state, or the "
            "comparison between them is not a comparison."))

        self.rows: dict[str, ColourRow] = {}
        for name, colour, alpha, _ in CONCEPTS:
            row = ColourRow(name, colour, alpha, self)
            row.colour_changed.connect(lambda c, n=name: self._recolour(n, c))
            row.alpha_changed.connect(lambda a, n=name: self._reweight(n, a))
            self.frame.add_left(row)
            self.rows[name] = row

        self.frame.section("Layers", HelpButton(
            "Layers",
            "Three rows, six layers: each one acts on both panels at once.\n\n"
            "threshold is the one that differs between them. On the left it is the segmentation "
            "threshold you set earlier — the boundary this screen extends from. On the right it "
            "is the measurement threshold you are choosing now."))
        self.both_panels = QLabel("(both panels)", self)
        self.both_panels.setProperty("role", "helper")
        self.frame.add_left(self.both_panels)

        self.toggles = LayerToggles(tuple((name, on) for name, _, _, on in CONCEPTS), self)
        self.toggles.toggled.connect(self._toggle)
        self.frame.add_left(self.toggles)
        self.frame.left.addStretch(1)

        # -- centre: the two panel names, the canvas, then the table -----
        #: **This screen alone carries no readout in the strip.** Every number here arrives on
        #: commit into the table below, so a figure in the strip would duplicate its last row.
        #: Items 2, 3 and 4 put one there; item 5 has nothing that is not already below.
        self.header = HeaderStrip(PANELS, "", None, self)
        self.frame.add_centre(self.header)
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        self.table = AccumulatingTable(TABLE_COLUMNS, self)
        self.frame.add_under_right_panel(self.table)

        # -- right: the live slider, then the commit-only number ---------
        self.mask_coefficient = ParameterSlider(
            "mask_coefficient", COEFFICIENT_RANGE[0], COEFFICIENT_RANGE[1],
            self.cell_coefficient,
            help_button=HelpButton(
                "mask_coefficient",
                "How much Iba1 signal counts as part of a cell for MEASUREMENT, as a multiple "
                "of each slice's automatic threshold. It changes how far each cell reaches, "
                "never how many cells there are.\n\n"
                "LOWER -> cells extend further, picking up dim thin processes\n\n"
                "The top of the slider is the segmentation threshold you set earlier, and at "
                "that position this screen does nothing — cells can never shrink, so anything "
                "above it would have no effect at all. Committing without moving is a real "
                "answer: it means no extension.\n\n"
                "LIVE: both panels follow it on this slice. The numbers arrive on COMMIT."),
            parent=self)
        self._wire_slider(self.mask_coefficient)
        self.frame.add_right(self.mask_coefficient)

        #: **The one new control in the debris patch, and it sits where item 3 has its own** —
        #: the user's decision, 2026-08-23. Item 5 shared item 3's `min_object_voxels` silently
        #: until then, which was defensible while item 3's mask decided the cell set; since the
        #: patch this mask is the one that gets split, so its floor decides what can become a
        #: cell at all. Same control as item 3's, same `?`, above the closing radius so the two
        #: thresholding boxes read in the same order on both screens.
        #:
        #: **Labelled `measurement_min_object_voxels`, not `min_object_voxels`.** §2's rule —
        #: *the control labels are the identifiers the parameters file uses* — has a test of its
        #: own, and the two boxes are different fields. Labelling both `min_object_voxels` would
        #: also put the same word on two screens for two numbers that need not agree, which is
        #: the confusion the separate field exists to end.
        self.measurement_min_object_voxels = self._number(
            "measurement_min_object_voxels", self.DEFAULT_MIN_OBJECT_VOXELS,
            "Drops connected blobs smaller than this many voxels from the measurement mask, "
            "measured in 3D across the whole stack.\n\n"
            "This is the measurement threshold's own floor. The segmentation screen has one "
            "too, and they no longer have to agree: this mask is the one that gets divided "
            "into cells, so what this removes can never become a cell or debris.\n\n"
            "Not previewed: a blob that looks small on one slice may be a long process in 3D."
            "\n\nTakes effect on COMMIT.")

        self.mask_closing_radius_um = self._number(
            "mask_closing_radius_um", self.DEFAULT_CLOSING_RADIUS_UM,
            "Bridges small gaps in the extended cell outline, then fills the hole left by the "
            "nucleus — the same job as on the segmentation screen, for this mask.\n\n"
            "Raising it above the value used there makes the top of the slider stop being a "
            "no-op: a fatter mask at the same coefficient still extends the cells. Lowering it "
            "does not, because cells can never shrink.\n\n"
            "Not previewed — part of it works across slices. Takes effect on COMMIT.")

        self.commit = CommitButton("Commit", self)
        self.commit.clicked.connect(self._on_commit)
        self.frame.add_right(self.commit)

        self.z_depth = self._number(
            "z-batch depth", "1",
            "How many z slices are combined into one view.\n\n"
            "Disabled until the first commit, and that is a consequence rather than a choice: "
            "before a commit only the slice you are looking at has been computed, so there is "
            "nothing to combine.\n\n"
            "Projected, the two panels answer the question stepping through planes cannot — "
            "whether a severed process reconnected is a 3D fact, and comparing plane 20 with "
            "plane 21 does not reveal it.")
        #: **On ENTER or when focus leaves, never per keystroke.** The user's
        #: decision, 2026-08-21. A z-batch is a whole re-projection of every layer,
        #: and applying one to each character of `10` means projecting at 1 on the
        #: way past — which is a dimensionality change of every layer, twice.
        self.z_depth.editingFinished.connect(self._on_depth_changed)
        self.batch_step = batch_step_box(self)
        self.batch_step.toggled.connect(self._on_batch_mode_changed)
        self.frame.add_right(self.batch_step)
        self.depth_note = QLabel("(disabled until first commit)", self)
        self.depth_note.setProperty("role", "helper")
        self.frame.add_right(self.depth_note)

        self._z_heading = self._heading(
            "z slice",
            "Which slice you are looking at, out of the whole stack. LEFT and RIGHT arrow keys "
            "move through z.\n\n"
            "At z-batch depth above 1 this counts groups of slices instead and reads "
            "\"z batch\".")
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

    def _heading(self, label: str, help_text: str) -> QLabel:
        name = QLabel(label, self)
        row = QHBoxLayout()
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(HelpButton(label, help_text, self))
        host = QWidget(self)
        host.setLayout(row)
        self.frame.add_right(host)
        return name


    def _wire_slider(self, slider: ParameterSlider) -> None:
        """**One place, because there are two sliders and they must behave identically.**

        The constructor builds one at the default ceiling and `set_cell_coefficient` replaces it
        when item 3's real value arrives. Wiring each site separately means the live control can
        quietly differ from the one the constructor tested — item 9a shipped exactly that, and a
        mutation pass found it because flipping the constructor's connection changed nothing a
        user could reach.
        """
        slider.value_changed.connect(self._on_coefficient_moved)
        slider.slider.sliderReleased.connect(self.refresh_preview)

    def set_cell_coefficient(self, cell_coefficient: float) -> None:
        """Re-range the slider to item 3's **committed** coefficient.

        > ### THE CEILING WAS A CONSTRUCTOR ARGUMENT NOBODY PASSED — task 26
        >
        > `slider_ceiling` says the top of the travel *"is set by ANOTHER SCREEN, and it moves
        > with it"*, and `build_screen` constructs this screen from a plain `{step: class}`
        > mapping with no arguments — so the ceiling was the **default 0.18** in every session,
        > whatever the user had committed at item 3. Reported on a real run: *"the slider was 0
        > to 0.18 even though i did not set 0.18 as my segmentation threshold."*
        >
        > **A constructor argument could not have worked anyway.** Screens are cached in
        > `self.screens`, and item 4's Back goes to item 3 — so a coefficient committed on the
        > second visit would arrive after this screen had already been built. It has to be
        > pushed on arrival, which is what `app.prepare` now does.

        The value moves with the ceiling: the slider **opens at the top**, where this screen is
        a deliberate no-op, and a value carried over from a higher ceiling would silently mean
        something different under a lower one.
        """
        reference = slider_ceiling(cell_coefficient)
        if reference == self.cell_coefficient:
            return
        self.cell_coefficient = reference
        old = self.mask_coefficient
        #: **The travel is fixed at `COEFFICIENT_RANGE` since 2026-08-25**; only the opening
        #: VALUE follows item 3. The slider is rebuilt rather than merely re-valued because
        #: that is what it has always done here, and the reconstruction is what carries the
        #: help button across.
        fresh = ParameterSlider("mask_coefficient", COEFFICIENT_RANGE[0],
                                COEFFICIENT_RANGE[1], reference,
                                help_button=old.help_button, parent=self)
        self._wire_slider(fresh)
        index = self.frame.right.indexOf(old)
        self.frame.right.removeWidget(old)
        old.setParent(None)
        self.frame.right.insertWidget(max(0, index), fresh)
        self.mask_coefficient = fresh
        self._sync()

    def _number(self, label: str, value: str, help_text: str) -> QLineEdit:
        self._heading(label, help_text)
        edit = QLineEdit(value, self)
        if label != "z-batch depth":
            edit.textChanged.connect(self._on_parameter_changed)
        self.frame.add_right(edit)
        return edit

    # -- loading --------------------------------------------------------

    def set_volumes(self, iba1: np.ndarray, cd68: np.ndarray | None) -> None:
        """`volume_contrast` and the fixed limits are computed **once**, exactly as at item 3.

        The single-slice trap is identical here — `threshold_channel`'s empty-slice guard is
        volume-relative, so a slice handed to it alone becomes its own reference and the guard
        can never fire.
        """
        #: **`cd68` is `None` for a study acquired without the channel** — `app.prepare` hands
        #: over `session.channels.get("cd68")` and there is nothing there. Dereferencing it for
        #: the shape check was the crash; the draw path below has always guarded for `None`.
        if cd68 is not None and iba1.shape != cd68.shape:
            raise ValueError(f"Iba1 {iba1.shape} and CD68 {cd68.shape} must have the same shape")
        self.iba1, self.cd68 = iba1, cd68
        self.volume_contrast = slice_contrast(iba1)
        self.contrast_limits = volume_contrast_limits(iba1)
        self.cache.invalidate()
        self.set_z(middle_slice(iba1.shape[0]))

    # -- z --------------------------------------------------------------

    def set_z(self, z: int) -> None:
        if self.iba1 is None:
            return
        n = self.iba1.shape[0]
        self.z = max(0, min(int(z), n - 1))
        self.z_indicator.setText(z_window_range(self.z, n, self.depth, self.batch_mode))
        self.draw()

    def bind_gestures(self) -> None:
        gestures.bind_z_stepping(self.viewer, self)

    def step_z(self, delta: int) -> None:
        """**The stride is the toggle**: one slice rolling, a whole batch when `Step by batch`
        is ticked. `step_window` also stops where the window last fills, so rolling cannot walk
        into the tail where the projection thins while the label still claims a full batch."""
        if self.iba1 is None:
            return
        self.set_z(step_window(self.z, delta, int(self.iba1.shape[0]),
                               self.depth, self.batch_mode))

    def _on_batch_mode_changed(self, batched: bool) -> None:
        self.batch_mode = BATCHED if batched else ROLLING
        self.set_z(self.z)

    def keyPressEvent(self, event):     # noqa: N802 - Qt's name
        if event.key() == Qt.Key.Key_Left:
            self.step_z(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.step_z(1)
        else:
            super().keyPressEvent(event)

    def _on_depth_changed(self) -> None:
        try:
            self.depth = max(1, int(self.z_depth.text().strip()))
        except ValueError:
            return
        self._z_heading.setText(z_window_word(self.depth))
        self.batch_step.setEnabled(self.depth > 1 and self.is_committed)
        self.set_z(self.z)

    # -- the preview ----------------------------------------------------

    def _preview(self, coefficient: float) -> np.ndarray:
        key = cache_key(self.z, coefficient)
        return self.cache.get(key, lambda: preview_mask(
            self.iba1, self.z, coefficient=coefficient,
            volume_contrast=self.volume_contrast))

    def refresh_preview(self) -> None:
        self.draw()

    def panel_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """What the two threshold layers hold, and **both are always in the same state**.

        | | left | right |
        |---|---|---|
        | **live** | raw threshold at item 3's committed `cell_coefficient` | raw threshold at the slider |
        | **committed** | the whole-stack cells, as a boolean | the whole-stack cells, as a boolean |

        Never one of each: with threshold 1's finished mask facing threshold 2's raw preview,
        most of the visible difference would be hole-filling and 3D closing rather than the
        coefficient, and **the screen would flatter threshold 2.**
        """
        if self.is_committed and self.segmentation_cells is not None \
                and self.measurement_cells is not None:
            #: **Two whole-volume bool copies were being made here — patch item A8.**
            #: Both were taken *before* `outline_of_batch` sliced the window, and neither was
            #: needed: its `any(axis=0)` already treats nonzero as true.
            left = cached_plane(
                self.windows, "left outline", self.segmentation_cells, self.z, self.depth,
                self.batch_mode,
                lambda: outline_of_batch(self.segmentation_cells, self.z, self.depth,
                                         mode=self.batch_mode))
            right = cached_plane(
                self.windows, "right outline", self.measurement_cells, self.z, self.depth,
                self.batch_mode,
                lambda: outline_of_batch(self.measurement_cells, self.z, self.depth,
                                         mode=self.batch_mode))
            return left, right
        left = self._preview(self.cell_coefficient)
        right = self._preview(self.mask_coefficient.value())
        return left, right

    def draw(self) -> None:
        """Push all six layers **in grid order**: 0-2 left, 3-5 right, at `stride=3`."""
        if self.viewer is None or self.iba1 is None:
            return

        #: **The batch, not the slice.** Both were `self.iba1[self.z]` until task 26, so the
        #: two mask outlines projected through `outline_of_batch` and the channels they were
        #: being judged against did not.
        #: > ### ⇒ THE CHANNELS IGNORED `batch_mode` AND THE OUTLINES DID NOT — found A9
        #: >
        #: > These two called `batch_plane` **without a mode**, so they took the default
        #: > `ROLLING` while `panel_masks` above projected its outlines through
        #: > `self.batch_mode`. With `Step by batch` ticked at any depth above 1 the two halves
        #: > of this screen were showing **different windows of the stack** — the mask outlined
        #: > over a picture it does not belong to, which is task 26's defect in a second door.
        #: >
        #: > Display only: nothing here reaches a number. Fixed while wiring the cache, because
        #: > the cache is keyed on the mode and a screen with two modes at once cannot be keyed
        #: > at all.
        plane_iba1 = cached_plane(
            self.windows, "iba1", self.iba1, self.z, self.depth, self.batch_mode,
            lambda: batch_plane(self.iba1, self.z, self.depth, self.batch_mode),
            projection=True)
        #: > ### ⇒ A MISSING CD68 TOOK THE Iba1 WITH IT — patch item B3, 2026-09-03
        #: >
        #: > This is a two-panel grid at `stride=3`, and **napari assigns panels by layer index
        #: > over the layers that actually exist**, not by visibility. With no CD68 channel the
        #: > two CD68 layers were never added, four layers remained, and the stride re-cut them:
        #: > the left panel got Iba1-segmentation, mask-segmentation **and** Iba1-measurement —
        #: > two copies of the same data, additively stacked, so doubled — and the right panel
        #: > got the measurement mask **with no Iba1 underneath it at all.** That is the user's
        #: > missing signal, and it is the trap §6i already records for hiding a layer, except
        #: > that hiding keeps it in the list and never creating it does not.
        #: >
        #: > **A zero plane, so all six indices stay put.** The precedent is in this codebase
        #: > already: `screen_background.panel_planes` returns `np.zeros_like(raw)` for layers
        #: > with no answer yet, *"zeros, rather than absent, so the layer indices that assign
        #: > the panels never move."* Zeros contribute nothing under additive blending.
        #: >
        #: > **The CD68 picker row is deliberately left as it is**, inert rather than greyed —
        #: > the user's decision, recorded here so nobody adds it uninvited.
        plane_cd68 = (cached_plane(
            self.windows, "cd68", self.cd68, self.z, self.depth, self.batch_mode,
            lambda: batch_plane(self.cd68, self.z, self.depth, self.batch_mode),
            projection=True) if self.cd68 is not None else np.zeros_like(plane_iba1))
        left_mask, right_mask = self.panel_masks()

        data = {
            LAYER_ORDER[0]: plane_iba1, LAYER_ORDER[1]: plane_cd68, LAYER_ORDER[2]: left_mask,
            LAYER_ORDER[3]: plane_iba1, LAYER_ORDER[4]: plane_cd68, LAYER_ORDER[5]: right_mask,
        }
        limits = {
            LAYER_ORDER[0]: self.contrast_limits, LAYER_ORDER[3]: self.contrast_limits,
            LAYER_ORDER[2]: (0.0, 1.0), LAYER_ORDER[5]: (0.0, 1.0),
        }
        for index, name in enumerate(LAYER_ORDER):
            concept = CONCEPTS[index % GRID_STRIDE][0]
            plane = data[name]
            if plane is None:
                continue
            row = self.rows[concept]
            self.viewer.show_image(name, plane, colour=row.colour().name, alpha=row.alpha(),
                                   visible=self.toggles.is_on(concept),
                                   contrast_limits=limits.get(name))

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
        assignment is computed from the layer's index, not from its visibility.
        """
        for layer in self._layers_for(concept):
            if self.viewer is not None and layer in self.viewer.layer_names():
                self.viewer.set_visible(layer, on)

    # -- the commit gate ------------------------------------------------

    def _on_coefficient_moved(self, value: float) -> None:
        """Moving the slider invalidates the commit **and drops both panels back to raw.**

        > ### ⇒ IT REDREW ON EVERY STEP, WHICH IS THE THING ITEM 3 REFUSES TO DO — item A6
        >
        > `value_changed` fires at every one of the slider's steps, and this called `draw()`
        > unconditionally: two `preview_mask` calls at **0.25 s** each — cache-missing every
        > time, because the coefficient is what changed — plus six full-field plane uploads.
        >
        > **`screen_threshold1._invalidate_view`'s guard, verbatim in intent:** *"Redraw only to
        > swap the panels back, never merely because the gate moved."* The panels only need
        > swapping if there was a commit to drop; after that first step there is nothing left to
        > swap and the recompute belongs on `sliderReleased`, where `refresh_preview` already
        > sits.
        >
        > **The invalidation itself still happens on every step, deliberately.** A user who
        > drags and never lets go must not be left with a gate that still says Continue.
        """
        was_committed = self.is_committed
        self.is_committed = False
        self.flow.invalidate()
        self._sync()
        if was_committed:
            self.draw()

    def _on_parameter_changed(self) -> None:
        self.is_committed = False
        self.flow.invalidate()
        self._sync()

    def parameter_error(self) -> str | None:
        try:
            if float(self.mask_closing_radius_um.text().strip()) < 0:
                return "mask_closing_radius_um cannot be negative"
        except ValueError:
            return "mask_closing_radius_um must be a number"
        #: Item 3's wording, and the same two failure modes in the same order — a negative
        #: floor and a non-integer one are different mistakes and say so separately.
        try:
            if int(self.measurement_min_object_voxels.text().strip()) < 0:
                return "min_object_voxels cannot be negative"
        except ValueError:
            return "min_object_voxels must be a whole number"
        return None

    def commit_result(self, segmentation_cells: np.ndarray, measurement_cells: np.ndarray,
                      cd68: np.ndarray | None = None) -> None:
        """Record a finished commit: both panels become cells, and the table gains a row."""
        problem = self.parameter_error()
        if problem is not None:
            self.commit.set_working(problem)
            return
        self.segmentation_cells = segmentation_cells
        self.measurement_cells = measurement_cells
        self.is_committed = True

        source = cd68 if cd68 is not None else self.cd68
        #: **`None`, not `0.0`.** The fallback zero was a plausible number for *no channel*, and
        #: `table_row` is where it becomes the em dash the user asked for.
        inside = cd68_inside_fraction(source, measurement_cells) if source is not None else None
        self.table.add_row(table_row(
            self.mask_coefficient.value(),
            float(self.mask_closing_radius_um.text().strip()),
            inside, mask_volume_fraction(measurement_cells)))

        self.flow.commit(measurement_cells)
        self.commit.set_working("")
        self._sync()
        self.draw()

    def _on_commit(self) -> None:
        """Flip the gate, then ask for the two masks the button names. ~59 s.

        `commit_result` is what finishes it — both panels become cells and the table
        gains a row — and it was built at task 13 for a caller that did not exist.
        """
        problem = self.parameter_error()
        if problem is not None:
            self.commit.set_working(problem)
            return
        self.flow.commit(None)
        self._sync()
        self.commit.set_working(COMMIT_PROGRESS)
        try:
            self.commit_requested.emit()
        finally:
            self.commit.set_working("")

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
        chosen = {"mask_coefficient": float(self.mask_coefficient.value())}
        chosen.update(parsed_numbers({
            "mask_closing_radius_um": (self.mask_closing_radius_um.text(), float),
            "measurement_min_object_voxels": (
                self.measurement_min_object_voxels.text(), int),
        }))
        return chosen

    def _sync(self) -> None:
        self.flow.session.chosen.update(self.parameters())
        self.commit.set_stale(self.flow.is_stale())
        self.continue_button.setEnabled(self.flow.can_continue())
        #: §6i.7's disabled state, **carried twice**: the greyed control for everyone, the
        #: sentence for anyone who cannot separate the two greys. That is why the visual half
        #: can afford to be subtle.
        self.z_depth.setEnabled(self.is_committed)
        #: **The toggle inherits the depth box's gate**, because it modifies it: before a commit
        #: only one slice has been computed, so there is no window to step by.
        self.batch_step.setEnabled(self.is_committed and self.depth > 1)
        self.depth_note.setVisible(not self.is_committed)

    def _on_continue(self) -> None:
        if self.flow.can_continue():
            self.continue_requested.emit()
