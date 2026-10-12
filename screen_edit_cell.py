"""Screen 7b — Cell Review, Edit Cell. `COMPONENT_05_PLAN.md` §3b, §3c, §3d.4–§3d.6, §3d.8.

**Frame C — item 2 and 4's frame, already built.** Colours and layers left, one view centre,
controls right. *"What it is for: everything a thumbnail cannot settle."*

> ### IT ALWAYS RESOLVES THE CELL, AND THAT IS WHAT MAKES TRIAGE TRUE
>
> §3a decision 4. There is no way to leave rich review with the cell still undecided — which is
> what makes triage's *"a single cell is resolved only by entering rich review"* a fact about the
> program rather than a convention. `Accept` and `Reject` are the only exits.

**`ndisplay=3` only. There is no slice mode and no z control of any kind** — no slider, no arrow
keys, no `(23/40)` indicator. Rotate, pan, zoom. **The second screen in the project with no z**,
after item 6, and Component 3's z conventions do not reach here.

> ### NO COMMIT, NO GATE, NO `(!)`, NO BACK
>
> **Fourth screen with no COMMIT**, after items 1, 6 and 9a. Nothing here lags a control, so
> there is nothing to guard: every edit is applied to the mask in memory the moment it is made,
> and `u` unwinds them until the cell resolves.

**Everything that is not Qt is in `editing.py`**, because both routes into this screen — item 7's
in-memory session and `review --pool` — need it and neither owns it.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import editing, palette
from .gestures import MARKER_COLOUR, MARKER_LAYER, SplitMarkers
from .frames import CanvasHost, FrameC
from .preview import contrast_limits
from .viewer import DEFAULT_RENDERING, OFFERED_RENDERING, CanvasLayout, ViewerPort
from .widgets import (
    ChoicePopup, ColourRow, ConfirmPopup, CountPopup, HelpButton, HelpPopup, LayerToggles,
    ParameterSlider, ask_rejection, cell_count_popup, reject_reason_popup,
)

__all__ = [
    "EditCellScreen", "LAYERS", "PICKER_LAYERS", "TOGGLES", "CELL_LAYER", "WIDENED_LAYERS",
    "DEFERRED_LAYERS", "MARGIN_RANGE", "DEFAULT_MARGIN_UM", "crop_box", "margin_voxels",
    "RenderingChoice",
]

#: The editable layer's name. **Not in `PICKER_LAYERS`, and it cannot be** — its colouring is a
#: per-region *scheme* (white soma, warm six graph-coloured) rather than a single hue, and its
#: alpha carries *inside versus outside the process you have entered*. §4's layer-appearance rule
#: therefore excludes it from both the picker and the translucency slider, exactly as it excludes
#: item 4's nucleus labels and item 6's size ramp. **Third such layer in the project.**
CELL_LAYER = "this cell"

#: §3d.5 and §3d.6, and the order is the reading order of the drawn `Colours` block.
#:
#: **Five pickers, not the three §3c drew** — the 2026-08-19 nucleus split created `other nuclei`
#: and `DAPI as detected` (named `raw DAPI` until 2026-08-25, when it started carrying the
#: detector's own view instead of nothing), and neither had a colour entry until §3d.8.
#:
#: **`everything else` IS GONE, AND IT HAD BEEN DEAD SINCE 2026-08-25.** It became `other
#: cells` + `debris` that day and its name was never in `LAYERS` afterwards, so no layer ever
#: asked `_volume_for` for it — but `ReviewStep._enter` went on computing `_scene_around` and
#: handing it over on every single cell entry, for a layer that could not be drawn. Deleted
#: end-to-end 2026-09-11 with the keyword, the attribute and `_scene_around` itself.
#:
#: **`DAPI as detected` shows every nucleus in the crop**, including hidden cells' and neurons',
#: because DAPI cannot be filtered by cell — *"precisely what it is for."*
#:
#: `None` colour and `None` alpha mark the scheme layer, as item 4's table does.
#: > ### FOUR ARRIVE ON AND THE OTHER THREE ARE NOT CUT AT ALL — 2026-09-11
#: >
#: > All seven layers used to be built, uploaded to the GPU and *then* hidden on every single
#: > entry: `viewer.show_image` assigns `layer.data` — the texture upload — and only afterwards
#: > sets `layer.visible`. Measured on the study's own pool, **12.5 of the 21.8 megavoxels
#: > pushed per cell were for a layer the reviewer could not see.**
#: >
#: > The user's decision: **`this cell`, `Iba1 signal`, `DAPI as detected` and `nucleus` arrive
#: > on.** `other nuclei`, `other cells` and `debris` arrive off, are **not cut and not
#: > uploaded**, and are built the first time they are switched on — a slow first toggle, which
#: > the user accepted in exchange for every entry being fast.
#: >
#: > `DAPI as detected` moved from off to on in the same decision. It shows every nucleus in
#: > the crop, including finished cells' and neurons', because DAPI cannot be filtered by cell —
#: > which is *"precisely what it is for"* and is now what the reviewer sees on arrival.
#: >
#: > **`other cells` is the one that mattered most for speed.** It is the layer the neighbour
#: > somata are computed for, so deferring it takes `_other_cells_around` — the single largest
#: > item in the compute half — off the entry path altogether.
LAYERS: tuple[tuple[str, str | None, float | None, bool], ...] = (
    ("Iba1 signal", "white", 0.15, True),
    (CELL_LAYER, None, None, True),
    ("nucleus", "blue", 1.0, True),
    ("other nuclei", "violet", 0.7, False),
    ("DAPI as detected", "blue", 0.35, True),
    ("other cells", "green", 0.75, False),
    ("debris", "yellow", 0.5, False),
)

#: The three that are **not built until the reviewer asks for them**, and the driver that owns
#: the scene is what builds them — see `context_wanted`. `other cells` and `debris` are two
#: readings of one scene crop and are therefore cut as a pair.
DEFERRED_LAYERS: frozenset[str] = frozenset({"other nuclei", "other cells", "debris"})

#: **`everything else` became two layers on 2026-08-25.** One flat yellow could not tell a
#: neighbouring cell from unclaimed material, so the question the layer exists to answer —
#: *does this belong to me or to the cell next to it* — had no visible answer.
#:
#: **Green for the neighbours.** `palette.LABEL_RING` splits deliberately: the warm six colour
#: this cell's decomposition and the cool six let the nucleus layers *"sit without competing"*.
#: Green is outside both families and away from the blue and violet nuclei, so a neighbour can
#: never be misread as one of this cell's own processes. PrestoCell reaches for the same hue
#: for the same job — its edited cell is green out of a red field.
#:
#: **Yellow stays with debris**, which is what `everything else` always meant: the leftover.
EVERYTHING_ELSE_SPLIT = ("other cells", "debris")

#: `other cells` renders in two treatments off ONE picked hue: the soma faint, the processes
#: solid. The user's choice, 2026-08-25, and the right way round — what has to be adjudicated
#: is whether a *process* belongs to this cell or the neighbour, so the processes stay legible
#: while the neighbour's body drops back to a landmark. Measured reason, not a guess: the soma
#: is 53-84% of cell volume on this tissue (`EXPORT_SPEC.md` §28), so a solid soma would be a
#: wall of colour with the processes lost against it.
OTHER_SOMA_ALPHA_SCALE = 0.33

#: **And the soma is also LIGHTENED — the user's decision, 2026-08-27.** They reported that the
#: neighbour's soma was not legible, and chose *"a few shades lighter (needs to be obviously
#: lighter)"* **beside** the existing alpha rather than instead of it. So `other cells` now
#: renders one picked hue in two treatments that differ on **two** channels: the soma pale and
#: faint, the processes saturated and solid.
#:
#: **Why lightness is the channel that was added.** Alpha alone had to carry the soma/process
#: distinction *and* answer the layer's translucency slider, so moving the slider degraded the
#: encoding it was moving. Lightness is independent of the slider and is the channel dichromacy
#: leaves intact — the same argument that gives this cell's own soma its white.
#:
#: **0.60 is a starting value, not a measured one.** It is deliberately strong because the user
#: asked for obvious; `palette.lighten`'s docstring records why the effect reads as *desaturated
#: pale against saturated solid* rather than as a brightness inversion, which it cannot be at a
#: reduced alpha on a dark canvas. **Adjust here after seeing it on the real crop.**
OTHER_SOMA_LIGHTEN = 0.60

#: Layers drawn as napari `Labels` rather than as volumes, so the rendering radio must skip
#: them. `this cell` was always one; `other cells` joined on 2026-08-25.
LABEL_LAYERS: frozenset[str] = frozenset({"other cells"})

#: Layers cut **wider than the entered cell** and therefore pushed with a `translate`. The
#: margin control drives their width, which is the first thing it has ever driven: `crop_box`
#: clips to the array it is handed, and every other layer is handed the bounding box itself.
#:
#: > ### IT WAS TWO LAYERS OUT OF FIVE, AND THE MARGIN LOOKED HALF-BROKEN — 2026-08-26
#: >
#: > `other cells` and `debris` widened; `Iba1 signal`, `other nuclei` and `DAPI as detected`
#: > did not, so pulling the margin out grew the neighbours and the debris **into a black
#: > field** — the channel they are made of stopped at the old boundary. The user's report was
#: > that the margin *"only seems to expand other cells and debris"*, which is exactly what it
#: > did.
#: >
#: > **The three that joined are the three with data outside the box.** `this cell` and
#: > `nucleus` are the entered cell's own masks and are bbox-shaped **by definition** — there
#: > is nothing of them beyond it, so widening them would pad zeros and show nothing. They stay
#: > narrow, and they are the frame the `translate` is measured against.
#: >
#: > **`Iba1 signal` needs a second array and the others do not.** `set_cell`'s `intensity`
#: > argument is what `CellEditor` decomposes, so it must stay the cell's own crop; widening it
#: > would move the editing frame. `dapi` and `other_nuclei` are display-only and simply arrive
#: > wide. That asymmetry is why `iba1_context` exists as its own parameter.
#: **Always** pushed whole: the caller builds them at the context box or not at all.
CONTEXT_LAYERS: frozenset[str] = frozenset({"other cells", "debris"})

#: Pushed whole **only when a context frame exists**, i.e. when `context_origin` is set.
#:
#: > ### THE DISTINCTION IS NOT DECORATION - IT IS THE OTHER ROUTE'S MARGIN
#: >
#: > These three are cut from full volumes, so who cuts them depends on the caller. `ReviewStep`
#: > cuts them at the context box and sets `context_origin`; a screen driven directly - the
#: > standalone route, and every test that builds one - hands over arrays **larger** than the
#: > cell and lets `draw` slice them with `crop_box`, which grows with the margin because the
#: > mask's extent is smaller than the array it sits in.
#: >
#: > **Putting them in `CONTEXT_LAYERS` outright broke that**, because a context layer bypasses
#: > `crop_box` entirely: the margin stopped changing the Iba1 crop on the route where it was
#: > the only thing that ever worked. Caught by
#: > `test_the_margin_takes_effect_on_RELEASE_and_not_while_dragging`, whose fixture is a small
#: > cell in a large volume for exactly this reason - and whose docstring already records being
#: > the third fixture of that build that could not exercise its own assertion.
WIDENED_LAYERS: frozenset[str] = frozenset({
    "Iba1 signal", "other nuclei", "DAPI as detected",
})

#: `set_context`'s "leave this one alone" marker. **`None` cannot do that job here**: it is the
#: legitimate value for a layer that has no data, and `_rewiden` replaces some layers while a
#: caller with no scene replaces none.
_KEEP = object()


def world_scale(voxel_size_um) -> tuple[float, float, float]:
    """The layer scale that makes this screen's 3D view **physically shaped**.

    > ### THE VOLUME WAS 5.5x TOO FLAT, AND ONLY THIS SCREEN COULD SHOW IT
    >
    > Every layer here was pushed **without a scale**, so napari rendered in raw voxel space —
    > 40 slices against 427 rows rather than 24 µm against 46 µm. Measured on crop2: the physical
    > z:y is 0.519 and the rendered z:y was 0.094, **flat by 5.54x, which is exactly the
    > anisotropy** 0.6 / 0.1083. That equality is what identifies the cause rather than merely
    > being consistent with it.
    >
    > **Pre-existing** — the pre-debris backup omits it too — and invisible from directly above,
    > which is how it survived. Rich review is the **only** screen at `ndisplay=3`; every other
    > one draws a z-slice or a z-projection, where the aspect is y:x and therefore isotropic. So
    > this is the one picture anisotropy could distort, and it did.
    >
    > **No measurement was ever affected *by this*.** Nothing in the export path reads a viewer.
    > Found by the user rotating the view, 2026-08-25.
    >
    > **The claim that used to follow — that the computation "handles anisotropy throughout" —
    > was too strong, and is corrected here rather than deleted.** Every item it listed is true:
    > physical structuring elements, `sampling=` on every distance transform,
    > `marching_cubes(spacing=...)` and µm-weighted skeleton edges. But *throughout* was not,
    > because the skeleton those edges weighted was **thinned in voxel space** — a correctly
    > weighted graph over the wrong skeleton. One process read 5.40 µm or 11.95 µm depending on
    > its orientation. Fixed 2026-08-31 by resampling to an isotropic grid before thinning
    > (`morphometrics.SKELETON_TARGET_UM`, `PROJECT_LOG.md` §54), so the sentence is now true —
    > but it was written before it was earned.
    """
    return tuple(float(v) for v in voxel_size_um)

PICKER_LAYERS: tuple[tuple[str, str, float, bool], ...] = tuple(
    (name, colour, alpha, on) for name, colour, alpha, on in LAYERS if colour is not None
)

#: Every layer with a **translucency slider** — the pickers plus `this cell`. §4's rule used to
#: exclude a scheme layer from both controls because they arrived as one row; they are separable
#: since 2026-08-25 and the user asked for this one's slider back. Its colour stays unpickable:
#: the warm six are graph-coloured so that *adjacent* processes always differ, which is
#: functional for editing rather than decorative, and its per-region alphas are MULTIPLIED by
#: the slider so the entered-process cue survives every setting.
SLIDER_LAYERS: tuple[tuple[str, str | None, float, bool], ...] = tuple(
    (name, colour, 1.0 if alpha is None else alpha, on) for name, colour, alpha, on in LAYERS
)

#: Six checkboxes, one per layer. **No `Outline` toggle here** — item 4 has one because a
#: nucleus fill hides what is under it; a decomposition *is* the thing being looked at.
TOGGLES: tuple[tuple[str, bool], ...] = tuple((name, on) for name, _, _, on in LAYERS)

#: §3b: *"The display margin is a slider in µm, taking effect on release."* PrestoCell's
#: equivalent is `crop_by_idx(pred, idx, padding=100)` — the bbox expanded by 100 **pixels** in
#: all three axes identically, which in this project's geometry is 10.8 µm laterally and **the
#: entire stack in depth**, since 100 slices exceeds 40. That is the isotropic-pixels trap in its
#: original form, the same one `closing_radius_um` and `plaque_dilate_um` are xy-only to avoid.
#: **Ours is in µm**, so it means the same distance in every direction.
MARGIN_RANGE = (0.0, 20.0)
DEFAULT_MARGIN_UM = 6.0


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

def margin_voxels(margin_um: float, voxel_size_um) -> tuple[int, ...]:
    """The margin in voxels **per axis**, which at 5.5:1 anisotropy is not one number.

    6 µm is 20 voxels in xy at 0.3 µm and 4 in z at 1.65 — and asking for the same *pixel*
    padding in both is what makes PrestoCell's crop take the whole stack in depth.
    """
    return tuple(max(0, int(round(float(margin_um) / float(v)))) for v in voxel_size_um)


def crop_box(mask: np.ndarray, margin_um: float, voxel_size_um) -> tuple[slice, ...]:
    """The cell's own bounding box, grown by the margin and clipped to the volume.

    **A view parameter rather than an artefact**, which is exactly why it can move at runtime:
    the crop is read from the source every time rather than being a stored thumbnail.
    """
    solid = np.asarray(mask).astype(bool)
    if not solid.any():
        return tuple(slice(0, dim) for dim in solid.shape)
    pad = margin_voxels(margin_um, voxel_size_um)
    box = []
    for axis, (extent, dim) in enumerate(zip(np.nonzero(solid), solid.shape)):
        box.append(slice(max(0, int(extent.min()) - pad[axis]),
                         min(dim, int(extent.max()) + 1 + pad[axis])))
    return tuple(box)


class RenderingChoice:
    """**One offered mode, which is therefore the decision.** §3c, and `viewer.OFFERED_RENDERING`.

    Held as a class so the screen and the tests name the same value, and so the default is a
    value rather than a radio button's index.

    > ### THE RADIO IS GONE, AND THE `LABELS` TABLE WENT WITH IT — 2026-09-11
    >
    > Three modes were offered and drawn as three radio buttons, with `LABELS` mapping napari's
    > identifiers to what the screen wrote (`attenuated mip`, without the underscore). The
    > user's decision leaves **one** mode, and a radio group with a single button is a control
    > that cannot be used — so the screen draws none and the table has nothing left to name.
    >
    > `set_rendering` stays, and still refuses anything not in `OFFERED`. What was a control is
    > now an invariant.
    """

    OFFERED = OFFERED_RENDERING
    DEFAULT = DEFAULT_RENDERING


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

class EditCellScreen(QWidget):
    """Frame C: five pickers and six toggles left, the 3D crop centre, controls right."""

    #: **The one screen in the project at ndisplay=3**, and the only reason this
    #: value is applied on every arrival rather than once: 3D is viewer state, so
    #: leaving it set is what put items 9a and 9b in 3D for the rest of a session.
    CANVAS = CanvasLayout(ndisplay=3)

    resolved = Signal(object)               # editing.Resolution
    children_produced = Signal(object)      # list[np.ndarray], after a manual split

    #: One take, in the **editable** frame, emitted after it has gone into the mask.
    #:
    #: **The screen does not know who lost it, and must not.** It holds one cell's crops; the
    #: payloads and the scene live on `ReviewStep`, which is the only place that can say which
    #: neighbour those voxels belonged to. So this carries the material and nothing else, and
    #: the join is made where the join can be made.
    material_taken = Signal(object, object)     # (status, material)
    #: The display margin moved. Carries the new value in µm, and is the seam that lets the
    #: context layers be **re-cut** from the full volume — this screen holds crops, not the
    #: scene, so it cannot widen them itself.
    margin_changed = Signal(float)
    #: A `DEFERRED_LAYERS` layer was switched on and has no data. **The same seam as
    #: `margin_changed`, for the same reason**: the array has to be cut from the whole volume
    #: and this screen holds crops rather than the scene. Carries the layer's name; whoever
    #: owns the scene cuts it and calls `set_context`, and the screen draws afterwards.
    #:
    #: A screen with no driver connected simply keeps the layer empty, which is what every
    #: standalone test presents and what the semi-automatic route has always presented for a
    #: layer it does not populate.
    context_wanted = Signal(str)

    def __init__(self, viewer: ViewerPort | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.viewer = viewer
        self.editor: editing.CellEditor | None = None
        #: **Item 6.5's four, held on the screen so re-entry keeps them.** Seeded from
        #: `CellEditor`'s own defaults, so a caller that supplies none behaves exactly as
        #: before; `set_cell` updates only what it is given. See `set_cell`.
        self.soma_params: dict = {}
        #: Shift-held split markers, shared with item 4 — `gestures.SplitMarkers`.
        self.markers = SplitMarkers()
        self.voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.margin_um = DEFAULT_MARGIN_UM

        #: The four volumes the screen draws besides the cell itself. Held rather than passed to
        #: `draw()`, for item 4's reason: `draw` is called from six places and a per-call
        #: argument list silently drops a layer the first time one of them forgets it.
        #: **Per-channel display range from item 1**, keyed by role; empty means the
        #: percentile stretch. Set by the flow driver on arrival — see `app.prepare`.
        self.display_range: dict[str, tuple[float, float]] = {}
        self.iba1: np.ndarray | None = None
        self.dapi: np.ndarray | None = None
        self.nucleus: np.ndarray | None = None
        self.other_nuclei: np.ndarray | None = None

        #: Replaced in tests, as item 4's `confirm_discard` is. The real ones show §3c's popups.
        self.ask_reject_reason = self._ask_reject_reason
        self.ask_cell_count = self._ask_cell_count

        #: Children waiting to be dealt with after a split. §9.2: *"presented one at a time in
        #: the same environment, each edited and then accepted or rejected, and the reviewer
        #: returns to triage only once all of them are dealt with."*
        self.pending: list[np.ndarray] = []

        self.frame = FrameC("Cell Review — Edit Cell", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: five pickers, then six toggles -------------------------
        self.frame.section("Colours", HelpButton(
            "Colours",
            "Each row sets one layer's colour and how much of the picture it is entitled to.\n\n"
            "Layers add rather than cover one another, so the slider is a weight and the "
            "stacking order changes nothing.\n\n"
            "This cell has a slider but no swatches: its soma is white and its processes take "
            "a scheme, so there is no single colour to pick. Its slider scales the whole layer "
            "and keeps the process you are inside brighter than the rest.", self))

        self.rows: dict[str, ColourRow] = {}
        for name, colour, alpha, _ in SLIDER_LAYERS:
            row = ColourRow(name, colour, alpha, self)
            row.colour_changed.connect(lambda c, n=name: self._recolour(n, c))
            row.alpha_changed.connect(lambda a, n=name: self._reweight(n, a))
            self.frame.add_left(row)
            self.rows[name] = row

        #: *"this cell: a scheme — no picker."* Written out on the screen, exactly as item 4
        #: writes its nucleus labels and item 6 its ramp, so the absence reads as a decision
        #: rather than as something missing.
        self.scheme_note = QLabel("this cell:\n a scheme — no picker", self)
        self.scheme_note.setProperty("role", "helper")
        self.frame.add_left(self.scheme_note)

        #: §3b's *"a thin fragment with no soma reads as one process with no soma colour, and
        #: the screen says why"*. Empty and hidden whenever there is a soma.
        self.soma_note = QLabel("", self)
        self.soma_note.setProperty("role", "helper")
        self.soma_note.setWordWrap(True)
        self.soma_note.setVisible(False)
        self.frame.add_left(self.soma_note)

        self.frame.section("Layers", HelpButton(
            "Layers",
            "DAPI as detected is the nuclei channel itself and cannot be filtered by cell, "
            "so it shows every nucleus in the crop — including those of finished cells and of "
            "neurons. It is the only place a nucleus shared between two cells is visible at "
            "all.\n\n"
            "Other cells and debris are the above-threshold signal this cell does not own: the "
            "neighbours, and the material that belongs to nothing.", self))
        self.toggles = LayerToggles(TOGGLES, self)
        self.toggles.toggled.connect(self._toggle)
        self.frame.add_left(self.toggles)
        self.frame.left.addStretch(1)

        # -- centre: the crop, and nothing written over it ----------------
        self.canvas = CanvasHost(self, minimum_height=360)
        self.frame.add_centre(self.canvas, 1)

        # -- right: margin, rendering, gestures, the two buttons ----------
        self.margin = ParameterSlider(
            "display margin", MARGIN_RANGE[0], MARGIN_RANGE[1], DEFAULT_MARGIN_UM,
            decimals=1, help_button=HelpButton(
                "display margin",
                "How much of the neighbourhood is shown around the cell, in microns.\n\n"
                "Takes effect when you let go of the slider.\n\n"
                "It is a view setting, not a measurement: the crop is read from the source "
                "every time, so nothing about the cell changes when you move it.", self),
            parent=self)
        #: **Release-only, and there is nothing to gate.** Every other slider in the project
        #: that recomputes on release also invalidates a commit at the first step of the drag;
        #: this screen has no commit, so the drag simply does nothing until it ends.
        self.margin.slider.sliderReleased.connect(self._on_margin_released)
        self.frame.add_right(self.margin)

        #: **No rendering control.** One mode is offered — see `RenderingChoice` — so there is
        #: nothing to choose between and a single radio button would be a control that cannot
        #: be used. The value is still applied to every volume layer on every draw.
        self.rendering = RenderingChoice.DEFAULT

        self.frame.right.addStretch(1)

        #: §3c: **the largest legend in the project, behind one mark.** Drawn out it consumed
        #: the right column, *"and a screen whose controls are mostly invisible gestures is
        #: better served by one honest mark than by a wall of text nobody reads twice."*
        gestures_row = QHBoxLayout()
        gestures_row.addWidget(QLabel("gestures"))
        gestures_row.addStretch(1)
        gestures_row.addWidget(HelpButton("gestures", editing.GESTURE_LEGEND, self))
        gestures_host = QWidget(self)
        gestures_host.setLayout(gestures_row)
        self.frame.add_right(gestures_host)

        #: **The split's progress line**, because at 0.7–2.5 s shift-release cannot be silent.
        self.progress = QLabel("", self)
        self.progress.setProperty("role", "helper")
        self.frame.add_right(self.progress)

        #: §3d.8: **`Reject` looks exactly like `Accept`.** A red label was proposed — `debris`
        #: rejection is the only irreversible action in the program — and **the user declined
        #: it: the popup is the guard.** Consistent with the interface's standing posture, and
        #: with the rejection popup already saying the deletion cannot be undone.
        buttons = QHBoxLayout()
        self.reject_button = QPushButton("Reject", self)
        self.accept_button = QPushButton("Accept", self)
        self.reject_button.clicked.connect(self.reject)
        self.accept_button.clicked.connect(self.accept)
        buttons.addWidget(self.reject_button)
        buttons.addWidget(self.accept_button)
        buttons_host = QWidget(self)
        buttons_host.setLayout(buttons)
        self.frame.add_right(buttons_host)

    # -- loading --------------------------------------------------------

    def set_cell(self, intensity, cell_mask, nucleus_mask=None, *,
                 voxel_size_um=(1.0, 1.0, 1.0), thickness_um: float | None = None,
                 soma_params: dict | None = None,
                 dapi=None, other_nuclei=None,
                 other_cells=None, debris=None, context_origin=None,
                 iba1_context=None, editor=None) -> None:
        """Take one cell and decompose it. **`ndisplay=3` is set here, once, at arrival.**

        > ### THE SOMA PARAMETERS PERSIST ACROSS RE-ENTRY, AND THEY HAD TO — 2026-08-30
        >
        > `thickness_um` was a keyword with a default of `0.9`, and **the two internal re-calls
        > did not pass it**: entering a split child, and moving to the next pending child, both
        > silently reset the soma to the built-in value mid-review. Nothing showed it, because
        > nothing had ever set a non-default.
        >
        > They are held on the screen instead and updated only when supplied, so `None` means
        > *keep what we were given* rather than *use the default*. The two re-calls need no
        > change, which is what makes the fix hold: there is no third caller to remember.
        """
        if thickness_um is not None:
            self.soma_params["thickness_um"] = float(thickness_um)
        if soma_params:
            self.soma_params.update(soma_params)
        self.voxel_size_um = tuple(float(v) for v in voxel_size_um)
        self.iba1 = np.asarray(intensity)
        #: **The same channel, cut at the context box, for the LAYER only.** `self.iba1` is what
        #: `CellEditor` decomposes and it must stay the cell's own crop; this is what
        #: `Iba1 signal` draws. `None` falls back to the narrow one, which is what every caller
        #: without a whole volume to cut from hands over.
        self.iba1_context = None if iba1_context is None else np.asarray(iba1_context)
        self.nucleus = None if nucleus_mask is None else np.asarray(nucleus_mask)
        self.dapi = None if dapi is None else np.asarray(dapi)
        self.other_nuclei = None if other_nuclei is None else np.asarray(other_nuclei)
        #: **Two arrays since 2026-08-25**, split out of the single `everything_else` mask.
        #: `other_cells` carries the neighbours' Iba1 as a small label volume — 1 soma, 2
        #: process — so one picked hue can render in two treatments; `debris` is a plain mask.
        self.other_cells = (None if other_cells is None else np.asarray(other_cells))
        self.debris = (None if debris is None else np.asarray(debris))
        #: Where the context layers sit relative to this cell's crop, in voxels — zero or
        #: negative on every axis, because the context box only grows outward. Handed to the
        #: viewer as `translate` so the wide layers stay registered against the narrow ones.
        #:
        #: > ### THE SIGN FLIPPED ON 2026-08-26, AND THAT IS THE POINT
        #: >
        #: > It used to be zero or negative on every axis, because the context box grew outward
        #: > past a frame that could not follow. The editable frame is now the bounding box
        #: > padded by the margin slider's **maximum**, so it is the widest box on the screen
        #: > and everything else lives inside it — the offset is zero or positive. The three
        #: > places that read it take either sign, because the pool-backed route can still
        #: > present the old arrangement.
        self.context_origin = (None if context_origin is None
                               else tuple(float(v) for v in context_origin))

        #: > ### ⇒ `editor` IS A PREFETCHED `CellEditor`, OR `None` FOR TODAY'S BEHAVIOUR
        #: >
        #: > `HANDOFF_PATCH_CPU.md` §4 B6. Building one is **1.83 s** measured — 18.7% of what
        #: > our own code does on a cell entry — and the pool route builds them ahead, on the
        #: > triage grid, while the reviewer is looking at thumbnails.
        #: >
        #: > **The caller has already checked that it was built from this cell's current mask**
        #: > (B5c, `ReviewStep._enter`), so there is nothing to verify here. `None` means build
        #: > it now, which is both the semi-automatic route and B3's fallback for a reviewer who
        #: > outran the prefetch.
        self.editor = editor if editor is not None else editing.CellEditor(
            self.iba1, cell_mask, self.nucleus,
            voxel_size_um=self.voxel_size_um, **self.soma_params)
        self.progress.setText("")
        self.markers.clear()
        if self.viewer is not None:
            self.viewer.set_ndisplay(3)
        self._show_soma_note()
        self.draw()

    def _world_offset(self, box=None):
        """`context_origin` in **world** units, which is the frame `translate` is read in.

        napari composes as `rotate · shear · scale + translate` — its own docstring — so the
        translate is applied **after** the scale and is therefore in µm, not voxels. The offset
        is captured in voxels because that is what the array slicing produced; multiplying here
        is the one conversion, at the one place that knows both.

        Get this wrong and the context layers slide by the anisotropy factor — a wrong picture
        that looks entirely plausible, which is the failure mode this screen keeps meeting.
        """
        if self.context_origin is None:
            return None
        #: **Relative to the DRAWN CROP, not to the editable frame.** `context_origin` is
        #: measured from the editable frame's corner; the layers it registers against are drawn
        #: from `_shown_box`'s corner, which since the frame widened is no longer the same
        #: place. Subtracting the box's own start is the one line that keeps the two in step,
        #: and it is zero in the old bounding-box world, which is why nothing needed it before.
        #:
        #: **`box` is that same crop, handed in rather than recomputed.** `crop_box` runs
        #: `np.nonzero` over the whole editable frame — 0.093 s per entry across the six calls
        #: `draw` used to make, one of its own and one per context layer through here. It is
        #: the same box every time, so `draw` takes it once and passes it down.
        if box is None:
            box = None if self.editor is None else self._shown_box()
        start = [0, 0, 0] if box is None else [int(part.start) for part in box]
        return tuple((float(o) - float(b)) * float(v)
                     for o, b, v in zip(self.context_origin, start, self.voxel_size_um))

    def _show_soma_note(self) -> None:
        note = "" if self.editor is None else self.editor.note()
        self.soma_note.setText(note)
        self.soma_note.setVisible(bool(note))

    # -- drawing --------------------------------------------------------

    def _volume_for(self, name: str):
        return {"Iba1 signal": (self.iba1 if self.iba1_context is None
                                else self.iba1_context),
                "nucleus": self.nucleus,
                "other nuclei": self.other_nuclei,
                "DAPI as detected": self.dapi,
                "other cells": self.other_cells,
                "debris": self.debris}.get(name)

    def _limits_for(self, name: str):
        """One layer's contrast limits, **computed only if that layer is being pushed.**

        **Only the two intensity layers take a range.** `nucleus`, `other nuclei`, `other
        cells` and `debris` are masks, and napari's own 0-1 handling is right for them.

        > ### ⇒ THE COMMENT SAID "FROM THE WHOLE VOLUME" AND IT NEVER WAS
        >
        > It read *"the ranges are taken from the WHOLE volume, not the crop: a cell sitting in
        > a dim corner would otherwise be stretched to look as bright as one in a plaque."*
        > `self.iba1` is the **editable frame**, so what is actually taken is a per-cell
        > stretch and the dim-corner cell is exactly what happens. The claim is removed rather
        > than the behaviour changed: taking them from the whole channel is a different picture
        > on every cell in the study, it costs `np.percentile` over ~212 Mvox, and it is not a
        > decision to make inside a speed fix. **Left for the user** — see the note in the
        > handover.
        """
        if name == "Iba1 signal":
            return (contrast_limits(self.iba1, self.display_range.get("cell"))
                    if self.iba1 is not None else None)
        if name == "DAPI as detected":
            return (contrast_limits(self.dapi, self.display_range.get("nuclei"))
                    if self.dapi is not None else None)
        return None

    def draw(self) -> None:
        """Push the five image layers, then the cell **last, so the labels sit on top**.

        The label layer is `translucent`, §6f.3's one exception, and not a style choice:
        additive would sum a region's hue with the signal beneath it and corrupt the one thing
        this layer exists to carry.
        """
        if self.viewer is None or self.editor is None:
            return

        box = crop_box(self.editor.mask, self.margin_um, self.voxel_size_um)
        #: **A layer that is off and has never been pushed is not pushed now.** `show_image`
        #: assigns `layer.data` — the texture upload — and only then sets `layer.visible`, so
        #: drawing a hidden layer uploads a multi-megavoxel volume to the GPU and immediately
        #: hides it. Once a layer exists it keeps being refreshed, because the reviewer can
        #: switch it back on without anything being recut.
        already = set(self.viewer.layer_names())
        #: **Every layer, or none.** A scaled image beside an unscaled label layer is worse than
        #: both being wrong together: the two would disagree about where the same voxel is.
        scale = world_scale(self.voxel_size_um)
        for name, _, _, _ in PICKER_LAYERS:
            data = self._volume_for(name)
            if data is None:
                continue
            if not self.toggles.is_on(name) and name not in already:
                continue
            row = self.rows[name]
            #: **`other cells` is a two-value LABEL volume, not a mask** — 1 soma, 2 process —
            #: so one picked hue renders in two treatments off a single slider. Faint soma,
            #: solid processes: what has to be adjudicated is whether a *process* belongs to
            #: this cell or the neighbour, so the processes stay legible while the neighbour's
            #: body drops back to a landmark it can still be located by.
            #: **The context layers are already the size they want to be**, cut wider than the
            #: cell by the caller, so they are pushed WHOLE with an offset rather than sliced by
            #: `box` — which is the bounding box and is exactly what used to throw the
            #: neighbourhood away.
            #: **Cut in the context frame, therefore drawn in it.** The two invariants the
            #: caller has to hold: widen all of `WIDENED_LAYERS` together, and set
            #: `context_origin` when you do. `ReviewStep._enter` and `._rewiden` do both, which
            #: is why they move as one picture.
            context = name in CONTEXT_LAYERS or (name in WIDENED_LAYERS
                                                 and self.context_origin is not None)
            volume = np.asarray(data) if context else np.asarray(data)[box]
            translate = self._world_offset(box) if context else None
            if name == "other cells":
                self.viewer.show_labels(
                    name, volume,
                    #: **`.hex`, not `.name`.** `show_image` takes a colour NAME and validates
                    #: it against the ten; `show_labels` takes a **hex string per label id**,
                    #: because napari's `DirectLabelColormap` wants RGBA. Passing the name here
                    #: crashed rich review on entry — `int("gr", 16)` — and no headless test
                    #: saw it, because `RecordingViewer.show_labels` parsed nothing. It does now.
                    #: **Two channels, not one — see `OTHER_SOMA_LIGHTEN`.** The soma takes the
                    #: picked hue blended toward white AND the reduced alpha; the processes take
                    #: the hue as picked at full alpha. Both come off the single slider, so the
                    #: reviewer still has one control for the layer.
                    colours={1: palette.lighten(row.colour().hex, OTHER_SOMA_LIGHTEN),
                             2: row.colour().hex},
                    alphas={1: row.alpha() * OTHER_SOMA_ALPHA_SCALE, 2: row.alpha()},
                    visible=self.toggles.is_on(name), translate=translate, scale=scale)
                continue
            self.viewer.show_image(name, volume, colour=row.colour().name,
                                   alpha=row.alpha(), visible=self.toggles.is_on(name),
                                   contrast_limits=self._limits_for(name),
                                   translate=translate, scale=scale)
            #: **The rendering control reaches the volume layers only.** `Labels` accepts
            #: `translucent` and `iso_categorical` and nothing else, so the radio cannot reach
            #: `this cell` even in principle — a napari fact, recorded rather than worked around.
            self.viewer.set_rendering(name, self.rendering)

        #: **One walk of the path, not two.** `displayed_labels` and `displayed_alphas` are
        #: the same `_displayed()` — which copies the whole editable frame — and asking for
        #: them separately paid for it twice on every frame.
        full_labels, alphas = self.editor.displayed()
        labels = full_labels[box]
        #: **The slider scales the region alphas, it does not replace them.** Setting them would
        #: erase the entered/not-entered cue the moment it moved, which is the one thing this
        #: layer's alpha carries. See `editing.region_alphas`.
        master = self.rows[CELL_LAYER].alpha()
        self.viewer.show_labels(
            CELL_LAYER, labels,
            colours=editing.region_colours(labels),
            alphas={r: a * master for r, a in alphas.items()},
            visible=self.toggles.is_on(CELL_LAYER), scale=scale)

    def _recolour(self, layer: str, colour: str) -> None:
        if self.viewer is not None and layer in self.viewer.layer_names():
            self.viewer.set_colour(layer, colour)

    def _reweight(self, layer: str, alpha: float) -> None:
        if self.viewer is not None and layer in self.viewer.layer_names():
            self.viewer.set_alpha(layer, alpha)

    def _toggle(self, name: str, on: bool) -> None:
        """**The first switch-on of a deferred layer is what builds it.**

        `DEFERRED_LAYERS` are neither cut nor uploaded on entry, so the layer does not exist in
        the viewer and its array may not exist either. Turning one on asks the driver for the
        array — see `context_wanted` — and then draws, which is the push that creates the
        layer. Slow once per cell, and free on every entry that never asks.
        """
        if self.viewer is None:
            return
        if name in self.viewer.layer_names():
            self.viewer.set_visible(name, on)
            return
        if not on:
            return
        if self._volume_for(name) is None:
            self.context_wanted.emit(name)
        self.draw()

    def _on_margin_released(self) -> None:
        """**The context has to be RE-CUT, not merely redrawn** — 2026-08-25.

        The four bbox-sized layers go through `crop_box`, which clips to the array it was handed
        and therefore cannot widen. The two context layers *can*, but they are already-cut
        arrays: redrawing pushes the same voxels at the same offset. So this emits first and
        draws second, and whoever owns the scene re-cuts in between.

        Without the emit the slider widens the context **once, at entry**, and then looks live
        while doing nothing — which is the same defect as item 6's cutoff wearing different
        clothes, and it was introduced in the same session that fixed that one.
        """
        self.margin_um = self.margin.value()
        self.margin_changed.emit(float(self.margin_um))
        self.draw()

    def set_context(self, other_cells=None, debris=None, context_origin=None, *,
                    iba1_context=_KEEP, dapi=_KEEP, other_nuclei=_KEEP) -> None:
        """Replace the context layers without disturbing the cell being edited.

        Deliberately **not** `set_cell`: that rebuilds `CellEditor` and would throw away every
        edit made so far. Widening the view must not cost the reviewer their work.

        **The three keyword layers default to `_KEEP`, not to `None`.** They joined the context
        set on 2026-08-26 and every existing caller passes only the first three positionally —
        defaulting them to `None` would have made each of those calls silently blank the Iba1,
        the DAPI and the neighbouring nuclei. `None` stays available and still means *this layer
        has no data*.
        """
        self.other_cells = None if other_cells is None else np.asarray(other_cells)
        self.debris = None if debris is None else np.asarray(debris)
        self.context_origin = (None if context_origin is None
                               else tuple(float(v) for v in context_origin))
        if iba1_context is not _KEEP:
            self.iba1_context = None if iba1_context is None else np.asarray(iba1_context)
        if dapi is not _KEEP:
            self.dapi = None if dapi is None else np.asarray(dapi)
        if other_nuclei is not _KEEP:
            self.other_nuclei = None if other_nuclei is None else np.asarray(other_nuclei)

    #: > ### `_world_offset` WAS DEFINED TWICE, AND THE SECOND ONE WON — 2026-08-26
    #: >
    #: > Two verbatim copies, ~170 lines apart. Python keeps the later one, so correcting
    #: > the first changed nothing and `_world_offset` kept returning the uncorrected
    #: > answer — the context layers sat one padding away from the cell they belong to.
    #: > **Found by a test that failed for the right reason and a debug print that
    #: > disagreed with the source**, which is the only way a shadowed method shows up.
    #: >
    #: > The duplicate is deleted rather than kept in step. `# -- the gestures ---` is
    #: > still doubled a few lines below, harmlessly, and is left as the marker that this
    #: > file was assembled by more than one hand.

    def set_rendering(self, mode: str) -> None:
        if mode not in RenderingChoice.OFFERED:
            raise ValueError(f"{mode!r} is not offered; expected {RenderingChoice.OFFERED}")
        self.rendering = mode
        if self.viewer is None:
            return
        #: **Volume layers only.** `other cells` became a labels layer on 2026-08-25 so one
        #: picked hue could render soma and processes differently, and `Labels` accepts
        #: `translucent` and `iso_categorical` and nothing else — the same napari fact that
        #: already keeps the radio away from `this cell`. Reaching it would raise in front of a
        #: user rather than in a test.
        for name, _, _, _ in PICKER_LAYERS:
            if name in LABEL_LAYERS:
                continue
            if name in self.viewer.layer_names():
                self.viewer.set_rendering(name, mode)

    # -- the gestures ---------------------------------------------------

    # -- the gestures ---------------------------------------------------

    def bind_gestures(self) -> None:
        """**The legend, wired.** Every handler below already existed; nothing reached them.

            CLICK a process        enter it
            CLICK the background   leave the process you are in
            DOUBLE-CLICK           remove - a part inside a process, the process outside one
            SHIFT + CLICK, RELEASE split this cell into that many
            u                      undo

        **This screen is `ndisplay=3`** - §3b: *"ndisplay=3 only, there is no slice mode"* - so
        a click is a ray rather than a point, and there are no arrow keys to bind: z is the
        camera here, and napari owns it.
        """
        if self.viewer is None:
            return
        self.viewer.bind_click(self._on_canvas_click)
        self.viewer.bind_double_click(self._on_canvas_double_click)
        self.viewer.bind_key("u", self.undo)
        self.viewer.bind_key("Shift", self._on_shift_held)

    def _shown_box(self):
        return crop_box(self.editor.mask, self.margin_um, self.voxel_size_um)

    def _to_voxel(self, position):
        """A world position -> this screen's **crop-voxel** index. One conversion, at capture.

        §4a.1's rule, and the reason it has its own method: *"store native full-image voxel
        coordinates, and convert once, at the point of capture."* `Click.position` is world
        coordinates, and world stopped equalling voxels the moment these layers gained a scale
        on 2026-08-25 — before that the two were the same number and every handler here read the
        position as an index directly, correctly but by coincidence.

        It cost `avg_nearest_neighbour_um` **doubling on every row ever written** the last time a
        frame was read as another one.
        """
        return tuple(float(c) / float(v)
                     for c, v in zip(tuple(position)[-3:], self.voxel_size_um))

    def _region_at(self, event) -> int:
        """Which region the pointer is over. **The ray is PrestoCell's own mechanism.**

        *"A click takes whatever is in front from the current angle, and you rotate to reach
        something occluded."* The walk tests only what is **drawn**, which is what makes hiding
        a confirmed cell safe rather than merely tidy - a hidden cell is not in this volume, so
        it cannot be clicked and needs no refusal popup.
        """
        if self.editor is None:
            return 0
        labels = self.editor.displayed_labels()[self._shown_box()]
        if event.ray is not None:
            #: The ray's endpoints are world coordinates too — it walks a **voxel** array, so
            #: both ends convert or neither does.
            return int(editing.first_hit_along_ray(
                labels, self._to_voxel(event.ray[0]), self._to_voxel(event.ray[1])))
        index = tuple(int(round(c)) for c in self._to_voxel(event.position))
        if any(c < 0 or c >= dim for c, dim in zip(index, labels.shape)):
            return 0
        return int(labels[index])

    def _crop_origin(self) -> tuple[int, ...]:
        """> ### THE CROP IS §4a.1's THIRD FRAME, AND IT IS THE ONE THAT ALREADY COST SOMETHING
        >
        > *"`export.py` added `cell.bbox` to a `soma_centroid` that already carried the crop
        > offset, which doubled `avg_nearest_neighbour_um` on every row ever written."*
        >
        > Every layer on this screen is drawn **cropped**, so a click arrives crop-relative. A
        > region id does not care - it is the same number in either frame - but a split marker
        > is a *position*, and `CellEditor.split` works on the full volume. This is the offset,
        > applied once, where the marker is captured.
        """
        return tuple(int(part.start) for part in self._shown_box())

    def _on_canvas_click(self, event) -> None:
        """**With SHIFT it places a split marker**; otherwise it enters or leaves a process."""
        if self.editor is None:
            return
        if event.shift():
            origin = self._crop_origin()
            local = tuple(int(round(c)) for c in self._to_voxel(event.position))
            self.markers.add(*(o + c for o, c in zip(origin, local)))
            self._draw_markers()
            return
        #: **The same walk as the double-click**, so the two gestures agree about what is under
        #: the pointer. A single click on a neighbour is not an action, but it must still count
        #: as *not this cell* — otherwise clicking a neighbour in front of your own process
        #: would enter the process behind it.
        kind, value = self._nearest_along_ray(event)
        self.click(value if kind == "cell" else 0)

    def _nearest_along_ray(self, event):
        """What the click passes through **first**, as `(kind, value)`.

        `kind` is `"cell"` with the region id, or an ownership status with the context-frame
        index of the voxel hit. `(None, None)` for a miss.

        > ### THE CONTEXT LOOKUP READ THE CLICK POINT, ON A SCREEN THAT IS ALWAYS 3D
        > — found by walking the screen, 2026-08-26
        >
        > `_region_at` walked the ray; `_context_index` read `event.position` and had no ray
        > branch at all. But `Click.ray` is documented as *"the pair napari gives for a click in
        > a 3D view"* and the position *"is already the voxel"* **only in a 2D slice view** —
        > and §3b makes this screen `ndisplay=3` with no slice mode. So the position is a point
        > on the camera ray, not the surface under the pointer, and the context lookup missed
        > every time. **Double-clicking a neighbour or debris did nothing at all, silently** —
        > no add, no transfer popup, nothing — while every gesture that went through
        > `_region_at` worked. That asymmetry is the whole reason only *add* looked broken.
        >
        > ### NEAREST WINS, AND THAT REPLACES "THIS CELL WINS TIES"
        >
        > The old rule resolved this cell's regions first and returned early. With a point that
        > meant *the same voxel*, which is a tie. With a ray it would mean **the cell wins even
        > when something is in front of it** — you would look straight at a neighbour's process,
        > double-click it, and remove a piece of your own cell hidden behind. The user's
        > decision of 2026-08-26 is that the nearest thing wins, which is the rule this screen
        > already states for its own regions: *"a click takes whatever is in front from the
        > current angle, and you rotate to reach something occluded."*
        >
        > **Only what is drawn is a candidate**, so a layer switched off cannot be hit — which
        > makes the toggles a way to aim, and is the same property that lets a `confirmed` cell
        > be hidden rather than refused.
        """
        if self.editor is None:
            return None, None
        labels = self.editor.displayed_labels()[self._shown_box()]
        if event.ray is None:
            region = self._region_at(event)
            if region:
                return "cell", region
            index = self._context_index(event)
            return self._context_status(index)

        start = self._to_voxel(event.ray[0])
        finish = self._to_voxel(event.ray[1])
        cell_on = self.toggles.is_on(CELL_LAYER)
        for local in editing.walk_ray(start, finish):
            if cell_on and not any(c < 0 or c >= dim
                                   for c, dim in zip(local, labels.shape)):
                region = int(labels[local])
                if region:
                    return "cell", region
            status, index = self._context_status(self._to_context(
                local, self._context_shape()))
            if status is not None:
                return status, index
        return None, None

    def _context_shape(self):
        """The shape of the context frame, from whichever context layer is present."""
        volume = self.debris if self.debris is not None else self.other_cells
        return None if volume is None else volume.shape

    def _context_status(self, index):
        """`(status, index)` for a context-frame voxel — **debris before neighbours at the SAME
        voxel**, which is the only tie left once the walk decides distance."""
        if index is None:
            return None, None
        if (self.debris is not None and self.toggles.is_on("debris")
                and bool(self.debris[index])):
            return "unowned", index
        #: **The neighbour's PROCESSES are takeable; its soma is not** — see `_neighbour_piece`.
        #: Value 2 is a process and 1 is the cell body, and offering the body would promise a
        #: transfer the gesture does not perform.
        if (self.other_cells is not None and self.toggles.is_on("other cells")
                and int(self.other_cells[index]) == 2):
            return "candidate", index
        return None, None

    def _on_canvas_double_click(self, event) -> None:
        """What is under the pointer, and **who owns it**.

        > ### THE OWNERSHIP TABLE WAS COMPLETE AND UNREACHABLE — closed 2026-08-25
        >
        > `editing.OWNERSHIP` has mapped four ownership states to four actions since the screen
        > was written — `confirmed -> hidden`, `real_cell -> transfer`, `debris -> add`,
        > `candidate -> transfer` — and `double_click` has taken `status` and `material` to act
        > on them. **This handler passed neither.** Every double-click therefore defaulted to
        > `status="this cell"` and resolved to `remove`, so `add` and `transfer` were dead code
        > and the *"This material belongs to another cell"* popup could never appear.
        >
        > It could not have been otherwise before now: nothing else was drawn. The neighbours
        > and the debris arrived on this screen in the same session as this fix.

        **This cell wins ties.** A voxel of the entered cell is never context, so its own
        regions are resolved first and the context layers are only consulted where it is absent
        — otherwise the margin would make a cell's own process takeable from itself.
        """
        kind, value = self._nearest_along_ray(event)
        if kind == "cell":
            self.double_click(value)
        elif kind is not None:
            self.double_click(0, status=kind, material=self._material_at(kind, value))

    def _context_at(self, event):
        """`(ownership status, material)` for a click that missed this cell, or `(None, None)`.

        **Debris is `unowned`, not `debris`.** `OWNERSHIP` maps both to `add`, and `unowned` is
        the honest word: the voxels belong to nothing, which is what makes taking them need no
        permission popup. `debris` as a *reject reason* is a verdict about a whole cell and is a
        different thing wearing the same word.

        A neighbour is `candidate` — takeable behind the transfer popup. A confirmed cell is
        never here at all, because `_repaint_scene` erased it: `hidden` means *not drawn*, and
        the ray walk can only hit what is drawn.
        """
        if self.editor is None:
            return None, None
        status, index = self._context_status(self._context_index(event))
        if status is None:
            return None, None
        return status, self._material_at(status, index)

    def _material_at(self, status: str, index):
        """The takeable piece under a context-frame voxel, in the **editable** frame."""
        if status == "unowned":
            return self._into_cell_frame(self._context_component(self.debris, index))
        if status == "candidate":
            return self._into_cell_frame(self._neighbour_piece(index))
        return None

    def _into_cell_frame(self, component):
        """A context-frame component in the **editable** frame, or `None` if it does not fit.

        > ### ⇒ THE CONTEXT IS WIDER THAN WHAT CAN BE EDITED, AND THAT IS A REAL LIMIT
        >
        > The context layers grow outward by the margin; `CellEditor.mask` is still the cell's
        > own bounding box, because widening *that* means handing the editor full-volume arrays
        > — the loading contract, which this work deliberately did not open.
        >
        > So a piece you can **see** may lie partly outside the piece you can **take**. This
        > returns `None` in that case and the take is declined, rather than silently absorbing
        > the part that happens to fall inside: a truncated take would give the cell a process
        > that stops at an invisible boundary, and the mask would be wrong in a way nothing on
        > screen could show.
        >
        > **⇒ FLAGGED FOR THE USER.** The alternatives are to widen the editable mask, or to
        > take the intersection and say so in the popup. Declining is the choice that cannot
        > produce a wrong mask while the question is open.
        """
        if component is None or self.editor is None:
            return None
        component = np.asarray(component).astype(bool)
        origin = self.context_origin or (0.0, 0.0, 0.0)
        shape = self.editor.mask.shape
        out = np.zeros(shape, bool)

        #: **The overlap of the two boxes, computed rather than sliced blind.** The context box
        #: sits at `origin` inside the editable frame; since 2026-08-26 that offset is normally
        #: positive and the context box fits entirely, so this is an embed. It stays written as
        #: an overlap because the pool-backed route can still present the old arrangement, where
        #: the context was the wider of the two and the offset was negative.
        src, dst = [], []
        for axis, (off, dim) in enumerate(zip(origin, shape)):
            off = int(round(off))
            lo = max(0, off)
            hi = min(dim, off + component.shape[axis])
            if hi <= lo:
                return None
            dst.append(slice(lo, hi))
            src.append(slice(lo - off, hi - off))
        out[tuple(dst)] = component[tuple(src)]

        #: **All of it or none of it**, which is the rule that has not changed: a truncated take
        #: gives the cell a process that stops at an invisible boundary, and the mask would be
        #: wrong in a way nothing on screen could show.
        return out if int(out.sum()) == int(component.sum()) else None

    def _context_index(self, event):
        """The clicked voxel in the **context** frame, or `None` when it is outside it."""
        volume = self.debris if self.debris is not None else self.other_cells
        if volume is None:
            return None
        local = tuple(int(round(c)) for c in self._to_voxel(event.position))
        return self._to_context(local, volume.shape)

    def _to_context(self, local, shape):
        """A drawn-crop voxel -> the same voxel in the **context** frame, or `None` if outside.

        > ### THE DRAWN CROP'S OWN START WAS DROPPED, AND IT WAS ZERO SO IT DID NOT SHOW
        >
        > This used to compute `local - origin` through two lines that cancelled each other's
        > `part.start`. That was right only because `crop_box` was handed an array **the cell
        > exactly fills**, so the box always started at 0. Since the editable frame became the
        > bounding box plus the margin maximum, the cell sits *inside* its array and the box
        > starts at the padding — so the term stopped being zero and the arithmetic had to
        > acquire it.

        `image = editable_start + box.start + local`, and the context frame is `image -
        context_start`, so what comes out is `box.start + local - context_origin`. Both signs
        of the offset work, which is what lets the pool-backed route keep the old frames.
        """
        #: **`shape` is `None` when the screen has no context layers at all**, which the
        #: pool-backed route and half this file's tests present. `zip`ping against it raised
        #: `TypeError` inside a double-click handler — where Qt turns a traceback into a warning
        #: — so the gesture would have died quietly on any screen without debris or neighbours.
        if self.editor is None or shape is None:
            return None
        origin = self.context_origin or (0.0, 0.0, 0.0)
        shown = self._shown_box()
        index = tuple(int(part.start) + int(c) - int(round(o))
                      for part, c, o in zip(shown, local, origin))
        if any(c < 0 or c >= dim for c, dim in zip(index, shape)):
            return None
        return index

    def _neighbour_piece(self, index):
        """The neighbour's **process** under the pointer, not the whole cell.

        > ### DOUBLE-CLICKING A NEIGHBOUR OFFERED YOU THE ENTIRE NEIGHBOUR — 2026-08-26
        >
        > `_context_at` took the connected component of `other_cells > 0`, and a neighbouring
        > cell is one connected blob, so pointing at any part of it offered **the whole cell,
        > soma included**. That is not the judgement this screen exists for: §3d.7's question is
        > *is this process mine or the neighbour's*, which is a question about one process.
        >
        > **`other_cells` is already 1-soma / 2-process**, computed by `_other_cells_around`
        > from the same `editing.soma_region` the export measures. Labelling the components of
        > `== 2` alone breaks each neighbour apart **at its soma**, so what comes back is
        > exactly the branch that was pointed at. No new computation: the split is already in
        > the array, it was simply being thrown away by the `> 0`.
        >
        > It also makes the take far likelier to fit: a process is a small fraction of a
        > neighbour, and a whole neighbouring cell would rarely sit inside any editable frame.

        **Pointing at the soma itself returns nothing**, deliberately. A neighbour's cell body
        is not a piece of this cell under any reading, and taking it would be a merge rather
        than a transfer — a different operation, and not one this gesture offers.
        """
        if self.other_cells is None:
            return None
        processes = np.asarray(self.other_cells) == 2
        if not bool(processes[index]):
            return None
        return self._context_component(processes, index)

    @staticmethod
    def _context_component(mask, index):
        """The connected piece the click landed on — **not the whole layer.**

        Taking a layer would hand a cell every fragment in the crop at once. A double-click is a
        statement about the thing under the pointer, so it takes that thing.
        """
        from scipy import ndimage

        labelled, _ = ndimage.label(np.asarray(mask).astype(bool),
                                    structure=np.ones((3, 3, 3), bool))
        hit = int(labelled[index])
        return labelled == hit if hit else None

    def _on_shift_held(self):
        """A **generator** binding: markers clear on press, the split runs on release."""
        self.markers.clear()
        self._draw_markers()
        yield
        markers = self.markers.array(
            None if self.editor is None else self.editor.mask.shape)
        self.markers.clear()
        self._draw_markers()
        if markers is not None:
            self.split(markers)

    def _draw_markers(self) -> None:
        """The numbered crosses, in the **crop** frame the canvas is drawn in."""
        if self.viewer is None:
            return
        origin = self._crop_origin() if self.editor is not None else (0, 0, 0)
        numbered = self.markers.numbered()
        points = [tuple(c - o for c, o in zip(point, origin)) for _n, point in numbered]
        #: **Scaled like every other layer.** The marker positions are crop-voxel coordinates,
        #: so without it the crosses would sit 5.5x too shallow against a scaled volume — the
        #: split markers would name a different place than the one that was clicked.
        self.viewer.show_points(MARKER_LAYER, points, colour=MARKER_COLOUR,
                                labels=[n for n, _p in numbered], visible=bool(points),
                                scale=world_scale(self.voxel_size_um))

    def click(self, region: int) -> None:
        """Single click. **A region descends into it; the background rises one level.**

        > ### THREE LEVELS SINCE 2026-08-27 — the depth rule moved into `CellEditor`
        >
        > This used to refuse a click on a part, because with two cuts there was nowhere further
        > to go and *"a part is not a thing to enter"*. With three there is, for one more step,
        > and the screen is the wrong place to know how many cuts exist — `CellEditor.enter`
        > holds `MAX_DEPTH` and returns unchanged at the bottom rather than raising, so pressing
        > on past the finest level is a no-op instead of an error.
        """
        if self.editor is None:
            return
        if not region:
            self.editor.leave()
        elif region == editing.SOMA_LABEL:
            return                      # the soma is not a persistence cluster; nothing to enter
        else:
            self.editor.enter(region)
        self.draw()

    def double_click(self, region: int, *, status: str = "this cell",
                     material: np.ndarray | None = None) -> str:
        """Double click. Returns what happened — `remove` · `add` · `transfer` · `hidden`.

        **The transfer popup is the only thing standing between a click and a neighbour's
        material**, and `debris` has none by design: §9.4e deleted its arrays, so its voxels
        cannot be identified as owned even in principle.
        """
        if self.editor is None:
            return "hidden"
        action = editing.double_click_action(status)
        if action == "remove":
            #: > ### ⇒ THIS ABORTED THE PROGRAM, AND IT TOOK THE EPISODE WITH IT — item B2
            #: >
            #: > `CellEditor.remove` already refuses the soma, correctly — *"a cell with no body
            #: > is not a repair"* — but this called it **unguarded, from inside a napari mouse
            #: > callback.** An unhandled Python exception in a Qt slot under PyQt6 calls
            #: > `qFatal()` and **aborts the process**, so a reviewer who double-clicked the cell
            #: > body lost the whole episode. `remove` raises `KeyError` as well, when the
            #: > clicked region is not in the displayed labels — the same crash by a second door.
            #: >
            #: > **Both halves, which is what the user chose:** the soma is named explicitly, so
            #: > the reviewer is told *why*; and the catch-all is behind it, so no refusal this
            #: > method has not thought of can ever abort the program either. Both route through
            #: > `report_unavailable`, which is how this screen already declines two other
            #: > things, and both return `"unavailable"`, as those do.
            if int(region) == editing.SOMA_LABEL:
                self.report_unavailable("soma")
                return "unavailable"
            try:
                self.editor.remove(region)
            except (ValueError, KeyError):
                #: **Its own status, not the caller's** — corrected 2026-09-03.
                #: This passed `status` through, which is the status of a **take**: a refused
                #: removal was then explained with the take note, describing a boundary the
                #: reviewer had not crossed and titled *Cannot take that* over a gesture that
                #: meant delete. What reaches here is a refusal this method does not know
                #: about — a region that is not in the displayed labels, most likely — so
                #: the note says only what is certainly true: nothing was changed.
                self.report_unavailable("not_removable")
                return "unavailable"
        elif action in ("add", "transfer"):
            #: > ### IT RETURNED `"add"` WHILE ADDING NOTHING — 2026-08-26
            #: >
            #: > The old branch was `elif action == "add" and material is not None`, so a take
            #: > that could not be framed fell through to the redraw and **still reported
            #: > `"add"`** to its caller and its tests. Nothing on screen said the take had been
            #: > refused, and nothing in the return value said it either. The user's rule:
            #: > *"we can't silently decline."*
            if material is None:
                self.report_unavailable(status)
                return "unavailable"
            if action == "transfer":
                #: **A transfer may not empty its donor.** Taking the whole of another cell is a
                #: merge, which this gesture does not offer — and a cell that drops to zero
                #: voxels vanishes from `payloads()`, which would slide every decision after it
                #: onto the wrong cell, since `apply_decisions` joins by position. Reachable
                #: only for a neighbour with no soma, which `NO_SOMA_NOTE` records as reading
                #: like a single process.
                if self.would_empty_the_donor(material):
                    self.report_unavailable("whole_cell")
                    return "unavailable"
                if not self.confirm_transfer(status):
                    return "declined"
            #: **The warning, and the reviewer may go through it.** Detached material is the
            #: ordinary case for a severed process, so this is a caution rather than a gate —
            #: it names the bridge length because that is what ends up in the morphometrics.
            gap = editing.bridge_length_um(self.editor.mask, material, self.voxel_size_um)
            if gap > 0.0 and not self.confirm_bridge(gap):
                return "declined"
            self.editor.add(material)
            self.material_taken.emit(status, material)
        self._show_soma_note()
        self.draw()
        return action

    def confirm_bridge(self, gap_um: float) -> bool:
        """*"not touching — taking it draws a bridge N um long"*. Replaced in tests."""
        return ConfirmPopup(editing.bridge_question(gap_um), "Take It Anyway", "Leave It",
                            title="Not touching", parent=self).ask()

    def would_empty_the_donor(self, material) -> bool:
        """Whether this take would leave the cell it comes from with nothing at all.

        **Answered by `ReviewStep`, which is the only place that knows.** This screen holds one
        cell's crops; the payloads belong to the step above it. Replaced in tests, and `False`
        when nobody is listening — the standalone route has no other cells to empty.
        """
        return False

    def report_unavailable(self, status: str) -> None:
        """Say that a gesture was refused, and why. Replaced in tests.

        **The title names the gesture, not always a take** - see `editing.REMOVAL_REFUSALS`.
        Item B2 added two refusals that answer a double-click meaning *delete*, and telling a
        reviewer they cannot *take* the cell body names something they did not try to do.
        """
        title = ("Cannot remove that" if status in editing.REMOVAL_REFUSALS
                 else "Cannot take that")
        HelpPopup(title, editing.unavailable_note(status), self).exec_()

    def confirm_transfer(self, status: str) -> bool:
        """*"this belongs to cell N — take it?"* Replaced in tests."""
        return ConfirmPopup(
            "This material belongs to another cell. Take it into this one?",
            "Take It", "Leave It", title="Take material", parent=self).ask()

    def split(self, markers: np.ndarray) -> list[np.ndarray]:
        """Shift-release. **The children queue up and are dealt with one at a time.**"""
        if self.editor is None:
            return []
        self.progress.setText(editing.split_progress(1.6))
        children = self.editor.split(markers)
        self.progress.setText("")
        self.pending = list(children[1:])
        self.children_produced.emit(children)
        first = children[0]
        self.set_cell(self.iba1, first, self.nucleus,
                      voxel_size_um=self.voxel_size_um,
                      dapi=self.dapi, other_nuclei=self.other_nuclei)
        return children

    def undo(self) -> editing.Operation | None:
        operation = None if self.editor is None else self.editor.undo()
        self._show_soma_note()
        self.draw()
        return operation

    def keyPressEvent(self, event):         # noqa: N802 - Qt's name
        """`u` undoes. **The keys and the buttons are deliberately different sets**, as on
        triage: the actions that dispose of work take a pointer, the one that recovers it is
        the fast one.
        """
        if event.key() == Qt.Key.Key_U:
            self.undo()
            return
        super().keyPressEvent(event)

    # -- resolution -----------------------------------------------------

    def _ask_reject_reason(self) -> str | None:
        """Popup one. **Three outcomes, and `None` is a real one** — see `ChoicePopup`.

        > ### ⇒ ESCAPE USED TO RECORD DEBRIS — fixed 2026-08-30
        >
        > This asked through `ConfirmPopup`, whose `ask()` returns a **bool**: `True` for
        > *Real Cell* and `False` for *Debris* — **and `False` for Escape, and for the window's
        > X.** So dismissing the question recorded the answer that removes the cell from every
        > count, and `reject`'s `if reason is None` could never fire.
        >
        > **The order of the buttons is unchanged** — Debris then Real Cell, reading left to
        > right after the way out — so a reviewer's hand goes where it went before.

        Neither answer is a default, on `ConfirmPopup`'s own reasoning: both are legitimate, and
        defaulting one would tell the reviewer which the screen expects.
        """
        return reject_reason_popup(self)

    def _ask_cell_count(self) -> int | None:
        """Popup two, on `real_cell` only. **The count is recorded as given.**

        A number, not a yes/no: §4.4's NEW-d is *"this is 3 cells I cannot afford to
        split"*, and this project already treats human marks as ground truth — calling the
        same person's cell count an estimate would be incoherent, so there is no separate
        estimated column and there is not going to be one.

        > ### ⇒ CANCEL USED TO COMMIT THE REJECTION AT 1 — fixed 2026-08-30
        >
        > It read `.ask() or 1`, so **Cancel committed** a rejection counting one cell. Cancel
        > aborted on the first popup and committed on the second: the same word meaning opposite
        > things one modal apart, which is the actual defect rather than the default itself.
        >
        > **`None` now means the reviewer wants the question back**, and the button says
        > **Back** rather than Cancel, because that is what it does. `1` is still a perfectly
        > good answer — it is the spin box's value, reached by pressing Record.
        """
        return cell_count_popup(self)

    def accept(self) -> editing.Resolution:
        resolution = editing.Resolution("accept")
        self._finish(resolution)
        return resolution

    def reject(self) -> editing.Resolution | None:
        """**Two popups, and only the second one is skipped.** `debris` is one press.

        > ### ⇒ THE SECOND POPUP GOES BACK TO THE FIRST — 2026-08-30
        >
        > *"Back one popup, not commit"* — the user's decision. The loop is the whole of it:
        >
        > * **Cancel on the reason popup** ends the rejection. Nothing is recorded and the cell
        >   is left as it was, which is what `if reason is None` always meant to do.
        > * **Debris** commits at once — one press, the asymmetry §3b records and accepts.
        > * **Real cell** asks the count. **Back** returns to the reason popup; **Record**
        >   commits.
        >
        > **It cannot trap anyone**, and that is why the reason popup had to gain a Cancel
        > first: going back to a question with no way out would leave a reviewer able to leave
        > only by answering it, and the cheapest answer is the one that discards the cell.
        >
        > ### ⇒ THE LOOP MOVED TO `widgets.ask_rejection` — patch item B5, 2026-09-03
        >
        > The delete button on a triage tile asks the same two questions in the same order,
        > because it is the same decision taken from a different place. **Two copies would
        > drift two ways** — and this one has already been corrected twice, once for Escape
        > recording debris and once for Cancel committing at 1.
        >
        > **The two hooks stay on this screen.** They are what the suite replaces to drive a
        > rejection with no modal on screen, and they are passed into the shared loop rather
        > than looked up inside it.
        """
        resolution = ask_rejection(self, ask_reason=self.ask_reject_reason,
                                   ask_count=self.ask_cell_count)
        if resolution is None:
            return None
        self._finish(resolution)
        return resolution

    def _finish(self, resolution: editing.Resolution) -> None:
        """**Finalisation ends the episode's undo**, deliberately — §3b, and it is why triage's
        own undo *"never reaches back into a finalised rich-review episode"*.

        A pending child from a split takes the screen next; only when the queue is empty does
        the reviewer return to triage.
        """
        if self.editor is not None:
            self.editor.record = editing.EditRecord()
        self.resolved.emit(resolution)
        if self.pending:
            nxt, self.pending = self.pending[0], self.pending[1:]
            self.set_cell(self.iba1, nxt, self.nucleus,
                          voxel_size_um=self.voxel_size_um,
                          dapi=self.dapi, other_nuclei=self.other_nuclei)

    def returns_to_triage(self) -> bool:
        """**Only once every child of a split has been dealt with.** §9.2."""
        return not self.pending
