"""Screen 9b — 6E10 Threshold. `COMPONENT_03_PLAN.md` §4 item 9, §6l.

**Frame C, and it is a COMMIT screen — unlike 9a.** The split is measured rather than assumed:
the grow mask alone is **617 ms** and is live, but labelling 212 M voxels and testing each
component for a seed is **9-11 s** and arrives on COMMIT. *"What splits is not the mask; it is
the CLASSIFICATION."* `Step.PLAQUE` is in `COMMIT_SCREENS`; `Step.CD68` is not.

> ### IT SHIPS DISPLAY-ONLY, AND THAT IS §5's DECISION RATHER THAN A SHORTCUT
>
> The seeded/unseeded recolour is computed **here, by the UI**, for the cyan/yellow outlines and
> the table. **Nothing about it reaches `plaques.csv` or `images.csv`** — that half is request 12
> (SEED-GROW), deferred at `HANDOFF_MERGE.md` §5. The screen demonstrates what it was drawn to do
> without pulling the largest item on either list forward.

> ### THE PARAMETER IS `plaque_threshold`, NOT THE DRAWN `plaque_grow_threshold`
>
> The user's decision, §5's first row: *"use the names in the code today, so the exports work as
> intended for the demo."* `plaque_grow_threshold` is request 12's name and rides the v5 -> v6
> bump. Item 2 shows `plaque_dilate_um` for the same reason.

**Two controls, and only one is gated.** Moving `plaque_threshold` reverts every outline to
red and blocks CONTINUE; typing `min_plaque_volume_um3` leaves them cyan and yellow, because
the floor only *removes* objects and **reclassifies nothing** — the components already exist with
their volumes, so filtering them is free. §6l.5.

> ### FIVE napari LAYERS UNDER THREE PICKER ROWS — item 2's and item 5's shape
>
> §6l.8 draws **three** swatch rows and **three** toggles, and the plaque row has to cover a mask
> that is drawn in three states. So `plaque` is a *concept* driving three layers, exactly as item
> 2's single `DAPI` toggle drives `DAPI kept` and `DAPI removed`, and item 5's three picker rows
> each drive both panels.
>
> **They are image layers under additive blending, not one `Labels` layer.** A `Labels` layer
> would put all three states in one object, which is closer to §6l.2's *"the same outlines,
> recoloured"* — but task 12 recorded the rule that closes it: a `Labels` layer *"is not in
> `PICKER_LAYERS` and cannot be — its colouring is a per-label **scheme**… so §4's layer-appearance
> rule excludes it from both the picker and the translucency slider."* §6l.8 gives `plaque` both a
> picker and a translucency slider, so it cannot be a `Labels` layer. Additive image layers also
> keep §6f.3 intact, with item 4 still its **only** exception, and let `outline_of_batch`'s
> project-then-outline rule do the drawing rather than napari's `contour`.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget
from scipy import ndimage

from ..markers import MIN_PLAQUE_VOLUME_UM3
from . import gestures, palette
from .frames import CanvasHost, FrameC
from .preview import middle_slice, volume_contrast_limits
from ..cache import WindowCache
from .screen_cd68 import batch_projection, slider_range, z_label
from .semiauto import Flow, Session, Step
from .viewer import (
    BATCHED, ROLLING, CanvasLayout, ViewerPort, cached_plane, outline_2d, step_window,
    window_bounds,
)
from .widgets import (
    AccumulatingTable, ColourRow, CommitButton, HelpButton, LayerToggles, NavigationRow,
    ParameterSlider, batch_step_box,
)

__all__ = [
    "PlaqueThresholdScreen", "LAYERS", "CONCEPTS", "TOGGLES", "TABLE_COLUMNS",
    "RAW_LAYER", "PLAQUE_LAYER", "SEEDED_LAYER", "UNSEEDED_LAYER", "SEED_LAYER",
    "SEEDED", "UNSEEDED", "RAW_6E10_ALPHA", "OUTLINE_ALPHA", "SEED_FILL_ALPHA",
    "COMMITTED_COLOURS", "PLAQUE_CONNECTIVITY", "SLIDER_STEPS", "FLOOR_CEILING_UM3",
    "COMMIT_PROGRESS",
    "grow_mask", "plaque_labels", "component_volumes", "seeded_ids", "class_lookup",
    "counts", "class_plane", "mask_projection", "table_row",
]

RAW_LAYER = "6E10 raw"

#: The live, unclassified outline. **One colour, because seeded and unseeded do not exist yet.**
PLAQUE_LAYER = "plaque"

#: The committed pair. Item 2's `DAPI kept` / `DAPI removed` naming — one mask under two states.
SEEDED_LAYER = "plaque seeded"
UNSEEDED_LAYER = "plaque unseeded"

SEED_LAYER = "seed mask"

#: The two ids the classification lookup ever assigns. **`UNSEEDED` is deliberately the LARGER**,
#: because the z-batch view is a `max` — see `class_plane`.
SEEDED = 1
UNSEEDED = 2

#: §6f.4's table sets 6E10 at **0.45 as a context channel** and names this screen as the one
#: anticipated exception: *"a screen where a context channel becomes the subject may raise its
#: own — item 9b is the candidate, and it is settled there."* §6l.1 takes it.
#:
#: **The screen's own "?" is the argument.** *"TOO HIGH -> the outline clings to the bright core
#: and loses the dim edge."* The dim periphery is precisely what this screen exists to judge, so
#: it cannot be held at a context channel's weight. Measured, magenta outline on dim/mid/
#: saturated amyloid: 0.45 gives 7.8x/4.1x/2.6x, **0.60 gives 5.9x/3.2x/2.0x**, 0.70 gives
#: 5.1x/2.8x/1.8x. 0.60 holds; the committed cyan and yellow run 3.7x and 3.8x on a saturated
#: core.
#:
#: **Every one of those ratios was measured against a MAGENTA outline at full strength, which is
#: no longer the default** — see `OUTLINE_ALPHA` and `_PLAQUE_CONCEPT`. They are kept because
#: they are what chose **0.60 for the raw channel**, which is the number this constant sets and
#: the only number they were ever asked to justify.
RAW_6E10_ALPHA = 0.60

#: **0.15 — the user's default, 2026-08-30**, set before walking the screen and together with
#: the switch from magenta to red.
#:
#: It replaces `palette.DERIVED_MASK_ALPHA`, whose reasoning was *"an outline is a thin line and
#: needs all of its brightness"*. That reasoning is kept here rather than deleted, because it
#: names what the default trades away, and both halves were measured rather than guessed:
#:
#: * **red's minimum relative luminance on black is 0.04** — joint-lowest in the palette with
#:   green — so a 2 px line at 0.15 over a 6E10 fill at 0.60 is faint, and faintest for a
#:   deuteranope;
#: * **the weight is shared by all three plaque states**, deliberately: `LAYERS` derives the
#:   committed pair's from this one and `_reweight` drives all three. So the cyan and yellow
#:   whose 3.7x and 3.8x were measured at full strength now arrive at 0.15 as well.
#:
#: **Both are recoverable without leaving the screen** — the plaque row's translucency slider is
#: live and reaches every state. This is an arrival default, not a ceiling.
OUTLINE_ALPHA = 0.15

#: **A fill, taking the derived-mask constant anyway, and the reason is §6l.6's own numbers.**
#: Its 103.4 against the magenta fill is the separation of spring green *at full strength*; a
#: weight below 1.0 would quietly spend part of the one margin that chose the colour. The layer
#: is off by default and switched on deliberately to answer one question, so it is never
#: competing as context — which is what a lower weight would be protecting against.
SEED_FILL_ALPHA = palette.DERIVED_MASK_ALPHA

#: The plaque concept's own row, named because the two committed states borrow from it below.
#:
#: > ### RED, NOT MAGENTA — the user's default, 2026-08-30, and it lands where §6l.4 left room
#: >
#: > §6l.4 ruled red out for the **committed** states: *"a red would have made unseeded look
#: > like a failure"*, because §6b sends seeded and unseeded forward into every measurement
#: > alike. **That reasoning is untouched** — `_colour_of` hard-codes cyan and yellow and the
#: > picker cannot reach them — and it never spoke about the *provisional* outline, which is the
#: > one state that genuinely is not finished. Red there pairs with the red `!` beside COMMIT,
#: > so the gate's full-canvas carrier and its one-character carrier now agree in hue.
#: >
#: > **What is given up is §6l.3's same-hue-separated-by-form arrangement**, magenta outline
#: > over magenta fill. Red over magenta is still separated by form — a 2 px line against a
#: > fill whose brightness varies — and for a dichromat the two hues are close, so form is doing
#: > the work it was already doing.
_PLAQUE_CONCEPT = (PLAQUE_LAYER, "red", OUTLINE_ALPHA, True)

#: The three the user thinks in — one swatch row and one checkbox each. §6l.8 draws exactly
#: these three, and `plaque` covers all three of its states.
#:
#: **`seed mask` spring green, §6l.6**: it is a FILL over the magenta raw, so it must beat
#: magenta by hue alone — form cannot help. Spring green scores **103.4 against the magenta
#: fill**, and it is the only candidate carrying no meaning from another screen: orange means
#: `CD68 > T` on 9a, the previous screen, and green is CD68's own channel colour. White fails
#: outright at 19.9 against magenta and 18.2 against cyan, both fill-against-fill. Its cost is
#: real and recorded — relative luminance **0.07** for a deuteranope — and is mitigated by the
#: layer being **off at arrival** rather than by design.
CONCEPTS: tuple[tuple[str, str, float, bool], ...] = (
    (RAW_LAYER, "magenta", RAW_6E10_ALPHA, True),
    _PLAQUE_CONCEPT,
    (SEED_LAYER, "spring green", SEED_FILL_ALPHA, False),
)

#: §6l.4. `#40d8f0` and `#ffd60a` are **96.6 apart for a dichromat** and both clear of the
#: magenta beneath them, 73.3 and 74.9. *Neither reads as an error, and that is correct* — §6b
#: sends both forward into every measurement, labelled, and a red would have made unseeded look
#: like a failure.
#:
#: **Colour is the only thing recorded here**, and that is the point: the two committed states
#: take the `plaque` row's weight and visibility, so there is no second copy of either number to
#: drift out of step with the control that actually drives them.
COMMITTED_COLOURS: dict[str, str] = {SEEDED_LAYER: "cyan", UNSEEDED_LAYER: "yellow"}

#: Every napari layer with the colour, weight and arrival state it is drawn at — §3's pattern,
#: so the palette decision is assertable without building the screen.
#:
#: **Derived rather than typed.** A hand-written fifth row would let this table say one thing
#: while `CONCEPTS` said another, and only one of them reaches a user — the failure task 17
#: recorded as *mutating code the user cannot reach*.
LAYERS: tuple[tuple[str, str, float, bool], ...] = (
    CONCEPTS[0],
    _PLAQUE_CONCEPT,
    (SEEDED_LAYER, COMMITTED_COLOURS[SEEDED_LAYER], _PLAQUE_CONCEPT[2], _PLAQUE_CONCEPT[3]),
    (UNSEEDED_LAYER, COMMITTED_COLOURS[UNSEEDED_LAYER], _PLAQUE_CONCEPT[2], _PLAQUE_CONCEPT[3]),
    CONCEPTS[2],
)

TOGGLES: tuple[tuple[str, bool], ...] = tuple((name, on) for name, _, _, on in CONCEPTS)

#: `ndimage.label`'s structure in `markers._plaque_sizes` and `markers.build_plaque_fields`.
#: **Full 26-connectivity, and it must match**, or this screen would count a different number of
#: plaques from the one `plaques.csv` reports at the same threshold.
PLAQUE_CONNECTIVITY = np.ones((3, 3, 3), bool)

#: §6l.7's table, moved out of the right column to below the canvas — the user's decision,
#: 2026-08-19, putting it where item 5's accumulating table already sits.
#: **`grow` became `threshold` at the user's request, 2026-08-30.** The column holds
#: `plaque_threshold`, and `grow` named the mask rather than the parameter — it sits beside
#: `floor`, which is `min_plaque_volume_um3` under the same shorthand.
TABLE_COLUMNS: tuple[str, ...] = ("threshold", "floor", "seeded", "unseeded")

#: 1000 log positions, as 9a. The two study-level pickers use the same control.
SLIDER_STEPS = 1000

#: **Shown, not enforced.** `plaque_floor_calibration.txt` adopts no value but establishes that
#: above 73 µm³ the weakest 5xFAD image loses **every** plaque, against the standing rule never
#: to zero out a real image. *"A choice, not a finding. Stay under 73."*
FLOOR_CEILING_UM3 = 73.0

#: Under Commit for the 9-11 s, **with no cancel** — Q2's component-wide rule, not a per-screen
#: choice.
COMMIT_PROGRESS = "…working"


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

def grow_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """**`6E10 > plaque_threshold`.** Takes a volume or an already-projected plane.

    > ### ⇒ THE PLAN DRAWS `>=` AND THE PIPELINE COMPUTES `>`. THE PIPELINE WINS AGAIN.
    >
    > §6l and §4 both write the layer as `6E10 >= plaque_grow_threshold`.
    > `markers._plaque_sizes` is `mask = np.asarray(plaque_channel) > float(threshold)`, and
    > every plaque column in `plaques.csv` and `images.csv` comes from that line through
    > `measure_plaque_burden`.
    >
    > **This is the same correction 9a made one screen earlier**, and it is the same argument:
    > the threshold is an absolute intensity the user types, so a voxel sitting exactly at it
    > falls on one side under the drawing and the other under the measurement. **A screen that
    > outlined objects the export will not count is the complete-looking-but-wrong failure this
    > project keeps rejecting.** Reversible in one character if the drawing is the intent, but it
    > would move a shipped measurement to suit a label.
    """
    return np.asarray(values) > float(threshold)


def plaque_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components of the grow mask, at the pipeline's own connectivity.

    **Unfloored, and that is what makes `min_plaque_volume_um3` live.** The floor is applied by
    `class_lookup` against volumes computed once, so typing a smaller number brings objects back
    without re-labelling — which is the only reason §4 moved the control to this screen.

    > **The 5.6 s optimisation §4 records cannot be taken here.** *"Applying the volume floor
    > first and testing only the ~36 survivors is the obvious fix."* It is — for a batch run. On
    > this screen the floor is live **after** the commit, so a component below the current floor
    > must already carry its seeded/unseeded answer for the moment the user lowers it. Testing
    > only the survivors would make lowering the floor either wrong or expensive.

    **The result is not cast.** `ndimage.label` already returns `int32`; an `astype(np.int32)`
    that looks like a safety net is a full-volume copy, which is **848 MB** at production width.
    """
    labels, n = ndimage.label(np.asarray(mask), structure=PLAQUE_CONNECTIVITY)
    return labels, int(n)


def component_volumes(labels: np.ndarray, voxel_size_um) -> np.ndarray:
    """µm³ per component id, indexed by id. **Index 0 is background** and is never a plaque.

    The same arithmetic as `markers._plaque_sizes` — `bincount` times the voxel volume — so the
    two cannot drift into disagreeing about which objects clear the floor.
    """
    voxel = float(np.prod([float(v) for v in voxel_size_um]))
    return np.bincount(np.asarray(labels).ravel()).astype(np.float64) * voxel


def seeded_ids(labels: np.ndarray, seed_mask: np.ndarray | None) -> frozenset[int]:
    """Which components contain at least one voxel of item 2's veto mask.

    **Seeded status is a property of the whole 3D component**, which is why it cannot be live:
    a component's only seed voxel may sit on slice 25 while the user is looking at slice 20, so
    a per-slice colouring would call it unseeded and be **wrong**. Same constraint as item 4's
    rule that label ids do not project — *a 2D view cannot carry a 3D fact.*
    """
    if seed_mask is None:
        return frozenset()
    seed = np.asarray(seed_mask).astype(bool)
    return frozenset(int(i) for i in np.unique(np.asarray(labels)[seed]) if i > 0)


def class_lookup(volumes_um3: np.ndarray, seeded: frozenset[int],
                 floor_um3: float) -> np.ndarray:
    """`id -> SEEDED | UNSEEDED | 0`. **The one place the floor is applied.**

    Both things that change at commit live here: objects under the floor map to `0` and vanish,
    and the survivors split into two classes. §4: *"what vanishes between the two states is
    itself the diagnostic — here you watch the floor do its work."*

    **`>=`, as `markers` filters.** `_plaque_sizes` ends `return sizes[sizes >= float(
    min_plaque_volume_um3)]` and `build_plaque_fields` writes
    `keep[1:] = sizes[1:] >= float(min_plaque_volume_um3)`. An object of exactly 25.0 µm³ is kept
    by the pipeline, so it is kept here.
    """
    volumes = np.asarray(volumes_um3, dtype=np.float64)
    kept = volumes >= float(floor_um3)
    if kept.size:
        kept[0] = False                     # background clears any floor and is never a plaque
    lookup = np.where(kept, UNSEEDED, 0).astype(np.uint8)
    if seeded:
        index = np.fromiter(sorted(seeded), dtype=np.int64)
        index = index[index < lookup.size]
        lookup[index] = np.where(kept[index], SEEDED, 0)
    return lookup


def counts(lookup: np.ndarray) -> tuple[int, int]:
    """`(n_seeded, n_unseeded)` above the floor, **read straight off the lookup.**

    Derived from the same array the picture is drawn from, so the table and the canvas cannot
    disagree about how many objects there are.
    """
    return int((lookup == SEEDED).sum()), int((lookup == UNSEEDED).sum())


def mask_projection(mask: np.ndarray, z: int, depth: int,
                    mode: str = ROLLING) -> np.ndarray:
    """`any` over the z batch — the mask analogue of `batch_projection`'s `max`.

    **The first half of `outline_of_batch`**, split out because this screen projects a slab and
    outlines the plane rather than materialising a whole-volume mask: at production width a bool
    copy of the channel is 212 MB, and the view only ever needs `depth` slices.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {mask.shape}")
    start, stop = window_bounds(z, mask.shape[0], depth, mode)
    return mask[start:stop].any(axis=0)


def class_plane(labels: np.ndarray, lookup: np.ndarray, z: int, depth: int,
                mode: str = ROLLING) -> np.ndarray:
    """The z-batch view of the classification. **`UNSEEDED` wins where two components overlap.**

    Two components cannot share a voxel, so at depth 1 — where §4 says to make the fusion call —
    there is no tie to break. Above it two outlines separated in z can land on the same pixel,
    and the `max` is what decides, which is why `UNSEEDED` is `2` and `SEEDED` is `1` rather than
    the other way round. **Nothing yellow may be hidden by cyan**: the unseeded count is the
    trend the screen exists to show, and a reviewer scanning for yellow that a cyan outline
    covered would under-read it.

    Only the slab is mapped, never the whole volume — see `mask_projection`.

    > ### IT TOOK NO WINDOW MODE, AND EVERY OTHER LAYER ON THE SCREEN DID — fixed 2026-08-30
    >
    > It sliced `z:z+depth` unconditionally while `batch_projection` and `mask_projection` both
    > went through `window_bounds`. The two agree exactly in `ROLLING`, whose start *is* `z`;
    > **in `BATCHED` the window snaps to `(z // depth) * depth`**, so the committed cyan and
    > yellow outlines would have projected a different slab from the 6E10 under them and the
    > seed mask beside them — at any `z` the batch does not divide.
    >
    > **It had never been reachable**, and that is the whole reason it survived: `batch_mode`
    > only ever changes from the `Step by batch` box, and on this screen that box was
    > permanently disabled. `set_depth` enabling it is what would have exposed this, so the two
    > are fixed in the same change rather than one shipping ahead of the other.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {labels.shape}")
    start, stop = window_bounds(z, labels.shape[0], depth, mode)
    return lookup[labels[start:stop]].max(axis=0)


def table_row(threshold: float, floor_um3: float,
              n_seeded: int, n_unseeded: int) -> tuple[str, str, str, str]:
    """One commit, formatted for §6l.7's table.

    **The floor is in the table because it moves.** Changing it updates the live row only, so
    earlier rows keep the floor they were taken at and two rows at different floors can never be
    read as comparable.
    """
    return (f"{threshold:g}", f"{floor_um3:.1f}", str(int(n_seeded)), str(int(n_unseeded)))


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

class PlaqueThresholdScreen(QWidget):
    """Frame C: three pickers and three toggles left, one overlaid view, controls right."""

    continue_requested = Signal()
    #: **Every screen but items 1, 9a and 7b, since 2026-08-30.**
    back_requested = Signal()

    #: One view with the table below it. Five layers, one panel.
    CANVAS = CanvasLayout()

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.PLAQUE)
        self.viewer = viewer
        self.plaque: np.ndarray | None = None
        self.seed: np.ndarray | None = None
        self.voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.contrast: tuple[float, float] | None = None
        self.z = 0
        self.depth = 1
        #: Rolling until the user ticks `Step by batch`. The two modes agree at depth 1, so
        #: the toggle can never move the picture on its own.
        self.batch_mode = ROLLING
        #: **The window cache — item A9.** The raw channel, the classification plane and the
        #: seed mask; the three outlines are 2D erosions over planes already held.
        self.windows = WindowCache()
        self.threshold = 0.0
        self.floor_um3 = float(MIN_PLAQUE_VOLUME_UM3)

        #: The commit's product. **All four are needed for the floor to stay live** — the labels
        #: to draw from, the volumes to filter, the seeded set to colour by, and the lookup that
        #: is the current answer.
        self.labels: np.ndarray | None = None
        self.volumes_um3: np.ndarray | None = None
        self.seeded: frozenset[int] = frozenset()
        self.lookup: np.ndarray | None = None

        self.flow.session.chosen["min_plaque_volume_um3"] = self.floor_um3

        self.frame = FrameC("6E10 threshold", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: three pickers, three toggles ---------------------------
        self.frame.section("Colours", HelpButton(
            "Colours",
            "Sets how each layer is drawn. Colour and translucency only.\n\n"
            "It changes nothing that is computed or saved.", self))

        self.rows: dict[str, ColourRow] = {}
        for name, colour, alpha, _ in CONCEPTS:
            row = ColourRow(name, colour, alpha, self)
            row.colour_changed.connect(lambda c, n=name: self._recolour(n, c))
            row.alpha_changed.connect(lambda a, n=name: self._reweight(n, a))
            self.frame.add_left(row)
            self.rows[name] = row

        self.frame.section("Layers", HelpButton(
            "Layers",
            "6E10 raw — the channel as imaged.\n\n"
            "plaque — the outlines of everything above your threshold. One colour until you "
            "commit; after that, coloured seeded and unseeded.\n\n"
            "seed mask — the plaque veto AS APPLIED on the earlier screen, your dilation "
            "included. It is what decides seeded from unseeded. Off by default.", self))
        self.toggles = LayerToggles(TOGGLES, self)
        self.toggles.toggled.connect(self._toggle)
        self.frame.add_left(self.toggles)
        self.frame.left.addStretch(1)

        # -- centre: the canvas, then the table BELOW it ------------------
        #: **No header strip**, as 9a: the table carries every number and there is one panel with
        #: nothing to name.
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        self.table = AccumulatingTable(TABLE_COLUMNS, self)
        table_strip = QHBoxLayout()
        table_strip.setContentsMargins(0, 4, 0, 0)
        table_strip.addWidget(self.table, 1)
        table_strip.addWidget(HelpButton(
            "the table",
            "One row per COMMIT: the threshold and floor used, and how many plaques came out "
            "seeded and unseeded.\n\n"
            "seeded — the object contains a voxel of the plaque veto as it was applied on the "
            "earlier screen, dilation included.\n\n"
            "unseeded — it does not.\n\n"
            "Both go forward into every measurement, labelled.", self), 0)
        table_host = QWidget(self)
        table_host.setLayout(table_strip)
        self.frame.add_centre(table_host)

        # -- right: the log slider, the box, commit, the floor, z ---------
        self.slider = ParameterSlider(
            "plaque_threshold", 1.0, 2.0, 1.0, decimals=0, steps=SLIDER_STEPS, log=True,
            help_button=HelpButton(
                "plaque_threshold",
                "The intensity above which 6E10 signal counts as part of a plaque. It sets how "
                "far each plaque extends — the outline you see.\n\n"
                "An ABSOLUTE value, not a multiplier. Picked once and applied unchanged to every "
                "image in the study.\n\n"
                "  TOO HIGH -> the outline clings to the bright core and loses the dim edge\n\n"
                "  TOO LOW  -> neighbouring deposits join into one\n\n"
                "LIVE: outlines follow the slider. The plaques themselves — floored and split "
                "into seeded and unseeded — arrive on COMMIT (about 9-11 s).", self),
            parent=self)
        self._wire_slider(self.slider)
        self.frame.add_right(self.slider)

        self.log_note = QLabel("LOG scale", self)
        self.log_note.setProperty("role", "helper")
        self.frame.add_right(self.log_note)

        #: > ### ⇒ THE ABSOLUTE VALUE BOX WAS REMOVED — 2026-08-30, at the user's request
        #: >
        #: > It read *"the BOX is authoritative, not the slider"*: 1000 log positions over two
        #: > orders of magnitude quantise finely enough to look right and coarsely enough to be
        #: > wrong in a file that is hashed and applied to every image in the study. 9a's box
        #: > came off the same day, for the same reason.
        #: >
        #: > **The number does not leave the screen with it.** `ParameterSlider` already draws a
        #: > live value above the slider; what is given up is typing an exact one past the
        #: > quantisation. **And the quantisation stopped being hidden in the same change** —
        #: > `set_threshold` stores what that label displays, so the glass and the file carry
        #: > the same number and the trade is guarded rather than merely accepted.

        self.commit = CommitButton("Commit", self)
        self.commit.clicked.connect(self._on_commit)
        self.frame.add_right(self.commit)

        self.floor_box = self._number(
            "min_plaque_volume_um3", f"{self.floor_um3:g}",
            "The smallest object that still counts as a plaque. Anything below it is dropped, "
            "whether seeded or not.\n\n"
            f"Stay under {FLOOR_CEILING_UM3:g}: above that, the weakest 5xFAD image in this "
            "study loses every plaque.\n\n"
            "LIVE after the first commit — objects appear and disappear as you type and the "
            "counts update. Nothing is recomputed.\n\n"
            "Before the first commit there are no objects to filter.")
        self.floor_box.editingFinished.connect(self._on_floor_edited)

        #: §6i.7's disabled state, **carried twice**: the greyed control for everyone, the
        #: sentence for anyone who cannot separate the two greys.
        self.floor_note = QLabel("live after the first commit", self)
        self.floor_note.setProperty("role", "helper")
        self.frame.add_right(self.floor_note)

        self.depth_box = self._number(
            "z-batch depth", "1",
            "How many z slices are combined into one view.\n\n"
            "  1  -> one slice at a time\n"
            "  10 -> the brightest value across each group of 10 slices\n\n"
            "Make the fusion call at depth 1: combining slices can make two separate deposits "
            "look joined.")
        self.depth_box.editingFinished.connect(self._on_depth_edited)
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

        #: **No BACK**, consistent with items 5, 6 and 9a. CONTINUE leads to item 8.
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
        """A labelled number box with its "?", item 2's arrangement."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addStretch(1)
        row.addWidget(HelpButton(label, help_text, self))
        host = QWidget(self)
        host.setLayout(row)
        self.frame.add_right(host)

        edit = QLineEdit(value, self)
        edit.setFixedWidth(96)
        self.frame.add_right(edit)
        return edit

    # -- loading --------------------------------------------------------

    def set_channels(self, plaque: np.ndarray, seed: np.ndarray | None,
                     voxel_size_um=(1.0, 1.0, 1.0), threshold: float | None = None) -> None:
        """Take the 6E10 channel and item 2's veto mask **as applied**, at the middle slice.

        > ### IT USED TO BE HANDED THE UNDILATED MASK — the user's correction, 2026-08-30
        >
        > *"The seed mask shown here should be the dilated one from the item 2 screen."* `app.py`
        > passed `products[Step.VETO]`, which is undilated **by contract** — `detect_nuclei`
        > applies `plaque_dilate_um` itself and a pre-widened mask would dilate twice. So the
        > spring-green layer drew something narrower than item 2 showed and narrower than what
        > actually suppressed DAPI, and `seeded_ids` classified against it.
        >
        > **The same shape as 2026-08-20's `DAPI -veto` defect**: the screen where a value is
        > judged, drawing a mask that value never reached. The margin is applied by
        > `SemiAutoApp._veto_as_applied`, the one helper item 4 also goes through, so two screens
        > cannot disagree about what *as applied* means.

        **The seed mask is optional and its absence is not an error.** §4: the 22.4 s precompute
        runs while 9a is on screen and *"press CONTINUE before it finishes and 9b shows its own
        progress"*. Without it every surviving object is unseeded, which is what a missing seed
        mask honestly means rather than a refusal to draw.
        """
        self.plaque = np.asarray(plaque)
        self.seed = None if seed is None else np.asarray(seed).astype(bool)
        self.voxel_size_um = tuple(float(v) for v in voxel_size_um)
        self.contrast = volume_contrast_limits(self.plaque)
        self.z = middle_slice(self.plaque.shape[0])
        self.depth = 1
        self.depth_box.setText("1")

        low, high = slider_range(self.plaque)
        opening = float(threshold) if threshold is not None else low
        self.slider = self._rebuild_slider(low, high, opening)
        self.set_threshold(opening)

    def _rebuild_slider(self, low: float, high: float, value: float) -> ParameterSlider:
        """The ends are two percentiles of the volume, so the slider is rebuilt on arrival.

        9a's arrangement exactly, and for 9a's reason: **a slider whose range is data cannot be
        constructed before the data.** The realised Otsu level on 6E10 spans 115 to 18,334 across
        slices and the spread among images holding plaques is 37x, so a linear slider fails the
        same way it would at 9a.
        """
        old = self.slider
        fresh = ParameterSlider("plaque_threshold", low, max(high, low * 1.000001), value,
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

        Task 17's slipped mutations: the constructor builds a placeholder and `set_channels`
        replaces it, so a separately wired constructor slider is dead code that a mutation cannot
        reach and a test cannot hold.

        **RELEASE recomputes; the FIRST STEP of a drag invalidates.** Unlike 9a this screen has a
        gate, and it must close the moment the value stops matching the commit — not when the
        hand lets go. Item 3 and item 5 do the same.
        """
        slider.slider.sliderReleased.connect(self._on_slider_released)
        slider.slider.valueChanged.connect(self._on_slider_moved)

    # -- the controls ---------------------------------------------------

    def is_committed(self) -> bool:
        """Whether a commit's components exist. **Item 5's `is_committed` flag**, which that
        screen also clears when a parameter moves."""
        return self.labels is not None

    def set_threshold(self, value: float) -> None:
        """The gated number. **Moving it reverts every outline to the provisional colour**, which
        is the gate's second carrier and falls out of the mechanism rather than being designed
        in: the live preview recomputes outlines the commit has not classified, and red is
        simply what an unclassified outline is.

        **Stored as the slider displays it** — `ParameterSlider.as_displayed`. With the box gone
        the slider is the only writer, and a label reading `2681` beside a parameters file
        holding `2681.4372` is a screen saying one thing while the hashed file records another.
        """
        self.threshold = self.slider.as_displayed(value)
        self.flow.session.chosen["plaque_threshold"] = self.threshold
        self.labels = None
        self.volumes_um3 = None
        self.seeded = frozenset()
        self.lookup = None
        self.flow.invalidate()
        self._sync()
        self.draw()

    def _on_slider_moved(self, _position: int) -> None:
        """The first step of a drag closes the gate. §4 item 2's rule for every commit screen."""
        self.flow.invalidate()
        self._sync()

    def _on_slider_released(self) -> None:
        """617 ms for the whole grow mask, so it recomputes on release rather than per step."""
        self.set_threshold(self.slider.value())

    def _on_floor_edited(self) -> None:
        try:
            value = float(self.floor_box.text())
        except ValueError:
            self.floor_box.setText(f"{self.floor_um3:g}")
            return
        self.set_floor(value)

    def set_floor(self, value: float) -> None:
        """`min_plaque_volume_um3`. **Live and UNGATED**, and the two are one decision.

        §6l.5: the floor only *removes* objects below it and **reclassifies nothing**, so there
        is nothing stale for a gate to guard — the outlines stay cyan and yellow. Gating it would
        move the control to this screen and then delete the only reason to move it: making the
        user wait 9 s to watch a free operation.
        """
        self.floor_um3 = max(0.0, float(value))
        self.floor_box.setText(f"{self.floor_um3:g}")
        self.flow.session.chosen["min_plaque_volume_um3"] = self.floor_um3
        if self.is_committed():
            self.lookup = class_lookup(self.volumes_um3, self.seeded, self.floor_um3)
            n_seeded, n_unseeded = counts(self.lookup)
            self.table.update_current_row(
                table_row(self.threshold, self.floor_um3, n_seeded, n_unseeded))
            self.draw()

    def _on_depth_edited(self) -> None:
        try:
            depth = int(float(self.depth_box.text()))
        except ValueError:
            self.depth_box.setText(str(self.depth))
            return
        self.set_depth(depth)

    def set_depth(self, depth: int) -> None:
        """**And it gates `Step by batch`**, which every other screen carrying that box did.

        > ### THE TOGGLE WAS DEAD ON THIS SCREEN, AND ONLY ON THIS SCREEN — 2026-08-30
        >
        > `batch_step_box` ships `setEnabled(False)` because the two window modes agree at depth
        > 1. Items 2, 4, 5 and 9a re-enable it as their depth rises; **9b never called
        > `setEnabled` at all**, so the box could not be clicked at any depth and
        > `self.batch_mode` could never leave `ROLLING`. The greyed state was hard to see
        > because the theme mutes the label text but leaves `QCheckBox::indicator` to the
        > platform style, so the tick box itself did not read as dead.
        >
        > **Which is why `class_plane`'s missing window mode had never been reachable** — see
        > there. Enabling this box is what exposes it, so both went in together.
        """
        n = 1 if self.plaque is None else self.plaque.shape[0]
        self.depth = max(1, min(int(depth), n))
        self.depth_box.setText(str(self.depth))
        self.batch_step.setEnabled(self.depth > 1)
        self.draw()

    def set_z(self, z: int) -> None:
        """**Absolute**, because napari's z slider reports a slice rather than a step."""
        if self.plaque is None:
            return
        self.z = max(0, min(int(z), self.plaque.shape[0] - 1))
        self.draw()

    def step_z(self, delta: int) -> None:
        if self.plaque is None:
            return
        self.set_z(step_window(self.z, delta, int(self.plaque.shape[0]),
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

    # -- the commit -----------------------------------------------------

    def _on_commit(self) -> None:
        """Label, measure, classify — **the 9-11 s, with progress and no cancel.**

        The screen runs this itself rather than taking a product, unlike item 5: nothing else in
        the flow holds the grow mask, and every function it calls is pure and in this module.
        """
        if self.plaque is None:
            return
        self.commit.set_working(COMMIT_PROGRESS)
        try:
            labels, _ = plaque_labels(grow_mask(self.plaque, self.threshold))
            self.labels = labels
            self.volumes_um3 = component_volumes(labels, self.voxel_size_um)
            self.seeded = seeded_ids(labels, self.seed)
            self.lookup = class_lookup(self.volumes_um3, self.seeded, self.floor_um3)
        finally:
            self.commit.set_working("")

        n_seeded, n_unseeded = counts(self.lookup)
        self.table.add_row(table_row(self.threshold, self.floor_um3, n_seeded, n_unseeded))
        self.flow.commit(labels)
        self._sync()
        self.draw()

    # -- drawing --------------------------------------------------------

    def draw(self) -> None:
        """Five layers, of which the three plaque states are outlines and two are always empty.

        **The live outline is taken from the projected plane, not from a whole-volume mask**, and
        that is an identity rather than a shortcut: `max` over z then `> T` is the same set as
        `> T` then `any` over z, so this is still §6f.6's *project, then outline*. It costs one
        2D erosion instead of a 212 MB bool copy of the channel.
        """
        if self.viewer is None or self.plaque is None:
            return

        raw = cached_plane(
            self.windows, "plaque", self.plaque, self.z, self.depth, self.batch_mode,
            lambda: batch_projection(self.plaque, self.z, self.depth, self.batch_mode),
            projection=True)
        self._image(RAW_LAYER, raw, contrast_limits=self.contrast)

        if self.is_committed():
            plane = cached_plane(
                self.windows, "classes", (self.labels, self.lookup), self.z, self.depth,
                self.batch_mode,
                lambda: class_plane(self.labels, self.lookup, self.z, self.depth,
                                    self.batch_mode))
            blank = np.zeros(plane.shape, bool)
            self._image(PLAQUE_LAYER, blank)
            self._image(SEEDED_LAYER, outline_2d(plane == SEEDED))
            self._image(UNSEEDED_LAYER, outline_2d(plane == UNSEEDED))
        else:
            blank = np.zeros(raw.shape, bool)
            self._image(PLAQUE_LAYER, outline_2d(grow_mask(raw, self.threshold)))
            self._image(SEEDED_LAYER, blank)
            self._image(UNSEEDED_LAYER, blank)

        #: > ### ⇒ THE SEED LAYER IGNORED `batch_mode` — found while wiring A9
        #: >
        #: > `mask_projection` was called **without a mode**, so it took the default `ROLLING`
        #: > while the raw channel, the classification plane and the indicator all used
        #: > `self.batch_mode`. With `Step by batch` ticked above depth 1 the seeds were drawn
        #: > from a different window of the stack than the plaques they seed — task 26's defect
        #: > in a second door, and display only: nothing here reaches a number.
        seed_plane = (np.zeros(raw.shape, bool) if self.seed is None else cached_plane(
            self.windows, "seeds", self.seed, self.z, self.depth, self.batch_mode,
            lambda: mask_projection(self.seed, self.z, self.depth, self.batch_mode)))
        self._image(SEED_LAYER, seed_plane)

        self.z_indicator.setText(
            z_label(self.z, self.plaque.shape[0], self.depth, self.batch_mode))

    def _image(self, layer: str, data: np.ndarray, contrast_limits=None) -> None:
        """Push one layer with its concept's colour, weight and visibility.

        The three plaque layers read the one `plaque` row, which is what makes the swatch row and
        the checkbox cover a mask drawn in three states — **except for the colour of the two
        committed states**, which `LAYERS` fixes and the picker does not reach.
        """
        colour = self._colour_of(layer)
        row = self.rows[self._concept_of(layer)]
        self.viewer.show_image(layer, np.asarray(data), colour=colour, alpha=row.alpha(),
                               visible=self.toggles.is_on(self._concept_of(layer)),
                               contrast_limits=contrast_limits)

    @staticmethod
    def _concept_of(layer: str) -> str:
        """Which picker row and checkbox a napari layer answers to."""
        return PLAQUE_LAYER if layer in (SEEDED_LAYER, UNSEEDED_LAYER) else layer

    def _colour_of(self, layer: str) -> str:
        """**Cyan and yellow are a scheme, not a choice**, so they do not come from the picker.

        §6l.4: they are a measured pair 96.6 apart for a dichromat, carrying the commit gate's
        second signal — the same class of decision as item 4's label ring, which §4's
        layer-appearance rule likewise keeps out of a picker. The `plaque` row recolours the
        provisional outline, which is the state that has a colour to spare.
        """
        if layer in (SEEDED_LAYER, UNSEEDED_LAYER):
            return COMMITTED_COLOURS[layer]
        return self.rows[layer].colour().name

    def _layers_for(self, concept: str) -> tuple[str, ...]:
        """The napari layers one picker row drives — item 5's `_layers_for`, generalised."""
        if concept == PLAQUE_LAYER:
            return (PLAQUE_LAYER, SEEDED_LAYER, UNSEEDED_LAYER)
        return (concept,)

    def _recolour(self, concept: str, colour: str) -> None:
        """**Only the provisional outline follows the swatch.** See `_colour_of`."""
        if self.viewer is None or concept not in self.viewer.layer_names():
            return
        self.viewer.set_colour(concept, colour)

    def _reweight(self, concept: str, alpha: float) -> None:
        """**Every state follows the translucency slider**, unlike the swatch.

        The slider is a weight — how much of the additive sum this layer is entitled to — and it
        means the same thing in all three states. A slider that dimmed the provisional outline
        and left the committed pair at full strength would look broken at the exact moment the
        user is checking the commit.
        """
        for layer in self._layers_for(concept):
            if self.viewer is not None and layer in self.viewer.layer_names():
                self.viewer.set_alpha(layer, alpha)

    def _toggle(self, concept: str, on: bool) -> None:
        """Item 2's arrangement: one checkbox, every layer of that concept."""
        for layer in self._layers_for(concept):
            if self.viewer is not None and layer in self.viewer.layer_names():
                self.viewer.set_visible(layer, on)

    # -- the gate -------------------------------------------------------

    def _sync(self) -> None:
        self.commit.set_stale(self.flow.is_stale())
        self.continue_button.setEnabled(self.flow.can_continue())
        #: §6i.7's disabled state. Item 5 clears `is_committed` when a parameter moves and its
        #: z-depth box greys out again; the floor does the same, because discarding the
        #: components leaves it nothing to filter.
        self.floor_box.setEnabled(self.is_committed())
        self.floor_note.setVisible(not self.is_committed())

    def _on_continue(self) -> None:
        if self.flow.can_continue():
            self.continue_requested.emit()
