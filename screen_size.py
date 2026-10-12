"""Screen 6 — Cell Size Cutoff. `COMPONENT_03_PLAN.md` §6j, §4 item 6.

**Frame D, and the only screen that uses it.** It sets `min_cell_volume_um3` and nothing else —
and it **decides which cells exist**, because the cutoff is applied inside `segment_image` before
any cell is written, so cells below it are never created. That is why it sits above cell review.

**`CellSizeCutoffGUI.m` with one change.** 3DMorph has the same three panels and sets the cutoff
*to* the selected cell's size; here selection only **populates** the box, which you then edit. You
get *"click the smallest thing I would still call a cell"* without being stuck with that number.

> ### NO Z, NO COMMIT, NO COLOUR PICKER, NO BACK — and every absence is forced
>
> Everything is max-projected, so there are no arrow keys, no depth box and **no position
> indicator**: the only screen in the component with a viewer and no z. Filtering labels that
> already exist is free, so there is **no gate, no exclamation and no *Not committed* popup**. The
> encoding is a ramp rather than a set of per-layer colours, so it belongs to the colour session
> in the same way item 4's label layer does. And every volume here is a threshold-2 volume, so
> going back would invalidate the whole screen.
>
> **Each absence removes a decision rather than creating one**, which is why the hardest ramp in
> the component sits on the simplest frame.

**There is no suggested value, and this is the one screen where one was genuinely computable.**
The distribution is bimodal with a trough at 100–200 µm³ across which only two cells are lost.
**The user declined it: the histogram is the evidence and the number is theirs.** Two reasons that
is defensible — the measurement is one image and a 5xFAD one, and it was run in **one-mask mode**,
so with threshold 2 set every volume grows and the whole distribution shifts right. *That is also
why the shipped `200.0` is stale*: it was calibrated against threshold-1 volumes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ..filters import labels_touching_xy_border
from . import palette
from .frames import CanvasHost, FrameD
from .semiauto import Flow, Session, Step, parsed_numbers
from .viewer import CanvasLayout, ViewerPort, outline_2d
from .widgets import HeaderStrip, HelpButton, NavigationRow, SelectionList

__all__ = [
    "CellSizeCutoffScreen", "LAYERS", "PANELS", "LAYER_ORDER", "GRID_SHAPE", "GRID_STRIDE",
    "kept_readout", "kept_mask", "rank_colours", "label_alphas", "list_rows",
    "border_ids", "field_projection", "cell_silhouette",
]

#: The two stacked panels. **Field view above, detail below.**
PANELS: tuple[str, str] = ("Field view", "Detail")

#: Three napari layers over two panels — `grid.shape=(2, 1)` at `stride=2` puts layers 0-1 in the
#: upper panel and layer 2 in the lower.
#:
#: **`border cells` is a separate layer because `contour` is layer-wide.** napari sets it on the
#: whole `Labels` layer, not per id, so a white outline around *only* the border-touching cells
#: cannot be expressed inside the ramp layer — it is drawn as its own outline image.
LAYER_ORDER: tuple[str, ...] = ("cells", "border cells", "Detail")
GRID_SHAPE = (2, 1)
GRID_STRIDE = 2

#: **No picker on any of them**, so the third element is the form rather than a colour name.
LAYERS: tuple[tuple[str, str, float, bool], ...] = (
    ("cells", "size rank ramp", palette.KEPT_ALPHA, True),
    ("border cells", "white outline", 1.0, True),
    ("Detail", "white", 1.0, True),
)


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

def kept_readout(n_kept: int, n_total: int) -> str:
    """`cells kept: 44 of 62`, right-aligned above the field view. **Updates live.**"""
    return f"cells kept: {n_kept} of {n_total}"


def kept_mask(volumes: np.ndarray, cutoff: float) -> np.ndarray:
    """Which cells survive. `filter_cells` computes `doomed_small` purely by volume.

    **A truncated border cell below the minimum is deleted like any other** — the cutoff does not
    exempt it, decided 2026-08-13. `remove_xy_border` is deprecated and always `False`, so a cell
    is never removed *for touching the border*; it can still be removed for being small.
    """
    return np.asarray(volumes, dtype=float) >= float(cutoff)


def rank_colours(volumes: np.ndarray) -> list[str]:
    """One hex per cell, **by size RANK rather than by absolute size**.

    Rank is what makes the ramp usable on a distribution that is bimodal with a 150x spread: 459
    of 537 objects in one measured field sat under 10 µm³. Mapping absolute volume onto the ramp
    would put nearly every cell in one colour.

    §6j.2: the ranks land on `cividis[0.25, 1.0]`, not the whole ramp. **The full ramp's floor is
    relative luminance 0.017 — effectively black — and the cells this screen adjudicates are the
    small ones, which sit exactly there.** Restricting lifts the floor to 0.078.
    """
    volumes = np.asarray(volumes, dtype=float)
    ramp = palette.size_ramp_colours(volumes.size)
    order = np.argsort(np.argsort(volumes, kind="stable"), kind="stable")
    return [ramp[rank] for rank in order]


def label_alphas(volumes: np.ndarray, cutoff: float) -> list[float]:
    """0.75 kept, 0.25 removed. **The user's own rule from 2026-08-13: *"use alpha for removed."***

    A flat **3.0x at every rank**, because alpha multiplies — the ratio is rank-independent, which
    no hue-based encoding manages.

    > **§4's stated reason for this is weaker than the true one, and the conclusion still stands.**
    > §4 argues fading is *"a presence change orthogonal to hue"*; on a black canvas it is not —
    > fading IS darkening, and on a sequential ramp dark IS small. **The collision cannot happen
    > for a different reason: removal is by volume and the ramp is by volume RANK, so the faded
    > set is always a PREFIX of the ramp.** There is never a faded mid-ramp cell to confuse with a
    > full-alpha small one. The one exception was the selected cell, which §6j.3 makes white.
    """
    keep = kept_mask(volumes, cutoff)
    return [palette.KEPT_ALPHA if k else palette.REMOVED_ALPHA for k in keep]


def border_ids(labels: np.ndarray) -> np.ndarray:
    """Cells touching the xy border, through the pipeline's own `labels_touching_xy_border`.

    **They are the one population where *small* does not mean *not a cell*** — part of the cell
    was never imaged, so it measures smaller than it really is. A cutoff chosen without seeing
    which small objects are merely cropped removes real cells for being at the edge of the frame.
    """
    return labels_touching_xy_border(labels)


def field_projection(labels: np.ndarray) -> np.ndarray:
    """Every cell from above, at its real xy position — **each cell's own silhouette, painted
    largest first so the smallest ends up on top.**

    Each cell contributes the max projection of its own region of the label volume. In two-mask
    mode that region is the **measurement mask** — `pipeline.segment_image` rebuilds `labels`
    from `measure.filled` whenever `mask_coefficient` is set — which is what keeps this picture
    and the volumes in the list beside it describing the same thing. The user chose that over
    the segmentation mask on 2026-08-23 for exactly that reason.

    > ### IT WAS `labels.max(axis=0)`, AND THAT IS A MAXIMUM OVER LABEL IDS — `O2`, closed here
    >
    > Ids come from connected-component order, which is arbitrary, so where two cells overlapped
    > in xy **the higher id won** and a small cell under a larger-id one was partly or wholly
    > erased. Item 4's own docstring names the trap: *"max-projecting ids 3 and 7 yields 7 — a
    > different nucleus."* On the one screen whose job is spotting small debris, the small thing
    > was the thing that could vanish.
    >
    > **Painting order replaces arithmetic order.** Cells are laid down by descending voxel
    > count, so a large cell can never hide a small one — only the reverse, and a small cell
    > sitting inside a large one's footprint is precisely the case the user needs to see. The
    > result is still one id per pixel, so the layer stays a `Labels` layer and nothing sums.
    >
    > **Ties keep a deterministic order** — equal sizes fall back to the id — so two runs on one
    > image cannot draw different pictures.
    """
    if labels.ndim != 3:
        raise ValueError(f"expected a (z, y, x) label volume, got shape {labels.shape}")

    labels = np.asarray(labels)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    projection = np.zeros(labels.shape[1:], dtype=labels.dtype)

    #: **Bounding boxes rather than a full pass per cell.** `find_objects` is one pass over the
    #: volume; the obvious `(labels == label).any(axis=0)` per cell is 537 passes over
    #: 40 x 2304 x 2304 on a measured field.
    boxes = ndimage.find_objects(labels)
    order = sorted((label for label in range(1, counts.size) if counts[label]),
                   key=lambda label: (-int(counts[label]), label))
    for label in order:
        box = boxes[label - 1]
        if box is None:
            continue
        silhouette = (labels[box] == label).any(axis=0)
        window = projection[box[1], box[2]]
        window[silhouette] = label
        projection[box[1], box[2]] = window
    return projection


def cell_silhouette(labels: np.ndarray, label: int | None) -> np.ndarray:
    """One cell's **own** max projection, at field position and **independent of occlusion.**

    > ### THE DETAIL VIEW WAS READ OUT OF THE PROJECTION, AND IT HAD HOLES IN IT
    >
    > It was `projection == selected_id`. `field_projection` paints largest-first so the smallest
    > cell ends up on top — which is right for the field view and **wrong as a source for the
    > detail**: every pixel where a smaller cell was laid over the selected one has been
    > reassigned to that cell's id, so reading the selection back out returns the cell *minus
    > every neighbour that overlaps it in xy*.
    >
    > The picture that produced was a cell with pieces missing, on the one panel whose entire job
    > is showing the shape of the cell you are judging — and the missing pieces looked exactly
    > like the holes in the field view above it, which is what made it read as "the same picture"
    > rather than as a defect. **Found by the user walking item 6, 2026-08-25.**
    >
    > Occlusion is a property of the *field* view. The detail view has one cell in it and nothing
    > to be occluded by, so it goes to the label volume directly.

    Bounding-boxed for the reason `field_projection` records: a full pass per cell is 537 passes
    over 40 x 2304 x 2304 on a measured field. One cell needs one box.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"expected a (z, y, x) label volume, got shape {labels.shape}")

    blank = np.zeros(labels.shape[1:], dtype=bool)
    if label is None or int(label) < 1:
        return blank

    boxes = ndimage.find_objects(labels, max_label=int(label))
    if len(boxes) < int(label):
        return blank
    box = boxes[int(label) - 1]
    if box is None:
        return blank

    blank[box[1], box[2]] = (labels[box] == int(label)).any(axis=0)
    return blank


def list_rows(volumes: np.ndarray) -> list[str]:
    """Volumes as the list shows them — **smallest first**, right-aligned by the widget.

    **The sorted list already IS the distribution.** A volume histogram panel was offered and
    withdrawn on exactly that ground: ~60 cells in order means the trough at 100–200 µm³ appears
    as a visible jump between consecutive rows, and a histogram beside it draws the same evidence
    twice.
    """
    return [f"{volume:.0f}" for volume in np.sort(np.asarray(volumes, dtype=float))]


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

class CellSizeCutoffScreen(QWidget):
    """Frame D: the list is the control, two stacked panels, one number box, Continue."""

    continue_requested = Signal()
    #: **Every screen but items 1, 9a and 7b, since 2026-08-30.**
    back_requested = Signal()

    #: Two panels STACKED, three layers, stride 2 — GRID_SHAPE and GRID_STRIDE
    #: were declared at task 14 and had no way to reach the viewer until now.
    CANVAS = CanvasLayout(shape=GRID_SHAPE, stride=GRID_STRIDE)

    #: **Opens at 0 — nothing filtered, so every move is visibly subtractive.** Not
    #: `SegmentationParams`' 200.0, which was calibrated on threshold-1 volumes and does not
    #: survive threshold 2. Starting there would hide cells before the user has seen anything.
    DEFAULT_CUTOFF = "0"

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.SIZE)
        self.viewer = viewer

        self.labels: np.ndarray | None = None
        self.cell_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.volumes: np.ndarray = np.zeros(0)
        self.border: np.ndarray = np.zeros(0, dtype=np.int64)
        self.selected_id: int | None = None
        #: **The field view and the border ring, built once in `set_cells`.** Neither can move
        #: while the screen is open: `labels` is set once and the cutoff is applied on the way
        #: *out* of the screen. See `draw`.
        self.projection: np.ndarray | None = None
        self.border_outline: np.ndarray | None = None

        self.frame = FrameD("cell size cutoff", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: the list, which IS the control on this screen ---------
        self.frame.section("List", HelpButton(
            "List",
            "Every cell, smallest first, with its volume in cubic micrometres.\n\n"
            "Click one to select it: it brightens in the field view, appears in the detail "
            "panel, and its size fills the cutoff box.\n\n"
            "Rows below the cutoff are faded.\n\n"
            "! marks a cell touching the edge of the image. Part of it was not imaged, so it "
            "measures smaller than it really is."))
        self.order_note = QLabel("smallest first", self)
        self.order_note.setProperty("role", "helper")
        self.frame.add_left(self.order_note)

        self.list = SelectionList(self)
        self.list.selection_changed.connect(self._on_row_selected)
        self.frame.add_left(self.list)
        self.frame.left.addStretch(1)

        # -- centre: two stacked panels, each with its own header ---------
        self.field_header = HeaderStrip(
            [(PANELS[0], HelpButton(
                PANELS[0],
                "Every cell, viewed from above, in its real position in the image.\n\n"
                "Colour is SIZE ORDER, not absolute size.\n\n"
                "Faded cells are below the cutoff and will be deleted.\n\n"
                "The WHITE cell is the one selected in the list.\n\n"
                "Outlined cells touch the edge of the image."))],
            kept_readout(0, 0),
            HelpButton("cells kept",
                       "How many cells survive the current value, out of every cell found.\n\n"
                       "Updates live."),
            self)
        self.frame.add_panel_header(self.field_header)
        #: **ONE host, although there are two panels** — task 23. Both panels are
        #: viewboxes inside the single canvas at `grid.shape=(2, 1)`, so a second
        #: Qt widget here would be an empty black rectangle under the picture, and
        #: the detail panel would render inside the upper half of the upper one.
        #: The second header strip still exists, because naming the panels IS what
        #: Qt contributes to this frame.
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        self.detail_header = HeaderStrip([PANELS[1]], "", None, self)
        self.frame.add_panel_header(self.detail_header)

        # -- right: one number box and Continue --------------------------
        self.min_cell_volume_um3 = self._number(
            "min_cell_volume_um3", self.DEFAULT_CUTOFF,
            "The smallest volume that still counts as a cell. Anything below it is deleted and "
            "never reaches review or measurement.\n\n"
            "Click a row in the list to fill this box with that cell's size, then adjust it up "
            "or down.\n\n"
            "LIVE: cells below the value fade as you type. Nothing is recomputed.")
        self.min_cell_volume_um3.textChanged.connect(self._on_cutoff_changed)

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
        name = QLabel(label, self)
        row = QHBoxLayout()
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(HelpButton(label, help_text, self))
        host = QWidget(self)
        host.setLayout(row)
        self.frame.add_right(host)

        edit = QLineEdit(value, self)
        self.frame.add_right(edit)
        return edit

    # -- loading --------------------------------------------------------

    def set_cells(self, labels: np.ndarray, volumes_um3) -> None:
        """The post-split, threshold-2 cells and their volumes, in label-id order.

        **By this point the cells are real and split.** What bounds this view since the debris
        patch of 2026-08-24 is **the nuclei, not a size floor**: the measurement mask is split by
        the microglial nuclei and every component none of them claims is debris, which never
        enters `derivation.labels`. So the screen shows one entry per nucleated cell — on the
        measured crop 9, where the old segment-then-grow path reached it with 21.

        `speck_floor_um3` used to be the answer here — *"537 objects to 62 on a measured
        field"* — and **it was deleted with this patch.** It removed ~88% of objects before
        anything could count them, which under the new architecture would have meant debris
        could never account for what the measurement mask held. Nothing replaced it because
        nothing needs to: an unclaimed speck is debris now rather than a tile.
        """
        volumes = np.asarray(volumes_um3, dtype=float)
        n = int(labels.max())
        if volumes.size != n:
            raise ValueError(f"{volumes.size} volumes for {n} labels")
        self.labels = labels
        self.cell_ids = np.arange(1, n + 1, dtype=np.int64)
        self.volumes = volumes
        self.border = border_ids(labels)
        #: ⇒ **THE FIELD VIEW IS BUILT HERE, ONCE — patch item A4, 2026-09-03**
        #:
        #: `draw()` used to rebuild both of these every time it ran, and it runs on every
        #: character typed into the cutoff box and on every row selected: `np.bincount`
        #: **2.63 s** plus `ndimage.find_objects` **0.68 s** over the full label volume, then
        #: `np.isin` and an erosion — roughly **4 s a character**.
        #:
        #: **None of it could ever change.** `labels` is set here and nowhere else, `border_ids`
        #: is a function of `labels` alone, and the cutoff is applied on the way *out* of the
        #: screen rather than to the picture. The box's own help text already claimed *"LIVE:
        #: cells below the value fade as you type. Nothing is recomputed."* — this makes that
        #: sentence true.
        self.projection = field_projection(labels)
        self.border_outline = outline_2d(np.isin(self.projection,
                                                 self.border)).astype(np.float32)
        self.selected_id = None
        self._refresh_list()
        self.draw()

    def cutoff(self) -> float:
        try:
            return float(self.min_cell_volume_um3.text().strip())
        except ValueError:
            return 0.0

    def kept(self) -> np.ndarray:
        return kept_mask(self.volumes, self.cutoff())

    # -- the list -------------------------------------------------------

    def _sorted_order(self) -> np.ndarray:
        """List position -> label id. Smallest first, as 3DMorph sorts (flipped: it is
        largest-first there, and the plan's drawing is smallest-first)."""
        return self.cell_ids[np.argsort(self.volumes, kind="stable")]

    def _refresh_list(self) -> None:
        order = self._sorted_order()
        flags = [bool(np.isin(cell_id, self.border)) for cell_id in order]
        self.list.set_items(list_rows(self.volumes), flags)
        self._refresh_dimming()

    def _refresh_dimming(self) -> None:
        """**Rows below the cutoff are muted, not removed** — the trough has to be visible by
        eye, and that is the judgement the screen exists to support.
        """
        order = self._sorted_order()
        keep = kept_mask(self.volumes, self.cutoff())
        self.list.set_dimmed([i for i, cell_id in enumerate(order) if not keep[cell_id - 1]])


    #: > ### THERE IS NO CANVAS SELECTION HERE, AND THAT IS A DECISION — 2026-08-21
    #: >
    #: > A click on the field view and the detail view does nothing. **The user's rule:**
    #: > selection can only be made from the list.
    #: >
    #: > Task 26 built the pointer path — item 6 had no `bind_gestures` at all, so the screen
    #: > answered neither the list nor the picture — and it was withdrawn once the list worked.
    #: > `SelectionList`'s rows are what became clickable, and that is the whole control.
    #: >
    #: > **So this screen deliberately has no `bind_gestures`**, which `app.show_step` calls
    #: > only where a screen offers one. It is also the one viewer screen with no z, so there is
    #: > no arrow-key binding to want either — every absence on this screen removes a decision.


    def _on_row_selected(self, index: int) -> None:
        """**Selection POPULATES the box; 3DMorph sets the cutoff to it.**

        That one difference is the whole point: you get *"click the smallest thing I would still
        call a cell"* without being stuck with exactly that number.
        """
        order = self._sorted_order()
        if not 0 <= index < order.size:
            return
        self.selected_id = int(order[index])
        self.min_cell_volume_um3.setText(f"{self.volumes[self.selected_id - 1]:.0f}")
        self.draw()

    def _on_cutoff_changed(self) -> None:
        """**Live.** Filtering labels that already exist needs no re-segmentation — almost
        nothing else in this component can do that.
        """
        self._refresh_dimming()
        self._refresh_readout()
        self.draw()

    def _refresh_readout(self) -> None:
        self.field_header.set_readout(
            kept_readout(int(self.kept().sum()), int(self.cell_ids.size)))

    # -- drawing --------------------------------------------------------

    def cell_colours(self) -> dict[int, str]:
        """Ramp by rank, **with the selected cell overridden to white.**

        A selected border cell is a white fill inside a white outline and loses its ring.
        Accepted: its list row still carries the `!`, and it is the row you just clicked.
        """
        colours = dict(zip((int(i) for i in self.cell_ids), rank_colours(self.volumes)))
        if self.selected_id is not None:
            colours[self.selected_id] = palette.SELECTED_FILL
        return colours

    def cell_alphas(self) -> dict[int, float]:
        return dict(zip((int(i) for i in self.cell_ids), label_alphas(self.volumes, self.cutoff())))

    def draw(self) -> None:
        """**Only what varies.** The colour and alpha dicts, and the detail silhouette.

        The field projection and the border ring are held from `set_cells` — see there for why
        neither can move — so what is left is a per-cell colour, a per-cell alpha, and one
        bounding-boxed silhouette. That is what makes the cutoff box live without the box having
        to move to Enter: after A4 there is nothing left for a keystroke to recompute.
        """
        if self.viewer is None or self.labels is None or self.projection is None:
            return

        self.viewer.show_labels(LAYER_ORDER[0], self.projection,
                                colours=self.cell_colours(), alphas=self.cell_alphas(),
                                contour=0, visible=True)

        self.viewer.show_image(LAYER_ORDER[1], self.border_outline,
                               colour="white", alpha=1.0, visible=True,
                               contrast_limits=(0.0, 1.0))

        #: **White on black, no nucleus, and that is necessary rather than tidy**: the selected
        #: cell is almost always a small one — that is what the screen adjudicates — and small
        #: cells sit at the ramp's dark end, where shape cannot be read at all.
        #: **From the labels, not from the projection.** See `cell_silhouette`: reading the
        #: selection back out of a largest-first projection returns the cell with every
        #: overlapping neighbour punched out of it.
        detail = cell_silhouette(self.labels, self.selected_id)
        self.viewer.show_image(LAYER_ORDER[2], detail.astype(np.float32),
                               colour="white", alpha=1.0, visible=True,
                               contrast_limits=(0.0, 1.0))
        self._refresh_readout()

    # -- the gate -------------------------------------------------------

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
            "min_cell_volume_um3": (self.min_cell_volume_um3.text(), float),
        })

    def _sync(self) -> None:
        """**No commit gate on this screen**, so Continue is live from arrival."""
        self.flow.session.chosen.update(self.parameters())
        self.continue_button.setEnabled(self.flow.can_continue())

    def _on_continue(self) -> None:
        if self.flow.can_continue():
            self.continue_requested.emit()
