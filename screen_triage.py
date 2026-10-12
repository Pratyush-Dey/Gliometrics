"""Screen 7a — Cell Review, Triage. `COMPONENT_05_PLAN.md` §3b, §3c, §3d.7, §3d.8, §3d.9.

**Frame E, and the only screen that uses it.** *"What it is for: disposing of the cells that need
no work, and finding the ones that do."*

> ### THE SCREEN TAKES A DATA PROVIDER, NOT A `Pool` — AND THAT IS ARCHITECTURAL
>
> §3d's layouts are specified against a pool — *"the stored `thumbnail`"*, `load_arrays`. **In
> semi-auto none of that is stored yet**, because §4a.5 keeps steps 1–7 entirely in memory and the
> first write is at item 8. Writing this screen against `Pool` directly would make item 7
> unbuildable without first committing the session — **exactly the property that amendment bought.**
>
> So it takes a list of `ReviewCell`. Semi-auto builds them from memory; `review --pool` builds
> them from `CellRecord`s. Everything either world needs is present in both.

**Three screens per file — HARD, MEDIUM, EASY — by within-file quantile.** The bands are relative
rather than absolute because **QC scores are not comparable across images**, which is the same
reason ordering is per file. A within-file quantile is self-calibrating and needs no constant.

> **The trade, recorded rather than left to be discovered:** every file presents all three screens
> whatever its quality, so a clean image still shows a HARD screen, and the bands mean *rank within
> this image* rather than *difficulty*. **The band is never on the face of the screen** — it lives
> in the `screen 1 of 3` popup, because the tool does not grade the reviewer's work in a heading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import qc
from . import theme
from .frames import GRID_COLUMNS, FrameE
from .widgets import ConfirmPopup, HelpButton, ask_rejection

__all__ = [
    "TriageScreen", "ReviewCell", "Tile", "BANDS", "HARD_FRACTION", "MEDIUM_FRACTION",
    "DELETE_GLYPH", "DELETE_HELP",
    "TILE_PX", "MARK_PX", "BORDER_HELP", "cell_pane", "FORCED_METHODS", "is_forced",
    "triage_keys", "review_order", "band_sizes", "split_into_screens",
    "file_position", "screen_position", "cells_on_screen", "accept_rest_warning",
    "FILE_DONE_MESSAGE", "proceed_label",
]

#: §3b: **15% / 35% / rest**, and the names never appear on the screen itself.
BANDS: tuple[str, str, str] = ("hard", "medium", "easy")
HARD_FRACTION = 0.15
MEDIUM_FRACTION = 0.35

#: Five across at this size fills the centre of a laptop window without a horizontal scrollbar,
#: which `FrameE` switches off outright.
TILE_PX = 96

#: The mark row beneath the thumbnail, **reserved on every tile whether or not it carries a
#: `!`**. Fixed rather than hinted because an empty `QLabel` and one holding `"!"` return
#: different height hints — 14 px against 12 px, measured — and a grid row takes the tallest
#: item in it, so tiles with a mark sat one pixel lower than tiles without. A reserved row also
#: means a `!` appearing costs no reflow: the space it needs was always there.
MARK_PX = 18

#: **`×` — U+00D7 — and not a trashcan, because the bundled font has no trashcan.**
#:
#: > ### ⇒ CHECKED AGAINST THE FONT, NOT ASSUMED — patch item B5, 2026-09-03
#: >
#: > IBM Plex Sans carries **no** U+1F5D1 wastebasket, no U+232B erase-to-the-left and no
#: > U+2716 heavy multiplication. It does carry `×`. The interface bundles its typeface and
#: > refuses to substitute — `FontNotBundled` — so a missing glyph is a tofu box on Bunya
#: > rather than a fallback that happens to look fine on this machine.
#: >
#: > **The user chose `×` over a painted icon.** One character, one line, and it sits on the
#: > same baseline as the `!` beside it.
DELETE_GLYPH = "\u00d7"

#: **Pinned to the left of the mark row, not centred.** The user's choice: the button never
#: drifts between tiles, so the hand learns one position. The `!` and its `?` stay centred as a
#: pair, which is what §3d.8 draws.
DELETE_HELP = (
    "Reject this cell without opening it.\n\n"
    "You are asked the same two questions the editor asks — whether it is debris or a real "
    "cell, and if it is a real cell, how many. The tile leaves the grid straight away, and "
    "`u` undoes it."
)

#: §3d's *Seven popups*: the `!`'s own "?", and **the second deliberate exception to the
#: controls-only rule** — the flag *"is the one thing on this screen a user could not possibly
#: infer, yet it is the whole reason border cells are marked."*
BORDER_HELP = (
    "This cell touches the edge of the image. Part of it was never imaged, so it "
    "measures smaller than it is.\n\n"
    "Small does not mean \"not a cell\" for these."
)


def cell_pane(thumbnail) -> np.ndarray:
    """The **cell** pane of a stored thumbnail, as a 2D boolean mask.

    > ### THE STORED THUMBNAIL HAS TWO PANES AND THIS SCREEN WANTS ONE — found by running
    >
    > `pipeline._thumbnail` stacks **two** max projections, `[cell, nucleus > 0]`, and stores
    > them as one `(2, H, W)` array. `ReviewCell.thumbnail` is specified as the *"mask-only max
    > projection — no nucleus channel, no intensity, nothing written on it"*, **so the judgement
    > is shape**. Both are right; nothing converted between them, and the tile unpacked
    > `mask.shape` into two names and raised `ValueError: too many values to unpack` **inside
    > `paintEvent`**, where Qt turns a traceback into a warning and paints an empty tile.
    >
    > **Every triage test passes a 2D fixture**, which is why 43 of them could not catch it —
    > the assertions were right and the fixture could not exercise them. §3's lesson 2 exactly,
    > and the test beside this function is built from `pipeline._thumbnail`'s own output rather
    > than from a hand-written array.
    >
    > It lives here, in one function, because **the pool-backed route reads the same stored
    > array** — Component 5 opens `load_arrays` rather than a `Derivation`, so a fix made at the
    > flow driver's seam would leave the identical defect waiting on the other route.

    Pane 0 rather than an `any` over the panes: the nucleus is *inside* the cell mask, so the
    two are almost the same picture — but only almost, and a nucleus voxel outside the mask
    would silently widen the shape being judged.
    """
    array = np.asarray(thumbnail)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"a thumbnail is 2D, or panes of 2D; got shape {array.shape}")
    return array.astype(bool)


@dataclass
class ReviewCell:
    """One tile's worth of data, from either world.

    `thumbnail` is the **mask-only max projection** — 3DMorph cell-size-cutoff style, no nucleus
    channel, no intensity, nothing written on it. **So the judgement is shape**, which is the same
    reason item 6's detail panel is white on black.
    """

    uid: str
    thumbnail: np.ndarray | None = None
    touches_xy_border: bool = False
    resolved: bool = False

    #: **The two triggers, and they force the hard band rather than entering the ranking.**
    #: No nucleus at all is unambiguous whatever one thinks of nucleus quality; a
    #: `centroid_fallback` soma means erosion consumed the cell and there is no soma to speak
    #: of. Neither can be outweighed by a clean score elsewhere, which is the whole point of a
    #: trigger. `largest_remnant` is deliberately NOT one: every cell taking that path has no
    #: nucleus and is already forced by the first rule, so naming it would restate one fact.
    has_nucleus: bool = True
    soma_method: str = ""

    #: **The soma category.** `soma_volume_um3` is two-sided against the image median; the other
    #: two are one-sided, because there is no such thing as too much of the nucleus inside the
    #: soma or too compact a cell body.
    nucleus_in_soma: float | None = None
    soma_volume_um3: float | None = None
    soma_bbox_fill: float | None = None

    #: **The process category.** `process_fraction` is two-sided — a bare soma and a cell with
    #: no soma at all are both wrong. `detached_fraction` reserves its best bracket for cells
    #: with none, so any detached material keeps a cell out of it.
    process_fraction: float | None = None
    detached_fraction: float | None = None


# --------------------------------------------------------------------------
# Ordering and banding — no Qt
# --------------------------------------------------------------------------

#: Soma rules that mean **there is no soma**, not merely a worse one. `soma.py` names
#: `largest_remnant` and `centroid_fallback` together as "a different definition"; only the
#: second is a trigger here, because the first happens exactly to the cells `has_nucleus`
#: already catches.
FORCED_METHODS: frozenset[str] = frozenset({"centroid_fallback"})


def is_forced(cell: ReviewCell) -> bool:
    """Whether this cell reaches the hard band regardless of how it ranks."""
    return (not cell.has_nucleus) or (cell.soma_method in FORCED_METHODS)


def triage_keys(cells) -> list[tuple[int, float, float]]:
    """Worst first, and **the key is a property of the population, not of one cell.**

    > ### ⇒ IT WAS A PER-CELL `triage_key` UNTIL 2026-09-03, AND IT COULD NOT STAY ONE
    >
    > The old key read two numbers off the cell in front of it. Ranking within the image cannot
    > be done that way: a cell's bracket depends on where every other cell fell. So the function
    > takes the list, and `review_order` is the only caller that needs to know.

    Two categories, each ranked and bracketed on its own signals, then summed — see
    `prestomorph.qc` for why that yields *bad in both beats bad in one* with no weights anywhere.
    Forced cells sort ahead of everything, and are still ordered among themselves by the same
    key, so the hard band is not arbitrary inside itself.
    """
    cells = list(cells)
    if not cells:
        return []

    soma = qc.category([
        qc.low_badness([c.nucleus_in_soma for c in cells]),
        qc.log_badness([c.soma_volume_um3 for c in cells]),
        qc.low_badness([c.soma_bbox_fill for c in cells]),
    ])
    process = qc.category(
        [
            qc.log_badness([c.process_fraction for c in cells]),
            #: **Passed through untransformed**: a larger detached share is already worse, and
            #: `zero_is_best` is what keeps a mostly-zero signal from collapsing the scale.
            [c.detached_fraction for c in cells],
        ],
        zero_is_best=(1,),
    )
    combined = qc.order_key(soma, process)
    return [(0 if is_forced(cell) else 1, bracket, raw)
            for cell, (bracket, raw) in zip(cells, combined)]


def review_order(cells) -> list[ReviewCell]:
    """**Sorted by position, not by identity.** The first version keyed a dict on `id(cell)`,
    which is safe only because the list holds every cell alive for the call — and §70 item 2
    was written the same week about exactly that reasoning being one refactor from wrong. An
    index needs no such argument, and it makes the sort stable into the bargain: equal keys keep
    the order the caller gave, which is label order."""
    cells = list(cells)
    keys = triage_keys(cells)
    order = sorted(range(len(cells)), key=lambda i: (keys[i], i))
    return [cells[i] for i in order]


def band_sizes(n_cells: int, n_forced: int = 0) -> tuple[int, int, int]:
    """How many cells land on each of the three screens.

    **HARD is definable by RULE and the rest by RANK, and the two can disagree.** §3b gives the
    quantile as 15% and also says a cell the rule condemns *is* hard. When the forced set is
    larger than 15%, the rule wins: **HARD grows to hold all of them.** If a third of the cells
    have no nucleus or no soma, a third of the cells genuinely are hard, and pushing the
    remainder into MEDIUM would file them under a band that means something else.

    **The rule was *no `agreement_score`* until 2026-09-03** and is now `is_forced` — no nucleus
    assigned, or a `centroid_fallback` soma. The first is the same population under a more
    direct test; the second is new, and is why this can grow for a reason the nucleus cannot
    explain.

    Recorded as a decision because §3b states both halves and does not say which yields.
    """
    if n_cells <= 0:
        return (0, 0, 0)
    hard = max(round(n_cells * HARD_FRACTION), min(n_forced, n_cells))
    hard = min(hard, n_cells)
    medium = min(round(n_cells * MEDIUM_FRACTION), n_cells - hard)
    return (hard, medium, n_cells - hard - medium)


def split_into_screens(cells) -> list[list[ReviewCell]]:
    """The three screens, worst first. Always three, even when one is empty."""
    ordered = review_order(cells)
    forced = sum(1 for cell in ordered if is_forced(cell))
    hard, medium, _ = band_sizes(len(ordered), forced)
    return [ordered[:hard], ordered[hard:hard + medium], ordered[hard + medium:]]


def file_position(index: int, total: int) -> str:
    """`file 3 of 712`. Files are taken in the order they were **merged**, and none is skipped."""
    return f"file {index + 1} of {total}"


def screen_position(index: int, total: int = 3) -> str:
    """`screen 1 of 3`. **The band name is not here** — it is in the popup."""
    return f"screen {index + 1} of {total}"


def cells_on_screen(n: int) -> str:
    return f"{n} cell{'' if n == 1 else 's'} on this screen"


def accept_rest_warning(n_remaining: int, n_unseen: int) -> str:
    """**The only gate on this screen**, and it guards the only action whose slip is bigger than
    one undo step. `Accept This Screen` has none: everything it clears is visible and `u` reaches
    all of it.
    """
    return (f"Accept {n_remaining} remaining cells in this file? "
            f"{n_unseen} of them are on screens you have not looked at.")


#: **The empty-grid state, and it is not an accept.** Shown when the file has nothing unresolved
#: left — every cell decided in rich review or cleared by an accept. Worded as a statement of fact
#: rather than a congratulation: the screen reports, it does not applaud.
FILE_DONE_MESSAGE = "This file has no unresolved cells left."


def proceed_label(file_index: int, file_total: int) -> str:
    """What the proceed button says once the file is done — **`Next File` or `Continue`.**

    The button always emits `file_accepted`, which is triage's *"this file is done"*; **what
    happens next is the host's decision, not this screen's.** In the batch route that signal takes
    the reviewer to the next file; in semi-auto `ReviewStep` forwards it as `continue_requested`
    and the flow moves to **item 9a**, because there is no next file to go to.

    So the *label* is the only half that can differ, and it is derived from the counter the screen
    already shows rather than from a mode flag handed down: with `file 1 of 1` there is no next
    file and the button says `Continue`. **A screen that named a file it could not open would be
    lying in the one place the reviewer is looking.**
    """
    return "Next File" if file_index + 1 < file_total else "Continue"


# --------------------------------------------------------------------------
# The tile
# --------------------------------------------------------------------------

class Tile(QWidget):
    """One thumbnail, **white on black**, with its `!` beneath when it touches the xy border.

    **No hover ring, and that was decided rather than skipped.** It was offered as the affordance
    that a tile is clickable; the user's reasoning ended it — *"why do we have to indicate that a
    tile is clickable, non-clickable tiles are gaps."* **An affordance that is uniform does not
    need a marker; a marker that is uniform is decoration.**
    """

    clicked = Signal(str)
    #: **The `×`, on every tile** — patch item B5.
    delete_requested = Signal(str)

    def __init__(self, cell: ReviewCell, parent: QWidget | None = None,
                 *, help_button: QWidget | None = None):
        super().__init__(parent)
        self.cell = cell
        #: > ### THE TILE MUST BE THE SIZE OF THE TILE — found by walking the screen, 2026-08-26
        #: >
        #: > A `QWidget` at the default `Preferred` policy takes whatever its grid cell offers,
        #: > and `FrameE`'s grid had no stretch anywhere, so `QGridLayout` handed every leftover
        #: > pixel of a resizable host to the occupied columns. **Measured at 2560x1400: one
        #: > cell on the screen produced a tile 2496 px wide and 1220 px tall**; seven produced
        #: > five columns 490 px wide and rows 604 px apart. The thumbnail stayed 96 px because
        #: > `_ThumbnailLabel` is fixed, so the picture looked almost right — but the `!` is
        #: > centred in the tile, and it was therefore centred in 2496 px, which put it in the
        #: > **middle of the window** rather than under its own thumbnail.
        #: >
        #: > **Every triage test asserts row and column indices**, which stayed correct
        #: > throughout: `grid_position` reads the layout, not the geometry. The defect lived
        #: > entirely in pixels, so the suite could not see it.
        self.setFixedSize(TILE_PX, TILE_PX + 2 + MARK_PX)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        self.image = _ThumbnailLabel(cell.thumbnail, self)
        self.image.clicked.connect(lambda: self.clicked.emit(self.cell.uid))
        column.addWidget(self.image)

        #: **`#e3b617`, inherited from item 6 unchanged.** A border cell is a caution, not an
        #: error, so it must not take the colour item 1 reserved for failure — and §3c requires
        #: it *"reuses item 6's `!` rather than inventing a second vocabulary"*.
        self.mark = QLabel("!" if cell.touches_xy_border else "", self)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        if cell.touches_xy_border:
            self.mark.setStyleSheet(f"color: {theme.WARNING};")

        #: **The `!` and its "?" are centred as a PAIR**, so the mark sits a few pixels left of
        #: the thumbnail's midline on the one tile that carries the button. The alternative was
        #: letting the "?" overhang into the gutter, which cannot be done: a child is clipped to
        #: its parent, and widening the tile would widen the whole grid column and step every
        #: column after it out of line. §3d.8 draws `! ?` as a pair, and a pair is what this is.
        self.help_button = help_button
        mark_row = QHBoxLayout()
        mark_row.setContentsMargins(0, 0, 0, 0)
        mark_row.setSpacing(4)
        #: **The delete button first, so it is pinned to the left edge** — the user's choice.
        #: Everything after it is inside the two stretches and stays centred, so the `!` and its
        #: `?` sit where they always sat and the `×` never moves between tiles.
        self.delete_button = QPushButton(DELETE_GLYPH, self)
        self.delete_button.setFlat(True)
        self.delete_button.setFixedSize(MARK_PX, MARK_PX)
        self.delete_button.setToolTip("Reject this cell")
        self.delete_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {theme.MUTED}; "
            f"border: none; padding: 0; }}")
        self.delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.cell.uid))
        mark_row.addWidget(self.delete_button)
        mark_row.addStretch(1)
        mark_row.addWidget(self.mark)
        if help_button is not None:
            help_button.setParent(self)
            help_button.setFixedHeight(MARK_PX)
            mark_row.addWidget(help_button)
        mark_row.addStretch(1)
        mark_host = QWidget(self)
        mark_host.setLayout(mark_row)
        column.addWidget(mark_host)


class _ThumbnailLabel(QLabel):
    """Paints a boolean mask white on black at `TILE_PX`, and reports clicks."""

    clicked = Signal()

    def __init__(self, thumbnail, parent=None):
        super().__init__(parent)
        self.setFixedSize(TILE_PX, TILE_PX)
        self.setStyleSheet(f"background: {theme.CANVAS};")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        #: **Reduced HERE, not in `paintEvent`, and a mutation is what forced the move.**
        #: Qt catches whatever a paint handler raises, prints a warning and paints nothing — so
        #: the original defect was invisible to a caller and a test that painted the tile and
        #: asserted *no exception* could not fail either way. Converting at construction puts
        #: the refusal somewhere a caller and a test can both see it, and it converts once
        #: instead of on every repaint of every tile in a scrolling grid.
        self._thumbnail = None if thumbnail is None else cell_pane(thumbnail)

    def mousePressEvent(self, event):       # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):            # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.CANVAS))
        if self._thumbnail is not None and self._thumbnail.size:
            mask = self._thumbnail
            height, width = mask.shape
            grey = np.where(mask, 255, 0).astype(np.uint8, copy=False)
            grey = np.ascontiguousarray(grey)
            image = QImage(grey.data, width, height, width, QImage.Format.Format_Grayscale8)
            painter.drawImage(self.rect(), image)
        painter.end()


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

@dataclass
class TriageAction:
    """One undoable triage action. **Both are pure status flips, so nothing is destroyed.**

    That is the whole distinction from rejection, which in the `debris` case drops the cell's
    arrays and is genuinely irreversible.
    """

    #: `"screen"` · `"file"` · `"delete"`. The first two are pure status flips; `"delete"` is
    #: a rejection taken from the grid, and it is undoable for the same reason — **within a
    #: session nothing is destroyed.** A cell's arrays are dropped at item 8, not here, so
    #: undoing a deletion is dropping a dictionary entry and repainting one crop.
    kind: str
    cleared: tuple[str, ...] = ()

    #: **The band the action was taken on**, so undo can go back to *where it happened* rather
    #: than one step back from wherever the reviewer now is. Those were the same thing until
    #: rich review could advance the band on its own — see `undo`.
    screen: int = 0


class TriageScreen(QWidget):
    """Frame E: header block, five-across scrolling grid, two buttons and one key."""

    cell_entered = Signal(str)
    #: **Item 7's Back, forwarded by `ReviewStep`.** It leaves review entirely — see there.
    back_requested = Signal()
    screen_accepted = Signal()
    file_accepted = Signal()
    #: **A rejection taken from the grid** — `(uid, editing.Resolution)`, patch item B5. The
    #: decision itself is recorded by `ReviewStep`, which is where every other decision is.
    cell_rejected = Signal(str, object)
    #: **What `undo` restored, so the decisions can go with the tiles** — patch item B5.
    #: A tuple of uids; empty when the action cleared nothing that is still on this file.
    undone = Signal(tuple)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.screens: list[list[ReviewCell]] = [[], [], []]
        self.screen_index = 0
        self.file_index = 0
        self.file_total = 1
        self.last_action: TriageAction | None = None
        self.tiles: dict[str, Tile] = {}

        #: Replaced in tests. The real one shows §3c's gate.
        self.confirm_accept_rest = self._ask_accept_rest
        #: **The delete button's two popups, replaceable the same way** — patch item B5.
        #: Every modal in this project is a named hook so the suite can answer it; an
        #: `exec_()` with no display does not fail, it blocks forever.
        self.ask_rejection = self._ask_rejection

        #: **The tile row height is handed over rather than duplicated.** `FrameE` needs it to
        #: put a floor under the grid — without one the whole-screen scroll it now sits in can
        #: never engage — and `frames` cannot import `TILE_PX` back from here without a cycle.
        #: It is the tile plus its 2 px gap and mark strip, which is `CellTile`'s own fixed size.
        self.frame = FrameE("Cell Review — Triage", row_px=TILE_PX + 2 + MARK_PX, parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- the header block --------------------------------------------
        self.path_line = QLabel("", self)
        self.frame.add_header_line(self.path_line)

        counters = QHBoxLayout()
        counters.setSpacing(24)
        self.file_label = QLabel("", self)
        self.file_label.setProperty("role", "helper")
        self.screen_label = QLabel("", self)
        self.screen_label.setProperty("role", "helper")
        self.count_label = QLabel("", self)
        self.count_label.setProperty("role", "helper")
        counters.addWidget(self.file_label)
        counters.addWidget(HelpButton(
            "file", "Files are taken in the order they were merged into the pool. None is "
                    "skipped. Files merged while you work appear at the bottom next time you "
                    "open the tool.", self))
        counters.addWidget(self.screen_label)
        counters.addWidget(HelpButton(
            "screen", "Each file is split into three screens — hard, medium and easy — by how "
                      "confident the pipeline is about each cell.\n\n"
                      "This is a RANK WITHIN THIS IMAGE, not a grade. The scores are not "
                      "comparable between images, so a clean image still has a hard screen.",
            self))
        counters.addWidget(self.count_label)
        counters.addStretch(1)
        counters_host = QWidget(self)
        counters_host.setLayout(counters)
        self.frame.add_header_line(counters_host)

        #: **Sits with the counters because it is one** — the honest reading of `0 cells on this
        #: screen` when there is nothing behind it either. Hidden until the file is done, so the
        #: header block does not reserve a line for a state most screens never reach.
        #:
        #: `helper`, following items 1.5, 5 and 9b: **the disabled state carried twice** — the
        #: greyed control for everyone and the sentence for anyone who cannot separate the two
        #: greys. Both accepts go dead in this state, and this is the line that says why.
        self.done_line = QLabel(FILE_DONE_MESSAGE, self)
        self.done_line.setProperty("role", "helper")
        self.done_line.setVisible(False)
        self.frame.add_header_line(self.done_line)

        #: **No "?" on this line, and that is a rule rather than an omission.** A key with no
        #: button has no home in the "?" convention — but the legend already says what the key
        #: does, so a mark beside it would explain an explanation. *Where a line is already its
        #: own explanation, it takes no mark.*
        self.undo_legend = QLabel("press  u  to undo the last action", self)
        self.undo_legend.setProperty("role", "helper")
        self.frame.add_header_line(self.undo_legend)

        # -- the bar: two buttons, and neither has a key ------------------
        #: **The asymmetry is the safety property, and it is the user's design.** The two actions
        #: that dispose of work need a deliberate pointer act and cannot be fat-fingered; the
        #: action that recovers work is the fast one. B had proposed keys for all three on
        #: throughput grounds — at ~20,000 decisions the accepts are already cheap, and it is the
        #: slip that is expensive.
        self.frame.add_to_bar(HelpButton(
            "Accept This Screen",
            "Accepts every cell still showing on this screen. Cells you already dealt with in "
            "the editor are unaffected.\n\nPress u to undo it.", self))
        self.accept_screen_button = QPushButton("Accept This Screen", self)
        self.accept_screen_button.clicked.connect(self.accept_this_screen)
        self.frame.add_to_bar(self.accept_screen_button)
        self.frame.bar.addStretch(1)

        self.frame.add_to_bar(HelpButton(
            "Accept Rest Of File",
            "Accepts everything left in this file, including screens you have not seen, and "
            "moves on to the next file.\n\nThere is no accept-all for the study.", self))
        self.accept_file_button = QPushButton("Accept Rest Of File", self)
        self.accept_file_button.clicked.connect(self.accept_rest_of_file)
        self.frame.add_to_bar(self.accept_file_button)

        #: **The third button, and the only one that accepts nothing.** It appears only when the
        #: file is done, in the same corner as `Accept Rest Of File` — the corner that means
        #: *forward* on this screen. It takes no confirm gate: §3c's popup guards accepting cells
        #: the reviewer has not seen, and by here there are none left to accept.
        self.proceed_help = HelpButton(
            "Continue",
            "Every cell in this file has been decided — in the editor, or by an accept.\n\n"
            "There is nothing left to review here.", self)
        self.proceed_help.setVisible(False)
        self.frame.add_to_bar(self.proceed_help)
        self.proceed_button = QPushButton("Continue", self)
        self.proceed_button.clicked.connect(self.proceed)
        self.proceed_button.setVisible(False)
        self.frame.add_to_bar(self.proceed_button)

        #: > ### BACK SITS IN THE BAR, NOT IN A RIGHT-HAND COLUMN — 2026-08-30
        #: >
        #: > The user described the shared row as *"back on the left continue on the right of the
        #: > bottom right panel"*, which is the geometry of every Frame-C screen. **This screen
        #: > has no right-hand column** — its controls live in the top bar — so the button goes
        #: > where this screen's own forward controls already are, rather than inventing a panel
        #: > to hold it.
        #: >
        #: > **Always visible, unlike `Continue`.** That one appears only once the file is done;
        #: > leaving is available whether or not there is anything left to review.
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.back_requested)
        self.frame.add_to_bar(self.back_button)

    # -- loading --------------------------------------------------------

    def _ask_accept_rest(self, message: str, n_remaining: int) -> bool:
        return ConfirmPopup(message, f"Accept All {n_remaining}", "Cancel",
                            title="Accept rest of file", parent=self).ask()

    def set_file(self, cells, path: str = "", file_index: int = 0, file_total: int = 1) -> None:
        """Take one file's cells and split them into the three screens."""
        self.screens = split_into_screens(cells)
        self.screen_index = 0
        self.file_index, self.file_total = file_index, file_total
        self.last_action = None
        self.path_line.setText(path)
        self._rebuild()

    # -- the grid -------------------------------------------------------

    def pending_uids(self) -> list[str]:
        """Every unresolved cell, **in the order this screen will offer them.**

        Worst band first, then medium, then easy — `split_into_screens`' own order, which is
        not the order the cells arrived in. The pool route's prefetcher builds `CellEditor`s
        against this list, so the budget is spent on the cells the reviewer reaches first
        rather than on whichever ones happen to be early in the file.
        """
        return [cell.uid for screen in self.screens for cell in screen if not cell.resolved]

    def visible_cells(self) -> list[ReviewCell]:
        """**Only unresolved cells show.** A resolved one leaves its gap and does not reflow."""
        return [cell for cell in self.screens[self.screen_index] if not cell.resolved]

    def _rebuild(self) -> None:
        while self.frame.grid.count():
            widget = self.frame.grid.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.tiles.clear()

        #: **Position is by index in the band, not by packing order.** A resolved cell is simply
        #: not placed, so the gap stays where the tile was — *"position is how a reviewer keeps
        #: their place, and reflowing moves every tile they have not looked at yet."*
        #: **One "?" per screen, on the first marked tile still showing** — §3d.8 draws `! ?`
        #: once and a bare `!` on every other bordered tile. Recomputed on each rebuild rather
        #: than pinned to a cell, so the button follows the first `!` when that cell is
        #: resolved and its tile leaves a gap. **A mark whose explanation can disappear while
        #: other marks are still on the screen is worse than no explanation**, and this is the
        #: one help text §3d calls *"the strongest exception in the project"*.
        explained = False
        for index, cell in enumerate(self.screens[self.screen_index]):
            if cell.resolved:
                continue
            help_button = None
            if cell.touches_xy_border and not explained:
                help_button = HelpButton("border cell", BORDER_HELP, self)
                explained = True
            tile = Tile(cell, self, help_button=help_button)
            tile.clicked.connect(self._on_tile_clicked)
            tile.delete_requested.connect(self.delete_cell)
            self.frame.place(tile, index)
            self.tiles[cell.uid] = tile
        self.frame.pack()

        self.file_label.setText(file_position(self.file_index, self.file_total))
        self.screen_label.setText(screen_position(self.screen_index))
        self.count_label.setText(cells_on_screen(len(self.visible_cells())))
        self._show_file_done(self.remaining_in_file() == 0)

    def _show_file_done(self, done: bool) -> None:
        """The done state, **driven from the file's count and never from the band index.**

        The reviewer reaches it on the last band in the ordinary case, but *"you are on screen 3"*
        and *"there is nothing left"* are not the same fact and they can come apart: an undo
        restores cells to an earlier band while the reviewer is standing on a later one. Gating on
        `remaining_in_file() == 0` means the line is true whenever it is showing.

        **The two accepts are disabled rather than hidden.** They would be no-ops — there is
        nothing unresolved for either to clear — and `Accept Rest Of File` would otherwise put up
        a popup offering to accept zero cells. Greying them keeps the bar the shape the reviewer
        learned instead of rearranging it under them.
        """
        label = proceed_label(self.file_index, self.file_total)
        self.done_line.setVisible(done)
        self.proceed_button.setText(label)
        #: The popup heads itself with the control's name, so the name has to follow the label —
        #: `HelpButton` reads `title` when it opens, not when it is built.
        self.proceed_help.title = label
        self.proceed_button.setVisible(done)
        self.proceed_help.setVisible(done)
        self.accept_screen_button.setEnabled(not done)
        self.accept_file_button.setEnabled(not done)

    def _ask_rejection(self):
        """The two popups, in rich review's order. **Replaced in tests**, as every modal is."""
        return ask_rejection(self)

    def delete_cell(self, uid: str):
        """The `×` on a tile: **reject this cell without opening it.**

        > ### ⇒ THE SAME TWO QUESTIONS, IN THE SAME ORDER — patch item B5, 2026-09-03
        >
        > `reject_question()` first, and `count_question()` after it when the answer is
        > `real_cell`. Cancel on the first ends it; Back on the second returns to the first.
        > **The loop is `widgets.ask_rejection`, which rich review now calls too** — one
        > implementation, because two copies of a sequence that has already been corrected
        > twice would drift two ways.
        >
        > **It is a `TriageAction`, unlike `mark_resolved`**, and that is the point of the
        > undo work above: `u` takes it back, the tile returns, and `ReviewStep` drops the
        > decision and puts the material back in the scene. Nothing is destroyed during a
        > session — the arrays go at item 8 — so the undo is complete rather than partial.
        >
        > **The tile leaves the grid immediately**, through `mark_resolved`, exactly as a rich
        > review resolution does.
        """
        resolution = self.ask_rejection()
        if resolution is None:
            return None
        #: **Recorded before the tile goes**, so a listener that reads the grid sees a
        #: consistent state — `ReviewStep._leave` records in the same order for the same reason.
        self.cell_rejected.emit(uid, resolution)
        self.last_action = TriageAction("delete", (uid,), self.screen_index)
        self.mark_resolved(uid)
        return resolution

    def mark_resolved(self, uid: str) -> bool:
        """One cell resolved from outside — **rich review's half of the gap rule.**

        > ### THE TILE NEVER LEFT THE GRID, AND NOTHING WAS MARKING IT — found by walking the
        > screen, 2026-08-26
        >
        > `visible_cells` filters on `resolved` and `_rebuild` places by index so a skipped
        > cell leaves its hole, but **`resolved` was written in exactly two places, both in
        > this file**: the two accepts. Rich review recorded its decision on `ReviewStep` and
        > switched the stack back, and no one ever touched the `ReviewCell`. So the gap
        > machinery was complete and unreachable, and **no outcome removed a tile** — not
        > accept, not either rejection.
        >
        > It is a method here rather than a flag set from `app.py` because the rebuild has to
        > follow the flag: setting `cell.resolved` alone leaves the grid showing a tile for a
        > cell that is no longer in `visible_cells`, which is the same defect wearing a
        > different cause.

        **It is not a `TriageAction` and it deliberately does not touch `last_action`.** §3b:
        triage's undo *"never reaches back into a finalised rich-review episode"* — that episode
        had its own undo and it expired at finalisation. Pressing `u` after resolving a cell
        undoes the last *triage* action, or nothing.

        **Resolving the band's last cell advances the band** — `_advance_past_cleared_bands`.
        Without it, deciding the final cell of a screen in rich review returned the reviewer to an
        empty grid reading `0 cells on this screen`, whose only way forward was `Accept This
        Screen` — a button that cleared nothing and existed only to move them on. **An accept that
        accepts nothing is a bad thing to teach a reviewer to press**, twenty thousand decisions
        into a study.

        Only when the resolved cell was on the band **being looked at**. A uid from another band —
        which the pool-backed route can hand over — must not move the grid under the reviewer.

        Returns whether a cell of that uid was found, so a caller can tell a no-op from a hit.
        """
        for index, band in enumerate(self.screens):
            for cell in band:
                if cell.uid == uid:
                    cell.resolved = True
                    if index == self.screen_index:
                        self._advance_past_cleared_bands()
                    self._rebuild()
                    return True
        return False

    def _advance_past_cleared_bands(self) -> None:
        """Move to the next band with something on it, **or stay put if the file is done.**

        The done check comes first and it is the whole of the last-band answer: when nothing is
        unresolved anywhere the reviewer stays on the band they were working on and `_rebuild`
        shows `FILE_DONE_MESSAGE` with the proceed button. **Advancing to a band that is empty for
        the same reason every other band is empty would just move the empty grid.**

        > **A loop rather than a single step, and the distinction is unobservable in normal use.**
        > Reaching this at all means the current band emptied under rich review, and the bands
        > after it are then still as `split_into_screens` left them — where an empty band is only
        > ever followed by empty bands. The loop costs one word and holds anyway if a later band
        > is emptied out of order, which undo plus a re-accept can do.
        """
        if self.remaining_in_file() == 0:
            return
        while not self.visible_cells() and self.screen_index < len(self.screens) - 1:
            self.screen_index += 1

    def grid_position(self, uid: str) -> tuple[int, int]:
        """Row and column of a tile, so the gap rule is assertable."""
        index = self.frame.grid.indexOf(self.tiles[uid])
        row, column, _, _ = self.frame.grid.getItemPosition(index)
        return row, column

    # -- actions --------------------------------------------------------

    def _on_tile_clicked(self, uid: str) -> None:
        """**A single cell is resolved only by entering rich review** — there is no per-cell
        accept on this screen, because a thumbnail is not enough to accept one on.
        """
        self.cell_entered.emit(uid)

    def accept_this_screen(self) -> None:
        """Clears everything still showing and advances. **A pure status flip.**"""
        cleared = tuple(cell.uid for cell in self.visible_cells())
        for cell in self.screens[self.screen_index]:
            if not cell.resolved:
                cell.resolved = True
        self.last_action = TriageAction("screen", cleared, self.screen_index)
        if self.screen_index < len(self.screens) - 1:
            self.screen_index += 1
        self._rebuild()
        self.screen_accepted.emit()

    def remaining_in_file(self) -> int:
        return sum(1 for band in self.screens for cell in band if not cell.resolved)

    def unseen_in_file(self) -> int:
        """Cells on screens the reviewer has not reached — what the gate has to name."""
        return sum(1 for band in self.screens[self.screen_index + 1:]
                   for cell in band if not cell.resolved)

    def proceed(self) -> None:
        """Leave a finished file. **`file_accepted` with nothing accepted and no `TriageAction`.**

        It reuses that signal because the fact it carries — *this file is done* — is the same one,
        and both routes are already listening for it. It records no undoable action because it
        clears nothing: everything it moves past was decided by an accept or in the editor, and
        each of those was its own action at the time.
        """
        self.file_accepted.emit()

    def accept_rest_of_file(self) -> None:
        remaining, unseen = self.remaining_in_file(), self.unseen_in_file()
        if not self.confirm_accept_rest(accept_rest_warning(remaining, unseen), remaining):
            return
        cleared = tuple(cell.uid for band in self.screens for cell in band if not cell.resolved)
        for band in self.screens:
            for cell in band:
                cell.resolved = True
        self.last_action = TriageAction("file", cleared, self.screen_index)
        self._rebuild()
        self.file_accepted.emit()

    def _emit_undone(self, restored) -> None:
        """Tell the flow driver which cells came back. **The undo was cosmetic without it.**"""
        self.undone.emit(tuple(restored))

    def undo(self) -> TriageAction | None:
        """**The last triage action only, and it expires on the next one.**

        Undoing a screen's accept restores only what that accept itself cleared — not cells
        resolved individually beforehand, which were separate earlier actions and stay resolved.
        **Triage's undo never reaches back into a finalised rich-review episode**: that episode
        had its own undo and it expired deliberately at finalisation.

        > ### IT GOES BACK TO THE BAND THE ACTION HAPPENED ON, NOT ONE STEP BACK FROM HERE
        >
        > This decremented `screen_index`, which was exactly right while an accept was the only
        > thing that could move the band: the action had just been taken here, so *here minus one*
        > was where it came from. **`mark_resolved` can now advance the band too**, and it
        > deliberately records no action — so accepting band 1, then emptying band 2 through rich
        > review, put the reviewer on band 3 with band 1's accept still the last action. `u` then
        > restored band 1's cells and landed them on **band 2**, looking at a screen that had
        > nothing to do with what had just been restored.
        >
        > `TriageAction` carries the band, so undo goes to it. For a file accept it is where the
        > reviewer already is, since that action never moved them — the two kinds only looked
        > different under the decrement.
        >
        > ### ⇒ AND IT DID NOT REMOVE THE DECISIONS — patch item B5, 2026-09-03
        >
        > This set `resolved = False` and rebuilt the grid, and **nothing removed the
        > decisions.** `ReviewStep.decisions` is written by `_record_resolved` from these very
        > flags; `undo` emitted no signal and `ReviewStep` had no connection to it. So: accept a
        > screen, twenty cells are recorded `confirmed`, press `u`, the twenty tiles come back —
        > **and all twenty are still recorded as confirmed**, and item 8 writes them that way.
        >
        > **The undo was cosmetic**, on the one gesture whose entire purpose is taking a
        > decision back. Fixed once, for both this and the delete button built on top of it:
        > `undo` now reports what it restored and `ReviewStep` drops those decisions and puts
        > their material back in the scene.
        """
        action = self.last_action
        if action is None:
            return None
        restore = set(action.cleared)
        restored = []
        for band in self.screens:
            for cell in band:
                if cell.uid in restore:
                    cell.resolved = False
                    restored.append(cell.uid)
        self.screen_index = action.screen
        self.last_action = None
        self._rebuild()
        self._emit_undone(restored)
        return action

    def keyPressEvent(self, event):     # noqa: N802 - Qt's name
        if event.key() == Qt.Key.Key_U:
            self.undo()
        else:
            super().keyPressEvent(event)
