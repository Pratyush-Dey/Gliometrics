"""The five frames. `COMPONENT_03_PLAN.md` §6f.9, §6j.7, `COMPONENT_05_PLAN.md` §3d.9.

    A   controls left, two image panels                 item 3 alone
    B   Colours/Layers left, two panels, controls right item 5 alone
    C   Colours/Layers left, ONE view, controls right   items 2, 4, 9a, 9b, rich review
    D   list left, two stacked panels, controls right   item 6
    E   header block, full-width scrolling grid, bar    triage

**Columns are 260 / stretch / 280 everywhere**, and that is not cosmetic: the controls need a
constant width to stay legible, and **the image is the only thing that benefits from more room**,
so every extra pixel a larger window gives goes to the centre.

**Panels are flat `#262930` with no tints and no dividers.** The canvas is `black`, not the chrome
grey, so it separates itself — and `#2e323a` stays reserved for popups, which is what keeps *one
step lighter* meaning *this is a modal*.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget,
)

from . import theme
from .widgets import screen_title

__all__ = ["LEFT_PX", "RIGHT_PX", "CENTRE_MIN_PX", "GRID_COLUMNS", "GRID_ROWS", "CanvasHost",
           "scrolling_body", "FrameA", "FrameB", "FrameC", "FrameD", "FrameE"]

LEFT_PX = 260
RIGHT_PX = 280
CENTRE_MIN_PX = 600

#: Triage's density — **five tiles across, scrolling vertically.** `COMPONENT_05_PLAN.md` §3c.
GRID_COLUMNS = 5

#: How many rows of tiles the grid must always be able to show. **The floor that makes item 7a's
#: whole-screen scroll work** — the grid's own area would otherwise collapse to nothing and the
#: outer bar would never engage. See `FrameE` for the mechanism. Five across and four down is
#: twenty tiles, the user's number on 2026-09-06.
GRID_ROWS = 4


def scrolling_body(owner: QWidget) -> tuple[QScrollArea, QWidget]:
    """A screen-wide vertical scroll area, and the widget the screen builds itself inside.

    > ### ⇒ EVERY SCREEN SCROLLS — patch item D, 2026-09-06
    >
    > **The window was forced taller than the display, and `fit_to_display` could not stop it.**
    > That function caps the window with `setMaximumSize`, but a layout minimum beats a maximum:
    > simulated against a 1280x800 desktop, the window asks for 1240x720 and **gets 1240x916**.
    > 116 px of window below the bottom of the screen, unreachable — including whatever sits at
    > the bottom of a control column.
    >
    > Measured minimum heights, headless, before this: **item 7b 916**, item 4 616, item 2 613,
    > item 5 526, the DAPI filter 502. Item 7b exceeds the app's own preferred 900 on **any**
    > display. And on 7b the tall column is the **left** one (870 px against the right's 308),
    > which is why this wraps the whole screen rather than the control column.
    >
    > **This is item 8's own fix, generalised** — patch item B4 of 2026-09-03 wrapped the save
    > screen and recorded the rest as a known hole in as many words: *"`FrameC.right` is also a
    > plain `QVBoxLayout` with no scroll, so the control columns on items 2, 4, 5, 9a and 9b can
    > hit the same wall."* Wrapping collapsed item 8's minimum from ~700 px to **68**, which is
    > what lets the window be any size the display allows.
    >
    > **The bar is always shown**, which is item 7a's rule rather than item 8's: wheel and pinch
    > are reserved for zoom on every screen with a canvas, so the 12 px bar has to be there to be
    > grabbed. The user chose this for all thirteen screens on 2026-09-06, and item 8 was moved
    > onto it for the same reason.
    >
    > **Item 7a is wrapped too, since 2026-09-06.** It was excluded on the day this was built,
    > on the grounds that `FrameE` already scrolled its tile grid and a second area would nest
    > two bars. The user reversed that: the grid's bar scrolls *the grid*, and what was missing
    > was any way to reach the bottom of the **screen** — the button bar included — on a window
    > taller than the display. That is the same defect this function exists to fix, and the
    > exclusion left one screen of thirteen still carrying it. See `FrameE`: nesting the two
    > needed a floor under the grid, which is the part that was not obvious.

    Returns the scroll area to put in the screen's outer layout, and the body widget the screen
    lays itself out in. The horizontal bar stays off — the house pattern, three times over.
    """
    body = QWidget(owner)
    scroll = QScrollArea(owner)
    scroll.setWidget(body)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    return scroll, body


def _scrolling_outer(screen: QWidget) -> tuple[QScrollArea, QVBoxLayout]:
    """`scrolling_body` plus the screen's own zero-margin outer layout. Used by the frames."""
    outer = QVBoxLayout(screen)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    scroll, body = scrolling_body(screen)
    outer.addWidget(scroll)
    return scroll, QVBoxLayout(body)


class CanvasHost(QWidget):
    """The centre panel of a viewer screen: **black, and empty until the canvas is moved in.**

    > ### THERE IS ONE CANVAS AND IT TRAVELS — task 23
    >
    > Seven screens show a picture and **all seven show the same one.** A napari viewer per
    > screen would be seven OpenGL contexts and seven copies of whatever is on them, on a
    > machine already holding a 3.17 GiB image; and the flow's own arrangement makes it
    > unnecessary, because exactly one screen is visible at a time. So the flow driver moves the
    > single canvas into whichever host is showing, and every other host stands empty.
    >
    > **`holds_canvas` is bookkeeping, not the widget.** It says *this host is where the canvas
    > belongs now*, which is true under `RecordingViewer` — where there is no widget at all —
    > as much as under the real one. That is what lets the headless suite hold the travelling
    > without a display, and it is deliberately the weaker of the two facts: whether an image
    > actually appeared is a thing only a person in front of a screen can say.

    **The black is scoped to this widget by object name.** An unscoped `background:` here would
    be inherited by every child, which after mounting means the z slider napari packs under its
    canvas — a black slider on black, which is what a scoped rule avoids.
    """

    #: §6f.9. The canvas is `black`, not the chrome grey, so it separates itself.
    COLOUR = theme.CANVAS

    def __init__(self, parent: QWidget | None = None, *, minimum_height: int = 0):
        super().__init__(parent)
        self.setObjectName("canvasHost")
        self.setStyleSheet(f"#canvasHost {{ background: {self.COLOUR}; }}")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        if minimum_height:
            self.setMinimumHeight(minimum_height)
        self.mounted: QWidget | None = None
        self.holds_canvas = False

    def take(self, widget: QWidget | None) -> None:
        """This host now holds the session's one canvas.

        ``None`` records the claim and mounts nothing, which is the headless case: a
        `RecordingViewer` has no widget to give and every layer decision is still assertable.
        """
        self.holds_canvas = True
        if widget is None:
            return
        #: > ### `mounted` GOES STALE THE MOMENT ANOTHER HOST TAKES THE CANVAS — task 26
        #: >
        #: > `give_up` releases the claim and **deliberately leaves the widget parented here**,
        #: > so `mounted` is true at that moment. It stops being true as soon as the next host
        #: > calls `addWidget`, which re-parents — and nothing tells this host that happened.
        #: >
        #: > So a second visit found `widget is self.mounted`, returned early, and **never
        #: > re-parented the canvas back.** The host reported `holds_canvas=True` while the
        #: > widget sat in a hidden screen: item 3 came back from item 4 with an empty centre,
        #: > its parameters and readout intact because only the canvas was missing.
        #: >
        #: > **Item 4's Back is the only path in the flow that visits a host twice**, which is
        #: > why one screen showed a defect that belongs to every host.
        #: >
        #: > The cached value cannot be trusted, so the widget's own parent is asked instead.
        if widget is self.mounted and widget.parentWidget() is self:
            return
        self._layout.addWidget(widget)     # addWidget re-parents
        widget.show()
        self.mounted = widget

    def give_up(self) -> QWidget | None:
        """Release the claim and hand the widget back, **without re-parenting it to `None`**.

        A `setParent(None)` here would turn the canvas into a top-level window, which is the
        very stray napari window this whole arrangement exists to remove. The caller decides
        where it goes next; until it does, the widget stays in this host's layout and is hidden
        along with the screen.
        """
        self.holds_canvas = False
        return self.mounted


class FrameC(QWidget):
    """Colours/Layers left, one overlaid view centre, controls right.

    Items 2, 4, 9a and 9b, plus Component 5's rich review. The most common frame in the project,
    which is why it is built once here.

    ``header`` is optional: items 2 and 4 carry a readout strip above the canvas, **9a and 9b
    carry none** — one panel means nothing to name, and 9a is purely visual with nothing to
    report.
    """

    def __init__(self, title: str, *, parent: QWidget | None = None):
        super().__init__(parent)
        #: **The whole screen scrolls** — patch item D. See `scrolling_body` for the measured
        #: minimum heights this fixes and for why the bar is always shown. `column` is what the
        #: screen builds itself in; the outer layout holds the scroll area and nothing else.
        #: **B and D inherit this**, so items 2, 4, 5, 6, 9a, 9b and 7b are all covered here.
        self.scroll, column = _scrolling_outer(self)
        outer = column
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)
        outer.addWidget(screen_title(title))

        body = QHBoxLayout()
        body.setSpacing(12)

        self.left = QVBoxLayout()
        self.left.setSpacing(10)
        left_host = QWidget(self)
        left_host.setLayout(self.left)
        left_host.setFixedWidth(LEFT_PX)

        self.centre = QVBoxLayout()
        self.centre.setSpacing(4)
        centre_host = QWidget(self)
        centre_host.setLayout(self.centre)
        centre_host.setMinimumWidth(CENTRE_MIN_PX)

        self.right = QVBoxLayout()
        self.right.setSpacing(10)
        right_host = QWidget(self)
        right_host.setLayout(self.right)
        right_host.setFixedWidth(RIGHT_PX)

        body.addWidget(left_host, 0)
        body.addWidget(centre_host, 1)      # only the centre stretches
        body.addWidget(right_host, 0)
        outer.addLayout(body)

        self.left_host, self.centre_host, self.right_host = left_host, centre_host, right_host

    def add_left(self, widget: QWidget) -> None:
        self.left.addWidget(widget)

    def add_right(self, widget: QWidget) -> None:
        self.right.addWidget(widget)

    def add_centre(self, widget: QWidget, stretch: int = 0) -> None:
        self.centre.addWidget(widget, stretch)

    def section(self, text: str, help_button: QWidget | None = None) -> None:
        """A `Colours` or `Layers` heading in the left column, with its optional "?"."""
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        row.addStretch(1)
        if help_button is not None:
            row.addWidget(help_button)
        host = QWidget(self)
        host.setLayout(row)
        self.left.addWidget(host)


class FrameB(FrameC):
    """Colours/Layers left, **two panels**, controls right, and a table under the right panel.

    **Item 5 alone.** Structurally it is `FrameC` with one addition, and that is worth stating
    rather than hiding: **the two-panel-ness is not a Qt fact.** Both panels live inside one
    napari canvas at `grid.shape=(1, 2)`, so the Qt layout of B and C is identical and the only
    thing B adds is §6i.6's accumulating table.

    **The table spans the RIGHT panel's width**, not the canvas's. Every row is a commit at a
    `mask_coefficient`, which is the right panel's parameter — putting it under the left panel
    would attach the history to the mask that is not moving.
    """

    def __init__(self, title: str, *, parent: QWidget | None = None):
        super().__init__(title, parent=parent)
        self.table_strip = QHBoxLayout()
        self.table_strip.setContentsMargins(0, 4, 0, 0)
        #: The left half stays empty so the table lines up under the right panel — the canvas
        #: above splits 50/50 and this keeps that division running through the strip.
        self.table_strip.addStretch(1)
        self.centre.addLayout(self.table_strip)

    def add_under_right_panel(self, widget: QWidget) -> None:
        self.table_strip.addWidget(widget, 1)


class FrameD(FrameC):
    """List left, **two stacked panels** centre, controls right. **Item 6 alone.**

    §6f.9 named three frames and left items 6 and 8 outside all of them; this is item 6's.

    **The right column is nearly empty here — one number box and Continue — and that is correct
    rather than unbalanced: the LIST is the control on this screen, and it is on the left.**

    Like `FrameB`, the stacking is a napari fact rather than a Qt one — both panels are viewboxes
    in one canvas at `grid.shape=(2, 1)`. What Qt contributes is the second header strip, because
    each panel needs its own name and only the upper one carries a readout.
    """

    def __init__(self, title: str, *, parent: QWidget | None = None):
        super().__init__(title, parent=parent)

    def add_panel_header(self, widget: QWidget) -> None:
        """A header strip. Called once per stacked panel, top to bottom."""
        self.centre.addWidget(widget)


class FrameE(QWidget):
    """Header block, **full-width scrolling grid**, button bar. Triage alone.

    **The only frame with no three-column layout**, which is why it does not subclass `FrameC` the
    way B and D do. There is nothing beside the grid: the header block sits above it and the two
    buttons below, and the grid takes the whole width because five tiles across is the density and
    a side column would cost one of them.

    **The gutters are `#262930` and no borders are drawn.** The tiles are black, so the chrome
    shows between them and *is* the separator — §6g.2's logic inverted: there the two panels were
    both black and the chrome had to supply the boundary from outside; here the tiles are black
    against chrome, so the boundary is already there.

    **Wheel and pinch are reserved for zoom, so scrolling is a grabbable 12 px scrollbar** — the
    same laptop-hardware reason that put Component 3's z on the arrow keys.

    > ### ⇒ TWO SCROLL AREAS, AND THE INNER ONE NEEDS A FLOOR OR THE OUTER ONE IS DEAD
    >
    > `self.scroll` is the whole screen — the same `scrolling_body` every other frame uses, so
    > `self.scroll` means the same thing on all thirteen. `self.grid_scroll` is the tile grid's
    > own, which it has always had. The user asked for both on 2026-09-06, reversing the
    > *"deliberately excluded"* call taken earlier the same day.
    >
    > **Wrapping alone would have done nothing, and this is the whole subtlety.** A scroll area
    > is willing to shrink to almost nothing — that is what makes it one. With the grid's area
    > free to collapse, the body's minimum height stays tiny however many tiles there are, the
    > outer viewport is never the smaller of the two, and the outer bar sits there permanently
    > without ever having anything to scroll. The grid area is given a floor of `GRID_ROWS`
    > rows, so the screen has a real minimum height and the outer bar engages exactly when the
    > window drops below it.
    >
    > **A floor, not a fixed height.** The grid keeps its stretch, so on a tall display it still
    > grows past four rows and shows as many tiles as there is room for. Fixing the height
    > instead would cap a 4K monitor at twenty tiles and leave the rest of the screen empty.
    >
    > The wheel is left alone. Nothing in this project ever bound it, so Qt's own rule applies:
    > over a tile it scrolls the grid, over the title, header or bar it scrolls the screen. The
    > 12 px bars are the mechanism either way, and both are always shown so both can be grabbed.
    """

    def __init__(self, title: str, *, row_px: int | None = None,
                 parent: QWidget | None = None):
        """``row_px`` is the height of one tile row, which **only the screen knows.**

        `screen_triage` owns `TILE_PX` and `MARK_PX` and imports *from* this module, so this
        module cannot import them back. Passing the number in keeps one source of truth rather
        than a copy here that would drift the first time a tile changed size. Left out, the grid
        gets no floor and the outer bar cannot engage — which is the pre-2026-09-06 behaviour.
        """
        super().__init__(parent)
        #: **The whole screen scrolls** — patch item D, extended to item 7a on 2026-09-06. See
        #: the class docstring for why the grid below needs a floor for this to do anything.
        self.scroll, column = _scrolling_outer(self)
        outer = column
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)
        outer.addWidget(screen_title(title))

        self.header = QVBoxLayout()
        self.header.setSpacing(2)
        header_host = QWidget(self)
        header_host.setLayout(self.header)
        outer.addWidget(header_host)
        self.header_host = header_host

        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        #: **A phantom sixth column takes the horizontal slack**, so the five real ones stay at
        #: tile width instead of sharing out the whole scroll area. `place` only ever writes
        #: columns 0-4, so nothing is ever put in it. The vertical half of the same job is
        #: `pack`, which cannot be done here because the last row is not known until the tiles
        #: are placed.
        self.grid.setColumnStretch(GRID_COLUMNS, 1)
        grid_host = QWidget(self)
        grid_host.setLayout(self.grid)

        self.grid_scroll = QScrollArea(self)
        self.grid_scroll.setWidget(grid_host)
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.grid_scroll.setFrameShape(QFrame.Shape.NoFrame)
        if row_px is not None:
            #: `GRID_ROWS` rows of tiles, the gaps between them, and the grid's own margins —
            #: read off the layout rather than assumed, so the floor is exactly the height at
            #: which the last of those rows is fully visible. The frame is `NoFrame` and the
            #: horizontal bar is off, so nothing else contributes.
            _, top, _, bottom = self.grid.getContentsMargins()
            self.grid_scroll.setMinimumHeight(
                GRID_ROWS * int(row_px) + (GRID_ROWS - 1) * self.grid.spacing()
                + top + bottom)
        #: **Stretch is kept**, so the floor is a floor and not a ceiling: on a tall display the
        #: grid still takes the leftover height and shows more than `GRID_ROWS` rows, and the
        #: button bar still sits at the bottom of the screen exactly as it did.
        outer.addWidget(self.grid_scroll, 1)
        self.grid_host = grid_host

        self.bar = QHBoxLayout()
        self.bar.setSpacing(24)
        outer.addLayout(self.bar)

    def add_header_line(self, widget: QWidget) -> None:
        self.header.addWidget(widget)

    def place(self, widget: QWidget, index: int) -> None:
        """**Position is by INDEX, not by packing order** — which is what leaves a gap.

        A resolved cell is simply not placed, and the grid does not reflow. *"Position is how a
        reviewer keeps their place, and reflowing moves every tile they have not looked at yet."*
        """
        self.grid.addWidget(widget, index // GRID_COLUMNS, index % GRID_COLUMNS)

    def pack(self) -> None:
        """Take the vertical slack below the last row, so the tiles sit at the top.

        **Called after the last `place`, because the row it goes in is the row after the last
        one used** — and a stretch factor set in `__init__` would land on row 0 and stretch the
        first row of tiles instead. A spacer rather than `setRowStretch` so it clears itself:
        the caller empties the grid with `takeAt` before every rebuild, and a stretch factor
        would survive that while an item does not.
        """
        self.grid.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum,
                                      QSizePolicy.Policy.Expanding),
                          self.grid.rowCount(), 0)

    def add_to_bar(self, widget: QWidget) -> None:
        self.bar.addWidget(widget)


class FrameA(QWidget):
    """Controls left, two image panels, Continue in a **bottom strip**.

    **Item 3 alone**, and it is the one screen with no `Colours` block, no `Layers` block, no
    z-batch box and no projection mode. That is *why* the controls take the left column: §4's note
    that they sit left here and right at item 2 is a consequence of the left column being free,
    not a stylistic alternation.

    The bottom strip exists because *bottom-right of the window* has to mean something on a frame
    with no right column — §4's 2026-08-17 correction, preserved rather than reversed.
    """

    def __init__(self, title: str, *, parent: QWidget | None = None):
        super().__init__(parent)
        #: **The whole screen scrolls**, bottom strip included — patch item D. See
        #: `scrolling_body`. Item 3 is the shortest viewer screen at 405 px, so this is
        #: future-proofing rather than a fix here; the rule is the same on every screen.
        self.scroll, column = _scrolling_outer(self)
        outer = column
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)
        outer.addWidget(screen_title(title))

        body = QHBoxLayout()
        body.setSpacing(12)

        self.left = QVBoxLayout()
        self.left.setSpacing(10)
        left_host = QWidget(self)
        left_host.setLayout(self.left)
        left_host.setFixedWidth(LEFT_PX)

        self.centre = QVBoxLayout()
        self.centre.setSpacing(4)
        centre_host = QWidget(self)
        centre_host.setLayout(self.centre)
        centre_host.setMinimumWidth(CENTRE_MIN_PX)

        body.addWidget(left_host, 0)
        body.addWidget(centre_host, 1)
        outer.addLayout(body)

        self.strip = QHBoxLayout()
        self.strip.addStretch(1)
        outer.addLayout(self.strip)

        self.left_host, self.centre_host = left_host, centre_host

    def add_left(self, widget: QWidget) -> None:
        self.left.addWidget(widget)

    def add_centre(self, widget: QWidget, stretch: int = 0) -> None:
        self.centre.addWidget(widget, stretch)

    def add_to_strip(self, widget: QWidget) -> None:
        """Bottom-right of the window, which is where Continue lives on every screen."""
        self.strip.addWidget(widget, 0, Qt.AlignmentFlag.AlignRight)
