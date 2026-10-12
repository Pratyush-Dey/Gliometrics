"""The blocks every screen reuses. `COMPONENT_03_PLAN.md` §6f.5, §6g.5, §6i.6, §6i.7, §6j.6.

**Built once because seven screens carry a colour picker and six carry a parameter slider.**
Writing the swatch grid on each screen would let it drift seven ways, and the whole value of
§6f.2's ten colours is that they are the same ten everywhere.

> **It was six and six until `screen_background.py` arrived on 2026-08-23**, which carries a picker
> but no slider. Same insertion that took the flow from ten screens to eleven, and the same lesson:
> **a count written into prose is a fact with nothing checking it.** The ten colours in the sentence
> above are safe for the opposite reason — `SWATCH_COLUMNS * SWATCH_ROWS` is derived, and
> `DATA_COLOURS` is the thing being counted.

Nothing in this module chooses a colour. Every value comes from `theme` (chrome) or `palette`
(data), so a screen that wants a new colour has to change one of those and answer for it there.

**Qt is imported at module level here**, unlike `theme` and `palette` — these *are* widgets, and
there is nothing to test about them without a display. `tests/test_ui_widgets.py` runs them under
`QT_QPA_PLATFORM=offscreen`, which works in this environment.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from . import palette, theme

__all__ = [
    "Swatch", "SwatchGrid", "ColourRow", "LayerToggles", "ParameterSlider",
    "HelpButton", "HelpPopup", "ConfirmPopup", "ChoicePopup", "CountPopup",
    "CommitButton", "HeaderStrip",
    "AccumulatingTable", "NavigationRow",
    "SelectionList", "screen_title",
    "reject_reason_popup", "cell_count_popup", "ask_rejection",
]


def screen_title(text: str) -> QLabel:
    """The one bold thing on any screen. 16 pt, Title Case, `#e2e4e6`."""
    label = QLabel(theme.title_case(text))
    label.setProperty("role", "title")
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    return label


# --------------------------------------------------------------------------
# The colour picker — §6f.5
# --------------------------------------------------------------------------

class Swatch(QWidget):
    """One 24 × 24 cell with a 16 × 16 painted square.

    **24 px because below that a swatch is fiddly on the laptop trackpad this tool is for**, and
    the square is painted at full value because *which colour this is* is the content here, not
    a state signal.

    Selection is a **2 px `#e2e4e6` ring with a 1 px `#262930` gap** between swatch and ring, and
    the gap is what makes the white swatch work — white square, dark gap, light ring still reads
    as a ring. Hover is 1 px `#9aa0a6`, same position, separated from selection by *weight*
    rather than by colour, so §6e.4 holds here too: the mark carries the state, not the hue.
    """

    clicked = Signal(str)

    def __init__(self, colour: palette.DataColour, parent: QWidget | None = None):
        super().__init__(parent)
        self.colour = colour
        self._selected = False
        self._hover = False
        self.setFixedSize(palette.SWATCH_CELL_PX, palette.SWATCH_CELL_PX)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(colour.name)

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self.update()

    def is_selected(self) -> bool:
        return self._selected

    def enterEvent(self, event):        # noqa: N802 - Qt's name
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):        # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):   # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.colour.name)
        super().mousePressEvent(event)

    def paintEvent(self, event):        # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        inset = (palette.SWATCH_CELL_PX - palette.SWATCH_PX) // 2
        painter.fillRect(inset, inset, palette.SWATCH_PX, palette.SWATCH_PX,
                         QColor(self.colour.hex))

        if self._selected:
            pen = QPen(QColor(theme.TEXT), 2)
        elif self._hover:
            pen = QPen(QColor(theme.MUTED), 1)
        else:
            painter.end()
            return
        painter.setPen(pen)
        half = pen.width() / 2
        painter.drawRect(self.rect().adjusted(int(half), int(half),
                                              -int(half) - 1, -int(half) - 1))
        painter.end()


class SwatchGrid(QWidget):
    """The ten, 5 × 2, in hue-wheel order red → white.

    **The order is load-bearing rather than tidy**: a user hunting *"something bluer"* moves
    rightward rather than searching, which is what makes the grid scannable instead of readable.
    """

    colour_changed = Signal(str)

    def __init__(self, selected: str = "white", parent: QWidget | None = None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self._swatches: dict[str, Swatch] = {}
        for index, colour in enumerate(palette.DATA_COLOURS):
            swatch = Swatch(colour, self)
            swatch.clicked.connect(self._on_clicked)
            grid.addWidget(swatch, index // palette.SWATCH_COLUMNS,
                           index % palette.SWATCH_COLUMNS)
            self._swatches[colour.name] = swatch

        self.setFixedSize(palette.SWATCH_COLUMNS * palette.SWATCH_CELL_PX,
                          palette.SWATCH_ROWS * palette.SWATCH_CELL_PX)
        self._selected = ""
        self.set_colour(selected)

    def _on_clicked(self, name: str) -> None:
        self.set_colour(name)
        self.colour_changed.emit(name)

    def set_colour(self, name: str) -> None:
        if name not in self._swatches:
            raise KeyError(f"{name!r} is not one of the ten: {sorted(self._swatches)}")
        self._selected = name
        for colour_name, swatch in self._swatches.items():
            swatch.set_selected(colour_name == name)

    def colour(self) -> palette.DataColour:
        return palette.COLOUR_BY_NAME[self._selected]

    def swatch(self, name: str) -> Swatch:
        return self._swatches[name]


class ColourRow(QWidget):
    """Label → swatches → translucency slider. One per layer.

    **No `translucency` caption under the slider.** §4's drawing carried one; items 2, 4, 5, 9a
    and 9b do not, and a slider directly beneath a swatch grid does not need one.

    Under additive blending the slider is a **weight** — how much of the sum this layer is
    entitled to — which is why the default comes from `palette.CHANNEL_ALPHA` rather than being
    a comfort setting the screen picks.
    """

    colour_changed = Signal(str)
    alpha_changed = Signal(float)

    def __init__(self, label: str, colour: str | None, alpha: float = 1.0,
                 parent: QWidget | None = None):
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        self.label = QLabel(label)
        #: **A `None` colour is a slider with no swatches — 2026-08-25.** A layer coloured by a
        #: per-region *scheme* has no single hue to pick, but it still has a translucency, and
        #: §4's rule excluded it from both because the two had never been separable. Item 7b's
        #: `this cell` is the case that forced the split: the user asked for its slider back,
        #: and its colouring is graph colouring, which is functional rather than chosen.
        self.grid = None if colour is None else SwatchGrid(colour, self)
        if self.grid is not None:
            self.grid.colour_changed.connect(self.colour_changed)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(round(alpha * 100)))
        self.slider.setFixedWidth(palette.SWATCH_COLUMNS * palette.SWATCH_CELL_PX)
        self.slider.valueChanged.connect(lambda v: self.alpha_changed.emit(v / 100.0))

        column.addWidget(self.label)
        if self.grid is not None:
            column.addWidget(self.grid)
        column.addWidget(self.slider)

    def alpha(self) -> float:
        return self.slider.value() / 100.0

    def colour(self) -> palette.DataColour:
        return self.grid.colour()


# --------------------------------------------------------------------------
# Layers, and the parameter slider
# --------------------------------------------------------------------------

class LayerToggles(QWidget):
    """The `Layers` block — one checkbox per layer, in the reading order of the picker rows.

    Stacking order is not offered and does not exist as a concept for the user: **addition
    commutes**, so *which layer is on top* changes nothing about the picture. The layer list is
    a reading order and nothing more.
    """

    toggled = Signal(str, bool)

    def __init__(self, layers: Sequence[tuple[str, bool]], parent: QWidget | None = None):
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(3)

        self._boxes: dict[str, QCheckBox] = {}
        for name, on in layers:
            box = QCheckBox(name, self)
            box.setChecked(on)
            box.toggled.connect(lambda checked, n=name: self.toggled.emit(n, checked))
            column.addWidget(box)
            self._boxes[name] = box

    def is_on(self, name: str) -> bool:
        return self._boxes[name].isChecked()

    def set_on(self, name: str, on: bool) -> None:
        self._boxes[name].setChecked(on)

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Grey a layer out — **for a layer that does not exist yet**, not one merely hidden.

        The background filter has two layers that only mean something after its COMMIT: before
        it there is no *kept* and no *removed*, only the channel. A checkbox that could be ticked
        to reveal nothing would be the control contradicting itself, which is §6e.4's rule.

        **The state is carried twice wherever this is used** — the greyed box for everyone, and a
        sentence beneath the block for anyone who cannot separate the two greys. Item 5's
        z-batch box set that precedent and the sentence is hidden again once it stops being true.
        """
        self._boxes[name].setEnabled(enabled)

    def is_enabled(self, name: str) -> bool:
        return self._boxes[name].isEnabled()

    def names(self) -> list[str]:
        return list(self._boxes)


class ParameterSlider(QWidget):
    """Name, live value, slider, bare endpoint labels. §6g.5, specified once program-wide.

    **The range labels stay bare** — `0` and `0.18`, not `min`/`max` and not `LIVE`. The user's
    decision: the number is the number, and the "?" popup is where any coupling to another
    screen's parameter gets explained. §4's own rule already requires every popup to say whether
    its control is live or takes effect on commit, so the slider does not repeat it.
    """

    value_changed = Signal(float)

    def __init__(self, name: str, minimum: float, maximum: float, value: float,
                 *, decimals: int = 2, steps: int = 1000, log: bool = False,
                 help_button: QWidget | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._min, self._max, self._steps, self._decimals = minimum, maximum, steps, decimals

        #: **Log scaling is item 9a's, and it is the THIRD time this component has met the same
        #: failure.** `cd68_threshold` is an absolute intensity whose useful range spans three
        #: orders of magnitude low in the nominal one — CD68 p99.9 runs 136 to 58,641 across 13
        #: images while puncta live at 10²–10⁴ — so a linear slider would put the entire
        #: decision in the bottom ~2% of its travel. §2b's *"dead across most of its travel"*,
        #: after threshold 1's range and threshold 2's cap.
        #:
        #: **A non-positive minimum is refused rather than clamped.** `log(0)` has no answer,
        #: and silently substituting one would put the slider's own floor somewhere the caller
        #: did not choose — on a screen whose whole point is that the number is exact.
        self._log = bool(log)
        if self._log and minimum <= 0:
            raise ValueError("a logarithmic slider needs a positive minimum; got "
                             f"{minimum}. Use the volume's smallest non-zero value.")

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        self.name_label = QLabel(name)
        self.value_label = QLabel(self._format(value))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, steps)
        self.slider.setValue(self._to_slider(value))
        self.slider.valueChanged.connect(self._on_slider)

        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        low = QLabel(self._format(minimum))
        high = QLabel(self._format(maximum))
        for end in (low, high):
            end.setProperty("role", "helper")
        ends.addWidget(low)
        ends.addStretch(1)
        ends.addWidget(high)

        #: **The "?" sits beside the NAME, not beside the live value.** §4's rule attaches one
        #: to every adjustable control, and the name is what identifies the control — the value
        #: is a readout of it. Optional so the widget stays usable in a test with no popup text.
        if help_button is None:
            column.addWidget(self.name_label)
        else:
            self.help_button = help_button
            heading = QHBoxLayout()
            heading.setContentsMargins(0, 0, 0, 0)
            heading.addWidget(self.name_label)
            heading.addStretch(1)
            heading.addWidget(help_button)
            column.addLayout(heading)
        column.addWidget(self.value_label)
        column.addWidget(self.slider)
        column.addLayout(ends)

    def _format(self, value: float) -> str:
        """Trailing zeros go, and **two things that look like tidiness are not.**

        > ### `rstrip("0")` ATE THE ZEROS OF INTEGERS — `100` RENDERED AS `1`
        >
        > It stripped unconditionally, which is right for `0.50` and catastrophic for `100`.
        > **Every screen before item 9a used `decimals=2` or more**, so the rounded text always
        > carried a `.` and the bug could not fire; 9a is the first with `decimals=0`, where
        > `f"{100:.0f}"` is `"100"` and stripping leaves `"1"`. **Only strip after a decimal
        > point.**

        > ### AND A NON-ZERO VALUE MUST NEVER RENDER AS `0`
        >
        > 9a's slider floor is the volume's smallest non-zero intensity, which at `decimals=0`
        > rounds to `"0"` — on a control whose whole argument is that zero is not a candidate
        > threshold, and beside a box that refuses one. **Falls back to three significant
        > figures**, so a small value reads as small rather than as nothing.
        """
        text = f"{value:.{self._decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if not text or float(text) == 0.0:
            return "0" if value == 0 else f"{value:.3g}"
        return text

    def _to_slider(self, value: float) -> int:
        if self._log:
            low, high = math.log(self._min), math.log(self._max)
            if high == low:
                return 0
            here = math.log(min(max(float(value), self._min), self._max))
            return int(round((here - low) / (high - low) * self._steps))
        span = self._max - self._min
        return 0 if span == 0 else int(round((value - self._min) / span * self._steps))

    def _from_slider(self, position: int) -> float:
        if self._log:
            low, high = math.log(self._min), math.log(self._max)
            return float(math.exp(low + (high - low) * position / self._steps))
        return self._min + (self._max - self._min) * position / self._steps

    def _on_slider(self, position: int) -> None:
        value = self._from_slider(position)
        self.value_label.setText(self._format(value))
        self.value_changed.emit(value)

    def value(self) -> float:
        return self._from_slider(self.slider.value())

    def as_displayed(self, value: float) -> float:
        """`value` as this slider **renders** it, parsed back to a number.

        > ### THE SCREEN SAID ONE NUMBER AND THE PARAMETERS FILE RECORDED ANOTHER
        >
        > `value_label` shows `_format(value)`; at `decimals=0` that is `f"{v:.0f}"`, so a log
        > position landing on 2681.4372 reads **2681** — and the screens stored the raw float.
        > That number is hashed into a parameters file and applied to every image in the study,
        > so the file and the glass disagreed by exactly the fraction the label had rounded away.
        > It arrived with 9a's absolute value box being removed: while the box existed it was
        > authoritative, so the stored number was one somebody had typed.
        >
        > **The user's rule, 2026-08-30:** *"if the slider says whole numbers then the recorded
        > value should be a whole number as well."* Storing what is displayed makes the
        > quantisation visible instead of hiding it, which is most of what the box was for.

        **Through `_format` rather than `round(value, decimals)`**, so the two cannot drift: the
        three-significant-figure fallback that stops a small non-zero threshold *rendering* as
        `0` is then also what stops it being *recorded* as 0.
        """
        return float(self._format(value))

    def set_value(self, value: float) -> None:
        self.slider.setValue(self._to_slider(value))


# --------------------------------------------------------------------------
# The "?" — §4's rule: every popup says whether its control is live or commits
# --------------------------------------------------------------------------

class HelpPopup(QDialog):
    """`#2e323a`, one step lighter than the screen, 1 px `#5a626c` border.

    **One step lighter, not the same grey.** A modal sharing the screen's exact background reads
    as part of the page — and every ratio survives the lift, which was checked rather than
    assumed.
    """

    def __init__(self, title: str, body: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(theme.title_case(title))
        self.setProperty("role", "popup")
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 14, 16, 14)
        column.setSpacing(10)
        text = QLabel(body, self)
        text.setWordWrap(True)
        text.setMaximumWidth(420)
        column.addWidget(text)
        close = QPushButton("Close", self)
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        column.addLayout(row)


class NavigationRow(QWidget):
    """**Back on the left, Continue on the right**, at the foot of the right-hand column.

    One factory, because from 2026-08-30 nearly every screen carries this pair and ten copies of
    a two-button row drift ten ways. **The layout is item 4's, unchanged** — it was the only
    screen with a Back button and the user named its arrangement when asking for the rest:
    *"the back button should be placed beside the continue button... back on the left continue
    on the right of the bottom right panel."*

    **Three shapes, because three are needed and no more:**

    | screen | |
    |---|---|
    | most | both buttons |
    | item 1, item 9a, item 7b | Continue only — they have no Back by decision |
    | item 8 | Back only. It is last in `FLOW_ORDER`, so it has never had a Continue |

    A button that is not asked for is **not built**, rather than built and hidden: `back_button`
    is `None` there, so a screen that wires a signal it does not have fails loudly at
    construction instead of offering a control that silently does nothing. That is the lesson
    from item 9b's `Step by batch`, which shipped disabled at every depth for as long as the
    screen existed.
    """

    back_clicked = Signal()
    continue_clicked = Signal()

    def __init__(self, back: bool = True, forward: bool = True,
                 parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self.back_button = QPushButton("Back", self) if back else None
        if self.back_button is not None:
            self.back_button.clicked.connect(self.back_clicked)
            row.addWidget(self.back_button)

        row.addStretch(1)

        self.continue_button = QPushButton("Continue", self) if forward else None
        if self.continue_button is not None:
            self.continue_button.clicked.connect(self.continue_clicked)
            row.addWidget(self.continue_button)

    def set_continue_enabled(self, enabled: bool) -> None:
        """The gate's half of the row. **Silent when there is no Continue** — item 8's row has
        none, and a screen calling this generically must not have to know that."""
        if self.continue_button is not None:
            self.continue_button.setEnabled(bool(enabled))

    def set_back_enabled(self, enabled: bool) -> None:
        """Item 8 disables Back once SAVE has been pressed — the user's rule, 2026-08-30:
        *"no back after save has been pressed."* The run is durable at that point."""
        if self.back_button is not None:
            self.back_button.setEnabled(bool(enabled))


def batch_step_box(parent: QWidget | None = None) -> QCheckBox:
    """`Step by batch` — the z-window toggle. **One factory so five screens say one thing.**

    Unticked is the rolling window the screens have always had: LEFT/RIGHT move **one slice**
    and the window slides across the stack. Ticked, they move a **whole batch**, so the windows
    tile it instead.

    **It is meaningless at depth 1** — both modes show the same single slice — so it ships
    disabled here and **each screen re-enables it as its own depth rises.**

    > ### AND ONE SCREEN NEVER DID — item 9b, found 2026-08-30
    >
    > `screen_plaque.set_depth` never called `setEnabled`, so the box was dead at every depth and
    > 9b's z-batch toggle could not be clicked at all. **The count guarding this factory could
    > not see it**: it counts the screens whose source mentions `batch_step_box`, which is five
    > either way, and a screen that wires the toggle looks identical to one that gates it.
    > `test_the_batch_toggle_can_be_reached_on_every_screen_that_has_one` now raises each
    > screen's depth and asserts the box is enabled — the property the count was standing in for.
    """
    box = QCheckBox("Step by batch", parent)
    box.setToolTip("LEFT and RIGHT move a whole z batch instead of one slice")
    box.setEnabled(False)
    return box


class ConfirmPopup(QDialog):
    """Two buttons, one question, no default. `COMPONENT_03_PLAN.md` §6e.6, §4 item 4.

    **Item 1's *"This image has been processed before"* and item 4's discard warning are the same
    popup with different words**, so it is built once. Same `#2e323a` surface and 1 px `#5a626c`
    border as `HelpPopup` — *one step lighter* keeps meaning *this is a modal*.

    **Neither button is a default and neither is destructive-red.** §4 item 4's warning fires only
    when there is something to lose, so both answers are legitimate: the question is *go back, or
    continue*, not *are you sure*. Colouring one as danger would make the other look like the
    right answer, which is a judgement the screen has no basis for.
    """

    def __init__(self, message: str, confirm: str, cancel: str,
                 title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(theme.title_case(title or confirm))
        self.setProperty("role", "popup")
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 14, 16, 14)
        column.setSpacing(14)

        text = QLabel(message, self)
        text.setWordWrap(True)
        text.setMaximumWidth(420)
        column.addWidget(text)

        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = QPushButton(theme.title_case(cancel), self)
        self.confirm_button = QPushButton(theme.title_case(confirm), self)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button.clicked.connect(self.accept)
        row.addWidget(self.cancel_button)
        row.addWidget(self.confirm_button)
        row.addStretch(1)
        column.addLayout(row)

    def ask(self) -> bool:
        """`True` if the user chose the confirming button."""
        return self.exec_() == QDialog.DialogCode.Accepted


class ChoicePopup(QDialog):
    """Two or more legitimate answers **and a way out**. `ConfirmPopup` with an exit.

    > ### ⇒ WHY `ConfirmPopup` COULD NOT DO THIS — 2026-08-30
    >
    > `ConfirmPopup.ask()` returns a **bool**, so a two-button question has exactly two
    > outcomes and *dismissing it* has to be folded into one of them. Item 7b's reject popup
    > asked *real cell or debris?* through it, and Escape — or the window's X — returned
    > `False`, which the caller read as **debris**: the answer that removes the cell from every
    > count, reachable by pressing a key that means *"I did not mean to do this"*.
    >
    > **`screen_edit_cell.reject` already had `if reason is None: return None`.** It was dead
    > code, because nothing could produce the `None`. The intent was written and never wired;
    > this widget is what wires it.

    **No button is a default**, as in `ConfirmPopup` and for its reason: every option here is a
    legitimate answer, and defaulting one would tell the reviewer which the screen expects.
    Cancel sits leftmost and is not one of the answers.

    ``options`` is ``(label, value)`` in the order they should read left to right. ``ask``
    returns the chosen value, or ``None`` if the reviewer cancelled or dismissed the dialog —
    **one outcome per way of leaving**, which is the whole point.
    """

    def __init__(self, message: str, options: Sequence[tuple[str, object]],
                 cancel: str = "Cancel", title: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        if not options:
            raise ValueError("a choice needs at least one option")
        self.setWindowTitle(theme.title_case(title or "Choose"))
        self.setProperty("role", "popup")
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 14, 16, 14)
        column.setSpacing(14)

        text = QLabel(message, self)
        text.setWordWrap(True)
        text.setMaximumWidth(420)
        column.addWidget(text)

        self._value: object | None = None
        self.option_buttons: list[QPushButton] = []

        row = QHBoxLayout()
        row.addStretch(1)
        #: **Leftmost, and separated from the answers by being first.** The count popup puts its
        #: way out on the left too, so the position means the same thing on both.
        self.cancel_button = QPushButton(theme.title_case(cancel), self)
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.cancel_button)
        for label, value in options:
            button = QPushButton(theme.title_case(label), self)
            button.clicked.connect(lambda _checked=False, v=value: self._choose(v))
            row.addWidget(button)
            self.option_buttons.append(button)
        row.addStretch(1)
        column.addLayout(row)

    def _choose(self, value: object) -> None:
        self._value = value
        self.accept()

    def ask(self):
        """The chosen value, or `None` if the reviewer cancelled or dismissed the dialog."""
        if self.exec_() != QDialog.DialogCode.Accepted:
            return None
        return self._value


class CountPopup(QDialog):
    """One question, one number, two buttons. Component 5's *"how many cells?"*.

    **A `ConfirmPopup` cannot express it.** The count is `1`, `2`, `3` — §4.4's task NEW-d,
    *"this is 3 cells I cannot afford to split"* — and two buttons can only ever answer a yes/no
    question about a number rather than carry the number itself.

    **The spin box takes focus with its value selected**, so the whole interaction is a digit and
    Enter. That is NEW-d's constraint restated: *"entering the number must cost almost nothing,
    or it will not be entered"* — and a count nobody enters reads as zero lost cells, which is
    worse than no measurement at all.

    Floors at 1: a rejection covers at least the cell being rejected.
    """

    def __init__(self, message: str, default: int = 1, minimum: int = 1, maximum: int = 99,
                 confirm: str = "Record", cancel: str = "Cancel",
                 title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(theme.title_case(title or confirm))
        self.setProperty("role", "popup")
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 14, 16, 14)
        column.setSpacing(14)

        text = QLabel(message, self)
        text.setWordWrap(True)
        text.setMaximumWidth(420)
        column.addWidget(text)

        self.spin = QSpinBox(self)
        self.spin.setRange(int(minimum), int(maximum))
        self.spin.setValue(max(int(minimum), int(default)))
        self.spin.selectAll()
        column.addWidget(self.spin)

        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = QPushButton(theme.title_case(cancel), self)
        self.confirm_button = QPushButton(theme.title_case(confirm), self)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button.clicked.connect(self.accept)
        #: **The confirming button is the default**, unlike `ConfirmPopup`'s deliberate absence
        #: of one. There both answers are legitimate; here Enter means *"the number as shown"*,
        #: which is the whole of the keystroke constraint.
        self.confirm_button.setDefault(True)
        row.addWidget(self.cancel_button)
        row.addWidget(self.confirm_button)
        row.addStretch(1)
        column.addLayout(row)

    def ask(self) -> int | None:
        """The number, or `None` if the reviewer cancelled."""
        if self.exec_() != QDialog.DialogCode.Accepted:
            return None
        return int(self.spin.value())


class HelpButton(QPushButton):
    """The `?`, in muted grey so it never competes with the control it explains."""

    def __init__(self, title: str, body: str, parent: QWidget | None = None):
        super().__init__("?", parent)
        self.title, self.body = title, body
        self.setFlat(True)
        self.setFixedWidth(16)
        self.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {theme.MUTED}; "
            f"border: none; padding: 0; }}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._open)

    def _open(self) -> None:
        HelpPopup(self.title, self.body, self).exec_()


# --------------------------------------------------------------------------
# Commit, header strip, table, list
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# The rejection sequence — one implementation, two screens. Patch item B5
# --------------------------------------------------------------------------
#: > ### ⇒ IT LIVED ON `EditCellScreen` AND THE TRIAGE GRID NEEDED IT TOO — 2026-09-03
#: >
#: > The delete button on a triage tile asks **the same two questions in the same order** as
#: > rich review's Reject, because it is the same decision taken from a different place. Two
#: > copies of a two-popup loop would drift two ways — and this one has been corrected twice
#: > already, once for Escape recording debris and once for Cancel committing at 1.
#: >
#: > `widgets.py` is the home because both screens already import it and `editing` imports
#: > nothing from here. The import is **inside the functions**, `PipelineCompute`'s rule: it
#: > keeps a module every screen loads free of `scipy` and the split machinery.


def reject_reason_popup(parent=None) -> str | None:
    """Popup one. **Three outcomes, and `None` is a real one** — Cancel ends the rejection.

    Neither answer is a default: both are legitimate, and defaulting one would tell the reviewer
    which the screen expects. The order is Debris then Real Cell, reading left to right after
    the way out, so a reviewer's hand goes where it went before.
    """
    from . import editing

    return ChoicePopup(editing.reject_question(),
                       [("Debris", "debris"), ("Real Cell", "real_cell")],
                       cancel="Cancel", title="Reject cell", parent=parent).ask()


def cell_count_popup(parent=None) -> int | None:
    """Popup two, on `real_cell` only. **`None` means the reviewer wants the question back.**

    A number, not a yes/no: *"this is 3 cells I cannot afford to split"*. The button says
    **Back** rather than Cancel because that is what it does — Cancel ending the rejection on
    the first popup and committing it on the second was the defect, not the default.
    """
    from . import editing

    return CountPopup(editing.count_question(), default=1, cancel="Back",
                      title="How many cells", parent=parent).ask()


def ask_rejection(parent=None, *, ask_reason=None, ask_count=None):
    """The two-popup loop, returning an `editing.Resolution` or `None`.

    * **Cancel on the reason popup** ends it. Nothing is recorded and the cell is left as it was.
    * **Debris** commits at once — one press, the asymmetry §3b records and accepts.
    * **Real cell** asks the count. **Back** returns to the reason popup; **Record** commits.

    **It cannot trap anyone**, which is why the reason popup has a Cancel: going back to a
    question with no way out would leave a reviewer able to leave only by answering it, and the
    cheapest answer is the one that discards the cell.

    The two questions are injectable so a screen can keep its own replaceable hooks — which is
    how the whole of this is driven in the suite, with no modal ever shown.
    """
    from . import editing

    ask_reason = ask_reason or (lambda: reject_reason_popup(parent))
    ask_count = ask_count or (lambda: cell_count_popup(parent))
    while True:
        reason = ask_reason()
        if reason is None:
            return None
        if reason != "real_cell":
            return editing.Resolution("reject", reason=reason, n_cells=1)
        n_cells = ask_count()
        if n_cells is not None:
            return editing.Resolution("reject", reason=reason, n_cells=n_cells)
        #: Back: round the loop and ask the reason again.


class CommitButton(QWidget):
    """Commit, its red `!`, and the working line beneath.

    **The `!` and the message never appear alone** — the mark carries *the view is stale* for a
    colour-blind reader and the colour carries it for everyone else. §6e.4's rule, satisfied the
    same way item 1 satisfies it.
    """

    clicked = Signal()

    def __init__(self, label: str = "Commit", parent: QWidget | None = None):
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton(theme.title_case(label), self)
        self.button.clicked.connect(self.clicked)
        self.mark = QLabel("!", self)
        self.mark.setStyleSheet(f"color: {theme.ERROR};")
        self.mark.setVisible(False)
        row.addWidget(self.button)
        row.addWidget(self.mark)
        row.addStretch(1)

        self.status = QLabel("", self)
        self.status.setProperty("role", "helper")

        column.addLayout(row)
        column.addWidget(self.status)
        self._stale = False

    def set_stale(self, stale: bool) -> None:
        """Show or hide the exclamation. Stale means *the view no longer matches the commit*."""
        self._stale = stale
        self.mark.setVisible(stale)

    def is_stale(self) -> bool:
        """The widget's own state, **not** `mark.isVisible()`.

        Qt reports a child of a window that has never been shown as not visible regardless of
        `setVisible(True)`, so reading the flag back off the label would make this return False
        for every screen still being assembled — and the gate it guards is the one stopping a
        user continuing from a stale view.
        """
        return self._stale

    def set_working(self, message: str = "") -> None:
        """The progress line, and **the repaint that makes it exist.**

        > ### A LINE THAT NEVER PAINTED IS NOT A PROGRESS LINE — task 24
        >
        > Every commit screen already set this before its work and cleared it after, and **not
        > one of them ever appeared on screen.** The work is synchronous, so between the two
        > calls the event loop never runs, no paint event is delivered, and the user sees an
        > unchanged window for 17 s at item 2 and **200-500 s at item 4** — with the button
        > apparently still enabled, because that repaint had not happened either.
        >
        > Q2's rule is *progress shown, no cancel*. Showing it is this: set the text, disable the
        > button, and **force the widget to repaint before returning**, so the caller's next
        > line — the blocking one — starts with the message already on the glass.

        `processEvents` rather than `repaint()` alone: the button's disabled state and the
        status text are two widgets, and on Windows the top-level window also has to be told to
        composite them. **Scoped to `ExcludeUserInputEvents`**, so a second click queued during
        the work cannot re-enter the commit that is already running.
        """
        self.status.setText(message)
        self.button.setEnabled(not message)
        if message:
            from qtpy.QtCore import QEventLoop
            from qtpy.QtWidgets import QApplication

            application = QApplication.instance()
            if application is not None:
                application.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


class HeaderStrip(QWidget):
    """Panel names on the left, the readout right-aligned to the canvas's right edge.

    **No text is ever drawn over image data, on any of the eleven screens.** Text over dark
    fluorescence is legible right up until the image is bright underneath it, and the
    alternative is inventing a scrim — a chrome element §6e deliberately does not have.
    """

    def __init__(self, panels: Iterable[str] = (), readout: str = "",
                 help_button: QWidget | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 4)
        row.setSpacing(12)

        #: A panel name may bring its own "?" as `(name, help_button)`. **Item 6 is the only
        #: screen that needs it**, and the reason is specific: its `FIELD VIEW` encoding carries
        #: four separate meanings — size, removed, selected, border — with nothing else on screen
        #: to explain any of them. Everywhere else a panel name is just a name.
        self._panels = []
        for entry in panels:
            name, panel_help = entry if isinstance(entry, tuple) else (entry, None)
            label = QLabel(name, self)
            label.setSizePolicy(QSizePolicy.Policy.Expanding if panel_help is None
                                else QSizePolicy.Policy.Preferred,
                                QSizePolicy.Policy.Preferred)
            row.addWidget(label, 0 if panel_help is not None else 1)
            if panel_help is not None:
                row.addWidget(panel_help, 0)
                row.addStretch(1)
            self._panels.append(label)
        if not self._panels:
            row.addStretch(1)

        self._panel_row = row
        self.readout = QLabel(readout, self)
        self.readout.setProperty("role", "readout")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.readout, 0)

        #: §6f.10 and §6g.3 both draw a "?" after the readout — `4.7%  ?` and `489  ?`. A
        #: readout is not an adjustable control, so §4's blanket rule does not reach it; these
        #: two carry one because the number itself needs explaining, not the control behind it.
        self.help_button = help_button
        if help_button is not None:
            row.addWidget(help_button, 0)

    def set_panels(self, names) -> None:
        """Retitle the panels in place. **Item 3 is the only screen that needs it.**

        Its panels change what they hold when the commit lands — raw and threshold become
        threshold and objects — so the names have to move with them or the strip lies about
        what is underneath it. Everywhere else a panel name is fixed for the life of the screen.

        Refuses a different number of names rather than silently retitling a subset: a strip
        that named two panels and now names three would be describing a grid this screen does
        not have.
        """
        names = list(names)
        if len(names) != len(self._panels):
            raise ValueError(
                f"this strip has {len(self._panels)} panel(s), got {len(names)}: {names}")
        for label, name in zip(self._panels, names):
            label.setText(name)

    def set_readout(self, text: str) -> None:
        self.readout.setText(text)


class AccumulatingTable(QFrame):
    """One row per commit, newest appended, the current row marked with `●` in the left margin.

    **A mark rather than a colour**, because the muted/normal pair is already carrying the
    header/data hierarchy and §6e.4 forbids colour being the only carrier of a state.

    Header row muted, data rows normal. Scrolls; item 5 and item 9b are the two screens that
    have one, and after 9b they sit in the same relation to their canvas — below it.
    """

    def __init__(self, columns: Sequence[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.columns = list(columns)
        self._rows: list[Sequence[object]] = []
        self._current = -1

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(2)

        marker = QLabel("", self)
        marker.setFixedWidth(12)
        self._grid.addWidget(marker, 0, 0)
        for index, name in enumerate(self.columns):
            header = QLabel(name, self)
            header.setProperty("role", "helper")
            self._grid.addWidget(header, 0, index + 1)

    def add_row(self, values: Sequence[object]) -> None:
        if len(values) != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} values, got {len(values)}")
        self._rows.append(tuple(values))
        self._current = len(self._rows) - 1
        row_index = self._current + 1

        marker = QLabel("●", self)
        marker.setFixedWidth(12)
        self._grid.addWidget(marker, row_index, 0)
        for column, value in enumerate(values):
            self._grid.addWidget(QLabel(str(value), self), row_index, column + 1)

        for previous in range(1, row_index):
            item = self._grid.itemAtPosition(previous, 0)
            if item and item.widget():
                item.widget().setText("")

    def update_current_row(self, values: Sequence[object]) -> None:
        """Rewrite the newest row in place. **Item 9b's floor, and only item 9b's.**

        §6l.7: *"the floor is in the table because it moves. Changing it updates the LIVE ROW
        ONLY; earlier rows keep the floor they were taken at, so two rows at different floors can
        never be read as comparable."* Appending instead would turn a free filter into a second
        commit and make the trend the table exists to show unreadable.

        **Silently does nothing before the first row**, because item 9b's floor is inert until
        there are components to filter — raising there would make the caller guard a state the
        screen already guards with a disabled box.
        """
        if self._current < 0:
            return
        if len(values) != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} values, got {len(values)}")
        self._rows[self._current] = tuple(values)
        row_index = self._current + 1
        for column, value in enumerate(values):
            item = self._grid.itemAtPosition(row_index, column + 1)
            if item and item.widget():
                item.widget().setText(str(value))

    def rows(self) -> list[Sequence[object]]:
        return list(self._rows)

    def current_index(self) -> int:
        return self._current


class _SelectableRow(QLabel):
    """One row of a `SelectionList`, and **it reports its own clicks.**

    > ### `select()` EXISTED AND NOTHING CALLED IT — task 26
    >
    > The rows were plain `QLabel`s, which have no mouse handling at all, so the list could only
    > be driven programmatically: `select`, `selection_changed` and item 6's `_on_row_selected`
    > were all built, wired and unreachable. A reviewer found it the only way it could be found
    > — *"i can't select anything, not the list or on the field view"* — because the class the
    > list is made of simply does not respond to a pointer.
    >
    > `screen_triage._ThumbnailLabel` already does this for the contact sheet; this is the same
    > two lines for the same reason.
    """

    clicked = Signal(int)

    def __init__(self, text: str, index: int, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.index = index
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):        # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class SelectionList(QWidget):
    """A list whose selected row sits on a `#414851` band. §6j.6, program-wide from item 6.

    Rows can be **dimmed** rather than hidden — item 6 drops everything below the cutoff to
    muted, so the trough is visible by eye rather than removed from the list. Values are
    right-aligned so a column of numbers lines up.

    > ### IT SCROLLS, AND THAT IS NOT A REFINEMENT — found by running item 6
    >
    > One row per cell, and **one measured field held 537 objects**. Without a scroll area the
    > column's size hint is the sum of every row, Qt honours it, and the whole window grows past
    > the bottom of the display — taking item 6's lower panel and its `Detail` header with it.
    > Photographed at 2470 px tall on a window asked for 950.
    >
    > **Bunya makes it worse rather than better.** The run happens in an Open OnDemand desktop
    > session, where the desktop is whatever the browser window is, and a window that sizes
    > itself to its content has no way back.
    >
    > Dimming still never removes a row, so the trough stays visible by scrolling to it.
    """

    selection_changed = Signal(int)

    #: Tall enough to show the trough — 12 rows at a glance — and short enough that the left
    #: column cannot drive the window's height.
    MAX_HEIGHT_PX = 320

    #: **And a floor, because a ceiling alone collapsed it to two rows.** The list is added to a
    #: column that ends in `addStretch(1)`, so a widget with no minimum gives all its height
    #: away — item 6's list photographed at two and a half rows on a 950 px window. *"The LIST
    #: is the control on this screen"*, and a control showing two of 45 rows is not one.
    MIN_HEIGHT_PX = 200

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._body = QWidget(self)
        self._column = QVBoxLayout(self._body)
        self._column.setContentsMargins(0, 0, 0, 0)
        self._column.setSpacing(0)

        self.scroll = QScrollArea(self)
        self.scroll.setWidget(self._body)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setMinimumHeight(self.MIN_HEIGHT_PX)
        self.scroll.setMaximumHeight(self.MAX_HEIGHT_PX)
        outer.addWidget(self.scroll)

        self._rows: list[_SelectableRow] = []
        self._dimmed: set[int] = set()
        self._flagged: set[int] = set()
        self._selected = -1

    def set_items(self, items: Sequence[str], flags: Sequence[bool] = ()) -> None:
        while self._column.count():
            widget = self._column.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self._rows.clear()
        self._dimmed.clear()
        self._flagged.clear()
        self._selected = -1

        marks = list(flags) + [False] * (len(items) - len(flags))
        self._flagged = {i for i, mark in enumerate(marks) if mark}
        for index, text in enumerate(items):
            row = _SelectableRow(f"{text}  !" if marks[index] else text, index, self._body)
            row.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.setContentsMargins(6, 2, 6, 2)
            #: **A dimmed row is still selectable.** Item 6 dims everything below the cutoff to
            #: show the trough, and clicking one is how a user says *"that is too small"* — the
            #: cutoff box takes its size. Muting is a statement about the cell, not the row.
            row.clicked.connect(self.select)
            self._column.addWidget(row)
            self._rows.append(row)
        self._column.addStretch(1)
        self._restyle()

    def set_dimmed(self, indices: Iterable[int]) -> None:
        """Rows below item 6's cutoff — muted, not removed."""
        self._dimmed = set(indices)
        self._restyle()

    def select(self, index: int) -> None:
        self._selected = index
        self._restyle()
        self.selection_changed.emit(index)

    def selected_index(self) -> int:
        return self._selected

    def _restyle(self) -> None:
        """**The flag outranks the dimming, and that is a decision rather than an ordering.**

        A row can be both flagged and dimmed — on item 6 that is a border-touching cell below
        the cutoff, which is *precisely* the case the flag exists to surface: a cell about to be
        deleted for being small when it is only cropped. **Letting the muted grey win there
        would erase the warning at the one moment it matters.**

        Found because `_restyle` rebuilt every row's stylesheet from scratch and silently
        dropped the colour `set_items` had applied — so the flag vanished the first time
        anything called `set_dimmed`. Item 6 is the first screen to use both.
        """
        for index, row in enumerate(self._rows):
            parts = []
            if index == self._selected:
                parts.append(f"background: {theme.LIST_SELECTION};")
            if index in self._flagged:
                parts.append(f"color: {theme.WARNING};")
            elif index in self._dimmed:
                parts.append(f"color: {theme.MUTED};")
            row.setStyleSheet(" ".join(parts))
