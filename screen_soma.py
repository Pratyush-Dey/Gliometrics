"""Screen 6.5 — Soma Allocation Parameters. The user's design, 2026-08-30.

**Item 8's shape: one stacked form, no viewer.** A method dropdown, then the boxes that method
reads, each with its "?" and its default filled in, and the shared Back/Continue row at the foot.
There is nothing to look at here, which is the whole reason it can be a form — and the "?" says
where to look instead.

> ### ⇒ WHAT THIS SCREEN CHANGES IS SEEN ON ITEM 7b, NOT HERE
>
> Every parameter on this screen moves the **white soma** in rich review. That is the argument
> for placing it at 6.5 rather than on item 8: item 8's own note refused these fields on the
> grounds that *"putting a live mechanism switch on the save screen would invite it to be
> flipped after the cells had already been reviewed under a different one."* **Before review,
> that objection dissolves** — the allocation is settled, then the cells are looked at.

> ### ⇒ THE METHOD IS OFFERED AFTER ALL — REVERSED BY THE USER, 2026-08-30
>
> This screen shipped that morning with `soma_method` fixed at `anchor_sheet`, on the recorded
> ground that `opening` reads **only** `soma_thickness_um`, so offering the switch would put
> three inert boxes on the screen — the exact fault that had `soma_bound_um` and
> `soma_occupancy_theta` deleted rather than shown.
>
> **The user reversed it the same day**, on the ground that the program then had *"no way of
> making it target nucleus surrounding soma only"* — and supplied the thing that dissolves the
> original objection: **the boxes follow the method.** Choose `opening` and the three sheet
> boxes are not merely inert, they are *not there*. Nothing on this screen is ever a control
> that cannot change anything, which is what that rule was protecting.
>
> `METHOD_FIELDS` is that rule in code, and it is the single place the two methods' parameter
> lists are written down.

> ### ⇒ AND THE THREE ARE RESET, NOT REMEMBERED — the user's instruction
>
> *"When opening is chosen the other values become redundant so no need to store those values or
> carry them forward."* They are dataclass fields and so are always **present** in the parameters
> file; what the screen controls is what they are present *at*. Switching to `opening` puts them
> back to their defaults and writes those, rather than leaving whatever was typed under the other
> method to travel in the file and in `params_hash`.
>
> **This is why the hidden boxes keep being written rather than skipped.** A value already in
> `session.chosen` from before the switch would otherwise sit there unchallenged: writing the
> default over it is what actually clears it. It also means two `opening` runs differing only in
> stale sheet numbers cannot land in two different pools while producing identical cells.
>
> *Switching back to `anchor_sheet` does not restore what was typed* — the walk 16 → 8 → 6 → 4
> would have to be retyped. Known, and cheap to change if it bites.

> ### IT NEEDS NO RE-DERIVATION, AND THAT IS NOT LUCK
>
> This screen sits **after** item 6 triggers `_derive`, so a screen that could only write
> `session.chosen` would arrive too late — the export would measure the soma at the defaults
> while 7b drew the new values. `Derivation` carries the five soma fields and `payloads()`
> measures the soma lazily from them, so writing them onto the cached derivation is enough.
> `_sync` does exactly that. See `pipeline.Derivation.payloads`.
"""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout,
                            QWidget)

from ..soma import (SOMA_ANCHOR_UM_DEFAULT, SOMA_METHOD_DEFAULT, SOMA_METHODS,
                    SOMA_MIN_SHEET_UM3_DEFAULT, SOMA_SHEET_AREA_UM2_DEFAULT,
                    SOMA_THICKNESS_UM_DEFAULT, SOMA_WAIST_UM_DEFAULT)
from .semiauto import Flow, Session, Step, parsed_numbers
from .frames import scrolling_body
from .widgets import HelpButton, NavigationRow, screen_title

__all__ = ["SomaParametersScreen", "FIELDS", "METHOD_FIELD", "METHOD_ORDER",
           "METHOD_FIELDS", "METHOD_HELP", "SOMA_THICKNESS_UM_DEFAULT",
           "SOMA_ANCHOR_UM_DEFAULT"]

#: **All five defaults now come from `soma.py`**, which is where their evidence is written.
#: `SOMA_THICKNESS_UM_DEFAULT` was defined here until 2026-08-31 and was the odd one out;
#: it is re-exported so importers of this module are unaffected.

#: The dropdown's own field name. Named rather than inlined because item 8 groups by it too.
METHOD_FIELD = "soma_method"

#: `(field name, default, help text)` — **the field names are `SegmentationParams`' own**, as
#: every other screen writes them, so `screen_save.session_params` carries them through without
#: a translation table. They print here in the order the mechanism applies them: the anchor
#: first, then what is added to it, then what is joined, then what is refused.
#:
#: **These four are the boxes; the method is a dropdown and is not one of them.** `METHOD_FIELDS`
#: says which of them each method actually reads, and the screen shows exactly those.
FIELDS: tuple[tuple[str, float, str], ...] = (
    ("soma_anchor_um", SOMA_ANCHOR_UM_DEFAULT,
     "How far the guaranteed soma reaches out from the nucleus, in micrometres.\n\n"
     "The nucleus plus the cytoplasm within this distance of it is ALWAYS soma — no other "
     "setting on this screen can take it away. It is a floor, not a limit: the cell body "
     "can reach further than this, it just cannot reach less.\n\n"
     "  TOO HIGH -> the base of each process is swallowed into the cell body\n\n"
     "  TOO LOW  -> a cell whose nucleus nearly fills it gets almost no guaranteed body\n\n"
     "Read under BOTH settings of the dropdown, because the guaranteed part is the same "
     "either way."),
    ("soma_thickness_um", SOMA_THICKNESS_UM_DEFAULT,
     "How thick material has to be, anywhere in the cell, to count as cell body on its "
     "own — in micrometres of radius, so material must be TWICE this across.\n\n"
     "This is not a distance from the nucleus. It asks how chunky the material is, "
     "wherever it sits, and anything that passes is joined to the body if it connects "
     "back to it through material that also passes.\n\n"
     "  TOO HIGH -> only the guaranteed part survives and lobes are lost\n\n"
     "  TOO LOW  -> processes start being claimed as cell body\n\n"
     "Read ONLY under 'opening'. It sized the guaranteed part as well until 2026-08-31, "
     "which is why the two could never be set independently."),
    ("soma_sheet_area_um2", SOMA_SHEET_AREA_UM2_DEFAULT,
     "How wide material must be, in square micrometres of cross-section, before it counts as "
     "cell body rather than a process.\n\n"
     "Measured in EVERY direction, so a long process fails it however thick it looks from one "
     "side. 5 is a disc about 2.5 micrometres across.\n\n"
     "  TOO HIGH -> only the ring around the nucleus is soma, and the lobes are lost\n\n"
     "  TOO LOW  -> processes start being claimed as cell body"),
    ("soma_waist_um", SOMA_WAIST_UM_DEFAULT,
     "How narrow a neck INSIDE the cell body may be and still be counted as body, in "
     "micrometres.\n\n"
     "Without it, a body pinched in the middle comes out as two pieces with a process-coloured "
     "band between them. This joins them across a gap of up to twice this value.\n\n"
     "  TOO HIGH -> it also swallows a process lying alongside the body\n\n"
     "  0 switches it off."),
    ("soma_min_sheet_um3", SOMA_MIN_SHEET_UM3_DEFAULT,
     "The smallest DETACHED piece that can still be claimed as part of this cell's body, in "
     "cubic micrometres. 30 is a sphere about 3.9 micrometres across.\n\n"
     "A piece already touching the cell body is never dropped for its size. This only applies "
     "to a piece sitting apart from it, which has to earn its place.\n\n"
     "  TOO LOW  -> a cell picks up junk far down one of its processes\n\n"
     "  TOO HIGH -> a genuine detached lobe is refused"),
)

#: The dropdown's entries, **in the order they are offered**. `SOMA_METHOD_DEFAULT` first and
#: selected, because it is what the study has been walked under; `opening` second, because it is
#: the departure a user goes looking for. `test_ui_screen_soma` asserts this is exactly
#: `soma.SOMA_METHODS` reordered, so a third method cannot be added to the program and quietly
#: miss the screen.
METHOD_ORDER: tuple[str, ...] = (SOMA_METHOD_DEFAULT,) + tuple(
    m for m in SOMA_METHODS if m != SOMA_METHOD_DEFAULT)

#: **Which boxes each method actually reads** — `soma.find_soma`'s own table, in code. This is
#: the whole of the visibility rule and the whole of the reset rule, so the two cannot disagree
#: with each other or with the mechanism.
#:
#: **Both read `soma_anchor_um`** — the anchor is identical under both methods, so mechanism 1's
#: radius is never hidden and never reset.
#:
#: `anchor_sheet` is still derived from `FIELDS` rather than restated, so a sixth box added
#: above is shown by it without a second edit — **less `soma_thickness_um`, which it genuinely
#: does not read.** Before 2026-08-31 it did read it, to size the anchor; the split ended that,
#: and leaving it visible would let an inert value reach `params_hash` — the exact thing
#: `_apply_method` exists to prevent.
METHOD_FIELDS: dict[str, tuple[str, ...]] = {
    "anchor_sheet": tuple(name for name, _, _ in FIELDS if name != "soma_thickness_um"),
    "opening": ("soma_anchor_um", "soma_thickness_um"),
}

#: The dropdown's "?". **It says what each choice does and what it costs**, because this is the
#: one control on the screen whose two settings are not more-or-less of the same thing.
METHOD_HELP = (
    "Which rule decides what counts as cell body BESIDES the nucleus and the cytoplasm "
    "wrapped around it. That part is guaranteed under both settings.\n\n"
    "anchor_sheet — material wide enough in cross-section to be a sheet rather than a line, "
    "in every direction, joined back to the nucleus. It picks up lobes sitting beside the "
    "nucleus, and it is what the boxes below it are for.\n\n"
    "opening — material anywhere in the cell thick enough to hold a ball of "
    "soma_thickness_um. This is the older rule and the only one with published validation "
    "behind it. On these images it finds very little, so the soma comes out close to the "
    "nucleus and the cytoplasm around it — which is the reason to choose it.\n\n"
    "soma_anchor_um is read under both settings and never disappears. The rest follow the "
    "dropdown: soma_thickness_um belongs to opening, and the other three to anchor_sheet, "
    "so each disappears under the setting that does not read it.\n\n"
    "Changing this changes soma volume and everything measured from the soma centre, so a "
    "study should be run under one setting throughout. The parameters file records which."
)


class SomaParametersScreen(QWidget):
    """Item 8's shape, with the shared Back/Continue row at the foot instead of SAVE."""

    continue_requested = Signal()
    back_requested = Signal()

    def __init__(self, flow: Flow | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.SOMA)

        #: **The whole screen scrolls** — patch item D, the rule every screen now follows. See
        #: `frames.scrolling_body`. **Held as `self.column`**, because `screen.layout()` is the
        #: outer one now and a test walking the form's order needs this one.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scroll, body = scrolling_body(self)
        outer.addWidget(self.scroll)

        self.column = column = QVBoxLayout(body)
        column.setContentsMargins(24, 18, 24, 18)
        column.setSpacing(10)
        column.addWidget(screen_title("soma allocation parameters"))

        #: **The block's own "?", as item 8 has one.** The individual boxes say what each number
        #: does; *"where do I see any of this?"* is a question about the screen, and nothing else
        #: on it answers that.
        block_help = QHBoxLayout()
        block_help.addStretch(1)
        block_help.addWidget(HelpButton(
            "These parameters",
            "They decide which part of each cell is counted as the cell BODY and which part is "
            "counted as processes.\n\n"
            "You see what they do on the cell review screen, where the soma is drawn in WHITE. "
            "Change a number here, go forward, and open a cell: the white region moves.\n\n"
            "They are applied to every cell in the study and are saved in the parameters file.",
            self))
        column.addLayout(block_help)

        #: **First on the screen, because it decides what the rest of it is.** The values shown
        #: are `SegmentationParams`' own, not friendly labels: every other row here prints the
        #: stored field name, and a dropdown reading *"Nucleus rim only"* would be the one place
        #: on this screen where the glass and the parameters file said different words. The "?"
        #: carries the meaning, which is how the boxes already work.
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel(METHOD_FIELD, self))
        method_row.addStretch(1)
        self.method_box = QComboBox(self)
        self.method_box.setFixedWidth(200)
        for method in METHOD_ORDER:
            self.method_box.addItem(method, method)
        method_row.addWidget(self.method_box)
        method_row.addWidget(HelpButton(METHOD_FIELD, METHOD_HELP, self))
        column.addLayout(method_row)

        #: **Each row is a widget rather than a bare layout**, which is the whole mechanism
        #: behind the dropdown: a layout cannot be hidden, a widget can.
        self.boxes: dict[str, QLineEdit] = {}
        self.rows: dict[str, QWidget] = {}
        self.defaults: dict[str, float] = {}
        for name, default, help_text in FIELDS:
            holder = QWidget(self)
            row = QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(name, holder))
            row.addStretch(1)
            edit = QLineEdit(f"{default:g}", holder)
            edit.setFixedWidth(200)
            edit.editingFinished.connect(self._sync)
            row.addWidget(edit)
            row.addWidget(HelpButton(name, help_text, holder))
            column.addWidget(holder)
            self.boxes[name] = edit
            self.rows[name] = holder
            self.defaults[name] = float(default)

        column.addStretch(1)

        self.navigation = NavigationRow(back=True, parent=self)
        self.navigation.back_clicked.connect(self.back_requested)
        self.navigation.continue_clicked.connect(self._on_continue)
        column.addWidget(self.navigation)

        #: **Connected after the items were added**, so populating the box does not fire the
        #: handler against half-built state.
        self.method_box.currentIndexChanged.connect(self._on_method_changed)
        self._apply_method()
        self._sync()

    # -- state --------------------------------------------------------------

    def method(self) -> str:
        """The selected `soma_method`. **The stored value, not the label** — they are the same
        string here on purpose."""
        return str(self.method_box.currentData())

    def live_fields(self) -> tuple[str, ...]:
        """The boxes the selected method reads, and so the boxes that are on screen."""
        return METHOD_FIELDS[self.method()]

    def values(self) -> dict:
        """`{field: value}` for the method and everything that parses. **Unparseable text is
        skipped, not guessed** — `semiauto.parsed_numbers`' rule, which every other screen's
        boxes obey.

        **The hidden boxes are included, and that is deliberate.** They have been reset to their
        defaults by `_apply_method`, so including them is what writes the default over whatever
        `session.chosen` was holding from before the switch. Skipping them would leave the old
        value in the parameters file under a method that never reads it.
        """
        chosen = parsed_numbers({name: (edit.text(), float)
                                 for name, edit in self.boxes.items()})
        chosen[METHOD_FIELD] = self.method()
        return chosen

    def _apply_method(self) -> None:
        """Show the boxes this method reads, hide the rest, **and reset what it hides.**

        The reset is the user's instruction and it is not cosmetic: an inert value left at
        whatever was typed still travels in the parameters file and still moves `params_hash`,
        so two runs that produced identical cells would key two different pools.
        """
        live = self.live_fields()
        for name in self.boxes:
            shown = name in live
            self.rows[name].setVisible(shown)
            if not shown:
                self.boxes[name].setText(f"{self.defaults[name]:g}")

    def _on_method_changed(self) -> None:
        self._apply_method()
        self._sync()

    def _sync(self) -> None:
        """Put the boxes onto the session, and onto the derivation if one exists.

        **Both, and the second is what makes the screen work at all.** `session.chosen` is what
        `screen_save.session_params` serialises; the cached `Derivation` is what item 8's
        `payloads()` measures each soma from, and it already exists by the time this screen is
        reached because item 6 built it. Writing only the first would give a parameters file
        that disagreed with the cells it was written beside.

        **`soma_method` rides the same path as the four numbers**, onto both, because
        `Derivation` carries it and `payloads()` hands it to `find_soma`. Item 7b reads it back
        off the derivation through `app._soma_params`, so the white soma follows the dropdown.
        """
        chosen = self.values()
        self.flow.session.chosen.update(chosen)
        derivation = getattr(self.flow.session, "derivation", None)
        if derivation is not None:
            for name, value in chosen.items():
                setattr(derivation, name, value)

    def _on_continue(self) -> None:
        self._sync()
        self.continue_requested.emit()
