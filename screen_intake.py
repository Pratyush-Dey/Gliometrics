"""Screen 1 — Image Intake. `COMPONENT_03_PLAN.md` §6e.5, §4 item 1, §4a.10.

**The only screen in the project with no image on it**, which is why the colour session took it
first: it settles the half of the palette that is least contentious and everything else inherits.

Two things here are load-bearing beyond this screen:

    TWO ROOTS      image root on the RDM, output root on scratch. One root is impossible
    THE ROLES      they must reach the parameters file, or `segment` exits 4 one command later

**Nothing is remembered between runs** — not the roots, not the roles, not the voxel size. The
one piece of cross-run state that was proposed (pre-filling from the most recent parameters file)
was withdrawn: invisible state is exactly what the decision removes.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from . import theme
from .semiauto import Flow, Session
from .frames import scrolling_body
from .widgets import HelpButton, screen_title

__all__ = ["IntakeScreen", "parse_voxel_size", "identity_from_path", "ROLE_CHOICES",
           "ROLE_LABELS"]

#: The four roles, plus the honest option. **`unused` is not padding** — a two-channel study is
#: real, and `verify_roles` is inert without a CD68 channel, so declaring the absence is the only
#: way the parameters file can say so. Presence counts, not just index.
ROLE_CHOICES = ("unused", "nuclei", "cell", "cd68", "plaque")

#: What the dropdown SHOWS against what the parameters file RECORDS. The user's decision,
#: 2026-08-20: *"i want it to say the stain first and then the role in brackets."*
#:
#: **A channel is identified by its stain at the microscope, not by the role the pipeline gives
#: it.** `Iba1` is on the acquisition sheet; `cell` is not. Naming the role alone asks the person
#: at the screen to do the translation every time, and getting it wrong is silent — a swapped
#: `cell` and `cd68` produces plausible numbers, never an error. The bracket keeps both
#: vocabularies on one line so neither has to be remembered.
#:
#: **THE LABEL IS DISPLAY ONLY AND THE VALUE MUST NOT FOLLOW IT.** `roles()` reads each item's
#: *data*, never its text: these strings become `<role>_channel` in the parameters file and
#: `segment --roles nuclei:0,...` parses the same names one command later. A label that leaked
#: into the file would exit 4 at the next command.
ROLE_LABELS: dict[str, str] = {
    "unused": "unused",
    "nuclei": "DAPI (nuclei)",
    "cell": "Iba1 (cell)",
    "cd68": "CD68 (lysosomal marker)",
    "plaque": "6E10 (plaque)",
}


def parse_voxel_size(text: str) -> tuple[float, float, float]:
    """`"0.6, 0.108, 0.108"` in (z, y, x) micrometres.

    **The one place the file can be wrong invisibly.** Every micron-sized parameter downstream —
    `closing_radius_um`, `nucleus_diameter_um`, `min_cell_volume_um3` — is scaled by this, and a
    wrong value produces plausible numbers rather than an error.
    """
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"expected three numbers (z, y, x), got {len(parts)}: {text!r}")
    try:
        z, y, x = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"voxel sizes must be numbers: {text!r}") from exc
    if min(z, y, x) <= 0:
        raise ValueError(f"voxel sizes must be positive: {text!r}")
    return z, y, x


def identity_from_path(image_root: str, image_path: str) -> tuple[str, str]:
    """`<genotype>/<animal>/<file>` — genotype and animal come from the path, nothing else.

    **Raises when the file does not sit exactly two folders under the root**, which is the error
    §6e.5 draws in red beneath Continue. It is refused rather than guessed because a wrong
    genotype is silent: every number downstream is plausible and the comparison is wrong.
    """
    try:
        relative = Path(image_path).resolve().relative_to(Path(image_root).resolve())
    except ValueError:
        raise ValueError("File is not inside the image root") from None
    if len(relative.parts) != 3:
        raise ValueError("File is not two folders under the image root")
    return relative.parts[0], relative.parts[1]


class IntakeScreen(QWidget):
    """One full screen, stacked top to bottom, Continue centred with red text beneath it."""

    continue_requested = Signal()

    def __init__(self, flow: Flow | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session())
        self.setStyleSheet(theme.stylesheet())

        #: **The whole screen scrolls** — patch item D, the rule every screen now follows. See
        #: `frames.scrolling_body`. **Held as `self.column`**, because `screen.layout()` is the
        #: outer one now and a test walking the form's order needs this one — exactly the note
        #: item 8 carries for the same reason.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scroll, body = scrolling_body(self)
        outer.addWidget(self.scroll)

        self.column = column = QVBoxLayout(body)
        column.setContentsMargins(28, 20, 28, 20)
        column.setSpacing(14)
        column.addWidget(screen_title("image intake"))

        self.image_root = self._picker(
            column, "Image root",
            "top folder of your images — genotype and animal come from it", folder=True)
        self.output_root = self._picker(
            column, "Output root",
            "pool, parameters and exports save here", folder=True)
        self.file_path = self._picker(column, "File", "", folder=False)

        self.role_boxes: list[QComboBox] = []
        self.range_boxes: list[tuple[QLineEdit, QLineEdit]] = []
        for index in range(4):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Channel {index}"))
            box = QComboBox(self)
            #: Label shown, role stored. See `ROLE_LABELS`.
            for role in ROLE_CHOICES:
                box.addItem(ROLE_LABELS[role], role)
            box.currentTextChanged.connect(self._sync)
            row.addWidget(box)
            row.addStretch(1)

            #: **The display range, set here once and used on every screen that shows the
            #: channel.** The user's decision, 2026-08-20: they set it in Fiji and type it in.
            row.addWidget(QLabel("min"))
            low = QLineEdit(self)
            low.setFixedWidth(90)
            low.setPlaceholderText("auto")
            low.textChanged.connect(self._sync)
            row.addWidget(low)
            row.addWidget(QLabel("max"))
            high = QLineEdit(self)
            high.setFixedWidth(90)
            high.setPlaceholderText("auto")
            high.textChanged.connect(self._sync)
            row.addWidget(high)
            if index == 0:
                row.addWidget(HelpButton(
                    "Display range",
                    "How this channel is stretched on screen — the same two numbers as Fiji's "
                    "Brightness/Contrast sliders, in the same units. Set it there, read the "
                    "min and max off the sliders, and type them here.\n\n"
                    "Leave both blank and the channel is stretched between its own 0.1 and "
                    "99.9 percentiles instead, which is usually reasonable but flattens a "
                    "sparse marker like 6E10.\n\n"
                    "It changes what you see and nothing that is measured. It is not written "
                    "to the parameters file.", self))

            self.range_boxes.append((low, high))
            column.addLayout(row)
            self.role_boxes.append(box)

        voxel_row = QHBoxLayout()
        self.voxel_mark = QLabel("!")
        self.voxel_mark.setStyleSheet(f"color: {theme.ERROR};")
        self.voxel_mark.setVisible(False)
        voxel_row.addWidget(self.voxel_mark)
        voxel_row.addWidget(QLabel("Voxel sizes (z, y, x) in micrometres"))
        self.voxel_edit = QLineEdit(self)
        self.voxel_edit.setPlaceholderText("0.6, 0.108, 0.108")
        self.voxel_edit.textChanged.connect(self._sync)
        voxel_row.addWidget(self.voxel_edit)
        voxel_row.addWidget(HelpButton(
            "Voxel size",
            "Read from the file when it carries one, and confirmed by you either way. Every "
            "micron-sized parameter downstream is scaled by it, so a wrong value produces "
            "plausible numbers rather than an error. Takes effect immediately."))
        column.addLayout(voxel_row)

        column.addStretch(1)

        self.continue_button = QPushButton("Continue", self)
        self.continue_button.clicked.connect(self._on_continue)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.continue_button)
        button_row.addStretch(1)
        column.addLayout(button_row)

        self.error = QLabel("", self)
        self.error.setProperty("role", "error")
        self.error.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.error.setWordWrap(True)
        column.addWidget(self.error)

        self._sync()

    # -- construction helpers -------------------------------------------

    def _picker(self, column, label: str, helper: str, *, folder: bool) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        if helper:
            note = QLabel(f"({helper})")
            note.setProperty("role", "helper")
            note.setWordWrap(True)
            row.addWidget(note, 1)
        else:
            row.addStretch(1)

        edit = QLineEdit(self)
        edit.textChanged.connect(self._sync)
        button = QPushButton("Select", self)
        button.clicked.connect(lambda: self._browse(edit, folder))
        row.addWidget(edit, 2)
        row.addWidget(button)
        column.addLayout(row)
        return edit

    def _browse(self, edit: QLineEdit, folder: bool) -> None:
        chosen = (QFileDialog.getExistingDirectory(self, "Select folder") if folder
                  else QFileDialog.getOpenFileName(self, "Select image")[0])
        if chosen:
            edit.setText(chosen)

    # -- state ----------------------------------------------------------

    def set_role(self, channel: int, role: str) -> None:
        """Select a channel's role **by role, not by the label shown.**

        `setCurrentText("cell")` cannot work once the box reads `Iba1 (cell)` — and a
        non-editable `QComboBox` handed text it does not have selects **nothing and says
        nothing**, which is §3's second lesson exactly: a fixture that cannot fail. This
        refuses instead.
        """
        position = self.role_boxes[channel].findData(role)
        if position < 0:
            raise ValueError(f"not a role: {role!r}. Expected one of {ROLE_CHOICES}")
        self.role_boxes[channel].setCurrentIndex(position)

    def display_ranges(self) -> dict[str, tuple[float, float]]:
        """`{role: (min, max)}` for the channels where **both** boxes hold a number.

        A half-filled pair is not a range, so it contributes nothing rather than pairing a
        typed value with an invented one — the same rule `parsed_numbers` applies to the
        parameter boxes, and for the same reason.
        """
        out: dict[str, tuple[float, float]] = {}
        for (low, high), box in zip(self.range_boxes, self.role_boxes):
            role = box.currentData()
            if role == "unused":
                continue
            try:
                pair = (float(low.text().strip()), float(high.text().strip()))
            except ValueError:
                continue
            if pair[0] < pair[1]:
                out[role] = pair
        return out

    def display_range_error(self) -> str:
        """The one message the boxes can produce. **Display-only, but never silently wrong.**

        A typo here does not corrupt a measurement — it corrupts what the user is judging by
        eye, on screens whose whole purpose is judging by eye. So it is refused in the open
        like the voxel size, rather than ignored.
        """
        for index, ((low, high), box) in enumerate(zip(self.range_boxes, self.role_boxes)):
            if box.currentData() == "unused":
                continue
            low_text, high_text = low.text().strip(), high.text().strip()
            if not low_text and not high_text:
                continue
            if not low_text or not high_text:
                return f"Channel {index}: give both a min and a max, or leave both blank"
            try:
                low_value, high_value = float(low_text), float(high_text)
            except ValueError:
                return f"Channel {index}: display range must be numbers"
            if low_value >= high_value:
                return f"Channel {index}: display min must be below max"
        return ""

    def roles(self) -> dict[str, int]:
        """Only assigned roles reach the parameters file. **`unused` means absent, not zero.**

        **Reads each box's data, never its text.** The box shows `Iba1 (cell)`; the file must
        say `cell`. See `ROLE_LABELS`.
        """
        return {box.currentData(): index
                for index, box in enumerate(self.role_boxes)
                if box.currentData() != "unused"}

    def _sync(self) -> None:
        session = self.flow.session
        session.image_root = self.image_root.text().strip()
        session.output_root = self.output_root.text().strip()
        session.image_path = self.file_path.text().strip()
        session.roles = self.roles()
        session.display_range = self.display_ranges()

        try:
            session.voxel_size_um = parse_voxel_size(self.voxel_edit.text())
            voxel_error = ""
        except ValueError as exc:
            session.voxel_size_um = None
            voxel_error = str(exc) if self.voxel_edit.text().strip() else ""

        self.voxel_mark.setVisible(bool(voxel_error))

        message = voxel_error or self.display_range_error()
        if not message and session.image_root and session.image_path:
            try:
                identity_from_path(session.image_root, session.image_path)
            except ValueError as exc:
                message = str(exc)
        if not message:
            message = self.flow.blocking_reason() or ""

        self.error.setText(message)
        self.continue_button.setEnabled(self.flow.can_continue() and not message)

    def _on_continue(self) -> None:
        self._sync()
        if self.flow.can_continue() and not self.error.text():
            self.continue_requested.emit()
