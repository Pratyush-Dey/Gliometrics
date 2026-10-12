"""The chrome — colour, type and Qt styling. `COMPONENT_03_PLAN.md` §6e.

**Settled 2026-08-19 and inherited by all eleven screens.** Nothing here is per-screen, and
nothing here is a taste decision that a later screen may revisit.

Two things in this module exist because napari's own theme is wrong for text, and both were
measured rather than assumed:

    napari `dark.error`     #99121f   1.71:1 on the window   dark maroon on dark grey
    napari `secondary`      #868e93   4.37:1                 below WCAG AA

The first is used by napari to tint icons and borders, never for body text — but item 1 needs
red error text and a red exclamation mark, and a validation message a user cannot read is worse
than no validation, because the screen looks merely unresponsive. The second matters because
**only the title is ever bold**, which makes the muted grey load-bearing rather than
decorative. `ERROR` and `MUTED` below replace them program-wide. **Do not "fix" either back to
the theme value.**

**Qt is imported lazily, inside the functions that need it.** The palette is strings, and a
test that checks a contrast ratio or a title-case rule should not have to find a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "WINDOW", "CANVAS", "TEXT", "MUTED", "ERROR", "WARNING",
    "CONTROL", "BUTTON", "POPUP", "POPUP_BORDER", "DISABLED_TEXT",
    "SLIDER_GROOVE", "SLIDER_FILL", "SLIDER_HANDLE",
    "LIST_SELECTION", "SCROLLBAR_THUMB", "SCROLLBAR_THUMB_HOVER",
    "TITLE", "LABEL", "HELPER", "BUTTON_LABEL", "ERROR_TEXT", "READOUT",
    "FontSpec", "STAIN_NAMES", "title_case", "contrast_ratio", "relative_luminance",
    "FontNotBundled", "FONT_DIR", "FONT_FAMILY", "REQUIRED_GLYPHS",
    "load_bundled_font", "missing_glyphs", "stylesheet",
]

# --------------------------------------------------------------------------
# Colour — §6e.3
# --------------------------------------------------------------------------

#: napari's own `dark.background`, inherited unchanged. Not a taste decision: image screens
#: must be dark, and napari's chrome surrounds every one of them, so a different grey would
#: mean two greys in one window.
WINDOW = "#262930"

#: `get_theme('dark').canvas` is `black`, **not** the window grey — verified in napari 0.7.1.
#: Every data colour is judged on this, and the two backgrounds differing is why no divider is
#: needed between the image and the panels (§6f.1).
CANVAS = "#000000"

TEXT = "#e2e4e6"            #: 11.42:1. Ours — napari's #f0f1f2 is 12.88:1, and pure-white-on-dark
                            #: causes halation across a four-hour review.
MUTED = "#9aa0a6"           #: 5.51:1. Ours — napari's `secondary` #868e93 is 4.37:1, below AA.
ERROR = "#f28b82"           #: 6.10:1. Ours, and it must be — see the module docstring.
WARNING = "#e3b617"         #: 7.61:1. napari's, unchanged. First used for item 6's border-cell `!`.

CONTROL = "#414851"         #: napari `foreground` — dropdowns, text boxes, slider groove.
BUTTON = "#5a626c"          #: napari `primary` — button surface, slider fill, popup border.
POPUP = "#2e323a"           #: One step lighter than the window. **Reserved for modals**, so that
                            #: "one step lighter" keeps meaning "this is a popup" (§6f.9).
POPUP_BORDER = BUTTON       #: 1 px.
DISABLED_TEXT = "#6a7380"   #: napari's own `highlight`, so no new colour is invented (§6i.7).

#: The parameter slider, specified once program-wide at §6g.5 and identical on every screen
#: that carries one. No new colour tokens — all three come from the five above.
SLIDER_GROOVE = CONTROL
SLIDER_FILL = BUTTON
SLIDER_HANDLE = TEXT

LIST_SELECTION = CONTROL        #: The selected row's band — §6j.6, program-wide from item 6.
SCROLLBAR_THUMB = BUTTON        #: §3d.7. 12 px and grabbable, because wheel and pinch are
SCROLLBAR_THUMB_HOVER = DISABLED_TEXT   #: reserved for zoom.


# --------------------------------------------------------------------------
# Type — §6e.2
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FontSpec:
    """One row of §6e.2's scale.

    ``case`` is carried as data rather than left to each screen because §6e.2 makes it a rule:
    **Title Case for titles and buttons, sentence case for labels and helper text.** Sentence
    case reads faster in a form, and the split keeps the title distinct without asking bold to
    do all the work.
    """

    size_pt: int
    bold: bool
    case: str          # "title" | "sentence"
    colour: str


#: **Only the title is bold. Nothing else, anywhere, on any screen.** The user's rule, and its
#: consequence is structural rather than cosmetic: the muted grey is doing work that bold would
#: normally do, so it cannot be treated as decoration and must keep its contrast.
TITLE = FontSpec(16, True, "title", TEXT)
LABEL = FontSpec(11, False, "sentence", TEXT)
HELPER = FontSpec(10, False, "sentence", MUTED)
BUTTON_LABEL = FontSpec(11, False, "title", TEXT)
ERROR_TEXT = FontSpec(11, False, "sentence", ERROR)

#: The header-strip readout — `DAPI removed by the veto  4.7%`. Same weight as a label; it sits
#: above the canvas, right-aligned, because **no text is ever drawn over image data** (§6f.8).
READOUT = FontSpec(11, False, "sentence", TEXT)

#: Stain names keep their capitals always. They are names, not prose, and `Cd68` or `iba1` is
#: simply wrong rather than merely inconsistent. §6e.2.
STAIN_NAMES = ("CD68", "6E10", "DAPI", "Iba1")

_LOWER_TO_STAIN = {name.lower(): name for name in STAIN_NAMES}

#: Words that stay lower-case inside a Title Cased string. Deliberately short — this is a UI
#: with titles like `Plaque DAPI Core Veto`, not prose.
_MINOR_WORDS = frozenset({"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or",
                          "the", "to", "vs"})


def title_case(text: str) -> str:
    """Title Case that never mangles a stain name.

    ``str.title()`` turns ``CD68`` into ``Cd68`` and ``6E10`` into ``6E10`` only by luck, so it
    cannot be used on any string in this interface. The first word is always capitalised even
    when it is a minor word.

    >>> title_case("plaque DAPI core veto")
    'Plaque DAPI Core Veto'
    >>> title_case("CD68 threshold")
    'CD68 Threshold'
    """
    words = text.split()
    out = []
    for i, word in enumerate(words):
        stain = _LOWER_TO_STAIN.get(word.lower())
        if stain is not None:
            out.append(stain)
        elif i and word.lower() in _MINOR_WORDS:
            out.append(word.lower())
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


# --------------------------------------------------------------------------
# Contrast — the numbers §6e.3 records, computable rather than transcribed
# --------------------------------------------------------------------------

def relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance of an ``#rrggbb`` string."""
    raw = hex_colour.lstrip("#")
    channels = []
    for offset in (0, 2, 4):
        value = int(raw[offset:offset + 2], 16) / 255.0
        channels.append(value / 12.92 if value <= 0.04045
                        else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two ``#rrggbb`` strings.

    Here so the palette's own claims are **derived rather than transcribed** — the rule
    `EXPORT_SPEC.md` §5.1 taught this project the expensive way, where 37 columns were written
    where 72 were advertised. `tests/test_ui_theme.py` asserts every ratio §6e.3 states.
    """
    high, low = sorted((relative_luminance(foreground), relative_luminance(background)),
                       reverse=True)
    return (high + 0.05) / (low + 0.05)


# --------------------------------------------------------------------------
# The typeface — §6e.1
# --------------------------------------------------------------------------

class FontNotBundled(RuntimeError):
    """The bundled typeface is missing, and the interface refuses to substitute silently.

    **This exception is the entire point of bundling.** Qt substitutes a font it cannot find
    without saying so, and the layout shifts — a failure that appears only on the cluster, not
    on the machine the screens were drawn on. Raising here converts an invisible, late,
    machine-dependent failure into a loud one at start-up.
    """


FONT_DIR = Path(__file__).parent / "fonts"
FONT_FAMILY = "IBM Plex Sans"

#: The interface uses all four — `µm`, `µm³`, `≥`, and `→` in the popups. Plex carries them,
#: but the check belongs in the build, because **a missing glyph renders as a box** and nothing
#: else announces it. §6e.1.
REQUIRED_GLYPHS = ("µ", "³", "≥", "→")


def load_bundled_font(*, strict: bool = True) -> str:
    """Register the bundled typeface and return the family name Qt actually resolved.

    ``strict=False`` downgrades a missing bundle to the platform default and is for tests and
    screenshots only. **The application calls this with the default**: §6e.1 chose Plex over
    DejaVu Sans — which is on every Linux and would have cost nothing — specifically because
    Plex distinguishes ``0``/``O`` and ``1``/``l``/``I``, and this interface shows filenames
    like ``1Hippocampus2`` and animal IDs like ``691619``. Falling back to a face without that
    property loses the thing the choice bought.
    """
    from qtpy.QtGui import QFontDatabase

    files = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf"))
    if not files:
        if strict:
            raise FontNotBundled(
                f"no font files in {FONT_DIR}. {FONT_FAMILY} ships with the application "
                "precisely so Qt cannot substitute one silently — see COMPONENT_03_PLAN §6e.1. "
                "Add the IBM Plex Sans Regular and Bold files, or pass strict=False to accept "
                "the platform default and a shifted layout.")
        return ""

    families: list[str] = []
    for path in files:
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id == -1:
            raise FontNotBundled(f"Qt refused to load {path.name}; the file is not a usable font")
        families.extend(QFontDatabase.applicationFontFamilies(font_id))

    if FONT_FAMILY not in families:
        raise FontNotBundled(
            f"the bundled files registered as {sorted(set(families))!r}, not {FONT_FAMILY!r}. "
            "The stylesheet names the family literally, so a mismatch substitutes silently.")
    return FONT_FAMILY


def missing_glyphs(family: str = FONT_FAMILY) -> list[str]:
    """Which of `REQUIRED_GLYPHS` the given family cannot draw. Empty is the passing case."""
    from qtpy.QtGui import QFont, QFontMetrics, QRawFont

    raw = QRawFont.fromFont(QFont(family))
    metrics = QFontMetrics(QFont(family))
    missing = []
    for char in REQUIRED_GLYPHS:
        supported = raw.supportsCharacter(char) if raw.isValid() else metrics.inFont(char)
        if not supported:
            missing.append(char)
    return missing


# --------------------------------------------------------------------------
# The Qt stylesheet
# --------------------------------------------------------------------------

def stylesheet(family: str = FONT_FAMILY) -> str:
    """The whole chrome as one QSS string, applied once to the top-level window.

    Everything a screen needs is here, so no screen sets a colour of its own — which is what
    keeps `#2e323a` meaning *popup* and stops the muted grey drifting toward decoration.

    **Disabled controls keep their surface and drop only their text**, per §6i.7, and the state
    is always said in words as well (`(disabled until first commit)`) — greyed text for
    everyone, a sentence for anyone who cannot separate the two greys. That redundancy is what
    lets the visual half be subtle.
    """
    return f"""
    QWidget {{
        background: {WINDOW};
        color: {TEXT};
        font-family: "{family}";
        font-size: {LABEL.size_pt}pt;
        font-weight: normal;
    }}
    QLabel[role="title"] {{
        font-size: {TITLE.size_pt}pt;
        font-weight: bold;
        color: {TITLE.colour};
    }}
    QLabel[role="helper"] {{ font-size: {HELPER.size_pt}pt; color: {MUTED}; }}
    QLabel[role="error"]  {{ font-size: {ERROR_TEXT.size_pt}pt; color: {ERROR}; }}
    QLabel[role="warning"] {{ color: {WARNING}; }}
    QLabel[role="readout"] {{ font-size: {READOUT.size_pt}pt; color: {TEXT}; }}
    QLabel:disabled {{ color: {DISABLED_TEXT}; }}

    QPushButton {{
        background: {BUTTON};
        color: {TEXT};
        border: none;
        padding: 5px 14px;
        font-size: {BUTTON_LABEL.size_pt}pt;
    }}
    QPushButton:hover   {{ background: {DISABLED_TEXT}; }}
    QPushButton:disabled {{ background: {CONTROL}; color: {DISABLED_TEXT}; }}

    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
        background: {CONTROL};
        color: {TEXT};
        border: none;
        padding: 3px 6px;
        selection-background-color: {BUTTON};
    }}
    QLineEdit:disabled, QComboBox:disabled,
    QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background: {CONTROL};
        color: {DISABLED_TEXT};
    }}
    QComboBox QAbstractItemView {{
        background: {CONTROL};
        color: {TEXT};
        selection-background-color: {BUTTON};
        border: none;
    }}

    QCheckBox {{ spacing: 6px; }}
    QCheckBox:disabled {{ color: {DISABLED_TEXT}; }}

    /* **The checked radio was INVISIBLE before this** — found by looking at rich review, where
       the three rendering modes showed two empty circles and one blank space, so the mode
       actually in use was the one you could not see. Nothing in the sheet named
       `QRadioButton`, so the indicator came from the platform style and its checked dot is
       drawn in a colour that only works on a light background.

       Painted from the same three chrome colours as everything else: `CONTROL` for the
       surface, `MUTED` for the ring, `TEXT` for the dot. §6e.4 holds — the mark carries the
       state, not the hue. */
    QRadioButton {{ spacing: 6px; }}
    QRadioButton:disabled {{ color: {DISABLED_TEXT}; }}
    QRadioButton::indicator {{
        width: 12px;
        height: 12px;
        border-radius: 7px;
        border: 1px solid {MUTED};
        background: {CONTROL};
    }}
    QRadioButton::indicator:checked {{
        border: 1px solid {TEXT};
        background: {TEXT};
    }}
    QRadioButton::indicator:disabled {{
        border: 1px solid {DISABLED_TEXT};
        background: {CONTROL};
    }}

    QSlider::groove:horizontal {{ background: {SLIDER_GROOVE}; height: 4px; }}
    QSlider::sub-page:horizontal {{ background: {SLIDER_FILL}; height: 4px; }}
    QSlider::handle:horizontal {{
        background: {SLIDER_HANDLE};
        width: 10px;
        margin: -5px 0;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal:disabled {{ background: {DISABLED_TEXT}; }}

    QListWidget, QTableWidget, QTreeWidget {{
        background: {CONTROL};
        color: {TEXT};
        border: none;
        outline: none;
    }}
    QListWidget::item:selected, QTableWidget::item:selected {{
        background: {LIST_SELECTION};
        color: {TEXT};
    }}

    QScrollBar:vertical {{ background: {WINDOW}; width: 12px; margin: 0; }}
    QScrollBar:horizontal {{ background: {WINDOW}; height: 12px; margin: 0; }}
    QScrollBar::handle {{ background: {SCROLLBAR_THUMB}; min-height: 24px; min-width: 24px; }}
    QScrollBar::handle:hover {{ background: {SCROLLBAR_THUMB_HOVER}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: {WINDOW}; }}

    QDialog, QWidget[role="popup"] {{
        background: {POPUP};
        border: 1px solid {POPUP_BORDER};
    }}
    QToolTip {{
        background: {POPUP};
        color: {TEXT};
        border: 1px solid {POPUP_BORDER};
        padding: 4px 6px;
    }}
    """
