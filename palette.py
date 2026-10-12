"""The data palette — every colour that lands on image data. `COMPONENT_03_PLAN.md` §6f.

**Separate from `theme.py` on purpose.** That module is §6e, the chrome: window, text, controls,
buttons, popups. This is §6f and its descendants — the ten colours a picker offers, which hue
each channel takes, how much of an additive sum each is entitled to, and the two label schemes.
The documents are split the same way and the next person reads them before reading this.

**Neither module imports Qt or napari.** `widgets.py` needs the ten hex values to draw a swatch
grid and `viewer.py` needs them to build colormaps; if they lived in `viewer.py` the picker
would require napari to draw a square.

Three things here are program-wide rather than per-screen, and that is the whole point:

    the ten colours       every picker on every screen offers the same ten
    the channel hues      6E10 magenta · Iba1 white · CD68 green · DAPI blue
    the default alphas    a property of the channel's ROLE, not of the screen
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DataColour", "DATA_COLOURS", "COLOUR_BY_NAME",
    "RED", "ORANGE", "YELLOW", "GREEN", "SPRING_GREEN",
    "CYAN", "BLUE", "VIOLET", "MAGENTA", "WHITE",
    "CHANNEL_HUE", "CHANNEL_ALPHA", "DERIVED_MASK_ALPHA",
    "LABEL_RING", "ring_colour", "MUTED_LABEL_ALPHA", "WARM_SIX", "COOL_SIX",
    "THIS_CELL_ALPHA", "OTHER_PROCESS_ALPHA",
    "SIZE_RAMP", "SIZE_RAMP_RANGE", "KEPT_ALPHA", "REMOVED_ALPHA", "size_ramp_colours",
    "SELECTED_FILL", "BORDER_OUTLINE",
    "lighten",
    "SWATCH_CELL_PX", "SWATCH_PX", "SWATCH_COLUMNS", "SWATCH_ROWS",
]


# --------------------------------------------------------------------------
# The ten — §6f.2
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DataColour:
    """One entry in every picker on every screen.

    ``min_relative_luminance`` is on black and is the **minimum** across normal vision,
    deuteranopia and protanopia — the figure that decides whether a dichromat can see the layer
    at all, which is a different question from whether they can tell it from its neighbour.
    """

    name: str
    hex: str
    min_relative_luminance: float


RED = DataColour("red", "#ff3b30", 0.04)
ORANGE = DataColour("orange", "#ff9f0a", 0.33)
YELLOW = DataColour("yellow", "#ffd60a", 0.63)
GREEN = DataColour("green", "#32d74b", 0.04)
SPRING_GREEN = DataColour("spring green", "#30d5a8", 0.07)
CYAN = DataColour("cyan", "#40d8f0", 0.14)
BLUE = DataColour("blue", "#4d7cff", 0.17)
VIOLET = DataColour("violet", "#a86bff", 0.20)
MAGENTA = DataColour("magenta", "#ff5cd6", 0.14)
WHITE = DataColour("white", "#ffffff", 1.00)

#: **Hue-wheel order, red → white**, and the order is load-bearing rather than tidy: a user
#: hunting *"something bluer"* moves rightward rather than searching. Laid out 5 × 2, so the
#: grid is scannable instead of readable.
#:
#: **We define these rather than picking from napari's thirty**, for two measured reasons.
#: Eight of napari's map value 0 to something that is not black — `I Blue`, `I Bordeaux`,
#: `I Forest`, `I Orange`, `I Purple` start at `#ffffff`; `ice` and `HiLo` at `#0000ff`; `fire`
#: at `#ffffff` — and under additive stacking **the empty part of such a layer paints the
#: screen**. And the pure primaries are too dim on black: `#0000ff` is relative luminance 0.072.
#:
#: The raised endpoints do not remove the collisions and cannot — with ten options on a wheel,
#: `orange`/`yellow` and `cyan`/`blue` will always be close for a dichromat. What they buy is
#: that pairs stop being **identical** and become merely **similar**, so a user who cannot
#: separate two can pick a third. **The defaults must be safe; the picker only has to be usable.**
DATA_COLOURS: tuple[DataColour, ...] = (
    RED, ORANGE, YELLOW, GREEN, SPRING_GREEN,      # row 1
    CYAN, BLUE, VIOLET, MAGENTA, WHITE,            # row 2
)

COLOUR_BY_NAME = {colour.name: colour for colour in DATA_COLOURS}

#: §6f.5's geometry. 24 px because below that a swatch is fiddly on the laptop trackpad this
#: tool is for; 16 px painted inside it, leaving the 1 px gap that makes the white swatch
#: readable against its own selection ring.
SWATCH_CELL_PX = 24
SWATCH_PX = 16
SWATCH_COLUMNS = 5
SWATCH_ROWS = 2


# --------------------------------------------------------------------------
# The channel convention — §6f.4
# --------------------------------------------------------------------------

#: One hue per channel, program-wide. A colour means the same thing on every screen it appears
#: on, which is what makes it possible to learn the interface once.
#:
#: **`DAPI removed` is not a channel — it is a *state* of one**, and it takes its own colour
#: rather than its channel's. That precedent is what settles item 9a's `CD68 ≥ T` as orange
#: rather than green.
#:
#: The one accepted collision: **6E10 magenta and DAPI blue are the same colour for a
#: protanope**, ΔE 12.9 — and would have been 0.0 on the pure primaries. Three non-hue channels
#: separate them: 0.45 alpha against 1.0, amyloid being a diffuse smear where nuclei are compact
#: blobs, and retained DAPI sitting mostly outside the veto outline.
CHANNEL_HUE: dict[str, DataColour] = {
    "6E10": MAGENTA,
    "Iba1": WHITE,
    "CD68": GREEN,
    "DAPI": BLUE,
    "DAPI removed": RED,
}

#: **A default alpha is a property of the channel's ROLE, not of the screen.** Under additive
#: blending the translucency slider is a *weight*, so this says how much of the sum a channel is
#: entitled to. Chosen per screen until item 5, which let one channel mean two brightnesses.
#:
#: **Item 9b takes the one anticipated exception**, raising 6E10 to 0.60 where it stops being
#: context and becomes the subject. That is settled on that screen, not here.
CHANNEL_ALPHA: dict[str, float] = {
    "6E10": 0.45,
    "Iba1": 0.40,
    "CD68": 0.60,
    "DAPI": 1.00,
}

#: An outline is a thin line and needs all of its brightness.
DERIVED_MASK_ALPHA = 1.0


# --------------------------------------------------------------------------
# The label ring — §6h.3, and its split at §3d.2
# --------------------------------------------------------------------------

#: **Twelve hues built in CIELAB at constant lightness**, L* 68 / C* 38 — all in gamut, all at
#: relative luminance 0.380, spread 1.00×. Worst pair anywhere in the cycle 19.5.
#:
#: The point is not prettier colour. **It is that lightness becomes free to carry exactly one
#: thing**, which is what makes the muted class readable. napari's own `label_colormap()` scores
#: better on hue — worst pair 11.9 across its first 40 — but its 128 labels span relative
#: luminance 0.008 to 0.877, and a quarter of them are already dimmer than our muted class, so
#: *dimmer means not microglial* would be false for a quarter of the palette.
#:
#: **You cannot fix label collisions by adding hues.** Worst pair in an evenly spaced wheel:
#: 12 → 17.3, 16 → 11.4, 20 → 3.9, 32 → 1.8. Past sixteen every pair gets worse.
LABEL_RING: tuple[str, ...] = (
    "#e68ba7", "#e78e85", "#d8986b", "#bca460", "#98ae69", "#6fb582",
    "#3eb8a5", "#00b7c7", "#31b2e0", "#73a9ea", "#a99de1", "#d191c9",
)

#: Item 4's *not microglial* class. **Alpha only, never desaturation.** Desaturating attacks the
#: very property that must survive — muted staying per-label distinct — while alpha attacks
#: brightness, which is the property being changed. Every hue is 0.380 at full strength, so
#: *dim* has exactly one meaning on that layer.
#:
#: > ### 0.6 WAS NOT A BIG ENOUGH DIFFERENCE, AND A REVIEWER SAID SO — 2026-08-20
#: >
#: > *"the microglial nuclei don't really look highlighted... the other nuclei should be way
#: > dimmer."* The old value was chosen to keep the muted class **per-label distinct**, costing
#: > ΔE 19.5 → 16.4 rather than desaturation's 13.6 — and that measurement was answering the
#: > wrong question. Item 4 asks *is this nucleus microglial*, one class against the other. Which
#: > individual rejected nucleus is which is not a judgement anyone makes on that screen, so
#: > protecting it was spending the whole contrast budget on something nobody reads.
#: >
#: > **0.25 rather than rich review's 0.30**, because there the muted class is still being worked
#: > with — up to twenty-nine processes competing, one entered — whereas here it is context the
#: > eye should skip over. Both are one constant; if it is still too subtle, lower it.
MUTED_LABEL_ALPHA = 0.25


#: **The order the ring is WALKED, and it is not 0,1,2,3.** `LABEL_RING` is twelve hues evenly
#: spaced at constant lightness, so **consecutive entries are the most similar pair it has** —
#: 30 degrees apart on the wheel. Walking it sequentially gives the first four objects four
#: adjacent hues spanning one warm quadrant, which is why a field of four objects read as two
#: colours. The user, 2026-08-20: *"objects detected 4 but across 40 slices i only see 2
#: different hues."*
#:
#: **And it was the worst possible pairing, not merely a weak one.** `ndimage.label` numbers
#: components in raster-scan order, so **spatially adjacent objects tend to get consecutive
#: ids** — and consecutive ids were being handed the closest available hues.
#:
#: This visits the wheel in halves, then quarters: 0, 6, 3, 9, ... Any two low ids land far
#: apart, the full cycle still uses all twelve exactly once, and **no colour changes** — only
#: the order they are claimed in. Affects item 3's objects and item 4's nuclei alike.
RING_ORDER: tuple[int, ...] = (0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11)


def ring_colour(label_id: int) -> str:
    """§6h.3's twelve hues, assigned by id and cycling.

    **Not §6f.2's ten**, and they are in no picker. Those ten are choices a user makes for a
    single-hue layer; these are a scheme, assigned automatically. All twelve sit at relative
    luminance 0.380 — a spread of 1.00x — which is the whole reason they were built in CIELAB
    rather than taken from a wheel: **lightness is freed to carry exactly one thing.**

    **It lives here rather than on item 4 because item 3 now uses it too**, for the objects its
    committed mask decomposes into. A second copy of a twelve-entry cycle is exactly the kind of
    duplicate that drifts.
    """
    if label_id < 1:
        raise ValueError(f"label ids start at 1, got {label_id}")
    return LABEL_RING[RING_ORDER[(label_id - 1) % len(RING_ORDER)]]

#: **The ring splits in half for rich review** — the decomposition takes the warm six and the
#: nucleus layers get the cool side to sit in without competing. The split costs nothing because
#: six already exceeds what the graph colouring needs.
WARM_SIX: tuple[str, ...] = LABEL_RING[:6]
COOL_SIX: tuple[str, ...] = LABEL_RING[6:]

#: Rich review: the entered process at full strength, every other process muted. **3.3×, and
#: stronger than item 4's 0.6 deliberately** — item 4 separates two classes, here up to
#: twenty-nine regions compete and only one is being worked on.
THIS_CELL_ALPHA = 1.0
OTHER_PROCESS_ALPHA = 0.30


# --------------------------------------------------------------------------
# The size ramp — §6j.1, §6j.2
# --------------------------------------------------------------------------

#: > ### `jet`, AND IT OVERRIDES §6d — THE USER'S DECISION, 2026-08-21
#: >
#: > This was `cividis`, picked by §6d's own criterion: that section **rejects `jet` because its
#: > lightness is not monotonic for deuteranopes**, and `cividis` is monotonic under normal
#: > vision, deuteranopia and protanopia. That reasoning has not been refuted — it has been
#: > outweighed.
#: >
#: > **3DMorph uses `jet` on this exact screen.** `CellSizeCutoffGUI.m:84` builds
#: > `jet(max(fullimg(:)))` over objects renumbered smallest-first — its own comment reads
#: > *"flip upside down so small is plotted first in blue"* — with the background forced black
#: > and a colorbar labelled `Small` to `Large`. Matching it means a reader who knows 3DMorph
#: > reads this screen without translating, which is worth more to this study than dichromatic
#: > monotonicity on a screen whose judgement is also carried by the list, the `!` and the
#: > readout. **The cost is recorded here rather than lost.**
SIZE_RAMP = "jet"

#: **Rank maps onto the middle three-quarters of the ramp, and with `jet` BOTH ends need it.**
#:
#: Measured: full `jet` runs `#000080` at relative luminance **0.016** to `#800000` at **0.046**
#: — near-black at both ends, on a black canvas. `cividis` needed only its floor lifted, because
#: it tops out at bright yellow.
#:
#: Over `(0.125, 0.875)` the ends become `#0000ff` and `#ff1e00` — **pure blue to pure red**, at
#: 0.072 and 0.222. The floor is 4.5x lighter, and blue-for-small is 3DMorph's own intent rather
#: than a compromise on it. **The cells this screen adjudicates are the small ones**, so what
#: the floor does is the one thing this ramp cannot get wrong.
SIZE_RAMP_RANGE = (0.125, 0.875)

#: Item 6's two alpha states. **A flat 3.0× at every rank, because alpha multiplies** — no
#: hue-based encoding is rank-independent like that. Selection is *not* a third alpha state: it
#: is a white fill, because 0.75 → 1.0 is 1.33× and at the dark end that is relY 0.019 against
#: 0.014, which is not a channel.
KEPT_ALPHA = 0.75
REMOVED_ALPHA = 0.25

#: **Selection is a white FILL, not a third alpha state.** §4 had *"alpha is one ordinal channel
#: with three states"*; measured, kept 0.75 -> selected 1.0 is **1.33x**, and at the dark end of
#: the ramp that is relative luminance 0.019 against 0.014 — both invisible on black. **A 33%
#: change to a number already near zero is not a channel.** As white the same cell runs 17.2x at
#: the smallest rank and 7.3x at the cutoff.
#:
#: **`jet` never reaches white either** — its lightest stop is `#7dff7a` at relative luminance
#: 0.773 — so a white fill cannot be mistaken for a ramp stop at any rank. That held for
#: `cividis`, whose top was `#fee838`, and it still holds. §4 is corrected in place; alpha now
#: has two states.
SELECTED_FILL = "#ffffff"

#: A border-touching cell is a white **outline** around its ramp colour — same colour, different
#: form, which is the split settled at items 2, 4 and 5 and costs the palette nothing.
BORDER_OUTLINE = "#ffffff"


def size_ramp_colours(n: int) -> tuple[str, ...]:
    """``n`` hex colours along `SIZE_RAMP` over `SIZE_RAMP_RANGE`, **smallest rank first**.

    **Sampled from matplotlib rather than transcribed.** A hand-copied LUT is the failure
    `EXPORT_SPEC.md` §5.1 records in another form — it rots beside its authority and nothing
    says so. matplotlib is imported lazily, exactly as Qt is in `theme`, so a test that checks a
    contrast ratio still pays nothing.

    **A single cell takes the TOP of the ramp**, not the floor. With one cell there is no rank to
    encode, and putting it at 0.25 would draw the only object on screen at the dimmest value the
    ramp offers.
    """
    if n <= 0:
        return ()
    from matplotlib import colormaps

    ramp = colormaps[SIZE_RAMP]
    low, high = SIZE_RAMP_RANGE
    positions = [high] if n == 1 else [low + (high - low) * i / (n - 1) for i in range(n)]
    return tuple(
        "#{:02x}{:02x}{:02x}".format(*(int(round(channel * 255)) for channel in ramp(p)[:3]))
        for p in positions
    )


def lighten(hex_colour: str, amount: float) -> str:
    """Blend a hex colour toward white. ``0.0`` unchanged, ``1.0`` pure white.

    **Added 2026-08-27 for item 7b's `other cells`**, where soma and processes had shared one
    hue and differed only in **alpha** — and alpha is also what that layer's translucency slider
    drives, so the control fought the encoding it controlled. The user's decision was to keep the
    alpha difference **and** add lightness beside it, so the soma is *"obviously lighter"* at the
    same reduced alpha.

    **Lightness is the right second channel.** It is the one dichromacy leaves intact — the same
    argument that gives the soma its white in `region_colours` — and it is independent of alpha,
    so the slider can move without erasing the distinction.

    > ### AT A REDUCED ALPHA THIS CHANGES SATURATION MORE THAN BRIGHTNESS
    >
    > On a dark canvas the composited luminance is roughly ``alpha x colour``, so a soma at
    > ``0.33`` alpha stays **dimmer overall** than a process at full alpha however far it is
    > lightened — the lightening would have to roughly treble the hue's luminance to win, which
    > is impossible for an already-light hue.
    >
    > **What it does deliver is a pale, desaturated tint against a solid saturated one**, which
    > is a real and separable channel. It is recorded here so the effect is judged on screen for
    > what it is rather than for a brightness inversion it cannot produce.

    Raises on anything that is not a 6-digit hex, because a silently-passed colour name is
    exactly the `show_labels` defect of 2026-08-25 — ``int("gr", 16)`` — in a new place.
    """
    value = str(hex_colour).lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected a 6-digit hex colour like '#aabbcc', got {hex_colour!r}")
    amount = max(0.0, min(1.0, float(amount)))
    return "#{:02x}{:02x}{:02x}".format(
        *(int(round(channel + (255 - channel) * amount))
          for channel in (int(value[i:i + 2], 16) for i in (0, 2, 4))))
