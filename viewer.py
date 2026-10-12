"""The napari layer, behind a Protocol. `COMPONENT_03_PLAN.md` §4a.7, §6f.3, §6f.6, §6h.6.

**The Protocol is for testing, and the reason is not "napari might be missing".** Cellpose *is*
installed here and still sits behind `Detector` — the durable reason is that the suite runs
headless, and **no test of "which screen follows which" should have to construct a Qt window or
find a display.** `RecordingViewer` below satisfies the Protocol with no napari at all, which
keeps the flow logic — screen order, what a nucleus edit does, when the cache is consulted —
testable permanently.

**napari is imported late, inside `NapariViewer`**, exactly as `nuclei.py:182` imports cellpose
*"imported late: heavy, and optional"*. Importing this module costs nothing.

Two mechanical facts drive nearly everything here:

    every image layer is 'additive'      so layers add rather than occlude, and
                                         stacking order stops mattering
    Labels stay 'translucent', on top    the one exception, and it is not a style choice
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from scipy import ndimage

from . import palette
from .preview import z_position

__all__ = [
    "Click", "ViewerPort", "RecordingViewer", "NapariViewer", "CanvasLayout",
    "hue_colormap", "outline_2d", "outline_of_batch", "strip_default_mouse_bindings",
    "NAPARI_DEFAULT_GESTURES", "batch_projection", "batch_plane", "cached_plane",
    "zoom_on_pinch", "install_pinch_zoom", "PINCH_ZOOM_PER_STEP", "OUTLINE_WIDTH_PX",
    "IMAGE_RENDERING", "LABELS_RENDERING", "OFFERED_RENDERING", "DEFAULT_RENDERING",
]

#: napari's seven, verified against 0.7.1 rather than transcribed from the docs.
IMAGE_RENDERING: tuple[str, ...] = (
    "translucent", "additive", "iso", "mip", "minip", "attenuated_mip", "average",
)

#: **`Labels` accepts two of them and neither is `mip`.** So Component 5's rendering control
#: cannot reach the `this cell` layer even in principle, and that is a napari fact rather than a
#: screen decision — the radio governs the volume layers, and the labels keep `translucent`.
LABELS_RENDERING: tuple[str, ...] = ("translucent", "iso_categorical")

#: > ### ONE MODE IS OFFERED, AND IT IS `attenuated_mip` — the user's decision, 2026-09-11
#: >
#: > `COMPONENT_05_PLAN.md` §3c offered **three of the seven** and made `translucent` the
#: > default, because rich review is the screen where depth *is* the judgement — whether two
#: > things are genuinely pressed together — and `mip`, napari's own default, *"reports only
#: > the single brightest voxel it passes"* and has no depth at all.
#: >
#: > **`translucent` is the most expensive of napari's modes**: full alpha-composited
#: > ray-casting, over several overlapping volumes, on an **A16** — a virtualised datacentre
#: > GPU, not a workstation card. Entering a cell took 30-60 s and the render is the bulk of
#: > it. `attenuated_mip` is brightest-voxel **dimmed with distance**, so it keeps the depth
#: > cue the choice of `translucent` was made for while costing a fraction of it.
#: >
#: > **The offer is one mode, so there is no control.** A radio group with a single button is
#: > a control that cannot be used; `screen_edit_cell` draws none. `set_rendering` still
#: > exists and still refuses anything not offered, which is what keeps this line the decision
#: > rather than a default someone can drift off.
#:
#: **PrestoCell inherits `mip` without knowing it.** It never sets `rendering`; what it does set
#: is `blending='additive'`, which is a different property entirely — how *layers* composite
#: with each other rather than how each volume is projected. §4a asked for this check because
#: the two names invite exactly that confusion.
OFFERED_RENDERING: tuple[str, ...] = ("attenuated_mip",)
DEFAULT_RENDERING = "attenuated_mip"

#: §6f.6. Thick enough to be unmistakable at 2.6x the local brightness, which is the worst case
#: measured — a magenta line on a saturated plaque core. The common case is 7.8x.
OUTLINE_WIDTH_PX = 2


# --------------------------------------------------------------------------
# How the one canvas is arranged for a screen — task 23
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CanvasLayout:
    """**The panel arrangement is a napari fact, not a Qt one**, so it is declared here.

    `frames.py` says it twice already — *"the two-panel-ness is not a Qt fact"* for `FrameB`,
    *"the stacking is a napari fact rather than a Qt one"* for `FrameD`. Both panels of a
    two-panel screen live in **one** canvas at `grid.shape`, so what Qt contributes is only the
    header strips. This is the value that says which arrangement, and the flow driver applies it
    when the canvas arrives on a screen.

    **It is applied on arrival rather than set once in a constructor**, and that is the whole
    reason it exists as data: grid and `ndisplay` are properties of the **viewer**, which is
    shared, while a screen is built once and shown many times. Rich review left `ndisplay=3`
    behind it and nothing put it back — items 9a and 9b would have rendered in 3D for the rest
    of the session. Declaring the arrangement per screen and applying it on every arrival makes
    that class of leak impossible rather than fixed once.

    ``stride`` is how many consecutive layers share a panel. Item 5 has six layers over two
    panels, so its stride is 3; item 6 has three over two, stride 2.
    """

    #: `(rows, columns)`. `(1, 1)` is the ordinary single view and disables the grid.
    shape: tuple[int, int] = (1, 1)
    stride: int = 1
    #: **2 everywhere except rich review**, which is `ndisplay=3` only — there is no slice mode.
    ndisplay: int = 2

    @property
    def gridded(self) -> bool:
        return self.shape != (1, 1)


# --------------------------------------------------------------------------
# Colormaps — §6f.2
# --------------------------------------------------------------------------

def hue_colormap(colour: palette.DataColour | str):
    """A black-to-hue colormap with the raised endpoint, built by us rather than taken.

    **Eight of napari's thirty map value 0 to something that is not black** — `I Blue`,
    `I Bordeaux`, `I Forest`, `I Orange` and `I Purple` start at `#ffffff`; `ice` and `HiLo` at
    `#0000ff`; `fire` at `#ffffff`. **Under additive stacking the empty part of such a layer
    paints the screen**, and every layer in this interface is mostly empty. That is mechanical,
    not aesthetic, and it is why the usable set was ten before anyone considered taste.

    The endpoint is raised because the pure primaries are too dim on black: `#0000ff` is
    relative luminance 0.072, and a dim blue layer on a black canvas is nearly not there.
    """
    from napari.utils.colormaps import Colormap

    if isinstance(colour, str):
        colour = palette.COLOUR_BY_NAME[colour]
    raw = colour.hex.lstrip("#")
    end = [int(raw[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return Colormap(colors=[[0.0, 0.0, 0.0, 1.0], end + [1.0]],
                    name=f"prestomorph {colour.name}")


# --------------------------------------------------------------------------
# Outlines — §6f.6, and the ordering that must not be rediscovered
# --------------------------------------------------------------------------

def outline_2d(mask: np.ndarray, width: int = OUTLINE_WIDTH_PX) -> np.ndarray:
    """The boundary of a 2D binary mask, `width` pixels thick, drawn inside the mask.

    Inside rather than outside so the line marks the mask's own extent — item 2's whole
    judgement is *containment*, and a line drawn outside would overstate the veto by its own
    thickness at exactly the boundary being judged.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    solid = mask.astype(bool, copy=False)
    if not solid.any():
        return np.zeros_like(solid)
    eroded = ndimage.binary_erosion(solid, iterations=max(1, width),
                                    border_value=0)
    return solid & ~eroded


#: napari's own canvas gestures, registered in `ViewerModel.__init__` and **live alongside
#: anything a screen binds** — the callback lists are appended to, never replaced.
#:
#: > ### ALL THREE WERE REACHING THE USER, AND TWO OF THEM BIT — task 26
#: >
#: > `double_click_to_zoom` ran on **every** double-click that added or deleted a nucleus, so
#: > item 4's edit gesture also zoomed the camera 2x. `dims_scroll` steps the z slider on
#: > `Ctrl`+wheel — and **Windows delivers a trackpad pinch as Ctrl+wheel**, so pinching to
#: > zoom stepped through z instead. `drag_to_zoom` had not been hit yet: it needs `Alt`.
#: >
#: > The user's rule, 2026-08-20: *"other key binds need to be turned off and then only our
#: > discussed key binds need to work."*
#:
#: **Matched by `__name__` rather than by identity**, so this neither imports from a private
#: napari module nor fails closed if the import path moves. What it cannot survive is napari
#: *renaming* the functions, which is why `strip_default_mouse_bindings` returns what it took
#: and `NapariViewer.stripped` keeps it — a test can then assert three, and say so when a
#: version bump makes it fewer.
NAPARI_DEFAULT_GESTURES: frozenset[str] = frozenset({
    "dims_scroll", "double_click_to_zoom", "drag_to_zoom",
})


def strip_default_mouse_bindings(viewer) -> tuple[str, ...]:
    """Remove napari's built-in canvas gestures. Returns the names actually removed."""
    taken: list[str] = []
    for attribute in ("mouse_wheel_callbacks", "mouse_double_click_callbacks",
                      "mouse_drag_callbacks"):
        callbacks = getattr(viewer, attribute, None)
        if callbacks is None:
            continue
        for callback in list(callbacks):
            if getattr(callback, "__name__", "") in NAPARI_DEFAULT_GESTURES:
                callbacks.remove(callback)
                taken.append(callback.__name__)
    return tuple(taken)




#: How much one notch of a pinch changes the zoom. 1.1 is a tenth per step, which is what a
#: trackpad's many small steps want — napari's own double-click zoom uses 2x, and a pinch
#: delivering twenty steps at 2x each would cross the whole zoom range in one gesture.
PINCH_ZOOM_PER_STEP = 1.1


def zoom_on_pinch(viewer, event) -> None:
    """Pinch to zoom. **The camera never sees this event, so we drive it ourselves.**

    > ### WINDOWS SENDS A PINCH AS Ctrl+SCROLL, AND NAPARI BLOCKS THAT PATH — task 26
    >
    > A precision-touchpad pinch is not a gesture as far as the application is concerned: the
    > OS delivers it as a scroll message **with Ctrl held**, the same convention browsers use
    > for zoom. napari's canvas then refuses it outright — `_vispy/canvas.py:97`, *"Ignore mouse
    > wheel events which have modifiers"* — so a modified scroll never reaches the camera at
    > all. It reaches `viewer.mouse_wheel_callbacks` and nothing else.
    >
    > That slot held `dims_scroll`, which is why pinching **stepped through z**. Removing it
    > stopped that and left the gesture doing nothing, because the one path to the camera was
    > closed by napari on purpose. Zooming here is the only way in.
    >
    > **Plain scroll is untouched** and still zooms through the camera, because it carries no
    > modifier and is never filtered. A touchscreen pinch is a different event again —
    > `gesture_zoom`, which vispy raises and the camera handles — and it needs nothing from us.

    The point under the pointer stays put, using napari's own expression for it from
    `double_click_to_zoom` rather than a second version of the same arithmetic.
    """
    if "Control" not in getattr(event, "modifiers", ()):
        return
    delta = float(np.asarray(event.delta).ravel()[1])
    if not delta:
        return

    factor = PINCH_ZOOM_PER_STEP ** delta
    viewer.camera.zoom *= factor
    position = np.asarray(event.position, dtype=float)
    centre = np.asarray(viewer.camera.center, dtype=float)
    if viewer.dims.ndisplay == 3 and viewer.dims.ndim == 3:
        displayed = np.asarray(viewer.dims.displayed)
        viewer.camera.center = centre + (position[displayed] - centre) * (1 - 1 / factor)
    else:
        viewer.camera.center = centre[-2:] + (position[-2:] - centre[-2:]) * (1 - 1 / factor)
    #: Marked handled so nothing downstream zooms a second time on the same gesture.
    event.handled = True


def install_pinch_zoom(viewer) -> bool:
    """Put `zoom_on_pinch` in the slot `dims_scroll` was taken out of."""
    callbacks = getattr(viewer, "mouse_wheel_callbacks", None)
    if callbacks is None or zoom_on_pinch in callbacks:
        return False
    callbacks.append(zoom_on_pinch)
    return True


#: The two ways LEFT/RIGHT can move a z-batch window, and **both already existed** — one in
#: `outline_of_batch`, one in `cache.ProjectionCache`, with no control to choose between them.
ROLLING = "rolling"
BATCHED = "batched"
WINDOW_MODES: tuple[str, str] = (ROLLING, BATCHED)


def window_bounds(z: int, n_slices: int, depth: int, mode: str = ROLLING) -> tuple[int, int]:
    """The slices a z-batch window covers, as `(start, stop)` with `stop` exclusive.

    > ### TWO WINDOWINGS EXISTED AND THE SCREENS SILENTLY USED ONE — task 27
    >
    > `outline_of_batch` took `z` as the window **start**, so an arrow key moved the window by
    > one slice: 1-10, 2-11, 3-12. `cache.ProjectionCache` did the opposite — fixed,
    > non-overlapping batches at `z // depth` — and **no screen ever used it**, so §4a.8's
    > projection precompute has been dead code and every arrow key recomputes.
    >
    > **Neither was chosen; the rolling one was simply what got written.** The user found it by
    > running the program and liked it: *"it shows 10 slices and moves one by one. i think thats
    > good."* This makes it a choice instead of an accident.
    >
    > **Rolling has a real advantage nobody wrote down.** `ProjectionCache`'s own docstring
    > records the fixed scheme's artefact — *"a nucleus occupying slices 8-20 appears partially
    > in two batches and wholly in neither"*. **Rolling cannot have it**, because every window
    > position exists.

    **`BATCHED` snaps to the batch containing `z`** rather than starting at it, so the two modes
    agree at depth 1 and the toggle never moves the picture on its own.

    **The tail differs between them, deliberately.** Rolling clamps its stop to the stack, so
    positions past `n - depth` project a *shrinking* window — which is why the last position
    worth stepping to is `n - depth`, and `last_window_start` is what says so. Batched has no
    such position: its final batch is simply short when `n` is not a multiple of `depth`.
    """
    if mode not in WINDOW_MODES:
        raise ValueError(f"window mode must be one of {WINDOW_MODES}, got {mode!r}")
    depth = max(1, int(depth))
    start = int(z) if mode == ROLLING else (int(z) // depth) * depth
    start = max(0, min(start, max(0, n_slices - 1)))
    return start, min(start + depth, n_slices)


def last_window_start(n_slices: int, depth: int, mode: str = ROLLING) -> int:
    """The furthest position stepping should reach.

    Rolling stops where the window last fills — `n - depth` — because beyond it the projection
    thins out while the indicator would still claim a full batch. Batched stops at the first
    slice of the final batch, however short that batch is.
    """
    depth = max(1, int(depth))
    if n_slices <= 0:
        return 0
    if mode == BATCHED:
        return ((n_slices - 1) // depth) * depth
    return max(0, n_slices - depth)


def step_window(z: int, delta: int, n_slices: int, depth: int, mode: str = ROLLING) -> int:
    """Where LEFT/RIGHT land. **This is what the toggle actually changes.**

    Rolling moves one slice per press whatever the depth. Batched moves a whole batch, so the
    windows tile the stack instead of sliding across it.
    """
    depth = max(1, int(depth))
    stride = 1 if mode == ROLLING else depth
    start, _ = window_bounds(z, n_slices, depth, mode)
    return max(0, min(start + int(delta) * stride, last_window_start(n_slices, depth, mode)))


def z_window_range(z: int, n_slices: int, depth: int, mode: str = ROLLING) -> str:
    """`(23/40)` at depth 1, **`(2-11/40)` above it.**

    > ### THE OLD INDICATOR NAMED ONE SLICE WHILE SHOWING TEN — task 27
    >
    > It read `(23/40)` at every depth, which is a single index for a window covering ten
    > planes. With two window modes now selectable that stops being merely vague and becomes
    > wrong: rolling and batched put **different slices** under the same number.
    >
    > **The range is the only thing on screen that distinguishes the two modes**, which is why
    > it ships with the toggle rather than after it.

    One-based for the reader and the end is inclusive, so `2-11` is ten slices.
    """
    if int(depth) <= 1:
        return z_position(z, n_slices)
    start, stop = window_bounds(z, n_slices, depth, mode)
    return f"({start + 1}-{stop}/{n_slices})"


def z_window_word(depth: int) -> str:
    """`z slice` at depth 1, `z batch` above it — the noun the range belongs to."""
    return "z slice" if int(depth) <= 1 else "z batch"


def z_window_label(z: int, n_slices: int, depth: int, mode: str = ROLLING) -> str:
    """Word and range together, for the screens that draw the indicator as one label."""
    return f"{z_window_word(depth)} {z_window_range(z, n_slices, depth, mode)}"


def batch_projection(volume: np.ndarray, z: int, depth: int,
                     mode: str = ROLLING) -> np.ndarray:
    """The brightest value across a group of slices, or **the volume untouched at depth 1.**

    > ### EVERY LAYER THAT IS NOT A LABEL PROJECTS, AND UNTIL NOW ALMOST NONE DID — task 26
    >
    > `outline_of_batch` projected the derived masks and nothing projected the channels
    > underneath them, so the z-batch box moved an outline over a picture that stayed on one
    > slice. Items 9a and 9b were already right; items 2, 4 and 5 pushed a single plane or the
    > whole volume and ignored `depth` entirely.
    >
    > **Depth 1 returns the 3D volume rather than a plane, and that is deliberate.** napari's
    > own z slider only exists while some layer is 3D, and it is the control the user reaches
    > for first. Projecting unconditionally would have removed it from every screen to serve a
    > mode nobody is in yet. Above depth 1 there is no slice to slide to, so it goes — and it
    > comes back when the box returns to 1.

    Max rather than mean: a projection answers *"is the signal anywhere in this group"*, and a
    mean dilutes a bright punctum across however many empty planes it is averaged with.
    """
    volume = np.asarray(volume)
    if int(depth) <= 1:
        if volume.ndim != 3:
            raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
        return volume
    return batch_plane(volume, z, depth, mode)


def batch_plane(volume: np.ndarray, z: int, depth: int, mode: str = ROLLING) -> np.ndarray:
    """The batch as a **2D plane, at every depth including 1.**

    **The difference from `batch_projection` is the depth-1 case and nothing else**, and which
    one a screen wants follows from how it pushes layers. Items 5, 9a and 9b push planes and own
    z outright, so they have no napari slider to preserve and want a plane at depth 1 too. Items
    2 and 4 push volumes and let napari slice them, so they want the volume back at depth 1.

    Written as two names rather than a flag because the choice is a property of the screen, and
    a boolean at the call site would have to be read back to `viewer.py` to be understood.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
    start, stop = window_bounds(z, volume.shape[0], depth, mode)
    return volume[start:stop].max(axis=0)


def outline_of_batch(mask: np.ndarray, z_start: int, depth: int,
                     width: int = OUTLINE_WIDTH_PX, mode: str = ROLLING) -> np.ndarray:
    """**Project the mask, then outline it — never outline then project.**

    Recorded at §6f.6 so it is not rediscovered. Outlining first and projecting after stacks ten
    slices' boundaries into **nested rings**; projecting first gives the one silhouette the
    containment judgement actually needs.

    Identical at depth 1, and the same total cost either way — which is why this one function is
    correct at every depth and the "slice a pre-outlined volume" shortcut is only ever an
    optimisation for depth 1, never a different answer.
    """
    if mask.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {mask.shape}")
    z_start, stop = window_bounds(z_start, mask.shape[0], depth, mode)
    projected = mask[z_start:stop].any(axis=0)
    return outline_2d(projected, width)


def cached_plane(cache, layer: str, source, z: int, depth: int, mode: str, compute,
                 *, projection: bool = False):
    """One window position's plane, computed once — `cache.WindowCache`, patch item A9.

    **The window semantics live here and not in `cache.py`**, which is the placement rule §4a.7
    already sets: the cache would run headless, the modes are this module's subject. `cache.py`
    therefore knows nothing about `ROLLING` and `BATCHED` and this function chooses between its
    two halves.

        ``projection=True``   the layer is exactly `volume[window].max(0)`, so **batched** mode
                              can be served by `ProjectionCache` — every batch built together on
                              a depth change, stepping then instant
        ``projection=False``  the layer is derived — an outline, a `np.where`, a threshold — so
                              there is nothing for `ProjectionCache` to project and the LRU
                              serves both modes

    **Depth 1 is never cached, deliberately.** A single slice is free; and depth 1 is the one
    place `batch_plane` and `batch_projection` disagree about whether the answer is a plane or
    the volume, so nothing here second-guesses the caller.

    Measured residual per arrow key after items A1-A8, at depth 10 on a full field: items 2 and
    9a ~250 ms, item 5 ~180 ms, item 9b ~90 ms, item 4 ~45 ms. One float32 window projection is
    77.8 ms and one bool `any` is 13.3 ms.
    """
    if cache is None or int(depth) <= 1:
        return compute()
    if projection and mode == BATCHED:
        return cache.batch(layer, source, z, int(depth))
    return cache.plane(layer, source, (int(z), int(depth), mode), compute)


# --------------------------------------------------------------------------
# The port
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Click:
    """One pointer event, in the **canvas's own coordinates**, with napari left behind.

    > ### THE COORDINATE FRAME IS THE TRAP, AND COMPONENT_03_PLAN 4a.1 SAYS SO
    >
    > *"Three coordinate frames are live in this pipeline at once, and they disagree... A point
    > written in one and read as another is off by a factor of two, or by the crop offset - and
    > the resulting number is always plausible, so nothing downstream can catch it."*
    >
    > **`position` is world coordinates**, which is the frame of an unscaled layer. Every screen
    > converts once, at the point of capture, and `screen_nuclei.click_frames` is where item 4
    > does it. Nothing here guesses on a screen's behalf.

    `ray` is the pair napari gives for a click in a 3D view - the start and end of the line
    through the volume under the pointer. It is `None` in a 2D slice view, where the position is
    already the voxel.
    """

    position: tuple[float, ...]
    modifiers: frozenset = frozenset()
    ray: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def shift(self) -> bool:
        return "Shift" in self.modifiers



def _parse_hex(value, *, where: str) -> list[float]:
    """`#rrggbb` -> three floats, or a `ValueError` naming the caller.

    Shared by the real viewer and the recorder **so that they cannot disagree about what is
    acceptable** — the recorder existing to predict napari's answer is worth nothing on any
    question it declines to ask.
    """
    raw = str(value).lstrip("#")
    if len(raw) != 6:
        raise ValueError(
            f"{where}: {value!r} is not a #rrggbb colour. `show_labels` takes a HEX string per "
            f"label id, unlike `show_image`, which takes one of the ten colour NAMES."
        )
    try:
        return [int(raw[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        raise ValueError(
            f"{where}: {value!r} is not a #rrggbb colour. Did you pass a colour NAME? "
            f"`show_image` takes names; `show_labels` takes hex."
        ) from None


def _apply_translate(layer, translate) -> None:
    """Place a layer's origin in world coordinates, in **voxels of its own scale**.

    > ### WHY ANY LAYER NEEDS AN OFFSET AT ALL — item 7b, 2026-08-25
    >
    > Every layer on rich review is a crop of the entered cell's own bounding box, so they are
    > all registered by having the same origin and nobody had to say what it was. The context
    > layers now want to be **wider than the cell**: the whole point of drawing a neighbour is
    > to see the one *beside* this cell, and a box that stops at the cell's own edge is exactly
    > the material it excludes.
    >
    > Widening one layer without an offset would slide it against the other four. `translate` is
    > napari's own answer and costs nothing — the alternative priced in `SHAKEOUT_FINDINGS.md`
    > was handing the editor full-volume arrays, which is the whole loading contract and would
    > widen the **editable** layer too.
    >
    > **Reset to zero rather than skipped when `None`**, because napari keeps `translate` across
    > a `layer.data` assignment: a layer that was once offset and is now not would keep the old
    > offset silently, which is the same class of defect `_apply_scale` records for `scale`.
    """
    layer.translate = ((0.0,) * layer.ndim if translate is None
                       else tuple(float(v) for v in translate))


def _apply_scale(layer, scale) -> None:
    """Set a layer's world scale, and **re-set it on every call rather than only at creation.**

    napari keeps `scale` when `layer.data` is replaced, so once would be enough today - but the
    same was true of `contrast_limits` until a data swap reset them, and this is the argument
    that decides whether a layer lands over the field or over a quarter of it.

    `None` means *native*, which is what every screen but item 4 passes.
    """
    if scale is not None:
        layer.scale = tuple(float(v) for v in scale)



def _while_pushing_layers(method):
    """Suppress napari's dims events **for the duration of a layer push.**

    > ### THE CRASH GOING FROM Z-BATCH 10 BACK TO 1 — task 26
    >
    > Pushing a layer whose dimensionality changed means removing it and adding it again, and
    > adding a 3D layer back into a 2D world **grows napari's dims by one axis**, which emits
    > `dims.events.current_step` *synchronously, from inside the layer insertion*. That reached
    > `_follow`, which called `screen.set_z`, which called `redraw`, which called `draw` — back
    > into `show_image` while napari was still mutating the layer list underneath itself.
    > Re-entrant layer mutation from inside a vispy event is a **segfault, not a traceback**.
    >
    > **It only crashed in one direction, and that is the tell.** Going 1 -> 10 the world loses
    > its third axis, so `on_dims`' own 2D guard already returned early and nothing re-entered.
    > Going 10 -> 1 the axis comes back, the guard passes, and the loop closes.
    >
    > A screen redrawing in response to the slider is still exactly right — it is only *during*
    > a push that the viewer is not in a state to be asked for anything.
    """
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._pushing = getattr(self, "_pushing", 0) + 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._pushing -= 1
    return guarded


@runtime_checkable
class ViewerPort(Protocol):
    """What a screen needs from a viewer, and nothing more.

    Deliberately small. Every method here is something a screen in `COMPONENT_03_PLAN.md` §4
    actually does; anything napari-specific stays inside `NapariViewer` so the flow never
    depends on it.
    """

    #: `translate` places a layer's origin in world coordinates, which is what lets item 7b's
    #: context layers be **wider than the entered cell** without sliding against the four that
    #: are not. See `_apply_translate`.
    def show_image(self, name: str, data: np.ndarray, *, colour: str,
                   alpha: float, visible: bool = True,
                   contrast_limits: tuple[float, float] | None = None,
                   scale: tuple[float, float, float] | None = None,
                   translate: tuple[float, float, float] | None = None) -> None: ...

    def show_labels(self, name: str, data: np.ndarray, *,
                    colours: dict[int, str], alphas: dict[int, float],
                    contour: int = 0, visible: bool = True,
                    scale: tuple[float, float, float] | None = None,
                    translate: tuple[float, float, float] | None = None) -> None: ...

    def show_points(self, name: str, points, *, colour: str, size: float = 6.0,
                    labels=None, visible: bool = True,
                    scale: tuple[float, float, float] | None = None) -> None: ...

    def set_visible(self, name: str, visible: bool) -> None: ...

    def set_colour(self, name: str, colour: str) -> None: ...

    def set_alpha(self, name: str, alpha: float) -> None: ...

    def set_rendering(self, name: str, mode: str) -> None: ...

    def set_ndisplay(self, ndisplay: int) -> None: ...

    def set_grid(self, shape: tuple[int, int], stride: int = 1) -> None: ...

    def remove(self, name: str) -> None: ...

    def clear(self) -> None: ...

    def reset_view(self) -> None: ...

    def layer_names(self) -> list[str]: ...

    # -- gestures, which is task 25 -------------------------------------

    def bind_click(self, callback) -> None: ...

    def bind_double_click(self, callback) -> None: ...

    def bind_key(self, key: str, callback) -> None: ...

    def bind_dims(self, callback) -> None: ...

    def unbind_all(self) -> None: ...

    def focus_canvas(self) -> None:
        """Give the canvas keyboard focus, so a key binding can fire at all."""

    def set_z(self, index: int) -> None:
        """Move the viewer's own z to `index`. **The other half of `bind_dims`.**"""



class RecordingViewer:
    """A `ViewerPort` that draws nothing and remembers everything. No napari, no Qt.

    This is what makes *"which layer is on at arrival"*, *"does the picker recolour the right
    layer"* and *"does the label layer disappear above z-batch depth 1"* testable in the same
    suite that runs on a cluster with no display.
    """

    def __init__(self) -> None:
        self.images: dict[str, dict] = {}
        self.labels: dict[str, dict] = {}
        self.points: dict[str, dict] = {}
        self.order: list[str] = []
        #: 2 until a screen asks for 3. Rich review is the only screen that does, and it asks
        #: once at arrival — §3b: *"`ndisplay=3` only, there is no slice mode."*
        self.ndisplay = 2
        #: The panel arrangement, recorded so a headless test can hold it. `(1, 1)` is one view.
        self.grid_shape: tuple[int, int] = (1, 1)
        self.grid_stride = 1
        #: How many times the camera has been refitted. See `reset_view`.
        self.views_reset = 0
        #: The gestures a screen has bound. **Held rather than ignored**, so the headless suite
        #: can fire a click and watch what the screen does with it - which is the only way the
        #: editing paths are testable without a display.
        self.click_callbacks: list = []
        self.double_click_callbacks: list = []
        self.key_callbacks: dict = {}
        self.dims_callbacks: list = []
        #: How many times the canvas has been asked for keyboard focus. Counted rather than
        #: performed: there is no widget here, and the question a test asks is *did the flow
        #: driver focus the canvas on arrival* — see `NapariViewer.focus_canvas`.
        self.focused = 0
        #: The z the screen last pushed onto the viewer. `None` until it does.
        self.z_index: int | None = None

    def _record(self, name: str) -> None:
        if name not in self.order:
            self.order.append(name)

    def show_image(self, name, data, *, colour, alpha, visible=True, contrast_limits=None,
                   scale=None, translate=None):
        if colour not in palette.COLOUR_BY_NAME:
            raise KeyError(f"{colour!r} is not one of the ten: "
                           f"{sorted(palette.COLOUR_BY_NAME)}")
        self.images[name] = {"data": data, "colour": colour, "alpha": alpha,
                             "visible": visible, "blending": "additive",
                             "contrast_limits": contrast_limits, "scale": scale,
                             "translate": None if translate is None else tuple(translate),
                             "rendering": self.images.get(name, {}).get("rendering")}
        self._record(name)

    def show_labels(self, name, data, *, colours, alphas, contour=0, visible=True, scale=None,
                    translate=None):
        #: **Parsed, not merely stored — 2026-08-25.** `show_image` has always refused a colour
        #: that is not one of the ten; this method accepted anything at all, so a caller passing
        #: a colour NAME where napari wants a **hex string** produced a green suite and a crash
        #: the moment a real viewer drew the layer. That is exactly what happened to item 7b's
        #: `other cells`, and it reached the user rather than a test.
        #:
        #: The recorder's whole purpose is to answer *"would napari have accepted this?"* without
        #: a display. A check napari makes and the recorder does not is a hole in that promise.
        for label_id, hex_colour in colours.items():
            _parse_hex(hex_colour, where=f"show_labels({name!r}) label {label_id}")
        self.labels[name] = {"data": data, "colours": dict(colours),
                             "alphas": dict(alphas), "contour": contour,
                             "visible": visible, "blending": "translucent", "scale": scale,
                             "translate": None if translate is None else tuple(translate)}
        self._record(name)
        #: > ### THE RECORDER DID NOT MODEL `_raise_labels_to_top`, AND IT AGREED BY LUCK
        #: >
        #: > `NapariViewer.show_labels` raises a labels layer to the top of the stack on every
        #: > push — it has to, because `translucent` labels over additive images must composite
        #: > last. The recorder only ever appended in first-push order, which **happened** to
        #: > put `this cell` last because item 7b drew all seven layers in one pass with the
        #: > labels at the end.
        #: >
        #: > Deferring the hidden layers (2026-09-11) broke that coincidence: a layer switched
        #: > on later is pushed after `this cell` has already been recorded, so the recorder
        #: > said the cell was no longer on top while napari had it exactly where it belongs.
        #: > **A recorder that disagrees with napari about layer order is a hole in the one
        #: > promise it makes**, so it models the raise rather than the assertion being relaxed.
        self.order.remove(name)
        self.order.append(name)

    def show_points(self, name, points, *, colour, size=6.0, labels=None, visible=True,
                    scale=None):
        if colour not in palette.COLOUR_BY_NAME:
            raise KeyError(f"{colour!r} is not one of the ten")
        self.points[name] = {"points": [tuple(float(c) for c in p) for p in points],
                             "colour": colour, "size": size,
                             "labels": list(labels) if labels is not None else None,
                             "visible": visible, "scale": scale}
        self._record(name)

    def _layer(self, name: str) -> dict:
        if name in self.images:
            return self.images[name]
        if name in self.labels:
            return self.labels[name]
        if name in self.points:
            return self.points[name]
        raise KeyError(f"no layer named {name!r}; have {self.layer_names()}")

    def set_visible(self, name, visible):
        self._layer(name)["visible"] = visible

    def set_colour(self, name, colour):
        if colour not in palette.COLOUR_BY_NAME:
            raise KeyError(f"{colour!r} is not one of the ten")
        self._layer(name)["colour"] = colour

    def set_alpha(self, name, alpha):
        self._layer(name)["alpha"] = alpha

    def set_rendering(self, name, mode):
        """**Refused on the wrong layer kind rather than silently ignored.**

        napari raises on `layer.rendering = 'mip'` for a `Labels` layer, so a recorder that
        accepted it would let a screen pass a test it would fail in front of a user.
        """
        allowed = LABELS_RENDERING if name in self.labels else IMAGE_RENDERING
        if mode not in allowed:
            raise ValueError(f"{mode!r} is not a rendering mode for {name!r}; "
                             f"expected one of {allowed}")
        self._layer(name)["rendering"] = mode

    def set_ndisplay(self, ndisplay):
        if ndisplay not in (2, 3):
            raise ValueError(f"ndisplay is 2 or 3, got {ndisplay}")
        self.ndisplay = ndisplay

    def set_grid(self, shape, stride=1):
        rows, columns = (int(v) for v in shape)
        if rows < 1 or columns < 1:
            raise ValueError(f"a grid needs at least one row and column, got {shape}")
        if int(stride) < 1:
            raise ValueError(f"stride is at least 1, got {stride}")
        self.grid_shape = (rows, columns)
        self.grid_stride = int(stride)

    def remove(self, name):
        self._layer(name)
        self.images.pop(name, None)
        self.labels.pop(name, None)
        self.order.remove(name)

    def clear(self):
        self.images.clear()
        self.labels.clear()
        self.points.clear()
        self.order.clear()

    def reset_view(self):
        """Counted rather than ignored, so a test can hold *that the camera was refitted*.

        **A lifetime count, and `clear` deliberately does not reset it** — the question a test
        asks is *did arriving at this screen refit the camera*, which is a before-and-after on
        one counter. Zeroing it in `clear` would make every such comparison read `1 > 1`.
        """
        self.views_reset += 1

    def layer_names(self):
        return list(self.order)

    # -- gestures -------------------------------------------------------

    def bind_click(self, callback) -> None:
        self.click_callbacks.append(callback)

    def bind_double_click(self, callback) -> None:
        self.double_click_callbacks.append(callback)

    def bind_key(self, key, callback) -> None:
        self.key_callbacks[str(key)] = callback

    def bind_dims(self, callback) -> None:
        self.dims_callbacks.append(callback)

    def focus_canvas(self) -> None:
        self.focused += 1

    def set_z(self, index: int) -> None:
        self.z_index = int(index)


    def unbind_all(self) -> None:
        """**Called on every arrival**, because there is one viewer and eleven screens.

        Without it item 7b's ray handler would still be attached while item 9a is showing, and
        the click would be answered by a screen the user cannot see - the same failure
        `clear()` fixes for layers, one level up.
        """
        self.click_callbacks.clear()
        self.double_click_callbacks.clear()
        self.key_callbacks.clear()
        self.dims_callbacks.clear()

    # -- firing them, which is what makes a gesture testable with no display

    def click(self, position, *, modifiers=(), ray=None) -> None:
        event = Click(tuple(position), frozenset(modifiers), ray)
        for callback in list(self.click_callbacks):
            callback(event)

    def double_click(self, position, *, modifiers=(), ray=None) -> None:
        event = Click(tuple(position), frozenset(modifiers), ray)
        for callback in list(self.double_click_callbacks):
            callback(event)

    def press(self, key: str) -> None:
        callback = self.key_callbacks.get(str(key))
        if callback is not None:
            callback()

    @contextmanager
    def hold(self, key: str):
        """Hold a key down, as a block. **SHIFT is the one that binds this way.**

        napari runs a generator binding up to `yield` on press and the remainder on release;
        this is the same contract, so a test holds SHIFT exactly the way the user does:

            with viewer.hold("Shift"):
                viewer.click((4, 10, 10))
                viewer.click((4, 20, 20))
        """
        callback = self.key_callbacks.get(str(key))
        generator = callback() if callback is not None else None
        if hasattr(generator, "__next__"):
            next(generator, None)
        try:
            yield
        finally:
            if hasattr(generator, "__next__"):
                try:
                    next(generator)
                except StopIteration:
                    pass

    def move_dims(self, axis: int, value: int) -> None:
        for callback in list(self.dims_callbacks):
            callback(axis, int(value))

    # -- conveniences the tests read, not part of the Protocol ------------

    def visible_names(self) -> list[str]:
        return [n for n in self.order if self._layer(n)["visible"]]

    def blending(self, name: str) -> str:
        return self._layer(name)["blending"]


class NapariViewer:
    """The real one. napari is imported inside `__init__`, never at module scope.

    **Every image layer is `additive`.** napari's alternative, `translucent`, lets the top layer
    win — and *its black pixels are still pixels*, so they paint black and hide what is beneath.
    Every layer here is mostly empty, so under `translucent` each layer's emptiness would blank
    the one below it. Three things follow, and the third is the one that bites:

        stacking order stops mattering       addition commutes
        the slider becomes a WEIGHT          not an occluder
        two layers of the same voxels DOUBLE  which is why item 2 arrives with one of them off
    """

    #: **`show=False` is the default, and flipping it is what task 23 was.** The canvas is
    #: *embedded* in the screen that is showing, so a second napari window beside the control
    #: window is not a variant arrangement — it is the canvas being in the wrong place, with the
    #: screens' centre panels left black. `show=True` remains available for a bare-viewer probe
    #: with no screens around it; nothing in the flow uses it.
    def __init__(self, *, title: str = "prestomorph", show: bool = False):
        import napari

        self._napari = napari
        self.viewer = napari.Viewer(title=title, show=show)
        #: What this viewer has bound on behalf of the screen now showing, so `unbind_all` can
        #: take exactly those off again rather than clearing napari's own bindings with them.
        self._bound: dict = {"click": [], "double": [], "key": [], "dims": []}
        self.stripped = strip_default_mouse_bindings(self.viewer)
        #: **Straight back into the slot `dims_scroll` just vacated.** Pinch is a
        #: Ctrl+scroll on Windows and napari refuses to route those to the camera, so
        #: the gesture is dead unless something here answers it. See `zoom_on_pinch`.
        install_pinch_zoom(self.viewer)
        #: Non-zero while layers are being pushed. See `_while_pushing_layers`.
        self._pushing = 0

    # -- the widget, which is what the screens actually hold -------------

    def widget(self):
        """The Qt widget to put in a screen's centre panel: **canvas plus the z slider**.

        `QtViewer` is a vertical `QSplitter` holding one `main_widget` of
        `canvas.native` + `QtDims`, and **nothing else** — verified against napari 0.7.1's own
        `qt_viewer.py`, not transcribed. The layer list, the layer controls and the menus are
        `QtViewerDockWidget`s on the *main window*, so taking this widget takes the picture and
        the z slider and leaves every piece of napari chrome behind. That is exactly the split
        this interface wants: §4 gives every screen its own `Colours` and `Layers` blocks, and
        napari's would be a second set of controls for the same layers.

        **`_qt_viewer` rather than `qt_viewer`, and the underscore is the supported one here.**
        The public property raises a `FutureWarning` on every access saying access is deprecated
        and removed in v0.8. Both names reach the same object today; the private one is the one
        that does not print a warning into the user's terminal on every screen change. **When
        0.8 lands this is the line that breaks**, and it breaks loudly rather than quietly.
        """
        return self.viewer.window._qt_viewer

    # -- the port -------------------------------------------------------

    @_while_pushing_layers
    def show_image(self, name, data, *, colour, alpha, visible=True, contrast_limits=None,
                   scale=None, translate=None):
        """``contrast_limits`` is **fixed across the stack** wherever a screen supplies it.

        §6g.4, and it is an honesty decision rather than a default. Auto-stretching each slice
        would make item 3 flatter itself: a dim slice stretched to full range *looks* as strong
        as a bright one, while the per-slice Otsu threshold underneath is genuinely different —
        so the user judges *"does the mask match the signal"* against a raw panel that has been
        silently normalised and a mask panel that has not. §4 gives the user no contrast control
        to compensate with, which is what makes fixed limits the only defensible option.

        **Re-applied on every call, not only at creation.** napari recomputes a layer's limits
        when its ``data`` is replaced, so setting them once at ``add_image`` would let the very
        per-slice restretch this exists to prevent creep back in on the first arrow key.

        ``None`` leaves napari's own behaviour untouched, which is what item 2 gets.

        ``scale`` is the same shape of argument for the other frame - see `show_labels`.
        """
        if colour not in palette.COLOUR_BY_NAME:
            raise KeyError(f"{colour!r} is not one of the ten")
        if self._reusable(name, data):
            layer = self.viewer.layers[name]
            layer.data = data
        else:
            layer = self.viewer.add_image(
                data, name=name, blending="additive",
                colormap=hue_colormap(colour), contrast_limits=contrast_limits)
        if contrast_limits is not None:
            layer.contrast_limits = list(contrast_limits)
        _apply_scale(layer, scale)
        _apply_translate(layer, translate)
        layer.colormap = hue_colormap(colour)
        layer.opacity = alpha
        layer.visible = visible
        layer.metadata["colour"] = colour

    @_while_pushing_layers
    def show_labels(self, name, data, *, colours, alphas, contour=0, visible=True, scale=None,
                    translate=None):
        """The one exception to additive blending, and it is not a style choice.

        Additive would sum a label's hue with the DAPI beneath it — **a red label over blue DAPI
        renders magenta** — corrupting the one thing the layer exists to carry. So `Labels` stays
        `translucent` and must be top of the stack.

        Per-id alpha survives `DirectLabelColormap`, verified by `.map()`: id 1 -> alpha 1.0 and
        id 2 -> alpha 0.35 from a single dict. The two-class scheme is expressible exactly.

        > ### ``scale`` IS COMPONENT_03_PLAN 4a.1's COORDINATE RULE, ITS DISPLAY HALF
        >
        > *"Nucleus labels are displayed and edited at WORKING resolution. The DAPI image is
        > shown at native and the labels layer carries `scale=(1, 2, 2)`, so napari registers
        > them correctly on screen while the array stays in the frame the pipeline wants."*
        >
        > Without it a `(z, 1152, 1152)` label volume and a `(z, 2304, 2304)` image both start
        > at the world origin with a unit voxel, so **the nuclei render over the top-left
        > quarter of the field** - a picture that is wrong in a way that looks like a detection
        > failure rather than a display bug.
        """
        from napari.utils.colormaps import DirectLabelColormap

        mapping = {}
        for label_id, hex_colour in colours.items():
            rgb = _parse_hex(hex_colour, where=f"show_labels({name!r}) label {label_id}")
            mapping[label_id] = rgb + [alphas.get(label_id, 1.0)]
        mapping[None] = [0.0, 0.0, 0.0, 0.0]

        if self._reusable(name, data):
            layer = self.viewer.layers[name]
            layer.data = data
        else:
            layer = self.viewer.add_labels(data, name=name)
        _apply_scale(layer, scale)
        _apply_translate(layer, translate)
        layer.colormap = DirectLabelColormap(color_dict=mapping)
        layer.blending = "translucent"
        layer.contour = contour
        layer.visible = visible
        self._raise_labels_to_top(name)

    def _raise_labels_to_top(self, name: str) -> None:
        index = self.viewer.layers.index(self.viewer.layers[name])
        last = len(self.viewer.layers) - 1
        if index != last:
            self.viewer.layers.move(index, last)

    @_while_pushing_layers
    def show_points(self, name, points, *, colour, size=6.0, labels=None, visible=True,
                    scale=None):
        """The split markers, and **the number beside each one is the whole point.**

        *"Markers show while SHIFT is held. One marker does nothing"* - the numbering is what
        makes that visible **before** the user lets go, so a misfire is seen rather than
        discovered. napari draws it as the points layer's text.
        """
        if colour not in palette.COLOUR_BY_NAME:
            raise KeyError(f"{colour!r} is not one of the ten")
        #: > ### AN EMPTY POINTS LAYER STILL DECLARES A DIMENSIONALITY — 2026-08-23
        #: >
        #: > This reshaped to `(0, 3)`, so a screen with **no markers placed** still contributed
        #: > a three-dimensional layer. `viewer.dims.ndim` is the maximum over every layer, so
        #: > one empty points layer kept napari's z slider alive on item 4 after all five of its
        #: > array layers had been made planes — *"the item 4 screen still has the napari z
        #: > slider under the image"*, reported on the running program.
        #: >
        #: > **The width now follows the points**, and an empty list means two: this layer never
        #: > leads, it follows whatever the screen is drawing, and every screen that draws points
        #: > draws them over a plane.
        width = len(points[0]) if len(points) else 2
        data = np.asarray([[float(c) for c in point] for point in points], dtype=float)
        if data.size == 0:
            data = data.reshape((0, width))
        hex_colour = palette.COLOUR_BY_NAME[colour].hex
        #: **One offset per axis, and the last of them is the one that matters** — the number is
        #: nudged along x so it sits beside the marker rather than on it. A three-element
        #: translation on a 2D layer is the same mismatch `screen_nuclei.plane_scale` exists for.
        text = {"string": "{n}", "color": hex_colour,
                "translation": [0.0] * (width - 1) + [6.0]} if labels is not None else None
        features = {"n": list(labels)} if labels is not None else {}

        if self._reusable(name, data):
            layer = self.viewer.layers[name]
            layer.data = data
            if features:
                layer.features = features
        else:
            layer = self.viewer.add_points(
                data, name=name, size=size, features=features,
                border_color=hex_colour, face_color=hex_colour)
            if text is not None:
                layer.text = text
        _apply_scale(layer, scale)
        layer.visible = visible

    def set_visible(self, name, visible):
        self.viewer.layers[name].visible = visible

    def set_colour(self, name, colour):
        if colour not in palette.COLOUR_BY_NAME:
            raise KeyError(f"{colour!r} is not one of the ten")
        layer = self.viewer.layers[name]
        layer.colormap = hue_colormap(colour)
        layer.metadata["colour"] = colour

    def set_alpha(self, name, alpha):
        self.viewer.layers[name].opacity = alpha

    def set_rendering(self, name, mode):
        """How each **volume** is projected — §3c's three-mode control, and its default.

        Distinct from `blending`, which is how layers composite with one another and stays
        `additive` everywhere. Two properties whose names invite the confusion PrestoCell fell
        into: it sets `blending` at two `add_image` calls and never sets this one at all, so it
        inherits napari's `mip` and has no depth on the screen where depth is the judgement.
        """
        layer = self.viewer.layers[name]
        is_labels = isinstance(layer, self._napari.layers.Labels)
        allowed = LABELS_RENDERING if is_labels else IMAGE_RENDERING
        if mode not in allowed:
            raise ValueError(f"{mode!r} is not a rendering mode for {name!r}; "
                             f"expected one of {allowed}")
        layer.rendering = mode

    def set_ndisplay(self, ndisplay):
        """2D or 3D. **Rich review is `ndisplay=3` only** and asks once, at arrival."""
        if ndisplay not in (2, 3):
            raise ValueError(f"ndisplay is 2 or 3, got {ndisplay}")
        self.viewer.dims.ndisplay = ndisplay

    def remove(self, name):
        del self.viewer.layers[name]

    def clear(self):
        """Empty the viewer. **One canvas means one screen's layers at a time** — task 23.

        > ### THE SECOND DEFECT A REAL RUN FOUND, AND IT WAS INVISIBLE TO 706 TESTS
        >
        > Every per-screen test builds its **own** `RecordingViewer`, so no test ever saw two
        > screens share one. The real viewer is shared, nothing removed anything, and by item 9a
        > it held every layer items 2 through 7 had ever added. Two consequences, and the second
        > is the one that stopped the flow:
        >
        > **The grid counts layers it cannot see.** napari assigns a panel from
        > `layers.index(layer)`, so items 3, 5 and 6's carefully ordered two-panel arrangements
        > would have been computed over a pile of earlier screens' layers.
        >
        > **And a name can change layer KIND between screens.** Item 6 draws the size ramp as a
        > `Labels` layer called `cells`; item 9a draws the confirmed cell set as an **image**
        > called `cells`. `show_image` found the `Labels` layer still sitting there and assigned
        > it a continuous colormap — *"Unable to interpret as labels colormap"*, and item 9a
        > never opened at all.
        """
        self.viewer.layers.clear()

    def reset_view(self):
        """Fit the camera to what is on screen. **Called when a screen arrives, and it has to be.**

        napari fits the camera when the *first* layer lands in an empty viewer, and never again.
        Two things then leave a screen looking wrong and neither is a drawing mistake:

        **The widget changes size.** The canvas is moved between screens whose centre columns
        differ — item 4 has a header strip above it, item 9b has a table below — so the camera
        that fitted the panel it was fitted in no longer fits this one. Item 4 was photographed
        with its DAPI occupying the middle 45% of a black panel.

        **And the grid changes under it.** A screen arriving from a two-panel screen has half the
        viewbox it had; the zoom that filled one fills a quarter of the other.
        """
        self.viewer.reset_view()

    def layer_names(self):
        return [layer.name for layer in self.viewer.layers]

    # -- gestures -------------------------------------------------------

    def _modifiers(self, event) -> frozenset:
        return frozenset(str(m) for m in getattr(event, "modifiers", ()) or ())

    def _ray(self, event):
        """The line through the volume under the pointer, or `None` in a slice view.

        napari gives `position` plus `view_direction` in 3D; the pair start/end is what
        `editing.first_hit_along_ray` walks, and it is PrestoCell's own mechanism.
        """
        direction = getattr(event, "view_direction", None)
        if direction is None or self.viewer.dims.ndisplay != 3:
            return None
        position = np.asarray(event.position, dtype=float)
        direction = np.asarray(direction, dtype=float)
        span = float(max(self.viewer.dims.range[i].stop for i in range(len(position))))
        return (tuple(position - direction * span), tuple(position + direction * span))


    def _reusable(self, name: str, data) -> bool:
        """Whether the existing layer can take `data` by assignment — **and removes it if not.**

        > ### ADDING OVER A NAME THAT EXISTS DOES NOT REPLACE IT — task 26
        >
        > A z-batch above 1 replaces a `(z, y, x)` volume with its `(y, x)` projection, and
        > napari does not re-derive a layer's dimensionality from a `data` assignment. The first
        > version of this returned *"rebuild"* and let the caller fall through to `add_image`
        > **with the old layer still in the list** — and napari's `LayerList` makes a duplicate
        > name unique by appending a suffix, so the result is `DAPI [1]` stacked additively on
        > top of the stale `DAPI` still showing one slice. The batch appeared not to work.
        >
        > **`RecordingViewer` keys its layers by name in a dict**, so it silently overwrote and
        > all 807 tests passed. That is the gap the headless suite has by construction, and it is
        > why this removal is done *here* rather than at three call sites: one place to be wrong.

        Dimensionality is read from the attribute rather than through `np.asarray`, so a lazy
        array is not materialised just to be measured.
        """
        if name not in self.viewer.layers:
            return False
        existing = getattr(self.viewer.layers[name].data, "ndim", None)
        arriving = getattr(data, "ndim", None)
        if arriving is None:
            arriving = np.asarray(data).ndim
        if existing is None or existing == arriving:
            return True
        self.viewer.layers.remove(name)
        return False

    def focus_canvas(self) -> None:
        """Give the canvas keyboard focus. **Without it no key binding can fire.**

        > ### THE ARROW KEYS WERE BOUND AND UNREACHABLE — task 26
        >
        > napari dispatches `viewer.keymap` from the canvas widget's **Qt key events**, so a
        > binding only runs while that widget holds focus. vispy gives it `WheelFocus`, so a
        > click or a scroll on the picture focuses it — but the flow driver never did, so
        > arriving on a screen left focus on whatever control the side column had, and LEFT and
        > RIGHT went there instead. The mouse gestures worked throughout, which is what made it
        > look like the keys specifically were unbound: mouse callbacks are vispy-level and need
        > no Qt focus at all.
        """
        widget = self.widget()
        canvas = getattr(getattr(self.viewer.window, "_qt_viewer", None), "canvas", None)
        target = getattr(canvas, "native", None) or widget
        if target is not None:
            target.setFocus()

    def bind_click(self, callback) -> None:
        """A press that is **not** a drag. napari delivers both through one callback list, so
        the generator form is how the two are told apart: everything after `yield` runs on
        release, and a drag has set `event.pos` moving in between.
        """
        def on_press(viewer, event):
            start = getattr(event, "pos", None)
            yield
            while getattr(event, "type", "") == "mouse_move":
                yield
            if start is not None and getattr(event, "pos", None) is not None:
                moved = max(abs(a - b) for a, b in zip(start, event.pos))
                if moved > 2:
                    return
            callback(Click(tuple(event.position), self._modifiers(event), self._ray(event)))

        self.viewer.mouse_drag_callbacks.append(on_press)
        self._bound["click"].append(on_press)

    def bind_double_click(self, callback) -> None:
        def on_double(viewer, event):
            callback(Click(tuple(event.position), self._modifiers(event), self._ray(event)))

        self.viewer.mouse_double_click_callbacks.append(on_double)
        self._bound["double"].append(on_double)

    def bind_key(self, key: str, callback) -> None:
        """**`overwrite=True`**, because one viewer serves eleven screens and napari raises rather
        than replacing a binding that is already there.

        > ### A GENERATOR BINDS PRESS *AND* RELEASE, AND SHIFT IS WHY THAT MATTERS
        >
        > *"SHIFT + CLICK each part, then RELEASE SHIFT"* - the split executes on release, which
        > is what gives N-way splits for free. napari supports it by running a generator up to
        > `yield` on press and the remainder on release, but it decides that from
        > `inspect.isgeneratorfunction` **on the function it is handed** - so wrapping a
        > generator callback in a plain `lambda` silently turns it into a press-only binding
        > that never runs its second half.

        > ### AND THE KEYMAP IS NOT KEYED BY THE STRING YOU PASSED — found on a real display
        >
        > `viewer.keymap` is keyed by **`KeyBinding` objects**: `bind_key("u", ...)` lands under
        > `<KeyBinding: U>` and `bind_key("Shift-Left", ...)` under `<KeyBinding: Shift+Left>`.
        > So `keymap.pop("u", None)` in `unbind_all` **removed nothing**, silently, and every
        > screen's bindings would have survived into the next one — which is the single thing
        > `unbind_all` exists to prevent.
        >
        > **The keys are taken from the difference rather than reconstructed**, so nothing here
        > depends on how a future napari spells a modifier.
        """
        import inspect

        if inspect.isgeneratorfunction(callback):
            def bound(viewer):
                yield from callback()
        else:
            def bound(viewer):
                callback()

        before = set(self.viewer.keymap)
        self.viewer.bind_key(key, bound, overwrite=True)
        self._bound["key"].extend(set(self.viewer.keymap) - before)

    def set_z(self, index: int) -> None:
        """Move napari's own z slider to `index`.

        > ### THE SYNC ONLY RAN ONE WAY, AND ITEM 2 SHOWED IT — task 26
        >
        > `bind_dims` tells the screen when the **slider** moves. Nothing told the slider when
        > the **screen** moved, and at z-batch depth 1 the image layers are whole `(z, y, x)`
        > volumes that napari slices with that slider. So an arrow key moved `screen.z`, the
        > 6E10 outline followed — it is a 2D array recomputed at `screen.z` on every draw — and
        > **every channel underneath it stayed on the slice napari was still showing.**
        >
        > Only the derived 2D layers moved, which is exactly what a reviewer reported: *"using
        > the left and right keys is only moving the 6e10 mask."*
        >
        > **The echo is already guarded**: this fires napari's own `current_step` event,
        > `_follow` compares the value to `screen.z`, finds them equal and returns.

        Silent in a 2D world — a z-batch above 1 leaves no third axis to set, and every layer is
        a projection the screen has already computed at the right slice.
        """
        dims = self.viewer.dims
        if getattr(dims, "ndim", 0) <= 2:
            return
        dims.set_current_step(0, int(index))


    def bind_dims(self, callback) -> None:
        """napari's own z slider is the one the user reaches for, so the screen has to hear it.

        Item 4's `z slice` readout sat blank through every screenshot of task 23 for exactly
        this reason: the slider moved the image and nothing told the screen.
        """
        def on_dims(event=None):
            #: **Silent while the world is 2D**, which is what a z-batch above 1 makes it: every
            #: layer becomes a projection, napari's slider goes with them, and `current_step`
            #: drops to the two displayed axes. Reporting `point[0]` then hands the screen a
            #: **y coordinate as a z**, and the first redraw of a batch would snap z to 0 and
            #: keep snapping it back while the user tried to move.
            #: **Never while a push is in flight.** See `_while_pushing_layers`.
            if getattr(self, "_pushing", 0):
                return
            if getattr(self.viewer.dims, "ndim", 3) <= 2:
                return
            point = self.viewer.dims.current_step
            callback(0, int(point[0]) if point else 0)

        self.viewer.dims.events.current_step.connect(on_dims)
        self._bound["dims"].append(on_dims)

    def unbind_all(self) -> None:
        """**Called on every arrival.** One viewer, eleven screens: a handler left attached
        answers a click on behalf of a screen the user is no longer looking at."""
        for callback in self._bound["click"]:
            if callback in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(callback)
        for callback in self._bound["double"]:
            if callback in self.viewer.mouse_double_click_callbacks:
                self.viewer.mouse_double_click_callbacks.remove(callback)
        for key in self._bound["key"]:
            self.viewer.keymap.pop(key, None)
        
        for callback in self._bound["dims"]:
            try:
                self.viewer.dims.events.current_step.disconnect(callback)
            except (TypeError, ValueError):                      # already gone
                pass
        self._bound = {"click": [], "double": [], "key": [], "dims": []}

    # -- frames ---------------------------------------------------------

    def set_grid(self, shape, stride=1):
        """Items 3, 5 and 6's frames — **one canvas, one camera, split into panels**.

        `GridCanvas`'s own docstring calls it *"the only grid mode that is still inside one
        canvas with one camera"*, and `spacing` defaults to `0.0`. So two panels are two
        viewboxes in a **single black canvas**, and any gap between them shows that same black.

        **This is why no divider is drawn.** §6f.9's *"the black canvas separates itself from the
        panels"* holds against the chrome and **not** between two panels, where both sides are
        black. The boundary comes from the header strip instead — one label over each panel, and
        the split between them marks the divide.

        **`(1, 1)` disables the grid rather than requesting a one-cell one**, because a screen
        arriving from a two-panel screen has to undo what that screen asked for. Applying the
        shape without disabling would leave every single-view screen rendering inside one cell of
        a stale two-cell grid.
        """
        rows, columns = (int(v) for v in shape)
        if rows < 1 or columns < 1:
            raise ValueError(f"a grid needs at least one row and column, got {shape}")
        if int(stride) < 1:
            raise ValueError(f"stride is at least 1, got {stride}")
        self.viewer.grid.enabled = (rows, columns) != (1, 1)
        if self.viewer.grid.enabled:
            self.viewer.grid.shape = (rows, columns)
            self.viewer.grid.stride = int(stride)

    def two_panels(self, on: bool = True) -> None:
        """`set_grid((1, 2))`, kept as the name items 3 and 5 were written against."""
        self.set_grid((1, 2) if on else (1, 1), 1)

    def set_z(self, z: int) -> None:
        self.viewer.dims.set_point(0, z)

    def run(self) -> None:
        """The only place `napari.run()` is called, and it is unreachable from Phase A.

        PrestoCell's failure was not calling this — every napari application does. It was calling
        it from *inside a pipeline function*, so the pipeline could not run without a human
        present. `tests/test_ui_boundary.py` asserts nothing outside `ui/` can reach here.
        """
        self._napari.run()
