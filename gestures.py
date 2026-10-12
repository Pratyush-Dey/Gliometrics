"""The gestures that are the same on every screen that has them — task 25.

**Six screens step through z and each one had its own copy of the binding.** Items 3, 4, 5, 9a
and 9b each carried `step_z` plus a Qt `keyPressEvent` matching `Key_Left`/`Key_Right`, and item
2 carried neither; that is five copies of one contract and one screen quietly outside it. The
same argument that produced `widgets.py` and `preview.py`: five copies of a rule drift five ways.

> ### AND NONE OF THE FIVE COPIES WORKED — task 25
>
> They were `keyPressEvent` overrides on the **screen** widget. The canvas is napari's, embedded
> since task 23, and once the user clicks on the picture napari's canvas has keyboard focus — so
> Qt delivers the arrow key to the canvas and the screen's handler never runs. Every one of them
> was correct code on an object the key could not reach.
>
> The Qt handlers are left in place: they are what answers when the *controls* have focus, which
> is where the keyboard is before the first click on the image.
"""

from __future__ import annotations

__all__ = ["Z_KEYS", "bind_z_stepping", "SplitMarkers",
           "MARKER_LAYER", "MARKER_COLOUR"]

from dataclasses import dataclass, field

import numpy as np

#: **LEFT and RIGHT, and the same two with SHIFT held.**
#:
#: > napari keys on the whole combination, so with SHIFT down an arrow press arrives as
#: > `Shift-Left` and a binding on `Left` alone never fires. On item 4 that would freeze the
#: > picture on one slice for exactly as long as the user was in split mode — and a clump's
#: > daughters are separated in z as often as in xy, so every seed would land on the wrong
#: > nucleus. The user's rule, 2026-08-20: *"the user should be able to change z with their
#: > arrow keys while still holding down shift to stay in split mode."*
#:
#: Bound on every z screen rather than only item 4, because a modifier changing what an arrow
#: key does is exactly the kind of inconsistency a reviewer discovers at the wrong moment.
Z_KEYS: tuple[tuple[str, int], ...] = (
    ("Left", -1), ("Right", 1),
    ("Shift-Left", -1), ("Shift-Right", 1),
)


def bind_z_stepping(viewer, screen) -> None:
    """LEFT and RIGHT step one slice, and **napari's own z slider moves the screen with it.**

    The slider under the canvas is the control the user reaches for first — it is the one piece
    of napari chrome that travels with the widget — and until this task **nothing was listening
    to it.** Item 4's `z slice` readout sat blank through every screenshot task 23 took while
    the slider moved z perfectly well: the viewer knew and the screen did not.

    **The echo guard is the second half.** `set_z` redraws, a redraw can move the slider, and a
    slider event calls back here; without the equality test the two chase each other.
    """
    if viewer is None:
        return
    for combination, delta in Z_KEYS:
        viewer.bind_key(combination, lambda d=delta: screen.step_z(d))
    viewer.bind_dims(lambda axis, value: _follow(screen, axis, value))


def _follow(screen, axis: int, value: int) -> None:
    if axis != 0:
        return
    if int(value) == int(getattr(screen, "z", -1)):
        return
    screen.set_z(int(value))


MARKER_LAYER = "split markers"

#: **White**, the one colour in the ten that is neither a channel nor a class on this screen —
#: the labels beneath carry twelve hues by id, so a marker in any of them would read as one of
#: them. `SELECTED_FILL` is white on item 6 for the same reason.
MARKER_COLOUR = "white"


@dataclass
class SplitMarkers:
    """Shift-held click markers — white crosses, **numbered in click order**. §6h.9.

    | | |
    |---|---|
    | **why white** | the ring sits at relative luminance 0.380 and white is 1.0, so a marker is **2.6x brighter** than any label it lands on |
    | **why a cross** | a shape nothing else on the screen uses, so it is found without colour and cannot be mistaken for a label |
    | **why numbered** | §4 records that *"releasing with one marker does nothing, silently."* A lone `1` on screen says the split will not happen **before** you release — and on a three-way split you can confirm you hit three nuclei rather than double-marking one |
    """

    points: list[tuple[int, int, int]] = field(default_factory=list)

    def add(self, z: int, y: int, x: int) -> int:
        """Drop a marker and return its click number, 1-based."""
        self.points.append((int(z), int(y), int(x)))
        return len(self.points)

    def numbered(self) -> list[tuple[int, tuple[int, int, int]]]:
        return [(n + 1, point) for n, point in enumerate(self.points)]

    def clear(self) -> None:
        """Shift released. Markers vanish whether or not they produced a split."""
        self.points.clear()

    def is_actionable(self) -> bool:
        """**Two or more.** One marker does nothing — the numbering is what says so in advance."""
        return len(self.points) >= 2

    def __len__(self) -> int:
        return len(self.points)

    def array(self, shape):
        """The markers as a volume, or `None` when there are fewer than two.

        `split_by_watershed` *"takes markers as an array and does not care where they came
        from"* - a reviewer's clicks are a valid marker array. **One marker returns `None`
        rather than a one-marker array**, because a watershed with a single seed returns the
        object unchanged and would record a split that did not happen.
        """
        if not self.is_actionable() or shape is None:
            return None
        volume = np.zeros(tuple(int(n) for n in shape), np.int32)
        for number, point in self.numbered():
            if all(0 <= c < dim for c, dim in zip(point, volume.shape)):
                volume[tuple(int(c) for c in point)] = number
        return volume if int(volume.max()) >= 2 else None
