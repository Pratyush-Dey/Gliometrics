"""The preview helpers three screens share. `COMPONENT_03_PLAN.md` §6g.4, §4 items 3, 5, 9a.

**These lived in `screen_threshold1.py` until 2026-08-19, and item 5 imported them from there.**
That coupling was recorded at the time as *"screen-to-screen, which is unusual"*, with the
condition written down: **if a third screen ever needs them, move them here — two callers did
not justify the churn mid-build, three would.** Item 9a is the third, so they moved.

**Nothing about them changed in the move.** Same functions, same defaults, same tests; only the
module they live in. Item 3 and item 5 now both import from a module that belongs to neither.

    middle_slice              where a screen opens, and 3DMorph opens there too
    z_position                `(20/40)`, one-based for the reader
    volume_contrast_limits    §6g.4 — fixed across the stack, never re-derived per slice
    preview_mask              one slice through `threshold_channel`, with the volume's contrast

**The percentiles are memoised per volume since 2026-09-03 — patch item A1.** They were being
recomputed from inside `draw()` on items 1.5, 2, 4 and 7b, at 2.4-3.6 s a call on a production
channel, for a number that is a property of the volume and cannot move on a z step. Nothing
about *what* the numbers are changed; see `_cached_limits`.
"""

from __future__ import annotations

import weakref

import numpy as np

from ..cache import cache_key
from ..threshold import threshold_channel

__all__ = ["middle_slice", "z_position", "volume_contrast_limits", "preview_mask",
           "log_view", "log_limits",
           "contrast_limits"]


def middle_slice(n_slices: int) -> int:
    """Where the screen opens. 3DMorph does the same — `midslice = round(zs/2,0)`.

    Zero-based, so a 40-slice stack opens at index 19 and the indicator reads `(20/40)` —
    §4's *"slice 20 of 40"*. For an odd count this is the exact middle.
    """
    if n_slices < 1:
        raise ValueError(f"a stack needs at least one slice, got {n_slices}")
    return max(0, (n_slices - 1) // 2)


def z_position(z: int, n_slices: int) -> str:
    """`(20/40)`. **One-based for the reader**, so the last slice reads `(40/40)`.

    The word in front of it is always `z slice` on this screen and never `z batch`: there is no
    projection mode here at all, so the arrow keys can only ever move one plane at a time.
    """
    return f"({z + 1}/{n_slices})"


def log_view(data: np.ndarray) -> np.ndarray:
    """`log1p`, the first half of `nuclei.normalize_for_detection`'s `log` method.

    The clip that follows it there is an affine rescale between two percentiles, and napari's
    `contrast_limits` is exactly that — so pushing `log1p` data with **log-space limits**
    reproduces what the detector sees without building a second copy of the volume.

    > ### ⇒ IT MOVED HERE ON 2026-08-30, ON A RULE WRITTEN BEFORE IT WAS NEEDED
    >
    > `screen_background.log_view` and `screen_nuclei.log_display` were byte-identical, and the
    > duplication was deliberate: *"item 4's copy is a display toggle on a screen that also shows
    > two other channels, this one is half of a screen's whole subject, and the shared thing is
    > one `log1p` call. **Task 13's rule is that a third caller earns a move to `preview.py`;
    > there are two.**"*
    >
    > **Item 2 is the third.** The rule fired on its own terms, and the two copies came here in
    > the same change that created the need for them — which is the point of writing the trigger
    > down rather than the conclusion.

    See `log_limits` for why the limits can be taken from the raw percentiles.
    """
    return np.log1p(np.clip(np.asarray(data, dtype=np.float32), 0, None))


#: The percentile memo's bound. An entry is two floats and a weak reference, so the cap is about
#: tidiness rather than memory — but it is a cap, for the reason `SliceCache` has one: a flow
#: that is handed a new channel per image would otherwise accumulate a key per image for the
#: life of the process. Six layers on the widest screen, so eight holds a screen and a spare.
_LIMIT_CAPACITY = 8

_LIMIT_CACHE: dict[tuple, tuple[float, float]] = {}
_LIMIT_REFS: dict[tuple, weakref.ref] = {}


def reset_limit_cache() -> None:
    """Drop every memoised limit. **For tests, and for nothing else in the flow.**

    Nothing in the program needs to invalidate this: an entry is keyed on the identity of the
    array it was computed from, so it can only ever be served back to that same array, and the
    array's own death removes it. A screen that recomputes a channel gets a new array, and
    therefore a new key, for free.
    """
    _LIMIT_CACHE.clear()
    _LIMIT_REFS.clear()


def _forget(key: tuple) -> None:
    """Drop one entry, value and reference together. Called by eviction and by the weakref."""
    _LIMIT_CACHE.pop(key, None)
    _LIMIT_REFS.pop(key, None)


def _cached_limits(volume, tail: tuple, compute) -> tuple[float, float]:
    """`compute(array)` once per (array identity, tail), and never again while that array lives.

    > ### ⇒ WHY IDENTITY, AND WHY A WEAK REFERENCE UNDERNEATH IT
    >
    > Measured on a production field: `percentile 0.1` **3.43 s**, `percentile 99.9` **2.41 s**,
    > `percentile [1, 98]` **3.57 s**. Items 1.5, 2, 4 and 7b called them from inside `draw()`,
    > so an arrow key paid two or three of those for a number that is a property of the volume
    > and cannot move on a z step. **The array itself is the whole key** — there is no
    > parameter here to go stale, which is what makes identity sufficient rather than merely
    > convenient.
    >
    > **`id()` alone would be a wrong answer waiting to happen.** CPython reuses the address of
    > a collected object, so a freed 850 MB channel and a fresh one can share an id, and
    > `cache.cache_key` already records that a stale key serving a wrong preview is the one
    > failure a preview must never have. The entry therefore holds a `weakref` whose callback
    > removes it, so an id cannot be reused while its entry is alive. The reference is **weak**,
    > so memoising a channel never keeps one alive — `SliceCache.release`'s rule, that memory
    > is saved by holding fewer arrays, is not undone here.
    >
    > **In-place mutation is not defended against, and does not occur.** Channels are loaded
    > once and treated as immutable; every derived view in the flow — item 1.5's filter, item
    > 2's `np.where` — allocates a new array and so takes a new key.
    """
    array = np.asarray(volume)
    key = (id(array), array.shape, array.dtype.str) + tuple(tail)
    if key in _LIMIT_CACHE:
        value = _LIMIT_CACHE.pop(key)           # reinsert: most recently used
        _LIMIT_CACHE[key] = value
        return value

    value = compute(array)
    try:
        reference = weakref.ref(array, lambda _dead, k=key: _forget(k))
    except TypeError:
        #: Not everything array-like can be weakly referenced. Computing every time is the right
        #: answer there — slower, never wrong — rather than keying on an id nothing guards.
        return value
    _LIMIT_CACHE[key] = value
    _LIMIT_REFS[key] = reference
    while len(_LIMIT_CACHE) > _LIMIT_CAPACITY:
        _forget(next(iter(_LIMIT_CACHE)))
    return value


def _log_percentiles(array: np.ndarray) -> tuple[float, float]:
    """`log_limits` without the memo. Separated so the cached and uncached paths cannot drift."""
    low, high = np.percentile(np.asarray(array, dtype=np.float32), [1.0, 98.0])
    return float(np.log1p(max(float(low), 0.0))), float(np.log1p(max(float(high), 0.0)))


def log_limits(volume: np.ndarray) -> tuple[float, float]:
    """The log panel's contrast limits, **taken from the RAW percentiles**.

    `log1p` is monotone, so the 1st and 98th percentiles of `log1p(v)` are `log1p` of the 1st
    and 98th percentiles of `v` — exact, not an approximation. Taking them this way avoids
    building a second float32 copy of the whole channel to take percentiles of, which is 849 MB
    at production width.

    **Memoised per volume since A1**, and the float32 cast above is now paid on a miss only.
    """
    return _cached_limits(volume, ("log1p", 1.0, 98.0), _log_percentiles)


def volume_contrast_limits(volume: np.ndarray, low: float = 0.1,
                           high: float = 99.9) -> tuple[float, float]:
    """§6g.4 — **computed once over the whole volume, never re-derived per slice.**

    Auto-stretching each slice would make the screen flatter itself: a dim slice stretched to
    full range *looks* as strong as a bright one, while the per-slice Otsu threshold underneath
    is genuinely different. And §4 gives the user no contrast control to compensate with, which
    is what makes fixed limits the only defensible option rather than merely the tidier one.

    **Percentiles rather than min/max**, because the study's own data spans four orders of
    magnitude within one plane — `min 0.0095 max 7169.57` — so a single hot voxel mapped to
    white would flatten everything real to black. 99.9 is already this codebase's *"top of
    signal"*: it is the percentile `slice_contrast` uses.
    **Memoised per volume since A1**, so a redraw at the same volume costs nothing. Items 3 and
    5 already called this once in `set_volume` / `set_volumes` and are unchanged by it.
    """
    return _cached_limits(volume, cache_key("percentile", low, high),
                          lambda array: (float(np.percentile(array, low)),
                                         float(np.percentile(array, high))))


def contrast_limits(volume, explicit=None, low: float = 0.1,
                    high: float = 99.9) -> tuple[float, float]:
    """The display range for one channel: **what the user typed, or the percentiles.**

    > ### ITEMS 2, 4 AND 7b PASSED NOTHING AT ALL, AND THAT MADE 6E10 INVISIBLE
    >
    > Measured on `5xFAD/1234/program training crop2.tif`: the 6E10 channel runs
    > `0.0155 .. 93567.4`, but its 99th percentile is `1724`. One hot voxel is **54x** the
    > brightest thing the user is looking for, so napari's own min-to-max default rendered
    > every plaque at under 2% of the ramp. DAPI fared better at ~47% and still looked washed.
    >
    > Items 3, 5, 9a and 9b already stretched by percentile; these three did not, and
    > `viewer.show_image` recorded that as a decision — *"None leaves napari's own behaviour
    > untouched, which is what item 2 gets."*

    ``explicit`` is item 1's per-channel min/max, the user's decision of 2026-08-20: they set
    the range in Fiji and type it in, and every screen showing that channel uses it. **The two
    scales are the same numbers** — confirmed against Fiji's own histogram on DAPI slice 1,
    `4385.32` from both readers — so nothing is converted on the way in.

    Blank boxes fall back to the percentiles, so a channel nobody has tuned still renders.
    """
    if explicit is not None:
        low_value, high_value = explicit
        return (float(low_value), float(high_value))
    return volume_contrast_limits(volume, low, high)


def preview_mask(volume: np.ndarray, z: int, *, coefficient: float,
                 volume_contrast: float) -> np.ndarray:
    """One slice's threshold — 0.25 s at production width, which is what makes it live.

    Otsu is **per slice**, so one slice's threshold is *exactly* what the stack would produce
    for that plane. That is the property the whole screen rests on, and it survives only because
    `volume_contrast` comes from the caller: see the module docstring.

    `min_object_voxels` is deliberately **not** applied — it is a 3D connected-component filter,
    and a per-slice version would be complete-looking but wrong. 3DMorph previews it in 2D
    anyway (`handles.FilteredImg`); this plan rejected that.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
    plane = volume[z:z + 1]
    return threshold_channel(plane, coefficient=coefficient,
                             volume_contrast=volume_contrast).mask[0]
