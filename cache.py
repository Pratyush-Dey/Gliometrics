"""Compute reuse for the interactive flow. `COMPONENT_03_PLAN.md` §4a.6, §4a.8.

**This module is `prestomorph/cache.py`, not `ui/cache.py`, and the placement is a decision.**
§4a.7's test is simple: *if it would run headless, it lives outside*. There is no napari in
compute reuse — so putting it in `ui/` would mean anything wanting the cache had to import a
napari-dependent package.

Two jobs, and they are unrelated except that both exist to stop the interface stuttering:

    the slice cache          one (slice, coefficient) preview costs 0.25 s; stepping back is free
    projection precompute    43 ms per 10-slice projection at 2304**2, which is a visible stutter
                             on every arrow key. Precomputed once per depth it is nothing
    the window cache         what the screens actually hold — the two above, chosen by mode

> ### ⇒ `ProjectionCache` HAD NO CALLERS AT ALL UNTIL 2026-09-03 — patch item A9
>
> It was written, documented against §4a.8, and **never wired to anything**: the third piece of
> complete, correct, uncalled code this project has produced, after item 6's inert cutoff and
> item 9b's dead `Step by batch`. `WindowCache` below is what finally calls it — and what
> serves the **rolling** mode it cannot serve, since rolling has a window at every z and caching
> all 31 positions of one layer is 657 MB at production width.

> ### ⇒ §4a.6's `uint16` MEMORY PLAN IS NOT BUILT, AND MUST NOT BE
>
> It says *"hold channels as `uint16`, it is the source dtype"*. **The source is `float32`** —
> verified on the study's own images. The cast truncates every fractional part and **collapses
> every voxel below 1.0 to zero**; one plane reads `min 0.0095 max 7169.57 mean 51.44`, so the
> floor is four orders of magnitude below the ceiling and rounding it away would reintroduce the
> Otsu divergence §6.1 records.
>
> **Replaced 2026-08-19: hold `float32`, and save memory by holding FEWER channels rather than
> narrower ones.** `release()` below is that plan.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Callable, Hashable

import numpy as np

__all__ = ["SliceCache", "ProjectionCache", "WindowCache", "NucleusCache",
           "cache_key", "detection_key", "DETECTION_FIELDS",
           "WINDOW_CACHE_CAPACITY", "NUCLEUS_CACHE_ENTRIES"]


def cache_key(*parts: Hashable) -> tuple:
    """**Derive the key, never transcribe it.**

    §4a.6 borrows this rule from `EXPORT_SPEC.md` §5.1, where a hand-maintained transcription
    rotted beside its authority badly enough that 37 columns were written where 72 were
    advertised. A key built by hand from *"the parameters that matter"* goes stale the first time
    a parameter is added and nobody notices, and a stale cache key serves a **wrong preview**,
    which is the one failure a preview must never have.

    Floats are rounded to 9 decimal places so a slider that reports 0.1 and 0.10000000000000001
    on two passes does not miss its own cache entry.
    """
    return tuple(round(p, 9) if isinstance(p, float) else p for p in parts)


@dataclass
class SliceCache:
    """One computed preview per (key) — `COMPONENT_03_PLAN.md` §4 item 3's *"cached afterwards"*.

    **Bounded, and the bound is the point.** Expected use on item 3 is *"three to five slices per
    coefficient"*, so the working set is small — but a user who drags the slider across its whole
    range would otherwise accumulate one full-resolution mask per position. At 2304 x 2304 that
    is 5 MB a step.

    Least-recently-used eviction, because the reason a preview is cached at all is that stepping
    **back** should be free.
    """

    capacity: int = 32
    _entries: dict[tuple, np.ndarray] = field(default_factory=dict, repr=False)
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple, compute: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the cached array for ``key``, computing it only on a miss."""
        if key in self._entries:
            self.hits += 1
            value = self._entries.pop(key)      # reinsert: most recently used
            self._entries[key] = value
            return value

        self.misses += 1
        value = compute()
        self._entries[key] = value
        while len(self._entries) > self.capacity:
            self._entries.pop(next(iter(self._entries)))
        return value

    def invalidate(self) -> None:
        """Drop everything. Called when a commit changes what a preview would mean."""
        self._entries.clear()

    def release(self) -> None:
        """§4a.6's memory plan, as amended: hold **fewer** arrays, never narrower ones."""
        self.invalidate()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple) -> bool:
        return key in self._entries


@dataclass
class ProjectionCache:
    """Every max projection for one z-batch depth, computed together. §4a.8.

    **43 ms per 10-slice projection at 2304 x 2304**, measured — recomputed on every arrow-key
    step that is a visible stutter; precomputed it is nothing. At depth 10 over a 40-slice stack
    there are **four** projections, 4 x 43 ms ~= **170 ms once**, and stepping is then instant.

    Rebuilt on change, because the user is expected to try a depth, find it too coarse, and try
    another.

    > **The known artefact, recorded so it is not later mistaken for a bug.** Fixed,
    > non-overlapping batches **straddle nuclei**. At `nucleus_diameter_um = 7.6` and 0.6 um
    > slices a nucleus spans ~13 slices, so at depth 10 a nucleus occupying slices 8-20 appears
    > partially in two batches and **wholly in neither** — the common case at 13-against-10, not
    > an edge case.
    >
    > **Accepted deliberately, because of the division of labour: projection mode is for
    > SPOTTING and slice mode is for DECIDING.** A straddled nucleus still shows up in both
    > batches well enough to notice, and the judgement is made in slice view regardless. The
    > alternative — a rolling window centred on the current slice — has no boundaries but
    > restores all 40 step positions, losing the speed that is the mode's entire purpose.
    """

    depth: int = 1
    _batches: list[np.ndarray] = field(default_factory=list, repr=False)
    _source_id: int | None = field(default=None, repr=False)

    def build(self, volume: np.ndarray, depth: int) -> None:
        """Precompute every batch for this depth. Cheap enough to call on every depth change."""
        if volume.ndim != 3:
            raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
        if depth < 1:
            raise ValueError(f"z-batch depth must be at least 1, got {depth}")

        self.depth = depth
        self._source_id = id(volume)
        self._batches = [] if depth == 1 else [
            volume[start:start + depth].max(axis=0)
            for start in range(0, volume.shape[0], depth)
        ]

    def batch_index(self, z: int) -> int:
        return z // self.depth

    def view(self, volume: np.ndarray, z: int) -> np.ndarray:
        """The plane to draw at slice ``z``.

        **At depth 1 this is the slice itself, not a one-slice projection.** Same answer, but it
        avoids materialising 40 copies of the volume to prove it.
        """
        if self.depth == 1:
            return volume[z]
        if not self._batches or self._source_id != id(volume):
            self.build(volume, self.depth)
        return self._batches[min(self.batch_index(z), len(self._batches) - 1)]

    def is_view_only(self) -> bool:
        """Projection mode takes no clicks, and that is **correctness, not simplification**.

        A max projection is 2D: a click yields (y, x) and **no z at all**, so *"add a nucleus
        here"* is underdetermined. Resolving z by `argmax` is available and cheap — a projection
        already knows which slice contributed each pixel — but it is **least reliable exactly
        where it would matter most**: two nuclei overlapping in xy and separated in z, which is
        the fusion the reviewer is hunting.
        """
        return self.depth > 1

    def release(self) -> None:
        self._batches = []
        self._source_id = None

    def __len__(self) -> int:
        return len(self._batches)
#: **Window positions kept per layer.** One float32 plane at production width is 21 MB, so this
#: is ~85 MB for a cached channel and ~270 MB across item 2's four layers — reached only if the
#: user actually revisits four positions, and paid back at ~80 ms a projection.
#:
#: **Per layer rather than shared**, so a screen drawing four layers holds four positions of all
#: of them rather than one position of one; the unit the user moves is a *position*, not a layer.
WINDOW_CACHE_CAPACITY = 4


@dataclass
class WindowCache:
    """The z-batch screens' projected planes. §4a.8, **both window modes**, one holder.

    > ### THE TWO MODES NEED DIFFERENT MACHINERY, AND THAT IS WHY THIS EXISTS
    >
    > **Batched** has four positions at depth 10 over a 40-slice stack, so every one of them can
    > be precomputed the moment the depth changes and stepping is then instant. That is exactly
    > what `ProjectionCache` was built for, and `batch` below is its first caller.
    >
    > **Rolling** has 31, because every window position exists — which is the property the user
    > chose it for (*"it shows 10 slices and moves one by one. i think thats good"*) and the
    > reason it has no nucleus-straddling artefact. Precomputing them is 657 MB **per layer**,
    > so rolling gets an LRU instead: revisiting a position is free, a fresh one still costs its
    > ~80 ms. The user steps forward all the way sometimes but goes back and forth often.
    >
    > **Both modes stay.** The user uses both.

    **A layer's entries are dropped when the array they came from is replaced.** The key the
    patch specifies is `(layer, z, depth, mode)`, which does not name the volume — so the volume
    is tracked beside it rather than hashed into it, and a commit that produces a new mask, or a
    new image, cannot be served the previous one's planes. `cache_key`'s rule holds: a stale key
    serving a wrong preview is the one failure a preview must never have.
    """

    capacity: int = WINDOW_CACHE_CAPACITY
    _planes: dict[str, SliceCache] = field(default_factory=dict, repr=False)
    _batches: dict[str, ProjectionCache] = field(default_factory=dict, repr=False)
    _sources: dict[str, tuple] = field(default_factory=dict, repr=False)
    #: One weak reference per source array, so a layer is forgotten when what it was built
    #: from dies. See `_track` for why the `id()` stamp beside it is not enough on its own.
    _refs: dict[str, list] = field(default_factory=dict, repr=False)

    def plane(self, layer: str, source, key: tuple, compute: Callable[[], np.ndarray]):
        """`compute()` once per `(layer, *key)`, while `source` is what it was computed from."""
        self._track(layer, source)
        cache = self._planes.get(layer)
        if cache is None:
            cache = self._planes[layer] = SliceCache(capacity=self.capacity)
        return cache.get(cache_key(layer, *key), compute)

    def batch(self, layer: str, volume: np.ndarray, z: int, depth: int) -> np.ndarray:
        """`ProjectionCache` as written: every batch for this depth, built together.

        Only ever called above depth 1 — at depth 1 the two window helpers disagree about
        whether the answer is a plane or the volume, and a slice is free either way.
        """
        self._track(layer, volume)
        cache = self._batches.get(layer)
        if cache is None:
            cache = self._batches[layer] = ProjectionCache()
        if cache.depth != int(depth) or not len(cache):
            cache.build(volume, int(depth))
        return cache.view(volume, int(z))

    def _track(self, layer: str, source) -> None:
        """Forget `layer` when the array or arrays behind it are not the ones it was built on.

        > ### id() ALONE RECYCLES, AND A9's SIBLING IN preview.py ALREADY SAID SO
        >
        > This compared `tuple(id(array) ...)` and nothing else. CPython hands a freed array's
        > address straight to its replacement - **measured at 4 of 5 trials on a
        > `(40, 1152, 1152)` bool** - so a source replaced while its predecessor was already
        > dead keeps the same stamp, this does not forget, and the layer is served the
        > **previous array's planes**. A stale key serving a wrong preview is the one failure
        > `cache_key` exists to prevent.
        >
        > **Unreachable today, and that is not a guarantee.** Every source is held by
        > `session.channels`, `session.products` or the screen itself while its replacement is
        > computed, so the new array lands elsewhere and the stamp does change. That ordering
        > is true at every call site and written down at none of them.
        >
        > **The weak reference closes it**, exactly as `preview._cached_limits` does for the
        > contrast limits: the entry dies with the array it was built from, so a recycled
        > address cannot reach a live entry. The stamp stays because it is what catches a
        > *replacement* - the weakref catches a *death*, and they are different events.
        """
        arrays = source if isinstance(source, tuple) else (source,)
        stamp = tuple(id(array) for array in arrays)
        if self._sources.get(layer) != stamp:
            self._sources[layer] = stamp
            self._watch(layer, arrays)
            self.forget(layer)

    def _watch(self, layer: str, arrays) -> None:
        """Hold a weak reference to each source, dropping `layer` when any of them dies.

        **An object that refuses a weak reference simply is not watched**, and the `id()` stamp
        is then all there is - which is what this was before. Nothing in the flow passes one;
        every source is a numpy array, and numpy arrays are weak-referenceable.
        """
        refs = []
        for array in arrays:
            try:
                refs.append(weakref.ref(array, self._on_death(layer)))
            except TypeError:
                continue
        self._refs[layer] = refs

    def _on_death(self, layer: str):
        """The callback one dead source fires. **Bound through a weak reference to `self`**, so
        a cache holding a callback that holds the cache does not keep either alive."""
        me = weakref.ref(self)

        def forget(_reference) -> None:
            cache = me()
            if cache is not None:
                cache._sources.pop(layer, None)
                cache._refs.pop(layer, None)
                cache.forget(layer)

        return forget

    def forget(self, layer: str) -> None:
        self._planes.pop(layer, None)
        self._batches.pop(layer, None)

    def invalidate(self) -> None:
        """Drop everything. A screen calls this when a commit changes what a window would mean."""
        self._planes.clear()
        self._batches.clear()
        self._sources.clear()
        self._refs.clear()

    def release(self) -> None:
        """§4a.6's memory plan, as amended: hold **fewer** arrays, never narrower ones."""
        self.invalidate()

    @property
    def hits(self) -> int:
        return sum(cache.hits for cache in self._planes.values())

    @property
    def misses(self) -> int:
        return sum(cache.misses for cache in self._planes.values())

    def __len__(self) -> int:
        return sum(len(cache) for cache in self._planes.values()) + len(self._batches)
# --------------------------------------------------------------------------
# The nucleus cache — patch item A10, 2026-09-03
# --------------------------------------------------------------------------

#: **What `detect_nuclei` actually reads**, and therefore the whole key. Read off its call sites
#: rather than off a list of *"the parameters that matter"* — `cache_key`'s rule, and the reason
#: `test_ui_commits.py` asserts this tuple against the arguments the two callers pass.
#:
#: **Everything else is excluded, and the exclusion is the feature.** `cell_coefficient`,
#: `mask_coefficient`, `membership_threshold`, `min_cell_volume_um3` and the soma parameters are
#: not here, which is what makes items 3, 5, 6 and 6.5 freely re-enterable: going back to change
#: one of them re-commits in seconds instead of re-running Cellpose for 200-500 s.
#:
#: **Item 2 is not re-enterable, and that is why `plaque_veto_coefficient` is in this list.**
#: The docs already say item 2 *"sits outside the freely re-enterable range"*; this is the line
#: that makes that true rather than merely stated.
DETECTION_FIELDS: tuple[str, ...] = (
    "nucleus_diameter_um",
    "downsample",
    "stitch_threshold",
    "nucleus_min_volume_voxels",
    "normalize",
    "plaque_dilate_um",
    "plaque_veto_coefficient",
    "dapi_background_scale_um",
    "dapi_background_strength",
)

#: Roles that change what the detector sees. `cell` and `cd68` do not reach it.
DETECTION_ROLES: tuple[str, ...] = ("nuclei", "plaque")


def detection_key(params, *, image, voxel_size_um, roles) -> tuple:
    """The identity of one detection: **its parameters, never its arguments.**

    > ### ⇒ THE TRAP THIS EXISTS TO AVOID, AND IT IS SILENT
    >
    > The two callers reach `detect_nuclei` with **different arguments that compute the same
    > answer**:
    >
    > | | item 4's COMMIT | `segment_image` |
    > |---|---|---|
    > | DAPI | the **filtered** channel from item 1.5 | the **raw** channel |
    > | background | `strength=0.0` | `sigma_um` + `strength` from the parameters |
    > | plaque mask | `products[Step.VETO]` | thresholded internally |
    >
    > A cache keyed on the input arrays therefore **misses every time**, and misses *silently* —
    > the feature does nothing while looking complete. Keying on the parameters is what makes
    > the two agree, and `params` is one `SegmentationParams` built by `session_params` in both
    > places precisely so there is no second set of defaults to drift.

    The image identity, the voxel size and the two roles that reach the detector are in the key
    as well: the same numbers on a different file are a different answer.
    """
    values = tuple(getattr(params, name, None) for name in DETECTION_FIELDS)
    where = tuple(int(index) for role, index in sorted((roles or {}).items())
                  if role in DETECTION_ROLES)
    voxel = tuple(float(v) for v in (voxel_size_um or ()))
    return cache_key(str(image), where, voxel, *values)


#: **Two to three entries, because one is the size of the thing being avoided.** A label volume
#: at production width is ~212 MB as int32, and the working set is *"the last parameters I tried
#: and the one before it"* — a user comparing two diameters, not a history.
NUCLEUS_CACHE_ENTRIES = 3


@dataclass
class NucleusCache:
    """Cellpose's answer, held for the length of the session. **Memory only, never disk.**

    > ### WHY THIS EXISTS AT ALL
    >
    > Arriving at item 6 called `Compute.derive`, which called the full batch `segment_image`,
    > which **re-ran Cellpose from scratch** — 200-500 s that item 4's COMMIT had paid minutes
    > earlier at the same detection parameters. Item 5 to item 6 was the whole pipeline again.
    >
    > **A persistent, on-disk tier was considered in full and rejected by the user.** Batch
    > already runs Cellpose once per image, so disk would only help across program restarts —
    > and it would put a folder under `output_root` with a lifetime question attached. In memory
    > there is nothing to clean up and nothing to invalidate but the process ending.

    **What it holds is pristine detector output.** `screen_nuclei` never writes into the array
    it is given: `set_detection` stores it as `nucleus_labels`, and every hand edit goes through
    `edited_labels`, which copies first and lands in `shown_labels`. So the entry stays what the
    detector produced, and the user's parked decision — that item 4's hand edits do not reach
    the numbers — holds **by construction** rather than by anyone being careful.
    """

    capacity: int = NUCLEUS_CACHE_ENTRIES
    _entries: dict[tuple, object] = field(default_factory=dict, repr=False)
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple, compute: Callable[[], object]) -> object:
        """The detection for ``key``, running ``compute`` only on a miss."""
        found = self.peek(key)
        if found is not None:
            return found
        self.misses += 1
        value = compute()
        self.put(key, value)
        return value

    def peek(self, key: tuple):
        """What is held for ``key``, or `None` — **without computing anything.**

        `derive` needs exactly this: reuse item 4's detection when it is there, and fall back to
        letting `segment_image` detect for itself when it is not.
        """
        if key not in self._entries:
            return None
        self.hits += 1
        value = self._entries.pop(key)          # reinsert: most recently used
        self._entries[key] = value
        return value

    def put(self, key: tuple, value) -> None:
        self._entries[key] = value
        while len(self._entries) > self.capacity:
            self._entries.pop(next(iter(self._entries)))

    def release(self) -> None:
        """Drop every held detection. ~212 MB each, so this is the one worth calling."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple) -> bool:
        return key in self._entries
