"""Foreground masks from raw channels.

Channel-agnostic by design: nuclei, cell, CD68 and plaque all threshold the same way,
each with its own coefficient. Nothing here is Iba1-specific.

Three mechanics, each answering a specific failure:

**Per-slice Otsu.** Confocal signal attenuates with depth, so a single global threshold
under-detects deep slices and over-detects shallow ones — biasing measured cell volume
by z position. 3DMorph thresholds per slice; PrestoCell uses one global threshold and
inherits the bias.

**An empty-slice guard.** Neither parent has one. Otsu on a slice containing only
background is unimodal, so it splits the noise near its mean and turns roughly half the
slice into "signal". Slices above or below the tissue therefore produce garbage
foreground unless explicitly suppressed.

**Hole filling.** Iba1 is cytoplasmic, so a nucleus is a *hole* in the cell mask, not
part of it. Measured on real data, filling raises the number of nuclei found inside
cells from 0 to 3 at Otsu, 1 to 8 at Otsu x 0.5, and 6 to 12 at Otsu x 0.25. Without
it, nucleus membership fails and clumped cells are never split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

from . import cpu

__all__ = [
    "ThresholdResult",
    "threshold_channel",
    "fill_soma_holes",
    "build_cell_mask",
    "slice_contrast",
]


def slice_contrast(plane: np.ndarray, high: float = 99.9) -> float:
    """Intensity spread of one slice: a scale-free proxy for 'does this contain signal'."""
    return float(np.percentile(plane, high) - np.median(plane))


@dataclass
class ThresholdResult:
    """A mask plus the reasoning behind it, kept for diagnostics and reproducibility."""

    mask: np.ndarray
    thresholds: np.ndarray            # per-slice threshold actually applied
    empty_slices: tuple[int, ...]     # slices suppressed by the guard
    coefficient: float

    @property
    def foreground_fraction(self) -> float:
        return float(self.mask.mean())

    def summary(self) -> str:
        empties = (f", {len(self.empty_slices)} slice(s) suppressed"
                   if self.empty_slices else "")
        return (f"{self.foreground_fraction:.2%} foreground at coefficient "
                f"{self.coefficient:g}{empties}")


def threshold_channel(
    volume: np.ndarray,
    *,
    coefficient: float = 1.0,
    min_object_voxels: int = 0,
    empty_slice_ratio: float = 0.05,
    max_foreground_fraction: float = 0.40,
    volume_contrast: float | None = None,
    threads: int | None = None,
) -> ThresholdResult:
    """Per-slice Otsu threshold of one channel, with an empty-slice guard.

    Args:
        volume: ``(z, y, x)`` intensities.
        coefficient: multiplier on each slice's Otsu threshold. Below 1 lowers the
            threshold to capture fine processes — which is what makes neighbouring
            cells touch, and hence why nucleus-gated splitting exists.
        min_object_voxels: drop connected components smaller than this (3DMorph's
            ``noise`` parameter). Applied in 3D, after thresholding.
        empty_slice_ratio: a slice whose intensity spread is below this fraction of the
            whole volume's spread is treated as containing no signal. Guards against
            Otsu splitting pure noise on slices outside the tissue.
        max_foreground_fraction: a slice yielding more foreground than this is also
            treated as empty. This is the absolute backstop the relative test cannot
            provide: in a volume that is blank *throughout*, every slice matches the
            volume's own contrast, so nothing looks anomalous — yet Otsu still splits
            the noise near its median and calls about half of it signal. Real fields
            here run around 2% foreground, so 40% is far above anything genuine.
        volume_contrast: the whole stack's contrast, when ``volume`` is only part of it.

            **Added 2026-08-19 for the interactive preview, and it fixes a real divergence.**
            The empty-slice guard is *volume-relative*: it suppresses any plane whose contrast
            falls below ``empty_slice_ratio`` of the **whole stack's**. Hand this function a
            single slice — which is exactly what items 3 and 5 do, 0.25 s per preview against
            ~10 s for the stack — and that slice becomes its own reference, so the test reads
            ``x < x * 0.05`` and **can never fire**.

            Measured on a dim but internally bimodal slice: suppressed in the stack with 0.0000%
            foreground, and 4.7852% previewed alone. The preview would show a mask the commit
            then blanks, which is the *complete-looking but wrong* failure this module already
            refuses for ``min_object_voxels``, arriving through a different door.

            ``max_foreground_fraction`` still backstops **pure noise**, where Otsu splits near
            the median and yields ~50% foreground. It does not catch dim-but-structured slices,
            which is what the top and bottom of a stack actually contain.

            Defaults to ``None`` = derive it from ``volume``, which is byte-for-byte today's
            behaviour. Callers previewing one plane pass ``slice_contrast(whole_volume)``,
            computed once when the image loads.
        threads: how many threads the per-slice loop may use. `None` takes the process count,
            which is 1 unless a scheduler or ``--threads`` said otherwise. See `cpu.count`.

    > ### ⇒ THE THREADING IS INSIDE THIS FUNCTION, AND THAT IS NOT A STYLE CHOICE
    >
    > `HANDOFF_PATCH_CPU.md` §1. The obvious way to parallelise this is from outside — chop the
    > volume into slabs and call this on each. **It is wrong, and it was measured wrong: 60,453
    > voxels differ on a 24x512x512 fixture.**
    >
    > ``volume_contrast`` is a *whole-volume* reduction and it is the reference the empty-slice
    > guard compares each plane against. Give a slab to this function and the slab becomes its
    > own reference — the same divergence the ``volume_contrast`` argument was added to fix in
    > 2026-08-19, arriving through a third door.
    >
    > Threading the loop **after** ``floor`` is computed makes the preamble run once by
    > construction, so the trap is unreachable rather than merely avoided. Each worker owns a
    > contiguous span of z and writes only ``mask[z]``, ``thresholds[z]`` and ``is_empty[z]``
    > for its own planes; nothing is shared and nothing is joined.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")

    if volume_contrast is None:
        volume_contrast = slice_contrast(volume)
    floor = volume_contrast * empty_slice_ratio

    mask = np.zeros(volume.shape, dtype=bool)
    thresholds = np.full(volume.shape[0], np.nan, dtype=np.float64)
    #: **An array, not the list this used to append to.** The loop below runs on several
    #: threads, and `list.append` would record the planes in whatever order the workers
    #: happened to finish — `empty_slices` is a z-ordered tuple and a `ThresholdResult` that
    #: reported its empty planes shuffled would be a different object every run. Setting a flag
    #: at index z and reading the array out in order gives back exactly the serial answer.
    is_empty = np.zeros(volume.shape[0], dtype=bool)

    def work(lo: int, hi: int) -> None:
        for z in range(lo, hi):
            plane = volume[z]
            if volume_contrast > 0 and slice_contrast(plane) < floor:
                is_empty[z] = True
                continue
            try:
                level = float(threshold_otsu(plane)) * coefficient
            except ValueError:          # constant slice
                is_empty[z] = True
                continue

            candidate = plane > level
            if candidate.mean() > max_foreground_fraction:
                is_empty[z] = True
                continue

            thresholds[z] = level
            mask[z] = candidate

    cpu.run_slabs(work, volume.shape[0], cpu.count(threads))
    empty = [int(z) for z in np.flatnonzero(is_empty)]

    if min_object_voxels > 0 and mask.any():
        mask = remove_small_objects(mask, min_size=min_object_voxels, connectivity=3)

    return ThresholdResult(
        mask=mask,
        thresholds=thresholds,
        empty_slices=tuple(empty),
        coefficient=coefficient,
    )


def fill_soma_holes(
    mask: np.ndarray,
    *,
    voxel_size_um: Sequence[float] | None = None,
    closing_radius_um: float = 0.0,
    fill_holes: bool = True,
    threads: int | None = None,
) -> np.ndarray:
    """Close the nucleus-shaped cavity inside each soma.

    Iba1 stains cytoplasm, so a soma appears as a shell around an unstained nucleus.
    Downstream steps — nucleus membership, soma erosion, cell volume — all need the
    soma solid, so filling happens once here rather than being re-derived (or
    forgotten) by each consumer.

    Two passes, because neither alone suffices. A 3D fill closes cavities fully
    enclosed by signal; thresholding almost always leaves gaps in the shell, so a
    per-slice fill then closes rings that are enclosed within their own plane even
    where the 3D shell leaks.

    ``closing_radius_um`` optionally bridges small gaps in the shell first. It is
    applied **in xy only** and sized in microns: with 5.5:1 anisotropy an isotropic
    structuring element in voxels would reach ~5.5x further in z than in the plane.

    > ### THE CLOSING IS PADDED, AND WITHOUT IT IT DELETED THE EDGE OF THE FIELD
    >
    > A closing is a dilation followed by an erosion, and ``ndimage.binary_closing``
    > takes scipy's default ``border_value=0``. So the erosion half asks *"is the whole
    > disk centred here inside the foreground?"* of a voxel near the array edge, finds
    > that part of the disk hangs off the array, is told that part is background — and
    > deletes the voxel. Not because of the image; because the disk ran out of array.
    >
    > **Every foreground voxel within one element radius of the field edge was erased,
    > however bright.** On the demo crop at ``closing_radius_um=0.8`` that is a 7 px
    > (0.76 um) band and **8491 voxels of thresholded Iba1**, 100% of them inside the
    > band; it cost the largest object 1.79% of its volume and an edge-hugging cell
    > **6.27%**. It also *manufactured objects*: severing the band left two slivers on
    > the first surviving column as their own connected components, 10 voxels and 1
    > voxel, which is `SHAKEOUT_FINDINGS.md` O1.
    >
    > **Padding by exactly the element radius is measured to be exact** — identical to
    > padding by 4x and by 8x, i.e. to an unbounded canvas, at radii 0.4 / 0.8 / 1.6 /
    > 3.0 um. It is not a tuning knob. Outside the band **0 voxels change**, and the
    > closing adds *less* at the edge than elsewhere (113 vs 216 added per 1000 mask
    > voxels), so nothing is invented there.
    >
    > ``border_value=1`` was the other candidate and is wrong: it dilates the frame
    > inward, inventing signal along every edge — 550 voxels added on a 1050-voxel test
    > shape. **The program's own position on edge material is `filters.py`'s
    > ``remove_xy_border``, which defaults to False because it "removes genuine cells,
    > so it is the caller's decision".** This was doing an unasked version of that one
    > layer earlier, in microns nobody chose.

    **The result is a superset of ``mask``**, which is the invariant everything
    downstream assumes — ``find_cell_objects`` can fuse objects but never divide one,
    so the object count may fall from raw to filled and must never rise. It holds with
    ``fill_holes=False`` too: a closing is extensive.

    > ### ``fill_holes=False`` IS CLOSING WITHOUT FILLING, AND THE MEASUREMENT MASK WANTS IT
    >
    > Added by the debris patch, 2026-08-24. Hole filling exists so ``membership_fractions``
    > works — Iba1 is cytoplasmic, so an unfilled mask leaves each nucleus in a hole and every
    > fraction collapses toward zero. **Membership runs against the segmentation mask**, so the
    > measurement mask needs none of it, and filling there inflates every exported volume by the
    > lumen of each soma.
    >
    > **The two operations are separated rather than the caller taking ``raw``**, and that is
    > the whole reason this flag exists instead of a field access. ``raw`` skips the closing as
    > well, which would leave ``mask_closing_radius_um`` — a control item 5 shows, with its own
    > help text — silently doing nothing to the mask it names. Gap-bridging and lumen-filling
    > are different operations answering different objections; only the second one was argued
    > against. **Flagged for the user's review**: the patch brief said to take an unfilled mask
    > and assumed the two were already separable, and they were not.

    > ### ⇒ TWO OF THE FOUR PIECES ARE THREADED, AND THE SPLIT IS EXACT — CPU patch, §3 A5
    >
    > **Threaded:** the padded xy closing, because ``element`` is ``disk[np.newaxis]`` — one
    > slice thick, so the erosion never reaches past the plane it is centred on and each output
    > plane depends only on its own input plane; and the per-slice `binary_fill_holes` loop,
    > which says per-slice in its own name.
    >
    > **Not threaded:** the 3D `binary_fill_holes` above, which is what closes cavities the
    > per-slice pass cannot see, and the trailing `remove_small_objects`, which labels in 3D.
    >
    > **The closing algorithm is not touched.** A distance-transform formulation was measured
    > and rejected: at the user's ``closing_radius_um`` of 0.8 it is **1.9x slower** and
    > produces a mask differing in **10.1% of voxels**. Recorded so nobody tries it again.
    >
    > At the user's measurement settings — ``mask_closing_radius_um`` 0 with
    > ``fill_holes=False`` — this function returns its input unchanged and neither branch runs.
    """
    n_threads = cpu.count(threads)
    #: **Serial, and it stays serial.** A 3D fill closes cavities enclosed across planes, so
    #: each output plane depends on every other one. Only the two xy-only pieces below are
    #: strictly per-plane, and only those are threaded.
    filled = ndimage.binary_fill_holes(mask) if fill_holes else mask

    if closing_radius_um > 0:
        if voxel_size_um is None:
            raise ValueError("closing_radius_um requires voxel_size_um")
        _, y_um, x_um = (float(v) for v in voxel_size_um)
        ry = max(1, int(round(closing_radius_um / y_um)))
        rx = max(1, int(round(closing_radius_um / x_um)))
        yy, xx = np.ogrid[-ry:ry + 1, -rx:rx + 1]
        disk = ((yy / ry) ** 2 + (xx / rx) ** 2) <= 1.0
        element = disk[np.newaxis]                      # xy only, one slice thick
        #: Padded in xy only, by exactly the element's reach, so no voxel of the real
        #: field ever has the disk fall off the array. z needs none: the element is one
        #: slice thick, so the erosion never reaches past the plane it is centred on.
        padded = np.pad(filled, ((0, 0), (ry, ry), (rx, rx)),
                        mode="constant", constant_values=False)
        #: Written into a fresh array rather than sliced out of one — the per-slice fill below
        #: writes into it, and a view would keep the whole padded array alive to be written
        #: through. That was a `.copy()` before it was threaded; it is now the output each
        #: worker writes its own planes into, which does the same job and one fewer allocation.
        closed = np.empty(filled.shape, dtype=bool)

        def close(lo: int, hi: int) -> None:
            closed[lo:hi] = ndimage.binary_closing(
                padded[lo:hi], structure=element)[:, ry:-ry, rx:-rx]

        cpu.run_slabs(close, filled.shape[0], n_threads)
        filled = closed

    if fill_holes:
        #: **Only this loop, and only ever this loop.** The 3D `binary_fill_holes` above stays
        #: serial: it closes cavities enclosed across planes, so its output at z depends on
        #: every other plane and slabbing it would change the answer.
        def fill(lo: int, hi: int) -> None:
            for z in range(lo, hi):
                filled[z] = ndimage.binary_fill_holes(filled[z])

        cpu.run_slabs(fill, filled.shape[0], n_threads)

    return filled


@dataclass
class CellMask:
    """The cell channel's foreground, before and after the soma cavities are closed.

    **``filled`` is the product, ``raw`` is the diagnostic**, and since the debris patch
    ``filled`` means *closed, and hole-filled unless the caller passed ``fill_holes=False``*.
    The measurement mask is the one caller that passes it — see `fill_soma_holes`.
    """

    raw: np.ndarray
    filled: np.ndarray
    threshold: ThresholdResult

    @property
    def filled_gain(self) -> float:
        """Extra volume recovered by filling — the soma interiors, essentially."""
        raw = int(self.raw.sum())
        return float((int(self.filled.sum()) - raw) / raw) if raw else 0.0


def build_cell_mask(
    cell_volume: np.ndarray,
    *,
    coefficient: float = 1.0,
    min_object_voxels: int = 0,
    voxel_size_um: Sequence[float] | None = None,
    closing_radius_um: float = 0.0,
    empty_slice_ratio: float = 0.05,
    fill_holes: bool = True,
    threads: int | None = None,
) -> CellMask:
    """Threshold the cell channel and close its soma cavities.

    Returns both masks: the filled one is what every downstream step should use, while
    the raw one is retained because the *cell's shape* — its processes — is better
    represented before filling, and because the difference between them is a useful
    diagnostic.

    > ### ``min_object_voxels`` IS ENFORCED TWICE, AND THE SECOND TIME IS THE ONE THAT COUNTS
    >
    > ``threshold_channel`` applies it to the **raw** mask. But ``filled`` is what
    > ``find_cell_objects`` labels, what item 3 counts and draws, and what splitting
    > runs on — and it is two operations downstream, so anything those two operations
    > *create* has never been past the floor. Both known creators are real: the closing's
    > border erosion severed 10- and 1-voxel slivers off a real object (O1), and even
    > once padded a closing can add a lone voxel whose whole disk fits inside the dilated
    > mask but which touches nothing — seen at ``closing_radius_um`` 0.4 and 1.6 on the
    > demo crop, mid-field, nowhere near an edge.
    >
    > **With the padded closing this second pass cannot touch a genuine object, and that
    > is structural rather than lucky.** ``fill_soma_holes`` returns a superset of its
    > input, so every filled component contains a raw component that already cleared the
    > floor and is therefore at least as large. What fails the floor here must contain
    > **no thresholded signal at all**. Verified across four floors x three radii on the
    > demo crop: 1-5 voxels removed, every one of them from a component with zero raw
    > signal.
    >
    > **It lands here rather than in ``fill_soma_holes``** because ``min_object_voxels``
    > already lives in this signature; the filler keeps its one job and its own tests
    > keep their meaning. Both the interactive commit and ``pipeline.py`` come through
    > here, so the screen's number stays the number the pipeline finds.
    >
    > **Cost:** one extra 3D labelling, measured at **5.1 s** on a 40 x 2304 x 2304 field
    > against a ~59 s commit. The alternative — leave it, and let item 6's cell-size
    > cutoff sweep up the specks downstream — was rejected because item 3's readout and
    > item 6's list would still both count them, and the control is named for the
    > property it would not have.
    """
    #: **Forwarded rather than re-resolved.** Both callees would reach `cpu.count` themselves
    #: and get the same answer, but a test asking this function for the threaded path has to be
    #: able to say so once and have it hold all the way down.
    result = threshold_channel(
        cell_volume,
        coefficient=coefficient,
        min_object_voxels=min_object_voxels,
        empty_slice_ratio=empty_slice_ratio,
        threads=threads,
    )
    filled = fill_soma_holes(
        result.mask,
        voxel_size_um=voxel_size_um,
        closing_radius_um=closing_radius_um,
        fill_holes=fill_holes,
        threads=threads,
    )
    if min_object_voxels > 0 and filled.any():
        filled = remove_small_objects(
            filled, min_size=min_object_voxels, connectivity=3
        )
    return CellMask(raw=result.mask, filled=filled, threshold=result)
