"""Moving between native and segmentation resolution.

Segmentation runs on xy-downsampled data; measurement runs at native resolution. The
reasoning (`DESIGN_RATIONALE.md` §4.11) is that cost and sensitivity sit in different
places: whole-volume clustering is expensive but tolerant of coarser voxels, while
per-cell morphometry is resolution-sensitive but cheap, running on one bounding box at
a time.

Downsampling by 2 in xy also puts nuclei at ~30 px, which is Cellpose's operating
point — at native resolution they are 55-74 px and the model internally rescales by
``diameter/30`` anyway, so full resolution is wasted work.

**z is never downsampled.** With 0.6 µm slices against 0.108 µm pixels the stack is
already coarse in z; reducing it further would destroy the axis that most limits
measurement.

The important operation here is :func:`assign_labels_to_native`. Upsampling a label
mask alone would give blocky, 2-voxel-quantised cell boundaries. Instead the upsampled
labels act only as a *region assignment*, and the cell's actual shape comes from the
native-resolution mask — which is what preserves the fine processes that carry the
biological signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "downsample_xy",
    "downsample_mask_xy",
    "upsample_xy",
    "assign_labels_to_native",
    "assign_labels_by_connectivity",
    "MeasurementMask",
    "voxel_size_after_downsample",
]


def _pad_to_multiple(volume: np.ndarray, factor: int) -> np.ndarray:
    """Edge-pad y and x up to a multiple of ``factor``.

    Padding rather than cropping, so no data is silently discarded: one of the test
    crops is 427 px wide, which is odd, and losing its last row would shift every
    coordinate that follows.
    """
    _, y, x = volume.shape
    pad_y, pad_x = (-y) % factor, (-x) % factor
    if pad_y or pad_x:
        volume = np.pad(volume, ((0, 0), (0, pad_y), (0, pad_x)), mode="edge")
    return volume


def downsample_xy(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """Block-average a ``(z, y, x)`` volume in xy only.

    Averaging rather than subsampling: it preserves signal-to-noise, which matters for
    the faint fine processes that a lowered threshold is meant to capture.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return volume

    padded = _pad_to_multiple(volume.astype(np.float32, copy=False), factor)
    z, y, x = padded.shape
    return padded.reshape(z, y // factor, factor, x // factor, factor).mean(axis=(2, 4))


def downsample_mask_xy(mask: np.ndarray, factor: int = 2,
                       occupancy: float = 0.5) -> np.ndarray:
    """Downsample a boolean mask, keeping blocks at least ``occupancy`` full.

    An occupancy of 0.5 is a majority vote. Lower values dilate thin structures rather
    than dropping them, which is the safer direction for microglial processes — they
    are frequently one or two voxels wide and would otherwise vanish.
    """
    if factor == 1:
        return mask.astype(bool, copy=False)
    return downsample_xy(mask.astype(np.float32), factor) >= occupancy


def upsample_xy(array: np.ndarray, factor: int,
                target_yx: Sequence[int] | None = None) -> np.ndarray:
    """Nearest-neighbour expansion in xy, trimmed back to the original size."""
    if factor == 1:
        out = array
    else:
        out = np.repeat(np.repeat(array, factor, axis=1), factor, axis=2)
    if target_yx is not None:
        ty, tx = int(target_yx[0]), int(target_yx[1])
        out = out[:, :ty, :tx]
    return out


def assign_labels_to_native(
    labels_small: np.ndarray,
    native_mask: np.ndarray,
    factor: int,
    *,
    max_search_um: float | None = None,
    voxel_size_um: Sequence[float] | None = None,
) -> np.ndarray:
    """Give every native-resolution foreground voxel the label of its region.

    The upsampled labels decide only *which cell* a voxel belongs to; the cell's shape
    comes from ``native_mask``. Upsampling the label mask on its own would quantise
    every boundary to the downsample factor and blunt exactly the thin processes the
    analysis measures.

    Foreground voxels left unlabelled — a process that survives at native resolution
    but fell outside the coarse label — are assigned to the nearest labelled voxel
    rather than discarded, so orphaned fragments rejoin the cell they came from.
    """
    if labels_small.shape[0] != native_mask.shape[0]:
        raise ValueError(
            f"z extent differs: labels {labels_small.shape[0]} vs "
            f"native {native_mask.shape[0]}. z is never downsampled."
        )

    upsampled = upsample_xy(labels_small, factor, native_mask.shape[1:])
    out = np.where(native_mask, upsampled, 0).astype(labels_small.dtype, copy=False)

    orphan = native_mask & (out == 0)
    if not orphan.any() or not (out > 0).any():
        return out

    # Distance transform in physical units, so an anisotropic stack does not make z
    # neighbours look artificially close.
    sampling = None
    if voxel_size_um is not None:
        sampling = tuple(float(v) for v in voxel_size_um)

    # `return_distances` defaults to True, so asking only for indices still allocates and
    # fills a float64 distance array the size of the volume — 1.7 GB at the production
    # 2304x2304x40, thrown away unread. Ask for the distances only when `max_search_um`
    # needs them, and then take both from the **same** call: the previous code discarded
    # them here and recomputed the identical whole-volume transform below.
    want_distance = max_search_um is not None
    background = out == 0
    result = ndimage.distance_transform_edt(
        background, sampling=sampling,
        return_distances=want_distance, return_indices=True,
    )
    distance, indices = result if want_distance else (None, result)
    nearest = out[tuple(indices)]

    if want_distance:
        nearest = np.where(distance <= max_search_um, nearest, 0)

    out[orphan] = nearest[orphan]
    return out


@dataclass
class MeasurementMask:
    """A permissive cell mask, plus what it could not account for.

    ``unassigned_*`` describes foreground that appears only at the measurement threshold
    and connects to no cell. It is deliberately **not** attached to anything and
    deliberately **not** size-filtered: nothing downstream reads it, because every
    measurement is per cell. It is reported instead, because a large unassigned component
    is evidence that the *segmentation* threshold missed a cell — which is information,
    not debris.
    """

    labels: np.ndarray
    n_unassigned: int = 0
    unassigned_voxels: int = 0
    largest_unassigned_um3: float = 0.0

    def summary(self) -> str:
        if not self.n_unassigned:
            return "all measurement foreground assigned to cells"
        return (f"{self.n_unassigned} unassigned components, "
                f"{self.unassigned_voxels} voxels, "
                f"largest {self.largest_unassigned_um3:.0f} um3")


def assign_labels_by_connectivity(
    cell_labels: np.ndarray,
    measure_mask: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
) -> MeasurementMask:
    """Grow cells into a permissive mask, but only where they are physically connected.

    The second of the two masks (`DESIGN_RATIONALE.md` §4.13.4). ``cell_labels`` comes
    from the conservative threshold and has already decided **how many cells there are
    and where they divide**; this only decides **how far each one extends**.

    A process must be continuous with its cell, so assignment follows connected
    components of the measurement mask rather than nearest-neighbour distance:

    * a component containing **exactly one** cell becomes that cell entirely — this is
      what recovers a dim process the segmentation threshold had severed;
    * a component containing **several** cells is divided between them by nearest label,
      because two interdigitating arbours genuinely need arbitrating;
    * a component containing **no** cell is left alone and counted.

    That last rule is the point. Nearest-neighbour assignment — what
    :func:`assign_labels_to_native` does — would attach a detached blob of debris to
    whichever cell happened to be closest, and it would then enter that cell's skeleton
    as a branch and its mask as CD68 territory, silently. PrestoCell avoids the same trap
    the same way, propagating only along adjacency from nucleus-bearing seeds
    (``main.py:664-678``); this is that guarantee on a binary mask, where connectivity is
    direct and needs no graph.

    The measurement mask is unioned with the existing cells first, so a cell can never
    *shrink* — even if the caller passes a stricter threshold than the one that produced
    ``cell_labels``.
    """
    if cell_labels.shape != measure_mask.shape:
        raise ValueError(
            f"cell labels {cell_labels.shape} do not match measurement mask "
            f"{measure_mask.shape}"
        )

    z_um, y_um, x_um = (float(v) for v in voxel_size_um)
    voxel_volume = z_um * y_um * x_um
    occupied = cell_labels > 0
    mask = measure_mask.astype(bool, copy=False) | occupied

    components, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    out = np.zeros_like(cell_labels)
    if n_components == 0:
        return MeasurementMask(labels=out)

    # How many distinct cells each component holds, counted in one pass.
    n_cells = int(cell_labels.max())
    both = occupied & (components > 0)
    per_component = np.zeros((n_components + 1, n_cells + 1), dtype=bool)
    if both.any():
        pair = (components[both].astype(np.int64) * (n_cells + 1)
                + cell_labels[both].astype(np.int64))
        seen = np.bincount(pair, minlength=(n_components + 1) * (n_cells + 1))
        per_component = seen.reshape(n_components + 1, n_cells + 1) > 0
    counts = per_component[:, 1:].sum(axis=1)

    # One cell: the component is that cell, whole.
    sole = np.zeros(n_components + 1, dtype=cell_labels.dtype)
    single = np.flatnonzero(counts == 1)
    if single.size:
        sole[single] = per_component[single, 1:].argmax(axis=1) + 1
    single_mask = np.isin(components, single) & mask
    out[single_mask] = sole[components[single_mask]]

    # Several cells: arbitrate by nearest label, in physical units.
    contested = np.isin(components, np.flatnonzero(counts >= 2)) & mask
    if contested.any():
        out[contested & occupied] = cell_labels[contested & occupied]
        gap = contested & ~occupied
        if gap.any():
            # `return_distances` defaults to True, so asking only for indices would also
            # allocate and fill a float64 distance array the size of the volume — **1.7 GB at
            # the production 2304x2304x40**, thrown away unread. Nothing here reads a distance:
            # the arbitration is *which* label is nearest, never *how far*.
            #
            # The same defect was fixed in `assign_labels_to_native` above on 2026-08-11 and
            # this call site was missed, because the fix was applied where the failure appeared
            # rather than to every match in the file. With `return_distances=False` the call
            # returns the indices **alone**, so there is no tuple to unpack.
            indices = ndimage.distance_transform_edt(
                ~occupied, sampling=(z_um, y_um, x_um),
                return_distances=False, return_indices=True,
            )
            out[gap] = cell_labels[tuple(indices)][gap]

    # No cell: left unassigned, and reported rather than filtered.
    orphan_ids = np.flatnonzero(counts == 0)
    orphan_ids = orphan_ids[orphan_ids > 0]
    if orphan_ids.size == 0:
        return MeasurementMask(labels=out)

    sizes = np.bincount(components.ravel(), minlength=n_components + 1)
    orphan_sizes = sizes[orphan_ids]
    return MeasurementMask(
        labels=out,
        n_unassigned=int(orphan_ids.size),
        unassigned_voxels=int(orphan_sizes.sum()),
        largest_unassigned_um3=float(orphan_sizes.max() * voxel_volume),
    )


def voxel_size_after_downsample(
    voxel_size_um: Sequence[float], factor: int
) -> tuple[float, float, float]:
    """Physical voxel size at segmentation resolution. z is unchanged."""
    z, y, x = (float(v) for v in voxel_size_um)
    return (z, y * factor, x * factor)
