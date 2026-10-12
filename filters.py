"""Removing what is not a whole cell.

Two filters, and they are asymmetric on purpose.

**Size** is applied by default. Thresholding leaves a great deal of debris: on the
marked test crop, 439 of 448 objects were fragments and only 9 exceeded 500 µm³. Those
carry no nucleus, so they are never split and never harm the segmentation — but they
must not reach the review interface, where 439 spurious entries per image would make
triage across 700 files impossible.

**Border removal is optional and off by default.** It discards real, correctly
segmented cells on the grounds that the frame cut them off, which is a scientific
judgement rather than a technical one, so it is asked rather than assumed.

**The xy/z asymmetry is deliberate.** Only cells touching the *lateral* edges are
candidates for removal. Cells touching the top or bottom of the stack are always kept:
the stack is 24 µm deep and microglia span ~50 µm, so almost every cell is truncated in
z and excluding them would discard nearly the whole dataset. 3DMorph reaches the same
place by padding a zero plane onto each end of z before calling `bwclearborder`; the
faces are inspected directly here, which does the same thing more legibly and without
copying the volume.

Nothing is dropped silently — every removed label is returned, so exclusions can be
counted and reported rather than quietly shrinking the n.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "FilterResult",
    "labels_touching_xy_border",
    "filter_cells",
]


@dataclass
class FilterResult:
    """Surviving cells, relabelled, plus a record of what went and why."""

    labels: np.ndarray
    n_before: int
    n_after: int
    removed_small: list[int] = field(default_factory=list)
    removed_border: list[int] = field(default_factory=list)
    relabel_map: dict[int, int] = field(default_factory=dict)   # old id -> new id

    def summary(self) -> str:
        parts = [f"{self.n_after}/{self.n_before} cells kept"]
        if self.removed_small:
            parts.append(f"{len(self.removed_small)} below size")
        if self.removed_border:
            parts.append(f"{len(self.removed_border)} touching the xy border")
        return ", ".join(parts)


def labels_touching_xy_border(labels: np.ndarray) -> np.ndarray:
    """Label ids present on any lateral face of the volume.

    The z faces are deliberately not examined — see the module docstring.
    """
    if labels.ndim != 3:
        raise ValueError(f"expected a (z, y, x) label volume, got {labels.shape}")

    found: set[int] = set()
    for face in (labels[:, 0, :], labels[:, -1, :], labels[:, :, 0], labels[:, :, -1]):
        found.update(np.unique(face).tolist())
    found.discard(0)
    return np.array(sorted(found), dtype=np.int64)


def filter_cells(
    labels: np.ndarray,
    *,
    min_volume_um3: float | None = None,
    min_voxels: int | None = None,
    voxel_size_um: Sequence[float] | None = None,
    remove_xy_border: bool = False,
) -> FilterResult:
    """Drop debris and, if asked, cells cut off by the lateral frame edges.

    Prefer ``min_volume_um3`` over ``min_voxels``: a voxel count means different things
    at native and segmentation resolution, so a threshold expressed in voxels silently
    changes meaning when the downsample factor changes.

    ``remove_xy_border`` defaults to ``False``. It removes genuine cells, so it is the
    caller's decision — surfaced as a question in the interactive mode and recorded in
    the parameters file, never applied on its own initiative.
    """
    if labels.ndim != 3:
        raise ValueError(f"expected a (z, y, x) label volume, got {labels.shape}")
    if min_volume_um3 is not None and min_voxels is not None:
        raise ValueError("give either min_volume_um3 or min_voxels, not both")
    if min_volume_um3 is not None and voxel_size_um is None:
        raise ValueError("min_volume_um3 requires voxel_size_um")

    n_before = int(labels.max())
    if n_before == 0:
        return FilterResult(labels=labels.copy(), n_before=0, n_after=0)

    sizes = np.bincount(labels.ravel(), minlength=n_before + 1)
    sizes[0] = 0

    threshold_voxels = 0
    if min_volume_um3 is not None:
        voxel = float(np.prod([float(v) for v in voxel_size_um]))   # type: ignore[arg-type]
        threshold_voxels = int(np.ceil(min_volume_um3 / voxel))
    elif min_voxels is not None:
        threshold_voxels = int(min_voxels)

    present = np.flatnonzero(sizes)
    doomed_small = (present[sizes[present] < threshold_voxels]
                    if threshold_voxels > 0 else np.zeros(0, dtype=np.int64))

    doomed_border = (labels_touching_xy_border(labels)
                     if remove_xy_border else np.zeros(0, dtype=np.int64))

    doomed = np.union1d(doomed_small, doomed_border).astype(np.int64)
    survivors = np.setdiff1d(present, doomed)

    lookup = np.zeros(n_before + 1, dtype=np.int32)
    lookup[survivors] = np.arange(1, survivors.size + 1, dtype=np.int32)

    return FilterResult(
        labels=lookup[labels],
        n_before=int(present.size),
        n_after=int(survivors.size),
        removed_small=[int(v) for v in doomed_small],
        removed_border=[int(v) for v in doomed_border],
        relabel_map={int(old): int(lookup[old]) for old in survivors},
    )
