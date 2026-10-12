"""Finding connected objects in the cell mask.

Deliberately the least clever module here. Connected components only ever *fuses*
touching structures — it can never divide one — which is exactly the property wanted:
every decision to separate cells belongs to the nuclei, and no earlier stage may
pre-empt them.

That was a real choice over PrestoCell's persistence clustering, which separates on
intensity before nuclei are consulted. Because a zero-nucleus blob is kept whole by
design, an over-segmenting coarse step would manufacture a spurious nucleus-free
"cell" that nothing downstream corrects. Connected components cannot do that. It also
confines the expensive clustering to the minority of blobs that actually hold two or
more nuclei, and runs in milliseconds rather than minutes.

**26-connectivity**, matching 3DMorph's `bwconncomp(NoiseIm, 26)`. Microglial processes
are frequently one voxel wide and often connect only diagonally; 6-connectivity would
sever them and scatter each cell into fragments.

Bounding boxes are produced here because everything downstream works on crops. Splitting
a clump inside its own bounding box rather than across the full volume is what keeps
that step affordable at 200 M voxels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

__all__ = ["Objects", "find_cell_objects"]


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    raise ValueError(
        f"connectivity must be 6, 18 or 26, got {connectivity}"
    )


@dataclass
class Objects:
    """Connected components of a cell mask, with the geometry callers need."""

    labels: np.ndarray                              # (z, y, x) int32; 0 = background
    n_objects: int
    sizes: np.ndarray                               # voxels; element i is label i+1
    slices: list[tuple[slice, slice, slice] | None]  # element i is label i+1
    connectivity: int = 26

    def __post_init__(self) -> None:
        if self.sizes.size != self.n_objects:
            raise ValueError(
                f"sizes has {self.sizes.size} entries for {self.n_objects} objects"
            )

    def summary(self) -> str:
        if not self.n_objects:
            return "no objects"
        return (f"{self.n_objects} objects, "
                f"{int(self.sizes.min())}-{int(self.sizes.max())} voxels "
                f"(median {int(np.median(self.sizes))})")

    def bbox(self, label: int) -> tuple[slice, slice, slice]:
        """Tight bounding box of one object, as numpy slices."""
        if not 1 <= label <= self.n_objects:
            raise KeyError(f"no object with label {label}")
        box = self.slices[label - 1]
        if box is None:
            raise KeyError(f"object {label} is not present in the label volume")
        return box

    def crop(
        self,
        label: int,
        *,
        pad_um: float = 0.0,
        voxel_size_um: Sequence[float] | None = None,
    ) -> tuple[slice, slice, slice]:
        """Bounding box widened by a **physical** margin and clipped to the volume.

        Padding is specified in microns rather than voxels because the stack is ~5.5:1
        anisotropic: a uniform voxel margin would reach five and a half times further
        in z than in plane, which is rarely what a caller means.
        """
        box = self.bbox(label)
        if pad_um <= 0:
            return box
        if voxel_size_um is None:
            raise ValueError("pad_um requires voxel_size_um")

        out = []
        for axis, (sl, size_um) in enumerate(zip(box, voxel_size_um)):
            pad = int(np.ceil(pad_um / float(size_um)))
            out.append(slice(max(0, sl.start - pad),
                             min(self.labels.shape[axis], sl.stop + pad)))
        return tuple(out)                                    # type: ignore[return-value]

    def volumes_um3(self, voxel_size_um: Sequence[float]) -> np.ndarray:
        """Object volumes in cubic microns."""
        voxel = float(np.prod([float(v) for v in voxel_size_um]))
        return self.sizes.astype(np.float64) * voxel

    def mask_for(self, label: int) -> np.ndarray:
        """Boolean mask of one object, cropped to its bounding box."""
        return self.labels[self.bbox(label)] == label


def find_cell_objects(
    cell_mask: np.ndarray,
    *,
    connectivity: int = 26,
) -> Objects:
    """Label the connected components of a boolean cell mask.

    ``cell_mask`` was documented as needing to be the **hole-filled** mask, because Iba1 is
    cytoplasmic and an unfilled soma is a shell around a nuclear cavity — so a membership test
    run against it would be measuring against a hole.

    **That warning no longer applies to this caller, since the debris patch of 2026-08-24.**
    The pipeline hands this the **unfilled** measurement mask on purpose, and the membership
    test it worried about runs somewhere else entirely: `filter_nuclei_by_membership` tests
    nuclei against the **segmentation** mask, which is still filled. Connected components on a
    shell finds the same one object either way.

    Recorded rather than deleted because the reasoning is still correct for any *other* caller
    that wants to ask a containment question of the result.
    """
    if cell_mask.ndim != 3:
        raise ValueError(f"expected a (z, y, x) mask, got shape {cell_mask.shape}")

    labels, n = ndimage.label(
        cell_mask.astype(bool, copy=False), structure=_structure(connectivity)
    )
    labels = labels.astype(np.int32, copy=False)

    sizes = (np.bincount(labels.ravel(), minlength=n + 1)[1:]
             if n else np.zeros(0, dtype=np.int64))
    slices = ndimage.find_objects(labels) if n else []

    return Objects(
        labels=labels,
        n_objects=int(n),
        sizes=sizes,
        slices=list(slices),
        connectivity=connectivity,
    )
