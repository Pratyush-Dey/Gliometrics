"""Deciding which nuclei belong to which cell.

Most nuclei in the field are not microglial. DAPI stains every cell in the tissue —
neurons, astrocytes, oligodendrocytes, endothelium — and only a minority sit inside an
Iba1-positive cell. On the marked test crop, 13 of 122 detected nuclei were microglial.
This module is the filter that separates them, and then decides which nucleus goes with
which cell object.

**Iba1 is cytoplasmic, so the nucleus is a hole in the signal.** That single fact
shapes everything here. The cell mask must be hole-filled before membership is
assessed, and even then a microglial nucleus is only partly enclosed. Measured against
hand-marked somata, confirmed microglial nuclei score 0.35-0.98 while 75 of the other
109 fall below 0.05 — so the discriminating threshold is around **0.30**, far below the
0.50 that seems natural.

PrestoCell uses a 1% overlap threshold (``intersection > 0.01``,
``main.py:1519-1520``). That looks careless but is not: with an unfilled, hollow mask a
nucleus barely grazes the cytoplasmic shell, so the threshold *has* to be permissive.
The cost is that a neighbouring neuron's nucleus qualifies just as easily. Filling the
mask removes the constraint and lets the criterion actually discriminate.

Every overlap here is computed by **label-pair counting in a single pass**, not by
scanning the volume once per (nucleus, cell) pair as PrestoCell does
(``main.py:1517-1518`` runs two full-volume ``np.where`` calls inside a nested loop).
At batch scale that difference is the whole feasibility of the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Membership",
    "membership_fractions",
    "overlap_table",
    "filter_nuclei_by_membership",
    "nuclei_per_object",
]


@dataclass
class Membership:
    """Which nuclei survived the membership test, and by how much."""

    labels: np.ndarray                  # surviving nuclei, relabelled 1..n
    fractions: np.ndarray               # fraction inside the cell mask, per input label
    kept: np.ndarray                    # input label ids that passed
    rejected: np.ndarray                # input label ids that did not
    threshold: float
    relabel_map: dict[int, int] = field(default_factory=dict)   # old id -> new id

    @property
    def n_kept(self) -> int:
        return int(self.kept.size)

    def summary(self) -> str:
        total = self.kept.size + self.rejected.size
        return (f"{self.n_kept}/{total} nuclei at least "
                f"{self.threshold:.0%} inside a cell")


def membership_fractions(
    nucleus_labels: np.ndarray,
    cell_mask: np.ndarray,
) -> np.ndarray:
    """Fraction of each nucleus lying inside ``cell_mask``.

    Returns an array indexed from 0, so element *i* describes label *i + 1*.

    Two ``bincount`` passes over the volume, regardless of how many nuclei there are.
    The obvious implementation — loop over labels, mask, and count — is quadratic in
    disguise and is what makes PrestoCell's matching step intractable at scale.

    ``cell_mask`` should be the **hole-filled** mask. Iba1 stains cytoplasm, so an
    unfilled mask leaves the nucleus sitting in a hole and every fraction collapses
    toward zero.
    """
    if nucleus_labels.shape != cell_mask.shape:
        raise ValueError(
            f"nucleus labels {nucleus_labels.shape} and cell mask {cell_mask.shape} "
            "must have the same shape"
        )
    n = int(nucleus_labels.max())
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    flat = nucleus_labels.ravel()
    total = np.bincount(flat, minlength=n + 1)[1:]
    inside = np.bincount(
        flat, weights=cell_mask.ravel().astype(np.float64), minlength=n + 1
    )[1:]
    return inside / np.maximum(total, 1)


def overlap_table(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel counts for every overlapping pair of labels.

    Returns ``(a_ids, b_ids, counts)``, listing only pairs that actually overlap.

    Restricting to voxels where both labels are non-zero keeps memory proportional to
    the overlaps present rather than to ``n_a x n_b``. A dense contingency table would
    be fine for a hundred labels and hopeless for tens of thousands.
    """
    if labels_a.shape != labels_b.shape:
        raise ValueError(
            f"label volumes must have the same shape, got {labels_a.shape} "
            f"and {labels_b.shape}"
        )
    both = (labels_a > 0) & (labels_b > 0)
    if not both.any():
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty

    a = labels_a[both].astype(np.int64)
    b = labels_b[both].astype(np.int64)
    stride = int(b.max()) + 1
    packed, counts = np.unique(a * stride + b, return_counts=True)
    return packed // stride, packed % stride, counts


def filter_nuclei_by_membership(
    nucleus_labels: np.ndarray,
    cell_mask: np.ndarray,
    *,
    threshold: float = 0.30,
) -> Membership:
    """Drop nuclei that are not sufficiently inside a cell, and relabel the rest.

    The default of 0.30 is measured, not assumed: at 0.50 three of thirteen confirmed
    microglial nuclei were discarded, and losing a nucleus means losing a split.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    fractions = membership_fractions(nucleus_labels, cell_mask)
    n = fractions.size
    ids = np.arange(1, n + 1, dtype=np.int64)
    passed = fractions >= threshold
    kept, rejected = ids[passed], ids[~passed]

    lookup = np.zeros(n + 1, dtype=np.int32)
    lookup[kept] = np.arange(1, kept.size + 1, dtype=np.int32)

    return Membership(
        labels=lookup[nucleus_labels],
        fractions=fractions,
        kept=kept,
        rejected=rejected,
        threshold=float(threshold),
        relabel_map={int(old): int(lookup[old]) for old in kept},
    )


def nuclei_per_object(
    nucleus_labels: np.ndarray,
    object_labels: np.ndarray,
    *,
    min_overlap_fraction: float = 0.10,
) -> dict[int, list[int]]:
    """Group nuclei by the cell object they sit in.

    This is what the splitting stage consumes: an object with two or more nuclei is a
    clump to divide, one nucleus means leave it whole, none means leave it whole too.

    A nucleus straddling two objects is assigned to whichever it overlaps most, so a
    nucleus is never counted twice and cannot cause both neighbours to be split.
    ``min_overlap_fraction`` ignores incidental grazing contact.
    """
    a_ids, b_ids, counts = overlap_table(nucleus_labels, object_labels)
    result: dict[int, list[int]] = {}
    if a_ids.size == 0:
        return result

    n = int(nucleus_labels.max())
    sizes = np.bincount(nucleus_labels.ravel(), minlength=n + 1)

    best: dict[int, tuple[int, int]] = {}       # nucleus -> (object, count)
    for nucleus, obj, count in zip(a_ids, b_ids, counts):
        nucleus, obj, count = int(nucleus), int(obj), int(count)
        if count / max(sizes[nucleus], 1) < min_overlap_fraction:
            continue
        if nucleus not in best or count > best[nucleus][1]:
            best[nucleus] = (obj, count)

    for nucleus, (obj, _) in sorted(best.items()):
        result.setdefault(obj, []).append(nucleus)
    return result
