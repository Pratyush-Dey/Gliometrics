"""Record definitions for the prestomorph cell pool.

This module defines *what a cell is* as a stored record, and nothing else — no I/O,
no database, no numpy arrays. Keeping it dependency-free makes it trivially testable
and means the record definition can be reasoned about in isolation.

Two record types, because some data is genuinely per-file and some is per-cell:

* :class:`FileRecord` — one per source image. Holds voxel size, channel roles,
  provenance, and whether that file's review has been cleared. The per-file review
  model ("accept rest of this file") needs somewhere to record that a file is done,
  and a cells-only schema has nowhere to put it.
* :class:`CellRecord` — one per segmented cell.

Conventions that are deliberate and load-bearing
------------------------------------------------
**The full source path is the identity, not the filename.** Filenames repeat across
genotype and region directories, so every on-disk name is keyed by a hash of the full
path (see :func:`make_file_uid`). Using the filename stem alone would cause two
``image1.tif`` in different folders to collide — silently overwriting one another's
data with no error raised.

**Array axis order is numpy ``(z, y, x)`` everywhere**, with physical scale carried
alongside as ``voxel_size_um = (z, y, x)``. MATLAB's row/column/plane convention is
converted only at the comparison boundary. 3DMorph mixes the two (``obj = [y,x,z]``)
and the resulting index swaps are a recurring source of confusion.

**Bounding boxes are half-open**, matching numpy slicing exactly: ``bbox = (z0, z1,
y0, y1, x0, x1)`` selects ``arr[z0:z1, y0:y1, x0:x1]``. PrestoCell's ``crop()`` uses
``data[z_min:z_max]`` where ``z_max`` is an inclusive maximum index, which silently
drops the last voxel on all three upper faces.

**Coordinates are in full-image voxel space, not crop-relative.** Inter-cell centroid
distance (3DMorph's dispersion metric) needs every centroid in one common frame;
storing them crop-relative would require every bounding box to reconstruct.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "CellStatus",
    "NOT_ANALYSED_STATUSES",
    "KEPT_STATUSES",
    "UNREVIEWED_STATUSES",
    "RejectReason",
    "SplitMethod",
    "HUMAN_EDIT_KINDS",
    "human_edit_kind",
    "FileRecord",
    "CellRecord",
    "make_file_uid",
    "make_cell_uid",
    "natural_sort_key",
    "utc_now",
]

#: **1 -> 2 on 2026-08-24**, the debris patch: `FileRecord` gained four debris fields.
#:
#: **2 -> 3 on 2026-08-25**, and it takes them straight back out. Debris is the only per-image
#: quantity that was ever written to `FileRecord`; every other one — `union_surveyed_um3`,
#: `integrated_intensity`, the CD68 figures — goes through `Pool.write_image_measurement`,
#: keyed by `params_hash` and computed in Phase C. `FileRecord` is written by `PoolWriter` at
#: **segment time**, before anyone has reviewed anything, so debris stored there answered
#: *"what did no DERIVED cell claim"* while the export needed *"what did no SURVIVING cell
#: claim"*. Those differ by exactly the cells rejected as debris — which is the whole point of
#: rejecting one. Two numbers, one name, and nothing to stop a reader joining the wrong one.
#:
#: Debris now lives with the other per-image measures and is recomputed in Phase C from the
#: measurement mask, which Phase C can rebuild because it already reopens the source image and
#: already reads `params.mask_coefficient`. See `phasec.measure_file`.
#:
#: **A removal needs the bump as much as an addition** — a v2 file names four fields v3 has no
#: home for. Free on the same terms as 1 -> 2: no pool exists.
#:
#: **3 -> 4 on 2026-08-30**: `CellRecord` gains `n_real_cells_here`, so that a `real_cell`
#: rejection carrying *"this is 3 cells I cannot afford to split"* reaches the counts instead of
#: being flattened to one row. `Resolution.n_cells` has carried that number out of the editor
#: correctly since the popup was built and **was read by nobody** — parked 2026-08-26, unparked
#: 2026-08-30. Free on the same terms again: **no pool exists on disk**, checked rather than
#: assumed, so there is nothing to migrate.
#: **4 -> 5 on 2026-09-01**: `CellRecord` gains `human_split`, so the per-cell provenance column
#: can say *how* a human was involved rather than only that one was. `split_method` cannot serve
#: — it records which **automatic** splitter produced a cell (`pipeline.py` sets it from
#: `split.method_by_cell`), and the user's rule is that an automatic split is not a human edit.
#: Free on the same terms again: no pool exists on disk.
SCHEMA_VERSION = 6

_MAX_STEM_CHARS = 48
_HASH_CHARS = 8
_NATSORT_WIDTH = 12

#: Roles a channel may be assigned. ``nuclei`` and ``cell`` are required; the others
#: are optional and simply absent when not acquired.
REQUIRED_CHANNEL_ROLES = ("nuclei", "cell")
OPTIONAL_CHANNEL_ROLES = ("cd68", "plaque")
VALID_CHANNEL_ROLES = REQUIRED_CHANNEL_ROLES + OPTIONAL_CHANNEL_ROLES


class CellStatus(str, Enum):
    """Lifecycle of a cell through the A -> B -> C phases.

    Subclasses :class:`str` so it serialises transparently to JSON and SQLite while
    still making a typo a lookup error rather than a silently non-matching string.
    """

    CANDIDATE = "candidate"   # written by Phase A, awaiting review
    CONFIRMED = "confirmed"   # accepted in Phase B, queued for Phase C
    REJECTED = "rejected"     # rejected in Phase B, never analysed
    ANALYSED = "analysed"     # Phase C complete


#: Statuses meaning "this cell will not be measured", i.e. terminal for Phase C.
#:
#: **One status, deliberately.** A ``DEFERRED`` status sat beside ``REJECTED`` between
#: 2026-08-11 and later the same day, so that a cell abandoned under time pressure could be
#: told apart from one judged to be debris. It was removed: the distinction it carried is
#: carried by :class:`RejectReason` instead, and a rejected cell already leaves Phase C
#: (``pending_measurement`` admits only ``CONFIRMED`` and ``CANDIDATE``), so
#: ``cells(status=REJECTED)`` filtered on ``reject_reason == real_cell`` recovers exactly the
#: same set with one state fewer for every consumer to handle.
#:
#: Exported through this constant rather than restated in ``export.py``, so what counts as
#: "not measured" is decided in one place.
NOT_ANALYSED_STATUSES = (CellStatus.REJECTED,)

#: Statuses meaning "a reviewer looked at this cell and **kept** it", i.e. judged it a real
#: microglion and left it in the analysis.
#:
#: ``ANALYSED`` is in here because it *is* ``CONFIRMED`` plus a finished measurement —
#: ``mark_analysed`` only ever promotes a confirmed cell — so a list holding ``CONFIRMED``
#: alone would silently drop every cell Phase C has already run on, which is most of them by
#: the time anyone exports.
#:
#: Used by ``images.csv``'s ``n_real_cells`` together with the ``real_cell`` reject reason:
#: kept cells and real cells removed for lack of time are both cells a reviewer judged real,
#: and that sum is the answer to "how many microglia were in this field".
KEPT_STATUSES = (CellStatus.CONFIRMED, CellStatus.ANALYSED)

#: Statuses meaning "nobody has judged this cell yet". A cell here is neither real nor debris
#: as far as the file is concerned, so **any** count that separates the two is undetermined
#: while one of these is present — see ``export._real_cells``, which blanks rather than
#: under-reports.
UNREVIEWED_STATUSES = (CellStatus.CANDIDATE,)


class RejectReason(str, Enum):
    """Why a cell was removed at review — **exactly two values.**

    This is the reviewer's flow and nothing more: on rejecting a cell they say whether it is
    debris or a real cell, and a real cell may carry a note. The one distinction that has to
    be countable is between

    * :attr:`DEBRIS` — not a real cell: an artefact, a false positive. Removing it is
      **correct** and the analysis is better for it; and
    * :attr:`REAL_CELL` — a real cell removed anyway, because the reviewer ran out of time or
      could not fix it. Removing it is a **data loss**, and one whose rate is expected to
      depend on genotype.

    **Why the second category exists at all**, since it is the reason the pair is worth
    recording: the cells that are hard to fix are the clumped, plaque-associated 5xFAD ones —
    this project's own tuning finds splitting failures to be 5xFAD-only. Deletions made under
    time pressure therefore fall preferentially on the population the study exists to measure.
    Without this category that loss leaves no trace anywhere, so it is not merely unmeasured
    but unmeasurable. The count must stay testable for genotype dependence; that is the point.

    **Two values and no more.** Every extra category is another decision at ~20,000
    repetitions, and one the reviewer does not use is worse than absent because it makes the
    counts look considered when they are not. An ``other`` category existed briefly and was
    removed for that reason. Anything a reviewer wants to add belongs in
    :attr:`CellRecord.reject_note`, not in a third box.

    An unrecognised value is **refused**, not coerced: ``RejectReason("debri")`` raises. There
    is no free-text history to migrate — no human has ever reviewed a cell — and guessing at
    what a stored string meant would move a cell between the two categories this enum exists
    to separate.
    """

    #: Not a real cell: debris, an artefact, a false positive. A correct removal.
    DEBRIS = "debris"

    #: A real cell that was removed anyway — the reviewer ran out of time, or could not fix
    #: it. A **data loss**, and the branch on which ``reject_note`` is offered.
    REAL_CELL = "real_cell"


#: The four values of the per-cell provenance column, in the order they are checked.
#:
#: **Named `kind` rather than `provenance` for a silly but real reason**: the word
#: *prove-nan-ce* contains `nan`, and `test_nan_is_written_as_an_empty_cell_not_the_word_nan`
#: scans the whole CSV for that substring. A column name that trips a NaN check is a
#: trap for whoever next edits that test. `kind` is also what the code already calls it.
HUMAN_EDIT_KINDS = ("none", "mask_edited", "split", "both")


def human_edit_kind(human_edited: bool, human_split: bool) -> str:
    """Was this cell fully automatic, or did a human make it? One value per cell.

    **`none` means the whole cell is the program's own work** — thresholded, split if at all by
    the automatic splitter, and never touched. That is what makes the column usable as an
    analysis filter: the reviewer can ask whether a WT/5xFAD difference survives dropping every
    hand-touched cell, which matters because **hand editing is genotype-correlated** — under
    splits are 5xFAD-only, so repair lands disproportionately on one arm of the comparison.

    **An automatic split is not a human edit**, the user's rule of 2026-09-01, and it is why
    `SplitMethod` cannot feed this. That enum records which splitter ran — `pbc` and `watershed`
    are what `pipeline.py` chose on its own, from `split.method_by_cell`. A cell marked
    `watershed` there had no human anywhere near it.

    ``human_split`` is a human splitting this cell's **nucleus** at item 4 or the **cell itself**
    in rich review. Neither is wired yet and `CellRecord.human_split` says why, so `split` and
    `both` are currently unreachable — defined, documented, and honestly always absent rather
    than quietly meaning something else.
    """
    if human_edited and human_split:
        return "both"
    if human_split:
        return "split"
    if human_edited:
        return "mask_edited"
    return "none"


class SplitMethod(str, Enum):
    """How this cell came to be a separate object."""

    UNSPLIT = "unsplit"          # a whole connected object; no split applied
    PBC = "pbc"                  # persistence-clustering fragment propagation
    WATERSHED = "watershed"      # nucleus-seeded intensity watershed
    MANUAL_GMM = "manual_gmm"    # hand-invoked Gaussian mixture split
    MANUAL = "manual"            # hand-drawn / hand-edited in the review UI


def utc_now() -> str:
    """Timezone-aware UTC timestamp in ISO-8601 form.

    Naive local timestamps are avoided: segmentation runs on Bunya and review happens
    on a local machine, potentially in a different timezone, and ordering by time
    across the two must remain meaningful.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _basename(path: str) -> str:
    """Last path component, handling both separators regardless of host OS.

    ``pathlib`` parses according to the *running* platform, so a Windows path string
    read on Linux would not split on backslashes. Paths recorded by a Phase A run on
    Bunya are read back during review on Windows, so separator handling must not
    depend on which machine is asking.
    """
    return re.split(r"[\\/]", str(path))[-1]


def _stem(path: str) -> str:
    """Filename without its final extension."""
    name = _basename(path)
    return name[: name.rfind(".")] if "." in name[1:] else name


def _sanitise(text: str) -> str:
    """Reduce to characters safe in a filename on every target filesystem."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def _normalise_path(path: str) -> str:
    """Canonical path string for hashing: forward slashes, no trailing separator."""
    return re.sub(r"[\\/]+", "/", str(path)).rstrip("/")


def make_file_uid(source_path: str) -> str:
    """Stable, collision-resistant identifier for a source image.

    Formed as ``<sanitised stem, truncated>_<sha1(full path)[:8]>``.

    The hash is over the *full path* because filenames repeat across directories and
    the path is the real identity. The stem is retained only so the value is
    human-recognisable when browsing the pool; it is truncated so that the resulting
    directory and file names stay comfortably inside Windows' 260-character limit.

    Deterministic by construction: re-running a file yields identical identifiers, so
    a failed batch task can simply be retried and will overwrite its own partial
    output rather than duplicating cells.

    Note that the identifier is bound to the path as recorded at write time. Moving
    the source data means a subsequent run produces different identifiers, which is
    the correct behaviour — it is a different location, hence different provenance.
    """
    stem = _sanitise(_stem(source_path))[:_MAX_STEM_CHARS]
    digest = hashlib.sha1(_normalise_path(source_path).encode("utf-8")).hexdigest()
    return f"{stem}_{digest[:_HASH_CHARS]}"


def make_cell_uid(file_uid: str, cell_index: int) -> str:
    """Identifier for one cell, namespaced under its source file."""
    if cell_index < 0:
        raise ValueError(f"cell_index must be non-negative, got {cell_index}")
    return f"{file_uid}__c{cell_index:04d}"


def natural_sort_key(text: str) -> str:
    """Return a string whose lexicographic order matches human 'natural' order.

    Digit runs are zero-padded to a fixed width so that plain ``ORDER BY`` in SQL
    sorts ``img2`` before ``img10``. Returning a *string* rather than a tuple is what
    lets the ordering live in the database instead of requiring every row to be
    pulled into Python and re-sorted.
    """
    return re.sub(r"\d+", lambda m: m.group().zfill(_NATSORT_WIDTH), text.lower())


def _as_int_tuple(value: Sequence[Any], n: int, name: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in value)
    if len(out) != n:
        raise ValueError(f"{name} must have {n} elements, got {len(out)}")
    return out


def _as_float_tuple(value: Sequence[Any], n: int, name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in value)
    if len(out) != n:
        raise ValueError(f"{name} must have {n} elements, got {len(out)}")
    return out


@dataclass(frozen=True, kw_only=True)
class FileRecord:
    """One source image and the state of its processing.

    Declared ``kw_only`` deliberately: with this many same-typed fields, positional
    construction would allow silent argument transposition.
    """

    file_uid: str
    source_path: str            # full path, exactly as supplied — the identity
    source_filename: str        # display label only
    source_stem: str            # display label only
    sort_key: str               # natural-sort key over the full path
    voxel_size_um: tuple[float, float, float]   # (z, y, x)
    image_shape: tuple[int, int, int]           # (z, y, x) of the full image
    channel_roles: dict[str, int]               # e.g. {"nuclei": 0, "cell": 1, ...}
    run_id: str
    params_hash: str | None = None
    created_at: str = ""
    review_complete: bool = False
    n_cells: int = 0

    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "voxel_size_um",
                           _as_float_tuple(self.voxel_size_um, 3, "voxel_size_um"))
        object.__setattr__(self, "image_shape",
                           _as_int_tuple(self.image_shape, 3, "image_shape"))
        object.__setattr__(self, "channel_roles", dict(self.channel_roles))
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())

        if any(v <= 0 for v in self.voxel_size_um):
            raise ValueError(f"voxel_size_um must be positive, got {self.voxel_size_um}")
        if any(v <= 0 for v in self.image_shape):
            raise ValueError(f"image_shape must be positive, got {self.image_shape}")
        if self.n_cells < 0:
            raise ValueError(f"n_cells must be non-negative, got {self.n_cells}")

        missing = [r for r in REQUIRED_CHANNEL_ROLES if r not in self.channel_roles]
        if missing:
            raise ValueError(f"channel_roles missing required role(s): {missing}")
        unknown = [r for r in self.channel_roles if r not in VALID_CHANNEL_ROLES]
        if unknown:
            raise ValueError(
                f"channel_roles has unknown role(s) {unknown}; "
                f"valid roles are {list(VALID_CHANNEL_ROLES)}"
            )
        indices = list(self.channel_roles.values())
        if any(i < 0 for i in indices):
            raise ValueError(f"channel indices must be non-negative, got {self.channel_roles}")
        if len(set(indices)) != len(indices):
            raise ValueError(f"two roles map to the same channel index: {self.channel_roles}")

    @classmethod
    def create(
        cls,
        *,
        source_path: str,
        voxel_size_um: Sequence[float],
        image_shape: Sequence[int],
        channel_roles: Mapping[str, int],
        run_id: str,
        params_hash: str | None = None,
    ) -> "FileRecord":
        """Build a record, deriving the identifier and display fields from the path.

        **No debris arguments, since schema 3.** This record is written at segment time, and
        debris is only meaningful against the cell set review produced — see `SCHEMA_VERSION`.
        """
        return cls(
            file_uid=make_file_uid(source_path),
            source_path=source_path,
            source_filename=_basename(source_path),
            source_stem=_stem(source_path),
            sort_key=natural_sort_key(_normalise_path(source_path)),
            voxel_size_um=tuple(voxel_size_um),      # type: ignore[arg-type]
            image_shape=tuple(image_shape),          # type: ignore[arg-type]
            channel_roles=dict(channel_roles),
            run_id=run_id,
            params_hash=params_hash,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "FileRecord":
        """Rebuild from JSON, restoring tuple types that JSON flattens to lists."""
        d = dict(data)
        d["voxel_size_um"] = tuple(d["voxel_size_um"])
        d["image_shape"] = tuple(d["image_shape"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True, kw_only=True)
class CellRecord:
    """One segmented cell.

    ``n_voxels`` and ``bbox_diag_um`` are stored rather than derived so that size
    outliers can be found with a database query, without opening a single array file.
    Both feed the review-priority ranking: two fused cells have roughly twice the
    voxels but a far larger spatial extent, which makes the diagonal the more
    diagnostic of the two.
    """

    cell_uid: str
    file_uid: str
    cell_index_in_file: int

    # geometry — half-open bbox in full-image voxel coordinates, (z, y, x) order
    bbox: tuple[int, int, int, int, int, int]   # (z0, z1, y0, y1, x0, x1)
    n_voxels: int
    bbox_diag_um: float

    # provenance — centroids in full-image voxel coordinates
    nucleus_count: int = 0
    nucleus_ids: tuple[int, ...] = ()
    nucleus_centroids: tuple[tuple[float, float, float], ...] = ()
    split_method: SplitMethod = SplitMethod.UNSPLIT

    #: Voxels of this cell under the **segmentation** mask (threshold 1), which decided
    #: how many cells there are and where they divide. ``n_voxels`` above is the
    #: **measurement** mask (threshold 2) when one is in use, and identical to this when
    #: it is not. Both are kept because mixing them silently inflates any ratio built from
    #: them — a CD68 volume fraction measured in one and divided by the other would be a
    #: ratio of two different objects.
    n_voxels_segmentation: int = 0

    # quality control

    #: The soma's centre **in full-image voxel coordinates**, like ``bbox`` and
    #: ``nucleus_centroids`` above — *not* in the cell's own crop. `None` when no soma remnant
    #: was found.
    #:
    #: **The frame is stated here because its absence was a defect.** This field sat in the QC
    #: block with no frame named while every neighbouring centroid declared one, and the two
    #: sides drifted apart: `pipeline.segment_image` adds the bounding-box origin **before**
    #: storing (`offset = [sl.start for sl in box]`), and `export.build_rows` added
    #: `cell.bbox` **again**. Every soma centroid was therefore displaced by its own bbox
    #: origin, so `avg_nearest_neighbour_um` on `images.csv` — the export's only measure of how
    #: microglia are distributed in space — was **doubled on every row ever written**, and by a
    #: different amount per cell. Two cells 2.166 µm apart exported as 4.332. Fixed 2026-08-12;
    #: `EXPORT_SPEC.md` §21 carries the decision and `EXPORT_AUDIT.md` §3.3 the finding.
    #:
    #: **Anything that consumes this must multiply by the voxel size and nothing else. Anything
    #: that writes it must add the crop offset first.** `pipeline.segment_image` is the only
    #: writer today; Component 5's editing and splitting paths will be the second, which is the
    #: reason this sentence exists rather than a comment at the one call site.
    soma_centroid: tuple[float, float, float] | None = None
    soma_remnant_found: bool = False
    touches_xy_border: bool = False

    #: True when the cell reaches the top or bottom slice of the stack, so part of it lies
    #: outside the imaged volume. **A flag, never a deletion** — the same reasoning as
    #: `touches_xy_border`: a 24 µm stack cannot contain a 50 µm microglion, so excluding
    #: z-truncated cells at segmentation time would discard most of the data, and it is
    #: irreversible. The literature (review §9) recommends excluding truncated objects; that
    #: is an analysis choice, and this flag is what makes it available without destroying
    #: anything now. 3DMorph forces the decision up front by padding z before
    #: `bwclearborder`; there is no reason to inherit that.
    touches_z_border: bool = False

    #: > ### ⇒ `agreement_score` AND `size_outlier_score` LEFT THIS RECORD ON 2026-09-03
    #: >
    #: > Both existed to order item 7 worst-first and **both were nucleus-derived** — soma /
    #: > nucleus co-location, and volume *per nucleus* against the image median. The nucleus
    #: > channel drives segmentation and nothing else in this program: it has no measurement
    #: > threshold, no equivalent of item 5, and was never built to be reconstructed in 3D. So
    #: > the ordering was ranking how well nucleus detection went, which is why a bare soma with
    #: > a few stubs sorted **last** — one nucleus, cleanly matched, small volume.
    #: >
    #: > `prestomorph.qc` replaced them with five signals over the mask and the soma, and
    #: > **none of them is stored**: the pool would hold a second source for a fact that goes
    #: > stale the moment a reviewer edits a mask. Both review routes recompute, which
    #: > `review --pool` already did for the soma centroid and already demands `--params` for.
    #: >
    #: > Removed rather than left unused because no pool held data yet, so there was nothing to
    #: > migrate and no reason to keep a column nothing writes.

    #: True when **this cell's mask was drawn by hand** — processes added or removed. Narrowed
    #: 2026-08-16; it previously read "added, removed or split nuclei or processes", which was
    #: broader than anything can usefully carry.
    #:
    #: **The line is input versus output, and it is scientific rather than tidy.** Morphometrics
    #: measure the mask, so a hand-drawn boundary means the ramification index and skeleton
    #: measures are partly measuring the human's hand. Correcting *which nuclei exist* — adding,
    #: deleting or splitting a nucleus — does not: the pipeline still derives the boundary by its
    #: normal rules, and at that point in the flow no cell exists to flag anyway.
    #:
    #: **A cell split is not this flag.** ``split_by_watershed`` places the dividing surface from
    #: human markers, so the child's mask is machine-drawn. That fact is ``created_by_split``,
    #: which ships with Component 5's splitting feature. Two independent booleans, so a cell that
    #: was split *and* then tidied carries both with no precedence rule.
    #:
    #: **Why it exists**, and note this is NOT the reason previously given here. The old text said
    #: "semi-auto and batch write to one pool" — **they do not**, and have not since separate
    #: pools were enforced by origin (`COMPONENT_03_PLAN.md` §4a.3) and then by Phase A parameter
    #: set (§4a.9). The real reason is that **batch review edits masks too**: Component 5 gives
    #: batch the same add/remove-process editing as semi-auto, "because without it a process error
    #: in a batch run has no repair path at all". So a batch pool holds hand-drawn masks and pool
    #: separation cannot substitute for the flag.
    #:
    #: **And hand-editing is genotype-correlated**, which is what makes it a confound rather than
    #: a curiosity: under-splits are 5xFAD-only, so repair lands disproportionately on one arm of
    #: the comparison. The analyst has to be able to ask whether a WT/5xFAD difference survives
    #: dropping hand-touched cells — a per-row filter, which is why this is per cell.
    #:
    #: **Provenance, not control flow.** Nothing in the pipeline branches on this field and
    #: nothing should; it is written, stored and exported.
    #:
    #: > ### ⇒ IT DOES **NOT** SURVIVE A RE-SEGMENTATION, AND THAT IS CORRECT — settled 2026-09-01
    #: >
    #: > This used to claim the opposite: *"the one behaviour it needs is not being lost — hence
    #: > joining `preserve_cell`, in the same change that first sets it."* **It is not in
    #: > `preserve_cell` and never was**, so the sentence described an intention nobody
    #: > implemented, and which of the two was wrong depended on what the column means.
    #: >
    #: > **The user settled the meaning: "was this cell's mask edited by a human".** A
    #: > re-segmentation regenerates the mask from the parameters, so the row genuinely is not
    #: > hand-edited any more and the flag should reset. The behaviour was right and this comment
    #: > was wrong. Status, reject reason, note and `n_real_cells_here` do survive, because those
    #: > are judgements about the cell rather than facts about the mask.
    human_edited: bool = False

    #: True when a **human split** produced this cell — either by splitting its nucleus at
    #: item 4, or by splitting the cell itself in rich review. The user's definition, 2026-09-01.
    #:
    #: > ### ⇒ NOTHING SETS THIS YET, AND THAT IS THE HONEST STATE RATHER THAN AN OVERSIGHT
    #: >
    #: > Neither source of provenance exists today:
    #: >
    #: > * **Cell splitting in rich review is not built** — parked with agenda item A9, which
    #: >   waits on the child-uid question and the label renumbering.
    #: > * **Nucleus splitting leaves no per-nucleus trace.** Item 4 records
    #: >   `Operation("split", label, points)` in the screen's own edit list, but what reaches
    #: >   the session is `nucleus_edits`, an **integer count**. Cells carry `nucleus_ids`, so
    #: >   the join exists the moment those operations mark the nuclei they produce — but they
    #: >   do not, so no cell can be attributed to a hand-split nucleus.
    #:
    #: It is a field now rather than when that lands, because the exported provenance column is
    #: defined by it: shipping the column with `split` undefined would mean changing what the
    #: column *means* later, which is worse than shipping a value that is honestly always False
    #: until the provenance arrives.
    #:
    #: **Not preserved across a re-segmentation**, for the same reason as `human_edited`:
    #: re-detection regenerates the nuclei, so the cell is no longer the product of that split.
    human_split: bool = False

    # lifecycle
    status: CellStatus = CellStatus.CANDIDATE

    #: **Structured since 2026-08-11**, one of :class:`RejectReason`, not free text. It was
    #: free text, which meant "this is debris" and "this is a real cell I ran out of time on"
    #: were the same column with no way to tell them apart or to count either. An unrecognised
    #: value raises rather than being coerced into a category it may not belong to.
    reject_reason: "RejectReason | str | None" = None

    #: The reviewer's own words. Offered on the :attr:`RejectReason.REAL_CELL` branch — "if
    #: they select real cell, give them the option of making a note" — so that removing the
    #: ``other`` category costs no detail. Not constrained here: the flow decides when a note
    #: is asked for, and a stored note beside any reason is data, not an inconsistency.
    reject_note: str | None = None

    #: **How many real cells this row is worth**, §4.4's NEW-d. ``1`` for an ordinary cell, and
    #: the reviewer's own number on a :attr:`RejectReason.REAL_CELL` rejection — *"this is 3
    #: cells I cannot afford to split"*. The count is **authoritative, not an estimate**: this
    #: project treats human marks as ground truth, so there is no separate estimated column and
    #: there is not going to be one.
    #:
    #: > ### ⇒ ``None`` ON DEBRIS, AND THAT IS THE WHOLE REASON IT IS NULLABLE
    #: >
    #: > Debris is not a cell, so it is worth **no** cells — not one. The user's decision,
    #: > 2026-08-30, taken against the alternative of storing ``1`` and excluding debris at every
    #: > reading site: *a stored 1 on a row that is not a cell is a number that means nothing,
    #: > and some later SUM will happily add it up.* Storing nothing makes the column summable
    #: > directly and correct by construction.
    #: >
    #: > **It is derived, not accepted** — `store.Pool.set_status` writes ``None`` whenever the
    #: > reason is ``debris``, whatever it was handed, on the same principle that has ``add_cell``
    #: > compute ``n_voxels`` rather than trust a call site.
    #:
    #: **Nothing reads it yet.** Phase 1 captures and persists; the counts that consume it —
    #: `n_real_cells`, `cells.csv`, the loss rate, `n_cells_within@d` — are Phase 2.
    n_real_cells_here: int | None = 1

    created_at: str = ""
    reviewed_at: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _as_int_tuple(self.bbox, 6, "bbox"))
        object.__setattr__(self, "nucleus_ids", tuple(int(i) for i in self.nucleus_ids))
        object.__setattr__(
            self,
            "nucleus_centroids",
            tuple(_as_float_tuple(c, 3, "nucleus_centroid") for c in self.nucleus_centroids),
        )
        object.__setattr__(self, "split_method", SplitMethod(self.split_method))
        # Status and reason are independent facts and are coerced independently. There is no
        # reconciliation step: the machinery that used to normalise a `DEFERRED` status
        # against a `not_fixed` reason went with the status itself, because one fact recorded
        # in two places is what made a reconciler necessary in the first place.
        object.__setattr__(self, "status", CellStatus(self.status))
        object.__setattr__(self, "reject_reason",
                           None if self.reject_reason is None
                           else RejectReason(self.reject_reason))
        if self.soma_centroid is not None:
            object.__setattr__(self, "soma_centroid",
                               _as_float_tuple(self.soma_centroid, 3, "soma_centroid"))
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())

        z0, z1, y0, y1, x0, x1 = self.bbox
        if not (z0 < z1 and y0 < y1 and x0 < x1):
            raise ValueError(
                f"bbox must be half-open with strictly increasing bounds, got {self.bbox}"
            )
        if min(self.bbox) < 0:
            raise ValueError(f"bbox bounds must be non-negative, got {self.bbox}")
        if self.n_voxels <= 0:
            raise ValueError(f"n_voxels must be positive, got {self.n_voxels}")
        if self.bbox_diag_um <= 0:
            raise ValueError(f"bbox_diag_um must be positive, got {self.bbox_diag_um}")
        #: **`None` is a legitimate value and zero is not.** `None` says *not a cell* — the debris
        #: row; `0` would say *worth zero cells*, which is a different claim and one nothing in
        #: the interface can make. `Resolution` refuses the same thing at the other end.
        if self.n_real_cells_here is not None:
            object.__setattr__(self, "n_real_cells_here", int(self.n_real_cells_here))
            if self.n_real_cells_here < 1:
                raise ValueError(
                    f"n_real_cells_here must be at least 1 or None, "
                    f"got {self.n_real_cells_here}"
                )

        volume = (z1 - z0) * (y1 - y0) * (x1 - x0)
        if self.n_voxels > volume:
            raise ValueError(
                f"n_voxels ({self.n_voxels}) exceeds bbox volume ({volume}) — "
                "mask and bounding box disagree"
            )
        if self.nucleus_count != len(self.nucleus_ids):
            raise ValueError(
                f"nucleus_count ({self.nucleus_count}) != len(nucleus_ids) "
                f"({len(self.nucleus_ids)})"
            )
        if self.nucleus_count != len(self.nucleus_centroids):
            raise ValueError(
                f"nucleus_count ({self.nucleus_count}) != len(nucleus_centroids) "
                f"({len(self.nucleus_centroids)})"
            )

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        d["split_method"] = self.split_method.value
        d["status"] = self.status.value
        d["reject_reason"] = (None if self.reject_reason is None
                              else RejectReason(self.reject_reason).value)
        return d

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "CellRecord":
        """Rebuild from JSON, restoring the tuple types that JSON flattens to lists."""
        d = dict(data)
        d["bbox"] = tuple(d["bbox"])
        d["nucleus_ids"] = tuple(d.get("nucleus_ids") or ())
        d["nucleus_centroids"] = tuple(
            tuple(c) for c in (d.get("nucleus_centroids") or ())
        )
        if d.get("soma_centroid") is not None:
            d["soma_centroid"] = tuple(d["soma_centroid"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
