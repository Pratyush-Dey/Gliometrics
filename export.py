"""Component 7 — the CSVs, figures and label images that get analysed.

Rebuilt 2026-08-10 to `docs/EXPORT_SPEC.md`, which is authoritative and was agreed with the
user measure by measure. Four things about the shape of this module are decisions, not
accidents.

**One shared column specification drives every header and every flattener.** :class:`Columns`
is built once from the run's parameters and is the only place a column name is written down.
The defect that caused this rebuild was a measure that was computed, stored in the pool, and
had no column to be written into — every plaque distance, contact area and overlap volume, and
the per-image burden this project calls its outcome measure. **37 columns were written where
72 were advertised.** Headers built from one list and rows from another is how that happens, so
they are the same list here, and `tests/test_export.py` asserts in both directions: every
specified column reaches a header, and every key a flattener produces has a column.

**CSV throughout, one folder per run.** `openpyxl` and `xlsxwriter` are deliberately absent, so
the previous xlsx path was ~150 lines of hand-rolled SpreadsheetML existing only to put several
tables in one file. A folder of named CSVs carries the same information with less machinery and
makes **all tables equally loadable** — cells, plaques and images are *different row types*, not
a primary table with two lesser ones.

**A re-run writes its own export.** No side-by-side sheet, no per-hash column suffixes; the
pool may still hold several parameter sets, but one export describes one of them and
`params_hash` lives in run-level metadata rather than on every row.

**Files by unit of analysis, columns grouped by goal within.** A goal spans units — morphology
is per-cell *and* per-image, pathology is per-cell, per-plaque *and* per-image — so a goal
cannot be a file, but it can be a contiguous block of columns.

Two rules survive from the previous build unchanged. It does not quote a threshold-dependent
figure without the threshold that produced it. And it does not drop removed cells: they appear
at the bottom with their reason and **blank** measures, because "how many were discarded" must
be answerable from the file itself — and a blank is never a zero.

**Since 2026-08-11 that second rule carries a distinction it could not before.**
`reject_reason` is a structured value with exactly two members, and a cell removed as `debris`
is a different event from a `real_cell` removed anyway — the first is a correct removal, the
second a data loss. It matters because the cells expensive enough to abandon are the clumped,
plaque-associated 5xFAD ones, so the loss is aimed at the population under study.
`export_summary.csv` carries the rate **per genotype** for that reason; a pooled figure is the
summary in which such a bias disappears.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .markers import (
    CONTACT_TOLERANCES_UM,
    CURVE_FACTORS,
    MIN_PLAQUE_VOLUME_UM3,
    RECRUITMENT_DISTANCES_UM,
)
from . import swc as swc_module
from .metadata import parse_metadata
from .morphometrics import SHOLL_STEP_UM
from .schema import (
    KEPT_STATUSES,
    NOT_ANALYSED_STATUSES,
    UNREVIEWED_STATUSES,
    CellStatus,
    RejectReason,
    human_edit_kind,
)

__all__ = [
    "CellRow",
    "Columns",
    "build_columns",
    "build_rows",
    "export_pool",
    "export_cell_masks",
    "write_mask_tiff",
    "write_label_image",
    "write_cell_figures",
]


# ---------------------------------------------------------------------------
# the one column specification
# ---------------------------------------------------------------------------

#: `cells.csv`, in this reading order: who the cell is, whether it counts, then the three
#: goals — morphology, phagocytosis, pathology — then the depth covariate. Blocks that do not
#: depend on a threshold are fixed; the rest are generated in :func:`build_columns`.
#: > ### ⇒ THE FIVE `CELL_*` TUPLES WERE DELETED — 2026-09-01
#: >
#: > `CELL_IDENTITY`, `CELL_QC`, `CELL_MORPHOLOGY`, `CELL_PHAGOCYTOSIS` and `CELL_COVARIATE`
#: > built one `cells.csv` header between them. That file is now seven sheets, defined below as
#: > `CELLS_IDENTITY` … `CELLS_PLAQUE`, and keeping the old five beside them would have left
#: > two sets of names for one thing with nothing to keep them in step.
#: >
#: > Two columns did not survive the move. **`human_edited` is gone**, fully determined by
#: > `human_edit_kind`; the field stays on `CellRecord` because the categorical is computed
#: > from it. And **`n_endpoints` is now `anchored_n_endpoints`**, because the soma-free
#: > skeleton measures the same thing and a join must not collide.

IMAGE_IDENTITY = ("file", "genotype", "animal", "section", "region", "image_index")

#: `images.csv` alone, and it is the join key — added 2026-09-01.
#:
#: **Every other sheet is keyed on `cell_uid` or `plaque_uid`, both of which embed `file_uid`
#: already.** This one had only `file`, which is `source.name` — the **basename**. Nothing
#: enforced basename uniqueness, and genotype and animal are read from the *path* precisely
#: because filenames are not trusted to identify an image, so the export was joining on the one
#: field its own design distrusts. Two images sharing a basename would have had their cell rows
#: merged and both image rows would have reported the union — wrong counts, wrong capture
#: fraction, silently. There are no collisions in the present data; this closes the class.
IMAGE_IDENTITY_UID = ("file", "file_uid", "genotype", "animal", "section", "region",
                      "image_index")

#: **One count, and it answers "how many microglia were in this field".** The user's decision,
#: 2026-08-11: `n_real_cells` is **confirmed cells plus cells rejected with reason
#: `real_cell`** — both are cells the reviewer judged to be real, one kept and one discarded
#: for lack of time. Cells rejected as `debris` are not cells and are excluded.
#:
#: **It mixes measured and unmeasured cells,** so it will exceed the number of rows in this
#: export carrying morphometrics. That is correct for the question it answers and **wrong as a
#: denominator for "what fraction of cells did we measure"** — anyone wanting that must count
#: rows in `cells.csv` with a non-blank `morphometrics_status`, which is also why the
#: `n_cells_measured` column that used to sit here was dropped rather than kept: the user asked
#: for one count, and the measured figure was already recoverable by counting rows.
#:
#: **Named `n_real_cells`, not `n_cells`.** A reader who guesses at `n_cells` guesses "the rows
#: this image contributes to `cells.csv`" or "the cells that were measured", and both are
#: wrong. `real` is a word this export already defines — `reject_reason` is exactly `debris` or
#: `real_cell` — so the name carries its own criterion: realness, not survival and not
#: measurement. It also stops colliding with `export_summary.csv`'s `n_cells`, which really is
#: every row and now differs in name as well as in meaning.
#:
#: Blank with a status, never a 0, while any cell of the image is still `candidate`: **0 is a
#: legitimate value** — an image where every detected object was debris — so it cannot also
#: mean "nobody has judged these yet".
#: `n_cells_lost_real` (was `n_cells_real_cell` until 2026-08-25) joined 2026-08-12 (NEW-k): **how many real microglia this image lost to
#: review time.** `export_summary.csv` carried the figure per *genotype* already; nothing carried
#: it per *image*, so it could not be related to anything else measured on the same image — and
#: its rate is expected to depend on genotype, because the cells that are expensive to fix are
#: the clumped, plaque-associated 5xFAD ones (`schema.RejectReason`).
#:
#: > ### ⇒ CORRECTED 2026-09-01. THE TERRITORY ARGUMENT HERE WAS CLOSED AND SAID OTHERWISE
#: >
#: > This read: *"`union_surveyed_um3` and the other hull measures are built from the
#: > **measured** cells only, so every `real_cell` loss shrinks them — silently, and
#: > preferentially on 5xFAD… This column is what lets the choice be made on evidence later."*
#: >
#: > **The choice was made.** The user reversed D-A2 on 2026-08-25 and a rejected real cell now
#: > contributes a hull (`phasec.measure_file`, where the reversal carries its own reasoning):
#: > territory is the tissue a cell reaches into, and a real cell reaches into it whether or not
#: > the reviewer had time to certify its outline. So `union_surveyed_um3`,
#: > `union_unsurveyed_um3`, `union_percent_surveyed`, `overlap_volume_um3` and `overlap_index`
#: > are **C + R**, and this column no longer reports a bias in any of them.
#:
#: **What it does report, and why it stays a column:** how many real microglia this image lost to
#: review time. That rate is expected to depend on genotype — the expensive-to-fix cells are the
#: clumped, plaque-associated 5xFAD ones — which makes it a result rather than bookkeeping, and
#: it is recoverable from nothing else here: `n_real_cells` folds it together with the cells that
#: were kept. `EXPORT_SPEC.md` §26.
#:
#: **The order is the specification.** `n_real_cells_status` sits after **both** counts because
#: it governs both (D-A14): they become determinate at the same instant — when no cell is still
#: a candidate — so a second status could never disagree with the first and would only be a
#: second place for it to be wrong.
#: `n_cells_touching_xy_border` and `n_cells_touching_z_border` joined 2026-08-15, on
#: Workstream B's request. **`images.csv` had thirty columns and none said how many cells the
#: frame had cut.** Every cell already carries `touches_xy_border` / `touches_z_border`
#: (`schema.CellRecord`), and the flags exist so that excluding truncated objects stays an
#: *analysis* choice rather than a segmentation-time deletion — but `union_surveyed_um3`,
#: `union_percent_surveyed`, `overlap_index` and `avg_nearest_neighbour_um` are **aggregated
#: here, before an analyst can filter**. Once the union is a number the truncated hulls cannot
#: be taken back out, so the exposure has to travel on the same row.
#:
#: **They count the MEASURED cells, which is the population territory is built from** (D-A2) —
#: not `n_real_cells`, which includes `real_cell` rejections that never reach Phase C and
#: contribute no hull. **The two counts therefore do not share a denominator**, deliberately,
#: and §5 says so: this is the same trap the frozen log §20 had to record for `n_real_cells`
#: itself, which is correct for its own question and wrong as a denominator for a different one.
#:
#: **No status column, because 0 is a real answer** — an image the frame cut nothing from is
#: the good case and has to read as a number. That is `n_cells_lost_real`'s rule, and the
#: territory block these qualify carries no status either.
#:
#: **The z count is expected to run high and is still worth a column.** `schema.CellRecord`
#: notes that a 24 µm stack cannot contain a 50 µm microglion, so most cells are z-truncated in
#: this data — which is exactly why territory is a lower bound everywhere rather than only at
#: the frame edge (`EXPORT_SPEC.md` §7), and this is the number that makes that statement
#: checkable per image instead of asserted once in prose.
#: **Renamed and extended 2026-08-25.** `n_real_cells` and `n_cells_real_cell` differed only in
#: word order and were read as duplicates; and the population most often wanted — the confirmed
#: cells — had no column at all, because `n_cells_measured` was dropped as "recoverable by
#: counting rows". Counting rows stopped answering it when rejected-real cells gained a partial
#: row, so the count is explicit now.
IMAGE_CELL_COUNTS = ("n_cells_confirmed", "n_cells_lost_real", "n_real_cells",
                     "n_real_cells_status",
                     "n_cells_touching_xy_border", "n_cells_touching_z_border")

#: `image_cd68_fraction_in_cell_masks` — **how much of the image's CD68 the per-cell numbers
#: account for.** Numerator: the summed per-cell `cd68_integrated`. Denominator:
#: `image_cd68_integrated`, the threshold-free whole-image sum sitting two columns to its left.
#:
#: **The numerator is every cell in the image that carries a CD68 measurement, whatever its
#: review status** — and that choice is the whole reason this is a column rather than a
#: `groupby` left to the analyst. Summing only confirmed cells would fold two unrelated things
#: into one number: how tightly the masks hug the CD68 signal, and how many cells a reviewer
#: threw away. Those move for entirely different reasons and a reader cannot separate them
#: after the fact. Including every measured cell makes the figure a property of the **masks**,
#: so it does not move when review happens — `test_the_capture_fraction_does_not_move_when_a_
#: cell_is_rejected` pins exactly that.
#:
#: **Named for masks, not for cells,** because "cells" is the word that invites the wrong
#: guess: masks have no review state, so `in_cell_masks` cannot be read as "of the confirmed
#: ones". The `image_` prefix follows `image_cd68_integrated` (§12.1) — unit of analysis first.
#:
#: **What it is for.** One cell mask serves every measurement in this export; it is not widened
#: for CD68. That is a deliberate choice and this column is what makes it auditable per image
#: rather than argued in prose: capture runs ~83% median, 67-93% across images, and an image
#: that falls out of that band is one where the masks are missing CD68 the whole-image measure
#: can see.
#:
#: Blank with a status, never a zero and never a fraction over a missing denominator — 0.0 is a
#: plausible capture and "we could not compute one" is not the same fact.
IMAGE_MEASURES = (
    "integrated_intensity", "burden_status",
    "image_cd68_integrated", "image_cd68_status",
    #: > ### ⇒ THE THIRD THRESHOLD-FREE BURDEN, AND IT COMPLETES A SET — 2026-09-01
    #: >
    #: > This sheet carried a threshold-free burden for 6E10 and one for CD68 and **none for
    #: > Iba1**, which is the microglial marker and the basis of the microgliosis claim. Every
    #: > other Iba1 number here — `iba1_volume_um3` and both its halves — depends on
    #: > `mask_coefficient`, so this is the robustness check on all three, in exactly the
    #: > relationship `integrated_intensity` already has to `total_volume_um3`.
    #: >
    #: > **It needs its own status rather than sheltering under `debris_status`.** That one
    #: > reports whether the *mask* could be rebuilt, and one of its failure modes is
    #: > `no_measurement_threshold` — under which this mask-free, threshold-free sum still
    #: > succeeds. Sharing a status would make an image that merely lacks a threshold read as
    #: > one that lacks Iba1 entirely. It is the same argument that keeps `debris_cd68_status`
    #: > separate from `debris_status`.
    "image_iba1_integrated", "image_iba1_status",
    "image_cd68_fraction_in_cell_masks", "image_cd68_fraction_in_mask",
    "image_cd68_fraction_status",
    "n_plaques", "total_volume_um3",
    "volume_fraction", "median_plaque_um3", "largest_plaque_um3", "smallest_plaque_um3",
    # `field_volume_um3` first of the territory block: the whole field, then how much of it the
    # microglia survey, then how much they do not, then that as a percentage. Added 2026-08-12
    # (item 12a) — it was computed by `phasec.measure_file` and stored in the territory payload
    # from the beginning, and **carried by no column**, while §5 stated it "still exists and is
    # still exported". Nothing was lost, since it is `union_surveyed + union_unsurveyed`; the
    # sentence was simply false, which is the defect §25 of the frozen log is about.
    "field_volume_um3",
    #: Each status sits immediately after the block it explains — `debris_status` covers the
    #: volumes and the component count, `debris_cd68_status` covers only the CD68 sum. A missing
    #: CD68 channel leaves the volumes perfectly measurable, so one status over both would make
    #: an image with no CD68 read like an image with no debris.
    "iba1_volume_um3", "iba1_volume_cells_um3", "iba1_volume_debris_um3",
    "debris_n_components", "debris_status", "debris_cd68_integrated", "debris_cd68_status",
    "union_surveyed_um3", "union_debris_um3", "union_unsurveyed_um3",
    "union_percent_surveyed",
    "overlap_volume_um3", "overlap_index",
    "avg_nearest_neighbour_um", "avg_nearest_neighbour_status",
    "cell_labels_status",
)

#: Statuses for the rasterised cell-label volume (NEW-k, 2026-08-12). **This one qualifies no
#: column on its own file** — it warns about `n_cells_within@d` and `covered_fraction@d` on
#: `plaques.csv`, which is why it sits last rather than beside a measure.
#:
#: It is per *image* because the gap is: NEW-b rasterises `real_cell`-rejected cells into
#: `cell_labels` from their stored masks, and a mask that has gone cannot be painted, so those
#: two columns **undercount for every plaque in that image**. Repeating the fact on each plaque
#: row would say the same thing many times.
CELL_LABELS_OK = "ok"
#: At least one `real_cell` rejection could not be rasterised. The uids are in the image
#: payload at `cell_label_gaps` (`phasec.measure_file`); this column is what tells a reader
#: holding only the CSVs to go and look, which a payload key alone could never do.
CELL_LABELS_ARRAYS_MISSING = "arrays_missing"

#: Statuses for `avg_nearest_neighbour_um` (item 11, 2026-08-12). **Three, because a blank had
#: two unrelated causes and no way to tell them apart** — `EXPORT_AUDIT.md` §7 could not
#: classify this column as WAITING or DEFECTIVE from outside for exactly that reason.
#:
#: Every other measure on `images.csv` names its reasons. This was the last one that did not,
#: and it is the export's **only** measure of how microglia are distributed in space.
NEAREST_NEIGHBOUR_OK = "ok"
#: Fewer than two somata to measure between — the legitimate shortage. Covers an image with no
#: cells, one cell, or several cells of which only one has a detected soma.
NEAREST_NEIGHBOUR_TOO_FEW = "too_few_somata"
#: **Two or more cells and not one of them carries a soma centroid.** The diagnostic case: this
#: is what a writer path that never populates `soma_centroid` looks like from outside the CSV,
#: and telling it from the line above is the whole reason this column exists. It is *not*
#: proof of a gap — every soma detection can legitimately fail — but it is the only signal a
#: reader gets, and a silent blank was none.
NEAREST_NEIGHBOUR_NO_CENTROIDS = "no_soma_centroids"

#: `cell_plaque_links.csv` — **the only join between cells and plaques.** Every other plaque
#: measure is anonymous: `nearest_voxel_to_plaque_um` says how far but not *which*,
#: `n_cells_within@d` says how many but not *who*. Without this table cell-plaque
#: relationships are binary, and the per-plaque compaction measures can only be analysed
#: against other plaque measures, never against the cells surrounding them. It is also the
#: only measure that can express a cell touching more than one plaque.
#: **Four columns joined on 2026-09-01**, taking this from one distance per pair to the table
#: that actually describes the study's question. `polarity_index` is the only one of them that
#: is inherently per cell *and* per plaque — a direction rather than a quantity — so no other
#: sheet could hold it. `is_nearest_plaque` is nearest by **nearest voxel**, not by soma: the
#: two disagree for a cell whose process reaches one plaque while its body sits closer to
#: another, and the voxel reading is the one the rest of this table is built on.
LINK_COLUMNS = ("cell_uid", "file", "plaque_uid", "tolerance_um", "distance_um",
                "contact_area_um2", "soma_distance_um", "is_nearest_plaque",
                "polarity_index")

#: **The 3DMorph comparison output was removed 2026-08-11** — `THREE_DMORPH_COLUMNS`,
#: `COMPARABILITY`, `write_3dmorph_csv` and the `--compare-3dmorph` flag with them. It existed
#: to validate this port against 3DMorph, and that job is substantially done: segmentation
#: parity is closed at -0.5% on volume, and only 3 of its 8 columns were ever comparable — the
#: other 5 diverge **by design**, because two 3DMorph defects are fixed rather than ported.
#: **Nothing measured was lost.** Every per-cell figure the sheet was built from still ships in
#: `cells.csv`, and `scripts/parity_3dmorph.py` — the parity harness, which consumes a
#: spreadsheet the user runs MATLAB by hand to produce — survives untouched. Recovering the
#: capability means writing a second writer again, not re-deriving a measurement.


def _g(value: float) -> str:
    """A threshold in a column name, formatted so it round-trips by eye."""
    return f"{float(value):g}"


# ---------------------------------------------------------------------------
# the seven per-cell sheets — 2026-09-01
# ---------------------------------------------------------------------------
#
# `cells.csv` became seven files. The split is by **kind of measurement**, and it exists so
# that each file's population is self-evident: a rejected-real cell has a row on the identity,
# review, mask, CD68 and plaque sheets and **no row at all** on the two skeleton sheets, because
# it was never skeletonised. Under one wide sheet that distinction could only be a blank, and a
# blank there is indistinguishable from a measurement that failed.
#
# Every sheet carries `cell_uid` and `file`, so any pair joins in R without ambiguity.

#: The join, on every sheet. `cell_uid` embeds `file_uid`, so it is unique across the run.
CELL_KEY = ("cell_uid", "file")

CELLS_IDENTITY = CELL_KEY + (
    "genotype", "animal", "section", "region", "image_index", "mean_z_um",
    #: **The first per-cell coordinates the export has ever carried.** The value was computed
    #: in whole-image µm, used once for `avg_nearest_neighbour_um`, and discarded — so no
    #: spatial analysis of cells was possible downstream at all. Available for C **and** R,
    #: because `soma_centroid` is written at segment time and survives rejection.
    "soma_centroid_z_um", "soma_centroid_y_um", "soma_centroid_x_um",
)

CELLS_REVIEW = CELL_KEY + (
    "reviewed", "status", "reject_reason", "reject_note", "human_edit_kind",
    "n_real_cells_here", "nucleus_count", "touches_xy_border", "touches_z_border",
    "morphometrics_status", "skeleton_status", "cd68_status", "collapse_status",
)

#: **Four columns left on 2026-09-02.** `hull_span_ratio` went with `hull_density`, whose
#: reciprocal was already exported as `ramification_index`; the three nucleus columns
#: (`nucleus_volume_um3`, `cell_to_nucleus_ratio`, `soma_to_nucleus_ratio`) were never planned
#: and arrived in a previous session's table. `nucleus_count` on `cells_review.csv` is a
#: different, older column and stays.
CELLS_MASK = CELL_KEY + (
    "cell_volume_um3", "surface_area_um2", "sphericity",
    "soma_volume_um3", "soma_surface_area_um2", "soma_sphericity",
    "cell_to_soma_ratio", "soma_to_cell_fraction",
    "territorial_volume_um3", "ramification_index",
    "hull_surface_area_um2", "z_span_um", "distance_to_xy_border_um",
)

#: A measure computable on **both** skeletons is prefixed so a join cannot collide; one
#: computable on only this skeleton is not, because only a shared name can collide.
#:
#: **The branch-point count joined that rule on 2026-09-02.** It was `n_branch_points` here and
#: `n_junctions` on the free sheet, on the reading that two different spellings cannot collide —
#: but a fork is a fork, so this is the *same* measure counted on the tree and on the graph, and
#: two unrelated names hid that from anyone joining the sheets. `total_skeleton_um` against
#: `total_length_um` is **not** the same case and keeps its names: those are genuinely different
#: computations.
CELLS_SKELETON_ANCHORED = CELL_KEY + (
    "anchored_n_endpoints", "anchored_n_branch_points",
    "min_branch_um", "max_branch_um", "mean_branch_um", "total_skeleton_um",
    "primary_um", "secondary_um", "tertiary_um", "quaternary_um",
    "sholl_max_intersections", "sholl_critical_radius_um",
    "sholl_geodesic_max_intersections", "sholl_geodesic_critical_radius_um",
    "n_primary_processes", "max_centrifugal_order", "mean_centrifugal_order",
    "strahler_order",
    "anchored_mean_process_radius_um", "anchored_max_process_radius_um", "trunk_radius_um",
    "anchored_mean_tortuosity",
    "anchored_n_loops", "anchored_n_triple_points", "anchored_n_quadruple_points",
    "anchored_n_slab_voxels",
)

CELLS_SKELETON_FREE = CELL_KEY + (
    "free_n_endpoints", "free_n_branch_points", "total_length_um",
    "n_segments", "mean_segment_um", "median_segment_um",
    "n_skeleton_components", "detached_skeleton_um",
    "free_n_loops", "free_n_triple_points", "free_n_quadruple_points", "free_n_slab_voxels",
    "free_mean_tortuosity", "free_mean_process_radius_um", "free_max_process_radius_um",
)

CELLS_CD68 = CELL_KEY + (
    "cd68_integrated", "cd68_mean", "cd68_volume_um3", "cd68_fraction",
    "cd68_in_soma_um3", "cd68_in_processes_um3", "cd68_soma_fraction",
)

#: One row per branch of the soma-anchored tree, and one per inter-junction segment of the
#: soma-free graph. **Long, since 2026-09-01** — they were `branch_1_um … branch_N_um` and
#: `shell_1 … shell_N`, schemas that grew with the data.
BRANCHES_ANCHORED = CELL_KEY + (
    "branch_id", "length_um", "centrifugal_order", "strahler_order",
    "mean_radius_um", "tortuosity",
)
SEGMENTS_FREE = CELL_KEY + (
    "segment_id", "component_id", "segment_type", "length_um", "mean_radius_um", "tortuosity",
)
SHOLL_ANCHORED = CELL_KEY + ("metric", "shell_index", "shell_radius_um", "n_intersections")
CD68_CURVE = CELL_KEY + ("cd68_status", "cd68_threshold_level",
                         "cd68_volume_um3", "cd68_fraction")


@dataclass(frozen=True)
class Columns:
    """Every column of every file in one run's export.

    Built once by :func:`build_columns` from the run's parameters, then used for **both** the
    headers and the flatteners. That is the structural fix: a measure cannot be produced
    without a column, because the same object decides each.
    """

    tolerances: tuple[float, ...] = CONTACT_TOLERANCES_UM
    #: The second distance family, for `n_somata_within@r`. Separate from `tolerances` because
    #: it answers a different question at a different scale — see `markers`.
    recruitment: tuple[float, ...] = RECRUITMENT_DISTANCES_UM
    cd68_levels: tuple[float, ...] = ()
    burden_levels: tuple[float, ...] = ()
    cd68_threshold: float | None = None
    plaque_threshold: float | None = None
    min_plaque_volume_um3: float = MIN_PLAQUE_VOLUME_UM3
    sholl_step_um: float = SHOLL_STEP_UM
    #: The skeleton pruning length, µm. **Carried here for `export_summary.csv` alone — it
    #: appears in no column name** (NEW-j, 2026-08-12). That is why it was missed: this file's
    #: stated job is decoding the names, and no name carries a pruning length. But
    #: `anchored_n_endpoints`, `anchored_n_branch_points`, the four branch-length columns,
    #: `total_skeleton_um`, every column of `branches_anchored.csv` and both Sholl profiles are
    #: functions of it, and at
    #: the shipped 1.0 it removes **31% of all branches**.
    min_spur_um: float = 1.0
    #: Carried for `export_summary.csv` alone, and for `min_spur_um`'s exact reason: since the
    #: collapse these four decide the tree's **topology**, not just its volumes, so every
    #: skeleton column in the export is a function of them.
    soma_method: str = ""
    soma_thickness_um: float = float("nan")
    soma_anchor_um: float = float("nan")
    loop_prune_method: str = ""

    @property
    def cell_pathology(self) -> tuple[str, ...]:
        return (("soma_to_plaque_um", "nearest_voxel_to_plaque_um",
                 "plaque_overlap_volume_um3")
                + tuple(f"plaque_contact_um2@{_g(t)}" for t in self.tolerances)
                + tuple(f"contact_fraction@{_g(t)}" for t in self.tolerances))

    @property
    def cells_plaque(self) -> tuple[str, ...]:
        return CELL_KEY + self.cell_pathology

    @property
    def plaques(self) -> tuple[str, ...]:
        return (IMAGE_IDENTITY
                + ("plaque_uid", "volume_um3", "surface_area_um2", "sphericity", "solidity",
                   "equivalent_diameter_um", "max_feret_um",
                   "centroid_z_um", "centroid_y_um", "centroid_x_um",
                   "touches_xy_border", "touches_z_border")
                + tuple(f"covered_fraction_cells@{_g(t)}" for t in self.tolerances)
                + tuple(f"covered_fraction_iba1@{_g(t)}" for t in self.tolerances)
                + tuple(f"n_cells_within@{_g(t)}" for t in self.tolerances)
                #: The second tolerance family. `@r` and not `@d` because these are recruitment
                #: distances, not contact tolerances, and `@1` beside `@5` must not read as two
                #: points on one scale.
                + tuple(f"n_somata_within@{_g(r)}" for r in self.recruitment))

    @property
    def images(self) -> tuple[str, ...]:
        return IMAGE_IDENTITY_UID + IMAGE_CELL_COUNTS + IMAGE_MEASURES

    @property
    def links(self) -> tuple[str, ...]:
        return LINK_COLUMNS

    @property
    def image_curves(self) -> tuple[str, ...]:
        return (IMAGE_IDENTITY + ("burden_status",)
                + tuple(f"burden_volume_um3@{_g(t)}" for t in self.burden_levels))

    def all_headers(self) -> dict[str, tuple[str, ...]]:
        """Every file and its header. **Sixteen, and all fixed-width since 2026-09-01.**

        There is no longer a second category for files whose width follows the data — the long
        reshape ended it, which is why `tests/_spec_columns.py` no longer carries a
        `VARIABLE_WIDTH` construct and every count in `EXPORT_SPEC.md` §5.1 is a number.
        """
        return {
            "cells_identity.csv": CELLS_IDENTITY,
            "cells_review.csv": CELLS_REVIEW,
            "cells_mask.csv": CELLS_MASK,
            "cells_skeleton_anchored.csv": CELLS_SKELETON_ANCHORED,
            "cells_skeleton_free.csv": CELLS_SKELETON_FREE,
            "cells_cd68.csv": CELLS_CD68,
            "cells_plaque.csv": self.cells_plaque,
            "plaques.csv": self.plaques,
            "images.csv": self.images,
            "cell_plaque_links.csv": self.links,
            "branches_anchored.csv": BRANCHES_ANCHORED,
            "segments_free.csv": SEGMENTS_FREE,
            "sholl_anchored.csv": SHOLL_ANCHORED,
            "cd68_curve.csv": CD68_CURVE,
            "image_curves.csv": self.image_curves,
        }


def build_columns(params=None) -> Columns:
    """The column specification for a run, derived from its parameters.

    **The curve columns are a function of the chosen threshold, not of the data.** They carry
    absolute levels in their names — `cd68_volume_um3@12.5` — because a threshold-dependent
    number must never appear without the threshold that produced it. With no threshold chosen
    there are no levels and therefore no columns: the levels are multiples of T, so an unset T
    yields nothing to compute rather than something to guess.
    """
    cd68 = getattr(params, "cd68_threshold", None)
    plaque = getattr(params, "plaque_threshold", None)
    #: **A threshold with no channel behind it describes nothing.** Item 9a is skipped for a
    #: study acquired without CD68, so the semi-auto route cannot write that pair — but a
    #: parameters file carrying a null channel and a stale threshold would otherwise emit four
    #: all-blank curve rows per cell, each labelled with a level nothing was measured against.
    #: Making the combination **inert** rather than refusing it: `load_params` already refuses a
    #: hand-edited file on the `params_hash` check.
    #:
    #: **The levels alone.** `cd68_threshold` below still records what the parameters said —
    #: the provenance file reports the run, and a value the run carried is not made untrue by
    #: there being nothing to measure it against.
    cd68_levels = (() if cd68 is None or getattr(params, "cd68_channel", None) is None
                   else tuple(float(cd68) * f for f in CURVE_FACTORS))
    return Columns(
        tolerances=tuple(float(t) for t in CONTACT_TOLERANCES_UM),
        cd68_levels=cd68_levels,
        burden_levels=(() if plaque is None
                       else tuple(float(plaque) * f for f in CURVE_FACTORS)),
        recruitment=tuple(float(r) for r in RECRUITMENT_DISTANCES_UM),
        cd68_threshold=None if cd68 is None else float(cd68),
        plaque_threshold=None if plaque is None else float(plaque),
        min_plaque_volume_um3=float(
            getattr(params, "min_plaque_volume_um3", MIN_PLAQUE_VOLUME_UM3)),
        min_spur_um=float(getattr(params, "min_spur_um", 1.0)),
        #: Read with `getattr` defaults rather than required, so a caller holding an older
        #: parameter object still builds columns — the four reach one provenance file and
        #: nothing depends on their presence to write a measurement.
        soma_method=str(getattr(params, "soma_method", "") or ""),
        soma_thickness_um=float(getattr(params, "soma_thickness_um", float("nan")) or
                                float("nan")),
        soma_anchor_um=float(getattr(params, "soma_anchor_um", float("nan")) or
                             float("nan")),
        loop_prune_method=str(getattr(params, "loop_prune_method", "") or ""),
    )


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

@dataclass
class CellRow:
    """One cell's worth of output, from every measurement source at once.

    Values are `None` where a measure does not apply — a **rejected cell has no morphometrics
    at all**, because rejection happens at review, before Phase C runs. `None` is written as
    an empty cell rather than 0, since a zero volume is a measurement and an absent one is not.
    """

    cell_uid: str
    file: str
    values: dict[str, object] = field(default_factory=dict)
    #: Every branch length for this cell, in µm — 3DMorph writes these to their own file and
    #: so do we. Kept because min/max/mean cannot show a **distribution** shift, and process
    #: retraction is one: a cell losing its longest branch and one gaining short branches can
    #: share a mean.
    branch_lengths_um: list[float] = field(default_factory=list)
    #: One dict per branch of the soma-anchored tree and per segment of the soma-free graph,
    #: keyed by their exported column names — **2026-09-01**, when `branch_lengths.csv` went
    #: long and `segments_free.csv` was added. `branch_lengths_um` above is kept because the
    #: per-cell min/max/mean are still computed from it.
    branches: list[dict[str, object]] = field(default_factory=list)
    segments: list[dict[str, object]] = field(default_factory=list)
    #: `{metric: {radius_um: intersections}}` for `euclidean` and `geodesic`.
    sholl: dict[str, dict[float, int]] = field(default_factory=dict)
    sholl_step_um: float = SHOLL_STEP_UM
    cd68_volume_curve: dict[float, float] = field(default_factory=dict)
    cd68_fraction_curve: dict[float, float] = field(default_factory=dict)
    #: `{tolerance_um: {plaque_uid: distance_um}}` — the cell-plaque link table's raw form.
    #: Keyed on `plaque_uid` since 2026-09-01; `plaque_id` alone was unique only within an
    #: image, so a join across files could collide.
    links: dict[float, dict[str, float]] = field(default_factory=dict)
    #: `{(tolerance_um, plaque_uid): {column: value}}` for the four columns that joined the
    #: link table in 2026-09-01 — contact area, soma distance, nearest-plaque flag and
    #: polarity. Separate from `links` so the distance, which every row has, is not buried in a
    #: dict that is empty whenever the extra measures could not be taken.
    link_extras: dict[object, dict[str, object]] = field(default_factory=dict)
    #: Soma centroid in whole-image microns, for the nearest-neighbour distance.
    centroid_um: tuple[float, float, float] | None = None
    #: True when this cell has a row in `measurements` **at this run's `params_hash`**. Carried
    #: as a fact rather than inferred from whether some column came out non-empty: a measure
    #: can legitimately be blank on a measured cell — every CD68 volume is, while
    #: `cd68_threshold` is unset — so a proxy would report attrition that did not happen.
    #:
    #: **No longer summed into a column.** `images.csv` carried `n_cells_measured` until
    #: 2026-08-11; the user asked for one count and it was dropped, because the measured figure
    #: is recoverable by counting rows in `cells.csv` with a non-blank `morphometrics_status`.
    #: The flag stays because it is the only *exact* answer — a measurement can exist with
    #: every one of its columns blank — and re-deriving it later would mean re-deriving it
    #: wrongly.
    measured: bool = False

    @property
    def file_uid(self) -> str:
        """The source file's id, recovered from `cell_uid`.

        `schema.make_cell_uid` builds `{file_uid}__c{index:04d}`, so the id is already carried
        here and needs no second field to drift from it. Used to group rows by **image** rather
        than by basename, which is the join the export used until 2026-09-01 and which nothing
        guaranteed to be unique.
        """
        head, sep, _ = self.cell_uid.rpartition("__c")
        return head if sep else ""

    def get(self, key: str):
        return self.values.get(key)


def _flatten_measurement(row: dict, columns: Columns) -> dict[str, object]:
    """One `measurements` row -> the flat name/value pairs `cells.csv` expects.

    **Every key produced here is a column in `columns.cells`**, and
    `test_no_measure_is_produced_without_a_column` asserts it. That is the whole point of the
    rebuild: the previous version emitted per-floor plaque keys that no header enumerated, so
    they were measured, stored and dropped at write time.
    """
    p = row.get("payload") or {}
    m = p.get("morphometrics") or {}
    cd = p.get("cd68") or {}
    prox = p.get("plaque_proximity") or {}
    contact = p.get("plaque_contact") or {}
    hierarchy = m.get("hierarchy_um") or {}

    def order(level: int):
        # JSON round-trips dict keys as strings; accept either.
        return hierarchy.get(str(level), hierarchy.get(level))

    contact_area = contact.get("contact_area_um2") or {}
    contact_fraction = contact.get("contact_fraction") or {}

    out: dict[str, object] = {
        "reviewed": row.get("reviewed"),
        "morphometrics_status": m.get("status"),
        "cd68_status": cd.get("status"),
        # morphology
        "cell_volume_um3": m.get("cell_volume_um3"),
        "soma_volume_um3": m.get("soma_volume_um3"),
        "territorial_volume_um3": m.get("territorial_volume_um3"),
        "ramification_index": m.get("ramification_index"),
        "sphericity": m.get("sphericity"),
        "surface_area_um2": m.get("surface_area_um2"),
        # -1 is `CellMorphology`'s "not measured" sentinel, because 0 endpoints is a real
        # measurement — a cell with no processes. It must reach the file as a **blank**, with
        # `morphometrics_status` giving the reason; a negative count in a column of counts
        # would be read as data.
        #: **Both prefixed, because both measures exist on the free skeleton.** §1.1's rule: a
        #: measure computable on both artefacts is written `anchored_`/`free_` on each so a join
        #: in R cannot collide. The branch-point count joined that rule on 2026-09-02; the
        #: measurement object still calls it `n_branch_points`, which is the name of the
        #: quantity rather than of the column.
        "anchored_n_endpoints": _drop_sentinel(m.get("n_endpoints")),
        "anchored_n_branch_points": _drop_sentinel(m.get("n_branch_points")),
        "min_branch_um": m.get("min_branch_um"),
        "max_branch_um": m.get("max_branch_um"),
        "mean_branch_um": m.get("mean_branch_um"),
        "total_skeleton_um": m.get("total_skeleton_um"),
        # Hierarchy in µm per order: >=4 paths primary, 3 secondary, 2 tertiary, 1
        # quaternary. Length rather than voxels, because at 5.5:1 a voxel is not a unit of
        # length. The four must sum to total_skeleton_um for any cell with a soma.
        "primary_um": order(4),
        "secondary_um": order(3),
        "tertiary_um": order(2),
        "quaternary_um": order(1),
        "sholl_max_intersections": m.get("sholl_max_intersections"),
        "sholl_critical_radius_um": m.get("sholl_critical_radius_um"),
        "sholl_geodesic_max_intersections": m.get("sholl_geodesic_max_intersections"),
        "sholl_geodesic_critical_radius_um": m.get("sholl_geodesic_critical_radius_um"),
        # phagocytosis
        "cd68_integrated": cd.get("integrated_intensity"),
        "cd68_mean": cd.get("mean_intensity"),
        "cd68_volume_um3": cd.get("positive_volume_um3"),
        "cd68_fraction": cd.get("positive_fraction"),
        # pathology
        "soma_to_plaque_um": prox.get("soma_to_plaque_um"),
        "nearest_voxel_to_plaque_um": prox.get("nearest_voxel_to_plaque_um"),
        "plaque_overlap_volume_um3": contact.get("overlap_volume_um3"),
        # covariate. Read from `morphometrics`, never from `cd68` — the CD68 payload carried
        # its own depth until 2026-08-12 and reading that one would make the covariate wait on
        # a channel it has nothing to do with. µm since the same date (NEW-a).
        "mean_z_um": m.get("mean_z_um"),
        # ---- the mask block, 2026-09-01 ---------------------------------------------------
        "soma_surface_area_um2": m.get("soma_surface_area_um2"),
        "soma_sphericity": m.get("soma_sphericity"),
        "cell_to_soma_ratio": m.get("cell_to_soma_ratio"),
        "soma_to_cell_fraction": m.get("soma_to_cell_fraction"),
        "hull_surface_area_um2": m.get("hull_surface_area_um2"),
        "z_span_um": m.get("z_span_um"),
        "distance_to_xy_border_um": m.get("distance_to_xy_border_um"),
        # ---- CD68 compartments, 2026-09-01 ------------------------------------------------
        "cd68_in_soma_um3": cd.get("in_soma_um3"),
        "cd68_in_processes_um3": cd.get("in_processes_um3"),
        "cd68_soma_fraction": cd.get("soma_fraction"),
        #: `phasec` writes it onto the anchored summary, not onto a `skeletons` key — the
        #: artefacts themselves never reach the payload, only their scalars do.
        "collapse_status": (p.get("skeleton_soma_anchored") or {}).get("collapse_status"),
        #: > ### ⇒ THE POPULATION FILTER, AND WHY IT IS NOT `morphometrics_status`
        #: >
        #: > `morphometrics_status` exists for the NaN-plus-status rule: say *why* a number is
        #: > missing so a blank never reads as a zero. While only confirmed cells were measured,
        #: > "non-blank" happened to select exactly the measured ones, and this export adopted
        #: > that coincidence as its official recipe in four places. The partial pass of
        #: > 2026-08-25 ended it — a rejected-real cell now writes
        #: > `rejected_real_cell_partial` into that same column, so the recipe returns **C + R**,
        #: > over-counting preferentially on 5xFAD because real_cell rejections fall on the
        #: > clumped plaque-associated cells.
        #: >
        #: > `skeleton_status` states the fact directly. `not_attempted` is exactly the
        #: > rejected-real case, and unlike `status == confirmed` it survives
        #: > `--include-unreviewed`, under which candidate cells are fully measured and fully
        #: > skeletonised.
        "skeleton_status": _skeleton_status(m, p),
        **(m.get("skeleton_anchored") or {}),
        **(m.get("skeleton_free") or {}),
    }
    for tol in columns.tolerances:
        out[f"plaque_contact_um2@{_g(tol)}"] = _by_number(contact_area, tol)
        out[f"contact_fraction@{_g(tol)}"] = _by_number(contact_fraction, tol)
    return out


#: The rejected-real partial pass's marker, as `phasec` writes it. Matched by value rather than
#: imported so `export` keeps no import of `phasec`, which imports this module.
PARTIAL_REJECTED_REAL = "rejected_real_cell_partial"


def _skeleton_status(morphometrics: dict, payload: dict) -> str | None:
    """`ok` / `failed` / `not_attempted`, or blank when nothing ran at all.

    Three states and they are not the same question `morphometrics_status` answers.
    **`not_attempted`** is a rejected-real cell, which never reaches `build_skeletons`;
    **`failed`** is a cell that did and produced nothing usable; **`ok`** is a real skeleton.
    A blank means no measurement row exists for this cell at this parameter hash.
    """
    status = morphometrics.get("status")
    if not status:
        return None
    if status == PARTIAL_REJECTED_REAL:
        return "not_attempted"
    free = (payload.get("skeleton_soma_free") or {}).get("status")
    if free and free != "ok":
        return "failed"
    return "ok" if status == "ok" else "failed"


def _by_number(mapping: dict, key: float):
    """Read a numeric key out of a payload dict whose keys JSON turned into strings."""
    if not mapping:
        return None
    for candidate in (key, str(key), str(float(key)), _g(key)):
        if candidate in mapping:
            return mapping[candidate]
    for k, v in mapping.items():
        try:
            if float(k) == float(key):
                return v
        except (TypeError, ValueError):
            continue
    return None


def _numeric_map(mapping: dict | None) -> dict[float, object]:
    """A payload dict keyed by stringified numbers, back to numbers."""
    out: dict[float, object] = {}
    for k, v in (mapping or {}).items():
        try:
            out[float(k)] = v
        except (TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def build_rows(pool, params_hash: str, columns: Columns | None = None) -> list[CellRow]:
    """The join: pool cells + stored measurements -> `CellRow`.

    Cells with **no** measurement at this hash still get a row, carrying their provenance and
    status with empty measures. A rejected cell has no morphometrics because rejection happens
    before Phase C runs, and "how many were discarded" must stay answerable from the file.

    Identity comes from `metadata.parse_metadata`, not from re-parsing the path here.
    **Genotype is read from the path and never from the filename** — the `691643` images are
    named `WT` and are 5xFAD. `section`, `region` and `image_index` come from the filename and
    are canonicalised there, including the `Hippiocampus` misspelling: re-deriving them
    downstream is where a silent factor-level split enters, and two spellings becoming two
    levels of one factor produces no error, just a wrong model.
    """
    columns = columns or Columns()
    measured = {m["cell_uid"]: m for m in pool.measurements(params_hash)}
    files = {f.file_uid: f for f in pool.files()}
    rows: list[CellRow] = []
    for cell in pool.cells():
        record = files.get(cell.file_uid)
        source = Path(record.source_path) if record else Path("")
        meta = parse_metadata(source)
        values: dict[str, object] = {
            "cell_uid": cell.cell_uid,
            "file": source.name,
            "genotype": meta.genotype,
            "animal": meta.animal,
            "section": meta.section,
            "region": meta.region,
            "image_index": meta.index,
            "status": cell.status.value if hasattr(cell.status, "value") else cell.status,
            # The structured value, never the enum's repr — `real_cell`, not
            # `RejectReason.REAL_CELL`, so the column groups in a spreadsheet.
            "reject_reason": (cell.reject_reason.value
                              if hasattr(cell.reject_reason, "value")
                              else cell.reject_reason),
            "reject_note": cell.reject_note,
            "human_edit_kind": human_edit_kind(cell.human_edited, cell.human_split),
            #: `None` on debris reaches the file as a blank, which is the point — see `CELL_QC`.
            "n_real_cells_here": cell.n_real_cells_here,
            "nucleus_count": cell.nucleus_count,
            "touches_xy_border": cell.touches_xy_border,
            "touches_z_border": cell.touches_z_border,
        }
        got = measured.get(cell.cell_uid)
        lengths: list[float] = []
        branches: list[dict[str, object]] = []
        segments: list[dict[str, object]] = []
        sholl: dict[str, dict[float, int]] = {}
        step = SHOLL_STEP_UM
        volume_curve: dict[float, float] = {}
        fraction_curve: dict[float, float] = {}
        links: dict[float, dict[str, float]] = {}
        link_extras: dict[object, dict[str, object]] = {}
        if got is not None:
            values.update(_flatten_measurement(got, columns))
            payload = got.get("payload") or {}
            morph = payload.get("morphometrics") or {}
            cd = payload.get("cd68") or {}
            lengths = [float(v) for v in (morph.get("branch_lengths_um") or [])]
            sholl = {
                "euclidean": _numeric_map(morph.get("sholl_profile")),
                "geodesic": _numeric_map(morph.get("sholl_profile_geodesic")),
            }
            step = float(morph.get("sholl_step_um") or SHOLL_STEP_UM)
            volume_curve = _numeric_map(cd.get("volume_curve_um3"))
            fraction_curve = _numeric_map(cd.get("fraction_curve"))
            #: Keys stay as written — `plaque_uid` is a string since 2026-09-01, where
            #: `plaque_id` was an int. Cast to `str` rather than `int` so a payload written
            #: before the change still reads, as its integer ids simply become their own
            #: spelling rather than raising.
            #: **The payload holds plaque *ids*; the file carries plaque *uids*.** The id is
            #: only unique within an image, so the join is built here where `file_uid` is in
            #: hand — the same `{file_uid}__p{id:04d}` shape `plaques.csv` writes, so the two
            #: tables join on a key that cannot collide across images.
            links = {tol: {_plaque_uid(cell.file_uid, pid): float(d)
                           for pid, d in (pairs or {}).items()}
                     for tol, pairs in _numeric_map(payload.get("plaque_links")).items()}
            link_extras = {
                (float(tol), _plaque_uid(cell.file_uid, pid)): dict(extra or {})
                for tol, pairs in (payload.get("plaque_link_extras") or {}).items()
                for pid, extra in (pairs or {}).items()
            }
            #: The per-branch and per-segment rows, produced by `graph_measures` at measurement
            #: time and carried through the payload. Read defensively: a payload written before
            #: 2026-09-01 has neither key, and the two long files are simply empty for it
            #: rather than the export failing.
            branches = [dict(b) for b in (payload.get("branches") or [])]
            segments = [dict(s) for s in (payload.get("segments") or [])]

        centroid = None
        if cell.soma_centroid is not None and record is not None:
            # **`soma_centroid` is already in whole-image voxel coordinates.** `pipeline.py`
            # adds the crop offset before storing it, exactly as it does for
            # `nucleus_centroids`, so the only thing left to do here is scale to microns.
            #
            # Adding `cell.bbox` as well — which this did until 2026-08-12 — displaced every
            # cell by **its own** crop origin. Not a constant factor: cells near the far corner
            # of a field were displaced most, so `avg_nearest_neighbour_um` was inflated
            # unevenly across an image, on every row the export had ever written. Two cells
            # 2.166 µm apart exported as 4.332 (`EXPORT_AUDIT.md` §3.3).
            #
            # The frame is stated in `EXPORT_SPEC.md` §21 rather than on
            # `schema.CellRecord.soma_centroid`, where it belongs, because `schema.py` is the
            # other workstream's file. That docstring edit is pending in `PROJECT_LOG_A.md`.
            cz, cy, cx = cell.soma_centroid
            vz, vy, vx = record.voxel_size_um
            centroid = (cz * vz, cy * vy, cx * vx)

        #: **Exported since 2026-09-01, and it is the same value.** The centroid above was
        #: computed, used once for `avg_nearest_neighbour_um`, and thrown away — so the export
        #: carried no per-cell coordinate at all and no spatial analysis was possible in R.
        #: Written for C and R alike, because `soma_centroid` is stamped at segment time.
        values["soma_centroid_z_um"] = None if centroid is None else centroid[0]
        values["soma_centroid_y_um"] = None if centroid is None else centroid[1]
        values["soma_centroid_x_um"] = None if centroid is None else centroid[2]

        rows.append(CellRow(
            cell_uid=cell.cell_uid, file=str(values["file"]), values=values,
            branch_lengths_um=lengths, branches=branches, segments=segments,
            sholl=sholl, sholl_step_um=step,
            cd68_volume_curve=volume_curve, cd68_fraction_curve=fraction_curve,
            links=links, link_extras=link_extras,
            centroid_um=centroid, measured=got is not None))
    return rows


def avg_nearest_neighbour_um(rows: Sequence[CellRow]) -> float:
    """Mean distance from each soma to its nearest neighbouring soma, in µm.

    **3DMorph's own centroid measure is deliberately not reproduced.** It takes the mean of
    *all pairwise* distances (`Script_3DMorph.m:591-593`) and calls it density or dispersion,
    but for points in a box that converges to a fixed fraction of the box diagonal almost
    regardless of count or clustering — and it rises if you simply crop a larger field. It
    barely responds to the thing 5xFAD changes.

    Reproducing it would not even serve as a check on our port, because their centroid is the
    *largest* eroded remnant while ours is the *nucleus-selected* one (§4.6). Their number
    would differ from ours for two independent reasons at once, formula and input.
    """
    points = np.array([r.centroid_um for r in rows if r.centroid_um is not None], float)
    if len(points) < 2:
        return float("nan")
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min(axis=1).mean())


def _image_identity(source: Path) -> dict[str, object]:
    meta = parse_metadata(source)
    return {"file": source.name, "genotype": meta.genotype, "animal": meta.animal,
            "section": meta.section, "region": meta.region, "image_index": meta.index}


def _flatten_image(payload: dict, identity: dict, rows: Sequence[CellRow],
                   columns: Columns) -> dict[str, object]:
    """One image measurement -> the flat pairs `images.csv` expects.

    `integrated_intensity` is the **threshold-free** burden figure and the study's outcome
    measure. Before the 2026-08-10 rebuild it reached no file: `export.py` read only `plaques`
    and `territory` from the per-image payload.

    `image_cd68_integrated` is the CD68 measure of the same kind, added 2026-08-11 — summed
    over the whole channel, no mask and no threshold. It is the only CD68 figure in the export
    that depends on neither the segmentation nor an unchosen parameter, which is what makes it
    a usable denominator for the per-cell CD68 that depends on both.

    **`n_real_cells` is one count, and the user chose what it counts** (2026-08-11): confirmed
    cells plus cells rejected with reason `real_cell`. Both are cells the reviewer judged real
    — one kept, one discarded for lack of time — so together they answer "how many microglia
    were in this field". Debris is excluded because debris is not a cell. It replaced the pair
    `n_cells` / `n_cells_measured`, which no document had ever specified.
    """
    burden = payload.get("plaque_burden") or {}
    territory = payload.get("territory") or {}
    cd68_image = payload.get("cd68_image") or {}
    iba1_image = payload.get("iba1_image") or {}
    out: dict[str, object] = dict(identity)
    out.update({
        **_real_cells(rows),
        **_border_counts(rows),
        "integrated_intensity": burden.get("integrated_intensity"),
        # Not in EXPORT_SPEC §5's list, added on the NaN-plus-status rule: without it a blank
        # `n_plaques` cannot be told apart from "no threshold chosen", "no plaques above the
        # floor" and "no plaque channel". This is the study's outcome measure, so the reason
        # a figure is missing has to travel with it.
        "burden_status": burden.get("status"),
        "image_cd68_integrated": cd68_image.get("integrated_intensity"),
        # Same rule, and it bites harder here: 0.0 is a plausible amount of CD68, so a blank
        # with no reason beside it would be read as "this image has none". `not_measured`
        # covers a pool measured before 2026-08-11, whose payload has no `cd68_image` at all —
        # that is a different thing from an image acquired without a CD68 channel.
        "image_cd68_status": cd68_image.get("status") or "not_measured",
        #: The third threshold-free burden, added 2026-09-01 — see `IMAGE_MEASURES` for why it
        #: carries a status of its own rather than sharing `debris_status`.
        "image_iba1_integrated": iba1_image.get("integrated_intensity"),
        "image_iba1_status": iba1_image.get("status") or "not_measured",
        **_cd68_capture(cd68_image, rows, payload.get("iba1")),
        "n_plaques": _drop_sentinel(burden.get("n_plaques")),
        "total_volume_um3": burden.get("total_volume_um3"),
        "volume_fraction": burden.get("volume_fraction"),
        "median_plaque_um3": burden.get("median_plaque_um3"),
        "largest_plaque_um3": burden.get("largest_plaque_um3"),
        "smallest_plaque_um3": burden.get("smallest_plaque_um3"),
        "field_volume_um3": territory.get("field_volume_um3"),
        **_iba1_columns(payload.get("iba1") or {}),
        "union_surveyed_um3": _territory(territory, "union_surveyed_um3",
                                         "union_occupied_um3"),
        #: Debris outside every hull. Absent from a payload written before 2026-08-25, and 0.0
        #: is the right reading there: those runs put debris in `union_unsurveyed_um3`, so the
        #: three still sum.
        "union_debris_um3": territory.get("union_debris_um3", 0.0),
        "union_unsurveyed_um3": _territory(territory, "union_unsurveyed_um3",
                                           "union_unoccupied_um3"),
        "union_percent_surveyed": _territory(territory, "union_percent_surveyed",
                                             "union_percent_occupied"),
        "overlap_volume_um3": territory.get("overlap_volume_um3"),
        "overlap_index": territory.get("overlap_index"),
        "avg_nearest_neighbour_um": avg_nearest_neighbour_um(rows),
        "avg_nearest_neighbour_status": _nearest_neighbour_status(rows),
        # `ok` when nothing was omitted — including for an image with no plaque channel, where
        # no label volume is built at all and so nothing could be missed. The key is absent
        # from the payload unless there is something to report (`phasec.measure_file`).
        "cell_labels_status": (CELL_LABELS_ARRAYS_MISSING if payload.get("cell_label_gaps")
                               else CELL_LABELS_OK),
    })
    return out



def _iba1_columns(iba1: Mapping[str, Any]) -> dict[str, object]:
    """The image's Iba1 burden and its cells/debris breakdown. Blank when Phase C could not run.

    **Measured off the material, not off the cell set.** A failed split moves voxels from cells
    into debris rather than out of the image, so `iba1_volume_um3` cannot be biased low by a
    segmentation failure — and this project's own tuning finds splitting failures to be
    5xFAD-only, which would otherwise put that bias in the same direction as the effect the
    study exists to test.

    `NaN` with a status rather than 0.0, on `DebrisLoad`'s rule: **an image whose Iba1 mask holds
    nothing unclaimed is a real and good outcome**, so it cannot read the same as one where the
    measurement never ran.
    """
    return {
        "iba1_volume_um3": iba1.get("iba1_volume_um3"),
        "iba1_volume_cells_um3": iba1.get("iba1_volume_cells_um3"),
        "iba1_volume_debris_um3": iba1.get("iba1_volume_debris_um3"),
        "debris_n_components": iba1.get("debris_n_components"),
        "debris_cd68_integrated": iba1.get("debris_cd68_integrated"),
        "debris_status": iba1.get("debris_status") or "not_measured",
        "debris_cd68_status": iba1.get("debris_cd68_status") or "not_measured",
    }


def _nearest_neighbour_status(rows: Sequence[CellRow]) -> str:
    """Why `avg_nearest_neighbour_um` is blank, when it is. Item 11, 2026-08-12.

    Three states, and the distinction that matters is between the last two:

    * ``ok`` — two or more somata, so there is a distance;
    * ``no_soma_centroids`` — **two or more cells and not one carries a centroid.** From
      outside the CSV this is what a writer path that never populates ``soma_centroid`` looks
      like. It is not proof of one — every soma detection can legitimately fail on a
      two-cell image — but it is the only signal a reader gets, and a bare blank was none;
    * ``too_few_somata`` — anything else: no cells, one cell, or several of which only one has
      a soma. A genuine shortage, correct behaviour, and **not** the case above.

    **Why `no_soma_centroids` requires two or more rows.** With one cell there is nothing to
    measure between whether or not it has a soma, so reporting a missing-field diagnosis there
    would raise a suspicion the data does not support. The diagnostic is reserved for the case
    where the count of cells was sufficient and the centroids were not there.
    """
    if len(rows) >= 2 and not any(r.centroid_um is not None for r in rows):
        return NEAREST_NEIGHBOUR_NO_CENTROIDS
    with_centroids = sum(1 for r in rows if r.centroid_um is not None)
    return (NEAREST_NEIGHBOUR_OK if with_centroids >= 2
            else NEAREST_NEIGHBOUR_TOO_FEW)


#: Statuses for `n_real_cells`. Two, because there are exactly two states: the image has been
#: reviewed through and the count is the count, or some cell in it has not been judged and the
#: count is not determined.
REAL_CELLS_OK = "ok"
REAL_CELLS_UNREVIEWED = "review_incomplete"

#: The lifecycle values `n_real_cells` reads, as the plain strings that reach the file — the two
#: halves of "a cell the reviewer judged to be real", plus the one that means "not judged yet".
#: Derived from the schema's own enums rather than restated, so a status or a reason added there
#: cannot be missed here.
_KEPT = tuple(s.value for s in KEPT_STATUSES)
_UNREVIEWED = tuple(s.value for s in UNREVIEWED_STATUSES)
_REJECTED = CellStatus.REJECTED.value
_REAL_CELL = RejectReason.REAL_CELL.value


def _multiplicity(row: CellRow) -> int:
    """How many real cells one row is worth. **R is always multiplicity-factoring** — the
    user's rule, 2026-08-30, and it holds in every column and every file.

    **Defaults to 1 rather than 0 for a missing or blank value.** A row with nothing in the
    column is an ordinary cell, not a non-cell: the blank that means *not a cell* only ever
    appears on debris, and every caller here has already filtered debris out by reason.
    """
    value = row.get("n_real_cells_here")
    if value is None or value == "":
        return 1
    return int(value)


def _real_cells(rows: Sequence[CellRow]) -> dict[str, object]:
    """How many microglia were in this field — the user's decision, 2026-08-11.

    **Confirmed cells plus cells rejected with reason `real_cell`.** Both are cells the
    reviewer judged to be real: one was kept, the other discarded because there was no time to
    repair it. Cells rejected as `debris` are excluded, because debris is not a cell — and that
    exclusion is the whole content of the definition, so
    `test_the_real_cell_count_keeps_the_time_losses_and_excludes_the_debris` builds a fixture
    holding all three states rather than asserting arithmetic on cells that happen to agree.

    **It mixes measured and unmeasured cells.** A `real_cell` rejection leaves Phase C, so it
    has no morphometrics, yet it counts here. The number will therefore exceed the rows of this
    export carrying measurements, which is correct for the question it answers and **wrong as a
    denominator for "what fraction of the cells did we measure"** — that one is a count of rows
    in `cells.csv` with a non-blank `morphometrics_status`, and it is why `n_cells_measured`
    could be dropped rather than kept when the user asked for a single count.

    **Blank plus a status while any cell is still a candidate, never a 0.** A candidate has not
    been judged real or debris by anybody, so a count that separates the two is not determined
    for that image — and reporting the confirmed-so-far total would under-report by exactly the
    unreviewed remainder, silently. **0 is a legitimate value here**: an image where every
    detected object turned out to be debris really did contain no microglia, and that finding
    must not be indistinguishable from "nobody has looked yet". Today, with Component 5
    unbuilt and no human having ever reviewed a cell, every image takes this branch — the same
    shape as every measure waiting on `cd68_threshold`, and honest for the same reason.

    An image with no cell rows at all has nothing unjudged in it and counts 0: no real cell was
    found. Whether that is because everything was debris or because nothing was detected is
    answerable from `cells.csv`, which keeps a row for every removal.

    **`n_cells_real_cell` is the second half of the same question** (NEW-k, 2026-08-12): of the
    real cells, how many were *lost*. It shares this status rather than carrying its own, on the
    user's decision (D-A14) — both counts become determinate at the same instant, so a second
    status could never disagree with the first.

    **0 is a real answer for the loss count and must not read as a blank**: an image where the
    reviewer lost nothing is the good case and has to be legible as such.
    """
    if any(r.get("status") in _UNREVIEWED for r in rows):
        return {"n_cells_confirmed": None, "n_cells_lost_real": None,
                "n_real_cells": None, "n_real_cells_status": REAL_CELLS_UNREVIEWED}
    kept = sum(1 for r in rows if r.get("status") in _KEPT)
    #: **Cells, not rows, since 2026-08-30.** A rejection saying *"this is 3 cells I cannot
    #: afford to split"* adds 3. `n_cells_confirmed` beside it stays a row count and is
    #: untouched: the count popup never fires on an accept, so a confirmed cell is always one.
    lost = sum(_multiplicity(r) for r in rows if r.get("status") == _REJECTED
               and r.get("reject_reason") == _REAL_CELL)
    #: Three explicit counts. `n_real_cells` keeps its name and meaning — every cell judged
    #: real, kept or not — and the two halves it is made of now stand on their own rather than
    #: being recovered by subtracting one badly-named column from another.
    return {"n_cells_confirmed": kept, "n_cells_lost_real": lost,
            "n_real_cells": kept + lost, "n_real_cells_status": REAL_CELLS_OK}


def _border_counts(rows: Sequence[CellRow]) -> dict[str, object]:
    """How many of this image's measured cells the frame cut — 2026-08-15, B's request 8.

    > ### ⇒ THESE ARE **C + R**, AND THE DOCSTRING SAID THE OPPOSITE — corrected 2026-09-01
    >
    > It read: *"Over the measured cells only… A `real_cell` rejection never reaches Phase C, so
    > it contributes no hull and is not part of the exposure these columns report."* Both
    > clauses stopped being true and the text did not move.
    >
    > **The population changed twice, by two separate decisions, and the code was right both
    > times by accident.** The filter is a non-blank `morphometrics_status`; the partial pass of
    > 2026-08-25 gave rejected-real cells one, so these counts silently became C + R on that
    > date. Independently, the user reversed D-A2 on the same date and rejected cells began
    > contributing hulls — so the counts and the territory they qualify moved to the same
    > population without anyone lining them up.
    >
    > **C + R is the right answer**, because territory is C + R — confirmed by the user
    > 2026-09-01. The filter is left as a non-blank `morphometrics_status`, which here means
    > *reached Phase C in some form* and is exactly the hull-contributing set. See the note at
    > the filter itself for why `skeleton_status` is **not** the right test in this one place,
    > despite replacing `morphometrics_status` everywhere else.

    **Why the population matters here at all.** A count taken over a different set than the one
    territory was built from would describe an exposure nobody has, and it would look correct in
    every image where nobody had rejected anything — which today is every image.

    **Never blank, and no status.** 0 is a real answer: an image the frame cut nothing from is
    the good case. Unlike `n_real_cells` these do not depend on review having happened — a cell
    either reached the edge of the stack or it did not, and that was settled at segmentation.
    """
    #: > ### ⇒ NOT `skeleton_status`. THIS IS THE ONE PLACE THAT WANTS C + R
    #: >
    #: > Switching this to `skeleton_status` looks like the same tidy-up as everywhere else and
    #: > is wrong: `not_attempted` excludes rejected-real cells, and **those cells contribute
    #: > hulls**. The count would then describe a smaller population than the territory it
    #: > qualifies, which is the exact mismatch this function exists to avoid.
    #: >
    #: > A non-blank `morphometrics_status` means *this cell reached Phase C in some form* —
    #: > fully for a confirmed cell, partially for a rejected-real one — and that is precisely
    #: > the set contributing hulls. `skeleton_status` answers a different question (was it
    #: > skeletonised) and is the right filter for the two skeleton sheets, not for this.
    measured = [r for r in rows if r.get("morphometrics_status")]
    return {
        "n_cells_touching_xy_border": sum(1 for r in measured if r.get("touches_xy_border")),
        "n_cells_touching_z_border": sum(1 for r in measured if r.get("touches_z_border")),
    }


#: Statuses for `image_cd68_fraction_in_cell_masks`. Four, because each is a **different**
#: reason the fraction is absent and a blank alone would make them one.
CD68_CAPTURE_OK = "ok"
CD68_CAPTURE_NO_IMAGE = "no_image_cd68"        # denominator absent: no channel, or an old pool
CD68_CAPTURE_ZERO_IMAGE = "image_cd68_zero"    # denominator is 0; 0/0 is not zero
CD68_CAPTURE_NO_CELLS = "no_cell_cd68"         # numerator absent: no cell carries a CD68 value


def _finite(value) -> float | None:
    """A usable number, or `None`. `None`, `NaN` and `inf` are all 'no value'."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _cd68_capture(cd68_image: dict, rows: Sequence[CellRow],
                  iba1: Mapping[str, Any] | None = None) -> dict[str, object]:
    """What fraction of the image's CD68 the per-cell numbers account for.

    **Every cell with a CD68 measurement counts, whatever its review status.** The alternative —
    confirmed cells only — reports mask tightness and review attrition multiplied together, and
    a reader holding the product cannot recover either. Summing all measured cells makes this a
    statement about where the masks are, which is the question the column exists to answer, and
    it is the only version that stays put while a reviewer works.

    A cell with no CD68 value contributes nothing and is not counted as a zero: the numerator is
    over the cells that were measured, not over every row.

    Four ways there is no answer, each named rather than collapsed into a blank:

    * the denominator is absent — no CD68 channel, or a pool measured before
      `image_cd68_integrated` existed;
    * the denominator is 0 — the division is undefined, and 0/0 is emphatically not 0;
    * no cell carries a CD68 value, so there is no numerator. **Not 0.0**: "the masks captured
      none of it" and "nothing was measured" are different findings, and only the first is a
      result;
    * otherwise `ok`, and the fraction is real.
    """
    denominator = _finite(cd68_image.get("integrated_intensity"))
    numerator = 0.0
    contributors = 0
    for row in rows:
        value = _finite(row.get("cd68_integrated"))
        if value is not None:
            numerator += value
            contributors += 1

    if denominator is None:
        status = CD68_CAPTURE_NO_IMAGE
    elif denominator == 0.0:
        status = CD68_CAPTURE_ZERO_IMAGE
    elif contributors == 0:
        status = CD68_CAPTURE_NO_CELLS
    else:
        #: **Both fractions, and their difference is exactly the debris contribution.**
        #: `in_cell_masks` answers *"how much CD68 sits inside identified microglia"*, which is
        #: arguably the biological question; `in_mask` answers *"inside Iba1 material at all"*.
        #: The gap between them is CD68 no surviving cell claimed, and the gap from `in_mask` to
        #: 1.0 is CD68 outside microglia entirely — a residual that ran at ~17% with nothing able
        #: to name it before debris was measured.
        debris_cd68 = None if iba1 is None else _finite(iba1.get("debris_cd68_integrated"))
        return {"image_cd68_fraction_in_cell_masks": numerator / denominator,
                "image_cd68_fraction_in_mask": (
                    None if debris_cd68 is None
                    else (numerator + debris_cd68) / denominator),
                "image_cd68_fraction_status": CD68_CAPTURE_OK}
    return {"image_cd68_fraction_in_cell_masks": None,
            "image_cd68_fraction_in_mask": None,
            "image_cd68_fraction_status": status}


def _territory(territory: dict, key: str, legacy: str):
    """Read a territory measure, accepting the pre-2026-08-11 key from an older pool.

    The three union measures were renamed `occupied` -> `surveyed`. The payloads already
    written into pools carry the old keys, and re-measuring an image to recover them costs a
    whole-image pass. Reading either is three lines; the alternative is a column of silent
    blanks in the file that carries the outcome measure.
    """
    value = territory.get(key)
    return territory.get(legacy) if value is None else value


def _drop_sentinel(value):
    """-1 means 'not measured'; a file gets a blank, never a negative count.

    Used by `n_plaques`, `n_endpoints` and `n_branch_points`, all of which carry -1 rather
    than 0 for an absent measurement — because **0 is a real answer** for each of them, and
    conflating the two is the 3DMorph defect (`DESIGN_RATIONALE.md` §5.3) this codebase exists
    not to repeat.
    """
    return None if value is None or int(value) < 0 else int(value)


def _plaque_uid(file_uid: str | None, plaque_id: object) -> str | None:
    """`{file_uid}__p{plaque_id:04d}`, matching `schema.make_cell_uid`'s shape."""
    if file_uid is None or plaque_id is None:
        return None
    try:
        return f"{file_uid}__p{int(plaque_id):04d}"
    except (TypeError, ValueError):
        return None


def _flatten_plaque(plaque: dict, identity: dict, columns: Columns,
                    file_uid: str | None = None) -> dict[str, object]:
    """One plaque record -> the flat pairs `plaques.csv` expects.

    **`surface_area_um2` is exported since 2026-09-01.** It was computed and withheld on the
    grounds that only the coverage columns needed it — but it is the *denominator* of six of
    them, and coverage pooled across plaques is `Sum(covered) / Sum(total)`, which the file
    could not express: the only computable figure was the mean of the fractions, which weights
    a 25 um3 plaque like one fifty times larger.

    `is_clustered` is still not emitted — it is fully derivable from `n_cells_within`, and a
    derived boolean beside its own source can only ever disagree with it.
    """
    out: dict[str, object] = dict(identity)
    out.update({
        #: **`plaque_uid` replaces `plaque_id` everywhere, 2026-09-01.** Shaped like `cell_uid`
        #: — `{file_uid}__p{plaque_id:04d}` — so it is unique across the run rather than within
        #: one image. It is **not** stable across runs: `plaque_id` is recomputed by
        #: `ndimage.label` and depends on the plaque threshold, so changing that makes plaque 7
        #: a different object. Plaques are not persisted in the pool the way cells are, which is
        #: what a cross-run identity would need.
        "plaque_uid": _plaque_uid(file_uid, plaque.get("plaque_id")),
        "volume_um3": plaque.get("volume_um3"),
        "surface_area_um2": plaque.get("surface_area_um2"),
        "sphericity": plaque.get("sphericity"),
        "solidity": plaque.get("solidity"),
        "equivalent_diameter_um": plaque.get("equivalent_diameter_um"),
        "max_feret_um": plaque.get("max_feret_um"),
        "centroid_z_um": plaque.get("centroid_z_um"),
        "centroid_y_um": plaque.get("centroid_y_um"),
        "centroid_x_um": plaque.get("centroid_x_um"),
        "touches_xy_border": plaque.get("touches_xy_border"),
        "touches_z_border": plaque.get("touches_z_border"),
    })
    for r in columns.recruitment:
        out[f"n_somata_within@{_g(r)}"] = plaque.get(f"n_somata_within@{_g(r)}")
    #: Two coverage families since 2026-08-25 — by **cell** and by **material**. The iba1 one
    #: is absent from a payload measured before that date, and `.get` leaves it blank rather
    #: than inventing a zero, which would read as "no Iba1 near this plaque".
    for tol in columns.tolerances:
        for family in ("covered_fraction_cells", "covered_fraction_iba1"):
            out[f"{family}@{_g(tol)}"] = plaque.get(f"{family}@{_g(tol)}")
        out[f"n_cells_within@{_g(tol)}"] = plaque.get(f"n_cells_within@{_g(tol)}")
    return out


def _flatten_image_curves(payload: dict, identity: dict,
                          columns: Columns) -> dict[str, object]:
    """Burden curve plus the dense-core compartments, across five DAPI coefficients.

    The 35 compartment columns are **kept deliberately**. Every other threshold in this
    codebase collapsed to a single chosen value; this one does not, because the measure is
    *unvalidated on this tissue* — Gotkiewicz cross-validated high-intensity DAPI against
    Thioflavin-S and X34 and this tissue has neither — and that is exactly where a robustness
    spread earns its place.
    """
    burden = payload.get("plaque_burden") or {}
    curve = _numeric_map(burden.get("volume_curve_um3"))
    out: dict[str, object] = dict(identity)
    out["burden_status"] = burden.get("status")
    for level in columns.burden_levels:
        out[f"burden_volume_um3@{_g(level)}"] = _by_number(curve, level)
    return out


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _text(value) -> str:
    """A cell's text. **`None` and `NaN` are both blank, never `0` and never the word nan.**"""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else repr(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, np.bool_):
        return str(bool(value))
    return str(value)


def write_table(path: Path, header: Sequence[str],
                rows: Iterable[Sequence[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([_text(v) for v in row])
    return path


def _write_records(path: Path, header: Sequence[str],
                   records: Iterable[dict]) -> Path:
    return write_table(path, header,
                       ([rec.get(c) for c in header] for rec in records))


#: Statuses meaning the cell was not measured. Derived from the schema's own list rather than
#: restated, so a status added there cannot be missed here. It holds `rejected` alone: the
#: `deferred` status that briefly sat beside it was removed, and the distinction it carried
#: now lives in `reject_reason`, where sorting never depended on it anyway — both kinds of
#: removal sort to the bottom for the same reason, that neither has measurements to sort by.
_NOT_ANALYSED = tuple(s.value for s in NOT_ANALYSED_STATUSES)


def _ordered(rows: Sequence[CellRow]) -> list[CellRow]:
    """Live cells first; removed ones at the bottom with their reason."""
    live = [r for r in rows if r.get("status") not in _NOT_ANALYSED]
    dead = [r for r in rows if r.get("status") in _NOT_ANALYSED]
    return live + dead


def _cell_sheet(path: Path, header: Sequence[str], rows: Sequence[CellRow],
                *, skip_unskeletonised: bool = False) -> Path:
    """One of the seven per-cell sheets, selected out of the shared `values` dict.

    **Seven column selections, not seven flatteners.** `CellRow.values` already holds every
    per-cell measure under its exported name, so a sheet is a projection of it. That is what
    keeps the split from multiplying the places a measure can be written wrongly.

    ``skip_unskeletonised`` omits the row entirely rather than writing it blank, and it is the
    whole reason the split is worth making. A cell rejected as `real_cell` was never
    skeletonised — `Pool.pending_measurement` admits only confirmed and candidate cells — so on
    the two skeleton sheets **absence means never attempted**, while a present row with a
    `failed` status means attempted and failed. Written as a blank row those two would be
    indistinguishable, which is the distinction this codebase exists not to lose.
    """
    chosen = _ordered(rows)
    if skip_unskeletonised:
        chosen = [r for r in chosen
                  if (r.values.get("skeleton_status") or "") not in ("", "not_attempted")]
    return _write_records(path, header, (r.values for r in chosen))


def write_branches_csv(path: Path, rows: Sequence[CellRow]) -> Path:
    """One row per branch of the soma-anchored tree — **long since 2026-09-01**.

    This was `branch_lengths.csv`: one row per cell, `branch_1_um … branch_N_um`, as wide as
    the busiest cell in the run. That shape had five faults and every one of them is structural
    rather than cosmetic — the header depended on the data, most of the file was blanks whose
    meaning differed from blanks elsewhere in the export, the column index carried no
    information (`branch_7_um` is the seventh branch *emitted*, not the seventh longest or the
    seventh from the soma), a second fact per branch would have cost another `N` columns, and
    `EXPORT_SPEC.md` §5.1 could only describe it as a rule rather than count it.

    The four measures beside the length needed **no new measurement**: the tree already carried
    parent pointers, per-segment radii and edge lengths. What they buy is the question the wide
    file could not ask — whether distal branches are *thicker* in 5xFAD, the other half of
    de-ramification, which a distribution of lengths alone cannot show.

    `n_branches` is not a column here. It is a row count, the same trade already made for
    `n_cells_measured`.
    """
    body = []
    for row in _ordered(rows):
        for branch in row.branches:
            body.append([row.cell_uid, row.file] +
                        [branch.get(c) for c in BRANCHES_ANCHORED[2:]])
    return write_table(path, BRANCHES_ANCHORED, body)


def write_segments_csv(path: Path, rows: Sequence[CellRow]) -> Path:
    """One row per inter-junction segment of the soma-free skeleton — new 2026-09-01.

    Agenda item A8's home, and the soma-free twin of `branches_anchored.csv`. Six of the eight
    columns match its sister; the two that differ are forced rather than chosen. **Order cannot
    cross over** — centrifugal and Strahler order are defined from a root and this skeleton has
    none — so `segment_type` stands in for them, saying what a segment connects. And
    **`component_id` exists only here**, because only this artefact keeps detached material:
    without it `detached_skeleton_um` is a single opaque number rather than one fragment or
    twenty.
    """
    body = []
    for row in _ordered(rows):
        for seg in row.segments:
            body.append([row.cell_uid, row.file] +
                        [seg.get(c) for c in SEGMENTS_FREE[2:]])
    return write_table(path, SEGMENTS_FREE, body)


def write_sholl_csv(path: Path, rows: Sequence[CellRow]) -> Path:
    """One row per cell, metric and shell — **long since 2026-09-01**, and renamed.

    `metric` was already a row value rather than a column suffix, for the reason §5 has always
    given: putting the distinction **in the data** keeps two similarly-named profiles from being
    read as one thing (`DESIGN_RATIONALE.md` §4.12). The shells were still columns, so this
    finishes an idiom the file had already half-adopted. `n_shells` and `step_um` become a row
    count and an `export_summary.csv` row.

    The two metrics are different measures, not one measure in two units. Euclidean is Sholl
    (1953) and what the literature reports; a tortuous process can cross one shell repeatedly,
    so counts can exceed the process count. Geodesic is monotone along the skeleton, so each
    process crosses each shell once and the count *is* the number of paths at least that long —
    cleaner, and comparable to nothing published. **Report Euclidean outward.** Their ratio is
    itself a tortuosity measure, which is why both are carried.

    `_anchored` in the filename because Sholl needs an origin and is therefore inherently
    soma-anchored — which the old name did not say.
    """
    body = []
    for row in _ordered(rows):
        for metric in ("euclidean", "geodesic"):
            profile = row.sholl.get(metric) or {}
            for index, radius in enumerate(sorted(profile), start=1):
                body.append([row.cell_uid, row.file, metric, index,
                             float(radius), profile[radius]])
    return write_table(path, SHOLL_ANCHORED, body)


def write_cd68_curve_csv(path: Path, rows: Sequence[CellRow], columns: Columns) -> Path:
    """One row per cell and threshold level — **long since 2026-09-01**.

    The wide form encoded the level in the **column name**, as an absolute intensity: at a
    chosen threshold of 50 the quarter-level column was `cd68_volume_um3@12.5`, so the header
    changed whenever the threshold did. That is the same "schema depends on the data" fault as
    the two variable-width files, wearing a different costume, and it is why §5 needed a
    paragraph explaining how to decode the suffix. As a row value the level is an ordinary
    number and the paragraph is unnecessary.

    **With no threshold chosen the file keeps its six columns and has no rows.** The levels are
    multiples of `T`, so an unset `T` yields nothing to compute rather than something to guess —
    §2's rule, now expressed in rows instead of in columns.
    """
    body = []
    for row in _ordered(rows):
        for level in columns.cd68_levels:
            body.append([row.cell_uid, row.file, row.get("cd68_status"), float(level),
                         _by_number(row.cd68_volume_curve, level),
                         _by_number(row.cd68_fraction_curve, level)])
    return write_table(path, CD68_CURVE, body)


def write_links_csv(path: Path, rows: Sequence[CellRow], columns: Columns) -> Path:
    """One row per cell, plaque and tolerance — the only join between the two.

    **Four columns joined on 2026-09-01.** Every other plaque measure is anonymous:
    `nearest_voxel_to_plaque_um` says how far but not which, `n_cells_within@d` says how many
    but not who. `contact_area_um2` and `soma_distance_um` make the pair say whether the cell
    body moved or only a process reached; `is_nearest_plaque` makes the nearest-plaque
    attribution explicit instead of an approximation documented elsewhere; and `polarity_index`
    is the one measure here that is a **direction** rather than a quantity, inherently per cell
    *and* per plaque, so no other sheet could hold it.
    """
    body = []
    for row in _ordered(rows):
        for tol in sorted(row.links):
            for plaque_uid, distance in sorted(row.links[tol].items()):
                extra = (row.link_extras.get((tol, plaque_uid))
                         or row.link_extras.get(plaque_uid) or {})
                body.append([row.cell_uid, row.file, plaque_uid, tol, distance]
                            + [extra.get(c) for c in LINK_COLUMNS[5:]])
    return write_table(path, columns.links, body)


def export_pool(
    pool,
    out_dir: str | Path,
    *,
    params_hash: str | None = None,
    params=None,
    parameters_path: str | Path | None = None,
) -> dict:
    """Write one run's export: a dated folder of CSVs, internally homogeneous.

    **A re-run produces a separate export rather than adding columns to an existing one.**
    That is the user's decision, 2026-08-10, and it is what removed the side-by-side sheet and
    the per-hash column suffixes. The `measurements` table's `(cell_uid, params_hash)` key is
    unchanged — the pool may still hold several parameter sets; only the *export* stops
    merging them. A pool holding more than one set and no `params_hash` is therefore an error
    to be resolved by the caller, not something to merge silently.

    **The file set is closed**: the eight CSVs below, plus `parameters.json` and
    `export_summary.csv`. There is no second, optional writer — the `3dmorph/` folder and its
    `--compare-3dmorph` flag were removed 2026-08-11.
    """
    out_root = Path(out_dir)
    hashes = [params_hash] if params_hash else pool.parameter_sets()
    if not hashes:
        raise ValueError("no measurements in this pool; run Phase C before exporting")
    if len(hashes) > 1:
        raise ValueError(
            "this pool holds " + str(len(hashes)) + " parameter sets ("
            + ", ".join(h[:8] for h in hashes) + "). One export describes one run; "
            "pass --params-hash to choose which.")
    chosen = hashes[0]

    # Item 6, 2026-08-12 — `EXPORT_AUDIT.md` §4.1. **Refuse rather than guess.**
    #
    # `build_columns(None)` cannot see what the run used and falls back to defaults, so an
    # export without the parameters file did not merely omit values — it **stated false ones**.
    # Against a pool measured at `cd68_threshold=100.0`, `plaque_threshold=500.0`,
    # `min_plaque_volume_um3=1.0`, `export_summary.csv` asserted `cd68_threshold: UNSET` and
    # `min_plaque_volume_um3: 25.0`, and 10 `cd68_curve.csv` columns plus 5 burden columns
    # disappeared, because those names carry absolute levels and an unset threshold yields none.
    #
    # That file exists so **"a reader holding only the CSVs can decode the column names without
    # opening a second file"**. `cli.py` printed a NOTE, but the CSVs are what survive.
    #
    # **Reading the values back out of the pool was the alternative and was not chosen** — the
    # user's decision. They are there, on every measurement at `payload["coefficients"]`
    # (`phasec.measure_file`), which is why the error can point at them.
    #
    # **This guard runs after the two pool guards above, deliberately.** The message promises
    # the values are recoverable from the pool, and that is only true once the pool is known to
    # hold exactly one measured run. On an unmeasured pool it would send the reader somewhere
    # empty. `test_the_pools_own_errors_come_first` pins the order.
    if params is None:
        raise ValueError(
            "export needs --params: the parameters file for this run.\n"
            "Without it the export cannot know the cutoffs the run used, and "
            "export_summary.csv would state defaults as though they were this run's values — "
            "asserting a threshold was UNSET when it was set, and the wrong plaque size "
            "floor. The curve columns would go missing too: their names carry absolute "
            "levels, so an unset threshold produces no columns rather than wrong ones.\n"
            f"The true values are recorded in the pool at {getattr(pool, 'root', '')} — every "
            f"measurement row at parameter set {chosen[:8]} carries them under "
            "payload['coefficients'], written by phasec.measure_file."
        )

    columns = build_columns(params)
    folder = out_root / f"export_{date.today():%Y%m%d}_{chosen[:8]}"
    folder.mkdir(parents=True, exist_ok=True)

    rows = build_rows(pool, chosen, columns)
    by_file: dict[str, list[CellRow]] = {}
    by_uid: dict[str, list[CellRow]] = {}
    for row in rows:
        by_file.setdefault(row.file, []).append(row)
        if row.file_uid:
            by_uid.setdefault(row.file_uid, []).append(row)

    files = {f.file_uid: f for f in pool.files()}
    image_records: list[dict] = []
    curve_records: list[dict] = []
    plaque_records: list[dict] = []
    for image in pool.image_measurements(chosen):
        record = files.get(image["file_uid"])
        source = Path(record.source_path) if record else Path("")
        identity = _image_identity(source)
        #: **`file_uid` on `images.csv` alone**, because it is the only sheet without one —
        #: `cell_uid` and `plaque_uid` both embed it. Until 2026-09-01 this sheet joined on
        #: `file`, which is the basename, in an export whose own design distrusts filenames
        #: enough to read genotype and animal from the path instead.
        identity["file_uid"] = image["file_uid"]
        payload = image.get("payload") or {}
        #: Grouped by `file_uid`, not by basename. The four image aggregates that used this
        #: group moved into `phasec.measure_file` in the same change, so nothing downstream
        #: depends on it any more — but two images sharing a basename would have had their
        #: cells merged here, and both rows would have reported the union.
        group = by_uid.get(image["file_uid"], by_file.get(source.name, []))
        image_records.append(_flatten_image(payload, identity, group, columns))
        curve_records.append(_flatten_image_curves(payload, identity, columns))
        for plaque in payload.get("plaques") or []:
            plaque_records.append(_flatten_plaque(plaque, identity, columns,
                                                  file_uid=image["file_uid"]))

    #: **Sixteen files since 2026-09-01**, where there were eight. `cells.csv` became seven
    #: sheets split by kind of measurement; `branch_lengths.csv` and `sholl.csv` went long and
    #: were renamed for the skeleton they belong to; `segments_free.csv` is new.
    #:
    #: **The two skeleton sheets skip unskeletonised cells rather than writing them blank.**
    #: That is what makes each file's population readable from the file itself: absence means
    #: never attempted, a present row with a `failed` status means attempted and failed.
    written = {
        "cells_identity": _cell_sheet(folder / "cells_identity.csv", CELLS_IDENTITY, rows),
        "cells_review": _cell_sheet(folder / "cells_review.csv", CELLS_REVIEW, rows),
        "cells_mask": _cell_sheet(folder / "cells_mask.csv", CELLS_MASK, rows),
        "cells_skeleton_anchored": _cell_sheet(
            folder / "cells_skeleton_anchored.csv", CELLS_SKELETON_ANCHORED, rows,
            skip_unskeletonised=True),
        "cells_skeleton_free": _cell_sheet(
            folder / "cells_skeleton_free.csv", CELLS_SKELETON_FREE, rows,
            skip_unskeletonised=True),
        "cells_cd68": _cell_sheet(folder / "cells_cd68.csv", CELLS_CD68, rows),
        "cells_plaque": _cell_sheet(folder / "cells_plaque.csv", columns.cells_plaque, rows),
        "plaques": _write_records(folder / "plaques.csv", columns.plaques,
                                  plaque_records),
        "images": _write_records(folder / "images.csv", columns.images, image_records),
        "cell_plaque_links": write_links_csv(folder / "cell_plaque_links.csv", rows,
                                             columns),
        "branches_anchored": write_branches_csv(folder / "branches_anchored.csv", rows),
        "segments_free": write_segments_csv(folder / "segments_free.csv", rows),
        "sholl_anchored": write_sholl_csv(folder / "sholl_anchored.csv", rows),
        "cd68_curve": write_cd68_curve_csv(folder / "cd68_curve.csv", rows, columns),
        "image_curves": _write_records(folder / "image_curves.csv", columns.image_curves,
                                       curve_records),
    }

    _write_parameters(folder / "parameters.json", parameters_path, params, chosen)
    n_rejected = sum(1 for r in rows if r.get("status") == "rejected")
    summary = _write_summary(folder / "export_summary.csv", pool, columns, chosen,
                             rows=rows, n_cells=len(rows), n_rejected=n_rejected,
                             n_plaques=len(plaque_records), n_images=len(image_records))

    return {"folder": folder, "files": written, "summary": summary,
            "params_hash": chosen, "n_cells": len(rows),
            "n_plaques": len(plaque_records), "n_images": len(image_records),
            "columns": columns}


def _write_parameters(path: Path, parameters_path, params, params_hash: str) -> Path:
    """A **verbatim copy** of the run's parameters file, so it cannot drift from the export.

    With no file supplied the parameters object is serialised instead, and the difference is
    recorded — a reconstructed file and a copied one are not the same evidence.
    """
    if parameters_path is not None and Path(parameters_path).is_file():
        shutil.copyfile(Path(parameters_path), path)
        return path
    from dataclasses import asdict, is_dataclass
    payload = {
        "params_hash": params_hash,
        "source": ("reconstructed from the parameters object; no parameters file was "
                   "supplied to the export"),
        "params": asdict(params) if is_dataclass(params) else None,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _real_cell_loss_rows(rows: Sequence[CellRow]) -> list[tuple[str, object]]:
    """The removal counts by reason, whole-run and **per genotype**.

    A cell removed as `debris` is a correct removal. A cell removed as `real_cell` is a **real
    cell that was removed anyway** — the reviewer ran out of time, or could not fix it — which
    is a data loss, and the log's own tuning says the expensive-to-fix cells are the clumped
    5xFAD ones. So the loss rate is expected to depend on genotype, and if it does, every
    downstream comparison is running on a sample thinned unevenly between the groups.

    What is emitted is the 2x2 contingency table that question needs — cells and real-cell
    losses for each genotype — not a single pooled rate, because a pooled rate is precisely
    the summary in which a genotype-dependent loss disappears. The per-cell rows in
    `cells.csv` carry `genotype`, `status` and `reject_reason`, so the same table is
    reconstructable per image and per animal as well; these rows exist so the headline number
    cannot be missed. **The reviewer never sees any of this** — it only aggregates the two
    reasons they were asked for.

    **Rates are blank rather than 0 when a genotype has no cells.** 0/0 is not zero.

    > ### ⇒ THE RATE IS R/(C+R), AND BOTH HALVES MOVED — 2026-08-30
    >
    > It was **R/(C+R+D)**, counting rows, which disagreed with this function's own docstring
    > and with `n_real_cells`' definition two files away: debris is not a cell, so it has no
    > business in the denominator of *what fraction of the cells were lost*. Two changes, one
    > formula:
    >
    > * **R is multiplicity-factoring** — a rejection worth 3 cells counts 3. The user's rule,
    >   and it holds wherever R appears rather than in the numerator alone.
    > * **D leaves the denominator.** The population is the cells: confirmed plus lost.
    >
    > **`n_cells_confirmed` and `n_real_cells` are emitted beside the rate so it can be checked
    > by hand.** Without them the rate is unrecomputable from this file: `n_cells` is a row
    > count and deliberately stays one, and once R counts cells rather than rows there is no
    > column left from which to recover the denominator. The user asked for both.
    """
    def counts(subset: Sequence[CellRow]) -> tuple[int, int, int, int]:
        debris = sum(1 for r in subset if r.get("reject_reason") == "debris")
        confirmed = sum(1 for r in subset if r.get("status") in _KEPT)
        #: **Cells, not rows.** `debris` and `confirmed` stay row counts: debris is not a cell
        #: and a confirmed cell is always exactly one, the count popup never firing on accept.
        real_cell = sum(_multiplicity(r) for r in subset
                        if r.get("reject_reason") == "real_cell")
        return len(subset), debris, confirmed, real_cell

    def block(label: str, subset: Sequence[CellRow]) -> list[tuple[str, object]]:
        total, debris, confirmed, real_cell = counts(subset)
        suffix = f"_{label}" if label else ""
        out: list[tuple[str, object]] = []
        if label:
            out.append((f"n_cells{suffix}", total))
        out.append((f"n_cells_debris{suffix}", debris))
        out.append((f"n_cells_confirmed{suffix}", confirmed))
        out.append((f"n_cells_lost_real{suffix}", real_cell))
        #: The denominator, named as `images.csv` already names the same quantity, so the two
        #: files agree on what *the cells* means.
        real_cells = confirmed + real_cell
        out.append((f"n_real_cells{suffix}", real_cells))
        out.append((f"real_cell_loss_rate{suffix}",
                    "" if real_cells == 0 else round(real_cell / real_cells, 6)))
        return out

    out = block("", rows)
    genotypes = sorted({str(r.get("genotype") or "unknown") for r in rows})
    for genotype in genotypes:
        out += block(genotype, [r for r in rows
                                if str(r.get("genotype") or "unknown") == genotype])
    return out


def _write_summary(path: Path, pool, columns: Columns, params_hash: str, *,
                   rows: Sequence[CellRow] = (), n_cells: int, n_rejected: int,
                   n_plaques: int, n_images: int) -> Path:
    """Counts, date and source pool — plus every value a column name depends on.

    The thresholds are here as well as in `parameters.json` because the curve columns carry
    absolute levels in their names, and a reader holding only the CSVs must be able to decode
    them without opening a second file.
    """
    unset = "UNSET - measures behind it are NaN with a status, never zero"
    summary = [
        ("exported_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("source_pool", str(getattr(pool, "root", ""))),
        ("params_hash", params_hash),
        ("n_images", n_images),
        ("n_cells", n_cells),
        # Every removal is a rejection again, so this is the total and the two reason counts
        # below decompose it. It stopped being the total for part of 2026-08-11, while a
        # `deferred` status held one of the two kinds; that status is gone.
        ("n_cells_rejected", n_rejected),
        *_real_cell_loss_rows(rows),
        ("n_plaques", n_plaques),
        ("cd68_threshold",
         unset if columns.cd68_threshold is None else columns.cd68_threshold),
        ("plaque_threshold",
         unset if columns.plaque_threshold is None else columns.plaque_threshold),
        ("min_plaque_volume_um3", columns.min_plaque_volume_um3),
        ("contact_tolerances_um", " ".join(_g(t) for t in columns.tolerances)),
        ("recruitment_distances_um", " ".join(_g(r) for r in columns.recruitment)),
        ("curve_factors", " ".join(_g(f) for f in CURVE_FACTORS)),
        # The morphometry knobs. `min_spur_um` is here even though it appears in
        # no column *name* (NEW-j): every branch, endpoint and skeleton figure in this export is
        # a function of it, and at 1.0 it removes 31% of all branches — so two exports differing
        # only in it would otherwise be indistinguishable from the evidence in the folder.
        ("sholl_step_um", columns.sholl_step_um),
        ("min_spur_um", columns.min_spur_um),
        #: > ### ⇒ THE SOMA PARAMETERS JOINED FOR EXACTLY `min_spur_um`'s REASON — 2026-09-01
        #: >
        #: > Since the collapse, `soma_method`, `soma_thickness_um` and `soma_anchor_um` move
        #: > junction counts and primary-process counts, not only volumes — the soma determines
        #: > the tree's topology. So the sentence above is now equally true of them: two exports
        #: > differing only in a soma parameter would otherwise be indistinguishable from the
        #: > evidence in the folder. `parameters.json` has always carried them; this file exists
        #: > so a reader holding only the CSVs does not need to open it.
        #: >
        #: > `loop_prune_method` joins for the same reason — it decides which loop edges survive
        #: > and therefore every count taken off the graph.
        ("soma_method", columns.soma_method),
        ("soma_thickness_um", columns.soma_thickness_um),
        ("soma_anchor_um", columns.soma_anchor_um),
        ("loop_prune_method", columns.loop_prune_method),
        #: The skeleton QC totals, which live here rather than on `images.csv`: they are run
        #: quality rather than a per-image result, and they qualify no image-level aggregate —
        #: which is the only thing that earned the cell border counts their place there.
        *_skeleton_qc_rows(rows),
    ]
    return write_table(path, ["key", "value"], summary)


def _skeleton_qc_rows(rows: Sequence["CellRow"]) -> list[tuple[str, object]]:
    """How the skeletonisation went across the run — added 2026-09-01.

    Counted over rows that **reached** skeletonisation, so `n_cells_skeletonised` is the
    denominator the other four are read against. A cell rejected as `real_cell` is not counted
    as a failure: it was never attempted, which `skeleton_status` says as `not_attempted`.
    """
    attempted = [r for r in rows
                 if (r.values.get("skeleton_status") or "") not in ("", "not_attempted")]
    failed = [r for r in attempted if r.values.get("skeleton_status") == "failed"]
    collapse = [r.values.get("collapse_status") for r in attempted]
    detached = [r for r in attempted
                if _positive(r.values.get("detached_skeleton_um"))]
    return [
        ("n_cells_skeletonised", len(attempted)),
        ("n_cells_skeleton_failed", len(failed)),
        ("n_cells_no_soma", sum(1 for c in collapse if c == "no_soma")),
        ("n_cells_collapse_fallback", sum(1 for c in collapse if c == "nearest_node")),
        ("n_cells_with_detached_material", len(detached)),
    ]


def _positive(value: object) -> bool:
    try:
        return float(value) > 0.0            # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def write_cell_figures(
    out_dir: str | Path,
    stem: str,
    mask: np.ndarray,
    *,
    voxel_size_um,
    skeleton: np.ndarray | None = None,
    endpoints: np.ndarray | None = None,
    branch_points: np.ndarray | None = None,
    hull: bool = True,
    dpi: int = 100,
) -> list[Path]:
    """3DMorph's four figures per cell, plus the convex hull it displays but never saves.

    Written for **every** cell, batch included. The user chose that and the arithmetic
    supports it: measured from their own 3DMorph output, four JPEGs come to ~295 KB per cell,
    so ~20,000 cells is **≈5.9 GB and 80,000 files** — 2% of their block quota and 8% of their
    file quota. Rendering parallelises per image across the Slurm array.

    JPEG to match 3DMorph, which is also what the size estimate assumes.

    Surfaces are drawn with `marching_cubes` in **physical microns**, so an anisotropic cell
    is not rendered squashed — the same reason every measurement here is physically scaled.
    """
    import matplotlib
    matplotlib.use("Agg")                          # no display on a compute node
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = tuple(float(v) for v in voxel_size_um)
    written: list[Path] = []

    def surface(ax, volume, colour, alpha):
        if volume is None or not volume.any() or min(volume.shape) < 2:
            return
        try:
            verts, faces, _, _ = marching_cubes(volume.astype(float), level=0.5,
                                                spacing=spacing)
        except (RuntimeError, ValueError):
            return                                 # too thin to triangulate; not an error
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        mesh.set_facecolor(colour)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, volume.shape[0] * spacing[0])
        ax.set_ylim(0, volume.shape[1] * spacing[1])
        ax.set_zlim(0, volume.shape[2] * spacing[2])

    def points(ax, volume, colour, size):
        if volume is None or not volume.any():
            return
        coords = np.argwhere(volume) * np.asarray(spacing)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colour, s=size, depthshade=False)

    panels = [("Original", lambda ax: surface(ax, mask, "tab:blue", 0.35))]
    if skeleton is not None:
        panels.append(("Skeleton", lambda ax: points(ax, skeleton, "tab:orange", 1)))
    if endpoints is not None:
        panels.append(("Endpoints", lambda ax: (surface(ax, mask, "tab:blue", 0.08),
                                                points(ax, endpoints, "tab:red", 12))))
    if branch_points is not None:
        panels.append(("Branchpoints", lambda ax: (surface(ax, mask, "tab:blue", 0.08),
                                                   points(ax, branch_points, "tab:green", 12))))
    if hull:
        panels.append(("ConvexHull", lambda ax: surface(ax, _hull_volume(mask),
                                                        "tab:purple", 0.20)))

    for name, draw in panels:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        draw(ax)
        ax.set_axis_off()
        path = out_dir / f"{stem}_{name}.jpg"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pil_kwargs={"quality": 85})
        plt.close(fig)
        written.append(path)
    return written


def _hull_volume(mask: np.ndarray) -> np.ndarray | None:
    """Filled convex hull of a cell — 3DMorph shows this and never writes it out.

    A cell too small or too flat to triangulate has no 3D hull. That is an ordinary
    condition, not a fault, so the warning skimage emits for it is suppressed — across 700
    images it would otherwise print once per degenerate cell and bury real messages.
    """
    import warnings

    from skimage.morphology import convex_hull_image

    if mask.sum() < 4:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return convex_hull_image(mask)
    except Exception:
        return None



# ---------------------------------------------------------------------------
# per-cell mask TIFFs — the handoff to Fiji, MATLAB, Imaris
# ---------------------------------------------------------------------------

#: The value a set voxel takes in an exported mask. **255, not 1.** Fiji's `Skeletonize (2D/3D)`
#: and `Analyze Skeleton` operate on 8-bit binary images, which ImageJ defines as 0 and 255; an
#: image of 0s and 1s opens as an almost-black frame that the plugin will still process but that
#: a person cannot see or threshold by eye. The cost of being wrong here is silent — a skeleton
#: comes out either way — so it is fixed here rather than left to the caller.
MASK_FOREGROUND = 255

#: Voxels of background added around every exported mask, each axis, both ends.
#:
#: **Not cosmetic.** The stored mask is cropped tight to its bounding box, so a cell touches the
#: array edge on all six faces by construction. A thinning algorithm cannot tell a face that is
#: the edge of the *array* from a face that is the edge of the *object*, so an unpadded crop
#: skeletonises differently at the border than the same cell would inside a larger frame.
#: `morphometrics._surface` pads by one voxel before `marching_cubes` for exactly this reason;
#: the same decision is made here so an external skeleton and this program's own skeleton see
#: the same geometry.
#:
#: It shifts every voxel by `(1, 1, 1)`, which is why `manifest.csv` carries `pad`: a skeleton
#: point mapped back to source-image coordinates is `bbox` origin + point - pad.
MASK_PAD = 1

#: `manifest.csv`'s header. **`cell_uid` first because it is the join.** `cells.csv` carries the
#: same column, so a skeleton measured in Fiji can be placed beside everything this program
#: measured about the same cell — which is the whole point of exporting the masks rather than
#: only the numbers derived from them.
MASK_MANIFEST_COLUMNS = (
    "cell_uid", "file", "path",
    #: The two skeletons beside the mask, and the sidecars that make them lossless — added
    #: 2026-09-01. Relative like `path`, so the folder stays portable.
    "anchored_swc", "anchored_extra_edges", "free_swc", "free_extra_edges",
    "status", "reject_reason", "human_edit_kind",
    "n_voxels", "volume_um3",
    "z0", "z1", "y0", "y1", "x0", "x1",
    "voxel_z_um", "voxel_y_um", "voxel_x_um", "pad", "source_path",
)


def _write_cell_skeletons(pool, cell, params_hash, sub, folder) -> dict[str, object]:
    """Both skeletons as SWC, each with its sidecar. Blank paths when none were stored.

    **Never raises.** A skeleton that cannot be read costs its four manifest cells, not the
    mask export — the same isolation rule `measure_one` follows, and for the same reason: this
    is a handoff, and a handoff that fails wholesale because one cell is odd is worse than one
    that is honest about which cell is missing.
    """
    blank = {"anchored_swc": "", "anchored_extra_edges": "",
             "free_swc": "", "free_extra_edges": ""}
    if params_hash is None:
        return blank
    try:
        arrays = pool.load_skeletons(cell.cell_uid, params_hash)
    except Exception:                       # noqa: BLE001 - see the docstring
        return blank
    if not arrays:
        return blank

    origin = getattr(cell, "soma_centroid", None)
    out = dict(blank)
    for kind, prefix in (("soma_anchored", "anchored"), ("soma_free", "free")):
        art = swc_module.artefact_from_arrays(arrays, kind)
        if art is None or art.n_nodes == 0:
            continue
        try:
            written = swc_module.write_swc(
                sub / f"{cell.cell_uid}_{prefix}.swc", art,
                cell_uid=cell.cell_uid, kind=kind, origin_um=origin)
            _, extra = swc_module.swc_rows(art, origin)
            side = swc_module.write_extra_edges(
                sub / f"{cell.cell_uid}_{prefix}_extra_edges.csv", art, extra)
        except Exception:                   # noqa: BLE001 - see the docstring
            continue
        out[f"{prefix}_swc"] = str(written.relative_to(folder)).replace("\\", "/")
        out[f"{prefix}_extra_edges"] = str(side.relative_to(folder)).replace("\\", "/")
    return out


def _mask_subfolder(record) -> str:
    """`<genotype>/<animal>/<image stem>` — the input tree, mirrored. **2026-09-01.**

    It was one flat directory per image. Mirroring the source layout is the user's choice: the
    folder is meant to be opened beside the images it came from, and a thousand sibling
    directories named for `.ome` stems is not that.

    **Falls back to `<file_uid>` when the path does not parse.** Genotype and animal come from
    the *path*, and not every input tree is arranged the way this study's is — a file sitting
    at the top level yields neither. `file_uid` is always present and unique, so the fallback is
    a valid folder rather than a blank one, and its shape says plainly that the path was not
    understood.

    `FileRecord.source_stem` is the schema's own display label and is filename-derived, so it is
    safe on every target filesystem by construction. The trailing `.ome` is dropped because
    `_stem` removes only the final extension, and `..._result.ome` reads as a mistake.
    """
    stem = str(getattr(record, "source_stem", "") or "image")
    stem = stem[:-4] if stem.lower().endswith(".ome") else stem
    meta = parse_metadata(Path(str(getattr(record, "source_path", "") or "")))
    genotype = str(getattr(meta, "genotype", "") or "")
    animal = str(getattr(meta, "animal", "") or "")
    if not genotype or not animal:
        return str(getattr(record, "file_uid", "") or stem)
    return f"{genotype}/{animal}/{stem}"


def write_mask_tiff(path: str | Path, mask: np.ndarray, voxel_size_um,
                    *, pad: int = MASK_PAD) -> Path:
    """One cell's mask as an 8-bit binary ImageJ TIFF, calibrated in microns.

    **The calibration is the point.** A bare array of voxels is not a usable handoff at this
    project's 5.5:1 anisotropy: a tool that assumes isotropic voxels reports a length along z
    as though it were a length along x, and nothing in the file contradicts it. Writing the
    ImageJ `spacing`/`unit` metadata means Fiji opens the stack already scaled, and
    `Analyze Skeleton` reports microns without anybody setting the scale by hand — the step
    most likely to be forgotten and impossible to detect afterwards from the numbers.

    **`imagej=True` rather than OME-XML**, unlike `write_label_image`. That writer targets
    Imaris, which reads OME; this one targets Fiji, which reads ImageJ metadata natively and
    needs no importer dialogue. Both are ordinary TIFFs to everything else.
    """
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_um, y_um, x_um = (float(v) for v in voxel_size_um)

    volume = np.asarray(mask).astype(bool)
    if pad:
        volume = np.pad(volume, pad)
    frames = volume.astype(np.uint8) * np.uint8(MASK_FOREGROUND)

    tifffile.imwrite(
        path, frames,
        imagej=True, resolution=(1.0 / x_um, 1.0 / y_um),
        metadata={"spacing": z_um, "unit": "um", "axes": "ZYX"},
    )
    return path


#: **Where masks live under the output root**, named once so the two routes cannot disagree —
#: patch item C6, 2026-09-03. `ui/screen_save.DEFAULT_MASKS_DIR` is this, and `cli.py`'s
#: `--with-masks` resolves to it as well. A literal in three places is how they drifted.
MASKS_DIR = "masks"


def export_cell_masks(
    pool,
    out_dir: str | Path,
    *,
    params_hash: str | None = None,
    include_unreviewed: bool = False,
) -> dict:
    """Every kept cell's mask as its own TIFF, plus the manifest that joins them to `cells.csv`.

    **A separate export from `export_pool`, on purpose.** A mask is a property of the *pool*: it
    is written once by `PoolWriter.add_cell` and no later phase alters it, so it does not depend
    on a parameter set and does not change when a run is re-measured. `export_pool`'s folder is
    keyed by `params_hash` because its contents do change; putting an immutable 20,000-file
    dataset inside it would mean every re-export either duplicated it or special-cased it.
    Separating them also means masks are available **before** Phase C — the reviewer can carry a
    cell into Fiji the moment the session is saved, rather than after choosing thresholds they
    may not care about.

    **One TIFF per cell rather than one label volume per image.** The label volume is
    `write_label_image`'s job and Imaris' preferred input; it is the wrong shape for
    skeletonisation, where the unit of work is a single cell and every tool wants it isolated.

    **Which cells.** `KEPT_STATUSES` by default — the cells a reviewer judged real and left in
    the analysis. Not `CellStatus.CONFIRMED` alone: `schema.KEPT_STATUSES` records why, and the
    same trap is why `include_unreviewed` exists rather than a bare status argument. Passing
    `params_hash` narrows further, to cells holding a measurement at that hash, which is exactly
    the population of `cells.csv` — the strongest form of the join when the point of the export
    is to compare an external skeleton against this program's own.
    """
    #: > ### ⇒ WITHOUT `--params-hash` IT SILENTLY DROPPED EVERY SKELETON — patch item C1
    #: >
    #: > Measured on one pool, the same command otherwise:
    #: >
    #: >     masks WITHOUT --params-hash : 12 .tif,  0 .swc,  1 .csv
    #: >     masks WITH    --params-hash : 12 .tif, 24 .swc, 25 .csv
    #: >
    #: > You got the TIFFs, no skeletons, and a `manifest.csv` whose `anchored_swc`,
    #: > `anchored_extra_edges`, `free_swc` and `free_extra_edges` columns were **blank in every
    #: > row**. Exit 0, no warning. And item 8's generated command omitted the flag — for a
    #: > reason that is right at item 8, where nothing is measured yet, and wrong by the time
    #: > anyone runs it.
    #: >
    #: > **So the parameter set is resolved here when the caller names none**, exactly as
    #: > `export_pool` resolves it: one set, use it; several, refuse and name them; none, write
    #: > the masks alone — which is correct at item 8's moment, and is the case the old default
    #: > was really serving.
    #: >
    #: > `export --with-masks` was already correct: it passes the export's own `params_hash`
    #: > through, so only the standalone command ever had the trap.
    if params_hash is None:
        known = pool.parameter_sets()
        if len(known) > 1:
            raise ValueError(
                "this pool holds " + str(len(known)) + " parameter sets ("
                + ", ".join(h[:8] for h in known) + "). The skeletons belong to one run; "
                "pass --params-hash to choose which.")
        params_hash = known[0] if known else None

    out_root = Path(out_dir)
    folder = out_root / f"masks_{date.today():%Y%m%d}"
    folder.mkdir(parents=True, exist_ok=True)

    #: > ### ⇒ "C ONLY" WAS ALREADY THE BEHAVIOUR — checked 2026-09-01, and briefly broken
    #: >
    #: > The user asked that only confirmed cells be exported. `KEPT_STATUSES` is
    #: > `(CONFIRMED, ANALYSED)` and **excludes every rejected status**, so it already was
    #: > exactly that: `ANALYSED` is not a third population, it is a confirmed cell that Phase C
    #: > has finished with.
    #: >
    #: > Narrowing this to `CONFIRMED` alone — done here for one commit — drops every cell that
    #: > has been measured, which is most of them by the time anyone exports, and the folder
    #: > comes out plausibly sized and quietly incomplete. That is the trap `schema.KEPT_STATUSES`
    #: > exists to record, and `test_an_analysed_cell_is_still_exported` caught it.
    #: >
    #: > The population **is** identical to the skeletons' by construction: a rejected cell is
    #: > never skeletonised, and no rejected cell is here.
    wanted = {s.value for s in KEPT_STATUSES}
    if include_unreviewed:
        wanted |= {s.value for s in UNREVIEWED_STATUSES}

    measured: set[str] | None = None
    if params_hash is not None:
        measured = {m["cell_uid"] for m in pool.measurements(params_hash)}

    files = {f.file_uid: f for f in pool.files()}
    records: list[dict[str, object]] = []
    #: Cells selected but whose arrays are gone. A crashed Phase A task can leave a manifest
    #: entry with no `.npz` (`Pool.check_integrity` reports exactly this), and a mask that
    #: cannot be read must be **named**, never skipped in silence — the count is what tells a
    #: reader the folder is short of the cells.csv rows it is supposed to match.
    missing: list[str] = []

    for cell in pool.cells():
        if cell.status.value not in wanted:
            continue
        if measured is not None and cell.cell_uid not in measured:
            continue
        record = files.get(cell.file_uid)
        if record is None:
            missing.append(cell.cell_uid)
            continue
        try:
            arrays = pool.load_arrays(cell.cell_uid)
        except FileNotFoundError:
            missing.append(cell.cell_uid)
            continue
        mask = arrays.get("mask")
        if mask is None:
            missing.append(cell.cell_uid)
            continue

        source = Path(record.source_path)
        sub = folder / _mask_subfolder(record)
        path = write_mask_tiff(sub / f"{cell.cell_uid}.tif", mask, record.voxel_size_um)
        skeleton_paths = _write_cell_skeletons(pool, cell, params_hash, sub, folder)
        z_um, y_um, x_um = (float(v) for v in record.voxel_size_um)
        z0, z1, y0, y1, x0, x1 = cell.bbox
        records.append({
            "cell_uid": cell.cell_uid,
            "file": source.name,
            # Relative, so the folder can be copied to another machine — which is the normal
            # thing to do with it — without every row pointing at a path that no longer exists.
            "path": str(path.relative_to(folder)).replace("\\", "/"),
            "status": cell.status.value,
            "reject_reason": getattr(cell.reject_reason, "value", "") or "",
            #: `human_edit_kind`, not `human_edited` — the boolean left the export on
            #: 2026-09-01 and this manifest was its last surviving home.
            "human_edit_kind": human_edit_kind(cell.human_edited, cell.human_split),
            **skeleton_paths,
            "n_voxels": cell.n_voxels,
            "volume_um3": round(cell.n_voxels * z_um * y_um * x_um, 4),
            "z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
            "voxel_z_um": z_um, "voxel_y_um": y_um, "voxel_x_um": x_um,
            "pad": MASK_PAD,
            "source_path": record.source_path,
        })

    manifest = _write_records(folder / "manifest.csv", MASK_MANIFEST_COLUMNS, records)
    return {"folder": folder, "manifest": manifest, "n_masks": len(records),
            "n_missing": len(missing), "missing": missing,
            "n_images": len({r["file"] for r in records})}

# ---------------------------------------------------------------------------
# Imaris
# ---------------------------------------------------------------------------

def write_label_image(path: str | Path, labels: np.ndarray, voxel_size_um,
                      cell_uids: dict[int, str] | None = None) -> Path:
    """A `uint16` label OME-TIFF for Imaris, with physical voxel size embedded.

    **Labels, never a binary mask.** Imaris builds Surfaces by thresholding and taking
    connected components, so a binary mask would silently re-merge every cell the nuclei
    channel exists to separate (§4.13).

    **Voxel size is not optional.** Without it Imaris assumes a default and every volume is
    wrong — by a large factor at 5.5:1 anisotropy.

    A `label_id ↔ cell_uid` map is written alongside, but the export **does not depend on
    Imaris preserving ids**: whether it renumbers on import is unresolved, so each cell's
    centroid and volume go in the same table and allow geometric matching regardless.
    """
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_um, y_um, x_um = (float(v) for v in voxel_size_um)
    tifffile.imwrite(
        path, labels.astype(np.uint16),
        imagej=True, resolution=(1.0 / x_um, 1.0 / y_um),
        metadata={"spacing": z_um, "unit": "um", "axes": "ZYX"},
    )

    sidecar = path.with_suffix(".labels.csv")
    with sidecar.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label_id", "cell_uid", "n_voxels", "volume_um3",
                         "centroid_z", "centroid_y", "centroid_x"])
        voxel_volume = z_um * y_um * x_um
        for label in range(1, int(labels.max()) + 1):
            where = np.argwhere(labels == label)
            if where.size == 0:
                continue
            centre = where.mean(axis=0)
            writer.writerow([
                label, (cell_uids or {}).get(label, ""), len(where),
                len(where) * voxel_volume,
                round(float(centre[0]), 3), round(float(centre[1]), 3),
                round(float(centre[2]), 3),
            ])
    return path
