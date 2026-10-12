"""Cell pool storage: writing, merging and querying segmented cells.

Layout on disk::

    pool/
      cells/<file_uid>/<cell_uid>.npz   mask, nucleus_mask, thumbnail
      nuclei/<file_uid>.npz             the whole image's surviving nucleus labels
      skeletons/<params_hash>/<file_uid>/<cell_uid>.npz   both skeletons
      fragments/<file_uid>.json         one per Phase A task
      measurement_fragments/<params_hash>/<file_uid>.json   one per Phase C task
      pool.db                           merged SQLite index (files + cells)
      pool.csv                          flat export for spreadsheets

Writing and reading are deliberately separate classes rather than one, because they
happen in different processes on different machines:

* :class:`PoolWriter` runs during Phase A — potentially hundreds of concurrent batch
  tasks. It writes arrays and *one* manifest fragment, and **never opens the
  database**. SQLite over a network filesystem corrupts under concurrent writers, so
  the fragment indirection is what makes parallel segmentation safe.
* :func:`merge_fragments` folds fragments into the database. Single writer, once.
* :class:`MeasurementFragment` runs during Phase C, for exactly the same reason, and
  :func:`merge_measurement_fragments` folds those in. **Phase C had no equivalent until
  2026-09-02** and inserted straight into `pool.db`, so hundreds of concurrent measure
  tasks collided on the lock and each died *after* doing all the expensive work.
* :class:`Pool` serves review and analysis: query, update status, load arrays. Opened
  ``read_only=True`` it runs no DDL and cannot take a write lock — which is what lets an
  array task read cells and masks without becoming a writer.

Making these separate types means the unsafe pattern — a batch task writing to the
shared database — cannot be expressed by accident.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import (
    SCHEMA_VERSION,
    CellRecord,
    CellStatus,
    FileRecord,
    RejectReason,
    SplitMethod,
    make_cell_uid,
    utc_now,
)

__all__ = [
    "PoolWriter",
    "Pool",
    "merge_fragments",
    "MergeResult",
    "IntegrityReport",
    "FragmentError",
    "ORIGINS",
    "pool_dir_name",
    "pool_origin",
    "nucleus_labels_path",
    "MeasurementFragment",
    "merge_measurement_fragments",
    "MeasurementMergeResult",
    "measurement_fragment_path",
]

CELLS_DIR = "cells"
#: One label volume per image — the nuclei **that survived membership**, at segmentation
#: resolution. Not keyed by `params_hash`, like `cells/` and unlike `skeletons/`: it is a
#: Phase A product, and a pool holds exactly one Phase A parameter set by construction
#: (`pool_dir_name` keys on `phase_a_hash`).
NUCLEI_DIR = "nuclei"
#: Skeleton artefacts, keyed by params hash because they move when a measurement
#: parameter moves. See `Pool.skeleton_path`.
SKELETONS_DIR = "skeletons"
FRAGMENTS_DIR = "fragments"
#: Phase C's counterpart, **keyed by `params_hash` in the path** so two parameter sets cannot
#: collide on one image. Phase A's fragments need no such key: a pool holds one Phase A
#: parameter set by construction, and Phase C is the phase that can be re-run against the same
#: cells under different measurement settings.
MEASUREMENT_FRAGMENTS_DIR = "measurement_fragments"
DB_NAME = "pool.db"
FRAGMENT_FORMAT = 1

#: The two origins a pool can have. **A category, not a label** — `run_id` is free text
#: (`--run-id experiment2` produces a batch pool whose records say nothing about being batch),
#: so enforcement and provenance need something structured. `COMPONENT_03_PLAN.md` §4a.5.
ORIGINS = ("semiauto", "batch")


def pool_dir_name(origin: str, phase_a_hash: str) -> str:
    """`pool_semiauto_<phaseA_hash>`. `COMPONENT_03_PLAN.md` §4a.9.

    **One pool per Phase A parameter set**, with the origin in the name: cells produced by
    different segmentation parameters live in different pools, and semi-auto and batch never
    share one. Keyed on `SegmentationParams.phase_a_hash()`, **not** on `hash()` — see that
    method for why a Phase C knob must not strand a cell set.

    **Refuses an unrecognised origin rather than accepting it**, matching the `reject_reason`
    pattern. A typo that produced `pool_semi_auto_...` would be a silently separate pool, which
    is the failure §4a.3's enforcement exists to prevent.
    """
    if origin not in ORIGINS:
        raise ValueError(f"origin is one of {ORIGINS}, got {origin!r}")
    if not phase_a_hash:
        raise ValueError("phase_a_hash is required; it is what separates one pool from another")
    return f"pool_{origin}_{phase_a_hash}"


def pool_origin(pool_root: str | os.PathLike[str]) -> str | None:
    """The origin a pool's **directory name** declares, or `None` when it declares nothing.

    The reader for what :func:`pool_dir_name` writes, and the whole of what enforcement can
    rest on: `run_id` is free text and the database carries no origin column, so the name is
    the only structured place the category lives. `COMPONENT_03_PLAN.md` §4a.5.

    **`None` is not an error.** An operator may point `--pool` at any directory they like, and
    a name that declares nothing is the ordinary case — a hand-made path, a scratch pool, every
    pool in the test suite. Only a name that positively declares the *wrong* origin is
    refusable, which is why this reports rather than raises.
    """
    name = Path(pool_root).name
    for origin in ORIGINS:
        prefix = f"pool_{origin}_"
        if name.startswith(prefix) and name[len(prefix):]:
            return origin
    return None


def nucleus_labels_path(pool_root: str | os.PathLike[str], file_uid: str) -> Path:
    """`pool/nuclei/<file_uid>.npz`. One per image, written by Phase A, read by review."""
    return Path(pool_root) / NUCLEI_DIR / f"{file_uid}.npz"


def _finite(value) -> float | None:
    """NULL rather than NaN in a promoted column, so `WHERE x > 0` behaves predictably."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


class FragmentError(RuntimeError):
    """A manifest fragment could not be read. Always names the offending file."""


# --------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write a file so that it is either wholly present or wholly absent.

    A batch task killed mid-write must not leave a truncated file that the merge step
    would then parse as valid. The sequence is: write to a process-unique temporary
    name, flush, ``fsync`` so the bytes are durable, then ``os.replace`` — which is
    atomic on both POSIX and Windows (plain ``os.rename`` fails on Windows when the
    target exists).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


# --------------------------------------------------------------------------
# Phase A: writing
# --------------------------------------------------------------------------

class PoolWriter:
    """Accumulates the cells found in one source image, then writes one fragment.

    Usable as a context manager::

        with PoolWriter(root, source_path=..., voxel_size_um=..., ...) as w:
            w.add_cell(mask=mask, bbox=bbox, ...)

    Re-running the same source file overwrites its own output rather than duplicating
    it, because identifiers are derived deterministically from the source path.
    """

    def __init__(
        self,
        pool_root: str | os.PathLike[str],
        *,
        source_path: str,
        voxel_size_um: Sequence[float],
        image_shape: Sequence[int],
        channel_roles: Mapping[str, int],
        run_id: str,
        params_hash: str | None = None,
    ) -> None:
        self.root = Path(pool_root)
        self.file_record = FileRecord.create(
            source_path=source_path,
            voxel_size_um=voxel_size_um,
            image_shape=image_shape,
            channel_roles=channel_roles,
            run_id=run_id,
            params_hash=params_hash,
        )
        self._cells: list[CellRecord] = []
        self._closed = False
        self.cells_dir = self.root / CELLS_DIR / self.file_record.file_uid
        self.fragment_path = self.root / FRAGMENTS_DIR / f"{self.file_record.file_uid}.json"

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "PoolWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()

    # -- writing -----------------------------------------------------------

    def add_cell(
        self,
        *,
        mask: np.ndarray,
        bbox: Sequence[int],
        nucleus_mask: np.ndarray | None = None,
        thumbnail: np.ndarray | None = None,
        nucleus_ids: Sequence[int] = (),
        nucleus_centroids: Sequence[Sequence[float]] = (),
        split_method: SplitMethod | str = SplitMethod.UNSPLIT,
        soma_centroid: Sequence[float] | None = None,
        soma_remnant_found: bool = False,
        touches_xy_border: bool = False,
        touches_z_border: bool = False,
        n_voxels_segmentation: int | None = None,
        human_edited: bool = False,
        human_split: bool = False,
    ) -> str:
        """Store one cell's arrays and queue its record. Returns the cell uid.

        ``mask`` is the *cropped* mask and ``bbox`` its half-open position in the full
        image, ``(z0, z1, y0, y1, x0, x1)``. Their consistency is checked here rather
        than trusted: a mask whose shape disagrees with its bounding box would place
        the cell wrongly in every later reconstruction, including the Imaris export.

        Quantities that can be derived (``n_voxels``, ``bbox_diag_um``) are computed
        here rather than accepted from the caller, so the record's invariants cannot
        be broken by a mistake at the call site.
        """
        if self._closed:
            raise RuntimeError("PoolWriter is closed; create a new one to write more cells")

        bbox_t = tuple(int(v) for v in bbox)
        if len(bbox_t) != 6:
            raise ValueError(f"bbox must have 6 elements, got {len(bbox_t)}")
        z0, z1, y0, y1, x0, x1 = bbox_t
        expected = (z1 - z0, y1 - y0, x1 - x0)

        mask = np.asarray(mask)
        if mask.ndim != 3:
            raise ValueError(f"mask must be 3-D (z, y, x), got shape {mask.shape}")
        if mask.shape != expected:
            raise ValueError(
                f"mask shape {mask.shape} does not match bbox extent {expected} "
                f"for bbox {bbox_t}"
            )
        mask_bool = mask.astype(bool, copy=False)
        n_voxels = int(mask_bool.sum())
        if n_voxels == 0:
            raise ValueError("mask is empty; refusing to store a cell with no voxels")

        if nucleus_mask is not None:
            nucleus_mask = np.asarray(nucleus_mask)
            if nucleus_mask.shape != expected:
                raise ValueError(
                    f"nucleus_mask shape {nucleus_mask.shape} does not match "
                    f"bbox extent {expected}"
                )

        vz, vy, vx = self.file_record.voxel_size_um
        diag_um = float(
            np.sqrt(((z1 - z0) * vz) ** 2 + ((y1 - y0) * vy) ** 2 + ((x1 - x0) * vx) ** 2)
        )

        index = len(self._cells)
        cell_uid = make_cell_uid(self.file_record.file_uid, index)

        arrays: dict[str, np.ndarray] = {"mask": mask_bool}
        if nucleus_mask is not None:
            arrays["nucleus_mask"] = nucleus_mask.astype(np.uint16, copy=False)
        if thumbnail is not None:
            arrays["thumbnail"] = np.asarray(thumbnail).astype(np.uint8, copy=False)

        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        _atomic_write_bytes(self.cells_dir / f"{cell_uid}.npz", buf.getvalue())

        self._cells.append(
            CellRecord(
                cell_uid=cell_uid,
                file_uid=self.file_record.file_uid,
                cell_index_in_file=index,
                bbox=bbox_t,                      # type: ignore[arg-type]
                n_voxels=n_voxels,
                bbox_diag_um=diag_um,
                nucleus_count=len(tuple(nucleus_ids)),
                nucleus_ids=tuple(int(i) for i in nucleus_ids),
                nucleus_centroids=tuple(tuple(float(v) for v in c) for c in nucleus_centroids),
                split_method=SplitMethod(split_method),
                soma_centroid=tuple(float(v) for v in soma_centroid) if soma_centroid else None,
                soma_remnant_found=bool(soma_remnant_found),
                touches_xy_border=bool(touches_xy_border),
                touches_z_border=bool(touches_z_border),
                n_voxels_segmentation=(n_voxels if n_voxels_segmentation is None
                                       else int(n_voxels_segmentation)),
                human_edited=bool(human_edited),
                human_split=bool(human_split),
            )
        )
        return cell_uid

    def write_nucleus_labels(self, labels: np.ndarray, *, downsample: int = 1) -> Path:
        """Store the whole image's surviving nucleus labels. **Once per image.**

        > ### ⇒ BATCH REVIEW CANNOT DRAW *OTHER* NUCLEI WITHOUT THIS
        >
        > Semi-auto holds `Derivation.nuclei_labels` — the whole-image label volume — in
        > memory, and rich review's `other nuclei` layer is cut straight out of it. The pool
        > stores only each cell's own nuclei, **clipped to that cell's mask** and cropped to
        > its box. Reconstructing the volume by pasting neighbours back together would give
        > the same nuclei with the same ids, **eroded at every cell boundary** — a picture
        > that is wrong in exactly the region the reviewer is judging.
        >
        > **Stored BEFORE upsampling, and that is the whole of why it is affordable.**
        > `pipeline` scales this array up by `downsample` in plane; at the shipped factor of 2
        > the full-resolution version is four times the size, and it is recomputable —
        > `upsample_xy` is nearest-neighbour and exact. On a 40 x 2304 x 2304 field that is the
        > difference between roughly 200 MB and 850 MB of int32 per image before compression,
        > for an array that is mostly zeros and so compresses hard.

        ``labels`` is **surviving** nuclei — membership has already dropped the ones no cell
        claimed — so nothing is added here that the reviewer would not otherwise see, and it
        is exactly the population semi-auto's editor receives.
        """
        labels = np.asarray(labels)
        if labels.ndim != 3:
            raise ValueError(f"nucleus labels must be 3-D (z, y, x), got {labels.shape}")
        buf = io.BytesIO()
        np.savez_compressed(buf, labels=labels.astype(np.int32, copy=False),
                            downsample=np.int32(int(downsample)))
        path = nucleus_labels_path(self.root, self.file_record.file_uid)
        _atomic_write_bytes(path, buf.getvalue())
        return path

    def close(self) -> Path:
        """Write the manifest fragment atomically. Returns its path."""
        if self._closed:
            return self.fragment_path

        file_rec = FileRecord(
            **{**self.file_record.to_json_dict(), "n_cells": len(self._cells)}
        )
        payload = {
            "fragment_format": FRAGMENT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "file": file_rec.to_json_dict(),
            "cells": [c.to_json_dict() for c in self._cells],
        }
        _atomic_write_text(self.fragment_path, json.dumps(payload, indent=1))
        self._closed = True
        return self.fragment_path


def measurement_fragment_path(pool_root: str | os.PathLike[str], params_hash: str,
                              file_uid: str) -> Path:
    """`pool/measurement_fragments/<params_hash>/<file_uid>.json`."""
    return Path(pool_root) / MEASUREMENT_FRAGMENTS_DIR / params_hash / f"{file_uid}.json"


class MeasurementFragment:
    """One image's Phase C output, accumulated then written as one file. **Never the DB.**

    > ### ⇒ THIS IS WHAT MAKES A MEASURE ARRAY POSSIBLE AT ALL
    >
    > `_connect` states the constraint outright: *"Access is strictly single-writer."* No WAL —
    > deliberate, the sidecars break the archive copy — and no `busy_timeout`, so the default
    > is 0. Phase A avoids the collision with fragments; **Phase C had none** and inserted
    > straight into `pool.db`, so hundreds of concurrent tasks contended for the lock and each
    > one died *after* doing all the expensive work.
    >
    > The interface is deliberately the two method names :class:`Pool` already carries —
    > `write_measurement` and `write_image_measurement` — so `measure_file` writes to a
    > fragment or to the database by being handed a different object, and there is **no second
    > code path** through the measuring itself to drift.

    **Skeletons are not in here.** They already go to per-cell `.npz` paths keyed by
    `params_hash` and touch no database, so they were never part of this problem.
    """

    def __init__(self, pool_root: str | os.PathLike[str], *, file_uid: str,
                 params_hash: str) -> None:
        self.root = Path(pool_root)
        self.file_uid = file_uid
        self.params_hash = params_hash
        self._cells: list[dict[str, Any]] = []
        self._image: dict[str, Any] | None = None
        self._closed = False
        self.path = measurement_fragment_path(self.root, params_hash, file_uid)

    def __enter__(self) -> "MeasurementFragment":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()

    def write_measurement(
        self,
        cell_uid: str,
        params_hash: str,
        *,
        payload: Mapping[str, Any],
        status: str = "ok",
        reviewed: bool = True,
        volume_um3: float | None = None,
        n_endpoints: int | None = None,
        ramification_index: float | None = None,
        cd68_integrated: float | None = None,
    ) -> None:
        """Queue one cell's row. Signature-identical to :meth:`Pool.write_measurement`."""
        if params_hash != self.params_hash:
            raise ValueError(
                f"fragment is keyed {self.params_hash} and was handed {params_hash}; "
                "the hash is in the path, so mixing them would file a measurement under "
                "parameters that did not produce it")
        self._cells.append({
            "cell_uid": cell_uid,
            "status": status,
            "reviewed": bool(reviewed),
            "volume_um3": _finite(volume_um3),
            "n_endpoints": None if n_endpoints is None else int(n_endpoints),
            "ramification_index": _finite(ramification_index),
            "cd68_integrated": _finite(cd68_integrated),
            "payload": dict(payload),
        })

    def write_image_measurement(self, file_uid: str, params_hash: str, *,
                                payload: Mapping[str, Any],
                                reviewed: bool = True) -> None:
        """Queue the image-level row. Signature-identical to `Pool`\'s."""
        if file_uid != self.file_uid:
            raise ValueError(f"fragment holds {self.file_uid} and was handed {file_uid}")
        if params_hash != self.params_hash:
            raise ValueError(f"fragment is keyed {self.params_hash}, got {params_hash}")
        self._image = {"payload": dict(payload), "reviewed": bool(reviewed)}

    def close(self) -> Path:
        """Write the fragment atomically. Returns its path."""
        if self._closed:
            return self.path
        payload = {
            "fragment_format": FRAGMENT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "params_hash": self.params_hash,
            "file_uid": self.file_uid,
            "written_at": utc_now(),
            "cells": self._cells,
            "image": self._image,
        }
        #: `allow_nan=True` matches `Pool.write_measurement`, which stores the payload the
        #: same way. A measure that could not be computed is `NaN` **with a status**, never a
        #: plausible zero, and that distinction has to survive the round trip.
        _atomic_write_text(self.path, json.dumps(payload, sort_keys=True, allow_nan=True))
        self._closed = True
        return self.path


@dataclass
class MeasurementMergeResult:
    fragments_read: int = 0
    cells_written: int = 0
    images_written: int = 0
    fragments_deleted: int = 0


def _write_pool_csv(root: Path) -> None:
    """`pool.csv`, after either merge. **Never fatal** — patch item C3, 2026-09-03.

    The merge itself has already committed by the time this runs, and a spreadsheet convenience
    must not be able to fail a command that has finished its real work: a locked file on a
    network share is the ordinary way this goes wrong, and losing the merge's result to it would
    be a much worse trade than an absent CSV.
    """
    #: **Never opens a pool that does not exist.** `Pool` connects, and connecting creates the
    #: file — so calling this on a directory that holds no database would leave an empty one
    #: behind, which is a worse outcome than the missing CSV it is trying to fix.
    if not (root / DB_NAME).exists():
        return
    try:
        with Pool(root) as pool:
            pool.export_csv()
    except Exception:                           # noqa: BLE001 - see the docstring
        pass


def merge_measurement_fragments(
    pool_root: str | os.PathLike[str],
    *,
    params_hash: str | None = None,
    delete: bool = True,
) -> MeasurementMergeResult:
    """Fold Phase C fragments into the database, then delete the ones that were merged.

    **Its own command, not part of `merge`** — the user\'s decision. `merge` runs after
    `segment` and before `review`; this runs after `measure` and before `export`. One command
    doing both would re-parse and re-insert all 573 segmentation fragments every time
    measurements are folded in.

    ``params_hash`` narrows to one parameter set; omitted, every set present is merged, which
    is what a pool holding a re-run beside the run it replaced needs.

    **Every fragment is parsed before any is written**, so a corrupt one aborts cleanly
    naming the file rather than half-importing — exactly as :func:`merge_fragments` does.

    > ### ⇒ DELETE ONLY AFTER THE TRANSACTION COMMITS
    >
    > Commit first, then delete. If the merge dies partway the fragments are still on disk and
    > re-merging is harmless, because these inserts replace rather than duplicate.
    > **Delete-before-commit is the one ordering that can lose a run.**
    >
    > Deleting these is safe in a way deleting *segmentation* fragments would not be: batch\'s
    > resume signal is fragment existence (`batch.fragment_exists`), whereas measure\'s queue
    > is the `measurements` table. `COMPONENT_05_PLAN` §9.4d flags that distinction — **the
    > deletion is not extended to Phase A.**
    >
    > A deletion that fails is benign for the same reason: the fragment simply merges again.
    """
    root = Path(pool_root)
    base = root / MEASUREMENT_FRAGMENTS_DIR
    result = MeasurementMergeResult()
    #: > **`pool.csv` is rewritten on EVERY exit from here, including these two** — item C3.
    #: > The command is run after reviewing, and *"nothing new to merge"* is exactly the case
    #: > where the file on disk is a Phase A snapshot and the statuses in it have all moved.
    if not base.exists():
        _write_pool_csv(root)
        return result

    folders = ([base / params_hash] if params_hash is not None
               else sorted(p for p in base.iterdir() if p.is_dir()))
    fragments = [f for folder in folders if folder.exists()
                 for f in sorted(folder.glob("*.json")) if ".tmp." not in f.name]
    if not fragments:
        _write_pool_csv(root)
        return result

    # Parse everything first. A corrupt fragment must abort without having half-imported.
    cell_rows: list[tuple] = []
    image_rows: list[tuple] = []
    for frag in fragments:
        try:
            payload = json.loads(frag.read_text(encoding="utf-8"))
            frag_hash = payload["params_hash"]
            file_uid = payload["file_uid"]
            now = utc_now()
            for cell in payload["cells"]:
                cell_rows.append((
                    cell["cell_uid"], frag_hash, now, cell["status"],
                    int(bool(cell["reviewed"])), cell.get("volume_um3"),
                    cell.get("n_endpoints"), cell.get("ramification_index"),
                    cell.get("cd68_integrated"),
                    json.dumps(cell["payload"], sort_keys=True, allow_nan=True)))
            image = payload.get("image")
            if image is not None:
                image_rows.append((
                    file_uid, frag_hash, now, int(bool(image["reviewed"])),
                    json.dumps(image["payload"], sort_keys=True, allow_nan=True)))
        except Exception as exc:
            raise FragmentError(
                f"could not read measurement fragment {frag}: {exc}") from exc
        result.fragments_read += 1

    conn = _connect(root / DB_NAME)
    try:
        with conn:                      # one transaction; rolls back entirely on error
            conn.executemany(
                "INSERT OR REPLACE INTO measurements "
                "(cell_uid, params_hash, measured_at, status, reviewed, volume_um3, "
                " n_endpoints, ramification_index, cd68_integrated, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", cell_rows)
            conn.executemany(
                "INSERT OR REPLACE INTO image_measurements "
                "(file_uid, params_hash, measured_at, reviewed, payload) "
                "VALUES (?, ?, ?, ?, ?)", image_rows)
        result.cells_written = len(cell_rows)
        result.images_written = len(image_rows)
    finally:
        conn.close()

    # ---- committed. Only now is it safe to remove the inputs. --------------
    if delete:
        for frag in fragments:
            try:
                frag.unlink()
                result.fragments_deleted += 1
            except OSError:             # benign: it simply merges again next time
                pass
        for folder in folders:
            try:
                if folder.exists() and not any(folder.iterdir()):
                    folder.rmdir()
            except OSError:
                pass
        #: > ### ⇒ AND THE PARENT, WHICH IS THE ONE THAT WAS SURVIVING — patch item C4
        #: >
        #: > The per-hash folders above have been removed since they were written. **What stayed
        #: > was `measurement_fragments/` itself** — checked rather than read off the code:
        #: > after a merge the tree holds `measurement_fragments/` and nothing inside it, and it
        #: > is the only empty directory anywhere in the output.
        #: >
        #: > Removed **only when it is empty**, so a pool holding a second parameter set that
        #: > has not been merged yet keeps its folder and its fragments.
        try:
            if base.exists() and not any(base.iterdir()):
                base.rmdir()
        except OSError:                         # benign: it is empty scaffolding either way
            pass

    #: **Rewritten, so review is reflected** — patch item C3. `merge` wrote it before anyone had
    #: reviewed anything; by the time measurements are folded in, every status in it has moved.
    _write_pool_csv(Path(pool_root))
    return result


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS files (
    file_uid        TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL UNIQUE,
    source_filename TEXT NOT NULL,
    source_stem     TEXT NOT NULL,
    sort_key        TEXT NOT NULL,
    voxel_z REAL NOT NULL, voxel_y REAL NOT NULL, voxel_x REAL NOT NULL,
    shape_z INTEGER NOT NULL, shape_y INTEGER NOT NULL, shape_x INTEGER NOT NULL,
    channel_roles   TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    params_hash     TEXT,
    created_at      TEXT NOT NULL,
    review_complete INTEGER NOT NULL DEFAULT 0,
    n_cells         INTEGER NOT NULL DEFAULT 0,
    schema_version  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS files_sort ON files(sort_key);

CREATE TABLE IF NOT EXISTS cells (
    cell_uid           TEXT PRIMARY KEY,
    file_uid           TEXT NOT NULL REFERENCES files(file_uid),
    cell_index_in_file INTEGER NOT NULL,
    z0 INTEGER NOT NULL, z1 INTEGER NOT NULL,
    y0 INTEGER NOT NULL, y1 INTEGER NOT NULL,
    x0 INTEGER NOT NULL, x1 INTEGER NOT NULL,
    n_voxels           INTEGER NOT NULL,
    bbox_diag_um       REAL NOT NULL,
    nucleus_count      INTEGER NOT NULL,
    nucleus_ids        TEXT NOT NULL,
    nucleus_centroids  TEXT NOT NULL,
    split_method       TEXT NOT NULL,
    soma_z REAL, soma_y REAL, soma_x REAL,
    soma_remnant_found INTEGER NOT NULL,
    touches_xy_border  INTEGER NOT NULL,
    touches_z_border   INTEGER NOT NULL DEFAULT 0,
    n_voxels_segmentation INTEGER NOT NULL DEFAULT 0,
    human_edited       INTEGER NOT NULL DEFAULT 0,
    human_split        INTEGER NOT NULL DEFAULT 0,
    status             TEXT NOT NULL,
    -- Structured since 2026-08-11: one of `RejectReason` — exactly `debris` or `real_cell` —
    -- and never free text. `reject_note` keeps the reviewer's own words, offered on the
    -- `real_cell` branch, so making the reason countable costs no detail. An unrecognised
    -- value is refused on read rather than coerced: there is no free-text history to migrate,
    -- and a guess would move a cell between the two categories the pair exists to separate.
    reject_reason      TEXT,
    reject_note        TEXT,
    -- How many real cells this row is worth. 1 ordinarily, the reviewer's own number on a
    -- `real_cell` rejection, and **NULL on debris** — deliberately nullable, and the only
    -- nullable count on the table. Debris is not a cell, so it is worth no cells rather than
    -- one: a stored 1 on a debris row is a number that means nothing and that a later SUM
    -- would add up. NULL makes the column summable as it stands. `set_status` derives the
    -- NULL from the reason rather than trusting a call site to pass it.
    n_real_cells_here  INTEGER DEFAULT 1,
    created_at         TEXT NOT NULL,
    reviewed_at        TEXT,
    schema_version     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS cells_review ON cells(file_uid, status, cell_index_in_file);
CREATE INDEX IF NOT EXISTS cells_status ON cells(status);

-- Phase C output. A separate table rather than columns on `cells`, for two reasons that are
-- properties of the measurements rather than preferences (DESIGN_RATIONALE.md §4.10.1):
-- curves have no fixed width — CD68 across five thresholds, plaque burden across four
-- coefficients, contact area across three tolerances, several of which will never have a
-- chosen value by design — and the measure set is still growing, so columns would pay a
-- migration each time.
--
-- `params_hash` in the PRIMARY KEY is the load-bearing part. Re-measuring after a parameter
-- change writes a NEW ROW rather than overwriting, so a cell measured under old settings can
-- never be silently mistaken for one measured under new. That is also what lets the export
-- show both side by side instead of having to refuse or drop one.
CREATE TABLE IF NOT EXISTS measurements (
    cell_uid     TEXT NOT NULL REFERENCES cells(cell_uid),
    params_hash  TEXT NOT NULL,
    measured_at  TEXT NOT NULL,
    status       TEXT NOT NULL,
    reviewed     INTEGER NOT NULL DEFAULT 1,
    -- Promoted measures: real columns so analysis can filter and sort without parsing every
    -- payload. Chosen 2026-08-10 for the analyst rather than the reviewer -- review happens
    -- before Phase C and sorts on QC fields that already live on `cells`. All four are
    -- threshold-free, so they stay meaningful across parameter sets. Plaque distance was
    -- considered and deferred: it is now a curve over size floors, so promoting it would mean
    -- silently committing to one floor.
    volume_um3        REAL,
    n_endpoints       INTEGER,
    ramification_index REAL,
    cd68_integrated   REAL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (cell_uid, params_hash)
);
CREATE INDEX IF NOT EXISTS measurements_hash ON measurements(params_hash);

-- Per-*image* Phase C output: plaque burden, the diffuse/dense-core split, background
-- levels. Separate from `measurements` because these are properties of the field, not of a
-- cell — storing them on each cell would repeat one number across every row of an image and
-- invite it being averaged, which would weight images by how many cells they happen to hold.
CREATE TABLE IF NOT EXISTS image_measurements (
    file_uid     TEXT NOT NULL REFERENCES files(file_uid),
    params_hash  TEXT NOT NULL,
    measured_at  TEXT NOT NULL,
    reviewed     INTEGER NOT NULL DEFAULT 1,
    payload      TEXT NOT NULL,
    PRIMARY KEY (file_uid, params_hash)
);
"""


def _connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the database with settings chosen for this workflow.

    Write-ahead logging is deliberately *not* enabled. WAL is the usual default
    advice, but it creates ``-wal`` and ``-shm`` sidecar files, and this pool gets
    synced between the compute cluster and a local machine — a partial sync that
    misses the sidecars yields a stale or corrupt database. Access is strictly
    single-writer, so the rollback journal with ``synchronous=FULL`` is both safer
    and simpler at this scale.

    > ### ⇒ `read_only=True` IS WHAT MAKES A MEASURE ARRAY POSSIBLE — 2026-09-02
    >
    > *"Access is strictly single-writer"* is the sentence above, and the default path
    > **contradicts it on every open**: `executescript(_DDL)` and `_add_missing_columns` are
    > writes, so merely opening the pool to read a cell's mask made a task a writer. With no
    > WAL and no `busy_timeout` — the default is 0 — hundreds of concurrent measure tasks
    > collide on the lock, and each one dies *after* doing all the expensive work.
    >
    > The read-only path opens the file `mode=ro` through a URI and **runs no DDL at all**, so
    > a reader cannot take a write lock even by accident. The additive-column migration is
    > skipped with it, which is correct rather than merely tolerable: `merge` runs before
    > `measure` in every chain and it opens read-write, so the schema is already current by
    > the time any array task looks at it.
    """
    if read_only:
        if not db_path.exists():
            raise FileNotFoundError(
                f"no pool database at {db_path}. A read-only open never creates one — "
                "run `prestomorph merge` first.")
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = FULL")
    conn.executescript(_DDL)
    _add_missing_columns(conn)
    return conn


#: Columns added to an existing table after it was first created. ``CREATE TABLE IF NOT
#: EXISTS`` does nothing to a table that already exists, so a pool written before a column
#: was added would be missing it and every insert would fail.
#:
#: **Additive only, and deliberately so.** Nothing here drops, renames or rewrites a value:
#: this pool is the only copy of hours of human review, and a migration that edits data is a
#: migration that can destroy it. A schema change that cannot be expressed as "add a nullable
#: column" should refuse the pool the way `params.py` refuses a v2 parameters file, not
#: quietly reshape it.
_ADDED_COLUMNS = {
    "cells": {"reject_note": "TEXT"},
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def _file_to_row(r: FileRecord) -> dict[str, Any]:
    return {
        "file_uid": r.file_uid,
        "source_path": r.source_path,
        "source_filename": r.source_filename,
        "source_stem": r.source_stem,
        "sort_key": r.sort_key,
        "voxel_z": r.voxel_size_um[0], "voxel_y": r.voxel_size_um[1], "voxel_x": r.voxel_size_um[2],
        "shape_z": r.image_shape[0], "shape_y": r.image_shape[1], "shape_x": r.image_shape[2],
        "channel_roles": json.dumps(r.channel_roles),
        "run_id": r.run_id,
        "params_hash": r.params_hash,
        "created_at": r.created_at,
        "review_complete": int(r.review_complete),
        "n_cells": r.n_cells,
        "schema_version": r.schema_version,
    }


def _row_to_file(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        file_uid=row["file_uid"],
        source_path=row["source_path"],
        source_filename=row["source_filename"],
        source_stem=row["source_stem"],
        sort_key=row["sort_key"],
        voxel_size_um=(row["voxel_z"], row["voxel_y"], row["voxel_x"]),
        image_shape=(row["shape_z"], row["shape_y"], row["shape_x"]),
        channel_roles=json.loads(row["channel_roles"]),
        run_id=row["run_id"],
        params_hash=row["params_hash"],
        created_at=row["created_at"],
        review_complete=bool(row["review_complete"]),
        n_cells=row["n_cells"],
        schema_version=row["schema_version"],
    )


def _column(row: sqlite3.Row, name: str, default=None):
    """Read a column that a given query may not have selected."""
    return row[name] if name in row.keys() else default


def _cell_to_row(r: CellRecord) -> dict[str, Any]:
    z0, z1, y0, y1, x0, x1 = r.bbox
    soma = r.soma_centroid or (None, None, None)
    return {
        "cell_uid": r.cell_uid,
        "file_uid": r.file_uid,
        "cell_index_in_file": r.cell_index_in_file,
        "z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
        "n_voxels": r.n_voxels,
        "bbox_diag_um": r.bbox_diag_um,
        "nucleus_count": r.nucleus_count,
        "nucleus_ids": json.dumps(list(r.nucleus_ids)),
        "nucleus_centroids": json.dumps([list(c) for c in r.nucleus_centroids]),
        "split_method": r.split_method.value,
        "soma_z": soma[0], "soma_y": soma[1], "soma_x": soma[2],
        "soma_remnant_found": int(r.soma_remnant_found),
        "touches_xy_border": int(r.touches_xy_border),
        "touches_z_border": int(r.touches_z_border),
        "n_voxels_segmentation": r.n_voxels_segmentation,
        "human_edited": int(r.human_edited),
        "human_split": int(r.human_split),
        "status": r.status.value,
        "reject_reason": None if r.reject_reason is None else RejectReason(
            r.reject_reason).value,
        "reject_note": r.reject_note,
        "n_real_cells_here": r.n_real_cells_here,
        "created_at": r.created_at,
        "reviewed_at": r.reviewed_at,
        "schema_version": r.schema_version,
    }


def _row_to_cell(row: sqlite3.Row) -> CellRecord:
    soma = (row["soma_z"], row["soma_y"], row["soma_x"])
    return CellRecord(
        cell_uid=row["cell_uid"],
        file_uid=row["file_uid"],
        cell_index_in_file=row["cell_index_in_file"],
        bbox=(row["z0"], row["z1"], row["y0"], row["y1"], row["x0"], row["x1"]),
        n_voxels=row["n_voxels"],
        bbox_diag_um=row["bbox_diag_um"],
        nucleus_count=row["nucleus_count"],
        nucleus_ids=tuple(json.loads(row["nucleus_ids"])),
        nucleus_centroids=tuple(tuple(c) for c in json.loads(row["nucleus_centroids"])),
        split_method=SplitMethod(row["split_method"]),
        soma_centroid=None if soma[0] is None else soma,
        soma_remnant_found=bool(row["soma_remnant_found"]),
        touches_xy_border=bool(row["touches_xy_border"]),
        touches_z_border=bool(row["touches_z_border"]),
        n_voxels_segmentation=row["n_voxels_segmentation"],
        human_edited=bool(row["human_edited"]),
        human_split=bool(row["human_split"]),
        # Not coerced here: `CellRecord.__post_init__` does it, and refuses a value that is
        # not one of the two reasons rather than guessing which one was meant.
        status=row["status"],
        reject_reason=row["reject_reason"],
        #: **`_column` with a default of 1**, like `reject_note` above, so a query that did not
        #: select this column yields an ordinary cell rather than raising. The default is 1 and
        #: not `None`: an unselected column means *not asked for*, and the row a partial query
        #: describes is a cell until something says otherwise.
        n_real_cells_here=_column(row, "n_real_cells_here", 1),
        reject_note=_column(row, "reject_note"),
        created_at=row["created_at"],
        reviewed_at=row["reviewed_at"],
        schema_version=row["schema_version"],
    )


def _upsert_many(conn: sqlite3.Connection, table: str, rows: Sequence[Mapping[str, Any]],
                 key: str, preserve: Sequence[str]) -> None:
    """Insert or update many rows in one statement, leaving ``preserve`` columns alone.

    Batched rather than row-at-a-time: at ~20,000 cells the per-row round trip
    dominated merge time (30 s measured, versus well under a second batched).
    """
    if not rows:
        return
    cols = list(rows[0])
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in preserve and c != key)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({key}) DO UPDATE SET {updates}",
        [dict(r) for r in rows],
    )


@dataclass
class MergeResult:
    files_inserted: int = 0
    files_updated: int = 0
    cells_inserted: int = 0
    cells_updated: int = 0
    reviews_preserved: int = 0
    fragments_read: int = 0


def merge_fragments(
    pool_root: str | os.PathLike[str],
    *,
    reset_review: bool = False,
) -> MergeResult:
    """Fold all manifest fragments into the pool database.

    Re-merging a fragment **preserves review decisions by default**. Segmentation
    fields are updated, but ``status``, ``reject_reason``, ``reject_note`` and
    ``reviewed_at`` are left alone — otherwise re-running one file's segmentation would
    silently reset every accept/reject already made for it, destroying human work that cannot
    be recovered. Pass ``reset_review=True`` to deliberately discard those decisions.

    Temporary files left by an interrupted writer are ignored; a fragment that cannot
    be parsed raises :class:`FragmentError` naming the file, rather than being skipped
    silently.
    """
    root = Path(pool_root)
    frag_dir = root / FRAGMENTS_DIR
    result = MergeResult()

    fragments = sorted(p for p in frag_dir.glob("*.json") if ".tmp." not in p.name) \
        if frag_dir.exists() else []

    conn = _connect(root / DB_NAME)
    #: **Everything a reviewer decided**, held across a re-segmentation. `n_real_cells_here`
    #: joined on 2026-08-30 with the column itself: it is part of the rejection, exactly as
    #: `reject_note` is, and without it re-running segmentation would quietly turn a clump the
    #: reviewer counted as 3 back into a 1. **Found by the test rather than by reading this
    #: line** — `test_a_remerge_preserves_the_reviewers_count` was written expecting it to pass.
    preserve_cell = () if reset_review else ("status", "reject_reason", "reject_note",
                                             "n_real_cells_here", "reviewed_at")
    preserve_file = () if reset_review else ("review_complete",)
    try:
        # Read existing state once rather than per row: at 20,000 cells the per-row
        # lookup was the whole cost of merging.
        prior_status = {
            r["cell_uid"]: r["status"]
            for r in conn.execute("SELECT cell_uid, status FROM cells")
        }
        prior_files = {r["file_uid"] for r in conn.execute("SELECT file_uid FROM files")}

        # Parse everything before writing anything, so a corrupt fragment aborts the
        # batch without having half-imported it.
        file_rows: list[dict[str, Any]] = []
        cell_rows: list[dict[str, Any]] = []
        for frag in fragments:
            try:
                payload = json.loads(frag.read_text(encoding="utf-8"))
                file_rec = FileRecord.from_json_dict(payload["file"])
                cell_recs = [CellRecord.from_json_dict(c) for c in payload["cells"]]
            except Exception as exc:
                raise FragmentError(f"could not read fragment {frag}: {exc}") from exc

            result.fragments_read += 1
            file_rows.append(_file_to_row(file_rec))
            if file_rec.file_uid in prior_files:
                result.files_updated += 1
            else:
                result.files_inserted += 1

            for cell in cell_recs:
                cell_rows.append(_cell_to_row(cell))
                seen = prior_status.get(cell.cell_uid)
                if seen is None:
                    result.cells_inserted += 1
                else:
                    result.cells_updated += 1
                    if not reset_review and seen != CellStatus.CANDIDATE.value:
                        result.reviews_preserved += 1

        with conn:  # single transaction; rolls back entirely on error
            _upsert_many(conn, "files", file_rows, "file_uid", preserve_file)
            _upsert_many(conn, "cells", cell_rows, "cell_uid", preserve_cell)
    finally:
        conn.close()

    #: > ### ⇒ `pool.csv` WAS PROMISED AND NEVER WRITTEN — patch item C3, 2026-09-03
    #: >
    #: > This module's own layout docstring lists it — *"flat export for spreadsheets"* — and
    #: > `Pool.export_csv` has existed all along with **zero callers**. The file simply never
    #: > appeared in anybody's output folder.
    #: >
    #: > **Written here and rewritten by `merge_measurement_fragments`**, so it is not a Phase A
    #: > snapshot that goes stale the moment anyone reviews a cell: the status, the reject reason
    #: > and `reviewed_at` are columns in it. The measured *values* stay `export`'s job — that is
    #: > what `cells.csv` is, keyed by parameter set, and duplicating them here would create a
    #: > second answer to the same question.
    _write_pool_csv(root)
    return result


# --------------------------------------------------------------------------
# Phase B / C: reading and updating
# --------------------------------------------------------------------------

@dataclass
class IntegrityReport:
    """Result of :meth:`Pool.check_integrity`."""

    orphan_arrays: list[str] = field(default_factory=list)
    missing_arrays: list[str] = field(default_factory=list)
    miscounted_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.orphan_arrays or self.missing_arrays or self.miscounted_files)


class Pool:
    """Query and update a merged pool. Single process; not for concurrent writers."""

    def __init__(self, pool_root: str | os.PathLike[str], *,
                 read_only: bool = False) -> None:
        self.root = Path(pool_root)
        self.read_only = bool(read_only)
        self.conn = _connect(self.root / DB_NAME, read_only=self.read_only)

    def _refuse_if_read_only(self, what: str) -> None:
        """Say which write was attempted, rather than letting sqlite say *a* write was.

        `attempt to write a readonly database` arrives with no indication of which of a
        dozen methods produced it, and the read-only path exists precisely so that hundreds
        of concurrent tasks never take a write lock — a call that got this far is a routing
        mistake worth naming.
        """
        if self.read_only:
            raise PermissionError(
                f"{what} on a read-only pool. Measure tasks open the pool read-only and "
                "write fragments; the database is written once, by a merge step.")

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # -- files -------------------------------------------------------------

    def files(self, *, include_complete: bool = True) -> list[FileRecord]:
        """All source files in natural path order.

        Ordered by a natural-sort key over the *full path*, so files group by
        directory (genotype, region) instead of interleaving, and ``img2`` precedes
        ``img10``.
        """
        sql = "SELECT * FROM files"
        if not include_complete:
            sql += " WHERE review_complete = 0"
        sql += " ORDER BY sort_key"
        return [_row_to_file(r) for r in self.conn.execute(sql)]

    def file(self, file_uid: str) -> FileRecord:
        row = self.conn.execute(
            "SELECT * FROM files WHERE file_uid = ?", (file_uid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such file in pool: {file_uid}")
        return _row_to_file(row)

    def mark_file_reviewed(self, file_uid: str, complete: bool = True) -> None:
        self._refuse_if_read_only("mark_file_reviewed")
        with self.conn:
            self.conn.execute(
                "UPDATE files SET review_complete = ? WHERE file_uid = ?",
                (int(complete), file_uid),
            )

    # -- cells -------------------------------------------------------------

    def cells(
        self,
        *,
        file_uid: str | None = None,
        status: CellStatus | str | None = None,
    ) -> list[CellRecord]:
        sql = "SELECT * FROM cells"
        clauses, params = [], []
        if file_uid is not None:
            clauses.append("file_uid = ?")
            params.append(file_uid)
        if status is not None:
            clauses.append("status = ?")
            params.append(CellStatus(status).value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY file_uid, cell_index_in_file"
        return [_row_to_cell(r) for r in self.conn.execute(sql, params)]

    def cells_for_review(self, file_uid: str) -> list[CellRecord]:
        """Unreviewed cells of one file, least confident first.

        Cells with no agreement score come first: a missing score means no nucleus
        was matched, which is precisely the case most in need of human eyes. SQLite
        would order NULLs first here anyway, but the intent is stated explicitly
        rather than left resting on a default.
        """
        rows = self.conn.execute(
            """
            SELECT * FROM cells
             WHERE file_uid = ? AND status = ?
             ORDER BY cell_index_in_file ASC
            """,
            (file_uid, CellStatus.CANDIDATE.value),
        )
        return [_row_to_cell(r) for r in rows]

    def set_status(
        self,
        cell_uid: str,
        status: CellStatus | str,
        reject_reason: RejectReason | str | None = None,
        reject_note: str | None = None,
        n_real_cells_here: int = 1,
    ) -> None:
        """Record one review decision.

        ``reject_reason`` is a :class:`RejectReason` — ``debris`` or ``real_cell`` — and an
        unrecognised value raises rather than being filed under a category nobody chose.
        ``reject_note`` is the reviewer's own words, offered on the ``real_cell`` branch.

        Status and reason are two independent facts written as they are given. Nothing
        reconciles them, because nothing needs to: the ``DEFERRED`` status that once encoded
        the same fact a second time is gone, and ``REJECTED`` + ``real_cell`` is now the only
        way to say a real cell was lost.

        > ### ⇒ ``n_real_cells_here`` IS THE ONE THING HERE THAT IS DERIVED — 2026-08-30
        >
        > *"This is 3 cells I cannot afford to split"* — §4.4's NEW-d, and until today it
        > reached no column at all. It is written as given **except on debris, where it is
        > forced to ``None``** whatever the caller passed. Debris is not a cell, so it is worth
        > no cells rather than one.
        >
        > **Derived rather than trusted**, on the same principle as ``add_cell`` computing
        > ``n_voxels``: one call site forgetting to pass ``None`` for a debris rejection would
        > put a 1 on a row that is not a cell, and nothing downstream could tell it apart from
        > a genuine single. The invariant is enforced in the one place that can enforce it.
        >
        > **This is the third fact about a rejection, and unlike the first two it is not
        > independent** — it is a function of the reason. That asymmetry is why it is computed
        > here and the other two are not.
        """
        self._refuse_if_read_only("set_status")
        status = CellStatus(status)
        reason = None if reject_reason is None else RejectReason(reject_reason)
        count = int(n_real_cells_here)
        if count < 1:
            raise ValueError(
                f"n_real_cells_here must be at least 1, got {n_real_cells_here}")
        stored_count = None if reason is RejectReason.DEBRIS else count
        with self.conn:
            cur = self.conn.execute(
                "UPDATE cells SET status = ?, reject_reason = ?, reject_note = ?, "
                "n_real_cells_here = ?, reviewed_at = ? WHERE cell_uid = ?",
                (status.value, None if reason is None else reason.value, reject_note,
                 stored_count, utc_now(), cell_uid),
            )
        if cur.rowcount == 0:
            raise KeyError(f"no such cell in pool: {cell_uid}")

    def replace_cell_mask(
        self,
        cell_uid: str,
        *,
        mask: np.ndarray,
        bbox: Sequence[int],
        nucleus_mask: np.ndarray | None = None,
        thumbnail: np.ndarray | None = None,
        soma_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Write a reviewer-edited mask back, and rebuild everything it invalidates.

        `COMPONENT_05_PLAN.md` §9.4b. **A transaction, not a file write**, and the one method
        of §9.4's storage foundation this build lands: cell splitting is out of scope, which
        is what removes `create_cell` and `delete_cell`.

        > ### ⇒ THE BOUNDING BOX CAN GROW, SO IT IS AN ARGUMENT AND NOT AN ASSUMPTION
        >
        > Rich review can add material outside the cell's original box — the editable frame is
        > the box grown by the margin's maximum — so `bbox` moves with the mask and
        > `bbox_diag_um` is recomputed from it. `add_cell` validates the same pair the same
        > way, and for the same reason: a mask disagreeing with its box misplaces the cell in
        > every later reconstruction.

        > ### ⇒ THE SOMA CENTROID IS REBUILT, AND `--params` EXISTS FOR THIS
        >
        > It is computed from the run's six soma settings and the pool stores only their hash,
        > so it cannot be rebuilt from the pool alone. Skipping the rebuild would leave a
        > stored centroid pointing at a mask that changed underneath it — **corrupting the
        > plaque-distance measures for exactly the cells a human worked hardest on.**
        >
        > `soma_remnant_found` comes off the **same** `find_soma` call rather than being left
        > alone: two facts from one computation cannot disagree, and a stale flag beside a
        > rebuilt centroid is the inconsistency this method exists to prevent.
        >
        > **It was three facts until 2026-09-03**, when `agreement_score` left `CellRecord`
        > with the rest of the nucleus-derived review ordering. The triage signals that
        > replaced it are recomputed at review time, so an edited mask needs nothing written
        > back here — the next ranking measures the mask as it now is.

        > ### ⇒ THE FILE WRITE COMES BEFORE THE ROW UPDATE
        >
        > There are two stores and no transaction spanning them. A crash between them leaves
        > new arrays with a stale `n_voxels` — **recoverable**, because `n_voxels` is
        > recomputable from the mask. The other order leaves a row claiming an edit that never
        > happened, which is not.

        > ### ⇒ AND IT DELETES THE WHOLE FILE'S MEASUREMENTS
        >
        > Not just this cell's — the user's decision of 2026-08-17. The per-image row carries
        > territory, overlap and plaque geometry, all derived from the cell masks, so editing
        > one cell makes the image's territory a figure computed from a mask that no longer
        > exists. **And forgetting it ships silently wrong:** Phase C's queue is *cells with no
        > measurement at this hash*, and an edit changes the cell **without changing any
        > hash** — so unless the rows go, the edited cell reports itself already measured and
        > the export ships the old numbers against the new mask.
        >
        > On a first pass this costs nothing: review runs before `measure`, so there are no
        > rows to delete.

        The thumbnail is **passed in, not computed**: `pipeline._thumbnail` is Workstream A's
        and private, and `add_cell` already takes one rather than making one. The UI computes
        it; storage stores it.
        """
        self._refuse_if_read_only("replace_cell_mask")
        row = self.conn.execute(
            "SELECT file_uid, nucleus_ids FROM cells WHERE cell_uid = ?", (cell_uid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such cell in pool: {cell_uid}")
        file_uid = row["file_uid"]
        record = self.file(file_uid)

        bbox_t = tuple(int(v) for v in bbox)
        if len(bbox_t) != 6:
            raise ValueError(f"bbox must have 6 elements, got {len(bbox_t)}")
        z0, z1, y0, y1, x0, x1 = bbox_t
        expected = (z1 - z0, y1 - y0, x1 - x0)
        mask = np.asarray(mask)
        if mask.ndim != 3:
            raise ValueError(f"mask must be 3-D (z, y, x), got shape {mask.shape}")
        if mask.shape != expected:
            raise ValueError(
                f"mask shape {mask.shape} does not match bbox extent {expected} "
                f"for bbox {bbox_t}")
        mask_bool = mask.astype(bool, copy=False)
        n_voxels = int(mask_bool.sum())
        if n_voxels == 0:
            raise ValueError(
                "the edited mask is empty. A cell that drops to zero voxels is a deletion, "
                "and deletion is out of scope — see COMPONENT_05_PLAN §9.4a.")
        if nucleus_mask is not None:
            nucleus_mask = np.asarray(nucleus_mask)
            if nucleus_mask.shape != expected:
                raise ValueError(
                    f"nucleus_mask shape {nucleus_mask.shape} does not match bbox "
                    f"extent {expected}")

        vz, vy, vx = record.voxel_size_um
        diag_um = float(np.sqrt(((z1 - z0) * vz) ** 2 + ((y1 - y0) * vy) ** 2
                                + ((x1 - x0) * vx) ** 2))

        from .soma import find_soma

        soma = find_soma(mask_bool, nucleus_mask, voxel_size_um=record.voxel_size_um,
                         **dict(soma_params or {}))
        centroid = (None if soma.centroid is None else
                    (float(soma.centroid[0]) + z0, float(soma.centroid[1]) + y0,
                     float(soma.centroid[2]) + x0))

        # ---- the file write, first ----------------------------------------
        arrays: dict[str, np.ndarray] = {"mask": mask_bool}
        if nucleus_mask is not None:
            arrays["nucleus_mask"] = nucleus_mask.astype(np.uint16, copy=False)
        if thumbnail is not None:
            arrays["thumbnail"] = np.asarray(thumbnail).astype(np.uint8, copy=False)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        _atomic_write_bytes(self.array_path(cell_uid, file_uid), buf.getvalue())

        # ---- then the rows, in one transaction ----------------------------
        with self.conn:
            self.conn.execute(
                "UPDATE cells SET z0 = ?, z1 = ?, y0 = ?, y1 = ?, x0 = ?, x1 = ?, "
                "n_voxels = ?, bbox_diag_um = ?, soma_z = ?, soma_y = ?, soma_x = ?, "
                "soma_remnant_found = ?, human_edited = 1 "
                "WHERE cell_uid = ?",
                (z0, z1, y0, y1, x0, x1, n_voxels, diag_um,
                 None if centroid is None else centroid[0],
                 None if centroid is None else centroid[1],
                 None if centroid is None else centroid[2],
                 int(bool(soma.remnant_found)), cell_uid),
            )
            self.conn.execute(
                "DELETE FROM measurements WHERE cell_uid IN "
                "(SELECT cell_uid FROM cells WHERE file_uid = ?)", (file_uid,))
            self.conn.execute(
                "DELETE FROM image_measurements WHERE file_uid = ?", (file_uid,))

    def counts_by_reject_reason(self, file_uid: str | None = None) -> dict[str, int]:
        """How many cells were removed, and for which structured reason.

        The question free text could not answer: ``debris`` is a correct removal and
        ``real_cell`` is a **data loss**, and the second is the one expected to depend on
        genotype — the cells that are expensive to fix are the clumped 5xFAD ones. A stored
        value that is neither raises, rather than being silently pooled into a bin.
        """
        sql = "SELECT reject_reason, COUNT(*) n FROM cells WHERE reject_reason IS NOT NULL"
        params: list[Any] = []
        if file_uid is not None:
            sql += " AND file_uid = ?"
            params.append(file_uid)
        sql += " GROUP BY reject_reason"
        out: dict[str, int] = {}
        for row in self.conn.execute(sql, params):
            reason = RejectReason(row["reject_reason"])
            out[reason.value] = out.get(reason.value, 0) + row["n"]
        return out

    def accept_rest_of_file(self, file_uid: str) -> int:
        """Confirm every still-unreviewed cell of a file. Returns how many changed.

        Only ``candidate`` rows are touched, so an earlier rejection is never
        resurrected by clearing the rest of the file.
        """
        self._refuse_if_read_only("accept_rest_of_file")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE cells SET status = ?, reviewed_at = ? "
                "WHERE file_uid = ? AND status = ?",
                (CellStatus.CONFIRMED.value, utc_now(), file_uid, CellStatus.CANDIDATE.value),
            )
        return cur.rowcount

    def pending_analysis(self) -> list[CellRecord]:
        """The Phase C queue: confirmed but not yet analysed."""
        return self.cells(status=CellStatus.CONFIRMED)

    def mark_analysed(self, cell_uids: Iterable[str]) -> int:
        self._refuse_if_read_only("mark_analysed")
        uids = list(cell_uids)
        with self.conn:
            cur = self.conn.executemany(
                "UPDATE cells SET status = ? WHERE cell_uid = ? AND status = ?",
                [(CellStatus.ANALYSED.value, u, CellStatus.CONFIRMED.value) for u in uids],
            )
        return cur.rowcount

    # -- Phase C measurements ---------------------------------------------

    def write_measurement(
        self,
        cell_uid: str,
        params_hash: str,
        *,
        payload: Mapping[str, Any],
        status: str = "ok",
        reviewed: bool = True,
        volume_um3: float | None = None,
        n_endpoints: int | None = None,
        ramification_index: float | None = None,
        cd68_integrated: float | None = None,
    ) -> None:
        """Store one cell's Phase C output under the parameters that produced it.

        ``volume_um3`` and ``n_endpoints`` are **promoted** out of the payload into real
        columns because they are what review and analysis sort and filter on; everything
        else, including every curve, lives in the JSON. Promoting more later costs a
        column and no migration of the data itself.

        Re-measuring the same cell under a *different* ``params_hash`` inserts a second
        row rather than replacing the first. That is the point of the composite key: the
        old measurement stays available for comparison and can never be mistaken for the
        new one. Re-measuring under the *same* hash replaces, since that is a repeat of
        the identical computation.
        """
        self._refuse_if_read_only("write_measurement")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO measurements "
                "(cell_uid, params_hash, measured_at, status, reviewed, volume_um3, "
                " n_endpoints, ramification_index, cd68_integrated, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cell_uid, params_hash, utc_now(), status, int(bool(reviewed)),
                 _finite(volume_um3), None if n_endpoints is None else int(n_endpoints),
                 _finite(ramification_index), _finite(cd68_integrated),
                 json.dumps(payload, sort_keys=True, allow_nan=True)),
            )

    def measurements(self, params_hash: str | None = None) -> list[dict[str, Any]]:
        """Stored measurements, newest parameter set first unless one is named."""
        sql = ("SELECT cell_uid, params_hash, measured_at, status, reviewed, volume_um3, "
               "n_endpoints, ramification_index, cd68_integrated, payload "
               "FROM measurements")
        params: list[Any] = []
        if params_hash is not None:
            sql += " WHERE params_hash = ?"
            params.append(params_hash)
        sql += " ORDER BY cell_uid"
        out = []
        for r in self.conn.execute(sql, params):
            row = dict(r)
            row["payload"] = json.loads(row["payload"])
            row["reviewed"] = bool(row["reviewed"])
            out.append(row)
        return out

    def write_image_measurement(self, file_uid: str, params_hash: str, *,
                                payload: Mapping[str, Any],
                                reviewed: bool = True) -> None:
        """Store one image's field-level measures — plaque burden, compartments, background."""
        self._refuse_if_read_only("write_image_measurement")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO image_measurements "
                "(file_uid, params_hash, measured_at, reviewed, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (file_uid, params_hash, utc_now(), int(bool(reviewed)),
                 json.dumps(payload, sort_keys=True, allow_nan=True)),
            )

    def image_measurements(self, params_hash: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT file_uid, params_hash, measured_at, reviewed, payload "
               "FROM image_measurements")
        params: list[Any] = []
        if params_hash is not None:
            sql += " WHERE params_hash = ?"
            params.append(params_hash)
        sql += " ORDER BY file_uid"
        out = []
        for r in self.conn.execute(sql, params):
            row = dict(r)
            row["payload"] = json.loads(row["payload"])
            row["reviewed"] = bool(row["reviewed"])
            out.append(row)
        return out

    def parameter_sets(self) -> list[str]:
        """Every distinct ``params_hash`` with measurements, most-measured first.

        The export uses this to decide how many labelled column groups to write. More
        than one is not an error — it is a pool holding a re-run beside the run it
        replaced, which is exactly what the composite key exists to permit.
        """
        return [r["params_hash"] for r in self.conn.execute(
            "SELECT params_hash, COUNT(*) n FROM measurements "
            "GROUP BY params_hash ORDER BY n DESC, params_hash")]

    def pending_measurement(self, params_hash: str,
                            *, include_unreviewed: bool = False) -> list[CellRecord]:
        """The resumable Phase C queue: cells with no measurement **at this hash**.

        This is what makes ``measure --resume`` correct rather than approximate. Resume
        is not "cells whose status is not ANALYSED" — that flag cannot distinguish a cell
        measured under today's parameters from one measured under last week's. Asking the
        `measurements` table directly means changing a parameter automatically re-queues
        every cell, with no bookkeeping to keep in step.

        ``include_unreviewed`` widens the queue from confirmed cells to candidates as
        well. It exists because Phase B has no interface yet; every measurement it
        produces is stamped ``reviewed = 0`` so the distinction survives to the
        spreadsheet.
        """
        allowed = [CellStatus.CONFIRMED.value]
        if include_unreviewed:
            allowed.append(CellStatus.CANDIDATE.value)
        placeholders = ",".join("?" * len(allowed))
        rows = self.conn.execute(
            f"SELECT c.* FROM cells c "
            f"LEFT JOIN measurements m "
            f"  ON m.cell_uid = c.cell_uid AND m.params_hash = ? "
            f"WHERE c.status IN ({placeholders}) AND m.cell_uid IS NULL "
            f"ORDER BY c.file_uid, c.cell_index_in_file",
            [params_hash, *allowed],
        ).fetchall()
        return [_row_to_cell(r) for r in rows]

    def measurable_images(self, *,
                          include_unreviewed: bool = False) -> list[dict[str, Any]]:
        """Images holding cells measure would take — **whether or not they have been taken.**

        > ### ⇒ THE STABLE LIST. `pending_measurement` IS NOT ONE, AND THAT WAS THE DEFECT
        >
        > `run_phase_c` built its array queue from `pending_measurement` — *cells not yet
        > measured* — and then indexed into it. **That list shrinks as tasks finish**, and
        > Slurm starts array tasks as resources free up rather than all at once. A task
        > starting late got a shorter list and its index pointed at a different image: some
        > images never measured, others twice, **and the array exits 0.**
        >
        > This list depends only on review status, so nothing a measure task does can move it.
        > It is what the pinned plan is built from, and re-running the plan step after more
        > review simply appends.

        Ordered by `sort_key` then `file_uid`: path order, like `files()`, so the plan reads
        the way the dataset is laid out. The tiebreak matters — two files could sort equal.
        """
        allowed = [CellStatus.CONFIRMED.value]
        if include_unreviewed:
            allowed.append(CellStatus.CANDIDATE.value)
        placeholders = ",".join("?" * len(allowed))
        rows = self.conn.execute(
            f"SELECT c.file_uid AS file_uid, f.source_path AS source_path, "
            f"       COUNT(*) AS n_cells "
            f"  FROM cells c JOIN files f ON f.file_uid = c.file_uid "
            f" WHERE c.status IN ({placeholders}) "
            f" GROUP BY c.file_uid ORDER BY f.sort_key, c.file_uid",
            allowed,
        ).fetchall()
        return [{"file_uid": r["file_uid"], "source_path": r["source_path"],
                 "n_cells": int(r["n_cells"])} for r in rows]

    def measurement_progress(self, params_hash: str, *,
                             include_unreviewed: bool = False) -> dict[str, Any]:
        """What measure still owes, at this parameter set. **The measure half of `status`.**

        > ### ⇒ AN ARRAY THAT LOSES TASKS REPORTS SUCCESS
        >
        > A stable index stops drift; it does not tell you that task 400 died of walltime. And
        > the more dangerous mismatch is an **undersized** array, precisely because it does not
        > fail at all — the last images simply never run and every task exits 0.
        >
        > Both show up here as the same fact: images that hold measurable cells with no
        > measurement at this hash. That is `pending_measurement` grouped by image, against
        > `measurable_images` as the denominator.

        ``outstanding`` names the images rather than only counting them, so a re-submit can be
        aimed rather than repeated wholesale.
        """
        measurable = self.measurable_images(include_unreviewed=include_unreviewed)
        pending = self.pending_measurement(params_hash,
                                           include_unreviewed=include_unreviewed)
        by_file: dict[str, int] = {}
        for cell in pending:
            by_file[cell.file_uid] = by_file.get(cell.file_uid, 0) + 1
        cells = sum(entry["n_cells"] for entry in measurable)
        outstanding = [{**entry, "n_pending": by_file[entry["file_uid"]]}
                       for entry in measurable if entry["file_uid"] in by_file]
        return {
            "params_hash": params_hash,
            "images": len(measurable),
            "images_outstanding": len(outstanding),
            "cells": cells,
            "cells_pending": len(pending),
            "cells_measured": cells - len(pending),
            "outstanding": outstanding,
        }

    def counts_by_status(self, file_uid: str | None = None) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) n FROM cells"
        params: list[Any] = []
        if file_uid is not None:
            sql += " WHERE file_uid = ?"
            params.append(file_uid)
        sql += " GROUP BY status"
        return {r["status"]: r["n"] for r in self.conn.execute(sql, params)}

    # -- arrays ------------------------------------------------------------

    def array_path(self, cell_uid: str, file_uid: str | None = None) -> Path:
        if file_uid is None:
            row = self.conn.execute(
                "SELECT file_uid FROM cells WHERE cell_uid = ?", (cell_uid,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no such cell in pool: {cell_uid}")
            file_uid = row["file_uid"]
        return self.root / CELLS_DIR / file_uid / f"{cell_uid}.npz"

    def skeleton_path(self, cell_uid: str, params_hash: str,
                      file_uid: str | None = None) -> Path:
        """Where this cell's two skeletons live for **this parameter set**.

        Keyed by ``params_hash``, like a measurement and unlike a mask. A mask is Phase A and
        does not move when a measurement parameter changes; a skeleton does — `min_spur_um` and
        `loop_prune_method` both reshape it — so storing it beside the mask would let one run
        silently overwrite another's, which is the `min_spur_um` hashing defect of 2026-08-12 in
        a new place.

        Arrays rather than JSON in the measurements table: a few hundred nodes per cell as text
        would put roughly a gigabyte into sqlite across the study, and slow every payload read
        to pay for it. Compressed `.npz` lands near 220 MB for the same content.
        """
        if file_uid is None:
            row = self.conn.execute(
                "SELECT file_uid FROM cells WHERE cell_uid = ?", (cell_uid,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no such cell in pool: {cell_uid}")
            file_uid = row["file_uid"]
        return self.root / SKELETONS_DIR / params_hash / file_uid / f"{cell_uid}.npz"

    def write_skeletons(self, cell_uid: str, params_hash: str,
                        arrays: Mapping[str, np.ndarray]) -> Path:
        """Persist both artefacts. **Never raises** — a lost skeleton must not lose a cell."""
        path = self.skeleton_path(cell_uid, params_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: np.asarray(v) for k, v in arrays.items()})
        _atomic_write_bytes(path, buf.getvalue())
        return path

    def load_skeletons(self, cell_uid: str, params_hash: str) -> dict[str, np.ndarray]:
        """The stored artefacts, or `{}` when this run never wrote any."""
        path = self.skeleton_path(cell_uid, params_hash)
        if not path.exists():
            return {}
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def nucleus_labels_path(self, file_uid: str) -> Path:
        return nucleus_labels_path(self.root, file_uid)

    def load_nucleus_labels(self, file_uid: str, *,
                            native: bool = True) -> np.ndarray | None:
        """The image's surviving nucleus labels, or ``None`` when this pool has none.

        ``native=True`` restores full resolution with `upsample_xy`, which is what review
        needs: every cell's `bbox` is in whole-image voxel coordinates, so a crop taken at
        segmentation resolution would be offset by `downsample` in plane. ``native=False``
        returns the stored array as it is, for a caller working in that frame already.

        **`None` rather than a raise, and only for a pool with no file there at all.** Every
        pool segmented before 2026-09-02 has none, and rich review can still draw everything
        else — the layer is simply empty, which is honest. A file that exists and cannot be
        read is a different thing and raises.
        """
        path = self.nucleus_labels_path(file_uid)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as npz:
            labels = npz["labels"]
            downsample = int(npz["downsample"]) if "downsample" in npz.files else 1
        if not native or downsample == 1:
            return labels
        from .resolution import upsample_xy

        record = self.file(file_uid)
        return upsample_xy(labels, downsample, record.image_shape[1:])

    def load_arrays(self, cell_uid: str,
                    file_uid: str | None = None) -> dict[str, np.ndarray]:
        """Load a cell's stored arrays. Keys present depend on what was written.

        > ### ⇒ `file_uid` IS WHAT MAKES THIS CALLABLE OFF THE OWNING THREAD — CPU patch, §4 B4
        >
        > Without it the path has to be looked up — `SELECT file_uid FROM cells` — and this
        > connection is `check_same_thread=True`, so a worker calling it raises
        > `sqlite3.ProgrammingError` rather than returning the wrong answer.
        >
        > **The review patch's plan recorded this method as *"pure filesystem, touching no
        > SQLite"*. It was not**, and threading `load_file_scene`'s per-cell loop is what found
        > it. Passing the file's own uid — which that caller has in the `FileRecord` it is
        > loading — resolves the path arithmetically and makes the claim true.
        >
        > It is not validated when supplied: a wrong one gives a path that does not exist and
        > the `FileNotFoundError` below, rather than the `KeyError` a lookup would raise.
        """
        path = self.array_path(cell_uid, file_uid)
        if not path.exists():
            raise FileNotFoundError(f"arrays missing for {cell_uid}: {path}")
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}

    # -- maintenance -------------------------------------------------------

    def check_integrity(self) -> IntegrityReport:
        """Look for the wreckage a crashed batch task leaves behind.

        A task killed after writing some arrays but before writing its fragment
        leaves arrays with no manifest entry. Those are harmless — deterministic
        identifiers mean a retry overwrites them — but silent is not the same as
        harmless, so they are reported.
        """
        report = IntegrityReport()
        known = {
            r["cell_uid"]: r["file_uid"]
            for r in self.conn.execute("SELECT cell_uid, file_uid FROM cells")
        }

        cells_root = self.root / CELLS_DIR
        if cells_root.exists():
            for npz in cells_root.glob("*/*.npz"):
                if npz.stem not in known:
                    report.orphan_arrays.append(str(npz))

        for cell_uid, file_uid in known.items():
            if not (cells_root / file_uid / f"{cell_uid}.npz").exists():
                report.missing_arrays.append(cell_uid)

        for row in self.conn.execute(
            "SELECT f.file_uid, f.n_cells, COUNT(c.cell_uid) actual "
            "FROM files f LEFT JOIN cells c ON c.file_uid = f.file_uid "
            "GROUP BY f.file_uid"
        ):
            if row["n_cells"] != row["actual"]:
                report.miscounted_files.append(row["file_uid"])

        return report

    def export_csv(self, path: str | os.PathLike[str] | None = None) -> Path:
        """Write a flat cells-joined-to-files view for spreadsheet use."""
        out = Path(path) if path is not None else self.root / "pool.csv"
        rows = self.conn.execute(
            """
            SELECT c.cell_uid, f.source_filename, f.source_path, c.cell_index_in_file,
                   c.status, c.reject_reason, c.reject_note, c.nucleus_count, c.split_method,
                   c.n_voxels, c.bbox_diag_um,
                   c.soma_remnant_found, c.touches_xy_border,
                   c.z0, c.z1, c.y0, c.y1, c.x0, c.x1,
                   f.voxel_z, f.voxel_y, f.voxel_x, f.run_id, c.created_at, c.reviewed_at
              FROM cells c JOIN files f ON f.file_uid = c.file_uid
             ORDER BY f.sort_key, c.cell_index_in_file
            """
        ).fetchall()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows([tuple(r) for r in rows])
        return out
