"""`prestomorph review --pool P --params F` — Component 5's pool-backed route.

> ### THE SCREENS WERE BUILT FOR THIS. THERE IS NO NEW INTERFACE HERE.
>
> `screen_triage.TriageScreen` takes a list of `ReviewCell`, **not** a `Pool`, and its own
> docstring says *"semi-auto builds them from memory; `review --pool` builds them from
> `CellRecord`s."* `screen_edit_cell` keeps everything non-Qt in `editing.py` *"because both
> routes into this screen — item 7's in-memory session and `review --pool` — need it and
> neither owns it."* `ui/app.ReviewStep` composes the two *"so Component 5's route can reuse
> both against a pool without inheriting Component 3's flow."*
>
> **What was missing was a driver.** This is it. Every widget on the glass is one the
> semi-automatic flow already shows.

## What it does per file

1. **Reconstructs a `Derivation`** from the pool and the source image, and hands it to
   `ReviewStep` exactly as `SemiAutoApp.prepare` does. That is the load-bearing decision of
   this module: `ReviewStep` already knows how to enter a cell, decompose it, transfer
   material between neighbours, repaint the scene and fold an edit back into a label volume —
   **all of it through `derivation`**. Building one here means the two routes run the same
   code rather than two implementations of the same rules, which is `pipeline.Derivation`'s
   own stated reason for existing.
2. **Writes each decision back as it is made** — `COMPONENT_05_PLAN.md` §5, measured at
   2.139 ms per decision on Bunya scratch. Across ~34,000 decisions that is about 73 s spread
   over many sittings, and what it buys is *"you were on cell 37 of 60"* rather than
   restarting the file.
3. **Never reopens a finished file.** §10: *"Within a pool, there is no backward movement."*
   The queue is files with `review_complete = 0`, and within a file it is the cells still
   `candidate`. **Resumable is not revisitable.**

## What it reads from the source, and why it has to

Rich review draws real Iba1 and DAPI, so the source TIFF is re-opened per file and resolved
the way `measure` resolves it — `ImageSource` with the file record's own `channel_roles`.
`COMPONENT_STATUS` records ~0.60 s per channel. Three things are then rebuilt from the source
plus the parameters file, because the pool stores none of them:

* the **measurement mask**, which is what the scene is made of — `build_cell_mask` at
  `mask_coefficient`, exactly the call `segment_image` makes;
* the **conditioned DAPI** the detector saw — filtered, vetoed and log-compressed, the same
  three steps `PipelineCompute._detector_dapi` applies for semi-auto;
* nothing else. The **nucleus label volume** is stored by Phase A since 2026-09-02 (item 6)
  precisely because it is *not* recomputable without re-running Cellpose.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

from .. import cpu
from . import theme
from .app import SCENE_CELL, SCENE_DEBRIS, SCENE_NOTHING, ReviewStep, fit_to_display
from .screen_triage import ReviewCell

__all__ = ["PoolReviewWindow", "FileScene", "Prefetcher", "load_file_scene",
           "detected_ram_bytes", "editor_budget_bytes", "editor_nbytes",
           "PREFETCH_THREADS", "EDITOR_CAP_BYTES", "main"]


class FileScene:
    """Everything one file's review episode needs, assembled once. **Held, not rebuilt.**

    A plain object rather than a dataclass because it is a bag of large arrays whose only
    consumer is `PoolReviewWindow`; giving it behaviour would be giving the driver a second
    place to keep the same facts.
    """

    def __init__(self, *, record, cells, tiles, payloads, derivation, scene,
                 iba1, dapi, nuclei_labels, owned_nuclei, labels_by_uid):
        self.record = record
        self.cells = cells                     # every CellRecord in the file
        self.tiles = tiles                     # ReviewCell per UNRESOLVED cell
        self.payloads = payloads               # cell_uid -> add_cell-shaped dict
        self.derivation = derivation
        self.scene = scene
        self.iba1 = iba1
        self.dapi = dapi
        self.nuclei_labels = nuclei_labels
        self.owned_nuclei = owned_nuclei
        self.labels_by_uid = labels_by_uid     # cell_uid -> label id in derivation.labels


def _conditioned_dapi(source, params, voxel_size_um):
    """**What Cellpose actually saw**: filtered, vetoed, log-compressed, at native resolution.

    The same three steps in the same order as `PipelineCompute._detector_dapi`, rebuilt from
    the parameters file rather than from a session. `nuclei.py` records the measurement that
    makes this worth doing: an aggressive percentile clip found **5** nuclei where log
    compression found **24**, so a reviewer judging whether a nucleus was missed has to be
    looking at the volume the detector judged, not at the raw channel.
    """
    from ..nuclei import normalize_for_detection, subtract_background, veto_mask_at_margin
    from ..threshold import threshold_channel

    if not source.has_role("nuclei"):
        return None
    dapi = np.asarray(source.channel("nuclei"))
    if params.dapi_background_scale_um > 0 and params.dapi_background_strength > 0:
        dapi = subtract_background(dapi, sigma_um=params.dapi_background_scale_um,
                                   voxel_size_um=voxel_size_um,
                                   strength=params.dapi_background_strength)
    if source.has_role("plaque"):
        veto = threshold_channel(source.channel("plaque"),
                                 coefficient=params.plaque_veto_coefficient,
                                 min_object_voxels=50).mask
        applied = veto_mask_at_margin(veto, dilate_um=params.plaque_dilate_um,
                                      voxel_size_um=voxel_size_um)
        dapi = np.where(applied, 0, dapi)
    conditioned = (normalize_for_detection(dapi, method=params.normalize)
                   if params.normalize else dapi)
    #: > ### float32, BECAUSE THIS VOLUME IS ONLY EVER LOOKED AT — 2026-09-11
    #: >
    #: > Every step above promotes to float64, so this came back at **8 bytes a voxel**: 1.70 GB
    #: > for a 40 x 2304 x 2304 field, held for the whole file, against the Iba1 channel's
    #: > 849 MB at float32. It is the `DAPI as detected` layer and nothing else — no measurement
    #: > reads it, and the detector has its own path in `PipelineCompute._detector_dapi` — so
    #: > the second four bytes buy a precision nobody can see and napari discards on upload.
    #: >
    #: > It began to matter when that layer moved from off to **on** for every entry: the crop
    #: > is pushed to the GPU per cell, and at float64 it was the largest single upload on the
    #: > screen — 47.8 MB against the Iba1 crop's 23.9 MB for the same voxels.
    return np.asarray(conditioned, dtype=np.float32)


def _nucleus_crop(nuclei_labels, mask, bbox, nucleus_ids):
    """This cell's own nuclei, clipped to its mask and cropped to its box.

    Byte for byte what `Derivation.payloads` writes, including the `nucleus_ids` filter —
    nuclei are upsampled nearest-neighbour and are not clipped to cell boundaries, so a
    neighbour's nucleus can graze this crop and counting it would imply a clump that was in
    fact correctly separated.
    """
    if nuclei_labels is None:
        return None
    z0, z1, y0, y1, x0, x1 = bbox
    crop = np.where(mask, nuclei_labels[z0:z1, y0:y1, x0:x1], 0).astype(np.uint16)
    own = np.isin(crop, list(nucleus_ids) or [0])
    return np.where(own, crop, 0).astype(np.uint16)


def load_file_scene(pool, record, params, *, thumbnail_max_px: int = 256,
                    threads: int | None = None) -> FileScene:
    """Assemble one file's episode. **No Qt, so the whole of it is testable headless.**

    > ### ⇒ THE PAYLOADS COVER EVERY CELL; THE TILES COVER ONLY THE UNRESOLVED ONES
    >
    > §10c: *"On entering a file, what shows? **Unresolved cells only.**"* — that is the tile
    > list. But rich review judges a cell **against its neighbours**, and a neighbour decided
    > in an earlier sitting is still next to it: `_other_cells_around` needs its mask to find
    > its soma, and `_donor_of` needs it to notice a transfer taking material from it. So the
    > payloads hold the whole file and only the tiles are filtered.

    The scene is pre-painted with the decisions already made, which is exactly the state
    `ReviewStep._repaint_scene` would have left it in had they been made in this sitting:
    a confirmed cell is erased and therefore untakeable, a `debris` rejection is unclaimed
    material, and a `real_cell` rejection is still a cell.
    """
    from .. import qc
    from ..io import ImageSource
    from ..soma import soma_for
    from ..pipeline import Derivation
    from ..schema import CellStatus, RejectReason
    from ..threshold import build_cell_mask

    cells = pool.cells(file_uid=record.file_uid)
    source = ImageSource(record.source_path, dict(record.channel_roles))
    voxel = source.voxel_size_um
    iba1 = np.asarray(source.channel("cell")) if source.has_role("cell") else None
    dapi = _conditioned_dapi(source, params, voxel)
    nuclei_labels = pool.load_nucleus_labels(record.file_uid)

    shape = tuple(int(v) for v in record.image_shape)
    #: uint16 rather than int32: a field holds tens of cells, not tens of thousands, and this
    #: array is the size of the image. `apply_cell_edit` is dtype-agnostic — it compares and
    #: assigns label values and never arithmetics on them.
    labels = np.zeros(shape, np.uint16)
    payloads: dict[str, dict] = {}
    labels_by_uid: dict[str, int] = {}
    tiles: list[ReviewCell] = []
    nuclei_by_cell: dict[int, list[int]] = {}

    #: > ### ⇒ B4 — THE PER-CELL WORK IS THREADED; THE ACCUMULATION IS NOT
    #: >
    #: > `HANDOFF_PATCH_CPU.md` §4 B4. Two independent pieces, and each was checked before it
    #: > was moved:
    #: >
    #: > * `pool.load_arrays` is I/O bound, which is where Lustre latency lives. **The plan
    #: >   recorded it as "pure filesystem, touching no SQLite". It was not** — with no
    #: >   `file_uid` it runs `SELECT file_uid FROM cells` to find the path, and this
    #: >   connection is `check_same_thread=True`, so the first threaded run raised
    #: >   `sqlite3.ProgrammingError`. Passing the file's own uid resolves the path
    #: >   arithmetically and makes the claim true. See `Pool.load_arrays`.
    #: > * `soma_for` is CPU bound and strictly per-cell.
    #: >
    #: > **`pool.cells` is NOT in here**, and that is the constraint rather than an oversight:
    #: > it touches SQLite, whose connection is `check_same_thread=True`, so it stays on the
    #: > owning thread and has already run above.
    #: >
    #: > **And the loop below keeps its order.** `index` is what assigns each cell's label value
    #: > in `labels`, and `labels_by_uid` maps a uid to it — so the compute is parallel and the
    #: > accumulation is serial, in the original order. `map_ordered` holds every cell's arrays
    #: > at once, which costs nothing here: `payloads` keeps all of them anyway.
    #: >
    #: > `qc.triage_signals` stays in the serial loop. It is a handful of array reductions
    #: > beside a `find_soma`, and the handoff names two pieces.
    def _per_cell(cell):
        #: **`file_uid` is not decoration.** Without it `load_arrays` looks the path up in
        #: SQLite, which is exactly the call this loop must not make from a worker — see
        #: `Pool.load_arrays`, and the note below on what the plan got wrong here.
        arrays = pool.load_arrays(cell.cell_uid, file_uid=record.file_uid)
        mask = np.asarray(arrays["mask"]).astype(bool)
        #: **The soma is rebuilt here, and `--params` is why the command demands them.** The
        #: pool stores the soma centroid and not the region, and every triage signal ranks on
        #: the region. This is the same recomputation the semi-automatic route makes in
        #: `review_cells`, through the same two functions, so the two routes rank identically
        #: by construction rather than by agreement. Only the unresolved cells need it — the
        #: rest are in `payloads` for their masks alone.
        soma = (soma_for(params, mask, arrays.get("nucleus_mask"), voxel)
                if cell.status is CellStatus.CANDIDATE else None)
        return arrays, mask, soma

    prepared = cpu.map_ordered(_per_cell, cells, cpu.count(threads))

    for index, (cell, (arrays, mask, soma)) in enumerate(zip(cells, prepared), start=1):
        z0, z1, y0, y1, x0, x1 = cell.bbox
        labels[z0:z1, y0:y1, x0:x1][mask] = index
        labels_by_uid[cell.cell_uid] = index
        nuclei_by_cell[index] = [int(n) for n in cell.nucleus_ids]
        payloads[cell.cell_uid] = {
            "mask": mask,
            "bbox": tuple(int(v) for v in cell.bbox),
            "nucleus_mask": arrays.get("nucleus_mask"),
            "thumbnail": arrays.get("thumbnail"),
            "nucleus_ids": list(cell.nucleus_ids),
            "nucleus_centroids": [tuple(c) for c in cell.nucleus_centroids],
            "split_method": cell.split_method.value,
            "soma_centroid": cell.soma_centroid,
            "soma_remnant_found": cell.soma_remnant_found,
            "touches_xy_border": cell.touches_xy_border,
            "touches_z_border": cell.touches_z_border,
            "human_edited": cell.human_edited,
            "n_voxels_segmentation": cell.n_voxels_segmentation,
        }
        if cell.status is CellStatus.CANDIDATE:
            tiles.append(ReviewCell(
                uid=cell.cell_uid,
                thumbnail=arrays.get("thumbnail"),
                touches_xy_border=cell.touches_xy_border,
                **qc.triage_signals(mask, arrays.get("nucleus_mask"), soma, voxel),
            ))

    #: The measurement mask, rebuilt with the run's own settings — the identical call
    #: `segment_image` makes, `fill_holes=False` included. It is what the scene is made of,
    #: and the pool does not store it: schema 3 took those fields back out because a mask
    #: stored at segment time answers a different question by the time anyone reads it.
    measurement = build_cell_mask(
        iba1, coefficient=float(params.mask_coefficient),
        min_object_voxels=params.measurement_min_object_voxels,
        voxel_size_um=voxel, closing_radius_um=params.mask_closing_radius_um,
        fill_holes=False,
    ).filled if iba1 is not None and params.mask_coefficient is not None else None

    scene = np.zeros(shape, np.uint8)
    if measurement is not None:
        scene[measurement] = SCENE_DEBRIS
    scene[labels > 0] = SCENE_CELL
    for cell in cells:
        if cell.status is CellStatus.CANDIDATE:
            continue
        if cell.status is CellStatus.REJECTED and cell.reject_reason is RejectReason.REAL_CELL:
            continue                          # still a cell, and still takeable
        z0, z1, y0, y1, x0, x1 = cell.bbox
        region = scene[z0:z1, y0:y1, x0:x1]
        mask = np.asarray(payloads[cell.cell_uid]["mask"]).astype(bool)
        region[mask] = (SCENE_DEBRIS if cell.status is CellStatus.REJECTED
                        else SCENE_NOTHING)

    derivation = Derivation(
        labels=labels,
        nuclei_labels=(np.zeros(shape, np.int32) if nuclei_labels is None
                       else np.asarray(nuclei_labels)),
        #: **Deliberately empty.** `segmentation_filled` exists for `payloads()`, which this
        #: route never calls — it reads cells out of the pool rather than deriving them — and
        #: the pool does not store the segmentation mask. A zero-length array makes any future
        #: caller fail loudly here rather than measure `n_voxels_segmentation` against zeros.
        segmentation_filled=np.zeros((0, 0, 0), bool),
        voxel_size_um=voxel,
        n_cells=len(cells),
        nuclei_by_cell=nuclei_by_cell,
        soma_thickness_um=params.soma_thickness_um,
        soma_anchor_um=params.soma_anchor_um,
        soma_method=params.soma_method,
        soma_sheet_area_um2=params.soma_sheet_area_um2,
        soma_waist_um=params.soma_waist_um,
        soma_min_sheet_um3=params.soma_min_sheet_um3,
        thumbnail_max_px=thumbnail_max_px,
    )
    return FileScene(
        record=record, cells=cells, tiles=tiles, payloads=payloads,
        derivation=derivation, scene=scene, iba1=iba1, dapi=dapi,
        nuclei_labels=None if nuclei_labels is None else np.asarray(nuclei_labels),
        owned_nuclei={int(n) for ids in nuclei_by_cell.values() for n in ids},
        labels_by_uid=labels_by_uid,
    )


def _footprint(labels, label, box):
    """One label's mask and tight bounding box, searched inside ``box`` only.

    ``box`` is the union of the cell's stored box and the frame it was edited in, which is
    the whole region an edit can have moved it into — the editable frame is where material
    can be added and the stored box is where it can be removed from. Searching the whole
    volume instead would be a 2304² pass per cell per decision for no extra correctness.
    """
    z0, z1, y0, y1, x0, x1 = box
    sub = labels[z0:z1, y0:y1, x0:x1] == label
    if not sub.any():
        return None, None
    zs, ys, xs = np.nonzero(sub)
    bz0, bz1 = int(zs.min()), int(zs.max()) + 1
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    return (sub[bz0:bz1, by0:by1, bx0:bx1],
            (z0 + bz0, z0 + bz1, y0 + by0, y0 + by1, x0 + bx0, x0 + bx1))


def _union_box(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]),
            min(a[4], b[4]), max(a[5], b[5]))



#: **8, measured on the A16 desktop and flat past it.** `HANDOFF_PATCH_CPU.md` §4 B6, building
#: `CellEditor`s: 1.00x / 1.81x / 2.42x / 2.72x / **2.95x** / 2.94x / 2.95x at 1 / 2 / 4 / 6 / 8 /
#: 12 / 16 threads. A constant rather than `cpu.count`, because it is a measured optimum for this
#: particular work on the machine review runs on, not a share of an allocation.
PREFETCH_THREADS = 8

#: The ceiling, before RAM is considered. §4 B6.
EDITOR_CAP_BYTES = 20 * 1024 ** 3

#: What share of detected RAM the cap may take. **A third**, so the 64 GB A16 desktop lands on
#: the 20 GB the plan asked for (21.3 GB, capped) and a 16 GB machine degrades to ~5 GB instead
#: of thrashing — which is what *"expressed as `min(20 GB, a fraction of detected RAM)` so a
#: smaller desktop degrades instead of thrashing"* is for.
RAM_SHARE = 3.0

#: When the machine will not say how much memory it has. Deliberately small: prefetching less
#: than it could costs seconds, and prefetching more than it has costs the sitting.
UNKNOWN_RAM_BUDGET_BYTES = 4 * 1024 ** 3

#: Bytes per voxel a `CellEditor` holds — intensity float32 (4), mask bool (1), nucleus uint16
#: (2), soma bool (1), three level volumes int32 (12) and `labels` int32 (4). Used to decide
#: whether one **will** fit before it is built; the real figure replaces it once it exists.
EDITOR_BYTES_PER_VOXEL = 24


def detected_ram_bytes() -> int | None:
    """Physical memory, or `None` when the machine will not say.

    Three sources, none of them required: `psutil` if it is installed (it is, on Bunya, pinned
    in `requirements-bunya.txt`), `sysconf` on a POSIX machine, and `GlobalMemoryStatusEx` on
    Windows. **Never raises** — a failure here must cost a smaller prefetch, not a sitting.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:                        # noqa: BLE001 - optional, never fatal
        pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _Status()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:                        # noqa: BLE001 - optional, never fatal
        pass
    return None


def editor_budget_bytes() -> int:
    """``min(20 GB, RAM / 3)``. See `RAM_SHARE` for why it is a share and not a constant."""
    ram = detected_ram_bytes()
    if ram is None:
        return UNKNOWN_RAM_BUDGET_BYTES
    return int(min(EDITOR_CAP_BYTES, ram / RAM_SHARE))


def editor_nbytes(editor) -> int:
    """What one built `CellEditor` actually holds, counted off its own arrays."""
    total = 0
    for value in vars(editor).values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
        elif isinstance(value, tuple):
            total += sum(int(v.nbytes) for v in value if isinstance(v, np.ndarray))
    return total


class Prefetcher(QObject):
    """Work done ahead of the reviewer, on one thread pool. `HANDOFF_PATCH_CPU.md` §4 B2, B6.

    > ### ⇒ WHAT IS BEING BOUGHT, AND WHAT IT IS NOT ALLOWED TO COST
    >
    > Measured on Bunya: a per-file load of **106.0 s**, run synchronously on the Qt thread
    > about 700 times; and a `CellEditor` at **1.83 s**, built on arrival for every cell.
    >
    > **Two kinds of work, one pool, cells first.** Editors are submitted before the file, and
    > the executor is FIFO — which is the priority the reviewer actually has: every cell of this
    > file is entered before the next file is opened. One pool rather than two so they never
    > compete for cores.
    >
    > **The file worker builds its own `Pool(root, read_only=True)`, inside the job.** A SQLite
    > connection belongs to the thread that created it (`check_same_thread=True`), so the
    > driver's pool cannot be handed over — and read-only because a prefetch must never take a
    > write lock on a database the driver is writing decisions into.
    >
    > **Nothing here touches a widget.** The scene arrives on `scene_ready`, and the window asks
    > for it with `take_scene` when it is ready to show it.
    >
    > **Depth 1.** A reviewer only ever moves forward one file, and a scene is ~3.5 GB — two is
    > already 7 GB before the editors are counted.
    >
    > **The cap is a rolling budget, not a ceiling.** An editor is freed the moment its cell is
    > resolved, and the freed bytes go to the next cell that has not been built. Within a file
    > the queue is the cells still `candidate` and *"resumable is not revisitable"*, so a
    > resolved cell is never re-entered.
    >
    > **When the budget is reached, prefetching stops.** It never evicts and it never fails: the
    > cells that were not built are built on arrival, which is exactly what this screen did
    > before any of this existed.
    """

    #: A finished `FileScene`. Emitted from the worker; Qt queues it onto the receiver's thread.
    scene_ready = Signal(object)

    def __init__(self, pool_root, params, *, threads: int = PREFETCH_THREADS,
                 budget_bytes: int | None = None, thumbnail_max_px: int = 256,
                 parent=None):
        super().__init__(parent)
        self._root = Path(pool_root)
        self._params = params
        self._thumbnail_max_px = int(thumbnail_max_px)
        self._budget = (editor_budget_bytes() if budget_bytes is None else int(budget_bytes))
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(threads)),
                                        thread_name_prefix="prestomorph-prefetch")
        self._lock = threading.Lock()
        #: uid -> (stamp, CellEditor). Only ever added to and removed from under the lock.
        self._editors: dict[str, tuple[str, object]] = {}
        #: uid -> bytes reserved for it, whether or not it has finished building.
        self._reserved: dict[str, int] = {}
        self._spent = 0
        self._stopped = False
        #: Bumped whenever the reviewer moves to another file, so a job still in flight for the
        #: previous one drops its result instead of storing it against the new file's uids.
        self._generation = 0
        self._scenes: dict[str, "FileScene"] = {}
        self._scene_wanted: str | None = None

    # -- cell editors ------------------------------------------------------

    def budget_bytes(self) -> int:
        return self._budget

    def spent_bytes(self) -> int:
        with self._lock:
            return self._spent

    def stopped(self) -> bool:
        """True once the budget refused a cell. **Not an error** — see the class docstring."""
        with self._lock:
            return self._stopped

    def start_editors(self, review, uids) -> None:
        """Queue `CellEditor`s for `uids`, in order, until the budget says no.

        `review` is the `ReviewStep`: `build_editor` is its one construction path and carries
        no Qt, so a worker may call it. Submitting in the order triage will offer them is what
        makes the budget spend itself on the cells the reviewer reaches first.
        """
        for uid in uids:
            self._submit_editor(review, uid)

    def _submit_editor(self, review, uid: str) -> bool:
        estimate = self._estimate(review, uid)
        with self._lock:
            if self._stopped or uid in self._reserved or estimate is None:
                return False
            if self._spent + estimate > self._budget:
                #: **Stop, do not evict.** The rest are built on arrival.
                self._stopped = True
                return False
            self._reserved[uid] = estimate
            self._spent += estimate
            generation = self._generation
        self._pool.submit(self._build_editor, review, uid, generation)
        return True

    def _estimate(self, review, uid: str) -> int | None:
        """How much a cell's editor will hold, **before it is built.**

        Admission has to be decided in advance: building one and then finding it does not fit
        would spend 1.83 s to throw it away, and the plan's rule is *stop*, not *evict*.
        """
        payload = review.cells.get(uid)
        if payload is None:
            return None
        z0, z1, y0, y1, x0, x1 = review.editable_box(payload)
        voxels = max(0, z1 - z0) * max(0, y1 - y0) * max(0, x1 - x0)
        return int(voxels) * EDITOR_BYTES_PER_VOXEL

    def _build_editor(self, review, uid: str, generation: int) -> None:
        """Worker side. **Swallows its own failure**: a prefetch that raises must cost the
        reviewer nothing beyond a cell built on arrival."""
        try:
            made = review.build_editor(uid)
        except Exception:                    # noqa: BLE001 - a prefetch may never break review
            made = None
        with self._lock:
            if generation != self._generation or uid not in self._reserved:
                return                       # the file moved on; drop it
            if made is None:
                self._spent -= self._reserved.pop(uid, 0)
                return
            stamp, editor = made
            actual = editor_nbytes(editor)
            self._spent += actual - self._reserved[uid]
            self._reserved[uid] = actual
            self._editors[uid] = (stamp, editor)

    def take_editor(self, uid: str, stamp: str):
        """The prefetched editor for this cell, **only if it was built from this mask.**

        B5c. `None` means build one now — nothing prefetched, the reviewer outran it, or the
        mask has moved since. The entry is handed over rather than copied, and the budget is
        released with it: the cell is being entered, so it is about to be resolved.
        """
        with self._lock:
            held = self._editors.get(uid)
            if held is None or held[0] != stamp:
                return None
            del self._editors[uid]
            self._spent -= self._reserved.pop(uid, 0)
            return held[1]

    def release(self, uid: str) -> None:
        """Free one cell's editor and give its bytes back to the budget.

        > ### ⇒ FREEING A RESOLVED CELL'S EDITOR IS SAFE, AND IT WAS CHECKED RATHER THAN ASSUMED
        >
        > The plan asked for exactly this to be confirmed. A cell **rejected as `real_cell`
        > stays in the scene as takeable material** — so does the reviewer still need its
        > editor? No: taking material runs off `ReviewStep.scene` and the payload masks
        > (`_donor_of`, `_taken_in_image_frame`, `_persist_edits`), and never off the donor's
        > `CellEditor`. The only editor a take touches is the entered cell's own, which is on
        > the screen. Asserted in `test_ui_review_pool.py`.
        """
        with self._lock:
            self._editors.pop(uid, None)
            self._spent -= self._reserved.pop(uid, 0)
            #: The budget rolls: freeing here is what lets the next unbuilt cell in.
            self._stopped = False

    def retain(self, uids) -> None:
        """Free every editor whose cell is no longer pending. **This is the free-on-resolve.**

        Stated as *keep exactly these* rather than *drop that one*, because a cell stops being
        pending three different ways — resolved in rich review, rejected from the grid, or swept
        up by `Accept This Screen` — and a driver that had to name the uid each time would
        eventually miss one. `ReviewStep.entered` is already `None` by the time the driver's
        slot runs, so naming it is not even available at the obvious place.
        """
        keep = set(uids)
        #: Snapshotted under the lock and released outside it — `release` takes the same lock,
        #: and it is a plain `Lock`, so holding it across the loop would deadlock.
        with self._lock:
            drop = [u for u in list(self._editors) + list(self._reserved) if u not in keep]
        for uid in drop:
            self.release(uid)

    def fill(self, review, uids) -> None:
        """Offer the budget to cells that have not been built. Called after a cell is freed."""
        for uid in uids:
            if not self._submit_editor(review, uid):
                if self.stopped():
                    return

    # -- the next file -----------------------------------------------------

    def start_file(self, record) -> None:
        """Queue the next file's `FileScene`. **Depth 1**: one record, and only one.

        Submitted after the editors, so the executor's FIFO order gives the cells the pool
        first — which is the order the reviewer needs them in.
        """
        if record is None:
            return
        with self._lock:
            if self._scene_wanted == record.file_uid or record.file_uid in self._scenes:
                return
            self._scene_wanted = record.file_uid
            generation = self._generation
        self._pool.submit(self._build_scene, record, generation)

    def _build_scene(self, record, generation: int) -> None:
        """Worker side. Its **own** read-only pool, opened and closed on this thread."""
        scene = None
        try:
            from ..store import Pool

            with Pool(self._root, read_only=True) as pool:
                scene = load_file_scene(pool, record, self._params,
                                        thumbnail_max_px=self._thumbnail_max_px)
        except Exception:                    # noqa: BLE001 - a prefetch may never break review
            scene = None
        with self._lock:
            if generation != self._generation or scene is None:
                self._scene_wanted = None
                return
            self._scenes[record.file_uid] = scene
            self._scene_wanted = None
        self.scene_ready.emit(scene)

    def take_scene(self, file_uid: str):
        """The prefetched scene for this file, or `None` — B3's fallback is the caller's."""
        with self._lock:
            return self._scenes.pop(file_uid, None)

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """The reviewer moved to another file. Drop every editor and let in-flight jobs die.

        The generation bump is what makes a job still building a cell of the previous file
        discard its result instead of storing it under a uid the new file may also use.
        """
        with self._lock:
            self._generation += 1
            self._editors.clear()
            self._reserved.clear()
            self._spent = 0
            self._stopped = False

    def shutdown(self) -> None:
        self.reset()
        with self._lock:
            self._scenes.clear()
            self._scene_wanted = None
        self._pool.shutdown(wait=False)


class PoolReviewWindow(QWidget):
    """One sitting. **The window is the driver and the driver is the whole of this component.**

    > ### ⇒ A SESSION IS A SITTING AND NOTHING MORE; THE QUEUE BELONGS TO THE POOL
    >
    > `COMPONENT_05_PLAN.md` §5 and §10. Closing this window loses nothing: every decision was
    > written when it was made, and the next sitting picks up the unresolved cells of the
    > first file that is not `review_complete`.
    >
    > That is why **Back closes the sitting and raises no popup.** In semi-auto, Back leaves
    > item 7 for item 6.5 and its popup warns that every triage decision and every edit goes,
    > because those live only in memory there. Here there is nothing to warn about.
    """

    def __init__(self, pool, params, *, viewer=None, parent: QWidget | None = None,
                 prefetch: "Prefetcher | None" = None):
        super().__init__(parent)
        self.pool = pool
        self.params = params
        self.viewer = viewer
        self.canvas_home = None
        #: > ### ⇒ B2 AND B6 — THE WORK DONE AHEAD OF THE REVIEWER
        #: >
        #: > `HANDOFF_PATCH_CPU.md` §4. The two synchronous costs this route pays are a **106 s**
        #: > file load on the Qt thread and a **1.83 s** `CellEditor` per cell entry; both are
        #: > built ahead here, on one thread pool, with the cells first.
        #: >
        #: > **Injected for tests, and constructed here otherwise.** A test that wants a
        #: > deterministic budget or a single thread hands one in; nothing about the driver
        #: > changes either way.
        self.prefetch = prefetch if prefetch is not None else Prefetcher(
            pool.root, params, thumbnail_max_px=getattr(params, "thumbnail_max_px", 256))
        self.scene: FileScene | None = None
        #: Decisions already written to the pool, so every flush is idempotent and the
        #: three signals below can all call it without counting how many times each fires.
        self.written: set[str] = set()
        self.files = [f for f in pool.files(include_complete=False)]
        self.file_index = 0
        self.finished = False
        self.n_decisions = 0

        self.setStyleSheet(theme.stylesheet())
        self.review = ReviewStep(viewer, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.review)

        #: **Connected after `ReviewStep`'s own**, and Qt honours connection order — so by the
        #: time these run, `_leave` and `_record_resolved` have already put the decisions on
        #: `review.decisions`. That is the whole seam this driver needs, and it needs no
        #: change to the screens to get it.
        self.review.editor.resolved.connect(self._on_cell_resolved)
        self.review.triage.screen_accepted.connect(self._flush)
        #: `Accept This Screen` resolves a band at once, which frees a band's worth of budget.
        self.review.triage.screen_accepted.connect(self._reclaim)
        self.review.continue_requested.connect(self._next_file)
        self.review.back_requested.connect(self._end_sitting)
        self.review.canvas_moved.connect(self.place_canvas)
        #: **B6's seam, and B3's fallback in the same line.** `take_editor` answers `None` for a
        #: cell nothing was built for, one the reviewer outran, or one whose mask has moved
        #: since (B5c) — and `_enter` then builds one, which is what this screen always did.
        self.review.editor_source = self.prefetch.take_editor
        #: Held so the scene is not garbage before `take_scene` asks for it, and so a test can
        #: see that the signal fired. **The window does no work here** — the plan's rule is that
        #: the worker touches no widget, and this only records that something landed.
        self.prefetched_files: list = []
        self.prefetch.scene_ready.connect(self._on_scene_prefetched)

        self._open_current_file()

    # -- the file queue --------------------------------------------------

    def _open_current_file(self) -> None:
        """Load the first file with unresolved cells, marking off any that have none.

        A file can be fully decided and not yet `review_complete` — the sitting that decided
        its last cell may have been closed before pressing Continue. Marking it here rather
        than showing an empty grid is §8a.2's rule: *"an empty queue is a SUCCESS state"*, and
        the same is true one file down.
        """
        from ..schema import CellStatus

        while self.file_index < len(self.files):
            record = self.files[self.file_index]
            if not self.pool.cells(file_uid=record.file_uid, status=CellStatus.CANDIDATE):
                self.pool.mark_file_reviewed(record.file_uid)
                self.file_index += 1
                continue
            self._show_file(record)
            return
        self._end_sitting()

    def _show_file(self, record) -> None:
        #: > ### ⇒ B2 AND B3 — TAKE THE PREFETCHED SCENE, OR BLOCK EXACTLY AS BEFORE
        #: >
        #: > A per-file load is **106.0 s** on Bunya and this line ran it on the Qt thread, so
        #: > the window froze for it about 700 times. `take_scene` answers `None` when the
        #: > reviewer got here first — there is no progress UI and no waiting, just the load
        #: > that always happened.
        self.scene = (self.prefetch.take_scene(record.file_uid)
                      or load_file_scene(self.pool, record, self.params,
                                         thumbnail_max_px=self.params.thumbnail_max_px))
        #: **Before the new file's editors are queued.** The generation bump makes any job still
        #: building a cell of the previous file drop its result rather than store it under a uid
        #: this file may also use.
        self.prefetch.reset()
        self.written.clear()
        scene = self.scene
        self.review.set_cells(
            scene.tiles, scene.payloads, path=record.source_path,
            voxel_size_um=scene.derivation.voxel_size_um, scene=scene.scene,
            iba1=scene.iba1, dapi=scene.dapi, nuclei_labels=scene.nuclei_labels,
            owned_nuclei=scene.owned_nuclei, derivation=scene.derivation)
        #: **Re-stated with the counters**, which `set_cells` cannot know: it calls
        #: `set_file` with the semi-automatic defaults of `file 1 of 1`, and `proceed_label`
        #: reads those to decide between `Next File` and `Continue`. A screen that named a
        #: file it could not open would be lying in the one place the reviewer is looking.
        self.review.triage.set_file(
            scene.tiles, path=record.source_path,
            file_index=self.file_index, file_total=len(self.files))
        self.review.bind_gestures()
        self.place_canvas()
        self._start_prefetch()

    def _start_prefetch(self) -> None:
        """Queue this file's editors, then the next file. **Cells first, and that is the order.**

        One thread pool, submitted FIFO: the reviewer enters every cell of this file before any
        of the next one is needed, so the cells have to be in front. `pending_uids` is triage's
        own offer order — worst band first — so the budget is spent on the cells that will be
        reached soonest.
        """
        self.prefetch.start_editors(self.review, self.review.triage.pending_uids())
        following = self.file_index + 1
        if following < len(self.files):
            #: **Depth 1.** A reviewer only ever moves forward one file, and a scene is ~3.5 GB.
            self.prefetch.start_file(self.files[following])

    def _on_scene_prefetched(self, scene) -> None:
        """A prefetched `FileScene` landed. **Nothing is drawn here** — see `_show_file`."""
        self.prefetched_files.append(getattr(scene.record, "file_uid", None))

    def _next_file(self) -> None:
        """Triage says this file is done. Flush, mark it, and open the next."""
        self._flush()
        if self.scene is not None:
            self.pool.mark_file_reviewed(self.scene.record.file_uid)
        self.file_index += 1
        self._open_current_file()

    def _end_sitting(self) -> None:
        self._flush()
        self.finished = True
        #: **Before the window goes.** The workers hold arrays and a read-only pool of their
        #: own; leaving them running past the sitting would keep both alive with nothing to
        #: hand them to.
        self.prefetch.shutdown()
        self.close()

    # -- writing decisions back -------------------------------------------

    def _on_cell_resolved(self, _resolution) -> None:
        self._flush()
        #: > ### ⇒ B6 — THE CAP IS A ROLLING BUDGET, WHICH IS WHAT THIS LINE MAKES IT
        #: >
        #: > A resolved cell is never re-entered — within a file the queue is the cells still
        #: > `candidate`, and *"resumable is not revisitable"* — so its editor is dead weight
        #: > the moment it is decided. Freeing it and offering the bytes to the next unbuilt
        #: > cell is what turns a 20 GB ceiling into a budget that keeps working through a file
        #: > with more cells than fit in it at once.
        self._reclaim()

    def _reclaim(self) -> None:
        """Give a resolved cell's editor back to the budget, and offer it to the next one."""
        pending = self.review.triage.pending_uids()
        self.prefetch.retain(pending)
        self.prefetch.fill(self.review, pending)

    def _flush(self) -> None:
        """Write every decision not yet written. **Per decision, and that is the design.**

        `COMPONENT_05_PLAN.md` §5: one decision is one `UPDATE` plus a commit, **2.139 ms on
        Bunya scratch** — about 73 s spread across 34,000 decisions, imperceptible after a
        keypress. What it buys is resumability: *"you were on cell 37 of 60"* rather than
        redoing forty decisions, which across many sittings is the difference between a
        review that gets finished and one that does not.

        Idempotent by `written`, so the three signals that call it need not agree on which of
        them fired first.
        """
        if self.scene is None:
            return
        for uid, decision in list(self.review.decisions.items()):
            if uid in self.written:
                continue
            #: **Arrays before rows**, the same order `replace_cell_mask` keeps internally.
            self._persist_masks(uid)
            self.pool.set_status(uid, decision.status,
                                 reject_reason=decision.reason,
                                 reject_note=decision.note,
                                 n_real_cells_here=decision.n_cells)
            self.written.add(uid)
            self.n_decisions += 1

    def _persist_masks(self, uid: str) -> None:
        """Write back any mask this decision changed — **this cell's, and a donor's.**

        A transfer touches two cells: this one gains material and the neighbour loses it.
        `apply_cell_edit` has already made that paired write in the label volume — a voxel
        carries one label, so painting a neighbour's process into this cell removes it from
        the neighbour by construction — and this is where both sides reach storage.

        The candidates are the entered cell plus every cell whose stored box overlaps the
        frame it was edited in. Nothing outside that frame can have moved.
        """
        scene = self.scene
        if scene is None or uid not in scene.payloads:
            return
        edited = self.review.editable_box(scene.payloads[uid])
        candidates = [uid] + [
            other for other, payload in scene.payloads.items()
            if other != uid and _overlaps(payload["bbox"], edited)]
        for candidate in candidates:
            self._persist_one(candidate, edited)

    def _persist_one(self, uid: str, edited_box) -> None:
        scene = self.scene
        payload = scene.payloads[uid]
        label = scene.labels_by_uid[uid]
        stored = tuple(int(v) for v in payload["bbox"])
        region = _union_box(stored, edited_box)

        #: > ### ⇒ *NOTHING MOVED* IS ASKED OF THE VOXELS, NOT OF THE BOUNDING BOX
        #: >
        #: > A stored box need not be tight — `add_cell` takes whatever box its caller gives —
        #: > so comparing the tight footprint's box against the stored one would call an
        #: > untouched cell edited and stamp `human_edited` on it. That column is what tells
        #: > an analyst a cell's numbers are not reproducible from the parameters file, so a
        #: > false positive there is not cosmetic.
        current = scene.derivation.labels[region[0]:region[1], region[2]:region[3],
                                          region[4]:region[5]] == label
        was = np.zeros(current.shape, bool)
        was[stored[0] - region[0]:stored[1] - region[0],
            stored[2] - region[2]:stored[3] - region[2],
            stored[4] - region[4]:stored[5] - region[4]] = \
            np.asarray(payload["mask"]).astype(bool)
        if np.array_equal(current, was):
            return

        mask, bbox = _footprint(scene.derivation.labels, label, region)
        if mask is None:
            #: A cell reduced to nothing cannot be written — `replace_cell_mask` refuses an
            #: empty mask, and deletion is out of scope. `would_empty_the_donor` is what makes
            #: this unreachable for a transfer; it is checked rather than assumed.
            return
        from ..pipeline import _thumbnail

        nucleus = _nucleus_crop(scene.nuclei_labels, mask, bbox, payload["nucleus_ids"])
        self.pool.replace_cell_mask(
            uid, mask=mask, bbox=bbox, nucleus_mask=nucleus,
            #: **The UI computes the thumbnail; storage stores it** — §9.4b, and the reason is
            #: layering: `_thumbnail` is Workstream A's and private, and `store.py` importing
            #: it would open a new dependency direction into A's code for no gain.
            thumbnail=_thumbnail(mask,
                                 nucleus if nucleus is not None
                                 else np.zeros(mask.shape, np.uint16),
                                 scene.derivation.thumbnail_max_px),
            soma_params={"thickness_um": self.params.soma_thickness_um,
                         "anchor_um": self.params.soma_anchor_um,
                         "method": self.params.soma_method,
                         "sheet_area_um2": self.params.soma_sheet_area_um2,
                         "waist_um": self.params.soma_waist_um,
                         "min_sheet_um3": self.params.soma_min_sheet_um3})
        payload["mask"] = mask
        payload["bbox"] = bbox

    def closeEvent(self, event):              # noqa: N802 - Qt's name
        """**The window can also be closed by its own button**, which never reaches `Back`.

        The prefetch workers hold cell arrays and a read-only pool of their own; leaving them
        running past the sitting keeps both alive with nothing to hand them to. `shutdown` is
        safe to call twice, which is what `_end_sitting` closing the window then does.
        """
        self.prefetch.shutdown()
        super().closeEvent(event)

    # -- the canvas --------------------------------------------------------

    def place_canvas(self) -> None:
        """Move the one canvas into whichever inner screen is showing.

        The same act `SemiAutoApp.place_canvas` performs and for the same reasons — one
        viewer moved rather than one per screen, and a screen with no canvas leaves it parked
        where it is rather than re-parenting it out into a stray top-level window.
        """
        host = self.review.canvas_host()
        layout = self.review.canvas_layout()
        if host is None:
            return
        if self.canvas_home is not None and self.canvas_home is not host:
            self.canvas_home.give_up()
        widget = self.viewer.widget() if hasattr(self.viewer, "widget") else None
        host.take(widget)
        self.canvas_home = host
        if layout is not None and self.viewer is not None:
            self.viewer.set_ndisplay(layout.ndisplay)
            self.viewer.set_grid(layout.shape, layout.stride)
            self.viewer.reset_view()
        if self.viewer is not None and hasattr(self.viewer, "focus_canvas"):
            self.viewer.focus_canvas()


def _overlaps(a, b) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3]
                or a[5] <= b[4] or a[4] >= b[5])


def main(pool_root: str, params_path: str) -> int:
    """Open the window and run it. **Everything GUI is imported inside this function's module**,
    which `cli.py` reaches through its one permitted crossing.

    The completion state is a line on the terminal rather than a screen: §8a.2 says an empty
    queue is a success, and there is no built screen that says so — inventing one would be a
    new visual element, which is the user's standing stop-and-ask.
    """
    from ..params import load_params
    from ..store import Pool
    from .viewer import NapariViewer

    params, _meta = load_params(params_path)
    application = QApplication.instance() or QApplication([])
    theme.load_bundled_font()

    with Pool(pool_root) as pool:
        outstanding = len(pool.files(include_complete=False))
        if not outstanding:
            print(f"nothing to review in {pool_root} — every file is complete.")
            return 0
        window = PoolReviewWindow(pool, params, viewer=NapariViewer(show=False))
        if window.finished:
            print(f"nothing to review in {pool_root} — every file is complete.")
            return 0
        window.setWindowTitle(f"prestomorph — review {Path(pool_root).name}")
        fit_to_display(window, application)
        window.show()
        code = int(application.exec_())
        remaining = len(pool.files(include_complete=False))
    print(f"{window.n_decisions} decision(s) written; {remaining} file(s) still to review.")
    return code
