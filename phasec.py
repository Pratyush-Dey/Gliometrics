"""Phase C: confirmed cells in the pool, measurements back into it.

The bridge between Component 6, which measures **one** cell from arrays, and Component 7,
which exports **many** rows. Until this existed the two could not be connected: nothing
outside tests called `measure_cell`, and `export.py` expected `CellRow` objects that nothing
constructed.

Three things shape this module.

**It is driven per image, not per cell.** Every cell needs its source channels, and opening a
2 GB stack once per cell would dominate the runtime. Cells are grouped by `file_uid`, the
image is opened once, and the whole-image plaque distance field is built once — which is
also *required* for correctness, not merely faster: a per-crop distance transform would
measure distance to the nearest plaque **inside the bounding box**, so a plaque just outside
a cell's box would be invisible and the cell would report itself further from amyloid than it
is.

**Failure is isolated per cell.** `measure_one` never raises. A degenerate mask costs that
cell a status string, not the image's other cells — the same contract `batch.run_one` gives
per image, and it composes with Component 6's rule of `NaN` plus a status rather than silent
zeros.

> ### ⇒ AND PER IMAGE, SINCE 2026-09-02 — the user asked for this explicitly
>
> An unreadable source file killed the whole task, because `measure_file` opens the image and
> `run_phase_c` did not guard the call. In the serial case that costs every image after the
> bad one. `run_phase_c` now records the traceback under `measurement_failures/` and carries
> on, exactly as `batch.run_one` does for Phase A.

**Phase C writes FRAGMENTS, not database rows** — since 2026-09-02, and it is what makes a
measure array possible at all. See `store.MeasurementFragment`. `measure_file` takes a
``sink``, which is a fragment in the batch route and could be a `Pool` in any caller that
genuinely holds the single-writer position; the measuring itself has one code path either
way.

**Two different plaque cutoffs are used, deliberately.** Geometry — distance, contact,
clustering, the per-plaque table and the cell-plaque links — uses `plaque_veto_coefficient`,
which is evidence-backed *as a plaque detector* (§4.13.1: recall 97% with it against 94%
without) and is not a measurement. Burden — how much amyloid there is — uses the absolute
`plaque_threshold`, which has **no default** and is chosen by the user in the semi-auto UI.
These answer different questions: "where is the nearest plaque" is far less sensitive to the
cutoff than "how much amyloid is there", because a plaque's centre barely moves as the
threshold changes while its volume changes several-fold. Both are recorded on every
measurement, so neither figure can be read without the value that produced it.

**One size floor, one distance transform.** Until 2026-08-10 every geometry measure was
reported across five `PLAQUE_SIZE_FLOORS_UM3` values, which meant **five whole-image
Euclidean distance transforms per image** — and the plaques table and the per-cell geometry
could still disagree about how many plaques exist, because the table used a *different*
floor. `min_plaque_volume_um3` is now one parameter applied everywhere, which removes 4 of the
5 transforms: roughly **7 minutes per image** at production size.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from . import cpu
from .markers import (
    CONTACT_TOLERANCES_UM,
    MIN_PLAQUE_VOLUME_UM3,
    NO_CD68_CHANNEL,
    ContactMeasures,
    DebrisLoad,
    build_plaque_fields,
    measure_cd68,
    measure_debris,
    measure_cd68_image,
    measure_iba1_image,
    measure_contact,
    measure_plaque_burden,
    measure_plaques,
    measure_plaque_proximity,
    plaques_within,
    sample_distance_um,
)
from .morphometrics import measure_cell
from .skeleton import LOOP_PRUNE_DEFAULT, build_skeletons
from .schema import CellRecord, CellStatus, FileRecord, RejectReason
from .soma import (SOMA_ANCHOR_UM_DEFAULT, SOMA_METHOD_DEFAULT, SOMA_MIN_SHEET_UM3_DEFAULT,
                   SOMA_SHEET_AREA_UM2_DEFAULT, SOMA_THICKNESS_UM_DEFAULT,
                   SOMA_WAIST_UM_DEFAULT, find_soma)
from .store import MeasurementFragment, Pool, _atomic_write_text
from .threshold import build_cell_mask, threshold_channel

__all__ = ["measure_one", "measure_file", "run_phase_c",
           "build_measure_plan", "read_measure_plan", "measure_plan_path"]


def _payload(obj: Any) -> Any:
    """Dataclass -> plain JSON-able dict, with dict keys stringified for JSON."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _payload(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {str(k): _payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_payload(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _stored_soma_point(cell) -> tuple[float, float, float] | None:
    """The run's soma centroid for `cell`, in **crop-local** voxel coordinates. `None` if absent.

    > ### ⇒ THE RULE — use the STORED centroid where you only need a POINT
    >
    > Two soma centroids exist and both are correct for their own job:
    >
    > * **the stored one**, `CellRecord.soma_centroid`, computed by Phase A with **the run's six
    >   soma parameters** (`pipeline.Derivation.payloads`) and re-derived from scratch whenever a
    >   reviewer edits the mask, because `apply_cell_edit` repaints the label volume and lets
    >   `payloads()` rebuild every field. It therefore never goes stale;
    > * **`find_soma`'s**, computed here in `measure_one`, which comes with the soma **region**
    >   the collapse, `soma_volume_um3`, `soma_surface_area_um2`, `soma_sphericity` and the CD68
    >   split all need. The pool stores only the point, never the region.
    >
    > **Use the stored centroid where you only need a point; use `find_soma`'s where the point
    > must match a region.** The skeleton root is the second case and deliberately keeps
    > `find_soma`'s: the root has to be the centre of the region it is collapsing, and pairing a
    > region from one call with a point from another is a worse bug than the one this fixes.
    >
    > What this repairs is `_measure_rejected_real`, which called `find_soma` with **default**
    > parameters while every other path uses the run's — so a rejected cell's
    > `soma_to_plaque_um` was measured from a soma nothing else in the export agrees with.

    **The coordinate frame is the hazard, and this is the one place it is handled.**
    `soma_centroid` is in whole-image voxel coordinates (`schema.CellRecord.soma_centroid`);
    every array in Phase C is a crop. Subtracting `bbox` twice — or adding it — is the defect of
    `EXPORT_AUDIT.md` §3.3, where two cells 2.166 µm apart exported as 4.332. Nothing downstream
    of here may touch the offset again.
    """
    centroid = getattr(cell, "soma_centroid", None)
    if centroid is None:
        return None
    bbox = getattr(cell, "bbox", None)
    if bbox is None:
        return None
    #: `bbox` is half-open `(z0, z1, y0, y1, x0, x1)`, so the crop origin is elements 0, 2, 4.
    return (float(centroid[0]) - float(bbox[0]),
            float(centroid[1]) - float(bbox[2]),
            float(centroid[2]) - float(bbox[4]))


#: > ### ⇒ THE ONE ARRAY IN `measure_one` THAT IS NOT THE CELL'S BBOX SHAPE — 2026-09-02
#: >
#: > Every other array handed to `measure_one` is cropped to the cell's bounding box, and three
#: > functions assert it (`measure_cd68`, `measure_plaque_proximity`, `measure_debris`). The
#: > exact `plaques_within` needs a **padded** window instead, because a plaque within tolerance
#: > can sit outside the cell's own box entirely — which is the whole reason the old
#: > nearest-label method existed at all.
#: >
#: > It is named for what it is and reaches only `plaques_within`. **Do not hand it to anything
#: > that compares shapes.**
def _padded_window(plaque_labels, box, shape, spacing, reach: float):
    """The plaque labels around one cell, padded by `reach` µm, and the window's origin.

    Padding is exact rather than generous: a voxel within `d` µm of the cell is at most
    ``ceil(d / spacing[i])`` indices outside its bounding box on axis `i`, so nothing that could
    qualify is cut off — and anything beyond the pad is further than `reach` by construction and
    cannot change an answer this window is used to compute.
    """
    if plaque_labels is None:
        return None, None
    pad = np.ceil(float(reach) / np.asarray(spacing, float)).astype(int)
    z0, z1, y0, y1, x0, x1 = box
    lo = np.maximum(np.asarray([z0, y0, x0]) - pad, 0)
    hi = np.minimum(np.asarray([z1, y1, x1]) + pad, np.asarray(shape))
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return np.asarray(plaque_labels)[crop], tuple(int(v) for v in lo)


def _cell_in_window(mask, box, offset, window_shape):
    """This cell's mask placed into the padded window, so the two arrays line up."""
    out = np.zeros(window_shape, bool)
    z0, _, y0, _, x0, _ = box
    dz, dy, dx = z0 - offset[0], y0 - offset[1], x0 - offset[2]
    out[dz:dz + mask.shape[0], dy:dy + mask.shape[1], dx:dx + mask.shape[2]] = mask
    return out


def _exact_links(mask, cell, padded_labels, labels_offset, voxel_size_um):
    """`plaques_within` on the padded window, or empty when there are no labels to read."""
    if padded_labels is None or labels_offset is None:
        return {float(t): {} for t in CONTACT_TOLERANCES_UM}
    return plaques_within(
        _cell_in_window(mask, cell.bbox, labels_offset, padded_labels.shape),
        padded_labels, voxel_size_um=voxel_size_um, tolerances=CONTACT_TOLERANCES_UM)


def _link_extras(mask, links, *, skeletons, soma_centroid, nearest_plaque,
                 distance_um, voxel_size_um,
                 measure_contact_area: bool = True,
                 ) -> dict[float, dict[int, dict[str, Any]]]:
    """The per-pair measures of `cell_plaque_links.csv` — added 2026-09-01.

    One entry per `(tolerance, plaque_id)` the link table already holds, so nothing here
    invents a pair: this enriches rows that exist rather than deciding which rows exist.

    ``soma_centroid`` is a **point in crop-local voxel coordinates** — `_stored_soma_point`'s
    output — not a soma object. Only a point is wanted here, so the rule in that function
    applies: this takes the *stored* centroid, which is also what `measure_plaque_proximity`
    reads, so `soma_distance_um` here and `soma_to_plaque_um` on `cells_plaque.csv` are
    measured from the same soma and cannot disagree. Taking a point rather than a region is
    also what lets `_measure_rejected_real` call this at all — it has no soma object.

    ``measure_contact_area`` is `False` for a rejected-real cell. See the population note
    below.

    * **`contact_area_um2`** — this cell's surface within the tolerance of *this* plaque, where
      `plaque_contact_um2@d` on the cell sheet is the total across all of them. **Meshed in
      physical microns since 2026-09-02**, by the same mechanism `markers.measure_contact`
      uses, so the two columns are the same quantity measured the same way.
    * **`soma_distance_um`** — soma centroid to this plaque, beside the nearest-voxel distance
      the row already carries. The pair is what says whether the cell body moved or only a
      process reached.
    * **`is_nearest_plaque`** — nearest by **nearest voxel**, the user's choice of 2026-09-01.
      It disagrees with a soma-based reading for a cell whose process reaches one plaque while
      its body sits closer to another, and the voxel reading is what the rest of this table is
      built on.
    * **`polarity_index`** — the fraction of the arbour's length in the hemisphere facing this
      plaque, and `NaN` rather than 0.5 when it cannot be computed. Measured on the
      **soma-anchored** tree since 2026-09-02; see the block below.

    > ### ⇒ `contact_area_um2` COUNTED VOXEL FACES, AND THE ANSWER DEPENDED ON DIRECTION
    > #### Rebuilt on the mesh, 2026-09-02
    >
    > It was `surface_voxel_count × spacing[1] × spacing[2]` — every exposed voxel charged the
    > area of **one xy face**. At this study's 0.6 × 0.1083 × 0.1083 µm voxel the xy face is
    > 0.0117 µm² while a side face is 0.0650 µm², **5.5× larger**, and the small one was used
    > for both. A surface facing z and a surface of the same true area facing xy therefore
    > measured 5.5× apart, so the number was a statement about which way a process pointed
    > rather than about how much of it lay near the plaque.
    >
    > **That error is not random with respect to the effect under study.** Microglial processes
    > run in every direction and may be *preferentially* oriented near plaques, so a
    > direction-dependent area can correlate with the thing being measured.
    >
    > Now: mesh the cell once in physical microns (`morphometrics.surface_mesh`), take each
    > triangle's centroid and area (`morphometrics._triangle_areas`), read this plaque's
    > distance at each centroid, and sum the areas within the tolerance. Identical in mechanism
    > to `markers.measure_contact`, which is what fills `plaque_contact_um2@d`. A cell below
    > `MIN_MESH_VOXELS` meshes to nothing and reports `NaN` — the same answer the cell sheet
    > already gives it, instead of a voxel count the cell sheet would not agree with.

    > ### ⇒ `polarity_index` MOVED TO THE ANCHORED TREE — 2026-09-02
    >
    > It was measured on the **soma-free** skeleton, on the reasoning that a cell with no
    > anchored tree should still get a direction. But the free skeleton is precisely the
    > artefact that **keeps detached fragments**, and a fragment on the far side of a cell
    > shifts the hemisphere fraction without any process pointing anywhere. That is worst in
    > the clumped, plaque-adjacent fields where detached material is commonest — again an error
    > that lines up with the effect under study rather than scattering around it.
    >
    > **The consequence, accepted:** a cell with no soma has no anchored tree and now reads
    > `NaN`. That is the honest answer. Without a cell body there is no origin for "facing" to
    > be measured from, and the free-skeleton reading was answering a question the data could
    > not support.

    > ### ⇒ THE POPULATION IS C + R SINCE 2026-09-02, AND WHICH COLUMNS R CARRIES
    >
    > `_measure_rejected_real` now calls this, so a rejected-real cell gets link rows. It
    > carries `distance_um`, `soma_distance_um` and `is_nearest_plaque` — all **pure
    > location**, the same class `cells_plaque.csv` already gives R.
    >
    > * `contact_area_um2` stays **blank** (`measure_contact_area=False`): it is a *surface
    >   area*, which is exactly what an uncertified outline gets wrong.
    > * `polarity_index` stays **blank** because R has no skeleton at all.
    >
    > **Why it changed.** Withholding link rows from R bought no protection. R already exports
    > `nearest_voxel_to_plaque_um` — the same distance, merely not broken out per plaque — so
    > the link table published nothing new about a rejected cell. What it did do was make two
    > files disagree about who is near a plaque: `n_cells_within@d` on `plaques.csv` is C + R,
    > and no aggregation of a C-only link table could reproduce it.
    """
    from . import graph_measures as gm
    from .morphometrics import _triangle_areas, surface_mesh

    spacing = np.asarray([float(v) for v in voxel_size_um], float)
    out: dict[float, dict[int, dict[str, Any]]] = {}
    if not links:
        return out

    #: **Meshed once for the cell, not once per pair.** The surface does not depend on which
    #: plaque is being asked about; only the distance read at each face does. Not meshed at all
    #: when the caller has withheld the surface — a rejected-real cell.
    mesh = surface_mesh(mask, voxel_size_um) if measure_contact_area else None
    face_area = face_index = None
    if mesh is not None:
        verts, faces = mesh
        face_area = _triangle_areas(verts, faces)
        #: Face centroids come back in microns in the **unpadded** crop frame, so dividing by
        #: the spacing gives voxel indices into the same arrays the distance fields use.
        idx = np.rint(verts[faces].mean(axis=1) / spacing).astype(int)
        for axis in range(3):
            np.clip(idx[:, axis], 0, mask.shape[axis] - 1, out=idx[:, axis])
        face_index = idx

    #: The nearest plaque over the whole cell, by nearest voxel. One id, so at most one pair
    #: per tolerance is flagged — a cell has one nearest plaque, not one per tolerance.
    nearest_id = None
    if nearest_plaque is not None and distance_um is not None:
        inside = np.asarray(distance_um)[mask]
        labels = np.asarray(nearest_plaque)[mask]
        if inside.size:
            nearest_id = int(labels[int(np.argmin(inside))])

    point = None if soma_centroid is None else np.asarray(soma_centroid, float)
    soma_um = None if point is None else point * spacing
    #: **The anchored tree, not the free skeleton** — see the block in the docstring. `None`
    #: for a cell with no soma, which is what makes `polarity_index` `NaN` there.
    arbour = getattr(skeletons, "soma_anchored", None)

    for tol, pairs in links.items():
        per_plaque: dict[int, dict[str, Any]] = {}
        for pid in (pairs or {}):
            entry: dict[str, Any] = {
                "is_nearest_plaque": (nearest_id is not None and int(pid) == nearest_id),
                "contact_area_um2": float("nan"),
                "soma_distance_um": float("nan"),
                "polarity_index": float("nan"),
            }
            this_plaque = (np.asarray(nearest_plaque) == int(pid)
                           if nearest_plaque is not None else None)
            if this_plaque is not None and this_plaque.any():
                #: Distance to **this** plaque alone, so the pair is measured against the object
                #: named in the row rather than against whichever plaque happens to be nearest.
                own = ndimage.distance_transform_edt(~this_plaque, sampling=tuple(spacing))
                if soma_um is not None:
                    #: **Read sub-voxel, not at the rounded index** — a half-voxel of rounding is
                    #: ±0.30 µm in z against ±0.054 in plane at this geometry, and a soma
                    #: centroid is a continuous quantity. `markers.sample_distance_um` carries
                    #: the reasoning; it clamps, which is the same guard the rounded index had.
                    entry["soma_distance_um"] = float(
                        sample_distance_um(own, point)[0])
                    if arbour is not None:
                        target = np.argwhere(this_plaque)
                        if target.size:
                            entry["polarity_index"] = gm.polarity_index(
                                arbour, soma_um, target.mean(axis=0) * spacing)
                if face_area is not None:
                    at_face = own[face_index[:, 0], face_index[:, 1], face_index[:, 2]]
                    entry["contact_area_um2"] = float(
                        face_area[at_face <= float(tol)].sum())
            per_plaque[int(pid)] = entry
        out[float(tol)] = per_plaque
    return out


def measure_one(
    cell: CellRecord,
    arrays: Mapping[str, np.ndarray],
    *,
    voxel_size_um: Sequence[float],
    cd68: np.ndarray | None = None,
    plaque_channel: np.ndarray | None = None,
    dapi_raw: np.ndarray | None = None,
    distance_um: np.ndarray | None = None,
    nearest_plaque: np.ndarray | None = None,
    plaque_mask: np.ndarray | None = None,
    field_status: str = "ok",
    cd68_threshold: float | None = None,
    iba1: np.ndarray | None = None,
    #: The whole image's shape, so `distance_to_xy_border_um` is measured in the frame
    #: rather than in this cell's own crop. `None` leaves that one column blank.
    image_shape: Sequence[int] | None = None,
    #: The plaque label volume on a window padded past this cell's bbox, and that
    #: window's origin. **Deliberately not the bbox shape** — see `_padded_window`.
    padded_labels: np.ndarray | None = None,
    labels_offset: Sequence[int] | None = None,
    min_spur_um: float = 1.0,
    loop_prune: str = LOOP_PRUNE_DEFAULT,
    soma_thickness_um: float = SOMA_THICKNESS_UM_DEFAULT,
    soma_anchor_um: float = SOMA_ANCHOR_UM_DEFAULT,
    soma_method: str = SOMA_METHOD_DEFAULT,
    soma_sheet_area_um2: float = SOMA_SHEET_AREA_UM2_DEFAULT,
    soma_waist_um: float = SOMA_WAIST_UM_DEFAULT,
    soma_min_sheet_um3: float = SOMA_MIN_SHEET_UM3_DEFAULT,
) -> dict[str, Any]:
    """Every per-cell measure for one cell. **Never raises.**

    Channel arguments are already cropped to this cell's bounding box by the caller;
    ``distance_um`` and ``nearest_plaque`` are crops of the *whole-image* fields, which is
    what keeps "nearest plaque" meaning nearest in the image rather than nearest inside the
    bounding box.

    ``cd68_threshold`` is the user's absolute intensity, or ``None``. There is no default and
    none is invented: unset means `NaN` plus a status on every CD68 volume measure.
    """
    out: dict[str, Any] = {"status": "ok", "payload": {}, "volume_um3": None,
                           "n_endpoints": None, "ramification_index": None,
                           "cd68_integrated": None}
    try:
        mask = np.asarray(arrays["mask"]).astype(bool)
        nucleus = arrays.get("nucleus_mask")
        nucleus = None if nucleus is None else np.asarray(nucleus)

        #: **`find_soma` still runs, and it must.** The collapse needs the soma *region*, and so
        #: do `soma_volume_um3`, `soma_surface_area_um2`, `soma_sphericity` and the CD68 split;
        #: the pool stores only the centroid. See `_stored_soma_point` for which of the two
        #: centroids each consumer is entitled to.
        soma = find_soma(mask, nucleus, voxel_size_um=voxel_size_um,
                         thickness_um=soma_thickness_um, anchor_um=soma_anchor_um,
                         method=soma_method,
                         sheet_area_um2=soma_sheet_area_um2,
                         waist_um=soma_waist_um,
                         min_sheet_um3=soma_min_sheet_um3)
        stored_soma_point = _stored_soma_point(cell)
        # `z_offset` is the one thing `measure_cell` cannot see from a crop: the crop's own
        # origin in the stack, in voxels. Without it `mean_z_um` is a cell thickness rather
        # than a depth. `measure_cell` converts the sum to µm; nothing here scales it.
        #: **Stage 1 runs once and stage 2 reads it.** Skeletonisation is its own step since
        #: 2026-09-01 (`PROJECT_LOG.md` §57); handing the artefacts to `measure_cell` rather
        #: than letting it build its own is what stops the same mask being thinned twice.
        #: ``soma`` goes in whole, not just its centroid — the collapse needs the *region*.
        skeletons = build_skeletons(mask, voxel_size_um=voxel_size_um, soma=soma,
                                    min_spur_um=min_spur_um, loop_prune=loop_prune,
                                    intensity=iba1)
        #: `bbox` and `image_shape` joined 2026-09-01 for the `cells_mask.csv` block: they give
        #: `distance_to_xy_border_um`, which has to be measured in the **whole image** — a
        #: crop-local answer would put every cell on its own edge. `nucleus_mask` joined with
        #: them and left on 2026-09-02 with the three nucleus columns it served. The nucleus is
        #: still read above, for `find_soma`.
        morph = measure_cell(mask, voxel_size_um=voxel_size_um, soma=soma,
                             skeletons=skeletons, min_spur_um=min_spur_um,
                             z_offset=float(cell.bbox[0]),
                             bbox=cell.bbox,
                             image_shape=image_shape)
        # `measure_cell` is handed a centroid, not a soma extent, so it cannot fill this in
        # itself. The join happens here, where both are already in hand. Until 2026-08-10
        # nothing filled it at all and the column was blank in every exported row.
        morph.soma_volume_um3 = soma.volume_um3
        #: **Detached before the payload is built, deliberately.** `_payload` walks every
        #: dataclass field, and `CellMorphology.skeletons` holds numpy arrays that no JSON
        #: encoder will take. The artefacts leave by their own door, as `out["skeletons"]`, and
        #: only their scalar summaries go in the payload below.
        morph.skeletons = None
        payload: dict[str, Any] = {"morphometrics": _payload(morph)}
        #: **Both skeletons summarised, neither serialised.** The scalars belong with the other
        #: measurements; the node and edge geometry is numeric array data and goes to its own
        #: `.npz`, because a few hundred nodes per cell as JSON in a `TEXT` column would put on
        #: the order of a gigabyte into sqlite across the study.
        #:
        #: The **difference between the two endpoint counts is agenda item A2 made visible**:
        #: soma-free counts detached material because it has no root to be unreachable from,
        #: soma-anchored cannot. Neither is wrong and they must not share a column name.
        payload["skeleton_soma_free"] = skeletons.soma_free.summary()
        payload["skeleton_soma_anchored"] = (
            skeletons.soma_anchored.summary() if skeletons.soma_anchored is not None
            else {"status": "no_soma"})
        payload["skeleton_soma_anchored"]["collapse_status"] = skeletons.collapse_status
        #: The two long files' rows, produced here rather than at export time because they are
        #: **measurements**: `graph_measures` reads the artefacts, which exist only in this
        #: stage. Export formats them and computes nothing (`EXPORT_SPEC.md` section 5,
        #: `images.csv`). One dict per branch and per segment, keyed by exported column name.
        try:
            from . import graph_measures as _gm
            if skeletons.soma_anchored is not None:
                payload["branches"] = _gm.branch_rows(skeletons.soma_anchored)
            payload["segments"] = _gm.segment_rows(skeletons.soma_free)
        except Exception:                       # noqa: BLE001 - never lose a cell to a row list
            payload.setdefault("branches", [])
            payload.setdefault("segments", [])
        out["status"] = morph.status
        out["volume_um3"] = (None if not np.isfinite(morph.cell_volume_um3)
                             else float(morph.cell_volume_um3))
        out["n_endpoints"] = None if morph.n_endpoints < 0 else int(morph.n_endpoints)
        out["ramification_index"] = float(morph.ramification_index)

        if cd68 is not None:
            #: `soma_mask` joined 2026-09-01 for the soma/process split. It is the collapse's
            #: own region, so the split inherits `soma_erode_um` exactly as `soma_volume_um3`
            #: does rather than introducing a second boundary.
            cd = measure_cd68(mask, cd68, voxel_size_um=voxel_size_um,
                              threshold=cd68_threshold,
                              soma_mask=getattr(soma, "region", None))
            payload["cd68"] = _payload(cd)
            out["cd68_integrated"] = float(cd.integrated_intensity)
        else:
            #: **A blank status beside blank values cannot be told from a failed measurement.**
            #: The image level and the debris level already say `no_cd68_channel`; the cell level
            #: said nothing at all, because a payload that is never written has no status to
            #: read. Every value column stays blank — export reads them through `.get()` — and
            #: only the status gains a value.
            payload["cd68"] = {"status": NO_CD68_CHANNEL}

        # One size floor, so one set of geometry measures rather than five. A plaque mask
        # that includes noise gives a cell a finite distance to something that is not there —
        # which in WT, where there is no amyloid at all, is the difference between a number
        # and NaN. That is the floor's primary job.
        #: **The STORED centroid, not `find_soma`'s** — `_stored_soma_point` carries the rule
        #: and the frame conversion. Only a *point* is wanted here, so nothing has to match the
        #: soma region, and this is the value `_measure_rejected_real` uses too: the two paths
        #: now measure `soma_to_plaque_um` from the same soma. For a confirmed cell the two
        #: centroids are the same number — Phase A and `measure_one` call `find_soma` on the
        #: same stored mask with the same six parameters — so nothing moves on C.
        payload["plaque_proximity"] = _payload(measure_plaque_proximity(
            mask, distance_um, soma_centroid=stored_soma_point, field_status=field_status))
        if distance_um is not None:
            payload["plaque_contact"] = _payload(measure_contact(
                mask, plaque_mask, voxel_size_um=voxel_size_um,
                other_distance_um=distance_um))
            #: Exact since 2026-09-02 — measured **from the cell** over a padded window
            #: rather than read off the nearest-plaque field. `markers.plaques_within`
            #: carries the measurement that justified the change.
            links = _exact_links(mask, cell, padded_labels, labels_offset,
                                 voxel_size_um)
            payload["plaque_links"] = _payload(links)
            #: The four columns that joined `cell_plaque_links.csv` on 2026-09-01. Keyed by
            #: tolerance and plaque id so the export can attach them to the row that already
            #: carries the distance, and computed here rather than at export time because they
            #: are measurements — export formats and computes nothing.
            #: `plaque_mask` was passed and never read — the per-pair distance comes off
            #: `nearest_plaque`, not off the mask. Dropped 2026-09-02.
            payload["plaque_link_extras"] = _payload(_link_extras(
                mask, links, skeletons=skeletons, soma_centroid=stored_soma_point,
                nearest_plaque=nearest_plaque,
                distance_um=distance_um, voxel_size_um=voxel_size_um))
        else:
            payload["plaque_contact"] = _payload(ContactMeasures(status=field_status))
            payload["plaque_links"] = {str(float(t)): {} for t in CONTACT_TOLERANCES_UM}

        out["payload"] = payload
        #: Handed back for the caller to persist. The geometry is array data and does
        #: not belong in the JSON payload beside the scalars.
        out["skeletons"] = skeletons
    except Exception:                      # noqa: BLE001 - isolation is the point
        out["status"] = "failed"
        out["payload"] = {"traceback": traceback.format_exc()}
    return out



# --------------------------------------------------------------------------
# Debris — rebuilt here, because it is only meaningful after review
# --------------------------------------------------------------------------

#: Reachable values of `debris_status`. **`source_unavailable` is deliberately not among them**:
#: `measure_file` opens the image before anything else it does, so a missing source has already
#: taken the whole call down long before debris is reached.
DEBRIS_OK = "ok"
DEBRIS_NONE = "no_debris"
DEBRIS_NO_CELL_CHANNEL = "no_cell_channel"
DEBRIS_NO_THRESHOLD = "no_measurement_threshold"


def _recompute_debris(src, params, voxel, cells_mask, cd68):
    """The measurement mask, the debris inside it, and the aggregate — or blanks with a status.

    **The mask is rebuilt from the Iba1 channel rather than read back from the pool.** The pool
    stores per-cell crops and no image-level array, and storing one would store a *derived*
    value: debris is the mask minus the surviving cells, and both operands are already durable
    — the channel plus three hashed parameters, and the cells themselves. The difference is the
    only part that can disagree with its own inputs, so it is the part not to keep.

    **`fill_holes=False`, matching `segment_image` exactly.** The measurement mask is the one
    caller that passes it, and a mask rebuilt with filling on would be a different mask —
    2.18% larger on the real crop, and every debris figure computed against it wrong by that
    much in the direction that hides unclaimed material.

    Returns `(measurement_mask, debris_mask, DebrisLoad)`. The first two are `None` when the
    mask could not be rebuilt, and the load then carries the reason rather than a zero.
    """
    if not src.has_role("cell"):
        blank = DebrisLoad()
        blank.status = DEBRIS_NO_CELL_CHANNEL
        blank.volume_um3 = float("nan")
        blank.cd68_integrated = float("nan")
        return None, None, blank
    if getattr(params, "mask_coefficient", None) is None:
        #: Phase A refuses to run without it, so reaching here means a pool written before that
        #: rule or a hand-built parameters file. Blank with a reason beats a plausible zero.
        blank = DebrisLoad()
        blank.status = DEBRIS_NO_THRESHOLD
        blank.volume_um3 = float("nan")
        blank.cd68_integrated = float("nan")
        return None, None, blank

    measurement = build_cell_mask(
        src.channel("cell"),
        coefficient=float(params.mask_coefficient),
        min_object_voxels=int(getattr(params, "measurement_min_object_voxels", 20)),
        voxel_size_um=voxel,
        closing_radius_um=float(getattr(params, "mask_closing_radius_um", 0.8)),
        fill_holes=False,
    ).filled

    debris_mask = measurement & ~cells_mask
    return measurement, debris_mask, measure_debris(debris_mask, voxel_size_um=voxel,
                                                    cd68=cd68)


def _iba1_payload(measurement_mask, debris: DebrisLoad, voxel_volume: float) -> dict:
    """The image's Iba1 burden and its cells/debris breakdown, or blanks with the status."""
    if measurement_mask is None:
        return {
            "iba1_volume_um3": float("nan"),
            "iba1_volume_cells_um3": float("nan"),
            "iba1_volume_debris_um3": float("nan"),
            "debris_n_components": None,
            "debris_cd68_integrated": None,
            "debris_status": debris.status,
            "debris_cd68_status": debris.cd68_status,
        }
    total = float(measurement_mask.sum()) * voxel_volume
    return {
        "iba1_volume_um3": total,
        #: Subtracted rather than counted, so the three cannot fail to sum. Counting the cells
        #: separately would let a voxel owned by a cell **outside** the rebuilt mask — possible
        #: if a parameter moved between segmentation and measurement — break the identity
        #: silently instead of showing up as a debris figure that does not fit.
        "iba1_volume_cells_um3": total - debris.volume_um3,
        "iba1_volume_debris_um3": debris.volume_um3,
        "debris_n_components": debris.n_components,
        "debris_cd68_integrated": (None if np.isnan(debris.cd68_integrated)
                                   else float(debris.cd68_integrated)),
        "debris_status": debris.status,
        "debris_cd68_status": debris.cd68_status,
    }



#: `morphometrics_status` on a cell measured only in part. **Its own value, not `ok` and not a
#: failure**: nothing went wrong, and the shape columns are blank because the reviewer declined
#: to certify the outline, not because the measurement broke. A reader filtering on `ok` gets
#: exactly the certified population; one filtering on failures does not pick these up.
PARTIAL_REJECTED_REAL = "rejected_real_cell_partial"


def _measure_rejected_real(cell, arrays, *, voxel_size_um, cd68, dapi_raw, distance_um,
                           nearest_plaque, field_status, cd68_threshold,
                           padded_labels=None, labels_offset=None):
    """The subset of measures a **rejected-as-`real_cell`** cell may carry. Never raises.

    > ### EXISTENCE AND LOCATION, NEVER SHAPE — the rule, and where the line falls
    >
    > `real_cell` means *"a real cell I could not certify"*. A boundary the reviewer declined to
    > vouch for gives a **broken skeleton** but only a **slightly wrong sum**, so the two classes
    > of measure do not degrade alike:
    >
    > * **carried** — `cell_volume_um3` (a voxel count), the CD68 measures (threshold-free at
    >   the integral, and `cd68_status` is already there to filter on), `mean_z_um` (the
    >   covariate every intensity comparison needs, so it must travel with CD68 or the CD68 is
    >   unusable), and the two plaque **distances**, which are pure location.
    > * **withheld** — every shape measure, `territorial_volume_um3` included. R's hull counts
    >   toward the image union, but a per-cell territory column invites cell-to-cell comparison
    >   in a way an image aggregate does not. The user's decision, 2026-08-25.
    > * **carried since 2026-09-02** — `plaque_links` and the location half of
    >   `plaque_link_extras`. A link row is a distance, not an area. See the block at the call
    >   below for why it changed and what is still withheld.
    > * **withheld** — `plaque_contact`, and `contact_area_um2` on the link rows. Contact is a
    >   *surface area*, which is exactly the quantity an uncertified outline gets wrong.
    """
    out: dict[str, Any] = {"status": PARTIAL_REJECTED_REAL, "payload": {},
                           "volume_um3": None, "n_endpoints": None,
                           "ramification_index": None, "cd68_integrated": None}
    try:
        from .morphometrics import CellMorphology

        mask = np.asarray(arrays["mask"]).astype(bool)
        #: The nucleus mask is **not** read here any more: its only consumer was the `find_soma`
        #: call this path no longer makes.
        voxel_volume = float(np.prod(np.asarray(voxel_size_um, float)))

        #: A real `CellMorphology` rather than a bespoke dict, so the stored payload keeps the
        #: shape every reader already expects and the withheld columns are the dataclass's own
        #: `NaN` defaults — blank because they were never measured, which is what blank means.
        morph = CellMorphology()
        morph.status = PARTIAL_REJECTED_REAL
        morph.cell_volume_um3 = float(mask.sum()) * voxel_volume
        filled = np.argwhere(mask)
        if filled.size:
            morph.mean_z_um = ((float(filled[:, 0].mean()) + float(cell.bbox[0]))
                               * float(voxel_size_um[0]))
        out["payload"]["morphometrics"] = _payload(morph)
        out["volume_um3"] = morph.cell_volume_um3

        if cd68 is not None:
            cd = measure_cd68(mask, cd68, voxel_size_um=voxel_size_um,
                              threshold=cd68_threshold)
            out["payload"]["cd68"] = _payload(cd)
            out["cd68_integrated"] = float(cd.integrated_intensity)
        else:
            #: The rejected-cell path says it too. A cell rejected in a study with no CD68
            #: channel has two reasons for its blank columns and the status names the one that
            #: applies to every cell in the run.
            out["payload"]["cd68"] = {"status": NO_CD68_CHANNEL}

        #: The soma centroid is written at SEGMENT time and survives rejection, which is the
        #: whole reason the distances are available for a cell nobody measured.
        #:
        #: > ### ⇒ IT IS READ, NOT RECOMPUTED — 2026-09-02
        #: >
        #: > This called `find_soma(mask, nucleus, voxel_size_um=voxel_size_um)` and took its
        #: > centroid. Every keyword after the voxel size was left at its **default**, so a
        #: > rejected cell's `soma_to_plaque_um` was measured from a soma built with parameters
        #: > the run never chose, while every other path in the program uses the run's six.
        #: > `_stored_soma_point` returns the value Phase A already computed with those six, so
        #: > this stops being a parameter-plumbing problem instead of solving one.
        #: >
        #: > **Expect R's `soma_to_plaque_um` to move**, and to become blank where the run's
        #: > parameters find no soma that the defaults did. That is the correct answer.
        stored_soma_point = _stored_soma_point(cell)
        out["payload"]["plaque_proximity"] = _payload(measure_plaque_proximity(
            mask, distance_um, soma_centroid=stored_soma_point,
            field_status=field_status))

        #: > ### ⇒ THE LINK TABLE BECAME C + R — 2026-09-02
        #: >
        #: > A link row is a **distance**, and distance is what this population is allowed to
        #: > carry. Withholding these rows bought nothing: `nearest_voxel_to_plaque_um` above
        #: > is the same quantity, merely not broken out per plaque. It did cost something —
        #: > `n_cells_within@d` on `plaques.csv` is C + R, so no aggregation of a C-only link
        #: > table could reproduce it and the two files disagreed about who is near a plaque.
        #: >
        #: > `contact_area_um2` is still withheld, because it is a **surface area** and that is
        #: > the quantity an uncertified outline gets wrong; `polarity_index` is blank because
        #: > this path builds no skeleton. `distance_um`, `soma_distance_um` and
        #: > `is_nearest_plaque` are carried — the same location-only rule `cells_plaque.csv`
        #: > already applies here.
        #: >
        #: > **The method gap is closed too, as of 2026-09-02.** This block once noted that
        #: > `plaques_within` read the nearest-plaque field and so missed a plaque that was
        #: > within tolerance but never nearest. Measured, that was not rare — a 0.25 µm
        #: > process probe missed up to 28.3% at 1.0 µm — and it is now exact for **both**
        #: > populations, so this file and `n_cells_within@d` agree by measurement.
        if distance_um is not None:
            links = _exact_links(mask, cell, padded_labels, labels_offset,
                                 voxel_size_um)
            out["payload"]["plaque_links"] = _payload(links)
            out["payload"]["plaque_link_extras"] = _payload(_link_extras(
                mask, links, skeletons=None, soma_centroid=stored_soma_point,
                nearest_plaque=nearest_plaque, distance_um=distance_um,
                voxel_size_um=voxel_size_um, measure_contact_area=False))
        else:
            out["payload"]["plaque_links"] = {
                str(float(t)): {} for t in CONTACT_TOLERANCES_UM}
    except Exception:
        out["status"] = "failed"
        out["payload"].setdefault("traceback", traceback.format_exc())
    return out


def measure_file(
    pool: Pool,
    file_record: FileRecord,
    cells: Sequence[CellRecord],
    params: Any,
    params_hash: str,
    *,
    reviewed: bool = True,
    roles: Mapping[str, int] | None = None,
    sink: Any | None = None,
    threads: int | None = None,
) -> dict[str, int]:
    """Measure every given cell of one image and write the results into ``sink``.

    ``sink`` carries `write_measurement` and `write_image_measurement` and defaults to the
    pool itself, which is what every direct caller and every test outside the batch route
    wants. The batch route hands a `store.MeasurementFragment` instead — same two method
    names, a file rather than the database, and **no second path through the measuring**.
    """
    from .io import ImageSource

    sink = pool if sink is None else sink

    src = ImageSource(file_record.source_path, dict(roles or file_record.channel_roles))
    voxel = src.voxel_size_um

    cd68_full = src.channel("cd68") if src.has_role("cd68") else None
    dapi_full = src.channel("nuclei") if src.has_role("nuclei") else None
    #: Iba1 is loaded and HELD, not opened transiently, because the
    #: intensity-based loop-pruning rules score edges against it. ~850 MB
    #: resident at production image size, one image at a time, nothing stored.
    iba1_full = src.channel("cell") if src.has_role("cell") else None
    plaque_full = src.channel("plaque") if src.has_role("plaque") else None

    # Geometry uses the veto coefficient; see the module docstring. **One** floor, so one
    # whole-image distance transform per image instead of five.
    floor = float(getattr(params, "min_plaque_volume_um3", MIN_PLAQUE_VOLUME_UM3))
    plaque_mask = None
    distance_um = nearest_plaque = plaque_labels = None
    field_status = "no_plaque_channel"
    if plaque_full is not None:
        plaque_mask = threshold_channel(
            plaque_full, coefficient=float(params.plaque_veto_coefficient),
            min_object_voxels=50).mask
        distance_um, nearest_plaque, plaque_labels, field_status = build_plaque_fields(
            plaque_mask, voxel, min_plaque_volume_um3=floor)

    # Accumulated as cells are measured, so the per-plaque coverage and clustering columns
    # can be computed without a second pass over the pool. uint16 rather than bool because
    # clustering needs to count *distinct* cells, not merely detect presence.
    cell_labels = np.zeros(file_record.image_shape, np.uint16) if plaque_mask is not None \
        else None
    cell_labels_index: list[str] = []
    #: **What each label in `cell_labels` is WORTH**, keyed by label value — 2026-08-30. One
    #: for an ordinary cell; the reviewer's own number on a `real_cell` rejection they judged
    #: to be several fused. It exists because `n_cells_within@d` counts cells and `cell_labels`
    #: holds objects, and a clump is one object worth several cells.
    #:
    #: Read off the record rather than assumed, on both loops below: a confirmed cell is always
    #: 1 — the count popup never fires on an accept — but hard-coding that here would be a
    #: second place for the rule to live.
    cell_label_weights: dict[int, int] = {}
    # How many convex hulls cover each voxel. One uint8 array yields all three territory
    # figures: the summed hull volume is its total, the union is where it is non-zero, and
    # the overlap is the difference. 3DMorph reports only the sum, which double-counts every
    # overlap and is why its "unoccupied volume" can go negative (DESIGN_RATIONALE §5.3).
    hull_counts = np.zeros(file_record.image_shape, np.uint8)

    #: Every voxel any surviving cell owns — **confirmed plus rejected-as-`real_cell`**, the
    #: same inclusion rule `cell_labels` follows. Debris is the measurement mask minus this, so
    #: this array is the subtrahend and nothing else.
    #:
    #: **Boolean, and separate from `cell_labels`, because that one only exists when the image
    #: has a plaque channel** — and an image with no 6E10 still has debris. Sharing it would
    #: make debris silently unavailable on exactly the WT fields where it is least expected.
    cells_mask = np.zeros(file_record.image_shape, bool)

    #: Soma centroids in **whole-image voxel coordinates**, for `n_somata_within@r` on
    #: `plaques.csv`. Gathered in both cell loops, so the population is **C + R** — the same as
    #: `n_cells_within@d` and for the same reason: a review verdict does not change where a
    #: cell body is. `soma_centroid` is written at segment time, which is what makes it
    #: available for a rejected cell that never reached measurement at all.
    soma_centroids: list[tuple[float, float, float]] = []

    def _note_centroid(cell) -> None:
        c = getattr(cell, "soma_centroid", None)
        if c is None:
            return
        try:
            soma_centroids.append((float(c[0]), float(c[1]), float(c[2])))
        except (TypeError, ValueError, IndexError):
            pass

    #: > ### ⇒ C1b — THE PER-CELL WORK RUNS IN PARALLEL; THE ACCUMULATION DOES NOT
    #: >
    #: > `HANDOFF_PATCH_CPU.md` §5. `measure_one` is already a pure function that takes arrays
    #: > and returns a dict and never raises, and `convex_hull_image` is the dominant per-cell
    #: > cost — measured at **~9 s per cell** on a comparable synthetic cell, and it runs in
    #: > this loop and in the rejected-real loop below.
    #: >
    #: > **Threads, not processes, and ~1.6-1.7x is the expectation rather than 2-3x.** Measured
    #: > on 4 threads: `skeletonize` 1.99x, `ndimage.convolve` 1.95x, `distance_transform_edt`
    #: > 1.70x, `convex_hull_image` **1.58x** (it only partially releases the GIL), and
    #: > `marching_cubes` 0.97x (it holds it). The dominant call is the one that releases least.
    #: > That is understood and accepted; processes would mean pickling these arrays.
    #: >
    #: > **The convex hull algorithm is NOT touched.** `offset_coordinates=False`, facet
    #: > reduction and coarser rasterisation were all specifically rejected, because each alters
    #: > `territorial_volume_um3` and `union_debris_um3`, which are exported measures.
    #: >
    #: > **The accumulation stays serial and in order, and it has to be.**
    #: > `label_value = len(cell_labels_index) + 1` assigns label ids by iteration position;
    #: > `cells_mask |= m` and `np.add(hull_counts...)` are commutative but write into shared
    #: > whole-image arrays; `sink.write_measurement` and `pool.write_skeletons` are writes.
    #: > So the compute is parallel and everything that touches shared state is not.
    #: >
    #: > **`file_uid` on `load_arrays` is load-bearing.** Without it that call looks the path up
    #: > in SQLite, whose connection is `check_same_thread=True` — see `Pool.load_arrays`.
    def _measure_one_cell(cell):
        """One cell's whole computation. **Touches only that cell's own arrays.**"""
        from skimage.morphology import convex_hull_image

        z0, z1, y0, y1, x0, x1 = cell.bbox
        box = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        try:
            arrays = pool.load_arrays(cell.cell_uid, file_uid=file_record.file_uid)
        except FileNotFoundError:
            return None
        mask = None
        hull = None
        if "mask" in arrays:
            mask = np.asarray(arrays["mask"]).astype(bool)
            try:
                # A point set's convex hull lies inside its own bounding box, so the crop is
                # sufficient — no need to rasterise into the full image.
                hull = convex_hull_image(mask)
            except Exception:
                hull = mask
        #: The padded plaque-label window for this cell. Built here rather than inside
        #: `measure_one` because only this loop holds the whole-image labels, and it is the
        #: **one** array in the call that is deliberately not the cell's bbox shape.
        padded_labels, labels_offset = _padded_window(
            plaque_labels, cell.bbox, file_record.image_shape, voxel,
            max(CONTACT_TOLERANCES_UM))
        return arrays, mask, hull, box, measure_one(
            cell, arrays, voxel_size_um=voxel,
            cd68=None if cd68_full is None else np.asarray(cd68_full[box]),
            plaque_channel=None if plaque_full is None else np.asarray(plaque_full[box]),
            dapi_raw=None if dapi_full is None else np.asarray(dapi_full[box]),
            distance_um=None if distance_um is None else distance_um[box],
            nearest_plaque=None if nearest_plaque is None else nearest_plaque[box],
            plaque_mask=None if plaque_mask is None else plaque_mask[box],
            field_status=field_status,
            cd68_threshold=getattr(params, "cd68_threshold", None),
            image_shape=file_record.image_shape,
            padded_labels=padded_labels, labels_offset=labels_offset,
            iba1=None if iba1_full is None else np.asarray(iba1_full[box]),
            min_spur_um=params.min_spur_um,
            loop_prune=getattr(params, "loop_prune_method", LOOP_PRUNE_DEFAULT),
            soma_thickness_um=params.soma_thickness_um,
            soma_anchor_um=params.soma_anchor_um,
            soma_method=params.soma_method,
            soma_sheet_area_um2=params.soma_sheet_area_um2,
            soma_waist_um=params.soma_waist_um,
            soma_min_sheet_um3=params.soma_min_sheet_um3,
        )

    done = {"ok": 0, "failed": 0}
    for cell, made in zip(cells, cpu.map_ordered(_measure_one_cell, cells,
                                                 cpu.count(threads))):
        #: **Before the arrays are looked at**, so a cell whose arrays are missing still
        #: contributes its soma centroid — `soma_centroid` is written at segment time and is
        #: available for a cell that never reached measurement at all. This is the order the
        #: serial loop had.
        _note_centroid(cell)
        if made is None:
            sink.write_measurement(cell.cell_uid, params_hash, payload={},
                                   status="arrays_missing", reviewed=reviewed)
            done["failed"] += 1
            continue
        arrays, m, hull, box, got = made

        if m is not None:
            cells_mask[box] |= m
            np.add(hull_counts[box], hull.astype(np.uint8), out=hull_counts[box],
                   casting="unsafe")

        if cell_labels is not None and m is not None:
            label_value = len(cell_labels_index) + 1
            region = cell_labels[box]
            np.copyto(region, np.uint16(label_value), where=m)
            cell_labels_index.append(cell.cell_uid)
            cell_label_weights[label_value] = int(cell.n_real_cells_here or 1)

        # Every cutoff travels on every measurement, so no figure can be read without the
        # value that produced it — and an unset one is visibly None, never a plausible number.
        got["payload"]["coefficients"] = {
            "plaque_veto": float(params.plaque_veto_coefficient),
            "plaque_threshold": (None if params.plaque_threshold is None
                                 else float(params.plaque_threshold)),
            "cd68_threshold": (None if params.cd68_threshold is None
                               else float(params.cd68_threshold)),
            "min_plaque_volume_um3": floor,
            "mask": (None if params.mask_coefficient is None
                     else float(params.mask_coefficient)),
        }
        sink.write_measurement(
            cell.cell_uid, params_hash, payload=got["payload"], status=got["status"],
            reviewed=reviewed, volume_um3=got["volume_um3"],
            n_endpoints=got["n_endpoints"],
            ramification_index=got.get("ramification_index"),
            cd68_integrated=got.get("cd68_integrated"))
        #: **The skeletons are kept, which is the whole point of building them as artefacts.**
        #: A failure to store one must not cost the cell its measurement, though — the numbers
        #: are already written above and the geometry is a convenience beside them — so this is
        #: guarded rather than allowed to propagate, in the same spirit as `measure_one` never
        #: raising.
        skeletons = got.get("skeletons")
        if skeletons is not None:
            try:
                pool.write_skeletons(cell.cell_uid, params_hash, skeletons.to_arrays())
            except Exception:                       # noqa: BLE001 - see above
                pass
        done["failed" if got["status"] == "failed" else "ok"] += 1

    # NEW-b, 2026-08-12 — `cell_labels` adopts `n_real_cells`' inclusion rule: confirmed cells
    # **plus cells rejected as `real_cell`**, every cell the reviewer judged real whether it
    # was kept or discarded for lack of time. Debris is not a cell and stays out.
    #
    # A rejected cell leaves Phase C **by construction** — `pending_measurement` admits only
    # `confirmed` and `candidate` — so it is not in `cells` above and cannot be rasterised from
    # the measured set. It has to be read back from the pool's stored masks. Without this, a
    # plaque engaged by a real cell nobody had time to fix reads one lower in
    # `n_cells_within@d` than `n_real_cells` says the image contains: two columns counting
    # cells by two rules, which is the defect class the export audit exists to remove.
    #
    # **`cell_labels` was the first of these to take rejected cells**, on the grounds that it
    # drives `n_cells_within@d` and `covered_fraction@d`, both of which ask what lies *near* the
    # plaque — and a review verdict does not change where a cell is.
    #
    # > ### ⇒ CORRECTED 2026-09-01. THIS PARAGRAPH SAID THE OPPOSITE OF WHAT THE CODE DOES
    # >
    # > It read: *"`hull_counts` above is untouched: territory is the tissue the **measured**
    # > cells survey, and a cell with no morphometrics has no business contributing a hull to
    # > it"*, citing D-A2 of 2026-08-12 and a test named
    # > `test_the_rejected_cells_do_not_enter_the_territory_measures`.
    # >
    # > **Both halves were wrong.** The user reversed D-A2 on 2026-08-25 and a rejected real
    # > cell now contributes a hull — the code that does it is about fifty lines below, and it
    # > carries the reversal's own reasoning. The test named here **no longer exists** in
    # > `tests/`; it went when the decision did.
    # >
    # > So `union_surveyed_um3`, `union_unsurveyed_um3`, `union_percent_surveyed`,
    # > `overlap_volume_um3` and `overlap_index` are all **C + R**, and `cell_labels` is no
    # > longer the exception it is described as being here. Left in place rather than deleted
    # > because `docs/MEASURE_LEDGER.html` recorded the reversal correctly while two code
    # > comments recorded it backwards, and that is worth one reader knowing about.
    #
    # Invisible in today's output: no human has ever reviewed a cell, so nothing is rejected
    # yet and this loop finds nothing. The test fixture is the only thing holding it.
    #
    # **The loop is no longer conditional on `cell_labels`, and that is a debris fix.** It used
    # to run only when the image had a plaque channel, because `cell_labels` was its only
    # consumer. `cells_mask` is now a second consumer and it exists on every image — leaving the
    # guard would have made every rejected-real cell on a WT field count as DEBRIS, which is the
    # exact opposite of what the user decided: a real cell stays a cell.
    label_gaps: list[str] = []
    measured = {c.cell_uid for c in cells}
    #: **Materialised before anything is threaded.** `pool.cells` reads SQLite, whose connection
    #: belongs to this thread, so the query runs here and the workers see a plain list.
    rejected_real = [
        cell for cell in pool.cells(file_uid=file_record.file_uid,
                                    status=CellStatus.REJECTED)
        if cell.reject_reason == RejectReason.REAL_CELL and cell.cell_uid not in measured]

    #: > ### ⇒ C1b, THE SECOND LOOP — SAME TREATMENT, AND IT DOES HAVE AN ORDER
    #: >
    #: > `HANDOFF_PATCH_CPU.md` §5 called this one *"simpler — it only accumulates into
    #: > `hull_counts` via `np.add`, so it has no ordering constraint at all"*. **It also
    #: > assigns `label_value = len(cell_labels_index) + 1`**, exactly as the first loop does,
    #: > so the ordering constraint is the same one and `map_ordered` supplies it here too.
    #: > Ordered accumulation costs nothing over unordered; getting it wrong renumbers cells.
    def _measure_rejected_cell(cell):
        from skimage.morphology import convex_hull_image

        try:
            arrays = pool.load_arrays(cell.cell_uid, file_uid=file_record.file_uid)
        except FileNotFoundError:
            return None
        if "mask" not in arrays:
            return None
        z0, z1, y0, y1, x0, x1 = cell.bbox
        box = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        m = np.asarray(arrays["mask"]).astype(bool)
        #: The padded plaque-label window for this cell. Built here rather than inside
        #: `measure_one` because only this loop holds the whole-image labels, and it is the
        #: **one** array in the call that is deliberately not the cell's bbox shape.
        padded_labels, labels_offset = _padded_window(
            plaque_labels, cell.bbox, file_record.image_shape, voxel,
            max(CONTACT_TOLERANCES_UM))
        #: **A partial measurement row, since 2026-08-25.** This loop used to rasterise the cell
        #: and move on, so R appeared in the image counts and in nothing else. It now carries
        #: bulk, signal and location — see `_measure_rejected_real` for where the line falls and
        #: why an uncertified outline may be summed but not skeletonised.
        partial = _measure_rejected_real(
            cell, arrays, voxel_size_um=voxel,
            cd68=None if cd68_full is None else np.asarray(cd68_full[box]),
            dapi_raw=None if dapi_full is None else np.asarray(dapi_full[box]),
            distance_um=None if distance_um is None else distance_um[box],
            nearest_plaque=None if nearest_plaque is None else nearest_plaque[box],
            field_status=field_status,
            cd68_threshold=getattr(params, "cd68_threshold", None),
            padded_labels=padded_labels, labels_offset=labels_offset)
        try:
            hull = convex_hull_image(m)
        except Exception:
            hull = m
        return m, hull, box, partial

    for cell, made in zip(rejected_real,
                          cpu.map_ordered(_measure_rejected_cell, rejected_real,
                                          cpu.count(threads))):
        _note_centroid(cell)
        if made is None:
            label_gaps.append(cell.cell_uid)
            continue
        m, hull, box, partial = made
        z0, z1, y0, y1, x0, x1 = cell.bbox
        cells_mask[box] |= m
        sink.write_measurement(
            cell.cell_uid, params_hash, payload=partial["payload"],
            status=partial["status"], reviewed=reviewed,
            volume_um3=partial["volume_um3"], n_endpoints=None,
            ramification_index=None, cd68_integrated=partial["cd68_integrated"])

        #: **A rejected real cell now contributes a hull — the user's decision, 2026-08-25.**
        #: It reverses D-A2 of 2026-08-12, which excluded it because *"a cell with no
        #: morphometrics has no business contributing a convex hull"*. That is a statement about
        #: mechanism, not about geometry: territory is the tissue a cell reaches into, and a real
        #: cell reaches into it whether or not the reviewer had time to certify its outline.
        #: NEW-b already won this argument for the plaque measures — *"a review verdict does not
        #: change where a cell is"* — and leaving the two on opposite rules meant one population
        #: for `n_cells_within@d` and another for `union_surveyed_um3` on the same image.
        np.add(hull_counts[box], hull.astype(np.uint8), out=hull_counts[box],
               casting="unsafe")

        if cell_labels is not None:
            label_value = len(cell_labels_index) + 1
            region = cell_labels[z0:z1, y0:y1, x0:x1]
            np.copyto(region, np.uint16(label_value), where=m)
            cell_labels_index.append(cell.cell_uid)
            #: **This is the loop where the weight is ever more than one.** Only a `real_cell`
            #: rejection can carry a count, and only this loop rasterises those.
            cell_label_weights[label_value] = int(cell.n_real_cells_here or 1)

    # ---- debris: the measurement mask minus every surviving cell -----------
    #
    # **Recomputed here rather than read back, and that is the whole design.** Debris is a
    # SUBTRACTION — *measurement mask − (confirmed + rejected-real)* — so it is only meaningful
    # against the cell set review produced, which does not exist until this point. A copy stored
    # at segment time answers a different question by the time anyone reads it, which is why
    # schema 3 took the four `FileRecord` fields back out.
    #
    # Recomputing is cheap on the terms that matter: this function has **already reopened the
    # source image** for CD68, DAPI and 6E10, so reading one more channel and thresholding it
    # adds no dependency the call did not already have. A stored mask would add ~212 MB per
    # production field and could still go stale; the inputs cannot.
    measurement_mask, debris_mask, debris = _recompute_debris(
        src, params, voxel, cells_mask, cd68_full)
    del cells_mask

    voxel_volume = float(np.prod(np.asarray(voxel, float)))
    field_volume = float(np.prod(np.asarray(file_record.image_shape, float))) * voxel_volume
    covered = hull_counts > 0
    union_um3 = float(covered.sum()) * voxel_volume
    rasterised_sum_um3 = float(hull_counts.sum(dtype=np.int64)) * voxel_volume
    debris_outside_hulls_um3 = (float((debris_mask & ~covered).sum()) * voxel_volume
                                if debris_mask is not None else 0.0)
    territory = {
        # Union basis - ours. Cannot produce an impossible value.
        #
        # **Surveyed, not occupied** (renamed 2026-08-11 at the user's direction). These are
        # the union of the per-cell convex *hulls*: the tissue the microglia collectively
        # reach into and monitor, which is nothing like the volume their bodies occupy —
        # `cell_volume_um3` is that, and it is smaller by an order of magnitude. "Occupied"
        # invited reading a territory measure as bulk. 3DMorph's own `PercentOccupiedVol`
        # kept its name in the export's comparison file, which was itself removed 2026-08-11
        # (`EXPORT_SPEC.md` §13); these union keys were never read by it and are unaffected.
        "field_volume_um3": field_volume,
        "union_surveyed_um3": union_um3,
        #: **Debris outside every hull, and the `~covered` is the whole point.** A fragment
        #: lying *inside* a cell's hull is tissue that cell already surveys; counting it in
        #: both categories would let the three parts exceed the field. Surveyed wins.
        #:
        #: This fixes an acknowledged error rather than adding a measure: `union_unsurveyed_um3`
        #: used to include debris, and the schema said so in as many words — the material was
        #: *"surveyed and then thrown away"*. Unsurveyed now means tissue no microglial anything
        #: reached, which is what the name has always claimed.
        "union_debris_um3": debris_outside_hulls_um3,
        "union_unsurveyed_um3": field_volume - union_um3 - debris_outside_hulls_um3,
        "union_percent_surveyed": (100.0 * union_um3 / field_volume
                                   if field_volume > 0 else float("nan")),
        # Overlap - information 3DMorph's summed figure destroys.
        "overlap_volume_um3": rasterised_sum_um3 - union_um3,
        "overlap_index": ((rasterised_sum_um3 - union_um3) / union_um3
                          if union_um3 > 0 else float("nan")),
        "rasterised_hull_sum_um3": rasterised_sum_um3,
    }
    del hull_counts, covered

    # Per-image measures, stored once against a sentinel row so they survive with the pool.
    #
    # Built unconditionally and added to, rather than in two branches. The threshold-free
    # whole-image measures do not depend on the plaque channel and must not be lost with it:
    # an image with no 6E10 still has territory and still has CD68.
    image_payload: dict[str, Any] = {"territory": territory}

    #: **Iba1 measured off the material, not off the cell set.** A failed split moves voxels from
    #: cells into debris rather than out of the image, so this total cannot be biased low by a
    #: segmentation failure — and this project's own tuning finds splitting failures to be
    #: 5xFAD-only, which would otherwise put that bias in the same direction as the effect the
    #: study exists to test. The three volumes sum by construction: debris **is** the complement.
    #:
    #: `NaN` with a status when the mask could not be rebuilt, never 0.0 — an image whose Iba1
    #: mask genuinely holds nothing unclaimed is a real and good outcome, and it must not read
    #: the same as one where the measurement never ran.
    image_payload["iba1"] = _iba1_payload(measurement_mask, debris, voxel_volume)

    # The CD68 counterpart of the 6E10 `integrated_intensity` below. Threshold-free and
    # mask-free, so it depends on neither the segmentation nor the `cd68_threshold` nobody has
    # chosen — which is exactly why it can serve as the denominator and the sanity check for
    # the per-cell CD68 that depends on both. Absent channel gives NaN plus a status, never 0.
    image_payload["cd68_image"] = _payload(measure_cd68_image(
        None if cd68_full is None else np.asarray(cd68_full)))

    #: **The Iba1 counterpart, added 2026-09-01.** The sheet had a threshold-free burden for
    #: 6E10 and one for CD68 and none for the microglial marker itself, so every Iba1 figure it
    #: carried was downstream of `mask_coefficient` with nothing to check it against.
    image_payload["iba1_image"] = _payload(measure_iba1_image(
        None if iba1_full is None else np.asarray(iba1_full)))

    if plaque_full is not None:
        burden = measure_plaque_burden(
            np.asarray(plaque_full), voxel_size_um=voxel,
            threshold=params.plaque_threshold, min_plaque_volume_um3=floor)
        image_payload["plaque_burden"] = _payload(burden)
        #: **The DAPI dense-core compartments are no longer measured** — the user's decision,
        #: 2026-09-01, removing the measure at both the per-plaque and the image level. It was
        #: unvalidated on this tissue (no Thioflavin-S, no X34), its per-plaque Otsu was computed
        #: against each plaque's own values so the fractions were not comparable between
        #: plaques, and no coefficient in the swept family separated the genotypes — at 4.0 all
        #: seven WT images still reported core, in animals that cannot contain human amyloid.
        #: `measure_plaque_compartments` and `DAPI_CORE_COEFFICIENTS` are dead with it.
        #:
        #: **Soma centroids reach the plaque measures for the first time**, for
        #: `n_somata_within@r`. They are the Phase A centroids read back with the cell labels,
        #: which is what makes the column C+R without a second soma-finding pass.
        image_payload["plaques"] = _payload(measure_plaques(
            plaque_mask, np.asarray(plaque_full), voxel_size_um=voxel,
            cell_labels=cell_labels, cell_weights=cell_label_weights,
            iba1_mask=measurement_mask, soma_centroids=soma_centroids,
            min_plaque_volume_um3=floor))
        image_payload["cell_label_index"] = list(cell_labels_index)
        # A `real_cell` rejection whose arrays have gone cannot be rasterised, and the two
        # cell-facing plaque columns would then under-count **silently** — precisely the defect
        # NEW-b closes, re-entering by a different door. Named, not dropped. Absent when empty,
        # so the key's presence is itself the signal.
        if label_gaps:
            image_payload["cell_label_gaps"] = list(label_gaps)

    sink.write_image_measurement(file_record.file_uid, params_hash,
                                 payload=image_payload, reviewed=reviewed)
    return done


MEASUREMENT_FAILURES_DIR = "measurement_failures"

#: The pinned array list, one per parameter set. `pool/measure_plans/<params_hash>.json`.
MEASURE_PLANS_DIR = "measure_plans"


def measure_plan_path(pool_root: str | Path, params_hash: str) -> Path:
    return Path(pool_root) / MEASURE_PLANS_DIR / f"{params_hash}.json"


def build_measure_plan(pool_root: str | Path, params_hash: str, *,
                       include_unreviewed: bool = False) -> list[dict]:
    """Pin the array's image list to a file, and return it. **Before the array, once.**

    > ### ⇒ THE INDEX POINTED AT A MOVING LIST, AND THE ARRAY EXITED 0
    >
    > `run_phase_c` built its queue from `pending_measurement` — *cells not yet measured* —
    > and indexed into it. That list **shrinks as tasks finish**, and Slurm starts array
    > tasks as resources free up rather than all at once. A task starting late got a shorter
    > list and its index pointed at a different image: **some images never measured, others
    > twice, and nothing said so.**
    >
    > `Pool.measurable_images` is the stable half — it depends on review status alone, so no
    > measure task can move it. Pinning it to a file is the other half, and it is the user's
    > choice over reading it live: **both phases then work identically.** A prepare step
    > prints a number, the array reads a file. One shape learned once, instead of two that
    > look alike and differ under pressure.

    Keyed by `params_hash` in the filename, like the fragments. The list itself does not
    depend on the parameters — that is the point — but two runs must not share one pinned
    bound, or review done between them would silently resize the first run's array.
    """
    with Pool(pool_root, read_only=True) as pool:
        entries = pool.measurable_images(include_unreviewed=include_unreviewed)
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params_hash": params_hash,
        "include_unreviewed": bool(include_unreviewed),
        "n_images": len(entries),
        "images": entries,
    }
    _atomic_write_text(measure_plan_path(pool_root, params_hash),
                       json.dumps(payload, indent=2))
    return entries


def read_measure_plan(pool_root: str | Path, params_hash: str) -> dict:
    """The pinned payload. Raises `FileNotFoundError` naming the step that writes it."""
    path = measure_plan_path(pool_root, params_hash)
    if not path.exists():
        raise FileNotFoundError(
            f"no measure plan at {path}. Run `prestomorph measure-plan --pool <pool> "
            "--params <file>` first — it pins the image list every array task indexes into "
            "and prints the --array bound.")
    return json.loads(path.read_text(encoding="utf-8"))


def _record_image_failure(pool_root: str, params_hash: str, file_uid: str,
                          source_path: str) -> Path:
    """One image's traceback, under `measurement_failures/<params_hash>/<file_uid>.txt`.

    **Its own directory, not Phase A's `failures/`.** The two phases fail for different
    reasons at different times, and item 9's status command reports on them separately; one
    mixed directory would make *"which images still owe me a measurement"* unanswerable from
    the filesystem. Keyed by `params_hash` for the same reason the fragments are: a failure
    under one parameter set says nothing about another.
    """
    out = (Path(pool_root) / MEASUREMENT_FAILURES_DIR / params_hash / f"{file_uid}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"{source_path}\n\n{traceback.format_exc()}", encoding="utf-8")
    return out


def run_phase_c(
    pool_root: str,
    params: Any,
    *,
    index: int | None = None,
    include_unreviewed: bool = False,
    roles: Mapping[str, int] | None = None,
    threads: int | None = None,
) -> dict[str, int]:
    """Measure everything outstanding at this parameter set. Resumable by construction.

    ``index`` selects one image, for a Slurm array. Resume needs no flag: the queue is
    "cells with no measurement at this hash", so re-running simply finds less to do.

    > ### ⇒ IT WRITES FRAGMENTS AND OPENS THE POOL READ-ONLY — 2026-09-02
    >
    > Both halves are one change. `_connect` runs `executescript(_DDL)` on every open, so a
    > task that merely wanted to read a mask was already a writer; and the measurements went
    > straight into `pool.db`, with no WAL and no `busy_timeout`. Hundreds of array tasks
    > therefore collided on the lock and each one died **after** doing all the expensive work.
    >
    > Now the pool is opened `read_only=True` — no DDL, and it cannot take a write lock even
    > by accident — and each image's output goes to one atomic fragment. `merge-measurements`
    > folds them in afterwards, single-writer, once. This is Phase A's pattern exactly.

    **A failed image writes no fragment**, which is what keeps it in the queue: the queue is
    the `measurements` table, so an image whose rows never landed is still outstanding and a
    resubmit re-runs it. Again, exactly what `batch.run_one` does with its own fragment.
    """
    params_hash = params.hash()
    totals = {"ok": 0, "failed": 0, "images": 0, "failed_images": 0}
    plan = None if index is None else read_measure_plan(pool_root, params_hash)
    with Pool(pool_root, read_only=True) as pool:
        pending = pool.pending_measurement(params_hash,
                                           include_unreviewed=include_unreviewed)
        by_file: dict[str, list[CellRecord]] = {}
        for c in pending:
            by_file.setdefault(c.file_uid, []).append(c)

        if index is None:
            #: **The serial path needs no plan and is not given one.** It computes its list
            #: once, in one process, so there is nothing for a later task to disagree with —
            #: the drift the plan exists to stop is a property of an array, not of a loop.
            file_uids = sorted(by_file)
        else:
            images = plan["images"]
            if index < 0:
                raise ValueError(f"index {index} is not a 0-based array index")
            if index >= len(images):
                #: Exit quietly, exactly as an out-of-range segment index does: an oversized
                #: array is not a failure, and a non-zero exit would hold the `afterok`
                #: dependency in the queue forever.
                print(f"index {index} is past the end of a {len(images)}-image measure plan, "
                      "nothing to do")
                return totals
            uid = images[index]["file_uid"]
            #: **A task whose image is already measured exits quietly.** That is what makes
            #: the stable list safe to index: the list holds every measurable image, so a
            #: resubmit finds the finished ones here and does nothing rather than shifting
            #: everyone else's index by one.
            if uid not in by_file:
                print(f"index {index} ({images[index]['source_path']}) is already measured "
                      f"at {params_hash[:8]}, nothing to do")
                return totals
            file_uids = [uid]

        for uid in file_uids:
            record = pool.file(uid)
            #: **Not a context manager**, deliberately. `__exit__` writes the fragment on a
            #: clean exit, and an exception caught *inside* this loop is a clean exit from
            #: its point of view — so a half-measured image would be written, merged, and
            #: then look complete. `close()` is called only where the measuring finished.
            fragment = MeasurementFragment(pool_root, file_uid=uid, params_hash=params_hash)
            try:
                got = measure_file(pool, record, by_file[uid], params, params_hash,
                                   reviewed=not include_unreviewed, roles=roles,
                                   sink=fragment, threads=threads)
            except Exception:                      # noqa: BLE001 — isolation is the point
                _record_image_failure(pool_root, params_hash, uid, record.source_path)
                totals["failed_images"] += 1
                continue
            fragment.close()
            totals["ok"] += got["ok"]
            totals["failed"] += got["failed"]
            totals["images"] += 1
    return totals
