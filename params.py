"""The parameters file — what makes a batch run reproduce a semi-auto one.

`SegmentationParams` already hashes itself and that hash is written onto every cell, so any
result traces to the settings that produced it. This adds the file, its schema version, and a
loader that fails loudly rather than quietly defaulting.

**It stays purely parameters.** Manual nucleus and process edits are *per-image overrides*
that deliberately do **not** transfer to batch (`COMPONENT_03_PLAN.md` §6a) — a hand-placed
split is a judgement about one image and cannot be replayed against a different one. Those
live in the pool as provenance, flagged `human_edited`. Keeping them out is what makes the
reproducibility gate mean anything: everything in this file is replayable by definition.

**Why the schema version matters.** Add a field to `SegmentationParams` and an old file
silently acquires its default — quietly changing what a rerun computes while appearing to
reproduce it. Loading checks the version and refuses rather than guessing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import SegmentationParams

__all__ = ["PARAMS_SCHEMA_VERSION", "save_params", "load_params"]

#: Bump when a field is added, removed or changes meaning. A file written under an older
#: version will not load without an explicit migration.
#: **2** since 2026-08-09: `plaque_coefficient` was split into `plaque_veto_coefficient`
#: and `plaque_measure_coefficient`. A v1 file loaded against this build would silently give
#: the veto its default while the renamed field went unread, so it is refused instead.
#: **3** since 2026-08-10: `plaque_measure_coefficient` (an Otsu multiplier) became
#: `plaque_threshold` (an absolute intensity), `cd68_threshold` was added, and
#: `min_plaque_volume_um3` replaced the two competing size floors. A v2 file loaded against
#: this build would carry a multiplier into a field that now means an intensity — the most
#: dangerous kind of silent migration, because both are plausible numbers.
#: **4** since 2026-08-12: `min_spur_um` became a real field (NEW-g). It had been read as
#: `getattr(params, "min_spur_um", 1.0)`, so it was on no dataclass and reached neither this
#: file nor `params_hash` — two runs pruned at different lengths were indistinguishable. A v3
#: file omitting it is refused here rather than silently given 1.0, which is the rule above
#: applied to the field that most needed it: at 1.0 it removes 31% of all branches and it sets
#: `n_endpoints`, the measure carrying the genotype effect.
#:
#: **The bump was the user's decision, 2026-08-12, and it corrects a premise.** The original
#: NEW-g note argued no bump was needed because "a file omitting the field gets 1.0, exactly
#: today's behaviour". It would not: `load_params` refuses a missing field *and* refuses a
#: `params_hash` that no longer matches, so such a file is rejected twice over either way.
#: **Nothing was refused in practice** — no parameters file existed anywhere when this landed.
#: **5** since 2026-08-15: the four channel roles became real fields (`nuclei_channel`,
#: `cell_channel`, `cd68_channel`, `plaque_channel`) and the dead `nuclei_coefficient` was
#: removed. Both change `asdict`, so both change `params_hash`; one bump covers them because
#: no v5 file existed between the two changes. A v4 file is refused rather than given the
#: shipped channel layout by default: the layout is exactly what a file written for a
#: differently-ordered acquisition would need to override, so filling it in silently is the
#: failure this version number exists to prevent.
#:
#: **Flat fields, not a nested `ChannelRoles`.** The unknown-field check below compares
#: `set(stored) - known` at the **top level only**, so a nested object would have its own key
#: recognised and nothing inside it validated — `{"nucleii": 0}` would pass and take its
#: default, punching a hole through the exact refusal this module exists to make.
#:
#: **6** since 2026-08-23: `dapi_background_strength` and `dapi_background_scale_um` became
#: real fields when the DAPI background filter got its own screen between items 1 and 2. A v5
#: file is missing both, and the refusal below is the point rather than an inconvenience: a
#: missing strength would default to `0.0` — the filter silently off — on a batch of ~700
#: images replaying a session where the user had turned it on. That is the same failure the
#: version number was created for, with the sign reversed.
#:
#: **Still free when it landed** — `HANDOFF_BUILD.md` §2 names this bump as free *"only while no
#: pool exists"*, and none does: no `pool.db` and no parameters file anywhere under the project,
#: re-checked 2026-08-23 before the field was added.
#: **6 -> 7 on 2026-08-24, the debris patch**, and it carries two changes that both move the
#: Phase A hash: `measurement_min_object_voxels` is added, and `speck_floor_um3` is **removed**.
#: A removal needs the bump exactly as much as an addition does — a v6 file names a field this
#: version has no home for, and loading it silently would drop a floor the run was tuned with.
#:
#: **Free again, and re-verified rather than assumed**: no `pool.db` and no parameters file
#: anywhere under the project root, searched 2026-08-24 before either change was made.
#:
#: **7 -> 8 on 2026-08-30**, when item 6.5 was built. Three changes, all moving the Phase A hash:
#: `soma_bound_um` and `soma_occupancy_theta` are **removed** with the `bounded_occupancy`
#: mechanism they were the whole of, and `soma_min_sheet_um3` is **added** — a module constant
#: the user had been walking by hand-editing the source, promoted so item 6.5 could offer it.
#:
#: > **The removals move the hash even at identical settings**, so a run after this lands in a
#: > different pool directory from an otherwise-identical run before it. That is the version
#: > number doing its job rather than a side effect: a v7 file names two fields this build has no
#: > home for, and one it has no value for, and `load_params` refuses all three cases.
#:
#: **Free a third time, checked rather than assumed**: no `pool.db` and no parameters file
#: anywhere under the project root, searched 2026-08-30 before the fields were changed.
#: **8 -> 9 on 2026-08-31**: `soma_anchor_um` is **added**, splitting mechanism 1's radius off
#: `soma_thickness_um`. It defaults to the value the shared field had, so a run at defaults
#: measures exactly what it measured before — but **an addition moves the Phase A hash all the
#: same**, so a run after this lands in a different pool directory from an otherwise-identical
#: run before it, and a v8 file names no value for a field this build requires.
#:
#: **Free a fourth time, checked rather than assumed**: no `pool.db` and no parameters file
#: anywhere under the project root, searched 2026-08-31 before the field was added. The only
#: JSON under the tree is the ground-truth marking output, which carries no `schema_version`.
PARAMS_SCHEMA_VERSION = 9


def save_params(
    params: SegmentationParams,
    path: str | Path,
    *,
    note: str = "",
    source_image: str | None = None,
) -> Path:
    """Write a parameters file, recording enough to explain itself later.

    ``note`` exists because six months on, the question is never *what* the value was — the
    file says that — but *why* it was chosen.

    ``source_image`` records which image the settings were tuned against. That matters for
    the measurement threshold in particular: it transfers to other images as a **coefficient**
    on per-slice Otsu (§4.13.4), so each image computes its own level, but the coefficient was
    still chosen by eye on one specific field and knowing which one is part of the provenance.
    """
    path = Path(path)
    payload = {
        "schema_version": PARAMS_SCHEMA_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params_hash": params.hash(),
        "source_image": source_image,
        "note": note,
        "params": asdict(params),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_params(path: str | Path) -> tuple[SegmentationParams, dict]:
    """Load a parameters file, or fail with a reason.

    Returns the parameters and the surrounding metadata. Three ways this refuses, all of them
    deliberate — a batch of 700 images inheriting a silently wrong parameter is far worse than
    a run that will not start:

    * **wrong schema version** — a field may have been added since, and the file would take
      its default without saying so;
    * **unknown field** — the file was written by newer code, or by hand with a typo, and the
      setting would be silently ignored;
    * **missing field** — the file predates a field, so it would silently default.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    version = payload.get("schema_version")
    if version != PARAMS_SCHEMA_VERSION:
        raise ValueError(
            f"{path.name}: parameters schema version {version!r}, this build expects "
            f"{PARAMS_SCHEMA_VERSION}. Refusing to load — fields may have been added since, "
            "and they would silently take defaults rather than the values you chose."
        )

    stored = payload.get("params")
    if not isinstance(stored, dict):
        raise ValueError(f"{path.name}: no 'params' object in the file.")

    known = {f.name for f in fields(SegmentationParams)}
    unknown = sorted(set(stored) - known)
    missing = sorted(known - set(stored))
    if unknown:
        raise ValueError(
            f"{path.name}: unknown parameter(s) {unknown}. Written by a newer build, or a "
            "typo — either way the setting would be ignored rather than applied."
        )
    if missing:
        raise ValueError(
            f"{path.name}: missing parameter(s) {missing}. Refusing to fill them from "
            "defaults, which would change what this run computes without saying so."
        )

    params = SegmentationParams(**stored)

    recorded = payload.get("params_hash")
    if recorded and recorded != params.hash():
        raise ValueError(
            f"{path.name}: params_hash {recorded} does not match the loaded values "
            f"({params.hash()}). The file has been edited by hand since it was written."
        )

    return params, {k: v for k, v in payload.items() if k != "params"}
