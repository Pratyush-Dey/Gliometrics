"""Phase A across hundreds of images: one independent Slurm task per file.

`store.py` already provides the contract that makes this safe — `PoolWriter` writes a fragment
per image and never touches the database, `merge_fragments` folds them in afterwards. This
supplies the driver, the manifest, and the failure isolation.

Three decisions from the user (`COMPONENT_04_PLAN.md` §5a), each chosen against a silent
alternative:

* **Glob, then write the manifest it used.** Discovery is convenient; a recorded list is
  auditable. Both, so a run can be replayed months later even if the directory has moved on.

  > ### ⇒ THE GLOB IS ITS OWN STEP NOW, AND `segment` ONLY READS — 2026-09-02
  >
  > Every array task used to re-glob the whole tree and rewrite `manifest.json` with a plain
  > `write_text`. Three faults in one line: **a race on the run's own audit record**, hundreds
  > of concurrent recursive globs against Lustre for no benefit, and **nothing anywhere printed
  > how many images there were** — so the `--array=0-N` bound was a hand-counted number that
  > could disagree with what the program found.
  >
  > `manifest` globs once, writes atomically and prints the count and the array bound.
  > `segment` reads that file. The count the operator types and the list the tasks index into
  > now come from one source and **cannot** disagree.
* **Refuse an ambiguous resubmit.** Skipping finished images automatically would make
  "reprocess everything" look like a no-op; reprocessing automatically would repeat 600
  finished files. It stops and asks for ``--resume`` or ``--force``.
* ~~**One pool, cells flagged.** Semi-auto and batch write to the same pool~~ — **superseded.**
  `COMPONENT_03_PLAN.md` §4a.3/§4a.9 settled on **two pools**: `pool_<origin>_<phase_a_hash>`,
  so semi-auto and batch never share one. The `human_edited` flag stays and does the job it
  always did — a cell a human edited is not reproducible from the parameters file — but it is
  no longer what separates the two routes. **`segment` refuses a pool whose name says
  `semiauto`**; see `main`.

**One failing image must not stop the array.** The fragment design already isolates failures;
the driver must not undo that by exiting on the first exception.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import cpu
from .io import ChannelRoles
from .metadata import parse_metadata
from .params import load_params
from .pipeline import SegmentationParams, segment_image
from .nuclei import describe_device, resolve_device
from .store import DB_NAME, Pool, _atomic_write_text, pool_origin

__all__ = ["build_manifest", "load_manifest", "read_manifest", "write_manifest",
           "run_one", "main", "manifest_main", "segment_progress",
           "IMAGE_SUFFIXES", "DEPTH_PATTERN"]

MANIFEST_NAME = "manifest.json"


#: Both spellings of the same format. **The RDM writes `.tiff` and the local test crops are
#: `.tif`** — measured 2026-09-02: 573 images under `Z:\Pratyush\Aim 2 Images` end `.tiff` and
#: **none** ends `.tif`, so the shipped `*/*/*.tif` matched zero of them and the run exited 2
#: before it started. Compared case-folded, because a suffix is not a statement about case.
IMAGE_SUFFIXES = (".tif", ".tiff")

#: `<genotype>/<animal>/<file>` and nothing else. **The depth is right and stays right**: every
#: image in the dataset sits at exactly this depth, and the 731 `.vsi` sidecar files sit two
#: levels deeper, out of its reach. Widening it would gain nothing and reach them.
DEPTH_PATTERN = "*/*/*"


def build_manifest(image_root: str | Path, pattern: str | None = None) -> list[str]:
    """Every image the run will process, discovered then recorded.

    Sorted so the array index to image mapping is stable — without that, a resubmitted task 47
    could process a different file than it did the first time.

    **One glob at the depth rule, then a suffix test** — rather than one glob per extension.
    A directory listing at depth 3 is the same cost either way, so a second pass over Lustre
    buys nothing, and the suffix test is where the two spellings and their case are decided in
    one place. ``pattern`` still overrides both, for a caller that wants a different tree.
    """
    root = Path(image_root)
    if pattern is not None:
        candidates = root.glob(pattern)
    else:
        candidates = (p for p in root.glob(DEPTH_PATTERN)
                      if p.suffix.lower() in IMAGE_SUFFIXES)
    found = sorted(str(p) for p in candidates
                   if not p.name.startswith(".") and "wselections" not in p.name)
    return found


def write_manifest(paths: list[str], pool_root: str | Path,
                   image_root: str | Path) -> Path:
    """Record the exact set processed, with each image's parsed provenance."""
    pool_root = Path(pool_root)
    pool_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in paths:
        meta = parse_metadata(path)
        entry = asdict(meta)
        entries.append(entry)
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_root": str(image_root),
        "n_images": len(paths),
        "images": entries,
    }
    out = pool_root / MANIFEST_NAME
    #: **Atomically, like every fragment.** This file is the run's audit record and the list
    #: every array task indexes into; a writer killed mid-write would leave a truncated JSON
    #: that `segment` then refuses to parse, taking the whole array down with it.
    _atomic_write_text(out, json.dumps(payload, indent=2))
    return out


def read_manifest(pool_root: str | Path) -> dict:
    """The whole recorded payload — the paths, the root they came from, and the count.

    Separate from :func:`load_manifest` because `segment` needs the root as well as the list:
    it checks the `--images` it was given against the one the manifest was built from, and a
    disagreement there is exactly the two-sources-for-one-fact defect this step exists to
    remove.
    """
    path = Path(pool_root) / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(pool_root: str | Path) -> list[str]:
    return [entry["path"] for entry in read_manifest(pool_root)["images"]]


def _same_root(a: str | Path, b: str | Path) -> bool:
    """Whether two spellings name the same directory, without resolving either.

    `normpath` and `normcase` only — **no `resolve()`**. Scratch and the RDM both reach the
    same tree through symlinks whose real paths differ per node, and `make_file_uid` hashes the
    path **as given**, so resolving here would refuse a run whose stored identifiers are
    perfectly consistent.
    """
    return (os.path.normcase(os.path.normpath(str(a)))
            == os.path.normcase(os.path.normpath(str(b))))


def fragment_exists(pool_root: str | Path, image_path: str) -> bool:
    """Whether this image already produced a fragment — the resume signal."""
    from .schema import make_file_uid

    try:
        uid = make_file_uid(str(image_path))
    except Exception:
        return False
    return (Path(pool_root) / "fragments" / f"{uid}.json").exists()


def segment_progress(pool_root: str | Path) -> dict:
    """What segment still owes: the manifest against the fragments actually on disk.

    > ### ⇒ AN UNDERSIZED ARRAY IS THE DANGEROUS ONE, BECAUSE IT DOES NOT FAIL
    >
    > `--array=0-500` on a 573-image manifest leaves 72 images unsegmented and **every task
    > reports success**. So does a task killed by walltime at index 400. Neither is visible
    > from any exit code, and the only thing that can see them is the manifest count compared
    > against what is on disk — which is what this does.

    ``in_pool`` is ``None`` rather than 0 before `merge` has ever run: *"no database yet"* and
    *"a database holding nothing"* are different facts and a status report must not conflate
    them.
    """
    root = Path(pool_root)
    try:
        payload = read_manifest(root)
    except FileNotFoundError:
        payload = None
    except ValueError as exc:
        return {"manifest": None, "error": f"{MANIFEST_NAME} could not be read: {exc}"}

    paths = [] if payload is None else [entry["path"] for entry in payload["images"]]
    missing = [p for p in paths if not fragment_exists(root, p)]

    frag_dir = root / "fragments"
    fragments = ([f for f in frag_dir.glob("*.json") if ".tmp." not in f.name]
                 if frag_dir.exists() else [])
    fail_dir = root / "failures"
    failures = sorted(f.name for f in fail_dir.glob("*.txt")) if fail_dir.exists() else []

    in_pool = None
    if (root / DB_NAME).exists():
        with Pool(root, read_only=True) as pool:
            in_pool = len(pool.files())

    return {
        "manifest": None if payload is None else len(paths),
        "image_root": None if payload is None else payload.get("image_root"),
        "fragments": len(fragments),
        "missing": missing,
        "in_pool": in_pool,
        "failures": failures,
    }


def _failure_name(image_path: str) -> str:
    """The stem of this image's failure file: its `file_uid`, or its basename if that fails.

    **The fallback is not defensive noise.** This runs inside the one `except` block whose job
    is to make sure a failure gets recorded; letting the identifier raise here would turn a
    recoverable one-image failure into a lost traceback, which is exactly what the block
    exists to prevent.
    """
    from .schema import make_file_uid

    try:
        return make_file_uid(str(image_path))
    except Exception:                              # noqa: BLE001 — see above
        return Path(image_path).name


def run_one(
    image_path: str,
    pool_root: str | Path,
    *,
    params: SegmentationParams,
    roles: ChannelRoles | dict,
    run_id: str,
    gpu: bool | None = None,
) -> tuple[bool, str]:
    """Segment one image, converting any failure into a recorded outcome.

    Returns ``(ok, message)`` and **never raises**. A corrupt file at index 400 of 700 must
    cost that one image, not the run.
    """
    meta = parse_metadata(image_path)
    notes = list(meta.warnings)
    try:
        result = segment_image(image_path, pool_root, roles=roles, params=params,
                               run_id=run_id, gpu=gpu)
        message = result.report()
        if notes:
            message += "  [" + "; ".join(notes) + "]"
        return True, message
    except Exception as exc:                       # noqa: BLE001 — isolation is the point
        failure = Path(pool_root) / "failures"
        failure.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).name
        #: > ### ⇒ NAMED BY `file_uid`, BECAUSE BASENAMES REPEAT ACROSS ANIMAL FOLDERS
        #: >
        #: > `failures/<basename>.txt` let two images with the same filename in different
        #: > animal folders overwrite each other's traceback — and this dataset is laid out
        #: > `<genotype>/<animal>/<file>` precisely so that filenames *can* repeat. The one
        #: > that failed second would be the only one you could read about.
        #: >
        #: > `make_file_uid` is `<stem>_<sha1(path)[:8]>`, so the name stays human-recognisable
        #: > in a directory listing while being unique across the study — the same identifier
        #: > the fragment and the cell arrays already carry, so a failure can be matched to the
        #: > rest of an image's output by name alone.
        (failure / f"{_failure_name(image_path)}.txt").write_text(
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"{image_path}\n\n{traceback.format_exc()}", encoding="utf-8")
        return False, f"FAILED {stem}: {type(exc).__name__}: {exc}"


def _roles_for_run(params: SegmentationParams, roles_text: str | None) -> dict[str, int]:
    """The channel roles to segment with — from the parameters file, or the flag, never both.

    Since 2026-08-15 the roles are fields on `SegmentationParams`, so a `segment` run has two
    possible sources for one fact. **A disagreement is refused**, in the same spirit as
    `load_params`' own four refusals: letting `--roles` win would record channels in
    `params_hash` that the run did not use — the `min_spur_um` defect in a new field — and
    letting the file win silently would ignore a flag the user deliberately passed.

    **Presence counts, not only index.** *"This image has no CD68"* against *"channel 1 is
    CD68"* is a real conflict with twelve export columns behind it, and a comparison over
    shared keys alone would pass it.

    **Why `--roles` now defaults to `None`.** It used to default to the shipped layout string,
    so the flag was *always* present — and a user whose file declared a non-standard layout
    and who passed no flag would have been refused on a conflict they never created. Absence
    now means *take the file's*, which is the only reading under which this check is safe.
    """
    from_file = {role: index for role, index in (
        ("nuclei", params.nuclei_channel),
        ("cell", params.cell_channel),
        ("cd68", params.cd68_channel),
        ("plaque", params.plaque_channel),
    ) if index is not None}

    if roles_text is None:
        return from_file

    given = {name.strip(): int(index)
             for name, _, index in (part.partition(":") for part in roles_text.split(","))}
    if given == from_file:
        return given

    rows = [f"    {role:<7} --roles {str(given.get(role, 'absent')):<8} "
            f"parameters file {from_file.get(role, 'absent')}"
            for role in sorted(set(given) | set(from_file))
            if given.get(role) != from_file.get(role)]
    raise ValueError(
        "--roles contradicts the parameters file. Refusing to guess — segmenting the wrong "
        "channel yields plausible but meaningless output in every column.\n"
        + "\n".join(rows)
        + "\n    Pass matching values, or omit --roles to use the parameters file."
    )


def manifest_main(argv: list[str] | None = None) -> int:
    """Glob once, record the list, print what the array bound has to be.

    > ### ⇒ THE COUNT IS PRINTED BECAUSE THE OFF-BY-ONE IS THE FAILURE
    >
    > **573 images means `--array=0-572`.** A hand-counted bound one too small leaves the last
    > images unsegmented and every task reporting success; one too large used to leave a task
    > raising `IndexError`. Both are removed by printing the number the program itself found,
    > and the bound is printed already formed so it can be copied rather than decremented.
    """
    ap = argparse.ArgumentParser(description=manifest_main.__doc__)
    ap.add_argument("--images", required=True, help="root directory of images")
    ap.add_argument("--pool", required=True, help="pool root directory")
    args = ap.parse_args(argv)

    paths = build_manifest(args.images)
    if not paths:
        print(f"no images under {args.images}", file=sys.stderr)
        return 2
    out = write_manifest(paths, args.pool, args.images)
    print(f"{len(paths)} image(s) under {args.images}")
    print(f"manifest {out}")
    print(f"--array=0-{len(paths) - 1}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, help="root directory of images")
    ap.add_argument("--pool", required=True, help="pool root directory")
    ap.add_argument("--params", required=True, help="parameters JSON")
    ap.add_argument("--run-id", default="batch")
    ap.add_argument("--index", type=int, default=None,
                    help="Slurm array index (0-based); omit to run every image serially")
    ap.add_argument("--resume", action="store_true",
                    help="skip images that already have a fragment")
    ap.add_argument("--force", action="store_true",
                    help="reprocess images that already have a fragment")
    ap.add_argument("--roles", default=None,
                    help="e.g. nuclei:0,cd68:1,cell:2,plaque:3. Omit to take the roles from "
                         "the parameters file, which has carried them since schema v5.")
    ap.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                    help="where nucleus detection runs. auto uses a GPU when torch reports "
                         "one. NOT a parameter: it is never hashed and never reaches "
                         "SegmentationParams, because the hash names the pool and a CPU run "
                         "and a GPU run of the same study belong in the same pool.")
    ap.add_argument("--threads", type=int, default=None,
                    help="cores to use inside this task. Defaults to SLURM_CPUS_PER_TASK, "
                         "and to 1 when that is unset. NOT a parameter, for exactly the "
                         "reason --device is not: it is never hashed and never reaches "
                         "SegmentationParams, so a 4-thread and a 16-thread run of the same "
                         "study land in the same pool. It is printed at startup and appended "
                         "to run_id instead.")
    args = ap.parse_args(argv)

    #: > ### ⇒ A SEMI-AUTO POOL IS REFUSED, AND ONLY REFUSED
    #: >
    #: > `pool_dir_name` and `ORIGINS` exist to keep the two apart — *"semi-auto and batch never
    #: > share one"* — and nothing consulted either, so `--pool` took any path and a batch run
    #: > could be aimed at a pool holding a human's work. What lands there is not recoverable by
    #: > re-running: `merge_fragments` preserves review decisions, but the cell set itself is
    #: > overwritten.
    #: >
    #: > **The user's decision: refuse, and leave `--pool` a path the operator types.** Deriving
    #: > `pool_batch_<phase_a_hash>` here was declined — it would make the destination something
    #: > the program chooses and the submit script cannot name.
    #: >
    #: > A name that declares *nothing* passes, which is the ordinary case; only a name that
    #: > positively says `semiauto` is refused. See `store.pool_origin`.
    origin = pool_origin(args.pool)
    if origin == "semiauto":
        print(f"{args.pool} is a SEMI-AUTO pool — its directory name says so.\n"
              "`segment` writes batch output and will not write it into one: semi-auto and "
              "batch keep separate pools so that a human's reviewed cell set is never "
              "overwritten by an array task.\n"
              "Point --pool at a batch pool, or at a new directory.", file=sys.stderr)
        return 6

    params, meta = load_params(args.params)
    try:
        roles = _roles_for_run(params, args.roles)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 4

    try:
        payload = read_manifest(args.pool)
    except FileNotFoundError:
        print(f"no {MANIFEST_NAME} in {args.pool}.\n"
              f"Run `prestomorph manifest --images {args.images} --pool {args.pool}` first — "
              "it globs once, records the list every task indexes into, and prints the "
              "--array bound.", file=sys.stderr)
        return 5
    except ValueError as exc:
        print(f"{Path(args.pool) / MANIFEST_NAME} could not be read: {exc}", file=sys.stderr)
        return 5

    #: **Refused rather than noted.** The manifest carries the root it was globbed from, so a
    #: `--images` that names a different tree is an operator error with a silent outcome: the
    #: tasks would segment the manifest's images while the command line says otherwise. This is
    #: the same disagreement `_roles_for_run` refuses one field over.
    recorded_root = payload.get("image_root")
    if recorded_root and not _same_root(recorded_root, args.images):
        print(f"--images does not match the manifest in {args.pool}.\n"
              f"    manifest  {recorded_root}\n"
              f"    --images  {args.images}\n"
              "Refusing to guess: the tasks would segment the manifest's images whatever the "
              "command line says. Re-run `prestomorph manifest` for the tree you mean.",
              file=sys.stderr)
        return 5

    paths = [entry["path"] for entry in payload["images"]]
    if not paths:
        print(f"no images in {Path(args.pool) / MANIFEST_NAME}", file=sys.stderr)
        return 2

    if args.index is None:
        selected = paths
    elif args.index < 0:
        print(f"index {args.index} is not a 0-based array index", file=sys.stderr)
        return 2
    elif args.index >= len(paths):
        #: > ### ⇒ EXIT 0, AND IT IS REQUIRED RATHER THAN COSMETIC
        #: >
        #: > This raised a bare `IndexError`. An oversized `--array` then produced tasks that
        #: > exit non-zero, and the second job waits on the first with `--dependency=afterok` —
        #: > **which would hold it in the queue forever.** A task with no image is not a
        #: > failure; it is an array bound one larger than the manifest, and saying so on
        #: > stdout and leaving is the whole of the correct behaviour.
        print(f"index {args.index} is past the end of a {len(paths)}-image manifest, "
              "nothing to do")
        return 0
    else:
        selected = [paths[args.index]]

    # The user chose friction over silence here: neither skipping nor reprocessing happens
    # by default, because both are wrong half the time and neither announces itself.
    already = [p for p in selected if fragment_exists(args.pool, p)]
    if already and not (args.resume or args.force):
        print(f"{len(already)} of {len(selected)} selected image(s) already have a "
              f"fragment in {args.pool}.\n"
              "Refusing to guess. Pass --resume to skip them, or --force to reprocess.\n"
              "First few:\n  " + "\n  ".join(Path(p).name for p in already[:5]),
              file=sys.stderr)
        return 3
    if args.resume:
        selected = [p for p in selected if not fragment_exists(args.pool, p)]

    #: > ### ⇒ THE THREAD COUNT RIDES BESIDE THE DEVICE, FOR THE SAME REASON
    #: >
    #: > It cannot go in `SegmentationParams` either — the hash names the pool, so a 4-thread
    #: > and a 16-thread run of one study would land in two. `run_id` is free text, stored per
    #: > file and reaching `pool.csv`, so it is the same provenance-shaped place.
    #: >
    #: > **And it is printed at startup**, because the failure this exists to fix — 16 cores
    #: > allocated and one used — is invisible from the output otherwise, exactly as an unused
    #: > GPU partition was.

    #: > ### ⇒ AND IT IS RESOLVED FIRST, BECAUSE THE DEVICE QUESTION IMPORTS TORCH
    #: >
    #: > `OMP_NUM_THREADS` has to be set **before** torch loads — OpenMP reads it once, at load
    #: > time, and nothing later undoes a pool that is already built. That is A2's whole point,
    #: > and this block sat *below* the device block until 2026-09-06, which quietly broke it:
    #: > `resolve_device(None)` calls `gpu_available`, and **that imports torch**. On the
    #: > default `--device auto` — which is every submit script — the import had therefore
    #: > already happened by the time the variable was set.
    #: >
    #: > Nothing downstream reads `threads` before `run_id`, and `configure_torch` needs no
    #: > answer from `resolve_device`, so the fix is the order and nothing else. **Do not move
    #: > this back below the device block.** `test_component04.py` asserts the ordering by
    #: > watching when the variable is set relative to the import.
    threads = cpu.use(cpu.resolve_threads(args.threads))
    cpu.configure_torch(threads)

    #: **Resolved once, before the loop.** `gpu_available` imports torch, and doing that per
    #: image would pay the import on every file for an answer that cannot change inside one
    #: task.
    gpu = {"auto": None, "gpu": True, "cpu": False}[args.device]
    gpu = resolve_device(gpu)
    device = describe_device(gpu)
    #: > ### ⇒ THE DEVICE IS RECORDED IN `run_id`, WHICH IS THE ONE FREE-TEXT FIELD PER FILE
    #: >
    #: > It cannot go in `SegmentationParams` — that is hashed and the hash names the pool, so
    #: > a hardware difference would split one study across two pools, and the user ruled that
    #: > out explicitly. `run_id` is stored on every `FileRecord`, reaches `pool.csv`, and
    #: > nothing parses it, so it is exactly the provenance-shaped place this belongs.
    #: >
    #: > **And it is printed at startup**, because the failure this item exists to fix — a GPU
    #: > partition paid for and not used — is invisible from the output otherwise.
    run_id = f"{args.run_id}:{device}:{cpu.describe_threads(threads)}"
    print(f"params {meta.get('params_hash')} | {len(selected)} image(s) | "
          f"run {run_id} | nucleus detection on {device} | {threads} thread(s)")
    failures = 0
    for path in selected:
        ok, message = run_one(path, args.pool, params=params, roles=roles,
                              run_id=run_id, gpu=gpu)
        failures += not ok
        print(("  " if ok else "!! ") + message, flush=True)

    if failures:
        print(f"\n{failures} image(s) failed; see {Path(args.pool) / 'failures'}",
              file=sys.stderr)
    return 1 if failures and len(selected) == 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
