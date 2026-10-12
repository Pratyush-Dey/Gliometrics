"""Component 8 — the single entry point that wires Phase A -> B -> C.

::

    prestomorph manifest --images R --pool P
    prestomorph segment --images R --pool P --params F [--index N] [--resume]
    prestomorph merge   --pool P
    prestomorph review  --pool P --params F
    prestomorph measure-plan --pool P --params F [--include-unreviewed]
    prestomorph measure --pool P --params F [--index N] [--include-unreviewed]
    prestomorph merge-measurements --pool P [--params-hash H]
    prestomorph export  --pool P --out D --params F [--params-hash H]
    prestomorph status  --pool P [--params F]
    prestomorph masks   --pool P --out D [--params-hash H] [--include-unreviewed]

Each subcommand is one Slurm job. `segment` and `measure` take `--index` so an array task
can claim one image; `manifest`, `merge` and `export` run once.

**`manifest` exists so the array bound is a measured number rather than a counted one.** It
globs once, writes `manifest.json` atomically, and prints `--array=0-N` already formed;
`segment` then *reads* that file instead of re-globbing per task. See `batch.build_manifest`
for why one glob at depth 3 is the whole discovery rule.

**`masks` sits outside that chain on purpose.** A cell's mask is written by Phase A and altered
by no later phase, so it depends on no parameter set and is available the moment cells are in the
pool — before `measure`, and without the `--params` that `export` requires. See
`export.export_cell_masks` and `EXPORT_SPEC.md` §43.

**`segment` delegates to `batch.main()` rather than reimplementing it.** That driver is built,
tested, failure-isolated and manifest-backed; a second code path for the same job is the most
likely way the batch and interactive routes would silently diverge, which is the whole failure
`params.py`'s refusals exist to prevent.

**`measure-plan` is to `measure` what `manifest` is to `segment`**, and deliberately so: a
prepare step pins the image list and prints the `--array` bound, and the array reads that file.
One shape learned once. It exists because `measure`'s queue used to be *cells not yet
measured* — a list that shrinks as tasks finish, so a task starting late indexed into a
different image and the array still exited 0.

**`merge-measurements` is its own command and not part of `merge`** — the user's decision.
`merge` runs after `segment` and before `review`; this runs after `measure` and before
`export`. One command doing both would re-parse and re-insert every segmentation fragment in
the pool each time measurements were folded in. The names are deliberately adjacent so the
pairing is visible in a submit script.

**`status` is the backstop both arrays need.** A stable index stops drift; it does not tell
you that task 400 died of walltime, and an **undersized** array is worse still because nothing
fails — the last images simply never run and every task exits 0. It reports both phases from
the only things that can see the gap: the manifest count against the fragments on disk, and
`pending_measurement` at this hash.

**`measure` needs no `--resume` flag.** Its queue is "cells with no measurement at this
parameter hash", so re-running simply finds less to do, and changing a parameter re-queues
everything automatically. `--resume` is accepted and ignored on `segment` only, where the
existing driver defines it.

**`review` and `interactive` are the two interactive commands**, and they reach `ui/` through
**one** import statement — `_ui_entry_points`. `tests/test_ui_boundary.py` asserts that this
file holds exactly one crossing into the interface and that it sits inside a function; a second
`from .ui import ...` anywhere here, even a lazy one, would be a second door to keep shut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cpu

__all__ = ["main"]


def _params_from(path: str):
    """The parameters only — `load_params` also returns the surrounding metadata."""
    from .params import load_params
    params, _metadata = load_params(path)
    return params


def _cmd_segment(args) -> int:
    from . import batch
    argv = ["--images", args.images, "--pool", args.pool, "--params", args.params,
            "--run-id", args.run_id]
    # Only forward the flag when it was actually given. Passing it unconditionally would
    # re-manufacture the hard-coded default one layer down and refuse on a conflict the
    # user never created — see `batch._roles_for_run`.
    if args.roles is not None:
        argv += ["--roles", args.roles]
    if args.index is not None:
        argv += ["--index", str(args.index)]
    if args.resume:
        argv += ["--resume"]
    if args.force:
        argv += ["--force"]
    if args.device != "auto":
        argv += ["--device", args.device]
    #: Forwarded rather than resolved here: `batch.main` owns the startup line and the
    #: `run_id` the count is appended to, and resolving twice on two sides of this argv is
    #: exactly how a printed number stops matching the pools that were actually used.
    if args.threads is not None:
        argv += ["--threads", str(args.threads)]
    return batch.main(argv)


def _cmd_manifest(args) -> int:
    from . import batch
    return batch.manifest_main(["--images", args.images, "--pool", args.pool])


def _cmd_merge(args) -> int:
    from .store import merge_fragments
    r = merge_fragments(args.pool)
    print(f"merged {r.fragments_read} fragment(s): "
          f"{r.files_inserted} file(s) inserted, {r.files_updated} updated; "
          f"{r.cells_inserted} cell(s) inserted, {r.cells_updated} updated; "
          f"{r.reviews_preserved} review decision(s) preserved")
    return 0


def _cmd_measure_plan(args) -> int:
    from .phasec import build_measure_plan, measure_plan_path

    params = _params_from(args.params)
    entries = build_measure_plan(args.pool, params.hash(),
                                 include_unreviewed=args.include_unreviewed)
    print(f"{len(entries)} image(s) with measurable cells at {params.hash()[:8]}")
    print(f"plan {measure_plan_path(args.pool, params.hash())}")
    if entries:
        print(f"--array=0-{len(entries) - 1}")
    else:
        print("NOTE: nothing to measure. Cells are written `candidate` and only review "
              "confirms them, so a pool that has not been reviewed plans nothing here. "
              "Pass --include-unreviewed to plan the candidates as they stand.")
    return 0


def _cmd_measure(args) -> int:
    from .phasec import run_phase_c
    _use_threads(args)
    params = _params_from(args.params)
    roles = _parse_roles(args.roles) if args.roles else None
    try:
        got = run_phase_c(args.pool, params, index=args.index,
                          include_unreviewed=args.include_unreviewed, roles=roles)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 5
    if args.include_unreviewed:
        # Not a warning in a log nobody reads later — the stamp travels to the spreadsheet.
        print("NOTE: measuring UNREVIEWED cells. Every row produced is stamped "
              "reviewed=false and the export will say so.")
    print(f"measured {got['ok']} cell(s) across {got['images']} image(s), "
          f"{got['failed']} failed")
    if got.get("failed_images"):
        print(f"NOTE: {got['failed_images']} image(s) could not be measured at all and were "
              f"recorded under {args.pool}/measurement_failures. They wrote no fragment, so "
              "they are still in the queue and a resubmit will retry them.")
    print("Fragments written. Run `prestomorph merge-measurements --pool "
          f"{args.pool}` before exporting.")
    return 0


def _cmd_merge_measurements(args) -> int:
    from .store import merge_measurement_fragments
    r = merge_measurement_fragments(args.pool, params_hash=args.params_hash)
    print(f"merged {r.fragments_read} measurement fragment(s): "
          f"{r.cells_written} cell row(s), {r.images_written} image row(s); "
          f"{r.fragments_deleted} fragment(s) deleted")
    if r.fragments_read and not r.fragments_deleted:
        print("NOTE: nothing was deleted. The rows are committed, so re-merging is harmless "
              "— but check the pool is writable.")
    return 0


def _cmd_status(args) -> int:
    """What each phase still owes. **Exit 0 either way, and that is deliberate.**

    Outstanding work is not an error — it is the ordinary state between two jobs — and a
    non-zero exit here would fail a submit script that was doing exactly the right thing. The
    report is what carries the signal, so it names the images rather than only counting them.
    """
    from . import batch
    from .store import DB_NAME, Pool

    segment = batch.segment_progress(args.pool)
    print(f"pool  {args.pool}")
    if segment.get("error"):
        print(f"  SEGMENT   {segment['error']}")
    elif segment["manifest"] is None:
        print(f"  SEGMENT   no manifest — run `prestomorph manifest` first. "
              f"{segment['fragments']} fragment(s) on disk")
    else:
        n = segment["manifest"]
        done = n - len(segment["missing"])
        print(f"  SEGMENT   {done}/{n} image(s) have a fragment"
              + ("" if not segment["missing"] else
                 f"  — {len(segment['missing'])} OUTSTANDING"))
        for path in segment["missing"][:10]:
            print(f"              missing  {path}")
        if len(segment["missing"]) > 10:
            print(f"              ... and {len(segment['missing']) - 10} more")
    if segment.get("failures"):
        print(f"  FAILURES  {len(segment['failures'])} image(s) recorded under "
              f"{args.pool}/failures")

    if segment.get("in_pool") is None:
        print("  MERGE     no pool database yet — run `prestomorph merge`")
    else:
        print(f"  MERGE     {segment['in_pool']} image(s) in the database, "
              f"{segment['fragments']} fragment(s) still on disk")

    if args.params is None:
        print("  MEASURE   pass --params to report this phase")
        return 0
    if not (Path(args.pool) / DB_NAME).exists():
        print("  MEASURE   nothing to report until the pool has been merged")
        return 0

    params = _params_from(args.params)
    with Pool(args.pool, read_only=True) as pool:
        progress = pool.measurement_progress(
            params.hash(), include_unreviewed=args.include_unreviewed)
    print(f"  MEASURE   {progress['images'] - progress['images_outstanding']}/"
          f"{progress['images']} image(s) measured at {params.hash()[:8]}; "
          f"{progress['cells_measured']}/{progress['cells']} cell(s)"
          + ("" if not progress["images_outstanding"] else
             f"  — {progress['images_outstanding']} OUTSTANDING"))
    for entry in progress["outstanding"][:10]:
        print(f"              {entry['n_pending']:>5} cell(s)  {entry['source_path']}")
    if len(progress["outstanding"]) > 10:
        print(f"              ... and {len(progress['outstanding']) - 10} more")

    from .phasec import MEASUREMENT_FAILURES_DIR, measure_plan_path

    plan = measure_plan_path(args.pool, params.hash())
    if plan.exists():
        pinned = json.loads(plan.read_text(encoding="utf-8"))["n_images"]
        print(f"  PLAN      {pinned} image(s) pinned — --array=0-{max(pinned - 1, 0)}"
              + ("" if pinned == progress["images"] else
                 f"  (the pool now holds {progress['images']}; re-run `measure-plan` "
                 "if review has moved on)"))
    else:
        print("  PLAN      none pinned — run `prestomorph measure-plan` before an array")

    fail_dir = Path(args.pool) / MEASUREMENT_FAILURES_DIR / params.hash()
    recorded = sorted(fail_dir.glob("*.txt")) if fail_dir.exists() else []
    if recorded:
        print(f"  FAILURES  {len(recorded)} image(s) could not be measured; see {fail_dir}")
    return 0


def _cmd_export(args) -> int:
    from .export import export_pool
    from .store import Pool
    params = _params_from(args.params) if args.params else None
    # `--params` is required as of 2026-08-12 (item 6), but the refusal is raised by
    # `export_pool` rather than by argparse: argparse would say only "the following arguments
    # are required", where the real error names the pool and says the values are recoverable
    # from it. Converted to SystemExit so a person gets the message and not a traceback.
    try:
        with Pool(args.pool) as pool:
            got = export_pool(pool, args.out, params_hash=args.params_hash, params=params,
                              parameters_path=args.params)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {got['folder']} — {got['n_cells']} cell(s), {got['n_plaques']} "
          f"plaque(s), {got['n_images']} image(s) at parameter set "
          f"{got['params_hash'][:8]}")

    if getattr(args, "with_masks", False):
        #: > ### ⇒ THE TWO MASK ROUTES WROTE TO TWO DIFFERENT PLACES — patch item C6, 2026-09-03
        #: >
        #: > This nested them at `<export folder>/masks/` while the standalone `masks` command
        #: > put them at `<--out>/masks_<date>/`. Same layout, same content, different locations
        #: > — so anyone who tried both ended up with two copies of a 20,000-file dataset.
        #: >
        #: > **`--out`'s parent, which after C2 is the output root.** Item 8 emits
        #: > `--out <output_root>/exports`, so this resolves to `<output_root>/masks/` and
        #: > `export_cell_masks` appends its own `masks_<date>` — byte-identical to where the
        #: > standalone command lands.
        #: >
        #: > **The trade, stated because it is real:** for any other `--out` the masks land in
        #: > that folder's *sibling* rather than inside it. Predictable, and the alternative —
        #: > nesting at `<--out>/masks/` — is the divergence this item exists to close.
        from .export import MASKS_DIR, export_cell_masks
        with Pool(args.pool) as pool:
            masks = export_cell_masks(pool, Path(args.out).parent / MASKS_DIR,
                                      params_hash=got["params_hash"])
        print(f"wrote {masks['n_masks']} mask(s) and their skeletons to {masks['folder']}")
        if masks.get("missing"):
            print(f"NOTE: {len(masks['missing'])} cell(s) had no readable arrays and were "
                  "named in the manifest rather than skipped in silence.")
    columns = got["columns"]
    for name, value in (("cd68_threshold", columns.cd68_threshold),
                        ("plaque_threshold", columns.plaque_threshold)):
        if value is None:
            print(f"NOTE: {name} is UNSET. Every measure behind it is blank with a "
                  "status, never a zero — see export_summary.csv.")
    return 0


def _cmd_masks(args) -> int:
    """Per-cell mask TIFFs for Fiji and friends.

    **Its own command rather than a flag on `export`.** A mask is fixed by Phase A and altered
    by no later phase, so it neither depends on a parameter set nor changes when a run is
    re-measured — and `export` refuses without `--params`, which would make the masks wait on a
    threshold choice that has nothing to do with them. See `export.export_cell_masks`.

    **The empty case is a message, not a silent success.** `PoolWriter` writes every cell
    `CANDIDATE` and only review promotes it, so a batch pool that has never been reviewed
    selects nothing — and an empty folder plus exit 0 is exactly what that looks like from
    outside. The same seam `ui/screen_save.apply_decisions` was written to close.
    """
    from .export import export_cell_masks
    from .store import Pool
    with Pool(args.pool) as pool:
        got = export_cell_masks(pool, args.out, params_hash=args.params_hash,
                                include_unreviewed=args.include_unreviewed)
    print(f"wrote {got['folder']} — {got['n_masks']} mask(s) across "
          f"{got['n_images']} image(s)")
    if args.include_unreviewed:
        print("NOTE: including UNREVIEWED cells. manifest.csv carries each cell's status, "
              "so an unreviewed mask can still be told from a confirmed one.")
    if got["n_missing"]:
        print(f"NOTE: {got['n_missing']} selected cell(s) had no stored mask and were "
              "skipped: " + ", ".join(got["missing"][:5])
              + ("..." if got["n_missing"] > 5 else "")
              + ". Run `prestomorph merge` first, or check pool integrity.")
    if not got["n_masks"]:
        print("NOTE: no cells matched. Cells are written as `candidate` and only review "
              "confirms them, so a pool that has not been reviewed selects nothing here. "
              "Pass --include-unreviewed to export the candidates as they stand.")
    return 0


def _ui_entry_points():
    """**The one crossing into `ui/`**, and the boundary test asserts it is the only one.

    > ### THE IMPORT IS INSIDE A FUNCTION, AND THAT IS THE WHOLE POINT
    >
    > §2's first constraint is *the UI calls the pipeline, and the pipeline never calls the
    > UI* — stated precisely at `COMPONENT_03_PLAN.md` §4a.7 as **`napari.run()` must be
    > unreachable from anything Phase A executes.** PrestoCell's failure was not calling it;
    > it was calling it from *inside a pipeline function*, so the pipeline could not run
    > without a human.
    >
    > **A command-line dispatcher is not Phase A.** `prestomorph segment` on a Slurm node
    > never reaches this line, and `tests/test_ui_boundary.py` proves it two ways: the import
    > must sit inside a function (asserted on this file's AST), and importing every headless
    > module must still pull in no GUI at all (asserted in a clean subprocess).
    >
    > **One statement for both interactive commands**, rather than one each. The test asserts
    > *exactly one* crossing precisely so that the exemption cannot quietly widen — and two
    > lazy imports would be two doors, not one door used twice.
    """
    from .ui import app, review_pool

    return app, review_pool


def _cmd_review(args) -> int:
    """Component 5's pool-backed route. **A driver over screens that were built for it.**

    `screen_triage` takes a list of `ReviewCell` rather than a `Pool`, and says in as many
    words that *"`review --pool` builds them from `CellRecord`s"*; `screen_edit_cell` keeps
    everything non-Qt in `editing.py` because both routes need it. What was missing was the
    driver — see `ui/review_pool.py`.

    **`--params` is required, and the reason is the soma centroid.** Editing a mask must
    rebuild it, it is computed from the run's six soma settings, and the pool stores only
    their hash. Skipping the rebuild would leave a stored centroid pointing at a mask that
    changed underneath it, corrupting the plaque-distance measures for exactly the cells a
    human worked hardest on.
    """
    _preload_detector(_use_threads(args))
    _app, review_pool = _ui_entry_points()
    return review_pool.main(args.pool, args.params)


def _cmd_interactive(args) -> int:
    """Open the semi-automatic flow. **The one place this package crosses into `ui/`.**

    > ### THE IMPORT IS INSIDE THIS FUNCTION, AND THAT IS THE WHOLE POINT
    >
    > §2's first constraint is *the UI calls the pipeline, and the pipeline never calls the UI* —
    > stated precisely at `COMPONENT_03_PLAN.md` §4a.7 as **`napari.run()` must be unreachable
    > from anything Phase A executes.** PrestoCell's failure was not calling it; it was calling
    > it from *inside a pipeline function*, so the pipeline could not run without a human.
    >
    > **A command-line dispatcher is not Phase A.** `prestomorph segment` on a Slurm node never
    > reaches this line, and `tests/test_ui_boundary.py` proves it two ways: the import must sit
    > inside a function (asserted on this file's AST), and importing every headless module must
    > still pull in no GUI at all (asserted in a clean subprocess).
    >
    > **Nothing else outside `ui/` may do this**, and the test names this file as the only
    > exemption rather than allowing the pattern generally.
    """
    _preload_detector(_use_threads(args))
    app, _review_pool = _ui_entry_points()
    return app.main(image=args.image)


def _preload_detector(threads: int = 1) -> None:
    """Import torch **before Qt**, or item 4 cannot run at all on Windows.

    > ### ⇒ MEASURED, AND IT IS THE ORDER THAT MATTERS — NOTHING ELSE
    >
    > ```
    >   import qtpy.QtWidgets ; import torch   ->  OSError [WinError 1114]
    >                                              c10.dll initialisation failed
    >   import torch ; import qtpy.QtWidgets   ->  both fine
    > ```
    >
    > **This is why `segment` has always worked and the interactive route could not.** Batch
    > never imports Qt, so torch loads first by default; the semi-automatic flow builds its
    > window and only then reaches `detect_nuclei`, by which time the conflict is locked in.
    > Nucleus detection is item 4 — *"this **is** the expensive step"* and the one the whole
    > component exists to supervise — so the failure takes the flow with it.
    >
    > **Ruled out before this was believed:** it is not memory (a 115 MB crop fails exactly like
    > a 741 MB field), not background execution, not `prestomorph.pipeline` being imported, and
    > not scipy or scikit-image having run. `segment_image` imports cellpose happily from a
    > process that never touched Qt.
    >
    > **Best-effort on purpose.** napari is an optional dependency and so is cellpose; a machine
    > without either must still reach the screens, and a user who never gets to item 4 must not
    > be stopped at startup by an import they do not need.

    > ### ⇒ AND IT IS WHERE THE THREAD COUNT IS APPLIED, BECAUSE THE ORDER IS THE SAME ORDER
    >
    > `OMP_NUM_THREADS` has to be set **before** torch loads — OpenMP reads it once, and a
    > `set_num_threads` afterwards does not undo a pool that is already built. This function
    > already exists to control exactly when torch is imported, so it is the one place in the
    > interactive route where *before the import* is a thing that can be said at all.
    >
    > Left alone, torch sizes its pool from the node's visible CPUs rather than from the
    > allocation, and **cellpose runs on CPU on both routes now**. `cpu.configure_torch` does
    > both halves and swallows the import failure the same way this did.
    """
    cpu.configure_torch(threads)


#: `--threads`' help, written once because four subcommands carry the flag. The wording
#: deliberately mirrors `--device`'s, because it is the same decision with the same reason.
THREADS_HELP = ("cores to use inside this task. Defaults to SLURM_CPUS_PER_TASK, and to 1 "
                "when that is unset — so a run outside Slurm is serial and says so rather "
                "than deciding for itself how much of a shared machine to take. NOT a "
                "parameter: never hashed and never on SegmentationParams, exactly as "
                "--device is not, so a 4-thread and a 16-thread run of one study land in the "
                "same pool. Printed at startup, and on `segment` appended to run_id too.")


def _add_threads(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threads", type=int, default=None, help=THREADS_HELP)


def _use_threads(args) -> int:
    """Fix the count for this process and say what it is. `HANDOFF_PATCH_CPU.md` §3 A3.

    **`segment` does not call this** — `batch.main` owns its own startup line and folds the
    count into `run_id` there, so calling it here would print the number twice and fix it
    from the wrong side of the argv it builds.
    """
    threads = cpu.use(cpu.resolve_threads(getattr(args, "threads", None)))
    print(f"{threads} thread(s)")
    return threads


def _parse_roles(text: str) -> dict[str, int]:
    out = {}
    for part in text.split(","):
        name, _, index = part.partition(":")
        out[name.strip()] = int(index)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="prestomorph", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    man = sub.add_parser(
        "manifest",
        help="glob the image tree once, record it, and print the --array bound")
    man.add_argument("--images", required=True)
    man.add_argument("--pool", required=True)
    man.set_defaults(func=_cmd_manifest)

    seg = sub.add_parser("segment", help="Phase A: image in, candidate cells in the pool")
    seg.add_argument("--images", required=True)
    seg.add_argument("--pool", required=True)
    seg.add_argument("--params", required=True)
    seg.add_argument("--run-id", default="batch")
    seg.add_argument("--index", type=int, default=None,
                     help="Slurm array index; one image per task")
    seg.add_argument("--resume", action="store_true")
    seg.add_argument("--force", action="store_true")
    seg.add_argument("--roles", default=None,
                     help="e.g. nuclei:0,cd68:1,cell:2,plaque:3. Omit to take the roles "
                          "from the parameters file, which has carried them since schema "
                          "v5. Passing values that contradict the file is refused.")
    seg.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                     help="where nucleus detection runs. `auto` uses a GPU when torch "
                          "reports one, which is what a GPU partition is being paid for. "
                          "**Never hashed**: the device is recorded in run_id and printed "
                          "at startup, and it does not reach SegmentationParams, because "
                          "the hash names the pool and a hardware difference must not split "
                          "one study across two.")
    _add_threads(seg)
    seg.set_defaults(func=_cmd_segment)

    mrg = sub.add_parser("merge", help="fold Phase A fragments into the pool database")
    mrg.add_argument("--pool", required=True)
    mrg.set_defaults(func=_cmd_merge)

    plan = sub.add_parser(
        "measure-plan",
        help="pin the image list a measure array indexes into, and print the --array bound")
    plan.add_argument("--pool", required=True)
    plan.add_argument("--params", required=True)
    plan.add_argument("--include-unreviewed", action="store_true",
                      help="plan the candidates too. Must match the flag `measure` is then "
                           "given, or the array is bounded for a different set of images "
                           "than the tasks will look for.")
    plan.set_defaults(func=_cmd_measure_plan)

    mea = sub.add_parser("measure", help="Phase C: measure confirmed cells. Resumable.")
    mea.add_argument("--pool", required=True)
    mea.add_argument("--params", required=True)
    mea.add_argument("--index", type=int, default=None,
                     help="Slurm array index into the file written by `measure-plan`; one "
                          "image per task. Omit to measure everything outstanding serially, "
                          "which needs no plan.")
    mea.add_argument("--roles", default=None)
    mea.add_argument("--include-unreviewed", action="store_true",
                     help="measure cells that have NOT been through Phase B review. "
                          "Every resulting row is stamped reviewed=false and the export "
                          "carries that stamp, so unreviewed output can never be read as "
                          "reviewed.")
    _add_threads(mea)
    mea.set_defaults(func=_cmd_measure)

    mrgm = sub.add_parser(
        "merge-measurements",
        help="fold Phase C fragments into the pool database, then delete them")
    mrgm.add_argument("--pool", required=True)
    mrgm.add_argument("--params-hash", default=None,
                      help="narrow to one parameter set. Omit to merge every set present, "
                           "which is what a pool holding a re-run beside the run it replaced "
                           "needs.")
    mrgm.set_defaults(func=_cmd_merge_measurements)

    exp = sub.add_parser("export", help="one run's CSVs from stored measurements")
    exp.add_argument("--pool", required=True)
    exp.add_argument("--out", required=True)
    exp.add_argument("--params", default=None,
                     help="REQUIRED. The run's parameters file. Copied verbatim into the "
                          "export as parameters.json, and the source of the thresholds the "
                          "curve column names carry. Without it the export refuses: it would "
                          "otherwise fall back to defaults and state them in "
                          "export_summary.csv as though they were this run's values.")
    exp.add_argument("--params-hash", default=None,
                     help="which parameter set to export. Required when the pool holds "
                          "more than one: a re-run gets its own export rather than being "
                          "merged into another run's.")
    exp.add_argument("--with-masks", action="store_true",
                     help="also write per-cell mask TIFFs and both skeletons as SWC into a "
                          "masks/ subfolder, mirroring the input tree. OFF by default and "
                          "deliberately so: masks depend only on the pool, not on any export "
                          "choice, so a re-export to fix a column would otherwise rewrite "
                          "every TIFF in the run. The standalone `masks` command writes the "
                          "same layout.")
    exp.set_defaults(func=_cmd_export)

    sts = sub.add_parser(
        "status", help="what each phase still owes — the backstop for a lost or undersized "
                       "array")
    sts.add_argument("--pool", required=True)
    sts.add_argument("--params", default=None,
                     help="the run's parameters file. Without it the measure phase cannot be "
                          "reported: its queue is keyed by the parameter hash, and guessing "
                          "one would report on a run nobody made.")
    sts.add_argument("--include-unreviewed", action="store_true",
                     help="count candidates as measurable, matching `measure` run the same "
                          "way.")
    sts.set_defaults(func=_cmd_status)

    msk = sub.add_parser("masks",
                         help="per-cell binary mask TIFFs, for Fiji / MATLAB / Imaris")
    msk.add_argument("--pool", required=True)
    msk.add_argument("--out", required=True)
    msk.add_argument("--params-hash", default=None,
                     help="narrow to the cells measured at this parameter set — exactly the "
                          "rows of that run's cells.csv, so the mask set and the spreadsheet "
                          "match one-to-one. Omit to export every kept cell.")
    msk.add_argument("--include-unreviewed", action="store_true",
                     help="also export cells that have NOT been through review. Every row of "
                          "manifest.csv carries its cell's status, so the distinction "
                          "survives into the folder.")
    msk.set_defaults(func=_cmd_masks)

    #: **BUILT 2026-09-02.** This was a stub naming §9.4's storage foundation as what it still
    #: needed; that foundation landed as `Pool.replace_cell_mask` — and only that method, cell
    #: splitting in batch review being out of scope, which is what removed `create_cell` and
    #: `delete_cell`. The screens were always built; what was missing was the driver.
    rev = sub.add_parser(
        "review", help="Phase B: confirm cells in a pool. Interactive, and resumable.")
    rev.add_argument("--pool", required=True)
    rev.add_argument("--params", required=True,
                     help="REQUIRED, and the soma centroid is why. Editing a mask must "
                          "rebuild it; it is computed from the run's six soma settings and "
                          "the pool stores only their hash. Without them an edited cell "
                          "would keep a centroid pointing at a mask that changed underneath "
                          "it, corrupting the plaque-distance measures for exactly the cells "
                          "a human worked hardest on.")
    _add_threads(rev)
    rev.set_defaults(func=_cmd_review)

    interactive = sub.add_parser(
        "interactive", help="Component 3: the semi-automatic flow, one image")
    interactive.add_argument("--image", required=False,
                             help="pre-fill the file box. Optional: the first screen asks for "
                                  "both roots and the file, and nothing is remembered between "
                                  "runs.")
    _add_threads(interactive)
    interactive.set_defaults(func=_cmd_interactive)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
