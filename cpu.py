"""The cores inside one task. `HANDOFF_PATCH_CPU.md` §3 A3, §3 A5, §5 C2a.

The program was built for multi-*process* parallelism — one Slurm array task per image — and
did nothing at all with the cores inside a task. Nothing called `ThreadPoolExecutor`, nothing
set `OMP_NUM_THREADS`, and torch sized its own pool from the node's visible CPUs rather than
from the allocation. This module is the one place that decides how many threads a task gets,
and the two shapes every threaded loop in the package is allowed to take.

**The one rule everything here rests on:** only work that is genuinely independent is
parallelised, and only where the output is bit-identical. Two shapes qualify — *strictly
per-plane*, where each output z-slice depends only on the same input z-slice, and *strictly
per-cell*, where each result depends only on that cell's own arrays. `run_slabs` is the first
and `map_ordered` is the second. Everything else stays serial.

**Threads, not processes**, throughout. `scipy.ndimage` releases the GIL; skimage does so
unevenly, which is why Phase C expects ~1.6-1.7x rather than 2-3x and that is understood and
accepted. Processes would mean pickling arrays that reach 850 MB.

> ### THE THREAD COUNT IS ENVIRONMENT, NOT A PARAMETER
>
> It follows the precedent `--device` set, for the same reason and with the same force:
> `SegmentationParams` is hashed, the hash names the pool, and `phase_a_hash` decides which
> pool a cell belongs in. A thread count on that dataclass would put the 4-thread half of a
> study in one pool and the 16-thread half in another. **So it is never hashed and never on
> `SegmentationParams`** — it is printed at startup and appended to `run_id`, which is free
> text that nothing parses, and that is what makes a wasted allocation visible in `pool.csv`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

__all__ = ["ENV_VAR", "resolve_threads", "describe_threads", "configure_torch",
           "use", "count", "slabs", "run_slabs", "map_ordered"]

T = TypeVar("T")
R = TypeVar("R")

#: The one source. Slurm sets it per task, and an Open OnDemand desktop session is a Slurm job
#: too, so the interactive route gets a real number from the same place the batch route does.
ENV_VAR = "SLURM_CPUS_PER_TASK"


def resolve_threads(explicit: int | None = None) -> int:
    """How many threads this task may use. **Falls back to 1, deliberately.**

    > ### ⇒ THE FALLBACK IS 1, AND IT IS NOT A CONSERVATIVE GUESS
    >
    > The suite runs on a Windows machine where `SLURM_CPUS_PER_TASK` is never set. A higher
    > fallback would have every threshold test spawn a pool, on arrays small enough that the
    > pool costs more than the work — thousands of times over. **Tests that need the threaded
    > path pass an explicit count**, which is the only way to reach it without a scheduler.
    >
    > It also means a run outside Slurm is serial and *says so* at startup, rather than
    > quietly deciding for itself how much of a shared machine to take.

    `--threads` wins over the environment; a value below 1 is raised to 1; and a variable that
    is not an integer is treated as absent rather than fatal — a malformed scheduler variable
    must not stop an image from being segmented.
    """
    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def describe_threads(threads: int) -> str:
    """The `run_id` token. Mirrors `nuclei.describe_device`, and for the same reason."""
    return f"{threads}t"


#: The count `--threads` fixed for this process, or `None` when nothing has fixed one and the
#: environment is still the source. **Not a parameter and not on any dataclass** — see the
#: module docstring for why that distinction is load-bearing rather than stylistic.
_FIXED: int | None = None


def use(threads: int) -> int:
    """Fix the count for this process. **Called once, by the CLI, at startup.**

    The flag has to reach functions that are called from both routes and from deep inside the
    pipeline, and threading them a `threads=` argument through every intermediate signature
    would put a hardware fact into the pipeline's own vocabulary. So the CLI states it once and
    `count` answers with it.
    """
    global _FIXED
    _FIXED = max(1, int(threads))
    return _FIXED


def count(explicit: int | None = None) -> int:
    """The count to use here. **The one call every threaded function in the package makes.**

    `explicit` is what a caller passed — a test asking for the threaded path, which is the only
    way to reach it on a machine with no scheduler. Otherwise: the count the CLI fixed, and
    failing that the environment, and failing that 1.
    """
    if explicit is not None:
        return max(1, int(explicit))
    return _FIXED if _FIXED is not None else resolve_threads()


def configure_torch(threads: int) -> None:
    """`OMP_NUM_THREADS` **before** torch is imported, `set_num_threads` after.

    > ### ⇒ LEFT ALONE, TORCH SIZES ITS POOL FROM THE NODE, NOT FROM THE ALLOCATION
    >
    > A Bunya node is 100 cores wide and a task is allocated 16. Torch reads the machine, not
    > the allocation, so it builds a pool far larger than the cores it has and thrashes.
    > **Cellpose runs on CPU on both routes now**, so this is the single largest item in the
    > Phase A half of the patch.
    >
    > The environment variable is the half that has to happen first: OpenMP reads it once, at
    > load time, and a `set_num_threads` afterwards does not undo a pool that is already
    > built. `set_num_threads` is called as well, because that is the half covering torch's
    > own intra-op pool.

    **Best-effort on the torch half, on purpose.** torch is an optional dependency — a machine
    without it must still reach the screens — so the import failing here is not an error. The
    environment variable is set either way, because OpenMP is used by more than torch.
    """
    value = str(max(1, int(threads)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = value
    try:
        import torch
    except Exception:                        # noqa: BLE001 - optional, and never fatal here
        return
    torch.set_num_threads(max(1, int(threads)))


def slabs(n_planes: int, threads: int) -> list[tuple[int, int]]:
    """Contiguous ``[lo, hi)`` z-ranges, one per thread, **balanced to within one plane**.

    Contiguous rather than strided so each worker writes into one span of the output array and
    the writes never interleave. Returns fewer ranges than `threads` when there are fewer
    planes than threads, and never returns an empty range.
    """
    n_planes = max(0, int(n_planes))
    threads = max(1, int(threads))
    if n_planes == 0:
        return []
    n = min(threads, n_planes)
    edges = [(i * n_planes) // n for i in range(n + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n)]


def run_slabs(work: Callable[[int, int], None], n_planes: int, threads: int) -> None:
    """Run ``work(lo, hi)`` over the z-ranges of `slabs`, **into a pre-allocated output.**

    > ### WORKERS WRITE INTO DISJOINT VIEWS; NOTHING IS COLLECTED AND CONCATENATED
    >
    > A map-and-concatenate would hold every worker's result at once and then a joined copy —
    > double peak memory on arrays that reach 850 MB. The caller allocates the output once and
    > each worker writes only its own planes, which is what makes contiguous ranges a
    > requirement rather than a tidiness.

    **One thread runs inline, with no pool at all.** That is the path the suite takes on a
    machine with no scheduler, and it must cost nothing.
    """
    n_planes = int(n_planes)
    if threads <= 1 or n_planes <= 1:
        if n_planes > 0:
            work(0, n_planes)
        return
    ranges = slabs(n_planes, threads)
    if len(ranges) <= 1:
        work(0, n_planes)
        return
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        for _ in pool.map(lambda lohi: work(*lohi), ranges):
            pass


def map_ordered(work: Callable[[T], R], items: Sequence[T] | Iterable[T],
                threads: int) -> list[R]:
    """``[work(x) for x in items]``, computed in parallel, **returned in the original order.**

    Every per-cell loop in this package accumulates in a way that depends on iteration
    position — `Derivation.add_cell` assigns cell ids in order, `load_file_scene`'s `index`
    assigns each cell's label value, and Phase C's ``label_value = len(cell_labels_index) + 1``
    does the same. So the compute is parallel and the accumulation is not: this returns a
    list, and the caller walks it exactly as it walked the input.

    `ThreadPoolExecutor.map` already yields in submission order, which is the whole mechanism.
    """
    items = list(items)
    if threads <= 1 or len(items) <= 1:
        return [work(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(threads, len(items))) as pool:
        return list(pool.map(work, items))
