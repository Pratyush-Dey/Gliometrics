"""The semi-automatic flow. `COMPONENT_03_PLAN.md` §3a, §4, §4a.5, §4a.7.

**The screen order and its gates, with no Qt anywhere in this module.** That is deliberate: §4a.7
keeps the flow logic — which screen follows which, what a commit invalidates, when the parameters
file is written — testable in a suite that runs on a cluster with no display.

    The UI calls the pipeline, and the pipeline never calls the UI.

`napari.run()` lives in `viewer.NapariViewer.run` and is unreachable from anything Phase A
executes; `tests/test_ui_boundary.py` asserts it structurally.

> ### STEPS 1-7 ARE ENTIRELY IN MEMORY — §4a.5, as amended 2026-08-16
>
> **Nothing reaches the pool until item 8.** Crash or close before saving and the image is left
> exactly as if the session had never run: no cells, no arrays, no file record, no operations
> record. **By construction, not by cleanup** — there is nothing in the pool to remove.
>
> This is why §9.4's storage foundation is not on the demo path. It stays required for the
> `review --pool` batch route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np

__all__ = [
    "Step", "FLOW_ORDER", "OPTIONAL_STEPS", "COMMIT_SCREENS", "LIVE_SCREENS", "NO_BACK",
    "BACK_FROM", "applicable_steps", "no_back_steps", "back_map", "parsed_numbers",
    "Decision", "Session", "Flow", "GateError",
]


class Step(Enum):
    """The screens, valued by the title the user actually sees. **Twelve, or eleven.**

    **It was ten until `BACKGROUND` was added on 2026-08-23**, and this sentence said so for a week
    afterwards. `test_ui_app` has asserted the count against `FLOW_ORDER` the whole time, which is
    why the stale number was prose and never behaviour — the count that matters is derived, not
    written. **Twelve since `SOMA` was added on 2026-08-30**, and this time the sentence moved
    with the enum.

    **Twelve is now the ceiling rather than the count, since 2026-09-02.** `OPTIONAL_STEPS` names
    the screens a study can lack: `CD68` is item 9a, and a study acquired without that channel
    walks eleven. `Flow.steps()` is what any question about *this session's screens* must ask;
    `FLOW_ORDER` remains the canonical order and every count in it is the four-channel one.

    **Item numbers are ours and never appear in the interface.** Nobody using the tool has seen
    "item 4"; they saw `Microglial Nuclei Detection` at the top of a screen. The item numbers
    survive only in comments and in the plan.
    """

    INTAKE = "Image Intake"                        # item 1
    #: **Between items 1 and 2, and the position is forced rather than chosen.**
    #: `nuclei.subtract_background` must run before the plaque veto — the veto is the only step
    #: that writes zeros into DAPI, and a Gaussian estimate rings around a zero region. Putting
    #: the control in the same order as the computation is what makes the screen honest.
    #: **No item number**: the numbering is frozen so that four years of `§4 item N` pointers
    #: keep meaning what they meant, exactly as item 9 preceding item 8 already does.
    BACKGROUND = "DAPI Background Filter"           # between items 1 and 2
    VETO = "Plaque DAPI Core Veto"                 # item 2
    THRESHOLD_1 = "Iba1 Segmentation Threshold"    # item 3
    NUCLEI = "Microglial Nuclei Detection"         # item 4
    THRESHOLD_2 = "Iba1 Measurement Threshold"     # item 5
    SIZE = "Cell Size Cutoff"                      # item 6
    #: **Item 6.5, and the half-number is the user's** — *"i propose this screen as item 6.5"*.
    #: `BACKGROUND` took no number at all when it was inserted, on the rule that the numbering is
    #: frozen; this one is named because the user named it.
    #:
    #: **The position is forced, not chosen.** Everything on it moves the white soma at item 7b,
    #: so it has to come before review — and item 8's own note refused these fields on exactly
    #: that ground, that a mechanism must not be flipped after the cells were reviewed under a
    #: different one. Before review, that objection dissolves.
    SOMA = "Soma Allocation Parameters"            # item 6.5
    REVIEW = "Cell Review"                         # item 7 - triage, then rich review
    CD68 = "CD68 Threshold"                        # item 9a - study-level
    PLAQUE = "6E10 Threshold"                      # item 9b - study-level
    SAVE = "Save Parameters"                       # item 8 - the session's commit point


#: §3a. **Flow order and item numbering are deliberately decoupled** — item 9 precedes item 8,
#: because both pickers are `None` until item 9 runs and writing the parameters file first would
#: have made it incomplete by design, with something else obliged to fill in two hashed fields
#: later. Written last, the file goes out complete, once.
#:
#: **The order is fixed; the membership is not, since 2026-09-02.** This tuple is every screen
#: the program has, in the one order they are ever met — and it is never mutated. A session that
#: lacks an optional role walks a **subsequence** of it, computed by `applicable_steps`, so the
#: relative order of what remains is the order written here by construction.
FLOW_ORDER: tuple[Step, ...] = (
    Step.INTAKE, Step.BACKGROUND, Step.VETO, Step.THRESHOLD_1, Step.NUCLEI, Step.THRESHOLD_2,
    Step.SIZE, Step.SOMA, Step.REVIEW, Step.CD68, Step.PLAQUE, Step.SAVE,
)

#: Screens where the view can lag the parameters, so Continue is gated on a fresh commit.
#: **9a is not one**: its whole-volume preview is 460 ms, under Q2's line, so the screen is live
#: throughout and there is no stale view for a gate to guard.
#:
#: **The background filter is one**, although its slider is live and its preview is a plane.
#: What COMMIT buys is the whole-field product: the filtered volume every later screen and the
#: detector read, and the *"DAPI removed"* readout, which is a whole-field number and cannot be
#: taken off one slice. It costs ~1 s on a crop and **~67 s on a full field**.
COMMIT_SCREENS: frozenset[Step] = frozenset({
    Step.BACKGROUND, Step.VETO, Step.THRESHOLD_1, Step.NUCLEI, Step.THRESHOLD_2, Step.PLAQUE,
})

#: Screens whose controls take effect immediately, so there is nothing to commit.
LIVE_SCREENS: frozenset[Step] = frozenset({Step.SIZE, Step.CD68})


#: **Screens that exist only when the study has the channel they need**, step to the role it
#: requires. The program is built for four channels; a study acquired without CD68 has no item
#: 9a to show, and `set_channels` would do `np.asarray(None).shape[0]` if it were shown one.
#:
#: **A map rather than a mutated `FLOW_ORDER`.** The order stays one canonical tuple and every
#: per-session answer is derived from it, so the full-flow case is not a second code path that
#: could drift from this one — it is the same two functions with nothing removed.
OPTIONAL_STEPS: dict[Step, str] = {Step.CD68: "cd68"}


def applicable_steps(session: "Session | None" = None) -> tuple[Step, ...]:
    """`FLOW_ORDER` minus any optional step whose role this study did not declare.

    > ### ⇒ AN EMPTY `roles` MEANS INTAKE IS NOT DONE, NOT THAT A DECISION WAS MADE
    >
    > A bare `Session()` has `roles={}`. Reading that as *"this study has no CD68"* would skip
    > item 9a for every session built before intake has run — test fixtures and drivers sent
    > straight to a late step included. The skip is gated on the roles being **declared and
    > lacking `cd68`**, and `has_roles()` already refuses to leave item 1 until they are
    > declared, so nothing real reaches the flow undeclared.
    """
    roles = getattr(session, "roles", None)
    if not roles:
        return FLOW_ORDER
    return tuple(step for step in FLOW_ORDER
                 if step not in OPTIONAL_STEPS or OPTIONAL_STEPS[step] in roles)


def no_back_steps(steps: Iterable[Step]) -> frozenset[Step]:
    """Item 1, plus the first step after review — **derived, so a skip cannot orphan it.**

    *Leaving review is final* is a property of the screen that **follows** review, not of item
    9a in particular. With CD68 present that is still 9a and this returns the pair it always
    returned; without it, item 9b inherits the pin rather than inheriting a Back button to a
    screen the user never saw.
    """
    order = tuple(steps)
    pinned = {Step.INTAKE}
    if Step.REVIEW in order:
        after = order[order.index(Step.REVIEW) + 1:]
        if after:
            pinned.add(after[0])
    return frozenset(pinned)


def back_map(steps: Iterable[Step]) -> dict[Step, Step]:
    """The previous applicable step, for every step that has a Back button.

    **Derived, never written out.** A hand-kept map is a list to maintain, and this project has
    now been bitten four times by a written fact with nothing counting it.
    """
    order = tuple(steps)
    pinned = no_back_steps(order)
    return {
        step: order[index - 1]
        for index, step in enumerate(order)
        if index > 0 and step not in pinned
    }


#: **The three screens with no Back**, the user's decision of 2026-08-30. Everything else goes
#: back one step.
#:
#: * **item 1** has nothing behind it.
#: * **the screen after review** is where *leaving review becomes final*. You can return to it
#:   from the screen after that, but you cannot re-open triage from it — re-entering review after
#:   leaving it would mean re-deciding cells that were already decided. **With four channels that
#:   screen is item 9a**, which is what this said outright until 2026-09-02; in a study with no
#:   CD68 channel item 9a does not exist and item 9b inherits the pin. The rule belongs to the
#:   *position*, and `no_back_steps` is where it is now written.
#: * **item 7b** is not a `Step`; it is inside `REVIEW`, and it is left by resolving a cell.
#:
#: > ### WHAT THIS REPLACED, AND WHY EVERY REASON IT GAVE HAS BEEN ANSWERED
#: >
#: > Until 2026-08-30 item 4 was the only screen with a Back button, and each other screen had
#: > its own recorded reason: item 2 *"sits outside the freely re-enterable range, because the
#: > veto feeds `detect_nuclei` and returning to it invalidates the 200-500 s detection
#: > downstream"*; item 3 because *"item 3 is unreachable from here, so the cap cannot move while
#: > you are on this screen"*; items 5 and 6 because **every volume on those screens is a
#: > threshold-2 volume**, so going back would invalidate the whole screen.
#: >
#: > **Every one of those said the same thing: going back invalidates what is downstream.** The
#: > user's rule makes that the *behaviour* rather than an objection — *"any time the user goes
#: > back they lose their settings for the screen they were on"*, and a screen is at its defaults
#: > whether it is reached forwards or backwards. Nothing downstream is left standing to be
#: > stale, so there is nothing left for those reasons to protect.
#:
#: **Both of these are the four-channel answer**, evaluated once at import from the full order.
#: `CD68` is the first step after `REVIEW`, so `NO_BACK` still comes out `{INTAKE, CD68}` and
#: full-flow behaviour is provably what it was. A session that skips a step asks `Flow` instead,
#: which resolves the same two functions against its own `steps()`.
NO_BACK: frozenset[Step] = no_back_steps(FLOW_ORDER)
BACK_FROM: dict[Step, Step] = back_map(FLOW_ORDER)


def parsed_numbers(fields: dict) -> dict:
    """`{field: (text, kind)}` -> `{field: value}`, **skipping anything that will not parse.**

    > ### NOTHING THE USER TYPED REACHED THE PIPELINE — task 24
    >
    > `Session.chosen` is what `screen_save.session_params` serialises and what
    > `PipelineCompute` reads, and before task 24 **only items 9a and 9b ever wrote to it.**
    > Items 2, 3, 4, 5 and 6 held their values in `QLineEdit`s and nowhere else, so every commit
    > ran at `SegmentationParams`' defaults and the parameters file recorded defaults — for
    > screens whose entire purpose is choosing the value.

    **A half-typed box contributes nothing rather than a zero.** `float("")` raises and a box
    mid-edit is not a choice; each screen's `parameter_error` refuses the commit on the same
    values, so a skipped field can never reach a commit either.

    Lives here rather than in eight screens because eight copies of a parser drift eight ways —
    the same reason `widgets.py` exists. (**It said five until 2026-08-30**, from before
    `screen_background.py` and `screen_intake.py` adopted it; `test_ui_widgets` now counts.)
    """
    chosen = {}
    for name, (text, kind) in fields.items():
        try:
            chosen[name] = kind(str(text).strip())
        except (TypeError, ValueError):
            continue
    return chosen


class GateError(RuntimeError):
    """Continue was refused. The message says which gate and what would clear it."""


@dataclass(frozen=True)
class Decision:
    """One cell's outcome at item 7, on its way to `Pool.set_status`.

    > ### ⇒ IT WAS A 3-TUPLE UNTIL 2026-08-30, AND THE FOURTH FACT IS WHY IT IS NOT
    >
    > `(status, reject_reason, reject_note)` was written in `app._leave`, unpacked positionally
    > in `screen_save.apply_decisions`, and built literally in fourteen test assertions. The
    > reviewer's cell count — *"this is 3 cells I cannot afford to split"* — had nowhere to sit
    > in it, which is the whole of why `Resolution.n_cells` reached no column for four days
    > while being carried correctly out of the editor.
    >
    > **A dataclass rather than a wider tuple**, on the user's instruction. A 4-tuple unpacked
    > positionally in two files is one insertion away from putting the note in the count's
    > place, and nothing would raise: both ends would still be tuples of the right length.
    > Names cannot be transposed silently.

    **Frozen, because a decision is a record of what a reviewer did.** Item 7 replaces one by
    assignment when a cell is re-entered; nothing edits one in place.
    """

    status: str
    reason: str | None = None
    note: str | None = None
    #: **1 unless the reviewer said otherwise**, and only a `real_cell` rejection can ask.
    #: `Pool.set_status` is what turns this into `None` on debris — see there for why that is
    #: derived rather than represented here.
    n_cells: int = 1


@dataclass
class Session:
    """Everything one semi-automatic run holds, and **all of it is in memory**.

    One image from intake to save — which is what makes `source_image` singular and unambiguous,
    and is why the *"N answers to a question that has one"* problem never arises here.
    """

    image_path: str = ""
    image_root: str = ""
    output_root: str = ""
    roles: dict[str, int] = field(default_factory=dict)
    voxel_size_um: tuple[float, float, float] | None = None

    #: **Per-channel display range, keyed by role** — item 1's min/max boxes. Display only:
    #: it changes nothing the pipeline computes, so it stays out of `SegmentationParams` and
    #: out of the parameter hash, which picks the pool directory. A role that is absent, or
    #: present as `None`, falls back to the percentile stretch — see `preview.contrast_limits`.
    display_range: dict[str, tuple[float, float]] = field(default_factory=dict)

    #: `(z, y, x)` of the source image, set when the channels are loaded. **`FileRecord` needs
    #: it and it must survive `release()`** — every array can be dropped to save memory, and the
    #: shape of the image the session ran on is not something to re-open a 3.17 GiB file for.
    image_shape: tuple[int, int, int] | None = None

    #: Channel volumes, held as `float32`. **Never cast to `uint16`** — the source is `float32`
    #: and the cast collapses every voxel below 1.0 to zero.
    channels: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    #: What each screen produces, keyed by the step that produced it.
    products: dict[Step, Any] = field(default_factory=dict, repr=False)

    #: Parameters chosen so far, by field name on `SegmentationParams`.
    chosen: dict[str, Any] = field(default_factory=dict)

    #: Steps whose committed product is current. A commit adds; a parameter change removes.
    committed: set[Step] = field(default_factory=set)

    #: The session's cells, as an **iterable** of `PoolWriter.add_cell` keyword arguments.
    #:
    #: **Deliberately not a dataclass of its own.** The payload *is* `add_cell`'s signature, and
    #: a parallel dataclass would have to be kept in step with it by hand; `**cell` cannot drift.
    #: Item 8 is the only reader — §4a.5 as amended: steps 1-7 are entirely in memory, and the
    #: first write happens at item 8 together with the parameters file.
    #:
    #: **An iterable rather than a list, and that is §4a.5's memory line.** In the real flow this
    #: is `pipeline.Derivation.payloads()`, a generator that crops one cell at a time — *"the
    #: memory cost is one array, not sixty."* A list is equally acceptable and is what the tests
    #: use; nothing may call `len()` on it.
    cells: Iterable[dict] = field(default_factory=list, repr=False)

    #: The label volume the cells came from, held so item 7 can review them and item 8 can crop
    #: them. Set by the flow driver from `pipeline.segment_image(..., pool_root=None)`.
    derivation: Any = field(default=None, repr=False)

    #: **Item 7's review decisions**, keyed by the position of the cell in label order and valued
    #: `Decision` — which is `Pool.set_status`'s arguments, and was a bare 3-tuple until
    #: 2026-08-30. See `Decision` for why it stopped being one.
    #:
    #: **Without these the whole of Phase C is empty.** `PoolWriter` writes every cell as
    #: `CANDIDATE`, and `measure` takes only `CONFIRMED` unless it is told to include unreviewed
    #: cells — so a user who reviews sixty cells, saves, and runs the two commands the SAVED popup
    #: hands them gets *"measured 0 cell(s) across 0 image(s)"* and empty CSVs. **Found by running
    #: one image end to end**; no per-screen test could see it, because it is the seam between
    #: item 7 and item 8 and neither screen owns it.
    decisions: dict[str, "Decision"] = field(default_factory=dict)

    #: How many nuclei the user edited by hand on item 4, carried forward for the sentence §6a
    #: requires item 8 to show: *"the save step has to say plainly that manual edits stay with
    #: this image."* A user who hand-splits nuclei, saves and expects batch to reproduce their
    #: work *"would be silently wrong"* — and batch cannot, by design, because §6a made nucleus
    #: edits per-image overrides that do not transfer.
    nucleus_edits: int = 0

    #: The `min_cell_volume_um3` already folded out of `derivation`, so leaving item 6 twice
    #: does not filter twice. `None` means the cutoff has never been applied.
    #:
    #: **Not about re-entering item 6** — `Flow.back()` raises on every screen but item 4, so a
    #: reviewer never does. `_apply_cutoff` runs from `prepare`, which fires on every arrival at
    #: every step after item 6: items 7, 9a, 9b and 8, four times in one forward walk. The fold
    #: is idempotent, so the answer would be right either way; the guard is what stops
    #: `filter_cells` allocating a native label volume four times over — 850 MB a pass on a
    #: production field.
    #:
    #: **A value rather than a flag** so that a driver sent to a step directly with a raised
    #: cutoff still folds the difference, rather than being told the work is already done.
    cutoff_applied_um3: float | None = None

    saved: bool = False

    def release(self, *keep: str) -> None:
        """Drop channels not named. §4a.6 as amended: **fewer channels, not narrower ones.**"""
        for name in [n for n in self.channels if n not in keep]:
            del self.channels[name]

    def has_roles(self) -> bool:
        """Item 1's gate, and the one that `segment` exits 4 over if it is fudged.

        **Presence counts, not just index.** A parameters file written without the roles set
        declares `nuclei=0, cell=2` and no CD68 or plaque channel — a *two-channel study* — so a
        later `--roles nuclei:0,cd68:1,cell:2,plaque:3` contradicts it and refuses.
        """
        return bool(self.roles) and {"nuclei", "cell"} <= set(self.roles)


class Flow:
    """Screen order and the gates between screens. No Qt, no napari, no pool.

    A screen calls `commit()` when its expensive step finishes and `invalidate()` when a control
    moves; `can_continue()` answers whether Continue is live, and `advance()` refuses rather than
    silently allowing a stale view forward.
    """

    def __init__(self, session: Session | None = None, start: Step = Step.INTAKE):
        self.session = session if session is not None else Session()
        self.step = start
        self.history: list[Step] = [start]

    # -- position -------------------------------------------------------

    def steps(self) -> tuple[Step, ...]:
        """**The screens this session actually has**, which is `FLOW_ORDER` unless a role is
        absent. Every movement below resolves against this rather than indexing the module
        tuple, so the skip case and the full case are one path."""
        return applicable_steps(self.session)

    def index(self) -> int:
        """Position in the **canonical** order. Unchanged by a skip on purpose: it answers
        *where is this screen in the flow as designed*, and `steps()` answers *what does this
        session walk*."""
        return FLOW_ORDER.index(self.step)

    def next_step(self) -> Step | None:
        order = self.steps()
        if self.step in order:
            position = order.index(self.step)
            return order[position + 1] if position + 1 < len(order) else None
        #: **A driver standing on a step this session skips** — jumped to directly, or left
        #: standing when roles were set afterwards. Moving on to the next screen it does keep is
        #: the only answer that is not a crash; walking off the end returns `None` as ever.
        following = FLOW_ORDER[FLOW_ORDER.index(self.step) + 1:]
        return next((step for step in following if step in order), None)

    def back_target(self) -> Step | None:
        """Where Back goes from here, or `None` where there is no Back.

        **The driver asks this rather than reading `BACK_FROM`**, because the module map is the
        four-channel answer and a session that skipped a screen must not be offered it."""
        return back_map(self.steps()).get(self.step)

    def can_go_back(self) -> bool:
        return self.back_target() is not None

    # -- commits --------------------------------------------------------

    def commit(self, product: Any = None) -> None:
        """Record that this screen's expensive step ran at the current parameters.

        > ### AND IT INVALIDATES EVERY COMMIT DOWNSTREAM OF IT - found by walking the flow
        >
        > **Item 4 is the only screen with a Back button and it goes to item 3.** So this path
        > exists: commit item 4's 200-500 s detection, go Back, move `cell_coefficient`,
        > COMMIT item 3 again, Continue. Item 4 is then showing nuclei whose membership
        > fractions were computed against **the mask that was just replaced** - and its own gate
        > still read as current, so there was no exclamation, Continue was live, and the
        > `microglial: 5 of 35` in the strip was answering for a segmentation nobody could see
        > any more.
        >
        > Nothing downstream reads `products[Step.NUCLEI]` - item 8 recomputes everything from
        > the parameters - so the damage was confined to what the screen showed. **That is still
        > the worst shape of wrong**: a number that looks settled and answers a superseded
        > question.
        >
        > **Only `commit` cascades, not `invalidate`.** Invalidate fires on every step of a drag;
        > cascading there would discard a 200-500 s detection because a hand moved a slider on
        > the screen behind it, and item 3's own gate already refuses Continue until it is
        > re-committed. Nothing can reach item 4 without passing that gate.
        >
        > On the forward path this changes nothing at all: everything downstream of the step
        > being committed is uncommitted already.
        """
        if self.step not in COMMIT_SCREENS:
            raise GateError(
                f"{self.step.value} has no commit. "
                + ("Its controls are live." if self.step in LIVE_SCREENS
                   else "It has no viewer."))
        self.session.committed.add(self.step)
        self.session.products[self.step] = product
        for later in FLOW_ORDER[FLOW_ORDER.index(self.step) + 1:]:
            self.session.committed.discard(later)

    def invalidate(self) -> None:
        """A control moved, so the view no longer matches the commit.

        On item 9b this is visible twice over: the exclamation beside Commit **and** the plaque
        outlines reverting to magenta. That second carrier falls out of the mechanism rather than
        being designed in — the live preview recomputes outlines the commit has not classified,
        and magenta is simply what an unclassified outline is.
        """
        self.session.committed.discard(self.step)

    def is_stale(self) -> bool:
        return self.step in COMMIT_SCREENS and self.step not in self.session.committed

    # -- gates ----------------------------------------------------------

    def blocking_reason(self) -> str | None:
        """Why Continue is refused, or `None` when it is live."""
        if self.step is Step.INTAKE:
            if not self.session.image_root or not self.session.output_root:
                return ("Both roots are needed. The image root is where "
                        "<genotype>/<animal>/<file> gives identity; the output root is where "
                        "the pool, parameters and exports are written.")
            if not self.session.image_path:
                return "No file selected."
            if not self.session.has_roles():
                return ("Channel roles are not set. They are written into the parameters file, "
                        "and `segment` exits 4 when --roles contradicts it.")
            if self.session.voxel_size_um is None:
                return "Voxel size is not confirmed."
            return None

        if self.is_stale():
            return (f"{self.step.value} has not been committed at these settings. "
                    "The view does not match what Continue would carry forward.")
        return None

    def can_continue(self) -> bool:
        return self.blocking_reason() is None

    # -- movement -------------------------------------------------------

    def advance(self) -> Step:
        reason = self.blocking_reason()
        if reason is not None:
            raise GateError(reason)
        following = self.next_step()
        if following is None:
            raise GateError(f"{self.step.value} is the last screen. The flow ends here: "
                            "`measure` and `export` are run from the CLI.")
        self.step = following
        self.history.append(following)
        return following

    def back(self) -> Step:
        """Go back one screen, **discarding everything from the target forward.**

        > ### THE RULE IS THE USER'S AND IT IS UNIFORM — 2026-08-30
        >
        > *"Any time the user goes back they lose their settings for the screen they were on"*,
        > and *"the screen is default whether u go forward or back to it."* So both ends are
        > cleared: the screen being left, and the screen being landed on.
        >
        > **Uniform is what makes it small.** The alternative — the screen you return to keeping
        > what you chose — would need `prepare` to know which direction it was called from and
        > every one of its twelve branches audited for what re-entry means. Under this rule the
        > existing behaviour is already correct: `prepare` resets, and that is now the specified
        > outcome rather than something to work around.
        >
        > **The derivation goes too, when the target is at or before item 6.** It was built from
        > parameters that are now back at their defaults, and item 6's cutoff fold is destructive
        > — `apply_cell_size_cutoff` renumbers survivors in place, so lowering the cutoff cannot
        > restore what raising it removed. Rebuilding is what makes the user's *"going back to
        > item 6 should mean that everything should be recoverable"* true, and it needs no second
        > copy of an 850 MB label volume to do it. **It costs a full re-derive on the way
        > forward**, which is the price of the rule rather than a defect in it.
        """
        if not self.can_go_back():
            raise GateError(
                f"{self.step.value} has no Back button. "
                "Item 1 has nothing behind it, item 9a is where leaving review becomes final, "
                "and rich review is left by resolving a cell.")
        target = self.back_target()
        assert target is not None                     # can_go_back() is exactly this question
        self.discard_from(target)
        self.step = target
        self.history.append(target)
        return target

    def discard_from(self, target: Step) -> None:
        """Drop every commit, product and derivation from `target` forward.

        **Separate from `back` so the driver can use it too** — the widgets are the driver's to
        rebuild, and both halves of *the screen is default either way* have to happen together
        or the model and the glass disagree about what was discarded.
        """
        for step in FLOW_ORDER[FLOW_ORDER.index(target):]:
            self.session.committed.discard(step)
            self.session.products.pop(step, None)
        if FLOW_ORDER.index(target) <= FLOW_ORDER.index(Step.SIZE):
            self.session.derivation = None
            self.session.cutoff_applied_um3 = None

    # -- the end --------------------------------------------------------

    def save(self) -> None:
        """Item 8, and it is **the session's commit point**.

        This writes both the parameters file and the session's cells. Before it, nothing exists
        in the pool; after it, the run is durable. There is no bridge to Phase C by decision —
        the user runs `measure` and `export` themselves, exactly as a batch user does.
        """
        if self.step is not Step.SAVE:
            raise GateError(f"save() belongs to {Step.SAVE.value}, not {self.step.value}")
        self.session.saved = True
