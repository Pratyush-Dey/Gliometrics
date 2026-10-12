"""Screen 8 — Save Parameters. `COMPONENT_03_PLAN.md` §4 item 8.

**The session's COMMIT POINT, and the only screen in the component with no viewer at all** — no
image, no layers, no camera, so no frame from `frames.py` and no `LAYERS` tuple. §6d excluded it
from the colour session for exactly that reason: *"item 8 has no viewer."* It takes §6e's chrome
unchanged. **Item 1's shape**: one full screen, stacked top to bottom, button centred at the
bottom with red text beneath it.

> ### STEPS 1-7 ARE IN MEMORY AND THIS SCREEN IS WHERE THEY LAND — §4a.5 as amended
>
> **Nothing reaches the pool until this screen completes**, and it writes **both** the parameters
> file and the session's cells. Crash or close before saving and the image is left exactly as if
> the session had never run — *"by construction, not by cleanup"*, because there is nothing in
> the pool to remove.
>
> **The objection is recorded beside the decision and was accepted knowingly.** At item 8 the
> blast radius includes item 7: a crash after review loses nucleus edits and per-cell process
> editing, *"human judgement on ~60 cells… the one thing §4a.6's cache cannot rebuild."* A
> mitigation was offered and declined; **the user's decision: start fresh.**

> ### ONE EDITABLE PARAMETER, AND THE RULE THAT LEAVES IT ALONE THERE
>
> `min_plaque_volume_um3` **left this screen on 2026-08-17** by this screen's own test —
> *"you are being asked NOW; nowhere else asks"* — because 9b now asks, and **task 18 built it
> there**. What remains is `min_spur_um`.
>
> **The principle, which holds for anything added later:** *a parameter is decided on the screen
> that can show its consequence, and item 8 holds only what no screen can show.* No screen in
> this component draws a skeleton, so nobody can look at a field and see 1 µm of spur.

**The read-only block is grouped by SCREEN TITLE, sentence case.** Item numbers are ours and
never appear in the interface — §2's tenth row, and the read-only rule's *"naming the screen to
go back to"* only became satisfiable when the screens gained titles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from ..export import MASKS_DIR
from ..params import save_params
from ..pipeline import SegmentationParams
from ..store import PoolWriter, merge_fragments, pool_dir_name
from . import theme
from .semiauto import Flow, Session, Step
from .widgets import HelpButton, NavigationRow, screen_title

__all__ = [
    "SaveParametersScreen", "SavedPopup", "SaveResult",
    "READ_ONLY_GROUPS", "READ_ONLY_FIELDS", "EDITABLE_FIELD", "HIDDEN_FIELDS",
    "DEFAULT_FILE_NAME", "POOL_ORIGIN", "DEFAULT_EXPORT_DIR",
    "session_params", "read_only_rows", "format_value", "field_value",
    "NO_CD68_CHANNEL_NOTE", "parameters_path",
    "collision_message", "hand_edit_statement", "measure_command", "export_command",
    "commit_session", "apply_decisions",
]

#: §4 item 8's layout: **eighteen values in eight groups, shown outright** — not collapsed, not
#: behind a disclosure. Grouping makes the screen name a heading instead of repeating it
#: seventeen times, and *"go back to that screen"* is the actionable part.
#:
#: **Fifteen in seven until 2026-08-23**, when the DAPI background filter got its own screen and
#: its two parameters joined the block under that screen's title. They had briefly sat on item 4
#: as one field and were taken off it while the name was still being decided, *"because a field
#: name reaches the parameters file and the pool key permanently."*
#:
#: **Sentence case, not the all-caps used at the top of each screen** — seven all-caps headings
#: in a stack would shout.
#:
#: **`plaque_dilate_um` and `plaque_threshold`, not the drawn `plaque_veto_margin_um` and
#: `plaque_grow_threshold`** — §5's first row, the user's decision, *"so the exports work as
#: intended for the demo."* The drawn names are request 12's and ride the v5 -> v6 bump. Items 2
#: and 9b show the same names on their own screens, so this block matches what the user saw.
READ_ONLY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    #: **First, because its screen is first.** The groups run in flow order, which is how
    #: *"go back to that screen"* stays actionable — a block ordered any other way makes the
    #: reader search for the screen instead of retracing the session.
    ("DAPI background filter",
     ("dapi_background_strength", "dapi_background_scale_um")),
    ("Plaque DAPI core veto", ("plaque_veto_coefficient", "plaque_dilate_um")),
    ("Iba1 segmentation threshold",
     ("cell_coefficient", "min_object_voxels", "closing_radius_um")),
    ("Microglial nuclei detection",
     ("nucleus_diameter_um", "stitch_threshold", "nucleus_min_volume_voxels",
      "membership_threshold")),
    #: **Three values since the debris patch, 2026-08-24.** `measurement_min_object_voxels`
    #: joined when item 5 got a box for it; it prints under item 5's title because that is the
    #: screen to go back to, which is the whole point of the grouping.
    ("Iba1 measurement threshold",
     ("mask_coefficient", "measurement_min_object_voxels", "mask_closing_radius_um")),
    ("Cell size cutoff", ("min_cell_volume_um3",)),
    #: **Added 2026-08-30 with item 6.5.** These were in `HIDDEN_FIELDS` until that screen
    #: existed, on the recorded ground that the user had never been offered them. They are now
    #: asked for on a screen of their own, so they print here like every other choice — which is
    #: the user's instruction: *"this should be shown on item 8 and treated how every other
    #: parameter is."*
    #:
    #: > ### ⇒ `soma_method` JOINED THEM LATER THE SAME DAY, AND HAD TO
    #: >
    #: > It stayed hidden when the other four moved, fixed at `anchor_sheet`; the user reversed
    #: > that and item 6.5 now offers it as a dropdown. **Moving it here is not presentation** —
    #: > `session_params` filters `session.chosen` against `READ_ONLY_FIELDS`, so a method
    #: > chosen on 6.5 while this tuple still hid it would have been **silently dropped from the
    #: > parameters file** while item 7b drew the soma under it. The glass and the file would
    #: > have disagreed, in the direction that looks fine.
    #: >
    #: > **It prints first in the group**, because the four below it are only read under one of
    #: > its two values: a reader who sees `opening` here should see it before the three
    #: > `anchor_sheet` numbers that sat out the run at their defaults.
    ("Soma allocation parameters",
     ("soma_method", "soma_anchor_um", "soma_thickness_um", "soma_sheet_area_um2",
      "soma_waist_um", "soma_min_sheet_um3")),
    ("CD68 threshold", ("cd68_threshold",)),
    ("6E10 threshold", ("plaque_threshold", "min_plaque_volume_um3")),
)

READ_ONLY_FIELDS: tuple[str, ...] = tuple(
    name for _, names in READ_ONLY_GROUPS for name in names)

#: The parameters this screen asks for. **Editable because no other screen asks for them**, and
#: used at measurement time rather than now.
EDITABLE_FIELD = "min_spur_um"
LOOP_FIELD = "loop_prune_method"

#: The dropdown, in the order the user named them. Label to stored value: the value is what
#: reaches `params_hash`, the label is what a person reads.
LOOP_CHOICES: tuple[tuple[str, str], ...] = (
    ("none", "none"),
    ("shortest branch", "shortest_branch"),
    ("lowest intensity branch", "lowest_intensity_branch"),
    ("lowest intensity voxel", "lowest_intensity_voxel"),
)

#: **Never asked, and not meaningfully changeable here, so NOT SHOWN AT ALL.** This third class
#: is what keeps the screen a decision point rather than a parameter dump: the block above is
#: exactly the set of choices this session made.
#:
#: **`normalize` is here and was in the wrong class until 2026-08-17.** The plan listed it under
#: *"you already chose it — item 4"*, so it would have printed read-only with *"go back to item
#: 4"* beside it; item 4's layout, settled the day after that table was written, **took it off
#: that screen** — it is fixed at `"log"` and the user is never asked.
#:
#: > ### ⇒ `soma_thickness_um` STOPPED BEING HIDDEN ON 2026-08-30
#: >
#: > It read *"hidden rather than read-only because it was never offered, and if it ever changes,
#: > every soma volume changes with it — an inherited constant governing an exported measure.
#: > **This screen is not where that is closed.**"* That was right, and it named the condition
#: > for its own retirement: it is closed on **item 6.5**, which is where it is now offered, and
#: > it prints here read-only like every other choice.
#:
#: > ### HALF OF THAT SENTENCE WAS FIXED ON 2026-08-29
#: >
#: > It called the field *"one of the ten `PARAMETERS.md` records with no purpose written
#: > down"*. **It now carries one**, above the field itself, and the bare count is **nine**.
#: > What did not change is that it is unreachable from any screen and uncalibrated against
#: > ground truth — `EXPORT_SPEC.md` §32's four validation steps are still open.
#: **`speck_floor_um3` left this tuple by being DELETED from the program**, 2026-08-24. It was
#: the clearest case this class ever held — a 20 µm³ floor nobody was ever shown, removing ~88%
#: of objects (537 -> 62 measured on a real image) before anything could count them. Under the
#: debris patch what it removed would have vanished *before debris existed*, so the accounting
#: could never close. Its stated purpose was cost, not science: *"purely to avoid promoting
#: hundreds of specks to native resolution."* `docs/SHAKEOUT_FINDINGS.md` carries the removal
#: with its measured justification and the rejected alternative of surfacing it as a box.
#: > ### AND THE OTHER FOUR WENT WITH IT, EXCEPT THE METHOD — 2026-08-30
#: >
#: > The note here read that the five soma-mechanism fields were hidden because the user was
#: > never offered them, they are *"evaluated by **looking at item 7b** rather than by setting a
#: > box"*, and **putting a live mechanism switch on the save screen would invite it to be
#: > flipped after the cells had already been reviewed under a different one.**
#: >
#: > **The second half is still true and is exactly why item 6.5 sits before review.** Looking at
#: > 7b is still how they are judged — item 6.5's own "?" says so and points there. What changed
#: > is that a screen now asks for them, at a place in the flow where flipping them cannot
#: > invalidate work already done.
#: >
#: > `soma_bound_um` and `soma_occupancy_theta` are **gone from the program**, deleted with
#: > `bounded_occupancy` on the same day: they were the whole of that mechanism, and
#: > `anchor_sheet` never read them, so they were controls that could not change anything.
#:
#: > ### ⇒ `soma_method` LEFT THIS TUPLE THE SAME DAY IT JOINED THE OTHER FOUR — 2026-08-30
#: >
#: > It read *"the one that stays hidden, fixed at `anchor_sheet`, on the same terms as
#: > `normalize`: it is never asked and the user's ruling is that it is not changeable"*, and
#: > cited `soma.SOMA_METHODS` — `opening` reads only `soma_thickness_um`, so offering the switch
#: > would leave three of item 6.5's four boxes inert.
#: >
#: > **The user reversed the ruling and dissolved the argument in the same instruction**: item
#: > 6.5's boxes now follow the dropdown, so under `opening` the three are not inert, they are
#: > gone. `normalize` is now the only member of this class that a screen could plausibly have
#: > asked for — everything else here is a channel index or a fixed internal.
HIDDEN_FIELDS: tuple[str, ...] = (
    "nuclei_channel", "cell_channel", "cd68_channel", "plaque_channel",
    "normalize", "split_method", "smooth_um", "min_overlap_fraction",
    "remove_xy_border", "downsample", "thumbnail_max_px",
)

#: **Left fixed rather than made unique per session**, deliberately: the collision block fires
#: from the second session onward, so the name is chosen on purpose every time rather than
#: accepted by reflex.
DEFAULT_FILE_NAME = "parameters.json"

#: Semi-auto never writes a batch pool. §4a.3's enforcement, and the reason `origin` is a
#: category rather than `run_id`'s free text.
POOL_ORIGIN = "semiauto"

#: A **suggestion inside a string**, not a location this screen decides. `export --out` is a
#: required argument the user chooses, no export folder exists, and this screen does not create
#: one — so the pasted command runs as written and can be edited before it does.
DEFAULT_EXPORT_DIR = "exports"

#: The masks' counterpart, and a **sibling of `exports/` rather than a child of it**. Masks are
#: fixed by segmentation and do not change when a run is re-measured, so they do not belong
#: inside a folder whose contents are keyed by `params_hash`. `export_cell_masks` adds the dated
#: run folder inside this one, exactly as `export_pool` does inside `exports/`.
#: **`export.MASKS_DIR`, not a second literal** — patch item C6. `cli.py`'s `--with-masks`
#: resolves to the same name, so the pasted command and the flag land in one place.
DEFAULT_MASKS_DIR = MASKS_DIR


# --------------------------------------------------------------------------
# Pure logic — no Qt
# --------------------------------------------------------------------------

def session_params(session: Session) -> SegmentationParams:
    """The parameters this session chose, as the dataclass `save_params` serialises.

    **The channel roles are set here or the file declares a two-channel study.** §2's third row:
    the defaults are `nuclei=0, cell=2, cd68=None, plaque=None`, so a file written without them
    says the study has no CD68 and no plaque channel — and `segment` then **exits 4** against a
    `--roles` flag that contradicts it. The roles are item 1's, carried on the session.

    Everything else comes from `session.chosen`, which each screen wrote under the field name on
    `SegmentationParams` rather than under the name it drew.
    """
    params = SegmentationParams()

    #: **An unassigned role is left alone, and that is already `None` for the two that matter.**
    #: `cd68_channel` and `plaque_channel` default to `None` precisely so *this image has no such
    #: channel* is expressible — `ChannelRoles.index` raises on it, and every measure behind it
    #: exports blank with a status rather than a plausible zero. A defaulting branch here that
    #: forced them to `None` would restate the dataclass, and a mutation proved it: flipping it
    #: to the defaults changed nothing, because the defaults **are** `None`.
    role_fields = {"nuclei": "nuclei_channel", "cell": "cell_channel",
                   "cd68": "cd68_channel", "plaque": "plaque_channel"}
    updates = {field: session.roles[role]
               for role, field in role_fields.items() if role in session.roles}

    known = set(READ_ONLY_FIELDS) | {EDITABLE_FIELD}
    updates.update({name: value for name, value in session.chosen.items() if name in known})
    return replace(params, **updates)


def format_value(value) -> str:
    """How a parameter prints in the read-only block.

    **`None` reads as `not set`, never as a blank or a zero.** `mask_coefficient`,
    `cd68_threshold` and `plaque_threshold` are all legitimately `None` before their screens run,
    and a blank row would be read as *the value is zero* on a screen whose whole job is showing
    what is about to be written.
    """
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


#: **Item 9a never ran because the study has no CD68 channel**, and the row says so rather than
#: reading the same `not set` as a slider somebody left alone. Those are different facts: one is
#: a choice not made, the other is a choice that was never available. The user approved this
#: exact string on 2026-09-02.
NO_CD68_CHANNEL_NOTE = " — no CD68 channel assigned"


def field_value(params: SegmentationParams, name: str) -> str:
    """How one named field prints, which for `cd68_threshold` depends on a second field.

    **`format_value` cannot answer this alone** — it is handed a value, and *not set* versus
    *there was never a channel* is a fact about `cd68_channel`. Kept here rather than inside
    `format_value` so that the general formatter stays a value-to-string function.
    """
    printed = format_value(getattr(params, name))
    if name == "cd68_threshold" and params.cd68_channel is None and printed == "not set":
        return printed + NO_CD68_CHANNEL_NOTE
    return printed


def read_only_rows(params: SegmentationParams) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """The block, as `(screen title, ((field, printed value), ...))`."""
    return tuple(
        (title, tuple((name, field_value(params, name)) for name in names))
        for title, names in READ_ONLY_GROUPS
    )


def parameters_path(output_root: str, file_name: str) -> Path:
    """**Derived folder, chosen name.** The folder follows the study and is not asked.

    §4 item 8: the pool is *never chosen* because a wrong choice there is **silent** — both
    routes write valid cells and nothing looks wrong. *"That argument does not transfer."* A
    misplaced parameters file fails loudly and immediately: `measure --params F` cannot find it.
    And every consumer is a human typing a path, so `params_a3f9c21b0e44.json` would be
    predictable and unusable, while `params_conservative.json` beside `params_permissive.json`
    carries meaning two hashes do not.

    **Name only, not a path** — otherwise the derived-location half of the decision quietly
    evaporates.
    """
    name = Path(str(file_name).strip()).name
    if not name:
        raise ValueError("the parameters file needs a name")
    if not name.endswith(".json"):
        name += ".json"
    return Path(output_root) / name


def collision_message(path: Path) -> str:
    """The inline red text. **The rename is always required — no identical-content carve-out.**

    Considered and rejected 2026-08-16. *"Identical"* could never have meant byte-identical —
    `written_at` moves on every write — and **two things legitimately differ between sessions at
    the same parameters**: the `note`, and `source_image`, because semi-auto is one image per
    session. So overwriting would erase the record that the earlier session happened: three
    sessions on three images at identical parameters would leave one file naming the third image
    while the pool held cells from all three.

    **It also protects a pool that already exists.** Overwrite a file that produced cells and the
    pool's `params_hash` matches no file on disk — the provenance points at something that no
    longer describes it.
    """
    return (f"{path.name} already exists in this folder. Saving is refused so an earlier "
            f"session's record is not overwritten — change the name above.")


def hand_edit_statement(n_edits: int) -> str | None:
    """§6a's required sentence. **`None` when nothing was edited.**

    Item 4's commit warning set the precedent: it fires only when there is something to lose,
    *"because a warning that appears when there is nothing to lose is one people learn to click
    through."*

    The gate it guards is real: §6.1's reproducibility gate is about an **UNEDITED** run, and
    that word *"is the whole correction"* — with edits it cannot hold, by design, because §6a
    made nucleus edits per-image overrides that deliberately do not transfer to batch.
    """
    if n_edits <= 0:
        return None
    noun = "nucleus" if n_edits == 1 else "nuclei"
    was = "was" if n_edits == 1 else "were"
    return (f"{n_edits} {noun} on this image {was} edited by hand. Those edits are in the pool "
            f"and are not in this file.")


def measure_command(pool: Path, params_file: Path) -> str:
    """**Complete as pasted** — `--pool` and `--params` are its only required arguments."""
    return f"prestomorph measure --pool {pool} --params {params_file}"


def export_command(pool: Path, params_file: Path, params_hash: str, output_root: str) -> str:
    """**`--params-hash` is emitted, which the plan's earlier version omitted.**

    `export_pool` refuses when a pool holds more than one parameter set and names the
    competitors. A study accumulating sessions will eventually have several, so filling it in
    makes the command **correct forever rather than only the first time.**

    > ### ⇒ THE HASH WAS IN THE PATH TWICE — patch item C2, 2026-09-03
    >
    > Observed on a real run: `exports/835a98debb6f/export_20260903_835a98de/`. The outer folder
    > came from here and the inner from `export_pool`, and **both name the same run** — a
    > directory whose only child is a directory saying the same thing.
    >
    > `--out` is now the `exports` folder itself, giving `exports/export_20260903_835a98de/`.
    > `export_pool` is unchanged: it is the one that has to key its folder by the hash, because
    > a re-export at different parameters must not land on top of the first.
    """
    out = Path(output_root) / DEFAULT_EXPORT_DIR
    return (f"prestomorph export --pool {pool} --params {params_file} "
            f"--params-hash {params_hash} --out {out}")


def masks_command(pool: Path, output_root: str, params_hash: str | None = None) -> str:
    """**It carries `--params-hash` now** — patch item C1, 2026-09-03.

    > ### ⇒ THE OMISSION WAS RIGHT AT THIS MOMENT AND WRONG BY THE TIME IT IS RUN
    >
    > The old reasoning: `--params-hash` narrows to cells holding a *measurement*, and at the
    > instant this popup is shown nothing has been measured, so the pasted command would write
    > an empty folder and exit 0.
    >
    > **What that missed is that the command is pasted later.** Run after measuring — which is
    > when anyone wants skeletons — the flagless form writes the TIFFs, **no `.swc` at all**,
    > and a manifest with four blank columns in every row. Measured: 0 skeletons against 24.
    >
    > It is safe now because `export_cell_masks` resolves the set itself when none is passed, so
    > the flag is no longer load-bearing at either moment: at item 8 there is nothing to resolve
    > and the masks are written alone; later there is, and the skeletons come with them.

    **The masks already exist by the time this string is read.** They were written by
    `PoolWriter.add_cell` during the commit this popup is announcing, and no later phase alters
    them, which is why this command carries no `--params` and sits outside the ordered pair.
    """
    command = f"prestomorph masks --pool {pool} --out {Path(output_root) / DEFAULT_MASKS_DIR}"
    return command if params_hash is None else f"{command} --params-hash {params_hash}"


@dataclass
class SaveResult:
    """What the SAVED popup shows. **The cell count is why this screen does two jobs** — without
    it the popup confirms the file was written and says nothing about whether the cells landed."""

    params_file: Path
    pool_root: Path
    params_hash: str
    n_cells: int

    def measure(self) -> str:
        return measure_command(self.pool_root, self.params_file)

    def export(self, output_root: str) -> str:
        return export_command(self.pool_root, self.params_file, self.params_hash, output_root)

    def masks(self, output_root: str) -> str:
        return masks_command(self.pool_root, output_root, self.params_hash)


def commit_session(session: Session, params: SegmentationParams, *, file_name: str,
                   note: str = "", run_id: str = "semiauto") -> SaveResult:
    """Write the parameters file **and** the session's cells. The one durable act of a session.

    **The derivation path is batch's own** — `PoolWriter` → fragment → `merge_fragments` — which
    is what §6.1's reproducibility gate requires: no second write path to diverge from the one
    batch uses. `cli.py`'s standing warning is that *"a second code path for the same job is the
    most likely way the batch and interactive routes would silently diverge."*

    **Semi-auto merges its own fragment**, unlike batch. Batch separates `merge` because hundreds
    of concurrent Slurm tasks cannot write one SQLite file; **none of that applies to one image in
    one process**, and requiring a manual `prestomorph merge` afterwards would mean finishing a
    session, opening review and finding nothing.

    **The parameters file is written FIRST, and refuses rather than overwrites.** If it collided
    after the cells were written the session would be half-committed — a pool holding cells whose
    `params_hash` names a file the user was then told to rename.
    """
    path = parameters_path(session.output_root, file_name)
    if path.exists():
        raise FileExistsError(collision_message(path))
    if session.image_shape is None:
        raise ValueError("the session has no image shape; the channels were never loaded, so "
                         "there is nothing to commit")

    pool_root = Path(session.output_root) / pool_dir_name(POOL_ORIGIN, params.phase_a_hash())
    save_params(params, path, note=note, source_image=session.image_path or None)

    #: **Counted as they are written, not with `len()`.** `session.cells` is an *iterable* of
    #: `add_cell` keyword arguments, and in the real flow it is `Derivation.payloads()` — a
    #: generator, so that §4a.5's *"the memory cost is one array, not sixty"* survives all the
    #: way to the write. A `len()` here would force it into a list and put sixty cell crops
    #: beside the channels the session is already holding.
    uids: list[str] = []
    with PoolWriter(
        pool_root,
        source_path=session.image_path,
        voxel_size_um=session.voxel_size_um or (1.0, 1.0, 1.0),
        image_shape=session.image_shape,
        channel_roles=dict(session.roles),
        run_id=run_id,
        params_hash=params.hash(),
    ) as writer:
        for cell in session.cells:
            uids.append(writer.add_cell(**cell))

    merge_fragments(pool_root)
    apply_decisions(pool_root, uids, session.decisions)
    session.saved = True
    return SaveResult(params_file=path, pool_root=pool_root,
                      params_hash=params.hash(), n_cells=len(uids))


def apply_decisions(pool_root: Path, uids: list[str], decisions: dict) -> int:
    """Write item 7's accept/reject decisions onto the cells just merged.

    > ### ⇒ WITHOUT THIS, PHASE C IS EMPTY AND NOTHING SAYS SO
    >
    > `PoolWriter` writes every cell as **`CANDIDATE`**, and `pending_measurement` takes only
    > **`CONFIRMED`** unless `--include-unreviewed` is passed. So a session that reviewed every
    > cell and saved produced *"measured 0 cell(s) across 0 image(s)"* from the exact `measure`
    > command the SAVED popup hands the user, and an export of six cells with **no measurements
    > at all**.
    >
    > **Found by running one image end to end.** It is the seam between item 7 and item 8 — item
    > 7 resolves cells in memory and item 8 writes cells to a pool, and neither screen owns the
    > join, so no per-screen test could have seen it.
    >
    > **Keyed by position in label order**, which is the same order `Derivation.payloads()` yields
    > and the same order `review_cells` numbers its tiles. `add_cell` returns the uid it minted,
    > so the join is made from what was actually written rather than from a uid guessed in
    > advance.

    **A cell with no decision is left `CANDIDATE`**, deliberately: that is the honest record of a
    session where review was skipped, and `measure --include-unreviewed` still stamps every such
    row `reviewed = false` so the distinction survives to the spreadsheet.

    > ### ⇒ THE REVIEWER'S CELL COUNT TRAVELS FROM HERE — 2026-08-30
    >
    > `Decision.n_cells` is passed through to `set_status`. It was a positional unpack of a
    > 3-tuple until this change, which is where *"this is 3 cells I cannot afford to split"*
    > stopped: the editor produced the number, `_leave` dropped it on the floor, and this
    > function had no field to carry it in.
    >
    > **Read by name, not unpacked.** A 4-tuple here would work exactly as well until someone
    > inserted a field, and then it would put the note where the count goes without raising.
    """
    if not decisions:
        return 0
    from ..store import Pool

    applied = 0
    with Pool(pool_root) as pool:
        for index, uid in enumerate(uids, start=1):
            decision = decisions.get(str(index))
            if decision is None:
                continue
            pool.set_status(uid, decision.status,
                            reject_reason=decision.reason,
                            reject_note=decision.note,
                            n_real_cells_here=decision.n_cells)
            applied += 1
    return applied


# --------------------------------------------------------------------------
# The screens
# --------------------------------------------------------------------------

class SavedPopup(QDialog):
    """**The only popup in this component's normal path**, and it is a dialog over the form.

    Decision 5 closed with **no bridge to Phase C**, so the moment after saving is the moment the
    user needs to know where everything landed. *"This is the direct cost of choosing no bridge,
    and the cheapest possible way to pay it."*
    """

    close_program_requested = Signal()

    def __init__(self, result: SaveResult, output_root: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Saved")
        self.setProperty("role", "popup")
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 14, 16, 14)
        column.setSpacing(10)

        self.count_label = QLabel(
            f"{result.n_cells} cells committed to the pool.", self)
        column.addWidget(self.count_label)

        #: **Two sections, and the second is deliberately not numbered into the first.** The
        #: measure/export pair is an *order*: export reads what measure wrote, and running them
        #: the other way round produces an export with no measurements in it. Masks depend on
        #: neither — they were written during the commit this popup announces — so listing them
        #: as a third step would state a dependency that does not exist and would imply the
        #: reviewer must measure before they can open a cell in Fiji. The heading says *any
        #: time* for that reason.
        #:
        #: > ### ⇒ AND IT NEVER MENTIONED THE SKELETONS — patch item C5, 2026-09-03
        #: >
        #: > The old heading was *"For per-cell mask images (Fiji, MATLAB, Imaris) — any time,
        #: > in any order"*, and it was honest **at this moment**: nothing has been measured, so
        #: > there are no skeletons to mention.
        #: >
        #: > The command is pasted later, though, and since C1 it resolves the parameter set for
        #: > itself — so run **after** measuring it also writes both SWCs per cell, 24 of them
        #: > against a measured pool's 12 masks. The line stayed true and became incomplete, and
        #: > an output nobody was told to expect is one nobody looks for.
        #: >
        #: > **The structure does not move.** *Any time* still applies to the masks, because it
        #: > is still true of them; the skeletons are named as coming with them once the run has
        #: > been measured, which is a fact about the pool rather than a step in an order.
        self.lines: list[str] = [
            "Parameters file", str(result.params_file),
            "Pool", str(result.pool_root),
            "Then run, in this order:",
            result.measure(),
            result.export(output_root),
            "For per-cell mask images (Fiji, MATLAB, Imaris) — any time, in any order. "
            "Once the run has been measured, the same command also writes each cell's "
            "skeletons:",
            result.masks(output_root),
        ]
        #: **Every line, including `lines[0]`.** It read `self.lines[1:]` until 2026-08-31, which
        #: dropped the `Parameters file` heading and left its path bare above a labelled `Pool` —
        #: the two paths are the reason this popup exists and one of them arrived unlabelled.
        #: `test_the_popup_names_both_paths_and_both_commands` could not see it: it joins
        #: `self.lines`, which is the model, while the defect was in what the view drew from it.
        for text in self.lines:
            label = QLabel(text, self)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            column.addWidget(label)

        #: **No confirmation.** Everything is written by then, so there is nothing to lose — and
        #: a warning at the one moment when nothing is at risk is exactly the kind people learn
        #: to click through.
        self.close_button = QPushButton("Close Program", self)
        self.close_button.clicked.connect(self.close_program_requested)
        self.close_button.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.close_button)
        column.addLayout(row)


class SaveParametersScreen(QWidget):
    """Item 1's shape: one stacked form, no viewer, SAVE centred with red text beneath."""

    saved = Signal(object)
    back_requested = Signal()
    close_program_requested = Signal()

    def __init__(self, flow: Flow | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.SAVE)
        self.result: SaveResult | None = None

        #: > ### ⇒ THE BUTTONS COULD NOT BE REACHED ON BUNYA — patch item B4, 2026-09-03
        #: >
        #: > This was one flat `QVBoxLayout` on the screen itself, carrying 23 read-only
        #: > parameter rows plus headings, a divider, two editable controls, the hand-edit
        #: > sentence, SAVE, the Back row and an error label. On the Bunya desktop the bottom
        #: > falls off the screen and **SAVE cannot be reached at all** — on the one screen
        #: > whose entire purpose is the button at the bottom of it.
        #: >
        #: > **The house pattern, used twice already** — `frames.py`'s tile grid and
        #: > `widgets.py`'s selection list: `setWidgetResizable(True)`, the horizontal bar off,
        #: > and `QFrame.Shape.NoFrame` so the scroll area draws no border of its own.
        #: >
        #: > **Scope is item 8 only, by the user's decision.** `FrameC.right` is also a plain
        #: > `QVBoxLayout` with no scroll, so the control columns on items 2, 4, 5, 9a and 9b
        #: > can hit the same wall — item 4 carries the most controls. Recorded as a known hole
        #: > and deliberately not fixed here.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QWidget(self)
        self.scroll = QScrollArea(self)
        self.scroll.setWidget(body)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #: **Always on since patch item D**, when the same scroll went onto every other screen.
        #: It was `AsNeeded` here; the user's decision of 2026-09-06 was that all thirteen
        #: screens show the bar, on item 7a's reasoning — wheel and pinch are reserved for zoom,
        #: so a bar that is only sometimes there is a bar a reviewer cannot rely on grabbing.
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(self.scroll)

        #: **Held, because it is the screen's real layout.** `screen.layout()` is the outer one
        #: now, and a test walking the form's order needs the column rather than the wrapper.
        self.column = column = QVBoxLayout(body)
        column.setContentsMargins(24, 18, 24, 18)
        column.setSpacing(10)
        column.addWidget(screen_title("save parameters"))

        block_help = QHBoxLayout()
        block_help.addStretch(1)
        #: **A fifth exception to the controls-only rule, kept deliberately.** The individual
        #: values each name their screen; *"why can I not edit any of these?"* is a question
        #: about the **block**, and nothing else on the screen answers it.
        block_help.addWidget(HelpButton(
            "The parameters above",
            "Everything you chose on the screens named above. It cannot be edited here: the "
            "cells already exist and were made with exactly these numbers, so a file claiming "
            "different ones would not describe them.\n\n"
            "To change one, run the session again.", self))
        column.addLayout(block_help)

        self.value_labels: dict[str, QLabel] = {}
        for title, names in READ_ONLY_GROUPS:
            heading = QLabel(title, self)
            column.addWidget(heading)
            for name in names:
                row = QHBoxLayout()
                row.addSpacing(16)
                row.addWidget(QLabel(name, self))
                row.addStretch(1)
                value = QLabel("", self)
                self.value_labels[name] = value
                row.addWidget(value)
                column.addLayout(row)

        #: §4 item 8's divider carries the plan's own distinction: **above it parameters, below
        #: it file metadata.** One group changes what the pipeline computes; the other only
        #: describes the artefact.
        self.divider = QLabel("", self)
        self.divider.setFixedHeight(1)
        self.divider.setStyleSheet(f"background: {theme.CONTROL};")
        column.addWidget(self.divider)

        #: **No evidence line beside it — the user declined to show it, 2026-08-17.** The
        #: evidence requirement did not retire, it moved with the other parameter: the 73 µm³
        #: ceiling is shown beside the floor on 9b, where the value is picked against a picture.
        #: Here there is no picture.
        self.min_spur_um = self._field(
            column, EDITABLE_FIELD, "0.0",
            "The shortest branch the skeleton keeps. Anything shorter is pruned before the "
            "morphology measures are computed.\n\n"
            "It is editable here because no screen asks for it — there is nothing to look at "
            "that would show you 1 micrometre of spur.\n\n"
            "Used at measurement time, not now.\n\n"
            "It opens at 0, which prunes nothing. The 1.0 this used to default to was derived "
            "twice and both derivations have been superseded, so no filter is applied unless "
            "you ask for one.")

        #: **Directly under the spur length, at the user's instruction 2026-09-01.** The two are
        #: the same kind of decision — what the skeleton keeps — and they are the only two
        #: measurement choices this screen makes.
        self.loop_prune_method = self._choice(
            column, LOOP_FIELD, LOOP_CHOICES,
            "What to do when the skeleton closes a loop.\n\n"
            "Loops are always counted, whatever this says. Pruning one means cutting one arm "
            "of it, and this chooses which arm: the shortest, or the one running through the "
            "faintest Iba1 signal.\n\n"
            "It opens at 'none' because a loop may be two processes of the same cell genuinely "
            "touching, which is a real feature rather than noise. Fiji's AnalyzeSkeleton "
            "defaults the same way.\n\n"
            "This applies to the soma-free skeleton only. The soma-anchored one is a tree and "
            "cannot contain a loop.")

        self.file_name = self._field(
            column, "File name", DEFAULT_FILE_NAME,
            "The name of the parameters file. The folder is fixed and shown above.\n\n"
            "If a file of that name already exists, saving is refused and you rename it here. "
            "Every save gets its own file, so a later session never overwrites an earlier "
            "one's record.")

        self.note = self._field(
            column, "Note (optional)", "",
            "Optional free text, and never required.\n\n"
            "It is copied into every export and shown in export_summary.csv.\n\n"
            "Six months on, the file says what the values were. The note is where you say why.")

        #: **Only when edits exist.** Item 4's commit warning set the precedent.
        self.hand_edits = QLabel("", self)
        self.hand_edits.setWordWrap(True)
        column.addWidget(self.hand_edits)

        column.addStretch(1)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self._on_save)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.save_button)
        button_row.addStretch(1)
        column.addLayout(button_row)

        #: > ### BACK ONLY, AND IT DISAPPEARS ONCE SAVE HAS BEEN PRESSED — 2026-08-30
        #: >
        #: > **There has never been a Continue here** — item 8 is last in `FLOW_ORDER`, and
        #: > *"SAVE is the only control that goes forward"*. So this is the one screen that takes
        #: > the shared row's back-only shape.
        #: >
        #: > **The user's rule: *"no back after save has been pressed."*** Before SAVE nothing
        #: > exists in the pool and going back costs only the session; after it the parameters
        #: > file and the cells are on disk, and a Back that walked away from a written run would
        #: > leave the file describing cells the session no longer holds.
        self.navigation = NavigationRow(back=True, forward=False, parent=self)
        self.navigation.back_clicked.connect(self.back_requested)
        column.addWidget(self.navigation)
        self.back_button = self.navigation.back_button

        #: **Inline, not a dialog** — matching item 1, where failures are red text under
        #: CONTINUE. Cheaper than a modal for something that fires on every session from the
        #: second onward, since the default name is fixed deliberately.
        self.error = QLabel("", self)
        self.error.setProperty("role", "error")
        self.error.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.error.setWordWrap(True)
        column.addWidget(self.error)

        self.refresh()

    # -- construction ---------------------------------------------------

    def _field(self, column, label: str, value: str, help_text: str) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label, self))
        row.addStretch(1)
        edit = QLineEdit(value, self)
        edit.setFixedWidth(200)
        row.addWidget(edit)
        row.addWidget(HelpButton(label, help_text, self))
        column.addLayout(row)
        return edit

    def _choice(self, column, label: str, choices, help_text: str) -> QComboBox:
        """A dropdown built the same way as `_field`, so the two rows line up.

        **The stored value rides on the item, not on its position.** Reading the selection by
        index would make the enum's order load-bearing in a second place, and a reordering of
        `LOOP_CHOICES` for readability would then silently change what a run measured.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(label, self))
        row.addStretch(1)
        box = QComboBox(self)
        for text, value in choices:
            box.addItem(text, value)
        box.setFixedWidth(200)
        row.addWidget(box)
        row.addWidget(HelpButton(label, help_text, self))
        column.addLayout(row)
        return box

    # -- state ----------------------------------------------------------

    def params(self) -> SegmentationParams:
        """What SAVE would write, including the one value typed on this screen."""
        params = session_params(self.flow.session)
        loop = self.loop_prune_method.currentData()
        try:
            spur = float(self.min_spur_um.text())
        except ValueError:
            #: The typed field can be mid-edit and unparseable; the dropdown never can, so it
            #: is still carried rather than being lost to its neighbour's error.
            return replace(params, loop_prune_method=loop)
        return replace(params, min_spur_um=spur, loop_prune_method=loop)

    def refresh(self) -> None:
        """Fill the read-only block and the hand-edit sentence from the session.

        **Both editable controls open at the shipped default every run — `min_spur_um` 0.0 and
        `loop_prune_method` none — not at whatever the last session chose.** Pre-filling from
        the most recent parameters file was proposed and withdrawn: it would have been the one
        piece of cross-run state left standing, and invisible state is what §4 item 1's
        *nothing is remembered between runs* removes.
        """
        params = session_params(self.flow.session)
        for name, label in self.value_labels.items():
            label.setText(field_value(params, name))

        statement = hand_edit_statement(self.flow.session.nucleus_edits)
        self.hand_edits.setText(statement or "")
        self.hand_edits.setVisible(statement is not None)

    def _on_save(self) -> None:
        """SAVE is the only control that goes forward.

        **Refuses rather than overwrites**, and the rename is offered in place — *"the block is
        the safe default and the rename is the way through it, in one place"* — rather than
        sending the user back a step.
        """
        try:
            spur = float(self.min_spur_um.text())
        except ValueError:
            self.error.setText(f"{EDITABLE_FIELD} must be a number.")
            return
        if spur < 0:
            self.error.setText(f"{EDITABLE_FIELD} cannot be negative.")
            return

        try:
            result = commit_session(
                self.flow.session, self.params(),
                file_name=self.file_name.text(), note=self.note.text())
        except FileExistsError as exc:
            self.error.setText(str(exc))
            self.file_name.setFocus()
            return
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            return

        self.error.setText("")
        self.result = result
        self.flow.save()
        #: **Back goes when the run becomes durable** — the user's rule, 2026-08-30: *"no back
        #: after save has been pressed."* Before this line nothing exists in the pool and going
        #: back costs only the session; after it the parameters file and the cells are on disk,
        #: and a Back that walked away would leave a file describing cells the session no longer
        #: holds. **Disabled rather than hidden**, so the screen does not reflow under the
        #: pointer at the moment the user is reading the result.
        self.navigation.set_back_enabled(False)
        self.saved.emit(result)

    def show_saved(self) -> SavedPopup:
        """The completion dialog. Built here so a caller does not assemble it from the result."""
        popup = SavedPopup(self.result, self.flow.session.output_root, self)
        popup.close_program_requested.connect(self.close_program_requested)
        return popup
