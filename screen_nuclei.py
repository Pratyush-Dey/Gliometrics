"""Screen 4 — Microglial Nuclei Detection. `COMPONENT_03_PLAN.md` §6h, §4 item 4.

**The hardest screen, and the only expensive one — Cellpose is 200–500 s on CPU.** Frame C, four
layers, three of which draw immediately: DAPI-minus-veto came from item 2, Iba1 raw is in memory,
and the threshold-1 mask was committed at item 3. **Only the label layer waits.**

**It is also the only screen with a Back button**, and it returns to item 3. Nucleus edits survive
that trip — `detect_nuclei` never receives the cell mask, so changing `cell_coefficient` moves
only *which* nuclei are microglial.

> ### ONE CONTROL IS LIVE AND THE REST COST 500 SECONDS — AND THE MECHANISM IS WHY
>
> `membership_threshold` **recomputes nothing about the labels.** `membership_fractions` is two
> `bincount` passes and runs once at commit; moving the slider only re-thresholds that array and
> **rebuilds the colormap dict** — a lookup over ~130 ids. The nuclei do not change; which palette
> each one draws from does. That is what `fractions_at_commit` holds, and it is why the live
> control sits at the TOP of the column, 500 seconds away from the ones below it.

> ### TWO THINGS HERE CANNOT BE GOT WRONG QUIETLY
>
> **Labels do not project.** Max-projecting ids 3 and 7 yields 7 — *a different nucleus*, not a
> merged one. So above z-batch depth 1 the label layer disappears and the three image layers
> remain. One control carries it, and its "?" says so.
>
> **A split is ONE operation, not delete-plus-two-adds.** Recorded as mechanism, the record would
> read three operations and the fact that it was a split would be lost — along with the
> under-split rate, which is the main QC number for this pipeline's known weakest point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from scipy import ndimage

from ..membership import membership_fractions
from . import gestures, palette
from .palette import ring_colour
from .gestures import MARKER_COLOUR, MARKER_LAYER, SplitMarkers
from .frames import CanvasHost, FrameC
from .preview import contrast_limits, log_view
from .semiauto import Flow, Session, Step, parsed_numbers
from ..cache import WindowCache
from .viewer import (
    BATCHED, ROLLING, CanvasLayout, ViewerPort, batch_plane, cached_plane, outline_of_batch,
    step_window, z_window_range, z_window_word,
)
from .widgets import (
    ColourRow, CommitButton, ConfirmPopup, HeaderStrip, HelpButton, LayerToggles,
    NavigationRow, ParameterSlider, batch_step_box,
)

__all__ = [
    "NucleiScreen", "LAYERS", "PICKER_LAYERS", "TOGGLES", "LABEL_LAYER",
    "OUTLINE_LAYER", "LOG_DAPI", "log_display", "microglial_only",
    "microglial_readout", "detection_progress", "labels_shown_at_depth",
    "ring_colour", "label_colours", "label_alphas", "microglial_ids",
    "SplitMarkers", "Operation", "OperationRecord", "OUTLINE_CONTOUR_PX",
    "working_scale", "click_frames", "grow_nucleus", "edited_labels",
]

#: The label layer's name. **It is not in `PICKER_LAYERS` and cannot be**: its colouring is a
#: per-label *scheme* driven by a `DirectLabelColormap` dict, not a single hue — so §4's
#: layer-appearance rule excludes it from both the picker and the translucency slider, and the
#: exclusion is one rule rather than two because **alpha carries meaning here**.
LABEL_LAYER = "nucleus labels"

#: The split markers, drawn while SHIFT is held. **Not in `PICKER_LAYERS`** for the same reason
#: the labels are not: it is a transient of the gesture, not one of the screen's data layers, and
#: it disappears the moment the user lets go.


#: §6h.8's table. **Both Iba1 layers start OFF**, on §4's own wording: *"the user identifies the
#: soma by **switching the Iba1 layers on**"*. That keeps the opening view to the two things the
#: screen is about — the nuclei, and which of them are microglial.
#:
#: `None` colour and `None` alpha mark the label layer as a scheme rather than a choice.
#: **The outline is its own layer as of 2026-08-22 — the user's decision.** It was a toggle
#: driving `Labels.contour` on `LABEL_LAYER` itself, and napari's `contour` *replaces* the
#: shaded region rather than adding to it, so **turning the outline on turned the fills off**.
#: Their words: *"currently turning that on turns off the lables layer… i want it to become its
#: own layer called microglial nuclei outline."* Two layers means the two can be on together.
OUTLINE_LAYER = "microglial nuclei outline"

#: **A display transform, not a layer, and not a pipeline change.** Detection runs on
#: `nuclei.normalize_for_detection(method="log")`; the screen drew DAPI **linearly**. Measured on
#: `5xFAD/1234/program training crop2.tif`: p90 intensity renders at **2.9% grey** while the
#: detector sees **70.8%**, and **35.9% of voxels inside a confident nucleus core render below 5%
#: grey**. So a label reaching into DAPI that is genuinely there looks like it is reaching into
#: nothing. This toggle draws DAPI through the detector's own transform so the two can be
#: compared by eye. **Off at arrival** — the linear view stays the default.
LOG_DAPI = "log DAPI"

LAYERS: tuple[tuple[str, str | None, float | None, bool], ...] = (
    (LABEL_LAYER, None, None, True),
    ("DAPI -veto", "blue", palette.CHANNEL_ALPHA["DAPI"], True),
    ("Iba1 raw", "white", palette.CHANNEL_ALPHA["Iba1"], False),
    ("Iba1 mask", "white", palette.DERIVED_MASK_ALPHA, False),
    (OUTLINE_LAYER, None, None, False),
)

#: Three of the five. Same swatch rows as item 2. **Both label layers carry `None`** and are
#: excluded by construction: their colouring is a per-id scheme, not a choice.
PICKER_LAYERS: tuple[tuple[str, str, float, bool], ...] = tuple(
    (name, colour, alpha, on) for name, colour, alpha, on in LAYERS if colour is not None
)

#: Six checkboxes — the five layers plus `log DAPI`, which is a display transform rather than a
#: layer and is the one non-layer entry in this block.
TOGGLES: tuple[tuple[str, bool], ...] = (
    (LABEL_LAYER, True), ("DAPI -veto", True), ("Iba1 raw", False), ("Iba1 mask", False),
    (OUTLINE_LAYER, False), (LOG_DAPI, False),
)

#: §6h.5. napari's own docstring: *"if greater than 0, displays contours of labels **instead of**
#: shaded regions"* — so it **replaces** the fill rather than adding to it. That is why the
#: outline is now a **second layer** rather than a property of the first, and why `LABEL_LAYER`
#: is pinned at `contour=0`: the fills are what make *microglial versus not* readable at a
#: glance, and they must survive the outline being switched on.
OUTLINE_CONTOUR_PX = 2

#: §4's drawn legend, verbatim. It is the only place the gestures are written down.
GESTURE_LEGEND = """All editing happens one slice at a time, at z-batch depth 1.

  DOUBLE-CLICK a nucleus        delete it — the whole 3D nucleus, not just this slice

  DOUBLE-CLICK the background   add a nucleus, centred on this slice

  SHIFT + CLICK each part,      split one nucleus into that many. Markers show while
  then RELEASE SHIFT            SHIFT is held. One marker does nothing.

  u                             undo, one step per press

Click near the middle of a nucleus in z. A nucleus spans about 13 slices, so a seed \
placed near the top sits half outside it."""


# --------------------------------------------------------------------------
# Pure logic — no Qt, no napari
# --------------------------------------------------------------------------

def microglial_readout(n_microglial: int, n_detected: int) -> str:
    """`microglial: 51 of 131`, right-aligned above the canvas. Updates **live**.

    Both halves are needed. *"51"* alone says nothing about whether detection worked, and the
    ratio is the thing the threshold is being moved against.
    """
    return f"microglial: {n_microglial} of {n_detected}"


def detection_progress(done: int, total: int) -> str:
    """`…detecting 23 of 40`, under Commit.

    **The wording is a collision fix, not a style choice.** Cellpose progress read *"slice 23 of
    40"* and the z indicator six lines above reads *"z slice (23/40)"* — the same shape, entirely
    different meanings. Found by drawing the screen; eight sessions of prose had not put the two
    strings next to each other.
    """
    return f"…detecting {done} of {total}"


def labels_shown_at_depth(depth: int) -> bool:
    """**Depth 1 only, and this is structural rather than a simplification.**

    Max-projecting label ids 3 and 7 yields 7 — *a different nucleus*, not a merged one. So the
    three image layers project and the label layer cannot, which composes with editing being
    per-slice: at depth 1 the labels are shown and editable, above it they disappear.
    """
    return depth <= 1


def plane_scale(scale):
    """A `(z, y, x)` layer scale as the `(y, x)` one a 2D layer takes.

    **The z factor is dropped rather than kept as 1.** napari matches the scale tuple to the
    array's dimensions, so handing a three-element scale to a two-dimensional layer registers
    the picture against the wrong axes — the y factor lands on x and the layer is displaced by
    exactly the downsample factor. `None` stays `None`, which is the native-resolution case.
    """
    return None if scale is None else tuple(float(v) for v in scale)[-2:]


def working_scale(native_shape, working_shape):
    """The layer scale that registers a working-resolution volume over the native image.

    > ### COMPONENT_03_PLAN 4a.1's COORDINATE RULE, AND IT SAYS THIS EXACTLY
    >
    > *"Nucleus detection, membership, objects and splitting all run at 1152x1152... Decided
    > 2026-08-13: nucleus labels are displayed and edited at WORKING resolution. The DAPI image
    > is shown at native and the labels layer carries `scale=(1, 2, 2)`, so napari registers
    > them correctly on screen while the array stays in the frame the pipeline wants."*
    >
    > **The factor is read off the two arrays rather than plumbed through as a parameter.** A
    > `downsample` carried alongside can disagree with the data - a stand-in that answers at
    > native resolution while the session says 2 would place the labels at half scale over an
    > image they match exactly. The shapes cannot lie about themselves.
    >
    > **Rounded to an integer, not left as the ratio.** `downsample_xy` truncates the last
    > partial block, so a 220-wide field at factor 3 comes back 73 wide and the ratio is 3.014
    > - but the working voxel really is exactly 3 native voxels across, and the difference is
    > the dropped edge rather than a stretch. The integer is the true mapping; and it is also
    > the multiplication 4a.1 requires at the point a gesture is captured.

    `None` when the two agree, which is what every native-resolution caller gets and what the
    viewer reads as *leave the layer alone*.
    """
    if native_shape is None or working_shape is None:
        return None
    if len(native_shape) < 3 or len(working_shape) < 3:
        return None
    factors = []
    for native, working in zip(native_shape[-2:], working_shape[-2:]):
        factors.append(max(1, round(float(native) / float(working))) if working else 1)
    if factors[0] == 1 and factors[1] == 1:
        return None
    return (1.0, float(factors[0]), float(factors[1]))


def click_frames(position, canvas_scale: float = 1.0, downsample: int = 1):
    """One canvas click -> **(index into the label volume, native full-image voxel)**.

    > ### COMPONENT_03_PLAN 4a.1's COORDINATE RULE, AND IT EXISTS BECAUSE THIS HAS ALREADY GONE
    > ### WRONG ONCE
    >
    > *"Three coordinate frames are live in this pipeline at once, and they disagree... A point
    > written in one and read as another is off by a factor of two, or by the crop offset - and
    > the resulting number is always plausible, so nothing downstream can catch it."* The rule:
    > **store NATIVE FULL-IMAGE VOXEL COORDINATES, and convert once, at the point of capture.**
    > This function is that point for item 4; nothing downstream converts anything.
    >
    > It cost `avg_nearest_neighbour_um` **doubling on every row ever written** the last time it
    > was got wrong - two cells 2.166 um apart exported as 4.332 um, a completely believable
    > distance between two microglia.

    **Two conversions, not one, because the canvas frame is not fixed.** The label volume is
    always at working resolution - that is `Compute.detect`'s contract. The *canvas* is at
    working resolution too in this build, and would be at native under the arrangement 4a.1
    describes; `canvas_scale` is the label layer's own scale and is exactly that difference.

        index  = position / canvas_scale        canvas -> the label array
        native = index * downsample             the label array -> the image on disk

    **`z` is never scaled.** `downsample` is an xy reduction: `voxel_size_after_downsample`
    leaves the z spacing alone and `downsample_xy` leaves the z axis alone.
    """
    scale = float(canvas_scale) if canvas_scale else 1.0
    factor = max(1, int(downsample))
    z, y, x = (float(c) for c in tuple(position)[-3:])
    index = (int(round(z)), int(round(y / scale)), int(round(x / scale)))
    native = (index[0], index[1] * factor, index[2] * factor)
    return index, native


def grow_nucleus(dapi: np.ndarray, seed, *, diameter_px: float,
                 voxel_size_um=(1.0, 1.0, 1.0)) -> np.ndarray:
    """**The user places a seed at the middle of a nucleus; this finds the nucleus.**

    > ### A SEED, NOT A STAMP - the user's rule, 2026-08-20
    >
    > PrestoCell writes a **fixed square block** of side `nucleus_diameter / 2` wherever you
    > double-click (`main.py:836-841`) - so a hand-added nucleus is a box, and its volume and
    > shape are the parameter rather than the cell. This program does not: *"the user is to add
    > a seed to the centre of a nucleus in xy and z, and from that our program should find the
    > nucleus."*
    >
    > **It is the same contract the split already has.** Shift-clicking each part places seeds
    > and `split_by_watershed` finds the parts; placing one seed and having the nucleus found is
    > that rule with one marker. The user marks *which* object; the program decides *what* it is.

    The window is bounded at twice the expected diameter, which is what makes this cheap enough
    to run inside a double-click: a nucleus is one object, and a threshold over the whole stack
    would be both slower and worse - a local Otsu separates one blob from its own background
    rather than from the brightest nucleus in the field.

    **`nucleus_min_volume_voxels` is deliberately NOT applied here.** That floor is the
    *detector's* - *"deletes detected nuclei smaller than this many voxels"* - and a hand-placed
    seed is the user overriding the detector on exactly the cases it got wrong. Holding the
    override to the filter it is overriding made a small nucleus unaddable **and said nothing**:
    the found blob was discarded and a ball of the expected diameter stamped in its place, so
    pointing at a real 75-voxel nucleus produced a fabricated 1550-voxel sphere. A speck
    honestly drawn is better than a plausible invention; the user can see it and press `u`.

    **When the seed lands on nothing at all, a ball of the expected diameter is used instead.** The
    user has asserted a nucleus is there and the gesture's contract is to add one; refusing
    silently would leave a double-click that sometimes does nothing, with no way to tell that
    case from a missed click. See HANDOFF section 6, task 25, for the alternative.
    """
    from skimage.filters import threshold_otsu

    volume = np.asarray(dapi)
    seed = tuple(int(round(c)) for c in seed)
    if any(c < 0 or c >= dim for c, dim in zip(seed, volume.shape)):
        return np.zeros(volume.shape, bool)

    #: > ### THE WINDOW WAS A CUBE IN VOXELS, AND VOXELS ARE NOT CUBES — task 26
    #: >
    #: > `diameter_px` is an **xy** measure: `diameter_in_pixels` divides by `(y_um + x_um)/2`
    #: > and leaves z alone. Using it as a half-width on all three axes made the search window
    #: > **the whole stack in z** — 35 px is 7.6 um across in xy and 21 um in z at this study's
    #: > working voxel — so `ndimage.label` was free to follow DAPI through every slice, and a
    #: > hand-placed seed grew a nucleus spanning far more planes than a nucleus has. The
    #: > fallback ball had it too: 17.5 voxels is 3.8 um in xy and 10.5 um in z.
    #: >
    #: > Reported by a reviewer: *"the seeds are way too big... it spans many more z planes."*
    #: >
    #: > **Every other place in this codebase already does this** — `veto_mask_at_margin` and
    #: > `fill_soma_holes` both size their elements in microns and say why: an isotropic element
    #: > in voxels reaches further in z than in plane, by exactly the anisotropy.
    z_um, y_um, _x_um = (float(v) for v in voxel_size_um)
    z_per_xy = (y_um / z_um) if z_um > 0 else 1.0
    radius_xy = max(1, int(round(float(diameter_px))))
    radius_z = max(1, int(round(float(diameter_px) * z_per_xy)))
    radii = (radius_z, radius_xy, radius_xy)
    lo = [max(0, c - r) for c, r in zip(seed, radii)]
    hi = [min(dim, c + r + 1) for c, r, dim in zip(seed, radii, volume.shape)]
    window = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    local_seed = tuple(c - o for c, o in zip(seed, lo))

    found = np.zeros(window.shape, bool)
    finite = window[np.isfinite(window)]
    if finite.size and float(finite.max()) > float(finite.min()):
        try:
            level = float(threshold_otsu(finite))
        except ValueError:                                   # a window of one value
            level = float(finite.max())
        foreground = window > level
        if foreground[local_seed]:
            labelled, _n = ndimage.label(foreground)
            found = labelled == labelled[local_seed]

    if not found.any():
        #: The ball, in the same window, so the fallback cannot escape the bound above.
        #: **An ellipsoid in voxels, which is a sphere in micrometres.** A ball of equal
        #: voxel radius on every axis is 2.8x too tall in z at this study's working voxel.
        grid = np.ogrid[tuple(slice(0, n) for n in window.shape)]
        half_xy = max(1.0, float(diameter_px) / 2.0)
        semi = (max(1.0, half_xy * z_per_xy), half_xy, half_xy)
        distance = sum((((axis - centre) / extent) ** 2
                        for axis, centre, extent in zip(grid, local_seed, semi)))
        found = distance <= 1.0

    mask = np.zeros(volume.shape, bool)
    mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = found
    return mask


def edited_labels(labels: np.ndarray, operations, *, dapi=None, diameter_px: float = 8.0,
                  downsample: int = 1, voxel_size_um=(1.0, 1.0, 1.0)) -> np.ndarray:
    """The committed detection **with the human's edits applied**, replayed from the record.

    > ### THE EDITS WERE RECORDED AND THE PICTURE NEVER CHANGED - task 25
    >
    > `delete_nucleus`, `add_nucleus` and `release_shift` each appended to `OperationRecord` and
    > called `draw()`, and `draw()` pushed the **committed** label volume. So a double-click on
    > a nucleus recorded a deletion and the nucleus stayed on screen; the readout kept answering
    > for the detection as it arrived. Every gesture in the legend was inert in the one way the
    > user would notice first.

    **Replayed rather than applied in place, and that is what keeps undo honest.**
    `OperationRecord`'s docstring claims the record and the undo stack are one mechanism *"which
    guarantees undo and the persisted history cannot disagree"* - true only if the picture is a
    function of the record. Undo is then `operations.pop()` and a recompute, with no inverse
    operation to write and no way for the two to drift.

    **The record is in NATIVE coordinates and this works in the label volume's**, so every point
    comes back through `downsample`. That is the one place the conversion is undone, and it is
    the mirror of `click_frames`.
    """
    from ..split import split_by_watershed

    result = np.asarray(labels).copy()
    factor = max(1, int(downsample))
    for operation in operations:
        if operation.kind == "delete" and operation.label:
            result[result == int(operation.label)] = 0

        elif operation.kind == "add" and operation.points:
            z, y, x = operation.points[0]
            seed = (int(z), int(y) // factor, int(x) // factor)
            grown = (grow_nucleus(dapi, seed, diameter_px=diameter_px,
                                  voxel_size_um=voxel_size_um)
                     if dapi is not None else None)
            if grown is None or not grown.any():
                continue
            #: **A new id above every existing one**, never a gap being reused: the ids are what
            #: the operations record names, and reusing one would make an undone deletion and a
            #: later addition indistinguishable in the history.
            result[grown & (result == 0)] = int(result.max()) + 1

        elif operation.kind == "split" and operation.label:
            target = result == int(operation.label)
            if not target.any():
                continue
            markers = np.zeros(result.shape, np.int32)
            for number, (z, y, x) in enumerate(operation.points, start=1):
                point = (int(z), int(y) // factor, int(x) // factor)
                if all(0 <= c < dim for c, dim in zip(point, result.shape)):
                    markers[point] = number
            if int(markers.max()) < 2:
                continue
            intensity = dapi if dapi is not None else target.astype(np.float32)
            #: **The program-wide splitter, settled 2026-08-18** — *"it takes markers as an
            #: array and does not care where they came from"*, so a reviewer's clicks are a
            #: valid marker array. It returns a label volume over the object, and **every voxel
            #: is assigned to exactly one marker, so a split never discards material.**
            parts = split_by_watershed(np.asarray(intensity, dtype=np.float32), target, markers,
                                       voxel_size_um=voxel_size_um)
            present = [int(m) for m in np.unique(parts) if m]
            if len(present) < 2:
                continue
            #: **No clearing first.** `split_by_watershed` assigns *every* voxel of the
            #: object to exactly one marker — *"a split never discards material"* — so each one
            #: is overwritten below. A `result[target] = 0` here was dead: a mutation deleting
            #: it could not be made to fail, which is what said so.
            next_id = int(result.max())
            for marker in present:
                next_id += 1
                result[(parts == marker) & target] = next_id
    return result


def label_colours(ids) -> dict[int, str]:
    """id -> hex, for `DirectLabelColormap`. **Hue is identical between the two classes.**

    §6d constraint 1 — *highlighting must MODULATE per-label colour, never replace it* — is
    satisfied by construction here: because the hue is untouched, a fused label is still uniform
    and a touching pair still shows a seam. Collapse every microglial nucleus to one colour and
    the split gesture stops being usable, because you cannot tell when to reach for it.
    """
    return {int(i): ring_colour(int(i)) for i in ids}


def label_alphas(ids, microglial) -> dict[int, float]:
    """1.0 for microglial, `MUTED_LABEL_ALPHA` for the rest. **Alpha only — never desaturation.**

    *Desaturation attacks the very thing §6d constraint 2 protects* — washing a colour toward
    grey is precisely how per-label distinctness is lost — while alpha attacks brightness, which
    is the property being changed.

    **And it works only because every hue sits at the same lightness**: 0.380 at full strength,
    so *dim* has exactly one meaning on this layer. On any lightness-varying palette it would
    have none — napari's own label colormap spans 0.008 to 0.877.

    **The muted value dropped from 0.6 to 0.25 on 2026-08-20**, on a reviewer's eye against real
    data. See `palette.MUTED_LABEL_ALPHA` for why the measurement behind 0.6 was answering a
    question this screen does not ask.
    """
    members = {int(i) for i in microglial}
    return {int(i): (1.0 if int(i) in members else palette.MUTED_LABEL_ALPHA) for i in ids}


def microglial_only(labels: np.ndarray, microglial, n_ids: int) -> np.ndarray:
    """The label volume with every **non**-microglial id erased, ids otherwise untouched.

    **Ids are preserved rather than renumbered**, which is what keeps `label_colours` giving the
    outline the same hue the fill has. A nucleus does not change colour when you switch the
    outline on; it gains a boundary in the colour it already had. Renumbering would recolour
    every one of them and make the two layers impossible to read together.

    A lookup table rather than `np.isin`: this runs on a `(40, 1152, 1152)` volume on every
    redraw, and one fancy-index pass is what that costs.
    """
    table = np.zeros(int(n_ids) + 1, dtype=labels.dtype)
    members = np.asarray(microglial, dtype=np.int64)
    members = members[(members >= 1) & (members <= int(n_ids))]
    table[members] = members
    return table[labels]


#: **`log_display` is `preview.log_view`** since 2026-08-30, when item 2 became the third caller
#: and task 13's rule moved the shared `log1p` out of both screens. The name is kept because this
#: screen's toggle and its tests both use it, and a rename would be churn for nothing.
log_display = log_view


def microglial_ids(fractions: np.ndarray, threshold: float) -> np.ndarray:
    """Which label ids pass, from fractions computed once. **The live half of the screen.**

    ``fractions[i]`` describes label ``i + 1``, matching `membership_fractions`' own indexing.
    A test asserts this selects **exactly** `filter_nuclei_by_membership(...).kept`, because the
    screen must highlight the nuclei the pipeline will actually keep, not a near-enough set.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return np.arange(1, fractions.size + 1, dtype=np.int64)[fractions >= threshold]


@dataclass(frozen=True)
class Operation:
    """One thing the human did. `kind` is `delete`, `add` or `split`."""

    kind: str
    label: int | None = None
    points: tuple[tuple[int, int, int], ...] = ()


@dataclass
class OperationRecord:
    """§4a.1's record, in session memory — **and the undo stack, which is one mechanism.**

    Holding both jobs in one structure is what guarantees undo and the persisted history cannot
    disagree. Undo is `u`, stepping back exactly one operation per press.

    **It buys a number nothing else can produce**: how often a human had to intervene, and in
    what way. Store only the final state and the intervention rate is unanswerable.
    """

    operations: list[Operation] = field(default_factory=list)

    def delete(self, label: int) -> Operation:
        """Double-click a nucleus. Removes the **whole 3D label**, not just the slice shown."""
        return self._record(Operation("delete", label=int(label)))

    def add(self, z: int, y: int, x: int) -> Operation:
        """Double-click background. The seed is **centred on the clicked z**.

        A nucleus spans ~13 slices at `nucleus_diameter_um = 7.6` and 0.6 µm slices, so a seed
        placed near a nucleus's top sits half outside it and scores badly on membership.
        """
        return self._record(Operation("add", points=((int(z), int(y), int(x)),)))

    def split(self, label: int, points) -> Operation:
        """**ONE operation carrying every marker**, however it is implemented underneath.

        Recorded as delete-plus-two-adds the record would read three operations and the fact that
        it was a split would be lost — with it `human_edited` kind `split`, and with that the
        under-split rate. **Intent, not mechanism.**
        """
        marks = tuple((int(z), int(y), int(x)) for z, y, x in points)
        if len(marks) < 2:
            raise ValueError("a split needs at least two markers; one does nothing")
        return self._record(Operation("split", label=int(label), points=marks))

    def _record(self, operation: Operation) -> Operation:
        self.operations.append(operation)
        return operation

    def undo(self) -> Operation | None:
        """One step per press. `None` when there is nothing left to undo."""
        return self.operations.pop() if self.operations else None

    def kinds(self) -> list[str]:
        return [operation.kind for operation in self.operations]

    def has_edits(self) -> bool:
        return bool(self.operations)

    def __len__(self) -> int:
        return len(self.operations)


def discard_warning(n_edits: int) -> str:
    """What Commit is about to throw away, named rather than implied.

    **It fires only when edits exist.** A warning that appears when there is nothing to lose is
    one people learn to click through, and then it is not there when it matters.
    """
    plural = "" if n_edits == 1 else "s"
    return (f"Committing re-runs detection and produces an entirely new set of nuclei. "
            f"Your {n_edits} hand edit{plural} will be lost.\n\n"
            "Going back to the segmentation threshold instead keeps them.")


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------

#: Q2's rule is *progress shown, no cancel*. **200-500 s** — the longest wait in the flow,
#: and the reason this screen has a gate at all.
COMMIT_PROGRESS = "Detecting nuclei — this takes several minutes…"


class NucleiScreen(QWidget):
    """Frame C: three pickers, five toggles, one live slider, four commit-only parameters."""

    continue_requested = Signal()
    back_requested = Signal()
    #: **The COMMIT's work, which this screen cannot run itself.** It needs the pipeline,
    #: and a screen importing `detect_nuclei` or `ImageSource` would break §2's first
    #: constraint. `app.SemiAutoApp.run_commit` is the one listener. Emitted **after** the
    #: gate flips, because the gate answers *have the parameters moved since the last
    #: commit*, which the press itself settles.
    commit_requested = Signal()

    #: One overlaid view: the labels sit on the two channels, not beside them.
    CANVAS = CanvasLayout()

    #: `SegmentationParams`' own defaults, asserted against the dataclass in the tests.
    DEFAULT_MEMBERSHIP_THRESHOLD = 0.25
    DEFAULT_NUCLEUS_DIAMETER_UM = "7.6"
    DEFAULT_STITCH_THRESHOLD = "0.25"
    DEFAULT_NUCLEUS_MIN_VOLUME_VOXELS = "200"

    def __init__(self, flow: Flow | None = None, viewer: ViewerPort | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.flow = flow if flow is not None else Flow(Session(), start=Step.NUCLEI)
        self.viewer = viewer

        self.nucleus_labels: np.ndarray | None = None
        self.cell_mask: np.ndarray | None = None
        #: `(1, d, d)` once a detection arrives at working resolution, `None` while the labels
        #: and the image are the same size. Set in `set_detection`, read by `draw` - and by the
        #: gesture capture, where it is 4a.1's one multiplication.
        self.working_scale: tuple[float, float, float] | None = None
        #: **Per-channel display range from item 1**, keyed by role; empty means the
        #: percentile stretch. Set by the flow driver on arrival — see `app.prepare`.
        self.display_range: dict[str, tuple[float, float]] = {}
        #: `(labels, microglial-ids bytes, volume)`, or `None`. See `_microglial_outline`.
        self._outline_cache: tuple | None = None
        #: DAPI's 1st/98th percentiles in log space, computed once. See `_dapi_limits`.
        self._log_limits: tuple[float, float] | None = None
        #: **The window cache — item A9.** Three planes a position here, and a layer's entries
        #: are dropped when the array behind it is replaced, so a new detection is never served
        #: the previous one's mask outline.
        self.windows = WindowCache()
        self.dapi: np.ndarray | None = None
        self.iba1: np.ndarray | None = None
        self.fractions_at_commit: np.ndarray = np.zeros(0)
        self.z = 0
        self.depth = 1
        #: Rolling until the user ticks `Step by batch`. The two modes agree at depth 1, so
        #: the toggle can never move the picture on its own.
        self.batch_mode = ROLLING
        self.markers = SplitMarkers()
        self.edits = OperationRecord()
        #: The committed detection **with the record replayed onto it** — what the canvas shows
        #: and what every count on this screen answers for. `nucleus_labels` stays the detection
        #: as it arrived, because that is what an undo has to be able to get back to.
        self.shown_labels: np.ndarray | None = None
        #: The ids actually present in what is shown. See `present_ids`.
        self._present: np.ndarray = np.zeros(0, np.int64)
        #: Which nucleus the split markers were placed in. Taken from the **first** marker: a
        #: split divides one object, and the markers after the first say into how many.
        self.split_target: int | None = None

        #: Replaced in tests. The real one shows §6e.6's two-button popup; making it a callable
        #: is what keeps `_on_commit` testable without a modal blocking the suite.
        self.confirm_discard = self._ask_discard

        self.frame = FrameC("microglial nuclei detection", parent=self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.frame)

        # -- left: Colours (three), then Layers (five) ------------------
        self.frame.section("Colours", HelpButton(
            "Colours",
            "Every picker offers the same ten colours on every screen, so a colour means one "
            "thing throughout. The slider beneath each row is a weight: layers add rather than "
            "cover each other. Takes effect immediately.\n\n"
            "The nucleus labels are not here. Their colouring is a scheme rather than a choice "
            "— each nucleus takes its own hue so you can see where one ends and the next "
            "begins, and brightness says whether it is microglial."))

        self.rows: dict[str, ColourRow] = {}
        for name, colour, alpha, _ in PICKER_LAYERS:
            row = ColourRow(name, colour, alpha, self)
            row.colour_changed.connect(lambda c, n=name: self._recolour(n, c))
            row.alpha_changed.connect(lambda a, n=name: self._reweight(n, a))
            self.frame.add_left(row)
            self.rows[name] = row

        self.frame.section("Layers", HelpButton(
            "Layers",
            "DAPI -veto is every nucleus left after the plaque veto on the earlier screen. "
            "Iba1 mask is the segmentation threshold you set before this one, drawn as an "
            "outline so you can see whether a nucleus sits inside a cell.\n\n"
            "microglial nuclei outline draws boundaries around the nuclei that pass the "
            "threshold, in the same colours they already have. It is its own layer, so the "
            "fills stay on underneath. Turn it on when two nuclei touch and you need to know "
            "whether that is one nucleus or two.\n\n"
            "log DAPI redraws the DAPI channel the way the detector sees it. Faint nuclear "
            "signal that the normal view renders as black becomes visible, so you can tell a "
            "label that has over-reached from one sitting on DAPI you simply could not see."))
        self.toggles = LayerToggles(TOGGLES, self)
        self.toggles.toggled.connect(self._toggle)
        self.frame.add_left(self.toggles)
        self.frame.left.addStretch(1)

        # -- centre: readout strip, then the overlaid view ---------------
        self.header = HeaderStrip(
            [], microglial_readout(0, 0),
            HelpButton("microglial",
                       "How many detected nuclei pass membership_threshold, out of all "
                       "detected.\n\nUpdates live as you move the threshold."),
            self)
        self.frame.add_centre(self.header)
        self.canvas = CanvasHost(self)
        self.frame.add_centre(self.canvas, 1)

        # -- right: the live control FIRST, 500 s away from the rest -----
        self.membership_threshold = ParameterSlider(
            "membership_threshold", 0.0, 1.0, self.DEFAULT_MEMBERSHIP_THRESHOLD,
            help_button=HelpButton(
                "membership_threshold",
                "How much of a nucleus must sit inside the Iba1 mask for it to count as "
                "microglial.\n\n"
                "HIGHER -> fewer nuclei highlighted\n"
                "LOWER  -> more highlighted\n\n"
                "Only highlighted nuclei become cells.\n\n"
                "LIVE: it recolours the nuclei and changes nothing else. No nucleus is added "
                "or removed."),
            parent=self)
        self.membership_threshold.value_changed.connect(self._on_threshold_moved)
        #: **The redraw waits for the release — item 3's arrangement**, and A5's fix. See
        #: `_on_threshold_moved`.
        self.membership_threshold.slider.sliderReleased.connect(self._on_threshold_released)
        self.frame.add_right(self.membership_threshold)

        lost = "\n\nTakes effect on COMMIT and re-runs detection (200-500 s).\nALL HAND EDITS ARE LOST."
        self.nucleus_diameter_um = self._number(
            "nucleus_diameter_um", self.DEFAULT_NUCLEUS_DIAMETER_UM,
            "The nucleus size Cellpose looks for, in micrometres.\n\n"
            "TOO SMALL -> one nucleus found as several\n"
            "TOO LARGE -> several found as one" + lost)
        self.stitch_threshold = self._number(
            "stitch_threshold", self.DEFAULT_STITCH_THRESHOLD,
            "How much two detections on neighbouring slices must overlap to be joined into one "
            "3D nucleus.\n\n"
            "TOO LOW -> one nucleus is shredded into many pieces; each piece then falls under "
            "nucleus_min_volume_voxels and is deleted, so the nucleus looks like it was never "
            "found at all." + lost)
        self.nucleus_min_volume_voxels = self._number(
            "nucleus_min_volume_voxels", self.DEFAULT_NUCLEUS_MIN_VOLUME_VOXELS,
            "Deletes detected nuclei smaller than this many voxels.\n\n"
            "It interacts with stitch_threshold: shredded pieces fall below this floor and "
            "disappear." + lost)

        self.z_depth = self._number(
            "z-batch depth", "1",
            "How many z slices are combined into one view.\n\n"
            "  1  -> one slice at a time\n"
            "  10 -> the brightest value across each group of 10 slices\n\n"
            "AT DEPTH 1 the nucleus labels are shown and can be edited. ABOVE 1 they disappear: "
            "combining slices would merge two nuclei into one, so the labels cannot be shown "
            "honestly or clicked.\n\n"
            "Changes only what you see. Immediate. LEFT and RIGHT move through z.")
        #: **On ENTER or when focus leaves, never per keystroke.** The user's
        #: decision, 2026-08-21. A z-batch is a whole re-projection of every layer,
        #: and applying one to each character of `10` means projecting at 1 on the
        #: way past — which is a dimensionality change of every layer, twice.
        self.z_depth.editingFinished.connect(self._on_depth_changed)
        self.batch_step = batch_step_box(self)
        self.batch_step.toggled.connect(self._on_batch_mode_changed)
        self.frame.add_right(self.batch_step)

        self._z_heading = self._heading(
            "z slice",
            "Which slice you are looking at, out of the whole stack. LEFT and RIGHT arrow keys "
            "move through z.\n\n"
            "At z-batch depth above 1 this counts groups of slices instead and reads "
            "\"z batch\".")
        self.z_indicator = QLabel("", self)
        self.z_indicator.setProperty("role", "helper")
        self.frame.add_right(self.z_indicator)

        self.commit = CommitButton("Commit", self)
        self.commit.clicked.connect(self._on_commit)
        self.frame.add_right(self.commit)

        #: **Nothing else on this screen says the gestures exist.** Three mouse gestures and a
        #: keystroke, with no label, no menu and no hint — and the "?" convention covers
        #: *controls*, which gave them no home. PrestoCell has no split at all, so there was no
        #: inherited convention to lean on either.
        self._heading("gesture legend", GESTURE_LEGEND)

        self.frame.right.addStretch(1)
        #: **This screen's own row became the shared one on 2026-08-30.** It was the only Back
        #: button in the program and the user named its arrangement when asking for the rest —
        #: *"back on the left continue on the right of the bottom right panel"* — so the widget
        #: is this layout, lifted rather than redesigned.
        self.navigation = NavigationRow(back=True, parent=self)
        self.navigation.back_clicked.connect(self._on_back)
        self.navigation.continue_clicked.connect(self._on_continue)
        self.frame.add_right(self.navigation)
        self.back_button = self.navigation.back_button
        self.continue_button = self.navigation.continue_button

        self._sync()

    # -- construction ---------------------------------------------------

    def _heading(self, label: str, help_text: str) -> QLabel:
        name = QLabel(label, self)
        row = QHBoxLayout()
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(HelpButton(label, help_text, self))
        host = QWidget(self)
        host.setLayout(row)
        self.frame.add_right(host)
        return name

    def _number(self, label: str, value: str, help_text: str) -> QLineEdit:
        self._heading(label, help_text)
        edit = QLineEdit(value, self)
        if label != "z-batch depth":
            edit.textChanged.connect(self._on_parameter_changed)
        self.frame.add_right(edit)
        return edit

    def _ask_discard(self, message: str) -> bool:
        return ConfirmPopup(message, "Commit anyway", "Go back",
                            title="Hand edits will be lost", parent=self).ask()

    # -- loading --------------------------------------------------------

    def set_detection(self, nucleus_labels: np.ndarray, cell_mask: np.ndarray) -> None:
        """Take the committed detection and compute the fractions **once**.

        Everything the slider does afterwards is a re-threshold of this array plus a rebuilt
        colormap dict. That is the whole reason one control on this screen can be live while the
        four below it cost 200–500 s.
        """
        if nucleus_labels.shape != cell_mask.shape:
            raise ValueError(f"labels {nucleus_labels.shape} and mask {cell_mask.shape} "
                             "must have the same shape")
        self.nucleus_labels = nucleus_labels
        self.cell_mask = cell_mask
        self.working_scale = working_scale(
            None if self.dapi is None else self.dapi.shape, nucleus_labels.shape)
        self.shown_labels = None
        self.split_target = None
        self.markers.clear()
        self.fractions_at_commit = membership_fractions(nucleus_labels, cell_mask)
        self._recount()
        self._refresh_readout()
        #: > ### THE COMMIT COMPUTED 35 NUCLEI AND DREW NONE OF THEM - found by looking
        #: >
        #: > This method took the detection, computed the fractions and updated the readout,
        #: > and **nothing redrew the canvas.** On a real display the readout went from
        #: > `microglial: 0 of 0` to `microglial: 5 of 35`, the exclamation cleared, Continue
        #: > enabled - and the picture was the same DAPI it had been before, with the
        #: > `nucleus labels` toggle ticked over an empty layer. The labels appeared only if
        #: > the user happened to move the membership slider afterwards, which redraws.
        #: >
        #: > `set_channels`, the sibling seam, has always ended with a draw. This is the same
        #: > kind of arrival and now ends the same way.
        self.draw()

    def set_channels(self, dapi: np.ndarray, iba1: np.ndarray) -> None:
        """The two image volumes, held rather than passed to every `draw`.

        **Three of the four layers draw immediately** — DAPI-minus-veto came from item 2 and
        Iba1 raw is already in memory — so the screen has them before detection finishes and
        must be able to redraw on a z step without being handed them again.
        """
        self.dapi, self.iba1 = dapi, iba1
        #: **New channel, new percentiles.** `_log_limits` is derived from `self.dapi` and would
        #: otherwise describe the previous image — the kind of stale value that looks like a
        #: contrast bug rather than a caching one.
        self._log_limits = None
        self.draw()

    def labels_shown(self):
        """What is on the canvas: the detection **with the record replayed onto it**."""
        return self.nucleus_labels if self.shown_labels is None else self.shown_labels

    def present_ids(self) -> np.ndarray:
        """The nuclei that are actually **there**, cached.

        > ### `n_detected` COUNTED ID SLOTS, NOT NUCLEI — found by deleting one, task 25
        >
        > It was `fractions_at_commit.size`, and `membership_fractions` returns one entry per id
        > **up to the largest**. Deleting nucleus 1 of 2 leaves id 2 in the volume, so the array
        > is still 2 long and the strip still read `of 2` with one nucleus on screen. Deleting
        > the *highest* id shrank it and deleting any other did not — a count that was right
        > exactly often enough to look right.
        >
        > **Cached rather than computed on demand**: `np.unique` over a 40 x 1152 x 1152 label
        > volume is about half a second, and `_refresh_readout` runs on every step of the
        > membership slider — the one control on this screen that has to stay live.
        """
        return self._present

    def _recount(self) -> None:
        volume = self.labels_shown()
        if volume is None:
            self._present = np.zeros(0, np.int64)
            return
        ids = np.unique(volume)
        self._present = ids[ids > 0].astype(np.int64)

    def n_detected(self) -> int:
        return int(self._present.size)

    def _int_field(self, box, fallback: int) -> int:
        try:
            return int(box.text().strip())
        except (AttributeError, ValueError):
            return fallback

    def downsample(self) -> int:
        """The pipeline's xy reduction, from the session — **the one authority on it.**"""
        return max(1, int(self.flow.session.chosen.get("downsample", 2)))

    def working_voxel_size(self):
        """The voxel size **at segmentation resolution**, which is the frame the labels are in."""
        from ..resolution import voxel_size_after_downsample

        voxel = self.flow.session.voxel_size_um or (1.0, 1.0, 1.0)
        return tuple(voxel_size_after_downsample(voxel, self.downsample()))

    def nucleus_diameter_px(self) -> float:
        """Item 4's box is micrometres; growing a nucleus needs pixels at working resolution."""
        from ..nuclei import diameter_in_pixels

        try:
            diameter_um = float(self.nucleus_diameter_um.text().strip())
        except (AttributeError, ValueError):
            diameter_um = self.DEFAULT_NUCLEUS_DIAMETER_UM
        voxel = self.flow.session.voxel_size_um or (1.0, 1.0, 1.0)
        return diameter_in_pixels(float(diameter_um), voxel, self.downsample())

    def microglial(self) -> np.ndarray:
        """**Intersected with what is on screen.** An id the user deleted still has an entry in
        the fractions array, and at a threshold of 0 it would otherwise be counted as microglial
        — `microglial: 3 of 2`, which is an impossible sentence."""
        passing = microglial_ids(self.fractions_at_commit, self.membership_threshold.value())
        if self._present.size == 0:
            return passing
        return passing[np.isin(passing, self._present)]

    # -- z and depth ----------------------------------------------------

    def set_z(self, z: int) -> None:
        if self.nucleus_labels is None:
            return
        n = self.nucleus_labels.shape[0]
        self.z = max(0, min(int(z), n - 1))
        self.z_indicator.setText(z_window_range(self.z, n, self.depth, self.batch_mode))
        #: > ### PLANES, NOT VOLUMES - SO THERE IS NO napari SLIDER. The user's rule
        #: >
        #: > Stated for item 2 on 2026-08-21 and extended here on 2026-08-23: *"remove the
        #: > napari z slider, the only way to change z should be the left and right arrow
        #: > key."* Item 4 was the last screen pushing whole volumes and letting napari
        #: > slice them, which is what put a second z control on the screen - one the
        #: > screen's own indicator, gate and gestures did not own.
        #: >
        #: > **`viewer.set_z` used to be called here** to keep that slider in step. With
        #: > every layer 2D there is no third axis for napari to offer and nothing to keep
        #: > in step; `self.z` is the only z on the screen.
        self.draw()

    def step_z(self, delta: int) -> None:
        """**The stride is the toggle**: one slice rolling, a whole batch when `Step by batch`
        is ticked. `step_window` also stops where the window last fills, so rolling cannot walk
        into the tail where the projection thins while the label still claims a full batch."""
        if self.nucleus_labels is None:
            return
        self.set_z(step_window(self.z, delta, int(self.nucleus_labels.shape[0]),
                               self.depth, self.batch_mode))

    def _on_batch_mode_changed(self, batched: bool) -> None:
        self.batch_mode = BATCHED if batched else ROLLING
        self.set_z(self.z)

    def keyPressEvent(self, event):     # noqa: N802 - Qt's name
        """LEFT/RIGHT step z; `u` undoes one operation. PrestoCell binds `u` to exactly this."""
        if event.key() == Qt.Key.Key_Left:
            self.step_z(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.step_z(1)
        elif event.key() == Qt.Key.Key_U:
            self.undo()
        else:
            super().keyPressEvent(event)

    def _on_depth_changed(self) -> None:
        """The depth box owns the label layer's visibility. **One control, not two.**

        At item 2 the projection checkbox was merely redundant with the depth box; here it also
        switched off the label layer, so folding it in removes a control *and* puts the
        consequence where the user will meet it.
        """
        try:
            self.depth = max(1, int(self.z_depth.text().strip()))
        except ValueError:
            return
        self._z_heading.setText(z_window_word(self.depth))
        self.batch_step.setEnabled(self.depth > 1)
        self.set_z(self.z)

    # -- the live control -----------------------------------------------

    def _on_threshold_released(self) -> None:
        """The recompute, when the hand lets go. Item 3 is the reference implementation.

        `draw` is ~5 s on a production field, and `value_changed` fires on **every one of the
        slider's 1000 steps** — so a drag queued hundreds of full redraws and the screen appeared
        to hang outright. `sliderReleased` fires once.
        """
        self.draw()

    def _on_threshold_moved(self, value: float) -> None:
        """**No invalidate.** Moving this recolours and changes nothing the commit produced.

        > ### ⇒ AND NO REDRAW EITHER, SINCE A5 — 2026-09-03
        >
        > This ended in `draw()`, on a `value_changed` that fires at every one of the slider's
        > 1000 steps. Item 3 already refuses to do that; item 4 did it on the screen with the
        > most expensive draw in the flow. **`sliderReleased` now carries the redraw**, and this
        > handler keeps everything that is cheap and everything that must not wait:
        >
        > * `_sync`, for the reason below — it must fire per step, or a value the user chose can
        >   be left unwritten, which a mutation test already caught once;
        > * `_refresh_readout`, a comparison over the ~35 fractions the commit already computed,
        >   and the feedback this threshold is actually chosen by. It stays live deliberately;
        > * the slider's own value label, which `ParameterSlider._on_slider` sets itself before
        >   it emits, so it needs nothing from here.

        > ### BUT IT MUST STILL REACH THE SESSION — found by a mutation test, task 24
        >
        > `membership_threshold` is one of the three parameters owned by a **slider** rather than
        > a box, so it never passes through `parsed_numbers`; `_sync` is the only writer. This
        > handler recoloured and redrew and called neither — so the slider moved, the picture
        > followed, and `session.chosen` kept the value the screen opened at. `session_params`
        > would then have written the default into the parameters file **while the screen showed
        > something else**, which is the whole failure task 24 exists to close.
        >
        > `_sync` is safe here: it writes, re-reads the gate and re-reads Continue. **It does not
        > invalidate** — that is `flow.invalidate()`, which this handler still deliberately does
        > not call.
        """
        self._sync()
        self._refresh_readout()

    def _refresh_readout(self) -> None:
        self.header.set_readout(microglial_readout(self.microglial().size, self.n_detected()))

    # -- editing --------------------------------------------------------

    def _replay(self) -> None:
        """Rebuild the shown labels and their membership **from the record.**

        > ### THE EDITS WERE RECORDED AND THE PICTURE NEVER CHANGED - task 25
        >
        > Each of the three editing methods appended to `OperationRecord` and called `draw()`,
        > and `draw()` pushed the **committed** volume. A double-click on a nucleus recorded a
        > deletion and left the nucleus on screen; `microglial: 5 of 35` kept answering for the
        > detection as it arrived. Every gesture in the legend was inert in the way the user
        > would notice first.

        **The membership is recomputed too, not only the picture.** A deleted nucleus is one
        fewer of `n_detected`, a split is one more, and an added one has a fraction of its own —
        so a readout left at the committed fractions would be a number about a picture that is
        no longer there. It is `membership_fractions` over the label volume, which is the same
        call the commit makes; it runs per **edit**, not per slider move, and the slider's
        cheapness is untouched.
        """
        if self.nucleus_labels is None:
            return
        self.shown_labels = edited_labels(
            self.nucleus_labels, self.edits.operations,
            dapi=self.dapi, diameter_px=self.nucleus_diameter_px(),
            downsample=self.downsample(), voxel_size_um=self.working_voxel_size())
        if self.cell_mask is not None and self.shown_labels.shape == self.cell_mask.shape:
            self.fractions_at_commit = membership_fractions(self.shown_labels, self.cell_mask)
        self._recount()

    def delete_nucleus(self, label: int) -> Operation:
        operation = self.edits.delete(label)
        self._replay()
        self._refresh_readout()
        self.draw()
        return operation

    def add_nucleus(self, z: int, y: int, x: int) -> Operation:
        """`(z, y, x)` is a **native full-image** voxel — §4a.1, converted at capture."""
        operation = self.edits.add(z, y, x)
        self._replay()
        self._refresh_readout()
        self.draw()
        return operation

    def mark_split_point(self, z: int, y: int, x: int) -> int:
        """One click while shift is held. Returns the click number shown beside the cross."""
        return self.markers.add(z, y, x)

    def release_shift(self, label: int | None = None) -> Operation | None:
        """**Release-to-execute gives N-way splits for free** — three fused nuclei, three clicks.

        `split_by_watershed` takes a markers array, so the count is not special-cased anywhere.
        **Releasing with one marker does nothing, silently** — which is exactly what the
        numbering makes visible before you let go.
        """
        operation = None
        target = self.split_target if label is None else label
        if self.markers.is_actionable() and target is not None:
            operation = self.edits.split(target, self.markers.points)
        self.markers.clear()
        self.split_target = None
        self._replay()
        self._refresh_readout()
        self.draw()
        return operation

    def undo(self) -> Operation | None:
        """One step per press. **Undo is a pop and a recompute**, with no inverse to write.

        That is what `OperationRecord`'s claim — *"undo and the persisted history cannot
        disagree"* — actually requires: the picture has to be a function of the record.
        """
        operation = self.edits.undo()
        self._replay()
        self._refresh_readout()
        self.draw()
        return operation

    # -- drawing --------------------------------------------------------

    def _recolour(self, layer: str, colour: str) -> None:
        if self.viewer is not None and layer in self.viewer.layer_names():
            self.viewer.set_colour(layer, colour)

    def _reweight(self, layer: str, alpha: float) -> None:
        if self.viewer is not None and layer in self.viewer.layer_names():
            self.viewer.set_alpha(layer, alpha)

    def _toggle(self, name: str, on: bool) -> None:
        """Two of the six need a redraw rather than a visibility flip.

        `OUTLINE_LAYER` is only pushed while it is on — it is a second copy of the label volume
        and there is no reason to hold one the user is not looking at — so switching it on has
        to build it. `LOG_DAPI` changes the DAPI layer's data and its limits together, which is
        a redraw by definition.
        """
        if name in (OUTLINE_LAYER, LOG_DAPI):
            self.draw()
            return
        if self.viewer is not None and name in self.viewer.layer_names():
            self.viewer.set_visible(name, on)

    def draw(self) -> None:
        """Push the three image layers, then the labels **last, so they sit on top**.

        The label layer is `translucent`, which is §6f.3's one exception and is not a style
        choice: additive would sum a label's hue with the DAPI beneath it — **a red label over
        blue DAPI renders magenta** — corrupting the one thing the layer exists to carry.
        """
        if self.viewer is None:
            return

        source = {"DAPI -veto": self.dapi, "Iba1 raw": self.iba1, "Iba1 mask": self.cell_mask}
        #: **The mask is an outline, so it takes `(0, 1)`; the two channels take item 1's
        #: range or their own percentiles.** This screen passed nothing at all until
        #: 2026-08-20 — see `preview.contrast_limits`.
        logged = self.toggles.is_on(LOG_DAPI)
        limits = {
            "DAPI -veto": (self._dapi_limits(logged) if self.dapi is not None else None),
            "Iba1 raw": (contrast_limits(self.iba1, self.display_range.get("cell"))
                         if self.iba1 is not None else None),
            "Iba1 mask": (0.0, 1.0),
        }
        for name, colour, alpha, _ in PICKER_LAYERS:
            data = source.get(name)
            if data is None:
                continue
            #: **`Iba1 mask` arrives from the detection, so it is at working resolution too** -
            #: `Compute.detect` returns the labels and the cell mask as one pair, at one
            #: resolution, precisely so membership means something. Both therefore carry the
            #: same scale; the two raw channels are native and carry none.
            scale = None
            if name == "Iba1 mask":
                #: **No `astype(bool)` here — patch item A8, 2026-09-03.** It copied the
                #: whole volume *before* `outline_of_batch` sliced the window out of it, and
                #: the cast is not needed at all: the function's own `any(axis=0)` treats
                #: nonzero as true, which is what `astype(bool)` was doing.
                data = cached_plane(
                    self.windows, name, data, self.z, self.depth, self.batch_mode,
                    lambda d=data: outline_of_batch(np.asarray(d), self.z, self.depth,
                                                    mode=self.batch_mode))
                #: **`plane_scale`, because `outline_of_batch` returns a PLANE.** This carried
                #: the full `(1, 2, 2)` for as long as every other layer was a volume, and it
                #: was harmless only because napari ignored the leading factor on a layer that
                #: had a z axis to spend it on. With the layer 2D the y factor would land on x
                #: and the outline would sit at half scale beside the mask it outlines.
                scale = plane_scale(self.working_scale)
            else:
                #: **The two channels project with the mask over them.** Only the mask did
                #: until task 26, so the outline moved across a picture stuck on one slice.
                #: **Cached per window position — item A9**, and `projection=True` because
                #: this is exactly `volume[window].max(0)`: in **batched** mode it comes from
                #: `ProjectionCache`, every batch built together on a depth change.
                #:
                #: **The raw plane is what is cached, not the logged one.** `log_display` is
                #: applied below, so toggling `log DAPI` costs one `log1p` over a plane rather
                #: than dropping the entry it was computed from.
                data = cached_plane(
                    self.windows, name, data, self.z, self.depth, self.batch_mode,
                    lambda d=data: batch_plane(d, self.z, self.depth, self.batch_mode),
                    projection=True)
                #: **After the projection, not before.** `log1p` is monotone so `max` commutes
                #: with it either way, but doing it here transforms one plane rather than the
                #: whole stack — 0.4 MB against 850 MB on a full field.
                if logged and name == "DAPI -veto":
                    data = log_display(data)
            row = self.rows[name]
            self.viewer.show_image(name, data, colour=row.colour().name,
                                   alpha=row.alpha(), visible=self.toggles.is_on(name),
                                   scale=scale, contrast_limits=limits.get(name))

        self._draw_markers()

        if self.nucleus_labels is None or not labels_shown_at_depth(self.depth):
            for name in (LABEL_LAYER, OUTLINE_LAYER):
                if name in self.viewer.layer_names():
                    self.viewer.remove(name)
            return

        #: **Every id up to the largest**, not the count of nuclei: `label_colours` is a lookup
        #: keyed by id, and a sparse volume — which is what a deletion leaves — has ids above
        #: its own count.
        n_ids = int(self.fractions_at_commit.size)
        ids = np.arange(1, n_ids + 1)
        microglial = self.microglial()
        #: **`contour=0` always.** The outline moved to its own layer precisely because napari's
        #: `contour` replaces the fill; setting it here would put that behaviour back.
        #: **The slice, not the volume** - see `plane_scale`. Labels are only ever shown at
        #: depth 1 (`labels_shown_at_depth`), so there is exactly one slice to choose and no
        #: projection question to answer here.
        self.viewer.show_labels(
            LABEL_LAYER, self._label_plane(self.labels_shown()),
            colours=label_colours(ids),
            alphas=label_alphas(ids, microglial),
            contour=0,
            visible=self.toggles.is_on(LABEL_LAYER),
            scale=plane_scale(self.working_scale))

        #: **Pushed last, so it lands on top of the fills** — an outline underneath its own
        #: filled label is invisible. `show_labels` raises whichever labels layer it touched to
        #: the top of the stack, so the order of these two calls is the stacking order.
        if not self.toggles.is_on(OUTLINE_LAYER):
            if OUTLINE_LAYER in self.viewer.layer_names():
                self.viewer.remove(OUTLINE_LAYER)
            return
        #: **Full alpha for every id here, not `label_alphas`.** Nothing but microglial nuclei
        #: is in this volume, so there is no second class to mute against — the muting on the
        #: fill layer is what carries *microglial versus not*, and repeating it here would only
        #: dim the thing the layer exists to sharpen.
        self.viewer.show_labels(
            OUTLINE_LAYER, self._label_plane(self._microglial_outline(microglial, n_ids)),
            colours=label_colours(ids),
            alphas={int(i): 1.0 for i in ids},
            contour=OUTLINE_CONTOUR_PX,
            visible=True,
            scale=plane_scale(self.working_scale))

    def _dapi_limits(self, logged: bool) -> tuple[float, float] | None:
        """DAPI's display range, in whichever space the layer is being drawn.

        > ### THE LOG LIMITS COME FROM THE **RAW** PERCENTILES, AND THAT IS EXACT
        >
        > `log1p` is monotone, so `percentile(log1p(x), q) == log1p(percentile(x, q))`. Taking
        > the percentiles on the raw volume and logging the two numbers gives byte-identical
        > limits to logging the volume first — without allocating a second 850 MB copy on a full
        > field. **1st and 98th**, which is `normalize_for_detection`'s own pair rather than
        > `contrast_limits`' 0.1/99.9, because the whole point of this view is to match what the
        > detector sees.
        >
        > **An explicit range typed on item 1 is ignored while this is on.** That range is in
        > raw intensity units and means nothing in log space; silently logging its endpoints
        > would show a number the user did not choose. Turning the toggle off restores it.
        """
        if self.dapi is None:
            return None
        if not logged:
            return contrast_limits(self.dapi, self.display_range.get("nuclei"))
        if self._log_limits is None:
            low, high = np.percentile(self.dapi, [1.0, 98.0])
            self._log_limits = (float(np.log1p(max(low, 0.0))),
                                float(np.log1p(max(high, 0.0))))
        return self._log_limits

    def _microglial_outline(self, microglial, n_ids: int) -> np.ndarray:
        """`microglial_only` over the shown labels, **cached against the two things it varies
        with** — which labels are on screen, and which of them pass the threshold.

        `membership_threshold` is the live control on this screen and `draw` runs on every arrow
        key, so without this the outline layer would rebuild a `(40, 1152, 1152)` volume on each
        one. The cache is keyed on the label array's identity rather than its contents: every
        edit path here replaces `shown_labels` with a new array rather than mutating it.
        """
        labels = self.labels_shown()
        key = np.asarray(microglial, dtype=np.int64).tobytes()
        if (self._outline_cache is not None
                and self._outline_cache[0] is labels
                and self._outline_cache[1] == key):
            return self._outline_cache[2]
        volume = microglial_only(labels, microglial, n_ids)
        self._outline_cache = (labels, key, volume)
        return volume

    def _draw_markers(self) -> None:
        """The numbered crosses, while SHIFT is held. **They are what makes *"one marker does
        nothing"* visible before the user lets go** rather than after.

        Drawn in the **canvas's** frame, which is `working` scaled back up by the label layer's
        own scale — the markers sit on the picture, not in the array.

        > ### THE POINTS ARE 2D, AND THAT IS WHAT ACTUALLY REMOVED napari's SLIDER — 2026-08-23
        >
        > Making the five array layers planes was not enough: **`viewer.dims.ndim` is the maximum
        > over every layer**, and this one was pushing `(z, y, x)` triples. One 3D points layer
        > kept the third axis alive and the slider with it, under a canvas where nothing else had
        > a z. Found by the user on the running program after the arrays were already 2D.
        >
        > **The markers keep their z in the record and lose it only in the drawing.** Each seed
        > carries the slice it was placed on — that is the whole reason `Shift-Left` is bound, so
        > z can move while markers are being placed — and `SplitMarkers` is unchanged. What is
        > drawn is the seeds **on the slice being shown**, which is also what napari itself did
        > with a 3D points layer in slice mode, so the picture is the same one.
        """
        if self.viewer is None:
            return
        scale = 1.0 if self.working_scale is None else float(self.working_scale[1])
        here = [(n, (y, x)) for n, (z, y, x) in self.markers.numbered() if int(z) == int(self.z)]
        self.viewer.show_points(MARKER_LAYER,
                                [(y * scale, x * scale) for _n, (y, x) in here],
                                colour=MARKER_COLOUR,
                                labels=[n for n, _p in here],
                                visible=bool(here))

    # -- the gestures ---------------------------------------------------

    def bind_gestures(self) -> None:
        """**The legend, wired.** Every handler below already existed; nothing reached them.

        `GESTURE_LEGEND` is the contract and it is four gestures and one key:

            DOUBLE-CLICK a nucleus        delete the whole 3D nucleus
            DOUBLE-CLICK the background   add one, from a seed
            SHIFT + CLICK each part       mark, then RELEASE SHIFT to split
            u                             undo

        LEFT and RIGHT step z as well, because napari owns the keyboard once its canvas has
        focus — the screen's own `keyPressEvent` never sees them.

        > ### AND `Shift-Left` IS A DIFFERENT BINDING FROM `Left` — the user's rule, 2026-08-20
        >
        > *"When trying to split a nucleus clump, the user should be able to change z with their
        > arrow keys while still holding down shift to stay in split mode. This will allow them
        > to place the seeds for the daughters in the right place."*
        >
        > **A clump's daughters are not on one slice.** A nucleus spans about 13 slices, so two
        > fused nuclei are separated in z as often as in xy — and the split markers each carry
        > their own z, which is only useful if z can move while the markers are being placed.
        >
        > napari keys on the whole combination, so with SHIFT down an arrow press arrives as
        > `Shift-Left` and a binding on `Left` alone **never fires**: the picture would have
        > frozen on one slice for exactly as long as the user was in split mode. Both are bound
        > to the same step. Stepping z redraws, and a redraw redraws the markers, so the seeds
        > already placed stay on screen and stay at the z they were placed at.
        """
        if self.viewer is None:
            return
        self.viewer.bind_double_click(self._on_canvas_double_click)
        self.viewer.bind_click(self._on_canvas_click)
        self.viewer.bind_key("u", self.undo)
        self.viewer.bind_key("Shift", self._on_shift_held)
        #: **The shared one, not a copy.** Item 4 had its own list of arrow keys and its own
        #: dims handler, which is how a mutation on `gestures.Z_KEYS` came back green: the
        #: contract was in one place and item 4 was obeying another. Six screens, one binding.
        gestures.bind_z_stepping(self.viewer, self)

    def _click_point(self, event):
        """One canvas position -> `(index into the shown labels, native voxel)`. §4a.1.

        > ### THE CLICK CARRIES NO Z ANY MORE, AND THE SCREEN SUPPLIES IT — 2026-08-23
        >
        > Every layer is a 2D plane now, so napari reports a **two-element** position and there
        > is no z in the event to read. `self.z` is the slice being shown, and it is the only z
        > on the screen — the same value the indicator prints and the arrow keys move.
        >
        > **This is §4a.1's rule, not an exception to it.** *"Store native full-image voxel
        > coordinates, and convert once, at the point of capture."* The point of capture is still
        > here and still converts once; what changed is that one of the three numbers now comes
        > from the screen's state rather than from the pointer. Getting this wrong is what
        > doubled `avg_nearest_neighbour_um` on every row ever written, so the z is prepended
        > **before** `click_frames` sees it and nothing downstream touches it.
        """
        scale = 1.0 if self.working_scale is None else float(self.working_scale[1])
        position = tuple(float(c) for c in event.position)
        if len(position) < 3:
            position = (float(self.z), *position[-2:])
        return click_frames(position, scale, self.downsample())

    def _label_plane(self, volume):
        """The slice of a label volume this screen is showing, or `None`.

        **Clamped rather than trusted.** `set_z` bounds `self.z` against the detection's own
        depth, but `draw` can run between a new detection arriving and `set_z` being called —
        a shorter volume would otherwise index out of range inside a paint path.
        """
        if volume is None:
            return None
        array = np.asarray(volume)
        return array[max(0, min(int(self.z), array.shape[0] - 1))]

    def _label_at(self, index) -> int:
        volume = self.labels_shown()
        if volume is None:
            return 0
        if any(c < 0 or c >= dim for c, dim in zip(index, volume.shape)):
            return 0
        return int(volume[index])

    @property
    def view_only(self) -> bool:
        """**Above z-batch depth 1 this screen looks and does not edit.**

        Not a restriction added for tidiness — `screen_cd68.batch_projection` already records
        the reason from the other side: *"item 4's reason for making projection view-only does
        not reach here: 9a has no clicks at all, so there is no missing z for a click to need."*
        A projection collapses a group of slices into one picture, so a click on it names a `y`
        and an `x` and **no z** — and every edit here is a 3D operation on one nucleus.

        The labels already vanish above depth 1 (`labels_shown_at_depth`), so there is nothing
        on screen to aim at either. `u` still works: an undo needs no pointer position.
        """
        return not labels_shown_at_depth(self.depth)

    def _on_canvas_double_click(self, event) -> None:
        """**A nucleus deletes; the background adds.** PrestoCell's own arrangement
        (`main.py:826-841`), and the one the legend states."""
        if self.nucleus_labels is None or self.view_only:
            return
        index, native = self._click_point(event)
        label = self._label_at(index)
        if label:
            self.delete_nucleus(label)
        else:
            self.add_nucleus(*native)

    def _on_canvas_click(self, event) -> None:
        """A plain click does nothing on this screen. **With SHIFT it places a split marker**,
        and the first one names the nucleus being split."""
        if not event.shift() or self.nucleus_labels is None or self.view_only:
            return
        index, _native = self._click_point(event)
        if not self.markers.points:
            self.split_target = self._label_at(index) or None
        self.markers.add(*index)
        self._draw_markers()

    def _on_shift_held(self):
        """A **generator** binding: markers clear on press and the split runs on release.

        *"SHIFT + CLICK each part, then RELEASE SHIFT"* — release-to-execute is what gives
        N-way splits for free, and one marker does nothing, silently.
        """
        self.markers.clear()
        self.split_target = None
        self._draw_markers()
        yield
        self.release_shift()

    # -- the commit gate ------------------------------------------------

    def _on_parameter_changed(self) -> None:
        self.flow.invalidate()
        self._sync()

    def set_progress(self, done: int, total: int) -> None:
        self.commit.set_working(detection_progress(done, total))

    def _on_commit(self, product=None) -> None:
        """Warn **only when there is something to lose**, then run.

        `confirm_discard` returning False is a real answer — the user chose to go back — so the
        commit does not happen and the gate stays closed.
        """
        if self.edits.has_edits() and not self.confirm_discard(discard_warning(len(self.edits))):
            return
        self.edits = OperationRecord()
        self.flow.commit(product)
        self._sync()
        if product is None:
            #: **The 200-500 s one.** `set_working` disables the button and forces the
            #: repaint before this returns, so the line is on the glass before the
            #: blocking call starts rather than after it ends.
            self.commit.set_working(COMMIT_PROGRESS)
            try:
                self.commit_requested.emit()
            finally:
                self.commit.set_working("")
        else:
            self.commit.set_working("")

    def parameters(self) -> dict:
        """The `SegmentationParams` fields this screen owns, under **those** names.

        > ### NOTHING THE USER TYPED REACHED THE PIPELINE — task 24
        >
        > `session.chosen` is what `screen_save.session_params` serialises and what
        > `PipelineCompute` reads, and before this task **only items 9a and 9b ever wrote to
        > it.** Items 2, 3, 4, 5 and 6 held their values in `QLineEdit`s and nowhere else, so
        > every commit ran at `SegmentationParams`' defaults and the parameters file recorded
        > defaults — for a screen whose entire purpose is choosing the value.
        >
        > **A half-typed box contributes nothing rather than a zero.** `float("")` raises, and a
        > box mid-edit is not a choice; the commit gate refuses on the same values through
        > `parameter_error`, so a skipped field can never reach a commit either.

        Written by `_sync`, which every control change and every commit already calls.
        """
        chosen = {"membership_threshold": float(self.membership_threshold.value())}
        chosen.update(parsed_numbers({
            "nucleus_diameter_um": (self.nucleus_diameter_um.text(), float),
            "stitch_threshold": (self.stitch_threshold.text(), float),
            "nucleus_min_volume_voxels":
                (self.nucleus_min_volume_voxels.text(), int),
        }))
        return chosen

    def _sync(self) -> None:
        self.flow.session.chosen.update(self.parameters())
        self.commit.set_stale(self.flow.is_stale())
        self.continue_button.setEnabled(self.flow.can_continue())
        self.back_button.setEnabled(self.flow.can_go_back())

    def _on_continue(self) -> None:
        """**The edit count is carried onto the session here, and nowhere else can carry it.**

        §6a requires item 8 to say plainly that manual edits stay with this image, and the
        sentence names a number. The edits themselves are per-image overrides that deliberately
        never reach the parameters file, so nothing downstream would otherwise know one was made
        — and a user who hand-splits nuclei, saves and expects batch to reproduce their work
        *"would be silently wrong."*

        **Read at CONTINUE rather than at commit**, because a commit re-runs detection and
        `_on_commit` clears the record: what survives to item 8 is exactly what was edited after
        the last commit, which is what the screen is showing when the user leaves it.
        """
        if self.flow.can_continue():
            self.flow.session.nucleus_edits = len(self.edits)
            self.continue_requested.emit()

    def _on_back(self) -> None:
        """**The only Back button in the component**, and it returns to item 3.

        Nucleus edits survive the trip: `detect_nuclei` never receives the cell mask, so changing
        `cell_coefficient` moves only *which* nuclei are microglial, and returning is a cache hit.
        The honest caveat is that an edit's **rationale** may not survive even though the edit
        does — you deleted something because it looked microglial and wrong, and under the new
        mask it might not have been highlighted at all.
        """
        if self.flow.can_go_back():
            self.back_requested.emit()
