"""Phase C, 6b and 6c: CD68 load and plaque proximity, per confirmed cell.

Both measures sit beside the morphometrics on the same cell ids, which is the point — CD68
load can be correlated against ramification index *per cell*, and plaque distance against
both, without joining across two tools (`DESIGN_RATIONALE.md` §4.13).

**CD68 is measured over the whole cell mask** (§4.13.2). A soma-restricted measure was
considered and rejected: process-resident CD68 is real signal, and restricting the region
would break the fraction-of-cell-volume measure against the ramification index it exists to
pair with.

**The threshold-free measures are primary.** Integrated and mean intensity need no cutoff and
capture "more CD68 = brighter" directly, which is the biology. They are trustworthy only
because 32-bit was kept with no per-image rescaling (§4.13) — under a per-image rescale they
would stop being comparable between files, which is the whole point of measuring them.

**CD68-positive volume is deliberately inert until someone chooses a threshold.** Passing none
yields ``NaN`` and a status, never a zero and never a guessed default — and, since 2026-08-10,
**no curve either**, because the curve's levels are multiples of the value nobody has picked.

Two schemes were tried and abandoned before the absolute threshold, and both are recorded so
neither is retried. ``median + k*MAD`` is degenerate on this data: Huygens CMLE leaves a
constant background floor, so the MAD is exactly zero and every *k* yields the same cutoff
(§4.13.3). Multiples of the image's background median are worse — CD68 is punctate, so
background sits at 0.013-0.048 while the puncta live at 10²-10⁴, and every level lands in the
noise floor (§14.2). What resolved it was not a better scheme but a fact: **acquisition was
held constant across images**, so one absolute intensity picked once in the semi-auto UI
applies unchanged to the whole batch. `DESIGN_RATIONALE.md` §4.13.8 records the general
lesson — *a curve is not a substitute for a decision somebody is in a position to make*.

**Plaque distance is reported two ways** because they answer different questions: from the
**soma centroid**, has the cell body migrated toward a plaque; from the **nearest cell
voxel**, does any process reach one even with the soma distant. The "plaque-associated"
cutoff (commonly 10-30 µm) stays an analysis-time threshold, never a pipeline constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "Cd68Measures",
    "Cd68ImageLoad",
    "DebrisLoad",
    "PlaqueBurden",
    "PlaqueProximity",
    "CURVE_FACTORS",
    "MIN_PLAQUE_VOLUME_UM3",
    "NO_CD68_CHANNEL",
    "PLAQUE_DAPI_COEFFICIENT",
    "build_plaque_distance_field",
    "build_plaque_fields",
    "measure_cd68",
    "measure_cd68_image",
    "measure_iba1_image",
    "measure_debris",
    "measure_plaque_burden",
    "measure_plaque_proximity",
    "plaques_within",
]

#: The curve a chosen threshold ships with: ``T/4, T/2, T, 2T, 4T``.
#:
#: **A robustness bracket around a decision, not a substitute for one** — which is the change
#: made 2026-08-10 (`EXPORT_SPEC.md` §2, `DESIGN_RATIONALE.md` §4.13.8). Both CD68 and plaque
#: burden previously shipped a curve anchored to nothing because no value was thought
#: defensible; the user pointed out that acquisition was held constant across images, which
#: makes a single absolute threshold picked once in the semi-auto UI both defensible and
#: standard practice. A curve is still worth carrying, but now it brackets the chosen value.
#:
#: **Log spacing** because both quantities span orders of magnitude. **f = 2** because visual
#: threshold-setting is reproducible to roughly a factor of two, so the 16x span from T/4 to
#: 4T brackets any plausible alternative pick.
#:
#: There is no curve when no threshold is chosen. That is the point: the levels are defined
#: relative to T, so an unset T yields `NaN` and a status, never a curve anchored to a guess.
CURVE_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)

#: **No CD68 channel is not zero CD68**, and this is the status every level says it with — the
#: image load, the debris load, and since 2026-09-02 the per-cell payload that `phasec` writes.
#: A name rather than four literals: the cell level was blank for as long as it was, because
#: nothing counted the places this string had to appear.
NO_CD68_CHANNEL = "no_cd68_channel"

#: The one plaque size floor, in µm³ — **replacing both `MIN_PLAQUE_UM3 = 5.0` and the
#: five-value `PLAQUE_SIZE_FLOORS_UM3` curve**, which could disagree about how many plaques an
#: image contains. Applied identically to per-cell geometry, the plaques table and burden.
#:
#: **A floor is not optional.** Without one there are 39,765 objects across the 5xFAD set at
#: ~1-2 µm mean spacing, so every cell reads as plaque-adjacent and distance collapses to
#: noise. Its primary job is **distance**, not burden — burden totals barely move.
#:
#: **Why 25.** It keeps ~9 plaques per field and removes 99.7% of specks (99% of objects in
#: *both* genotypes are <= 7 µm³). The binding constraint is a ceiling of **73 µm³** — the
#: largest object in `692137_6Hippocampus4`, above which that image loses every plaque. That
#: is a *sample minimum from one image* and will fall as data is added, so margin matters: 25
#: leaves 48 µm³ of it, 50 leaves 23.
#:
#: **Diameter equivalents, because this decision is made in Fiji rather than here:**
#: 25 µm³ ~ a 3.6 µm sphere, 50 ~ 4.6 µm, 73 ~ 5.2 µm.
MIN_PLAQUE_VOLUME_UM3 = 25.0

#: DAPI dense-core cutoffs, as multiples of Otsu on the DAPI values *inside* the plaque mask.
#:
#: **Calibrated 2026-08-10** (`tuning_runs/dapi_core_calibration.txt`, 20 images), replacing a
#: range I had simply chosen. Unlike every other curve here, this one has a **maximum**:
#: 5xFAD/WT separation rises 26.6x -> 29.3x -> 30.9x -> **32.3x at 1.5** and then falls to
#: 28.6x, 14.5x, 7.9x. Above ~1.5 real 5xFAD cores are lost faster than WT signal, so the
#: range is set to bracket that peak rather than to run off into a monotone tail.
#:
#: **WT never reaches zero**, which the negative-control logic predicted it would — and the
#: reason is instructive. DAPI is a *nuclear* stain and WT tissue is full of nuclei, so what
#: WT reports here is very likely genuine nuclei caught inside a **false plaque mask**, not
#: noise in the DAPI cutoff. The error is inherited from the plaque mask upstream, which means
#: **the fix is `MIN_PLAQUE_VOLUME_UM3`, not this coefficient.**
#:
#: All five are kept at **image level** — 35 columns in `image_curves.csv` — deliberately, and
#: on grounds that do not apply anywhere else in this codebase: the measure is *unvalidated on
#: this tissue* (Gotkiewicz cross-validated high-intensity DAPI against Thioflavin-S and X34;
#: this tissue has neither), and that is exactly where a robustness spread earns its place.
DAPI_CORE_COEFFICIENTS = (0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass
class Cd68Measures:
    """CD68 load for one cell. `NaN` plus a status, never a silent zero.

    **No depth measure here, deliberately, since 2026-08-12.** This class carried a ``mean_z``
    until then — computed correctly in stack coordinates, exported by nothing, and sitting
    beside a crop-local ``CellMorphology.mean_z`` that *was* exported. One name, two meanings,
    and the test that asserted the right principle asserted it against this copy, which is why
    the wrong number shipped (`EXPORT_AUDIT.md` §3.1). Depth is a property of where the cell
    sits in the stack, not of the CD68 channel — it belongs to morphometrics, which runs
    whether or not this image has a CD68 channel at all.

    The surviving one is ``CellMorphology.mean_z_um``, renamed and converted to µm later the
    same day (NEW-a). ``test_cd68_does_not_carry_a_depth_of_its_own`` asserts **both** spellings
    are absent here, so the deleted field cannot return under its old name through the gap a
    rename opens.
    """

    status: str = "ok"
    integrated_intensity: float = float("nan")
    mean_intensity: float = float("nan")
    positive_volume_um3: float = float("nan")
    positive_fraction: float = float("nan")
    #: ---- the soma/process split, added 2026-09-01 ---------------------------------------
    #: CD68⁺ volume inside the soma mask and outside it, and the first as a share of the cell's
    #: **total** CD68 volume (the user's choice of denominator). The two volumes sum to
    #: ``positive_volume_um3`` by construction, since "processes" is the complement inside the
    #: cell rather than a second measurement — which is what lets a reader check them.
    #: All three stay `NaN` when no soma mask is supplied.
    in_soma_um3: float = float("nan")
    in_processes_um3: float = float("nan")
    soma_fraction: float = float("nan")
    #: The **absolute intensity** the user chose in the semi-auto UI, with live preview. Not
    #: an Otsu multiplier: Otsu is computed per slice per image, so the same multiplier lands
    #: on a different intensity in every image, which defeats choosing once and applying to a
    #: batch (`EXPORT_SPEC.md` §2). ``None`` means unchosen, and unchosen stays visibly so.
    threshold: float | None = None

    #: CD68⁺ volume and volume fraction at ``T x CURVE_FACTORS``, keyed by absolute level.
    #: Empty when no threshold is set — the levels are defined relative to T, so there is
    #: nothing to compute rather than something to guess.
    volume_curve_um3: dict[float, float] = field(default_factory=dict)
    fraction_curve: dict[float, float] = field(default_factory=dict)

    @property
    def has_volume(self) -> bool:
        """Whether the threshold-dependent measures are populated."""
        return not np.isnan(self.positive_volume_um3)


@dataclass
class Cd68ImageLoad:
    """Whole-image CD68 load — the CD68 counterpart of `PlaqueBurden.integrated_intensity`.

    **Threshold-free by construction, and that is its entire value.** Every other CD68 figure
    in this export depends on two things nobody has settled: the segmentation that decided
    where the cells are, and the absolute `cd68_threshold` the user has not yet chosen. This
    one depends on neither, so it exists *today* and it exists for every image regardless of
    how review goes — which makes it usable as a denominator for the per-cell CD68 that does
    depend on both, and as the sanity check that catches a segmentation or threshold choice
    that has quietly lost most of the signal.

    Comparable across files for the same reason `Cd68Measures` is: 32-bit is kept with **no
    per-image rescaling** (§4.13). Under a per-image rescale this number would stop meaning
    anything between images, which is the only way it is ever read.

    **Not a burden fraction and not a mean.** No area or volume normalisation is applied,
    because the marked crops are not all one size and dividing by field volume would answer a
    different question than the 6E10 `integrated_intensity` it sits beside. Field volume is
    already on the row via the territory block if a reader wants the ratio.
    """

    status: str = "ok"
    integrated_intensity: float = float("nan")


def measure_cd68_image(cd68_channel: np.ndarray | None) -> Cd68ImageLoad:
    """Summed CD68 intensity over the **whole image** — no mask, no threshold.

    The direct analogue of ``measure_plaque_burden``'s threshold-free
    ``integrated_intensity``, and computed the same way: a float64 sum over every voxel of
    the channel, so no segmentation and no user choice enters it.

    **No CD68 channel is not zero CD68.** It returns `NaN` with ``no_cd68_channel``, which the
    export writes as a blank beside a status — the rule everywhere in this codebase, and one
    that matters more here than usual: 0.0 is a perfectly plausible-looking amount of CD68 and
    would be read as a measurement.
    """
    out = Cd68ImageLoad()
    if cd68_channel is None or getattr(cd68_channel, "size", 0) == 0:
        out.status = NO_CD68_CHANNEL
        return out
    out.integrated_intensity = float(np.asarray(cd68_channel).astype(np.float64).sum())
    return out


def measure_iba1_image(iba1_channel: np.ndarray | None) -> Cd68ImageLoad:
    """Summed Iba1 intensity over the **whole image** — no mask, no threshold.

    **Added 2026-09-01 to complete a set that had a hole in it.** `images.csv` carried a
    threshold-free burden for 6E10 (`integrated_intensity`) and one for CD68
    (`image_cd68_integrated`) and **none for Iba1** — the microglial marker, and the basis of
    every microgliosis claim the study makes. Every other Iba1 number on that sheet,
    `iba1_volume_um3` and both its halves, depends on `mask_coefficient`; this one depends on
    nothing anyone chose, so it is the robustness check on all three, in exactly the
    relationship `integrated_intensity` already has to `total_volume_um3`.

    Returns the same dataclass as `measure_cd68_image` — the measurement is identical in kind
    and a second class would be two places for one rule to be written. Its status reads
    `no_cd68_channel` for a missing channel, which is the one wart of sharing it; the export
    renames nothing and the column it lands in is `image_iba1_status`.
    """
    out = Cd68ImageLoad()
    if iba1_channel is None or getattr(iba1_channel, "size", 0) == 0:
        out.status = "no_iba1_channel"
        return out
    out.integrated_intensity = float(np.asarray(iba1_channel).astype(np.float64).sum())
    return out


@dataclass
class DebrisLoad:
    """What the measurement mask held that no cell did — one set of numbers per image.

    **Debris is exactly two things**, agreed with the user 2026-08-23: components of the
    measurement mask that **no microglial nucleus claims**, and cells falling under
    `min_cell_volume_um3`. Both reach `measure_debris` the same way, as whatever the
    measurement mask holds once the surviving cells are taken out of it, so the two halves
    cannot drift apart the way two accumulators would.

    **Not** material outside the Iba1 mask entirely — that was never signal. The user's
    argument for keeping the rest: *"anything genuinely not worth keeping is deleted by
    minimum voxels filter in the iba1 thresholding step."*

    > ### AGGREGATE PER IMAGE, NOT PER COMPONENT — the user's choice
    >
    > Three numbers on the file record and **no debris rows in the pool**. That keeps the
    > pool schema's shape and adds fields rather than a table, and it is why debris cannot
    > enter the territory block: `union_surveyed_um3` is a convex hull over per-cell records,
    > and an aggregate volume has no hull. Raised with the user 2026-08-24 and parked.

    > ### `cd68_integrated` IS THRESHOLD-FREE, AND IT HAS TO BE
    >
    > It is the same quantity as `Cd68ImageLoad.integrated_intensity` — a float64 sum over
    > the debris voxels, no threshold anywhere. That is not a simplification: `cd68_threshold`
    > is in `pipeline.PHASE_C_ONLY_FIELDS` and is `None` until item 9a, which runs **after**
    > items 3-7, so nothing computed inside `segment_image` has a threshold available. See
    > `SegmentationParams.phase_a_hash` for why that ordering is load-bearing rather than
    > incidental — *"the pickers would invalidate the pool they were picked from."*
    >
    > It is also the only version that answers the question this measure exists for.
    > `image_cd68_fraction_in_cell_masks` runs ~83% median, and its denominator
    > `image_cd68_integrated` is threshold-free; a thresholded numerator would not subtract
    > from it. A thresholded debris volume at item 9a's T is possible, but only as a **Phase
    > C** measure with the debris mask stored in the pool. Raised 2026-08-24 and parked.

    **`NaN` with a status when there is no CD68 channel, never 0.0** — `measure_cd68_image`'s
    rule, and it matters more here than usual: zero CD68 in debris is a perfectly plausible
    finding and would be read as one.
    """

    status: str = "ok"
    volume_um3: float = 0.0
    n_components: int = 0
    cd68_integrated: float = float("nan")
    cd68_status: str = "ok"

    def summary(self) -> str:
        return f"debris {self.volume_um3:.0f} um3 in {self.n_components} components"


def measure_debris(
    debris_mask: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    cd68: np.ndarray | None = None,
) -> DebrisLoad:
    """Aggregate the debris mask into the three numbers the file record carries.

    ``debris_mask`` is the measurement mask minus every surviving cell, so it already
    contains both halves of the definition and this function does not need to know which
    voxel came from which. ``cd68`` must be the whole channel at native resolution, the
    same array `measure_cd68_image` sums.

    **Components are counted with full 26-connectivity**, matching
    `assign_labels_by_connectivity` — the stage that decided which components were
    unclaimed in the first place. Counting them under a different structure would report a
    different number of debris pieces than the number the split actually left behind.
    """
    if debris_mask.ndim != 3:
        raise ValueError(f"expected a (z, y, x) mask, got {debris_mask.shape}")

    out = DebrisLoad()
    mask = debris_mask.astype(bool, copy=False)
    voxel_volume = float(np.prod([float(v) for v in voxel_size_um]))
    out.volume_um3 = float(mask.sum()) * voxel_volume

    if not mask.any():
        out.n_components = 0
        out.cd68_integrated = 0.0 if cd68 is not None else float("nan")
        out.cd68_status = "ok" if cd68 is not None else NO_CD68_CHANNEL
        out.status = "no_debris"
        return out

    _, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    out.n_components = int(n)

    if cd68 is None or getattr(cd68, "size", 0) == 0:
        out.cd68_status = NO_CD68_CHANNEL
        return out
    if cd68.shape != mask.shape:
        raise ValueError(
            f"cd68 shape {cd68.shape} does not match the debris mask {mask.shape}"
        )
    out.cd68_integrated = float(np.asarray(cd68)[mask].astype(np.float64).sum())
    return out


@dataclass
class PlaqueBurden:
    """Amyloid load for one image — the study's outcome measure, reported two ways.

    The user confirmed fields were chosen by anatomical region, **blind to plaque content**,
    so burden here is a legitimate outcome rather than a consequence of where the crop was
    taken. That is what makes it worth measuring carefully.

    ``integrated_intensity`` needs **no threshold** and is the defensible number. Everything
    below it depends on ``coefficient``, and §4.13.6 showed there is **no plateau** — burden
    varies 7-16x across plausible thresholds — so the segmented figures travel with the
    coefficient that produced them and should never be quoted bare.

    The **size distribution** is kept, not just the total: a treatment that shifts many small
    plaques into fewer large ones without changing total amyloid would be invisible in a
    single burden number.
    """

    status: str = "ok"
    integrated_intensity: float = float("nan")      # threshold-free
    n_plaques: int = -1
    total_volume_um3: float = float("nan")
    volume_fraction: float = float("nan")           # of the imaged field
    median_plaque_um3: float = float("nan")
    largest_plaque_um3: float = float("nan")
    #: The smallest plaque **above the floor**, so in any image with a reasonable plaque count
    #: it sits near the floor and is nearly a constant. Reported because it is the direct
    #: check that one floor is applied uniformly; **do not read it as a finding.**
    smallest_plaque_um3: float = float("nan")
    plaque_volumes_um3: list[float] = field(default_factory=list)
    #: The absolute 6E10 intensity chosen in the semi-auto UI. `NaN` when unchosen.
    threshold: float = float("nan")
    min_plaque_volume_um3: float = float("nan")

    #: Burden volume at ``T x CURVE_FACTORS``. **Volume only** — a count across thresholds
    #: mostly reports how segmentation fragmented rather than how much amyloid there is
    #: (`EXPORT_SPEC.md` §5). Empty when no threshold is set.
    volume_curve_um3: dict[float, float] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """Whether the threshold-dependent measures are populated."""
        return self.n_plaques >= 0

    def summary(self) -> str:
        if self.n_plaques < 0:
            return f"plaque burden unmeasured ({self.status})"
        return (f"{self.n_plaques} plaques, {self.total_volume_um3:.4g} um3 "
                f"({self.volume_fraction:.2%} of field) at threshold {self.threshold:g}")


@dataclass
class PlaqueProximity:
    """Distance from one cell to the nearest amyloid, two ways."""

    status: str = "ok"
    soma_to_plaque_um: float = float("nan")
    nearest_voxel_to_plaque_um: float = float("nan")


#: Tolerances (µm) at which contact between two objects is reported. A curve, not a constant,
#: because "touching" has no ground truth. **0.3 is half the z-spacing** — the smallest
#: tolerance that can mean anything when one z step is 0.6 µm and segmentation noise routinely
#: leaves a one-voxel gap between objects that are physically in contact. A tolerance of
#: exactly 0 would report zero contact for most genuinely touching pairs, which would be an
#: artefact of the z sampling rather than a measurement.
#: **0.5 became 0.6 on 2026-09-01, and the old justification for 0.3 was wrong.** `EXPORT_SPEC`
#: §2 claimed 0.3 µm was *"the smallest tolerance that can mean anything at 5.5:1 anisotropy"*
#: because it is half the z-spacing. Half a voxel is **below** what z can express: the distance
#: transform is sampled at (0.6, 0.108, 0.108) µm, so the smallest non-zero z displacement is a
#: whole 0.6 µm voxel and any tolerance under that admits none at all. `@0.3` therefore measures
#: **in-plane adjacency only** and is kept deliberately on those terms; 0.6 is the smallest
#: tolerance that admits a single z step, which is what 0.3 was believed to be doing.
CONTACT_TOLERANCES_UM = (0.3, 0.6, 1.0)

#: **Recruitment scale, a different question from contact.** `n_somata_within@r` asks whether
#: cell bodies have gathered around a plaque, and at contact tolerances that is zero for every
#: plaque in every image — a soma is 5-10 µm across, so its centroid is several µm from anything
#: it touches. Kept as its own constant, and its own `@r` suffix in the export, so that `@1` and
#: `@5` are never read as two points on one scale.
RECRUITMENT_DISTANCES_UM = (5.0, 10.0, 20.0)

#: The DAPI dense-core coefficient used **per plaque** — one value, not a curve, because
#: `plaques.csv` is a per-object table and 5 coefficients would multiply every row by five.
#:
#: **1.5 is the measured separation peak** (`tuning_runs/dapi_core_calibration.txt`, 20
#: images), replacing an unjustified 1.0. Unlike every other curve here this one has a
#: **maximum**: 5xFAD/WT separation rises 26.6x -> 29.3x -> 30.9x -> **32.3x at 1.5** and then
#: falls to 28.6x, 14.5x, 7.9x. Above ~1.5 real 5xFAD cores are lost faster than WT signal.
#:
#: The image-level compartments keep all five coefficients — see `DAPI_CORE_COEFFICIENTS`.
PLAQUE_DAPI_COEFFICIENT = 1.5


@dataclass
class ContactMeasures:
    """How much of one object's surface lies within *d* µm of another, and how much overlaps.

    Gotkiewicz et al. (Glia 2025) report this as percent of a plaque's surface covered by
    Iba1+ processes — 11-30% of the diffuse shell against 20-94% of the dense core — which is
    the per-plaque direction of the same measurement.
    """

    status: str = "ok"
    surface_area_um2: float = float("nan")
    #: Contact area in µm² at each tolerance in `CONTACT_TOLERANCES_UM`.
    contact_area_um2: dict[float, float] = field(default_factory=dict)
    #: The same, as a fraction of this object's own surface.
    contact_fraction: dict[float, float] = field(default_factory=dict)
    #: Voxels in **both** masks. Reported separately and never folded into contact: a voxel
    #: can be above threshold in two different channels, and internalised amyloid is a
    #: different quantity from adjacent amyloid — which is the distinction the engulfment
    #: literature exists to draw.
    overlap_volume_um3: float = float("nan")


def measure_contact(
    mask: np.ndarray,
    other: np.ndarray | None,
    *,
    voxel_size_um: Sequence[float],
    tolerances: Sequence[float] = CONTACT_TOLERANCES_UM,
    other_distance_um: np.ndarray | None = None,
) -> ContactMeasures:
    """Surface of ``mask`` lying within each tolerance of ``other``.

    The surface is **meshed in physical microns** (`morphometrics.surface_mesh`), and a face
    counts as contact when its centroid is within *d* µm of ``other``. One mechanism serves
    every tolerance, so the curve is internally consistent.

    ``other_distance_um`` lets the caller pass a distance field computed once whole-image —
    which is how the plaque case is driven, since a per-crop transform would measure distance
    to the nearest plaque *inside the bounding box* rather than in the image.
    """
    from .morphometrics import _triangle_areas, surface_mesh

    out = ContactMeasures()
    if mask is None or not mask.any():
        out.status = "empty_mask"
        return out

    spacing = np.asarray([float(v) for v in voxel_size_um], float)
    voxel_volume = float(spacing.prod())

    if other_distance_um is None:
        if other is None or not other.any():
            out.status = "no_other_object"
            out.surface_area_um2 = float("nan")
            return out
        other_distance_um = ndimage.distance_transform_edt(
            ~other.astype(bool), sampling=tuple(spacing))

    if other is not None:
        out.overlap_volume_um3 = float((mask & other.astype(bool)).sum()) * voxel_volume

    mesh = surface_mesh(mask, voxel_size_um)
    if mesh is None:
        out.status = "too_small_to_mesh"
        return out
    verts, faces = mesh
    areas = _triangle_areas(verts, faces)
    out.surface_area_um2 = float(areas.sum())

    # Face centroids are in microns; convert back to voxel indices to read the field.
    centroids = verts[faces].mean(axis=1) / spacing
    idx = np.rint(centroids).astype(int)
    for axis in range(3):
        np.clip(idx[:, axis], 0, other_distance_um.shape[axis] - 1, out=idx[:, axis])
    face_distance = other_distance_um[idx[:, 0], idx[:, 1], idx[:, 2]]

    for tol in tolerances:
        area = float(areas[face_distance <= float(tol)].sum())
        out.contact_area_um2[float(tol)] = area
        out.contact_fraction[float(tol)] = (area / out.surface_area_um2
                                            if out.surface_area_um2 > 0 else float("nan"))
    return out


def background_level(
    channel: np.ndarray,
    exclude: Sequence[np.ndarray] = (),
    *,
    percentiles: Sequence[float] = (50.0, 75.0, 90.0),
) -> dict[float, float]:
    """Signal-free reference level, as percentiles of voxels outside every mask given.

    **A percentile, not `median + k*MAD`.** That scheme is degenerate on this data: Huygens
    CMLE converges to a constant positive floor in signal-free regions, so **MAD is exactly
    zero on every image** and `median + k*MAD` reduces to the median for every `k`. A sweep
    over `k` would have produced identical columns and been read as a reassuring plateau —
    the robustness check manufacturing the artefact it exists to detect (§4.13.3).

    Returns the level at each percentile so the sensitivity is inspectable, rather than
    committing the pipeline to one assumption about the background's shape.
    """
    outside = np.ones(channel.shape, bool)
    for m in exclude:
        if m is not None:
            outside &= ~m.astype(bool)
    if not outside.any():
        return {float(p): float("nan") for p in percentiles}
    values = channel[outside].astype(np.float64)
    return {float(p): float(np.percentile(values, float(p))) for p in percentiles}


def build_plaque_distance_field(
    plaque_mask: np.ndarray,
    voxel_size_um: Sequence[float],
    *,
    min_plaque_volume_um3: float = 0.0,
) -> tuple[np.ndarray | None, str]:
    """Distance in microns from every voxel to the nearest plaque, computed once per image.

    Built whole-image rather than per cell because the transform costs the same for one cell
    as for all of them, and a per-cell crop would measure distance to the nearest plaque
    *inside the crop* — which is not the nearest plaque.

    ``sampling`` carries the voxel size into the transform. Without it the result would be in
    voxels, and at 5.5:1 anisotropy a distance in voxels is not proportional to a distance in
    microns in any direction-independent way.

    Returns ``(field, status)``; the field is ``None`` when there is nothing to measure from,
    which is the expected case for WT tissue.
    """
    if plaque_mask is None or not plaque_mask.any():
        return None, "no_plaques_found"

    mask = plaque_mask.astype(bool, copy=False)
    if min_plaque_volume_um3 > 0:
        z_um, y_um, x_um = (float(v) for v in voxel_size_um)
        min_voxels = int(round(min_plaque_volume_um3 / (z_um * y_um * x_um)))
        labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
        if n:
            sizes = np.bincount(labels.ravel())
            keep = np.zeros(sizes.size, bool)
            keep[1:] = sizes[1:] >= min_voxels
            mask = keep[labels]
        if not mask.any():
            return None, "no_plaques_above_size_cutoff"

    field = ndimage.distance_transform_edt(~mask, sampling=tuple(
        float(v) for v in voxel_size_um))
    return field.astype(np.float32), "ok"


def _plaque_sizes(plaque_channel: np.ndarray, threshold: float, voxel_volume: float,
                  min_plaque_volume_um3: float) -> np.ndarray:
    """Volumes in µm³ of the plaques found above one **absolute** intensity, size-filtered.

    Absolute, not a per-slice Otsu multiple: acquisition was held constant across images, so
    one intensity picked once means the same thing in every file, which is what a batch
    measurement needs and what a multiplier cannot give (`EXPORT_SPEC.md` §2).
    """
    mask = np.asarray(plaque_channel) > float(threshold)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return np.zeros(0, float)
    sizes = np.bincount(labels.ravel())[1:] * voxel_volume
    return sizes[sizes >= float(min_plaque_volume_um3)]


def build_plaque_fields(
    plaque_mask: np.ndarray,
    voxel_size_um: Sequence[float],
    *,
    min_plaque_volume_um3: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Distance field **and** the id of the nearest plaque, from one transform.

    `distance_transform_edt(..., return_indices=True)` already computes, for every voxel, the
    coordinates of the nearest background voxel — so carrying the *label* costs one extra
    array and no extra traversal. That is what makes per-plaque clustering counts
    (§5.5 of the literature review — Tsering's ">=2 touching glia") essentially free.

    Returns ``(distance_um, nearest_label, labels, status)``. Kept separate from
    :func:`build_plaque_distance_field` rather than widening its return, so existing callers
    are untouched.

    **`labels` joined the return on 2026-09-02** and is what makes `plaques_within` exact. It
    was already computed here and thrown away: the size floor is applied to it, and
    ``nearest_label`` is a lookup *into* it. Returning it costs nothing and removes the reason
    the link table was approximate.
    """
    if plaque_mask is None or not plaque_mask.any():
        return None, None, None, "no_plaques_found"

    spacing = tuple(float(v) for v in voxel_size_um)
    voxel_volume = float(np.prod(spacing))
    labels, n = ndimage.label(plaque_mask.astype(bool), structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return None, None, None, "no_plaques_found"

    if min_plaque_volume_um3 > 0:
        sizes = np.bincount(labels.ravel()) * voxel_volume
        keep = np.zeros(sizes.size, bool)
        keep[1:] = sizes[1:] >= float(min_plaque_volume_um3)
        labels = np.where(keep[labels], labels, 0)
        if not labels.any():
            return None, None, None, "no_plaques_above_size_cutoff"

    mask = labels > 0
    distance, indices = ndimage.distance_transform_edt(
        ~mask, sampling=spacing, return_indices=True)
    nearest = labels[indices[0], indices[1], indices[2]]
    return (distance.astype(np.float32), nearest.astype(np.int32),
            labels.astype(np.int32), "ok")


def plaques_within(
    cell_mask: np.ndarray,
    plaque_labels: np.ndarray | None,
    *,
    voxel_size_um: Sequence[float],
    tolerances: Sequence[float] = CONTACT_TOLERANCES_UM,
) -> dict[float, dict[int, float]]:
    """Which plaque ids this cell reaches within each tolerance, **and how far**. Exact.

    ``{tolerance: {plaque_id: closest distance in µm}}``. This is what fills
    `cell_plaque_links.csv`, the only join between cells and plaques — every other plaque
    measure is anonymous (`nearest_voxel_to_plaque_um` says how far but not *which*,
    `n_cells_within` says how many but not *who*). Per-pair distance is carried rather than
    bare membership so proximity can be weighted, instead of treating everything inside a
    tolerance as equivalent.

    ``plaque_labels`` is the label volume **cropped to this cell's window, padded by at least
    the largest tolerance** — the caller's job, because only it knows the bbox.

    > ### ⇒ EXACT SINCE 2026-09-02. IT READ THE NEAREST-PLAQUE FIELD BEFORE, AND MISSED PAIRS.
    >
    > The old version looked up `nearest_label` at each cell voxel, so it recorded a plaque only
    > when that plaque was the **closest** one to at least one voxel. A plaque within tolerance
    > that was never nearest — a third plaque behind two others — was missed, and §5 called that
    > *"rare at 0.3-1.0 µm"* without ever measuring it.
    >
    > **Measured, it was not rare.** `scripts/link_table_gap.py`, probing only the voxels where a
    > miss is geometrically possible: a 3 µm blob missed **nothing** across 362 pairs, but a
    > **0.25 µm process** — the width of a real one — missed **up to 2.2% at 0.3 µm, 8.8% at 0.6
    > and 28.3% at 1.0**. A miss needs *every* one of the cell's voxels to prefer another plaque,
    > so only something thin can produce one. What was lost is a cell reaching a **second**
    > plaque with a process, in exactly the clumped population the study is about.
    >
    > **The transform is inverted, which is why this fits here at all.** The exact method was
    > recorded as "one distance transform per plaque", which belongs in the plaque loop and
    > therefore in the wrong stage. Measuring **from the cell** and reading the result at the
    > plaque voxels gives the same number — the closest approach between the two — as **one
    > transform per cell**, which is where this function already runs.
    >
    > **Exact despite the window.** Anything outside a window padded by the largest tolerance is
    > by construction further than that tolerance from the cell, so it cannot lower any minimum
    > this reports.
    """
    out: dict[float, dict[int, float]] = {float(t): {} for t in tolerances}
    if plaque_labels is None or not cell_mask.any():
        return out
    labels = np.asarray(plaque_labels)
    if labels.shape != cell_mask.shape or not (labels > 0).any():
        return out

    spacing = np.asarray([float(v) for v in voxel_size_um], float)
    reach = float(max(float(t) for t in tolerances))
    cell = cell_mask.astype(bool)

    #: > ### ⇒ ONE TRANSFORM PER **PLAQUE**, NOT PER CELL — measured 2026-09-02
    #: >
    #: > The first implementation inverted it: one transform from the cell over the cell's own
    #: > padded window, read at the plaque voxels. Same answer, and **2.4 seconds per cell** —
    #: > 13 hours across a 20,000-cell study. A microglial cell's bounding box spans ~50 µm and
    #: > the cell fills about 1% of it, so transforming over that box pays for ten million
    #: > voxels to measure sixty thousand.
    #: >
    #: > A plaque is compact, so its own padded box is small. This is why `measure_plaques` can
    #: > already afford exactly this computation for `n_cells_within@d`, and the estimate that
    #: > said otherwise was never measured.
    #: **Every plaque's box in one pass.** Locating them with `argwhere` per label re-scanned
    #: the whole window each time, which dominated the cost once the transforms were small —
    #: `find_objects` returns all of them from a single traversal.
    pad = np.ceil(reach / spacing).astype(int)
    boxes = ndimage.find_objects(labels)
    for index, extent in enumerate(boxes, start=1):
        if extent is None:
            continue
        pid = index
        lo = np.maximum([s.start for s in extent] - pad, 0)
        hi = np.minimum([s.stop for s in extent] + pad, labels.shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

        sub_cell = cell[box]
        if not sub_cell.any():
            continue
        own = ndimage.distance_transform_edt(~(labels[box] == int(pid)),
                                             sampling=tuple(spacing))
        closest = float(own[sub_cell].min())
        for tol in tolerances:
            if closest <= float(tol):
                out[float(tol)][int(pid)] = closest
    return {t: dict(sorted(v.items())) for t, v in out.items()}


def measure_plaque_burden(
    plaque_channel: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    threshold: float | None = None,
    factors: Sequence[float] = CURVE_FACTORS,
    min_plaque_volume_um3: float = MIN_PLAQUE_VOLUME_UM3,
) -> PlaqueBurden:
    """Amyloid load for one image: threshold-free intensity, plus a segmented distribution.

    Both are wanted, for the same reason CD68 keeps both. **`integrated_intensity` needs no
    cutoff and is the study's outcome measure** — it is recorded unconditionally, before
    anything that depends on a threshold, so it lands even in an image where segmentation
    finds nothing. Segmentation costs a threshold but buys the **count**, the **size
    distribution**, and the objects the distance transform needs.

    ``threshold`` is an **absolute 6E10 intensity chosen by the user in the semi-auto UI**,
    with live preview. It replaces the `plaque_measure_coefficient` Otsu multiplier, which had
    no default and could not acquire one: across 5 WT and 10 5xFAD images
    (`tuning_runs/plaque_sweep.txt`) the 5xFAD/WT separation rises **monotonically** — 16x at
    0.25 to 59x at 2.0 — with no plateau and therefore no turning point to read a value off.
    An absolute threshold dissolves that problem rather than solving it: acquisition was held
    constant across images, so one intensity picked once applies unchanged to the batch.

    **Passing none is not an error and never becomes a guess.** The threshold-free intensity
    is returned, every threshold-dependent field is `NaN`, the curve is empty, and the status
    reads ``threshold_unset`` — so a blank can never be read as a zero.
    """
    z_um, y_um, x_um = (float(v) for v in voxel_size_um)
    voxel_volume = z_um * y_um * x_um
    out = PlaqueBurden(
        threshold=float(threshold) if threshold is not None else float("nan"),
        min_plaque_volume_um3=float(min_plaque_volume_um3))

    if plaque_channel is None or plaque_channel.size == 0:
        out.status = "no_plaque_channel"
        return out

    # Threshold-free first, so it is recorded even when segmentation finds nothing.
    out.integrated_intensity = float(plaque_channel.astype(np.float64).sum())
    field_volume = plaque_channel.size * voxel_volume

    if threshold is None:
        # Not an error: integrated intensity above is complete and primary. The status
        # records why every other column is NaN, so a blank is never read as a zero. There
        # is deliberately no curve — its levels are multiples of T, so without T there is
        # nothing to bracket.
        out.status = "threshold_unset"
        return out

    for factor in factors:
        level = float(threshold) * float(factor)
        sizes = _plaque_sizes(plaque_channel, level, voxel_volume, min_plaque_volume_um3)
        out.volume_curve_um3[level] = float(sizes.sum())

    sizes = _plaque_sizes(plaque_channel, float(threshold), voxel_volume,
                          min_plaque_volume_um3)
    if sizes.size == 0:
        out.status = "no_plaques_above_size_cutoff"
        out.n_plaques = 0
        out.total_volume_um3 = 0.0
        out.volume_fraction = 0.0
        return out

    out.n_plaques = int(sizes.size)
    out.total_volume_um3 = float(sizes.sum())
    out.volume_fraction = out.total_volume_um3 / field_volume
    out.median_plaque_um3 = float(np.median(sizes))
    out.largest_plaque_um3 = float(sizes.max())
    out.smallest_plaque_um3 = float(sizes.min())
    out.plaque_volumes_um3 = sorted(float(v) for v in sizes)
    return out


@dataclass
class PlaqueCompartments:
    """Diffuse shell against dense core, each by volume **and** by intensity.

    **Exploratory, and it must be labelled so wherever it is used.** Gotkiewicz et al. (Glia
    2025) classify plaques as diffuse, fibrillar and dense-core using **high-intensity DAPI**,
    cross-validated against Thioflavin-S and X34. This tissue has neither dye, so the
    cross-validation cannot be repeated here.

    What exists is consistency, not validation: the plaque veto measurements show DAPI inside
    plaques at **0.26-25.4% in 5xFAD against 0.0-1.7% in WT**. WT has nuclei everywhere and no
    amyloid, so that asymmetry is what fibril signal looks like and is not what nuclei look
    like. A strong prior; not proof.

    Why it is worth having: Farrar et al. (Acta Neuropathol 2020) found volume-matched cored
    and diffuse plaques differ roughly **three-fold in total amyloid output**, so a pipeline
    reporting only plaque volume can show two groups as pathologically equivalent when their
    amyloid density differs substantially.
    """

    status: str = "ok"
    #: Keyed by the DAPI coefficient — a curve, since this cutoff has no validated value.
    core_volume_um3: dict[float, float] = field(default_factory=dict)
    shell_volume_um3: dict[float, float] = field(default_factory=dict)
    core_fraction: dict[float, float] = field(default_factory=dict)
    core_integrated_6e10: dict[float, float] = field(default_factory=dict)
    shell_integrated_6e10: dict[float, float] = field(default_factory=dict)
    core_mean_6e10: dict[float, float] = field(default_factory=dict)
    shell_mean_6e10: dict[float, float] = field(default_factory=dict)


def measure_plaque_compartments(
    plaque_mask: np.ndarray,
    plaque_channel: np.ndarray,
    dapi_raw: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    dapi_coefficients: Sequence[float] = DAPI_CORE_COEFFICIENTS,
) -> PlaqueCompartments:
    """Split amyloid into dense core and diffuse shell using DAPI as a fibril marker.

    ``dapi_raw`` **must be the raw DAPI channel, before the plaque veto.** The veto zeroes
    DAPI inside plaques before Cellpose sees it (§4.13.1) — which is precisely the signal this
    measure depends on. Handed the vetoed channel it would report an empty core, and it would
    do so silently: a well-formed zero rather than an error.
    ``test_compartments_read_raw_dapi_not_the_vetoed_channel`` pins this.

    The cutoff is a **curve** over ``dapi_coefficients`` (multiples of Otsu on the DAPI values
    *inside* plaques), because no validated value exists and inventing one would decide the
    diffuse/dense question by fiat.
    """
    from .threshold import threshold_channel  # noqa: F401  (kept for symmetry of imports)
    from skimage.filters import threshold_otsu

    out = PlaqueCompartments()
    if plaque_mask is None or not plaque_mask.any():
        out.status = "no_plaques_found"
        return out
    if dapi_raw is None or dapi_raw.shape != plaque_mask.shape:
        out.status = "no_dapi_channel"
        return out

    spacing = np.asarray([float(v) for v in voxel_size_um], float)
    voxel_volume = float(spacing.prod())
    mask = plaque_mask.astype(bool)
    inside = dapi_raw[mask].astype(np.float64)
    amyloid = plaque_channel[mask].astype(np.float64)
    if inside.size == 0 or float(inside.max()) <= float(inside.min()):
        out.status = "dapi_uniform_inside_plaques"
        return out

    base = float(threshold_otsu(inside))
    for coef in dapi_coefficients:
        level = base * float(coef)
        core = inside > level
        shell = ~core
        c = float(coef)
        out.core_volume_um3[c] = float(core.sum()) * voxel_volume
        out.shell_volume_um3[c] = float(shell.sum()) * voxel_volume
        total = out.core_volume_um3[c] + out.shell_volume_um3[c]
        out.core_fraction[c] = out.core_volume_um3[c] / total if total > 0 else float("nan")
        out.core_integrated_6e10[c] = float(amyloid[core].sum())
        out.shell_integrated_6e10[c] = float(amyloid[shell].sum())
        out.core_mean_6e10[c] = (float(amyloid[core].mean()) if core.any()
                                 else float("nan"))
        out.shell_mean_6e10[c] = (float(amyloid[shell].mean()) if shell.any()
                                  else float("nan"))
    return out


def _plaque_geometry(local: np.ndarray, box, spacing: np.ndarray,
                     image_shape) -> dict[str, float]:
    """Shape and position of one plaque — added 2026-09-01.

    **`solidity` is why `sphericity` can be left alone.** Sphericity's marching-cubes ceiling is
    ~0.93 and is *worse for smaller objects* (`morphometrics.sphericity`), so across a sheet
    whose plaques run from the 25 µm³ floor to the largest cored plaque it is confounded with
    volume. Solidity — volume over convex-hull volume — has no such size term, so the pair
    together distinguish a genuinely compact plaque from a small one.

    `max_feret_um` is the hull's diameter and is nearly free once the hull exists.
    `equivalent_diameter_um` carries no shape information at all — it is the volume restated as
    a length — and is here only because the amyloid literature quotes plaque size in µm.

    **The border flags qualify three columns, not one.** A plaque cut by the frame has an
    understated volume, a sphericity corrupted by a flat cut face, and a systematically
    understated `covered_fraction_*`: the cut face is surface area no cell can ever lie within
    `d` of, so it enters the denominator and never the numerator.
    """
    from scipy.spatial import ConvexHull

    pts = np.argwhere(local)
    origin = np.asarray([s.start for s in box], float)
    voxel_volume = float(spacing.prod())
    volume = float(pts.shape[0]) * voxel_volume
    centroid = (pts.mean(axis=0) + origin) * spacing

    out = {
        "centroid_z_um": float(centroid[0]),
        "centroid_y_um": float(centroid[1]),
        "centroid_x_um": float(centroid[2]),
        "equivalent_diameter_um": float((6.0 * volume / np.pi) ** (1.0 / 3.0))
        if volume > 0 else float("nan"),
        "solidity": float("nan"),
        "max_feret_um": float("nan"),
        #: A plaque whose bounding box reaches any face of the image was cut by the frame.
        #: The two axes are kept apart for the same reason cells keep them apart: z is an
        #: optical limit and xy is a field-of-view edge, and they are not the same caveat.
        "touches_xy_border": bool(box[1].start == 0 or box[2].start == 0
                                  or box[1].stop >= int(image_shape[1])
                                  or box[2].stop >= int(image_shape[2])),
        "touches_z_border": bool(box[0].start == 0 or box[0].stop >= int(image_shape[0])),
    }

    #: A hull needs four non-coplanar points. A flat or linear plaque has none, which is a real
    #: condition rather than an error — both hull measures stay `NaN` and the rest still report.
    physical = pts * spacing
    try:
        hull = ConvexHull(physical)
        verts = physical[hull.vertices]
        if verts.shape[0] >= 2:
            diff = verts[:, None, :] - verts[None, :, :]
            out["max_feret_um"] = float(np.sqrt((diff ** 2).sum(-1)).max())
    except Exception:
        pass

    #: > ### ⇒ SOLIDITY IS RASTERISED, NOT `ConvexHull.volume` — and the first version was wrong
    #: >
    #: > `ConvexHull` is built over voxel **centres**, so its volume excludes the half-voxel
    #: > shell around the outermost points, while `volume` above is a voxel **count**. Dividing
    #: > one by the other mixes two conventions and **exceeds 1**: measured at **1.149** on a
    #: > 6-voxel-radius sphere, for a quantity that is a fraction by definition.
    #: >
    #: > Rasterising the hull and counting its voxels puts numerator and denominator in the same
    #: > units, which is what `skimage`'s own `regionprops.solidity` does. Bounded cost, because
    #: > it runs on the plaque's own bounding box rather than the image.
    try:
        from skimage.morphology import convex_hull_image
        filled = convex_hull_image(local)
        n_hull = int(filled.sum())
        if n_hull > 0:
            out["solidity"] = float(pts.shape[0]) / float(n_hull)
    except Exception:
        pass
    return out


def measure_plaques(
    plaque_mask: np.ndarray,
    plaque_channel: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    cell_labels: np.ndarray | None = None,
    cell_weights: Mapping[int, int] | None = None,
    iba1_mask: np.ndarray | None = None,
    tolerances: Sequence[float] = CONTACT_TOLERANCES_UM,
    recruitment: Sequence[float] = RECRUITMENT_DISTANCES_UM,
    soma_centroids: Sequence[Sequence[float]] | None = None,
    min_plaque_volume_um3: float = MIN_PLAQUE_VOLUME_UM3,
) -> list[dict]:
    """One record per plaque — the per-object view the image-level burden cannot give.

    A single burden figure hides the thing a treatment is most likely to change: §5.2 of the
    literature review notes that a shift from many small plaques to fewer large ones leaves
    total amyloid unmoved. So each plaque reports its own volume, **sphericity** (a compaction
    proxy, §5.2), how much of its surface microglia cover (§5.5, Gotkiewicz's statistic), how
    many distinct cells reach it (§5.5, Tsering's clustering), and its dense-core fraction if
    DAPI is supplied (§5.4).

    ``cell_labels`` is a whole-image label volume — distinct integers per cell, 0 elsewhere,
    the same shape as ``plaque_mask``. Without it the coverage and clustering columns are
    omitted rather than guessed.

    Both cell-facing measures are **per-plaque and non-exclusive**: ``covered_fraction@d`` is
    the share of *this* plaque's surface with a cell within ``d``, and ``n_cells_within@d`` the
    number of distinct cells with a voxel within ``d`` of *this* plaque. A cell between two
    plaques is counted by both, so the counts do not sum to the number of cells in the image.
    The nearest-plaque attribution — each cell assigned to one plaque — is a different measure
    and lives in ``cell_plaque_links.csv`` (`EXPORT_SPEC.md` §5).
    """
    from .morphometrics import _triangle_areas, sphericity, surface_area_um2, surface_mesh

    if plaque_mask is None or not plaque_mask.any():
        return []
    spacing = np.asarray([float(v) for v in voxel_size_um], float)
    voxel_volume = float(spacing.prod())

    labels, n = ndimage.label(plaque_mask.astype(bool), structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return []

    cell_distance = None
    if cell_labels is not None:
        cell_distance = ndimage.distance_transform_edt(
            cell_labels == 0, sampling=tuple(spacing))

    #: **A second distance field, from the MATERIAL rather than from the cells.** `iba1_mask` is
    #: the whole measurement mask, so it carries the debris a cell-based field cannot see — and
    #: unclaimed material concentrates at plaques, because that is where the clumping that
    #: defeats splitting happens. The cell-based coverage therefore under-reports exactly at the
    #: plaques the study cares most about, and the gap between the two columns measures it.
    #:
    #: A separate column rather than a redefinition: `covered_fraction_cells@d` has already been
    #: reported under its own meaning, and folding debris into it would change a validated
    #: number without changing its name.
    iba1_distance = None
    if iba1_mask is not None:
        iba1_distance = ndimage.distance_transform_edt(
            ~np.asarray(iba1_mask).astype(bool), sampling=tuple(spacing))

    out: list[dict] = []
    objects = ndimage.find_objects(labels)
    for pid, box in enumerate(objects, start=1):
        if box is None:
            continue
        local = labels[box] == pid
        volume = float(local.sum()) * voxel_volume
        if volume < float(min_plaque_volume_um3):
            continue

        area = surface_area_um2(local, voxel_size_um)
        rec: dict = {
            "plaque_id": pid,
            "volume_um3": volume,
            #: **Exported since 2026-09-01.** It was computed and withheld on the grounds that
            #: only the coverage columns needed it — but it is the *denominator* of six of
            #: them, and coverage pooled across plaques is `Σcovered / Σtotal`, which the file
            #: could not express. Without it the only computable figure was the mean of the
            #: fractions, which weights a 25 µm³ plaque like one fifty times larger.
            "surface_area_um2": area,
            "sphericity": sphericity(volume, area),
            **_plaque_geometry(local, box, spacing, labels.shape),
        }

        #: > ### ⇒ THE IBA1 COVERAGE NO LONGER HIDES BEHIND THE CELL LABELS — 2026-09-01
        #: >
        #: > This sat inside `if cell_distance is not None:`, so an image with no cell labels
        #: > lost `covered_fraction_iba1@d` as well — a measure taken off the **measurement
        #: > mask**, which does not depend on the cell set in any way. The two families exist
        #: > precisely so the gap between them reads as material no surviving cell claimed;
        #: > making one conditional on the other's input meant that gap vanished silently in
        #: > exactly the images where the cells were hardest to find. The mesh is built once and
        #: > each family now guards only on its own distance field.
        mesh = (surface_mesh(local, voxel_size_um)
                if (cell_distance is not None or iba1_distance is not None) else None)
        if mesh is not None:
            verts, faces = mesh
            areas = _triangle_areas(verts, faces)
            origin = np.array([s.start for s in box], float) * spacing
            centroids = (verts[faces].mean(axis=1) + origin) / spacing
            idx = np.rint(centroids).astype(int)
            shape_ref = cell_distance if cell_distance is not None else iba1_distance
            for axis in range(3):
                np.clip(idx[:, axis], 0, shape_ref.shape[axis] - 1, out=idx[:, axis])
            total_area = float(areas.sum())
            for field_, family in ((cell_distance, "covered_fraction_cells"),
                                   (iba1_distance, "covered_fraction_iba1")):
                if field_ is None:
                    continue
                d = field_[idx[:, 0], idx[:, 1], idx[:, 2]]
                for tol in tolerances:
                    covered = float(areas[d <= float(tol)].sum())
                    rec[f"{family}@{float(tol):g}"] = (
                        covered / total_area if total_area > 0 else float("nan"))

        if cell_distance is not None:
            # Clustering: distinct cells reaching this plaque. Tsering's ">=2 touching"
            # is reported as one labelled column among the distances, not as the only form.
            #
            # **What the number means.** Cells with at least one voxel within `d` µm of *this*
            # plaque. It is **not** "cells for which this is the nearest plaque": a cell lying
            # between two plaques is counted by both, and the per-plaque counts therefore do
            # not partition the cells in the image. That is the statistic the column names, and
            # it is deliberately not the nearest-plaque attribution `cell_plaque_links.csv`
            # uses (`EXPORT_SPEC.md` §5) — reusing that field here would have been cheaper and
            # would have silently answered a different question.
            #
            # The window is the plaque's bounding box **padded by the largest tolerance**.
            # Until 2026-08-12 both the distance transform and the cell labels were cropped to
            # `box` itself, so a cell was countable only if it interdigitated the plaque's
            # bounding cuboid: the neighbourhood was clipped at the box faces and could not
            # look outward at all. `EXPORT_AUDIT.md` §3.2 — a cell 0.2166 µm away scored 0 at
            # every tolerance while `covered_fraction@1` in the same row read 0.995, and 0 is
            # a plausible enough answer that nothing downstream could notice.
            #
            # Padding is exact, not merely generous: a voxel within `d` µm of the plaque is at
            # most `ceil(d / spacing[i])` indices outside the plaque's bbox on axis `i`, so
            # every cell that can qualify has voxels inside the padded window. The plaque
            # itself is whole inside that window, so the distances are the whole-image ones —
            # the crop bounds which voxels are *asked*, never what the answer is. Cost stays
            # per-plaque and bounded: at 0.6 × 0.1083 × 0.1083 µm the 1.0 µm pad is 2 planes
            # and 10 columns, and no second whole-image transform is built.
            reach = float(max((float(t) for t in tolerances), default=0.0))
            pad = np.ceil(reach / spacing).astype(int)
            near_box = tuple(
                slice(max(0, int(s.start) - int(p)), min(int(dim), int(s.stop) + int(p)))
                for s, p, dim in zip(box, pad, labels.shape))
            near = labels[near_box] == pid
            grown = ndimage.distance_transform_edt(~near, sampling=tuple(spacing))
            sub = cell_labels[near_box]
            #: > ### ⇒ A LABEL MAY BE WORTH MORE THAN ONE CELL — 2026-08-30
            #: >
            #: > This counted **labels**, `ids.size`, which was the same number as *cells*
            #: only while every label was one cell. A `real_cell` rejection carrying the
            #: reviewer's *"this is 3 cells I cannot afford to split"* is **one label worth
            #: three**, so a plaque engaged by that clump now reports 3.
            #: >
            #: > **The weight is per label and defaults to 1**, so an image with no clump
            #: measures exactly what it measured before — and a caller passing no weights at
            #: all gets `ids.size` back, which is what every test fixture predating this does.
            #: >
            #: > `covered_fraction@d` is deliberately untouched beside it: it is an **area**,
            #: and how many cells a reviewer judged a clump to be does not change how much of
            #: the plaque's surface that clump covers.
            #: > ### ⇒ MULTIPLICITY IS NO LONGER FACTORED HERE — the user, 2026-09-01
            #: >
            #: > This weighted each label by `n_real_cells_here`, per §42's rule that *R is
            #: > always multiplicity-factoring*. **The user made this an exception**, for the
            #: > same reason `avg_nearest_neighbour_um` is one: a clump the reviewer judged to
            #: > be three cells is still **one physical object** touching this plaque, and
            #: > counting it three times invents two more contacts than exist. The rule and its
            #: > exceptions are registered in `EXPORT_SPEC.md` §42.
            #: >
            #: > `cell_weights` stays on the signature — it is still passed by `phasec` and is
            #: > still what a future multiplicity-factoring column would use.
            for tol in tolerances:
                ids = np.unique(sub[(grown <= float(tol)) & (sub > 0)])
                rec[f"n_cells_within@{float(tol):g}"] = int(ids.size)
            # `is_clustered` is deliberately NOT emitted: Tsering's ">=2 touching glia" is
            # fully derivable from `n_cells_within`, and a derived boolean beside the count
            # it comes from is a column that can only ever disagree with it.

        #: ---- somata within a RECRUITMENT distance, added 2026-09-01 ----------------------
        #: **A different question at a different scale, which is why it has its own distances.**
        #: Every other cell-facing column here is contact scale (0.3-1.0 µm) and answers whether
        #: a membrane abuts the plaque. This answers whether cell *bodies* have gathered around
        #: it, which is the recruitment phenotype, and at contact distances it would be zero for
        #: every plaque in every image — a soma is 5-10 µm across, so its centroid is several µm
        #: from anything it touches. `RECRUITMENT_DISTANCES_UM` is (5, 10, 20).
        #:
        #: **C+R, and no multiplicity**, matching `n_cells_within@d` above.
        if soma_centroids is not None and len(soma_centroids):
            pts = np.asarray(soma_centroids, dtype=float)
            far = float(max((float(r) for r in recruitment), default=0.0))
            pad_r = np.ceil(far / spacing).astype(int)
            wide_box = tuple(
                slice(max(0, int(s.start) - int(p)), min(int(dim), int(s.stop) + int(p)))
                for s, p, dim in zip(box, pad_r, labels.shape))
            wide = labels[wide_box] == pid
            grown_r = ndimage.distance_transform_edt(~wide, sampling=tuple(spacing))
            base = np.asarray([s.start for s in wide_box], float)
            local = pts - base
            #: **The window test stays on the rounded index**, so which somata are in range is
            #: unchanged; only the distance is now read sub-voxel. See `sample_distance_um`.
            idx = np.rint(local).astype(int)
            inside = np.ones(idx.shape[0], bool)
            for axis in range(3):
                inside &= (idx[:, axis] >= 0) & (idx[:, axis] < grown_r.shape[axis])
            #: A soma outside the padded window is further than the largest distance asked
            #: about, so it is `inf` rather than clipped to the window's edge — clipping would
            #: report the box face's distance and count somata that are nowhere near. This is
            #: why `sample_distance_um` is called only for the ones already known to be inside:
            #: it clamps, and clamping is exactly the behaviour this line exists to prevent.
            dist = np.full(idx.shape[0], np.inf)
            if inside.any():
                dist[inside] = sample_distance_um(grown_r, local[inside])
            for r in recruitment:
                rec[f"n_somata_within@{float(r):g}"] = int((dist <= float(r)).sum())

        out.append(rec)
    return out


def measure_cd68(
    cell_mask: np.ndarray,
    cd68: np.ndarray,
    *,
    voxel_size_um: Sequence[float],
    threshold: float | None = None,
    factors: Sequence[float] = CURVE_FACTORS,
    soma_mask: np.ndarray | None = None,
) -> Cd68Measures:
    """CD68 load within one cell.

    ``soma_mask`` is optional and drives the soma/process split alone; without it those three
    fields stay `NaN` and every other measure is unaffected. It must be the same shape as
    ``cell_mask`` — the soma region the collapse builds already is.

    ``cd68`` must already be cropped to ``cell_mask``'s shape — this module has no knowledge
    of the pool's bounding boxes, which keeps it usable from a test, the batch driver or an
    interactive session without change.

    ``threshold`` is an **absolute intensity** the user picks once in the semi-auto UI and
    applies unchanged to the batch. Passing none yields `NaN` volumes and status
    ``threshold_unset``; the intensity measures above are complete and primary regardless.

    > **Why this is not a background multiple, and why that was tried and reverted.** On
    > 2026-08-10 the curve was re-anchored to multiples of each image's background median.
    > Measured afterwards: **background median is 0.013-0.048 while CD68 puncta live at
    > 10²-10⁴**, so `20 x background` = 0.45 — four orders of magnitude below anything
    > meaningful. All five levels landed in the noise floor and the curve became ~4x *less*
    > discriminating (a 2-9 point spread against the old 26-point one).
    >
    > The root cause is biology: **CD68 is punctate.** Most of the tissue *and* most of the
    > inside of a cell has ~zero CD68 — the in-cell median is 0.0 in three of six images
    > checked — so a background-referenced threshold separates "any signal" from "exactly
    > zero", which is not a useful question. And unlike plaque, CD68 has **no negative
    > control**: every microglion has some, so nothing anchors the scale. Do not retry it.

    This function took a ``z_offset`` until 2026-08-12, to place a depth measure it also
    computed. Both are gone: the covariate is `morphometrics.measure_cell`'s ``mean_z_um``,
    where it does not depend on a CD68 channel existing. See `Cd68Measures`.
    """
    if cell_mask.shape != cd68.shape:
        raise ValueError(
            f"cd68 shape {cd68.shape} does not match cell mask {cell_mask.shape}"
        )

    out = Cd68Measures(threshold=threshold)
    if not cell_mask.any():
        out.status = "empty_mask"
        return out

    z_um, y_um, x_um = (float(v) for v in voxel_size_um)
    voxel_volume = z_um * y_um * x_um
    values = cd68[cell_mask].astype(np.float64)

    out.integrated_intensity = float(values.sum())
    out.mean_intensity = float(values.mean())

    cell_volume = float(cell_mask.sum()) * voxel_volume

    if threshold is None:
        # Not an error: the intensity measures above are complete and primary. The status
        # records why the volume columns are NaN so that a blank is never read as a zero.
        # No curve either — its levels are multiples of T.
        out.status = "threshold_unset"
        return out

    # The curve brackets the chosen value at f = 2 rather than standing in for a decision.
    for factor in factors:
        level = float(threshold) * float(factor)
        volume = float((values > level).sum()) * voxel_volume
        out.volume_curve_um3[level] = volume
        out.fraction_curve[level] = (volume / cell_volume
                                     if cell_volume > 0 else float("nan"))

    positive = values > float(threshold)
    out.positive_volume_um3 = float(positive.sum()) * voxel_volume
    if cell_volume > 0:
        out.positive_fraction = out.positive_volume_um3 / cell_volume

    #: ---- soma / process split, added 2026-09-01 --------------------------------------
    #: **Where the phagocytic signal sits, not just how much of it there is.** A cell can raise
    #: its CD68 by filling its soma with lysosomes or by carrying them out along its processes,
    #: and `positive_volume_um3` alone cannot tell those apart. The boundary is the soma mask
    #: the collapse already builds, so this introduces no new definition and no second erosion —
    #: it inherits `soma_erode_um` exactly as `soma_volume_um3` does.
    #:
    #: The two volumes sum to `positive_volume_um3` by construction, because "processes" is
    #: defined as the complement inside the cell rather than measured separately. That is what
    #: makes the pair auditable from the file.
    if soma_mask is not None:
        soma_bool = np.asarray(soma_mask).astype(bool)
        if soma_bool.shape == cell_mask.shape:
            in_soma = soma_bool[cell_mask]
            out.in_soma_um3 = float((positive & in_soma).sum()) * voxel_volume
            out.in_processes_um3 = float((positive & ~in_soma).sum()) * voxel_volume
            #: Denominator is the **cell's total CD68 volume**, the user's choice 2026-09-01 —
            #: so this reads as "what share of the phagocytic load is in the cell body",
            #: independent of how big the soma happens to be.
            out.soma_fraction = (out.in_soma_um3 / out.positive_volume_um3
                                 if out.positive_volume_um3 > 0 else float("nan"))
    return out


def sample_distance_um(field: np.ndarray, points_vox: np.ndarray) -> np.ndarray:
    """Read a µm distance field at **sub-voxel** positions. `points_vox` is `(N, 3)` → `(N,)`.

    > ### ⇒ WHY NOT `rint`, WHICH IS WHAT THIS REPLACED — 2026-09-02
    >
    > Three columns read a distance field at a soma centroid, and all three rounded the centroid
    > to the nearest voxel first. A half-voxel of rounding is **±0.30 µm in z against ±0.054 µm
    > in plane** at this study's geometry — the same 5.54:1 asymmetry that made
    > `morphometrics.detached_skeleton_um` ask its question on the wrong grid. A soma centroid is
    > a **continuous** quantity, computed as a centre of mass; quantising it to the coarse axis
    > throws away precision the value actually has.
    >
    > Trilinear interpolation (`map_coordinates`, `order=1`) is the right tool for a Euclidean
    > distance transform: the field is **locally linear** wherever it is not near the background's
    > medial axis, so interpolation is near-exact over the range that matters, and it is never
    > worse than half-voxel rounding.
    >
    > **This changes values, and it should.** Numbers move by up to a third of a micron in z on
    > every affected column. Rounding was symmetric, so the old error was noise rather than bias
    > — nothing computed from it is *wrong* in direction, it was simply less precise than the
    > inputs allowed.

    Positions are clamped into the array, so a caller that has already decided a point is
    in-range gets the same in-range answer. **A caller that must distinguish "outside" from
    "at the edge" has to test that itself before calling** — `measure_plaques` does, because a
    soma beyond its padded window must read `inf` rather than the window face's distance.
    """
    f = np.asarray(field, float)
    p = np.asarray(points_vox, float).reshape(-1, 3)
    coords = np.stack([np.clip(p[:, a], 0.0, f.shape[a] - 1) for a in range(3)])
    return np.asarray(ndimage.map_coordinates(f, coords, order=1, mode="nearest"), float)


def measure_plaque_proximity(
    cell_mask: np.ndarray,
    distance_field: np.ndarray | None,
    *,
    soma_centroid: Sequence[float] | None = None,
    field_status: str = "ok",
) -> PlaqueProximity:
    """How close one cell comes to amyloid, from its soma and from its nearest voxel.

    ``distance_field`` is the crop of the whole-image field from
    :func:`build_plaque_distance_field`, so "nearest plaque" means nearest in the *image*,
    not merely within this cell's bounding box.

    A cell in tissue with no plaques reports ``NaN``, never 0. Zero would be
    indistinguishable from a cell sitting directly on a plaque, which is the opposite of the
    truth — and the WT images are entirely this case.
    """
    out = PlaqueProximity()
    if distance_field is None:
        out.status = field_status if field_status != "ok" else "no_plaques_found"
        return out
    if cell_mask.shape != distance_field.shape:
        raise ValueError(
            f"distance field shape {distance_field.shape} does not match "
            f"cell mask {cell_mask.shape}"
        )
    if not cell_mask.any():
        out.status = "empty_mask"
        return out

    out.nearest_voxel_to_plaque_um = float(distance_field[cell_mask].min())

    if soma_centroid is None:
        out.status = "no_soma"
        return out

    #: **The in-crop test still rounds; only the value is interpolated.** Keeping the test on the
    #: rounded index means no cell changes from carrying a distance to `soma_outside_crop` or
    #: back — this makes the number more precise without moving the population.
    z, y, x = (int(round(float(c))) for c in soma_centroid)
    if not (0 <= z < cell_mask.shape[0] and 0 <= y < cell_mask.shape[1]
            and 0 <= x < cell_mask.shape[2]):
        out.status = "soma_outside_crop"
        return out
    out.soma_to_plaque_um = float(sample_distance_um(distance_field, soma_centroid)[0])
    return out
