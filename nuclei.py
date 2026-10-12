"""Nucleus detection.

Nuclei are the sole authority on cell separation, so this module carries more
consequence than its size suggests. Two failure directions, and they are opposite:

* **Under-merge** — one nucleus spanning fifteen z-slices detected as fifteen separate
  nuclei. Since splitting triggers on two or more nuclei, a single microglia is then
  shredded into fragments.
* **Over-merge** — nuclei stacked in z fused into one. Clumped cells are then never
  split, which is the failure the nuclei channel exists to prevent.

PrestoCell max-projects blocks of z and runs 2D, which fuses anything stacked in depth
and silently loses splits. Here detection runs **per slice with IoU stitching across
z**, which trades that for the opposite risk — hence ``stitch_threshold`` being a
parameter to validate rather than a default to accept. Hand-marked somata give a hard
ceiling: nuclei found inside cells should not exceed the number of cells present.

The detector is injectable so that everything except the model itself — the plaque
veto, size filtering, relabelling, diameter conversion — is testable without loading
Cellpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy import ndimage

from . import cpu
from .resolution import downsample_xy, voxel_size_after_downsample

__all__ = [
    "normalize_for_detection",
    "subtract_background",
    "HALF_RES_SIGMA_PX",
    "NucleiResult",
    "Detector",
    "CellposeDetector",
    "detect_nuclei",
    "gpu_available",
    "resolve_device",
    "describe_device",
    "veto_plaque_signal",
    "veto_mask_at_margin",
    "diameter_in_pixels",
]


def gpu_available() -> bool:
    """Whether CUDA is usable in this process. **Never raises.**

    torch is an optional dependency and so is cellpose — a machine with neither must still
    import this module and run every function in it that does not detect nuclei. An
    unimportable or misbuilt torch is therefore *no GPU*, not an error.
    """
    try:
        import torch
    except Exception:                              # noqa: BLE001 — optional, see above
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:                              # noqa: BLE001 — a broken driver is no GPU
        return False


def resolve_device(gpu: bool | None = None) -> bool:
    """Turn the three-valued request into the two-valued answer the detector needs.

    ``None`` means *use a GPU if there is one*; ``True`` and ``False`` force it either way.

    > ### ⇒ THE ANSWER MUST NOT REACH `SegmentationParams`, AND THAT IS WHY IT LIVES HERE
    >
    > That dataclass is hashed, the hash names the pool, and `phase_a_hash` decides which pool
    > a cell belongs in — so a device field would put the CPU half of a study in one pool and
    > the GPU half in another. **The user agreed explicitly.** The device is provenance, not a
    > parameter: it is recorded in `run_id`, which is free text and stored per file, and
    > printed at startup.

    **Forcing `True` on a machine with no CUDA is honoured, not corrected.** Cellpose raises
    its own error there, which names the real problem; silently falling back would make
    `--device gpu` a request the program can ignore, and the whole point of the override is
    to find out whether the GPU is being used.
    """
    if gpu is None:
        return gpu_available()
    return bool(gpu)


def describe_device(gpu: bool) -> str:
    return "cuda" if gpu else "cpu"


class Detector(Protocol):
    """Anything that turns a ``(z, y, x)`` volume into integer nucleus labels."""

    def __call__(self, volume: np.ndarray, *, diameter: float,
                 stitch_threshold: float) -> np.ndarray: ...


@dataclass
class NucleiResult:
    """Detected nuclei at *segmentation* resolution, plus how they were obtained."""

    labels: np.ndarray                 # (z, y, x) int32; 0 = background
    n_nuclei: int
    diameter_px: float
    stitch_threshold: float
    downsample: int
    plaque_vetoed_fraction: float = 0.0
    n_removed_small: int = 0

    def summary(self) -> str:
        veto = (f", {self.plaque_vetoed_fraction:.1%} of DAPI vetoed by plaque signal"
                if self.plaque_vetoed_fraction else "")
        dropped = (f", {self.n_removed_small} below size cutoff"
                   if self.n_removed_small else "")
        return (f"{self.n_nuclei} nuclei at diameter {self.diameter_px:.0f}px, "
                f"stitch {self.stitch_threshold:g}{veto}{dropped}")


def normalize_for_detection(
    volume: np.ndarray,
    *,
    method: str = "log",
    low_percentile: float = 1.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """Rescale intensities so Cellpose can see the nuclear body.

    Nuclei here have very bright **nucleoli** — small dots an order of magnitude
    above the surrounding nuclear body. Cellpose's own percentile normalisation is
    therefore dominated by those dots, compressing the body itself into almost no
    contrast. Measured on one slice containing ~20-25 visible nuclei, the default
    found **5**; log compression followed by a gentle clip found **24**.

    ``log`` is the default rather than an aggressive percentile clip because it does
    not depend on how much bright signal a particular image happens to contain. A
    fixed clip at, say, the 90th percentile discards a different amount of real signal
    in every image, which is exactly the kind of per-image variation that breaks a
    batch of 700.
    """
    data = np.asarray(volume, dtype=np.float32)
    if method == "log":
        data = np.log1p(np.clip(data, 0, None))
    elif method == "sqrt":
        data = np.sqrt(np.clip(data, 0, None))
    elif method != "percentile":
        raise ValueError(
            f"unknown method {method!r}; expected 'log', 'sqrt' or 'percentile'"
        )
    lo, hi = np.percentile(data, [low_percentile, high_percentile])
    return np.clip((data - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)


#: Above this sigma **in pixels** the background estimate is computed on a half-resolution
#: copy and interpolated back. Below it the exact filter is used.
#:
#: **Measured on `program training crop2.tif`, 2026-08-23**, comparing the half-resolution
#: estimate against the exact one over the whole 40 x 427 x 422 DAPI volume. The error is
#: quoted as a share of DAPI's own 95th percentile, and it is the *estimate* that is compared,
#: so the error reaching a filtered voxel is this times ``strength``:
#:
#: | sigma px | exact | half-res | speedup | worst voxel |
#: |---|---|---|---|---|
#: | 32 | 1.15 s | 0.50 s | 2.3x | 9.04% |
#: | 48 | 1.82 s | 0.57 s | 3.2x | 4.22% |
#: | **64** | **2.39 s** | **0.63 s** | **3.8x** | **2.54%** |
#: | 96 | 3.66 s | 0.78 s | 4.7x | 1.52% |
#:
#: **Below 32 px it is both wrong and slower** — 201% at 4.6 px, where the half-resolution
#: grid no longer samples the kernel at all and the exact filter is already cheap. The cut is
#: at 64 rather than 32 because the useful range of this control starts well above it: the
#: scale has to exceed one nucleus diameter or the subtraction eats the nuclei, which at
#: 7.6 µm nuclei and 0.1083 µm pixels is 70 px before the control does anything worth doing.
#:
#: > ### WHY THIS EXISTS AT ALL — the recorded cost was measured at the wrong sigma
#: >
#: > `SHAKEOUT_FINDINGS.md` records the Gaussian as *"0.65 s / ~19 s a field"*. scipy's kernel
#: > radius is ``4 * sigma``, so cost is **linear in sigma**, and that figure cannot have been
#: > taken at the 12 µm scale the screen defaults to. Re-measured on the same crop: 1.33 s at
#: > 1.3 µm, **10.5 s at 12 µm**, which extrapolates to **~310 s** on a 40 x 2304 x 2304 field.
#: > At half resolution the same field is **~67 s**. The screen that owns this parameter sits
#: > in front of every other screen in the flow, so a five-minute commit there is paid before
#: > anything has been decided.
#:
#: **The gate is on sigma in PIXELS, not microns**, because that is what the accuracy actually
#: depends on — the same micron scale on a coarser voxel is a smaller kernel. Two images with
#: different voxel sizes can therefore take different paths at one `dapi_background_scale_um`;
#: in this study the voxel size is constant, and the path is a function of the parameters
#: rather than of the image, so the same settings always give the same answer.
HALF_RES_SIGMA_PX = 64.0


def _background_estimate(data: np.ndarray, sigma_y: float, sigma_x: float) -> np.ndarray:
    """The blurred copy `subtract_background` takes away. See `HALF_RES_SIGMA_PX`.

    **Padded to an even size before halving, then cropped back**, so the interpolation factor
    is exactly 2.0 in both axes and the round-trip cannot come back one pixel short on an odd
    dimension. `mode="edge"` rather than zeros: this is a background estimate, and a zero
    column at the border would drag the estimate down exactly where `subtract_background`'s
    own ordering rule says a Gaussian must never meet a hole.
    """
    if min(sigma_y, sigma_x) < HALF_RES_SIGMA_PX:
        return ndimage.gaussian_filter(data, sigma=(0.0, sigma_y, sigma_x))

    z, y, x = data.shape
    padded = np.pad(data, ((0, 0), (0, y % 2), (0, x % 2)), mode="edge")
    _, ey, ex = padded.shape
    small = padded.reshape(z, ey // 2, 2, ex // 2, 2).mean(axis=(2, 4))
    blurred = ndimage.gaussian_filter(small, sigma=(0.0, sigma_y / 2.0, sigma_x / 2.0))
    return ndimage.zoom(blurred, (1.0, 2.0, 2.0), order=1)[:, :y, :x]


def subtract_background(
    volume: np.ndarray,
    *,
    sigma_um: float,
    voxel_size_um: Sequence[float],
    strength: float = 1.0,
    threads: int | None = None,
) -> np.ndarray:
    """Remove broad-scale background from a channel: ``v - k * blur(v)``, clipped at zero.

    **Why this exists.** Detection runs on `normalize_for_detection`'s log compression, which
    lifts the dim end hard: measured on the demo crop, DAPI at its 90th percentile renders at
    **2.9% grey** on screen and is **70.8%** to the detector. Real diffuse background is
    therefore bright to Cellpose while being invisible to the person judging the result.

    > ### THE ESTIMATOR IS A GAUSSIAN HIGH-PASS RATHER THAN A TOP-HAT, AND IT IS A COST DECISION
    >
    > A white top-hat is the better-behaved operator and was measured first. On the demo crop at
    > a 5 µm radius it took **18.6 s**, which extrapolates to **~546 s on one 40 x 1152 x 1152
    > field** — nine minutes per image before any tuning, on a study of ~700. Subtracting a
    > blurred copy is **0.65 s** on the same data, ~19 s a field, and is separable.
    >
    > **The top-hat's advantage is real and is bought back by ordering instead.** Near a region
    > of zeros the Gaussian estimate is dragged down and leaves a bright collar — measured at
    > **+189 max** against a background 95th percentile of 290, while the top-hat moved by
    > **+0.11**. That is why this must run **before the plaque veto**, which is the only step
    > that puts zeros into DAPI. Filtered first, there are no holes for it to ring around.

    > ### ⇒ ``strength`` IS THE CONTROL AND ``sigma_um`` IS NOT — measured, 2026-08-22
    >
    > Sweeping sigma on the demo crop gives **a cliff and then a plateau**: at 2 µm the nucleus
    > cores keep 7% of their signal, at 8 µm they keep 63% and the background 95th percentile is
    > **already entirely gone**, and nothing further happens out to 80 µm. Below one nucleus
    > diameter it destroys the nuclei; at or above it there is nothing left to remove. *"How
    > many microns"* is therefore a switch with a trap on its left, which is the wrong question
    > to put to a user.
    >
    > **Fix the scale and vary how much of the estimate is taken away** and the control becomes
    > smooth, monotone and bounded at both ends, with both ends meaning something — at sigma
    > 12 µm the nucleus-to-background contrast runs 5.8x at ``k = 0``, 14.6x at 0.4, **58x at
    > 0.6**, and above 0.8 the background is gone and only the nuclei are still being eaten.
    > That is the same shape as `cell_coefficient`. **Both this and the scale are number boxes
    > on the screen** — the strength was a slider until 2026-08-23, when the user made the
    > screen's derived layers appear only after COMMIT, which left nothing for a live control
    > to preview.

    **``sigma_um`` is a scale.** It must be comfortably larger than a nucleus or the subtraction
    eats the nucleus along with the background; at a 7.6 µm nucleus that means tens of microns,
    not single digits. **No default is offered here** — the caller's default is ``0.0``, off.

    **Both must be positive or the filter is off, and the ARRAY ITSELF comes back.** The screen
    holds the scale at 12 µm and puts ``strength`` at 0 to mean *off*, so a zero strength at a
    live scale has to be exactly as free as a zero scale — identity, not a copy and not a blur.

    **xy only, and z is left alone deliberately.** The stack is ~5.5:1 anisotropic, so a sigma
    that is right in plane spans several slices in z and would blend nuclei that are merely
    stacked. It also makes the operation exactly per-plane, which is what lets a preview filter
    the one slice on screen and get the same answer as filtering the whole volume — true of the
    half-resolution path too, because both the block mean and the interpolation act within a
    plane and neither reaches across z.

    > ### ⇒ AND *EXACTLY PER-PLANE* IS WHY IT IS THREADED BY z — `HANDOFF_PATCH_CPU.md` §3 A5
    >
    > The sentence above is not decoration, it is the licence. `gaussian_filter` is given
    > ``sigma=(0, ...)``, the half-resolution path reshapes within each plane, and
    > ``ndimage.zoom`` is given a z factor of exactly 1.0 — so no output plane can see another
    > input plane, and splitting the volume into contiguous z slabs cannot change one voxel.
    > Verified against a control that must differ; see `tests/test_nuclei.py`.
    >
    > ``threads`` is the count for this call. `None` takes the process count, which is 1 unless
    > a scheduler or `--threads` said otherwise. See `cpu.count`.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
    if sigma_um <= 0 or strength <= 0:
        return volume
    _, y_um, x_um = (float(v) for v in voxel_size_um)
    sigma_y, sigma_x = float(sigma_um) / y_um, float(sigma_um) / x_um
    take = float(strength)
    data = np.asarray(volume, dtype=np.float32)
    out = np.empty_like(data)

    #: Each worker owns a contiguous span of z and writes only into its own planes, so no two
    #: touch the same voxel and nothing is collected and joined afterwards. The blurred copy is
    #: now per-slab rather than whole-volume, which lowers the peak as well.
    def work(lo: int, hi: int) -> None:
        blurred = _background_estimate(data[lo:hi], sigma_y, sigma_x)
        np.clip(data[lo:hi] - take * blurred, 0.0, None, out=out[lo:hi])

    cpu.run_slabs(work, data.shape[0], cpu.count(threads))
    return out


def diameter_in_pixels(
    diameter_um: float,
    voxel_size_um: Sequence[float],
    downsample: int = 1,
) -> float:
    """Convert a physical nucleus diameter into pixels at segmentation resolution.

    Cellpose operates around 30 px and rescales internally by ``diameter/30``. At
    native resolution these nuclei are 55-74 px, so downsampling by 2 lands them near
    the model's own working size and saves it the rescale.
    """
    _, y_um, x_um = voxel_size_after_downsample(voxel_size_um, downsample)
    return float(diameter_um / ((y_um + x_um) / 2.0))


def veto_mask_at_margin(
    plaque_mask: np.ndarray,
    *,
    dilate_um: float = 0.0,
    voxel_size_um: Sequence[float] | None = None,
) -> np.ndarray:
    """The veto **as it is actually applied** — the thresholded plaque mask, widened or shrunk.

    > ### IT WAS COMPUTED HERE AND NEVER RETURNED, AND THAT WAS ITEM 2'S BUG
    >
    > This lived inside `veto_plaque_signal`, which returns only the vetoed volume and a
    > fraction. So `app.PipelineCompute.veto` handed item 2 the **undilated** mask, and the
    > screen derived all three of the things the user judges from it — the 6E10 outline, the
    > removed-DAPI overlay and the `DAPI removed by the veto` readout.
    >
    > **Every one of them sat still while `plaque_dilate_um` moved**, on the one screen whose
    > entire purpose is choosing that value by eye. The coefficient appeared to work only
    > because it changes the mask upstream, through `threshold_channel`.
    >
    > Detection was never affected: `detect_nuclei` re-applies the veto with its own
    > `plaque_dilate_um`. **This was a display defect, not a measurement one** — which is why
    > `products[Step.VETO]` must keep receiving the undilated mask, or the dilation lands twice.

    > ### NEGATIVE ERODES, AND IT IS WHAT THE SCREEN'S NAME ASKS FOR — 2026-08-20
    >
    > *Plaque DAPI **Core** Veto.* The 6E10 mask covers a whole plaque; the DAPI-positive
    > material is the **dense core inside it**, and microglia cluster in the halo around that
    > core — precisely the nuclei the veto must not eat. Eroding shrinks the mask toward the
    > core, so the veto keeps killing the false nucleus at the centre and stops reaching the
    > real ones around it. The user's framing: *"the 6e10 mask is being eroded to capture the
    > dense DAPI core within."*
    >
    > **Measured on `5xFAD/1234/program training crop2.tif`, and the gradient is real.** Mean
    > DAPI by depth inside the mask: `138` at the rim, then `266`, `425`, `646` at 1-2 um deep —
    > a 4.7x rise. 6E10's own intensity is flat across the same bins (`1546 / 2301 / 2247 /
    > 2046`), so the gradient belongs to DAPI rather than to brightness in general. DAPI
    > captured per veto voxel climbs `214 -> 319 -> 474 -> 624` as erosion goes `0 -> 0.25 ->
    > 0.5 -> 1.0 um`.
    >
    > **Small components vanish, and that is the behaviour rather than a defect.** At 0.5 um
    > only 9 of 326 components survive. A speck too small to survive erosion has no dense core,
    > so it is not a cored plaque and there is nothing there to veto.

    > ### SPHERICAL IN MICRONS — REPLACED THE xy-ONLY ELEMENT 2026-08-20
    >
    > **It used to be `disk[np.newaxis]`: one slice thick, so the margin reached its stated
    > distance in xy and ZERO distance in z at any value.** That was deliberate — an element
    > sized isotropically *in voxels* reaches 5.54x further in z than in plane at this study's
    > anisotropy — but the cure was worse than the disease. Plaque-associated DAPI one slice
    > away survived the veto, was detected as a nucleus, and then **joined the microglial set
    > whenever `membership_threshold` was lowered to admit genuine microglial nuclei.** So the
    > workaround for the false positive cost real cells.
    >
    > **A distance transform removes the element entirely.** `sampling=voxel_size_um` measures
    > true physical distance, so the margin is a real sphere in microns: exact at any radius, no
    > structuring element to build, no rounding, and the anisotropy handled natively rather than
    > compensated for.
    >
    > **Measured on a single voxel at (0.6, 0.1083, 0.1083):** 0.108 and 0.3 and 0.5 um reach
    > the centre slice only; **0.6 um reaches +/-1**; 1.2 um reaches +/-2. z is therefore coarsely
    > quantised — everything below 0.6 behaves identically in z — and **that is accepted**: it is
    > the slice spacing, not a choice. The user's own model, confirmed: *"if its going 1 voxel in
    > y so 0.108um then it should go up by 0.108 as well, so in this case it would not show up on
    > the next slice"*, and *"if it was 1.2um then it would show up two slices up as well."*
    >
    > **A 0.5 um margin still does not reach the next slice, and that is correct** — the next
    > slice centre is 0.6 um away and a 0.5 um sphere does not touch it.

    **The radius carries a one-voxel floor IN PLANE ONLY.** Any nonzero margin moves at least one
    in-plane voxel, which is what the old `max(1, int(round(...)))` did. **There is deliberately
    no z floor**: forcing 0.108 um to reach a slice 0.6 um away would reintroduce the exact
    anisotropy bug this replaced.

    **The border behaviour is preserved and was verified rather than assumed.** `distance_
    transform_edt` measures to the nearest background voxel *inside the array*, so a plaque
    running off the edge of the field keeps its border column and loses only its true interior
    edge — which is what `border_value=1` bought on the old erosion. Dilation needs no such care:
    it only ever grows.

    **Cost: the transform returns `float64`**, ~1.7 GB on a full 40 x 2304 x 2304 field. Accepted
    — this runs on a cluster. If it ever needs to shrink, run it inside the mask's bounding box
    expanded by the margin and OR the result back; that is exact rather than approximate.
    """
    veto = plaque_mask.astype(bool, copy=False)
    if not dilate_um:
        return veto
    if voxel_size_um is None:
        raise ValueError("dilate_um requires voxel_size_um")
    spacing = tuple(float(v) for v in voxel_size_um)
    radius = max(abs(float(dilate_um)), min(spacing[1], spacing[2]))
    if dilate_um > 0:
        return ndimage.distance_transform_edt(~veto, sampling=spacing) <= radius
    return ndimage.distance_transform_edt(veto, sampling=spacing) > radius


def veto_plaque_signal(
    dapi: np.ndarray,
    plaque_mask: np.ndarray,
    *,
    dilate_um: float = 0.0,
    voxel_size_um: Sequence[float] | None = None,
) -> tuple[np.ndarray, float]:
    """Suppress DAPI wherever plaque signal sits, before detection runs.

    Dense-core amyloid plaques are DAPI-positive. A plaque read as a nucleus would
    manufacture a spurious split — and would do so precisely where microglia cluster
    most tightly, so the error concentrates exactly where accuracy matters. Using the
    6E10 channel to veto is an independent measurement rather than a shape heuristic,
    so it removes those false positives without endangering genuine nuclei that happen
    to look unusual.

    ``dilate_um`` is a **signed margin**. Positive widens the veto, since plaque-associated
    DAPI often extends a little beyond the antibody signal; negative erodes it toward the dense
    core, which is where the DAPI-positive material actually sits. Applied in xy only and sized
    in microns: an isotropic element in voxels would reach ~5.5x further in z than in plane.
    See `veto_mask_at_margin`.

    Returns the vetoed volume and the fraction of total DAPI intensity removed.
    """
    if dapi.shape != plaque_mask.shape:
        raise ValueError(
            f"dapi shape {dapi.shape} does not match plaque mask {plaque_mask.shape}"
        )

    veto = veto_mask_at_margin(plaque_mask, dilate_um=dilate_um, voxel_size_um=voxel_size_um)

    total = float(dapi.sum())
    removed = float(dapi[veto].sum())
    out = dapi.copy()
    out[veto] = 0
    return out, (removed / total if total else 0.0)


class CellposeDetector:
    """Cellpose behind a minimal interface, so version churn stays contained.

    Targets the v3 ``nuclei`` model: purpose-trained for this task, faster than the
    v4 generalist, and it exposes an explicit diameter — which is worth having when
    the nucleus size is known precisely. v4 can be substituted by writing another
    :class:`Detector`; both are available on the cluster, so the two can be compared.
    """

    def __init__(self, model_type: str = "nuclei", gpu: bool | None = None) -> None:
        """``gpu=None`` auto-detects; ``True`` and ``False`` force the choice.

        > ### ⇒ IT DEFAULTED TO `False` AND NOTHING ANYWHERE PASSED `True`
        >
        > Detection **is** the expensive step of Phase A, and it was the one step running on
        > the CPU on a GPU partition the run was already paying for. `detect_nuclei` built this
        > bare, so there was no route to a GPU even for a caller who wanted one.
        >
        > **`None` rather than `resolve_device()` as a default argument**, so the detection
        > happens when the detector is constructed rather than when this module is imported —
        > a default argument is evaluated once, at import, which on a login node would pin
        > every later run in the process to CPU.
        """
        self.model_type = model_type
        self.gpu = resolve_device(gpu)
        self._model = None

    @property
    def device(self) -> str:
        """`cuda` or `cpu` — what to put in front of a person. See `describe_device`."""
        return describe_device(self.gpu)

    def _get_model(self):
        if self._model is None:
            from cellpose import models          # imported late: heavy, and optional
            self._model = models.Cellpose(gpu=self.gpu, model_type=self.model_type)
        return self._model

    def __call__(self, volume: np.ndarray, *, diameter: float,
                 stitch_threshold: float) -> np.ndarray:
        result = self._get_model().eval(
            volume,
            channels=[0, 0],            # single greyscale channel
            diameter=diameter,
            do_3D=False,                # 2D per slice...
            stitch_threshold=stitch_threshold,   # ...then merged across z by IoU
            # Intensities are already conditioned by normalize_for_detection; letting
            # Cellpose renormalise would reinstate the nucleoli-dominated scaling that
            # this pipeline exists to avoid.
            normalize=False,
        )
        return np.asarray(result[0], dtype=np.int32)


def _drop_small(labels: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int]:
    if min_voxels <= 0 or labels.max() == 0:
        return labels, 0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    doomed = np.flatnonzero((counts > 0) & (counts < min_voxels))
    if doomed.size == 0:
        return labels, 0
    labels = np.where(np.isin(labels, doomed), 0, labels)
    return labels, int(doomed.size)


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    present = np.unique(labels)
    present = present[present != 0]
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lookup[labels]


def detect_nuclei(
    dapi: np.ndarray,
    *,
    diameter_px: float,
    detector: Detector | None = None,
    stitch_threshold: float = 0.25,
    downsample: int = 2,
    plaque_mask: np.ndarray | None = None,
    plaque_dilate_um: float = 0.0,
    voxel_size_um: Sequence[float] | None = None,
    min_volume_voxels: int = 0,
    normalize: str | None = "log",
    background_sigma_um: float = 0.0,
    background_strength: float = 1.0,
    gpu: bool | None = None,
) -> NucleiResult:
    """Detect nuclei, returning labels at segmentation resolution.

    Args:
        dapi: nuclei channel at **native** resolution, ``(z, y, x)``.
        diameter_px: expected nucleus diameter in pixels *after* downsampling — see
            :func:`diameter_in_pixels`.
        detector: injectable for testing; defaults to Cellpose.
        stitch_threshold: IoU required to merge labels between adjacent slices. The
            parameter most worth validating: too low shreds one nucleus into many,
            too high fuses nuclei stacked in z.
        plaque_mask: 6E10 foreground at native resolution; suppressed before detection.
        min_volume_voxels: drop detections smaller than this, in downsampled voxels.
        background_sigma_um: remove background at scales broader than this, in microns,
            **before** the veto and before downsampling. ``0.0`` disables it, which is what
            every existing caller gets. See :func:`subtract_background` for why it must come
            first and why the estimator is a Gaussian rather than a top-hat.
        background_strength: how much of that estimate is taken away. ``1`` subtracts all of it
            and **there is no upper bound** — the screen takes it in a number box, which is
            what ended a ceiling that moved three times in one sitting. **This is
            the control the user actually moves**, and ``0`` disables the filter just as a zero
            sigma does — `SegmentationParams` holds the scale at its default and turns the
            filter on and off with this. The default here is ``1.0`` so that a caller passing
            only a sigma gets the full subtraction it asked for.
    """
    if dapi.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {dapi.shape}")

    #: **BEFORE the veto, and that ordering is load-bearing** — see `subtract_background`. The
    #: veto is the only step that writes zeros into DAPI, and a Gaussian background estimate
    #: rings around a zero region: measured at +189 against a background 95th percentile of 290.
    #: Filtering first means there are no holes to ring around. Default 0.0 = untouched, so
    #: every existing call is byte-for-byte what it was.
    if background_sigma_um > 0 and background_strength > 0:
        if voxel_size_um is None:
            raise ValueError("background_sigma_um requires voxel_size_um")
        dapi = subtract_background(dapi, sigma_um=background_sigma_um,
                                   voxel_size_um=voxel_size_um,
                                   strength=background_strength)

    vetoed_fraction = 0.0
    if plaque_mask is not None:
        dapi, vetoed_fraction = veto_plaque_signal(
            dapi, plaque_mask,
            dilate_um=plaque_dilate_um,
            voxel_size_um=voxel_size_um,
        )

    small = downsample_xy(dapi, downsample)
    if normalize:
        small = normalize_for_detection(small, method=normalize)

    if detector is None:
        #: **`gpu` is forwarded, not defaulted here.** `None` reaches `resolve_device` and
        #: becomes *use one if there is one*; an injected detector is left entirely alone,
        #: which is what keeps every test in this module free of a device question.
        detector = CellposeDetector(gpu=gpu)
    labels = np.asarray(
        detector(small, diameter=diameter_px, stitch_threshold=stitch_threshold),
        dtype=np.int32,
    )
    if labels.shape != small.shape:
        raise ValueError(
            f"detector returned {labels.shape}, expected {small.shape}"
        )

    labels, n_removed = _drop_small(labels, min_volume_voxels)
    labels = _relabel_consecutive(labels)

    return NucleiResult(
        labels=labels,
        n_nuclei=int(labels.max()),
        diameter_px=float(diameter_px),
        stitch_threshold=float(stitch_threshold),
        downsample=downsample,
        plaque_vetoed_fraction=vetoed_fraction,
        n_removed_small=n_removed,
    )
