"""Loading and validating multi-channel microscopy stacks.

Handles OME-TIFF (production), ImageJ hyperstacks (the test crops) and plain TIFF,
preferring embedded metadata and falling back to explicit parameters where a file
does not describe itself.

Three things this module insists on, each because of a specific failure:

**Channels are loaded lazily, one role at a time.** Phase A needs only the nuclei and
cell channels; CD68 and the plaque marker are not read until Phase C. At 32-bit and
full field size that is the difference between a 1.7 GB and a 3.4 GB working set.

**Channel roles are verified against the data, not merely declared.** CD68 is
lysosomal, so it must lie inside the Iba1 cytoplasm. Measuring that containment
detects a mislabelled or re-ordered file — which is not hypothetical: the first test
images arrived with channels 1 and 2 described the wrong way round, and the declared
order would have segmented CD68 puncta as microglia, producing plausible-looking
rather than obviously broken output.

**Voxel size is read and checked.** All morphometry is physical, and the anisotropy
here is ~5.5:1, so a wrong or assumed voxel size silently corrupts every measurement
downstream rather than raising an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile

__all__ = [
    "ChannelRoles",
    "ImageMeta",
    "ImageSource",
    "RoleCheck",
    "ImageFormatError",
    "VoxelSizeError",
    "REQUIRED_ROLES",
    "OPTIONAL_ROLES",
    "VALID_ROLES",
]

REQUIRED_ROLES = ("nuclei", "cell")
OPTIONAL_ROLES = ("cd68", "plaque")
VALID_ROLES = REQUIRED_ROLES + OPTIONAL_ROLES

_SPATIAL = ("Z", "Y", "X")


class ImageFormatError(RuntimeError):
    """The file's layout could not be understood, or contradicts what was declared."""


class VoxelSizeError(RuntimeError):
    """Voxel size is missing, or disagrees with what the caller expected."""


# --------------------------------------------------------------------------
# Channel roles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelRoles:
    """Which channel index carries which stain.

    Never positional: PrestoCell hardcodes ``data[:, 0]`` and ``data[:, 1]``, and
    applies its one configurable swap twice so that it cancels out
    (``main.py`` lines 2159-2163 and 2188-2191).
    """

    nuclei: int
    cell: int
    cd68: int | None = None
    plaque: int | None = None

    def __post_init__(self) -> None:
        declared = self.as_dict()
        if any(v < 0 for v in declared.values()):
            raise ValueError(f"channel indices must be non-negative, got {declared}")
        if len(set(declared.values())) != len(declared):
            raise ValueError(f"two roles map to the same channel index: {declared}")

    def as_dict(self) -> dict[str, int]:
        return {r: v for r in VALID_ROLES if (v := getattr(self, r)) is not None}

    def index(self, role: str) -> int:
        if role not in VALID_ROLES:
            raise KeyError(f"unknown role {role!r}; valid roles are {list(VALID_ROLES)}")
        value = getattr(self, role)
        if value is None:
            raise KeyError(f"role {role!r} is not present in this image")
        return value

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> "ChannelRoles":
        unknown = set(mapping) - set(VALID_ROLES)
        if unknown:
            raise ValueError(f"unknown role(s) {sorted(unknown)}")
        missing = [r for r in REQUIRED_ROLES if r not in mapping]
        if missing:
            raise ValueError(f"missing required role(s) {missing}")
        return cls(**{r: mapping.get(r) for r in VALID_ROLES})


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageMeta:
    """What the file says about itself."""

    path: Path
    axes: str                                   # e.g. "ZCYX"
    shape: tuple[int, ...]                      # matching `axes`
    dtype: np.dtype
    n_z: int
    n_channels: int
    yx_shape: tuple[int, int]
    voxel_size_um: tuple[float, float, float] | None    # (z, y, x)
    source_format: str                          # "ome" | "imagej" | "plain"

    @property
    def anisotropy(self) -> float | None:
        if self.voxel_size_um is None:
            return None
        z, _, x = self.voxel_size_um
        return z / x if x else None

    def describe(self) -> str:
        vox = ("unrecorded" if self.voxel_size_um is None
               else "z={:g} y={:g} x={:g} um".format(*self.voxel_size_um))
        return (f"{self.path.name}: {self.source_format}, {self.axes}{self.shape}, "
                f"{self.dtype}, voxel {vox}")


def _imagej_voxel_size(tif: tifffile.TiffFile) -> tuple[float, float, float] | None:
    meta = tif.imagej_metadata or {}
    page = tif.pages[0]
    try:
        xres = page.tags["XResolution"].value
        yres = page.tags["YResolution"].value
        x = xres[1] / xres[0]
        y = yres[1] / yres[0]
    except (KeyError, ZeroDivisionError, TypeError, IndexError):
        return None
    z = float(meta.get("spacing", 0.0))
    return (z, float(y), float(x)) if z > 0 else None


def _ome_voxel_size(tif: tifffile.TiffFile) -> tuple[float, float, float] | None:
    xml = tif.ome_metadata or ""
    found: dict[str, float] = {}
    for axis in ("X", "Y", "Z"):
        m = re.search(rf'PhysicalSize{axis}="([0-9.eE+-]+)"', xml)
        if m:
            found[axis.lower()] = float(m.group(1))
    if {"x", "y", "z"} <= found.keys():
        return (found["z"], found["y"], found["x"])
    return None


def _read_meta(path: Path, axes_override: str | None = None) -> ImageMeta:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(v) for v in series.shape)

        if axes_override is not None:
            if len(axes_override) != len(shape):
                raise ImageFormatError(
                    f"{path.name}: axes {axes_override!r} has {len(axes_override)} "
                    f"entries but the data has {len(shape)} dimensions {shape}"
                )
            axes = axes_override.upper()

        # A plain TIFF carries no dimension names, so tifffile labels unknown axes 'Q'
        # and sample axes 'S'. Guessing which is z and which is channel would be a
        # coin flip that silently mixes up the data, so require the caller to say.
        if axes_override is None and set(axes) & {"Q", "S"}:
            raise ImageFormatError(
                f"{path.name}: layout is not self-describing (axes {axes!r}). "
                "Pass axes= explicitly, e.g. axes='ZCYX'."
            )

        # drop degenerate leading axes (T, S, Q) that some writers emit as size 1
        keep = [i for i, a in enumerate(axes)
                if a in ("Z", "C", "Y", "X") or shape[i] > 1]
        if len(keep) != len(axes):
            axes = "".join(axes[i] for i in keep)
            shape = tuple(shape[i] for i in keep)

        unsupported = set(axes) - {"Z", "C", "Y", "X"}
        if unsupported:
            raise ImageFormatError(
                f"{path.name}: unsupported axes {sorted(unsupported)} in {axes!r}; "
                "only Z, C, Y and X are handled"
            )
        if "Y" not in axes or "X" not in axes:
            raise ImageFormatError(f"{path.name}: no Y/X axes in {axes!r}")

        if tif.is_ome:
            fmt, voxel = "ome", _ome_voxel_size(tif)
        elif tif.is_imagej:
            fmt, voxel = "imagej", _imagej_voxel_size(tif)
        else:
            fmt, voxel = "plain", None

        return ImageMeta(
            path=path,
            axes=axes,
            shape=shape,
            dtype=np.dtype(series.dtype),
            n_z=shape[axes.index("Z")] if "Z" in axes else 1,
            n_channels=shape[axes.index("C")] if "C" in axes else 1,
            yx_shape=(shape[axes.index("Y")], shape[axes.index("X")]),
            voxel_size_um=voxel,
            source_format=fmt,
        )


def _page_indices(axes: str, shape: Sequence[int], channel: int,
                  z_indices: Sequence[int] | None = None) -> list[int]:
    """Flat page indices holding one channel, optionally restricted in z.

    Pages of a TIFF series are laid out in C order over the axes preceding Y and X, so
    the page holding a given (z, c) is the ravelled index of that position. Computing
    it explicitly — rather than reading everything and slicing — is what makes lazy
    per-channel loading possible.
    """
    lead_axes = axes[:-2]
    lead_shape = tuple(shape[: len(lead_axes)])
    if not lead_axes:
        return [0]

    ci = lead_axes.index("C") if "C" in lead_axes else None
    zi = lead_axes.index("Z") if "Z" in lead_axes else None
    wanted_z = None if z_indices is None else set(int(v) for v in z_indices)

    out: list[int] = []
    for pos in np.ndindex(*lead_shape):
        if ci is not None and pos[ci] != channel:
            continue
        if zi is not None and wanted_z is not None and pos[zi] not in wanted_z:
            continue
        out.append(int(np.ravel_multi_index(pos, lead_shape)))
    return out


# --------------------------------------------------------------------------
# Role verification
# --------------------------------------------------------------------------

@dataclass
class RoleCheck:
    """Whether the declared channel roles are supported by the pixel data.

    Read as a comparison between two hypotheses rather than an absolute score. Real
    images give a high containment either way — CD68 and Iba1 occupy the same tissue,
    so a generous mask of one overlaps the other substantially by coincidence. What
    separates them reliably is that the *correct* assignment always scores higher.
    Measured on two real crops: 0.92 / 0.87 correct against 0.66 / 0.44 swapped.
    """

    performed: bool
    declared_containment: float | None = None
    swapped_containment: float | None = None
    chance_level: float | None = None
    margin: float = 0.10
    reason: str = ""

    @property
    def suspect_swap(self) -> bool:
        """True when the swapped reading is better supported than the declared one."""
        if not self.performed or self.declared_containment is None:
            return False
        return self.swapped_containment > self.declared_containment + self.margin  # type: ignore[operator]

    def summary(self) -> str:
        if not self.performed:
            return f"channel roles not verified: {self.reason}"
        verdict = "CHANNELS APPEAR SWAPPED" if self.suspect_swap else "consistent"
        return (
            f"CD68 peaks inside Iba1: {self.declared_containment:.1%} as declared, "
            f"{self.swapped_containment:.1%} if swapped "
            f"(chance ~{self.chance_level:.0%}) — {verdict}"
        )


# --------------------------------------------------------------------------
# The image
# --------------------------------------------------------------------------

class ImageSource:
    """Lazy, role-addressed access to one multi-channel stack.

    ``channel()`` reads from disk each time and does not cache — the caller decides
    what to keep in memory, which matters when a single 32-bit channel at full field
    size is ~850 MB.
    """

    def __init__(
        self,
        path: str | Path,
        roles: ChannelRoles | Mapping[str, int],
        *,
        axes: str | None = None,
        expected_voxel_size_um: Sequence[float] | None = None,
        voxel_size_tolerance: float = 0.01,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.roles = (roles if isinstance(roles, ChannelRoles)
                      else ChannelRoles.from_mapping(roles))
        self.meta = _read_meta(self.path, axes)

        declared = self.roles.as_dict()
        beyond = {r: c for r, c in declared.items() if c >= self.meta.n_channels}
        if beyond:
            raise ImageFormatError(
                f"{self.path.name} has {self.meta.n_channels} channel(s) but roles "
                f"{beyond} reference higher indices"
            )

        self.voxel_size_um = self._resolve_voxel_size(
            expected_voxel_size_um, voxel_size_tolerance
        )

    # -- voxel size --------------------------------------------------------

    def _resolve_voxel_size(
        self,
        expected: Sequence[float] | None,
        tolerance: float,
    ) -> tuple[float, float, float]:
        found = self.meta.voxel_size_um
        if expected is None:
            if found is None:
                raise VoxelSizeError(
                    f"{self.path.name} records no voxel size and none was supplied. "
                    "Every measurement is physical, so this cannot be guessed."
                )
            return found

        expected_t = tuple(float(v) for v in expected)
        if len(expected_t) != 3:
            raise VoxelSizeError(f"expected_voxel_size_um must be (z, y, x), got {expected}")
        if found is not None:
            drift = [abs(f - e) / e for f, e in zip(found, expected_t)]
            if max(drift) > tolerance:
                raise VoxelSizeError(
                    f"{self.path.name} records voxel size {found} but "
                    f"{expected_t} was expected (max deviation {max(drift):.1%}). "
                    "Refusing to proceed: one of the two is wrong, and guessing "
                    "would corrupt every physical measurement."
                )
        return expected_t

    # -- pixels ------------------------------------------------------------

    def channel(
        self,
        role: str | int,
        *,
        z_indices: Sequence[int] | None = None,
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        """Read one channel as ``(z, y, x)``, reading only the pages required."""
        index = role if isinstance(role, int) else self.roles.index(role)
        pages = _page_indices(self.meta.axes, self.meta.shape, index, z_indices)
        with tifffile.TiffFile(self.path) as tif:
            data = tif.asarray(key=pages)
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[np.newaxis]
        return data.astype(dtype, copy=False)

    def has_role(self, role: str) -> bool:
        return getattr(self.roles, role, None) is not None

    # -- verification ------------------------------------------------------

    def verify_roles(
        self,
        *,
        n_sample_slices: int = 10,
        peak_percentile: float = 99.9,
        generous_percentile: float = 95.0,
        margin: float = 0.10,
    ) -> RoleCheck:
        """Check the declared roles against the pixel data.

        CD68 is a lysosomal marker, so its signal must sit inside the Iba1 cytoplasm.
        Measuring what fraction of CD68's brightest voxels fall within a generous Iba1
        mask tests the assignment directly. The same quantity is computed with the two
        channels exchanged; if the swapped arrangement scores better by more than
        ``margin``, the file's channels are not in the order declared.

        The percentile defaults were tuned on real crops rather than chosen a priori.
        A first attempt used a 90th-percentile mask and a fixed 25-point margin, which
        missed a genuine swap on the larger image: both channels occupy the same
        tissue, so a generous mask overlaps either way and the absolute gap shrinks.
        Tightening the mask to the 95th percentile and comparing the two hypotheses
        against a modest margin separates them on both crops with room to spare
        (0.92 vs 0.66, and 0.87 vs 0.44).
        """
        if not self.has_role("cd68"):
            return RoleCheck(False, reason="no cd68 channel declared")
        if self.meta.n_z < 2:
            return RoleCheck(False, reason="not a z-stack")

        step = max(1, self.meta.n_z // n_sample_slices)
        zs = list(range(0, self.meta.n_z, step))[:n_sample_slices]
        cd68 = self.channel("cd68", z_indices=zs)
        cell = self.channel("cell", z_indices=zs)

        def containment(peaks_of: np.ndarray, mask_of: np.ndarray) -> float:
            peaks = peaks_of > np.percentile(peaks_of, peak_percentile)
            mask = mask_of > np.percentile(mask_of, generous_percentile)
            return float((peaks & mask).sum() / max(peaks.sum(), 1))

        return RoleCheck(
            performed=True,
            declared_containment=containment(cd68, cell),
            swapped_containment=containment(cell, cd68),
            chance_level=1.0 - generous_percentile / 100.0,
            margin=margin,
        )

    def __repr__(self) -> str:
        return f"<ImageSource {self.meta.describe()}>"
