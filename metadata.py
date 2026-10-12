"""Where each image came from: genotype, animal, section, region, index.

**Genotype and animal come from the path, never the filename** (§4.14). The `691643` images
are named `WT` and are 5xFAD; `692137`'s clump crops are named `WT` too. Trusting a filename
would silently mislabel a genotype, which is the one error that would invalidate the entire
comparison while leaving every number looking plausible.

**The filename supplies section, region and index**, confirmed by the user 2026-08-09::

    croptest[clump]5xFAD_HPL_C3_691619_4Subiculum1_result
                                       ^ ^        ^
                            section ---+ |        +--- index within section+ROI
                            region ------+

* ``4Subiculum1`` -> section 4, Subiculum, image 1
* ``2Hippiocampus3`` -> section 2, hippocampus, image 3
* ``5Subiculum`` -> section 5, Subiculum, **no index** — it is optional

``HPL`` and ``C3`` are constant across every file and carry no discriminating information.
They are captured verbatim rather than interpreted; inventing meanings for them would be
guessing.

**Treatment is deliberately absent.** The user is blinded, and treatment joins at analysis
time on animal id. No treatment field is invented, and none is left null looking like data.

**Region spelling is not normalised silently.** Some filenames read ``Hippiocampus``. The raw
token is the evidence and is always kept; a canonical form is offered beside it, never
instead of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ImageMetadata", "parse_metadata", "GenotypeMismatch"]

#: `<section number><Region><optional index>` — the region is letters only, so the trailing
#: digits are the index and the leading digits the section.
_FIELD = re.compile(r"^(?P<section>\d+)(?P<region>[A-Za-z]+?)(?P<index>\d*)$")

#: Known genotype folder names, matched case-insensitively against path parts.
_GENOTYPES = {"WT": "WT", "5XFAD": "5xFAD"}

#: Spelling variants seen in the data. Maps to a canonical form for grouping; the raw token
#: is always retained alongside.
_REGION_ALIASES = {"hippiocampus": "Hippocampus"}


class GenotypeMismatch(UserWarning):
    """The filename claims a genotype the path disagrees with. The path wins."""


@dataclass(frozen=True)
class ImageMetadata:
    """Everything the pipeline can know about an image before opening it."""

    path: str
    genotype: str | None            # from the PATH
    animal: str | None              # from the PATH
    section: int | None
    region_raw: str | None          # exactly as written, including misspellings
    region: str | None              # canonical form for grouping
    index: int | None
    filename_genotype: str | None   # what the name claims, for the cross-check only
    tokens: tuple[str, ...]         # the uninterpreted parts, e.g. HPL, C3
    warnings: tuple[str, ...] = ()

    @property
    def genotype_disagrees(self) -> bool:
        return (self.filename_genotype is not None and self.genotype is not None
                and self.filename_genotype != self.genotype)


def _genotype_and_animal(path: Path) -> tuple[str | None, str | None]:
    """Walk the path for `<genotype>/<animal>/file`, matching case-insensitively."""
    parts = list(path.parts)
    for i, part in enumerate(parts[:-1]):
        canonical = _GENOTYPES.get(part.upper())
        if canonical is None:
            continue
        animal = parts[i + 1] if i + 1 < len(parts) - 1 else None
        return canonical, animal
    return None, None


def parse_metadata(image_path: str | Path) -> ImageMetadata:
    """Parse provenance from a path, preferring the path wherever the two disagree."""
    path = Path(image_path)
    genotype, animal = _genotype_and_animal(path)

    stem = path.name
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"_result$", "", stem)

    parts = stem.split("_")
    filename_genotype = None
    section = region_raw = region = index = None
    tokens: list[str] = []

    for part in parts:
        upper = part.upper()
        for name, canonical in _GENOTYPES.items():
            # `croptest5xFAD` and `croptestclumpWT` both embed the genotype in a prefix
            if upper.endswith(name):
                filename_genotype = canonical
                break
        else:
            match = _FIELD.match(part)
            if match and match.group("region"):
                section = int(match.group("section"))
                region_raw = match.group("region")
                region = _REGION_ALIASES.get(region_raw.lower(), region_raw)
                index = int(match.group("index")) if match.group("index") else None
                continue
            if part and not part.isdigit():
                tokens.append(part)

    warnings: list[str] = []
    if filename_genotype and genotype and filename_genotype != genotype:
        warnings.append(
            f"filename says {filename_genotype} but the path says {genotype}; "
            "the path is authoritative (rationale 4.14) and is used"
        )
    if genotype is None:
        warnings.append("no genotype folder in the path; genotype unknown")
    if section is None:
        warnings.append(f"no <section><Region><index> field found in {stem!r}")

    return ImageMetadata(
        path=str(path),
        genotype=genotype,
        animal=animal,
        section=section,
        region_raw=region_raw,
        region=region,
        index=index,
        filename_genotype=filename_genotype,
        tokens=tuple(tokens),
        warnings=tuple(warnings),
    )
