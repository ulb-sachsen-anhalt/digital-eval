from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml.etree import _Element

from ocr_util.corpus.GtResources import GtResource

GT_TARGET_SUBDIR = "GT-PAGE"
GT_METS_FILEGROUP = "OCR-D-GT-FULLTEXT"


@dataclass(frozen=True)
class Args:
    input_dir: Path
    output_dir: Path
    temp_dir: Path
    limit: int
    corpus_label: str = "Ground Truth Corpus"


@dataclass(frozen=True)
class MetsResource:
    identifier_urn: str
    file_path: Path


@dataclass(frozen=True)
class MetsGeneratorResource:
    gt: GtResource
    mets: MetsResource


@dataclass(frozen=True)
class MetsGeneratorResult:
    mets: MetsResource


@dataclass(frozen=True)
class MetsExtract:
    phys_div: _Element
    file_image: _Element
    file_fulltext: _Element
    sm_link: _Element
    log_div: _Element
    dmd_sec: _Element | None
