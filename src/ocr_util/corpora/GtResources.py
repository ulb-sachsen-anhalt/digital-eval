from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from re import Match
from typing import Final


@dataclass
class GtResource:
    identifier: str
    file_base_name: str
    file_path: Path
    relative_file_path: Path
    languages: list[str]


class GtResources:
    __PATTERN_FILE_NAME: Final[str] = r'^((urn\+nbn\+de\+gbv\+\d+\+\d+-\d+-p\d+-\d+)_((?:\w{3}\+?)+))(?:\.gt)?.xml$'

    @classmethod
    def from_dir_copy(cls, in_dir: Path, out_dir: Path, limit: int = 0) -> list[GtResource]:
        gt_resources: list[GtResource] = cls.from_dir(in_dir, limit)
        for gt_resource in gt_resources:
            src_abs_path: Path = gt_resource.file_path
            src_rel_path: Path = gt_resource.file_path.relative_to(in_dir)
            dest_abs_path: Path = out_dir.joinpath(src_rel_path).absolute()
            os.makedirs(dest_abs_path.parent, exist_ok=True)
            shutil.copy2(src_abs_path, dest_abs_path)
            gt_resource.file_path = dest_abs_path
            gt_resource.relative_file_path = src_rel_path
        return gt_resources

    @classmethod
    def from_dir(cls, gt_dir: Path, limit: int = 0) -> list[GtResource]:
        resources: list[GtResource] = []
        current_dir: str
        child_dirs: list[str]
        files: list[str]
        for (current_dir, current_child_dirs, files) in os.walk(gt_dir):
            for file in files:
                file_path: Path = Path(current_dir).joinpath(file)
                match: Match[str] | None = re.match(GtResources.__PATTERN_FILE_NAME, file_path.name)
                if match is not None:
                    urn_enc: str = match.group(2)
                    urn_dec: str = urn_enc.replace('+', ':')
                    langs_enc: str = match.group(3)
                    langs_dec: list[str] = langs_enc.split('+')
                    resources.append(GtResource(
                        identifier=urn_dec,
                        file_base_name=match.group(1),
                        file_path=file_path,
                        relative_file_path=file_path.relative_to(gt_dir),
                        languages=langs_dec
                    ))
                    if 0 < limit <= len(resources):
                        break
            else:
                continue
            break
        return sorted(resources, key=lambda r: r.file_path.name)
