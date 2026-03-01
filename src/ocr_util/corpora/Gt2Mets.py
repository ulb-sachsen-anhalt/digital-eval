from __future__ import annotations

import os
import shutil
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from ocr_util.corpora.GtResources import GtResources
from ocr_util.corpora.MetsFileObtainer import MetsFileObtainer
from ocr_util.corpora.MetsGenerator import MetsGenerator
from ocr_util.corpora.common import Args, GtResource, MetsGeneratorResource, MetsResource, GT_TARGET_SUBDIR


class Gt2Mets:
    __URL_URN_RESOLVER: Final[str] = "https://nbn-resolving.org/process-urn-form"
    __URL_OAI_PMH_DATA: Final[str] = "https://opendata.uni-halle.de/oai/dd"
    __URL_OAI_PMH_ID_PREFIX: Final[str] = "oai:opendata.uni-halle.de"
    __NUM_TRHEADS: Final[int] = int(os.cpu_count() * 0.85)
    __DEFAULT_TEMP_DIR: Final[str] = os.path.join(os.path.expanduser("~"), '.cache', 'odem_gt_2_mets')
    __DEFAULT_LIMIT: Final[int] = 0

    @staticmethod
    def __parse_args() -> Args:
        parser: ArgumentParser = ArgumentParser()
        parser.add_argument("input", help="Path to the input directory")
        parser.add_argument("output", help="Path to the output directory")
        parser.add_argument(
            "-l",
            "--limit",
            help="Number of Files being processed, default = 0 (unlimited)",
            required=False,
            type=int,
            default=Gt2Mets.__DEFAULT_LIMIT
        )
        parser.add_argument(
            "-t",
            "--temp-dir",
            help="Path to the temporary directory",
            required=False,
            default=Gt2Mets.__DEFAULT_TEMP_DIR
        )
        args: Namespace = parser.parse_args()
        return Args(
            input_dir=Path(args.input).absolute(),
            output_dir=Path(args.output).absolute(),
            temp_dir=Path(args.temp_dir).absolute(),
            limit=int(args.limit),
            corpus_label="Ground Truth Corpus"
        )

    def __init__(self, args: Args | None = None):
        if args is None:
            args = Gt2Mets.__parse_args()
        if not args.input_dir.exists():
            raise RuntimeError(f"The input directory '{args.input_dir}' does not exist")
        self.__args: Final[Args] = args

    def run(self) -> None:
        self.__args.temp_dir.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.__args.output_dir):
            shutil.rmtree(self.__args.output_dir)
        self.__args.output_dir.mkdir(parents=True, exist_ok=True)
        gt_resources: list[GtResource] = GtResources.from_dir_copy(
            in_dir=self.__args.input_dir,
            out_dir=self.__args.output_dir.joinpath(GT_TARGET_SUBDIR),
            limit=self.__args.limit
        )
        mets_resources: list[MetsResource] = self.__obtain_mets_files(gt_resources)
        mets_generator_resources: list[MetsGeneratorResource] = [
            MetsGeneratorResource(gt=gt_resource, mets=mets_resources[i])
            for i, gt_resource
            in enumerate(gt_resources)
        ]
        mets_generator: MetsGenerator = MetsGenerator(
            self.__args.output_dir,
            mets_generator_resources,
            corpus_label=self.__args.corpus_label
        )
        mets_generator.run()

    def __obtain_mets_files(self, gt_resources: list[GtResource]) -> list[MetsResource]:
        mets_dir_path: Path = Path(f'{self.__args.temp_dir}').joinpath('mets')
        if not mets_dir_path.exists():
            mets_dir_path.mkdir()
        mets_file_obtainers: list[MetsFileObtainer] = [
            MetsFileObtainer(
                urn=gt_resource.identifier,
                file_path=mets_dir_path.joinpath(f'{gt_resource.file_base_name}.mets.xml'),
                url_urn_resolver=Gt2Mets.__URL_URN_RESOLVER,
                url_oai_pmh_data=Gt2Mets.__URL_OAI_PMH_DATA,
                url_oai_pmh_id_prefix=Gt2Mets.__URL_OAI_PMH_ID_PREFIX,
            )
            for gt_resource
            in gt_resources
        ]
        # Use ThreadPoolExecutor directly for parallel execution
        with ThreadPoolExecutor(max_workers=self.__NUM_TRHEADS) as executor:
            futures = [executor.submit(obtainer.run) for obtainer in mets_file_obtainers]
            return [future.result() for future in futures]
