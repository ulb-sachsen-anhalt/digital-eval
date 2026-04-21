# -*- coding: utf-8 -*-
"""OCR Utils"""

import argparse
import os
import re
from pathlib import Path, PurePath

import ocr_util.eval.model as do
import ocr_util.eval.model.filter as dofi
import ocr_util.eval.cli as eval_cli
from ocr_util.corpus.Gt2Mets import Gt2Mets
from ocr_util.corpus.common import Args

# script constants
DEFAULT_VERBOSITY = 0
SUB_CMD_FRAME = "frame"
SUB_CMD_GROUNDTRUTH_CORPUS = "corpus"
CORPUS_CACHE_DIR_NAME = "ocr_util_corpus_mets_cache"
CORPUS_CACHE_DIR = os.path.join(os.path.expanduser("~"), '.cache', CORPUS_CACHE_DIR_NAME)

SUB_CMD_EVALUATE = "eval"



def points_type(points: str) -> str:
    match: re.Match = re.match(dofi.PolygonFrameFilterUtil.POINT_LIST_PATTERN, points)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid point coordinates: '{points}'")
    return points


def start() -> None:
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="ocr-util",
    )
    sub_arg_parsers = arg_parser.add_subparsers(
        title="Subkommandos",
        dest="subcommand",
        required=True,
    )
    frame_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_FRAME,
        help="Filter Contents of provided ALTO-v3-Data by provided Coordinates, where Coordinates span a rectangular"
        " box with",
    )
    frame_arg_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')",
    )
    frame_arg_parser.add_argument(
        "-i", "--input-ocr-file", help="Path of OCR-Data file to process", required=True
    )
    frame_arg_parser.add_argument(
        "-o",
        "--output-ocr-file",
        help="Path of resulting OCR-Data file",
        required=False,
        default=None,
    )
    frame_arg_parser.add_argument(
        "-p",
        "--points",
        required=True,
        type=points_type,
        help="""
        Frame to slice words/lines/regions from input OCR-Data
        f.e.: --frame "2892,2480 5072,2480 5072,5148 2892,5148"
        """,
    )
    
    # groundtruth-corpus subcommand
    groundtruth_corpus_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_GROUNDTRUTH_CORPUS,
        help="Create METS files from ground truth PAGE-XML files with URN identifiers",
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-i",
        "--input",
        dest="input_dir",
        help="Path to the input directory containing GT PAGE-XML files",
        required=True,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        help="Path to the output directory for generated corpus",
        required=True,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=0,
        help="Number of files to process (default: 0 = unlimited)",
        required=False,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-t",
        "--temp-dir",
        dest="temp_dir",
        default=CORPUS_CACHE_DIR,
        help=f"Path to temporary directory for caching METS files (default: {CORPUS_CACHE_DIR})",
        required=False,
    )
    groundtruth_corpus_arg_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')",
    )
    groundtruth_corpus_arg_parser.add_argument(
        "--corpus-label",
        dest="corpus_label",
        default="Ground Truth Corpus",
        help="Label for the corpus in the METS logical structure (default: 'Ground Truth Corpus')",
        required=False,
    )
    
    # evaluate subcommand
    evaluate_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_EVALUATE,
        help="Evaluate OCR candidates against ground truth data",
        add_help=True,
    )
    evaluate_arg_parser.add_argument(
        "candidates",
        help="Root directory for evaluation candidates / Path to single candidate file",
    )
    evaluate_arg_parser.add_argument(
        "-ref", "--reference",
        required=False,
        help="Root directory for reference/groundtruth data",
    )
    evaluate_arg_parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag (optional; default: '{DEFAULT_VERBOSITY}')",
    )
    evaluate_arg_parser.add_argument(
        "--metrics",
        default=eval_cli.DEFAULT_OCR_METRICS,
        required=False,
        help=f"Comma-separated list of metrics (default: '{eval_cli.DEFAULT_OCR_METRICS}')",
    )
    evaluate_arg_parser.add_argument(
        "--utf8",
        default=eval_cli.DEFAULT_UTF8_NORM,
        required=False,
        help=f"UTF-8 normalization form (default: '{eval_cli.DEFAULT_UTF8_NORM}')",
    )
    evaluate_arg_parser.add_argument(
        "-s", "--sequential",
        action="store_true",
        required=False,
        help="Execute calculations sequentially (default: False)",
    )
    evaluate_arg_parser.add_argument(
        "-x", "--extra",
        required=False,
        help="Pass additional information to evaluation (e.g. 'ignore_geometry')",
    )
    evaluate_arg_parser.add_argument(
        "-l", "--language",
        required=False,
        help="Language code for LanguageTool (ISO 639-2)",
    )
    evaluate_arg_parser.add_argument(
        "-u", "--lt-api-url",
        required=False,
        help="LanguageTool API URL",
    )
    evaluate_arg_parser.add_argument(
        "--aggregate-by",
        required=False,
        help="Comma-separated aggregation dimensions (e.g. 'directory', 'type', 'mods:DIMENSION')",
    )
    evaluate_arg_parser.add_argument(
        "--mets-file",
        required=False,
        help="Path to METS/MODS file for MODS-based aggregation",
    )
    evaluate_arg_parser.add_argument(
        "--mods-dimensions",
        required=False,
        help="[LEGACY] Comma-separated MODS dimensions; use --aggregate-by instead",
    )

    args = arg_parser.parse_args()

    verbosity: int = getattr(args, 'verbosity', DEFAULT_VERBOSITY)
    
    if args.subcommand == SUB_CMD_FRAME:
        input_ocr_file: str = args.input_ocr_file
        output_ocr_file: str = args.output_ocr_file
        points: str = args.points
        if verbosity > 1:
            print(
                f"[DEBUG] args: {input_ocr_file}, {output_ocr_file}, {points}, {verbosity}"
            )
        polygon_frame_filter: dofi.PolygonFrameFilter = dofi.PolygonFrameFilter(
            input_ocr_file, points, verbosity
        )
        piece_result: do.DigitalObjectTree = polygon_frame_filter.process()
        file_result: PurePath = do.from_digital_object(piece_result, output_ocr_file)
        if verbosity > 0:
            print("[INFO ] file_result", file_result)
    
    elif args.subcommand == SUB_CMD_GROUNDTRUTH_CORPUS:
        # Create Args object for Gt2Mets
        gt2mets_args = Args(
            input_dir=Path(args.input_dir).absolute(),
            output_dir=Path(args.output_dir).absolute(),
            temp_dir=Path(args.temp_dir).absolute(),
            limit=int(args.limit),
            corpus_label=args.corpus_label
        )
        
        if verbosity > 0:
            print(f"[INFO ] Input directory: {gt2mets_args.input_dir}")
            print(f"[INFO ] Output directory: {gt2mets_args.output_dir}")
            print(f"[INFO ] Temp directory: {gt2mets_args.temp_dir}")
            print(f"[INFO ] Limit: {gt2mets_args.limit if gt2mets_args.limit > 0 else 'unlimited'}")
        
        try:
            gt2mets = Gt2Mets(gt2mets_args)
            gt2mets.run()
            if verbosity > 0:
                print("[INFO ] GT-METS generation completed successfully")
        except Exception as e:
            print(f"[ERROR] Failed to generate METS files: {e}")
            raise

    elif args.subcommand == SUB_CMD_EVALUATE:
        eval_args = vars(args)
        eval_args.pop('subcommand', None)
        eval_cli.start_evaluation(eval_args)


if __name__ == "__main__":
    start()
