# -*- coding: utf-8 -*-
"""OCR Utils"""

import argparse
import re
from pathlib import PurePath

import digital_object as do
import digital_object.filter as dofi

# script constants
DEFAULT_VERBOSITY = 0
SUB_CMD_FRAME = 'frame'


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
        title='Subkommandos',
        dest='subcommand',
        required=True,
    )
    frame_arg_parser = sub_arg_parsers.add_parser(
        SUB_CMD_FRAME,
        help='Filter Contents of provided ALTO-v3-Data by provided Coordinates, where Coordinates span a rectangular'
             ' box with'
    )
    frame_arg_parser.add_argument(
        "-v", "--verbosity",
        action='count',
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')"
    )
    frame_arg_parser.add_argument(
        "-i", "--input-ocr-file",
        help="Path of OCR-Data file to process",
        required=True
    )
    frame_arg_parser.add_argument(
        "-o", "--output-ocr-file",
        help="Path of resulting OCR-Data file",
        required=False,
        default=None
    )
    frame_arg_parser.add_argument(
        "-p", "--points",
        required=True,
        type=points_type,
        help="""
        Frame to slice words/lines/regions from input OCR-Data
        f.e.: --frame "2892,2480 5072,2480 5072,5148 2892,5148"
        """
    )
    args = arg_parser.parse_args()

    verbosity: int = args.verbosity
    if args.subcommand == SUB_CMD_FRAME:
        input_ocr_file: str = args.input_ocr_file
        output_ocr_file: str = args.output_ocr_file
        points: str = args.points
        if verbosity > 1:
            print(f"[DEBUG] args: {input_ocr_file}, {output_ocr_file}, {points}, {verbosity}")
        polygon_frame_filter: dofi.PolygonFrameFilter = dofi.PolygonFrameFilter(
            input_ocr_file,
            points,
            verbosity
        )
        piece_result: do.DigitalObjectTree = polygon_frame_filter.process()
        file_result: PurePath = do.from_digital_object(piece_result, output_ocr_file)
        if verbosity > 0:
            print('[INFO ] file_result', file_result)


if __name__ == "__main__":
    start()
