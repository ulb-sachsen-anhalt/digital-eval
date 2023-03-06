# -*- coding: utf-8 -*-
"""OCR Utils"""

import argparse
import sys
from argparse import _SubParsersAction, ArgumentParser
from typing import Any, Union, Final

from FrameFilterAltoV3 import FrameFilterAltoV3
from datatypes import Point2DList, Point2D
from Util import Util

# script constants

DEFAULT_VERBOSITY: int = 0
SUB_CMD_FRAME: Final[str] = 'frame'


def argparse_point2d_type(point: str) -> Point2D:
    try:
        return Util.str_to_point_2d(point)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Invalid point coordinates: '{point}'")


def start() -> None:
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="ocr-util",
    )
    arg_parser.add_argument(
        "-v", "--verbosity",
        action='count',
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')"
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
        nargs=4,
        required=True,
        type=argparse_point2d_type,
        help=f"""
        Frame to slice words/lines/regions from input OCR-Data
        f.e.: --frame 2892,2480 5072,2480 5072,5148 2892,5148
        """
    )
    args = arg_parser.parse_args()

    verbosity: int = args.verbosity
    if args.subcommand == SUB_CMD_FRAME:
        input_ocr_file: str = args.input_ocr_file
        output_ocr_file: str = args.output_ocr_file
        points: Point2DList = args.points
        frame_filter: FrameFilterAltoV3 = FrameFilterAltoV3(
            path_alto_in=input_ocr_file,
            path_alto_out=output_ocr_file,
            points=points,
            verbosity=verbosity
        )
    filter_result: str = frame_filter.process()

    print('filter_result', filter_result)


if __name__ == "__main__":
    start()
