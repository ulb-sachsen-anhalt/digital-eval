# -*- coding: utf-8 -*-
"""OCR Utils"""

import argparse
import os

import datetime as dt

# script constants
DEFAULT_VERBOSITY = 0
VERBOSITY = DEFAULT_VERBOSITY
EVAL_VERBOSITY = DEFAULT_VERBOSITY


def get_info():
    here = os.path.abspath(os.path.dirname(__file__))
    _v = ''
    _t = ''
    _fp = os.path.join(here, 'VERSION')
    with open(_fp) as fp:
        _v = fp.read()
    _t = dt.datetime.fromtimestamp(os.stat(_fp).st_mtime).strftime("%Y-%m-%d")
    return f'v{_v}/{_t}'


def main(path_input_ocr, **kwargs):
    """Determine main workflow"""

    print(f"{path_input_ocr} with {kwargs}")


def start():
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=f"""
        Evaluate Mass Digital Data. ({get_info()})
        """)
    arg_parser.add_argument(
        "input_ocr",
        help="Path of OCR-Data to process"
    )
    arg_parser.add_argument(
        "-f", "--frame",
        required=False,
        help=f"Frame to slice words/lines/regions from input OCR-Data"
    )
    arg_parser.add_argument(
        "-v", "--verbosity",
        action='count',
        default=DEFAULT_VERBOSITY,
        required=False,
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')"
    )

    args = vars(arg_parser.parse_args())
    path_input_ocr = args["input_ocr"]
    # TODO
    # fail-fast: check if given input path is valid
    del args["input_ocr"]

    global VERBOSITY
    VERBOSITY = args["verbosity"]
    del args["verbosity"]

    main(path_input_ocr, **args)


if __name__ == "__main__":
    start()
