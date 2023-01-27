# -*- coding: utf-8 -*-
"""OCR Utils"""

import argparse
import os

import datetime as dt


# script constants
DEFAULT_VERBOSITY = 0
VERBOSITY = DEFAULT_VERBOSITY
EVAL_VERBOSITY = DEFAULT_VERBOSITY


def _get_info():
    here = os.path.abspath(os.path.dirname(__file__))
    _v = ''
    _t = ''
    _fp = os.path.join(here, 'VERSION')
    with open(_fp) as fp:
        _v = fp.read()
    _t = dt.datetime.fromtimestamp(os.stat(_fp).st_mtime).strftime("%Y-%m-%d")
    return f'v{_v}/{_t}'


def _main(path_input_ocr, **kwargs):
    """Determine main workflow"""

    print(f"{path_input_ocr} with {kwargs}")


def start():
    PARSER = argparse.ArgumentParser(description=f"""
        Evaluate Mass Digital Data. ({_get_info()})
        """)
    PARSER.add_argument(
        "input_ocr", 
        help="Path of OCR-Data to process"
        )
    PARSER.add_argument("-f","--frame", 
        required=False,
        help=f"Frame to slice words/lines/regions from input OCR-Data"
        )
    PARSER.add_argument("-v", "--verbosity", 
        action='count', 
        default=DEFAULT_VERBOSITY,
        required=False, 
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')"
        )
    
    ARGS = vars(PARSER.parse_args())
    path_input_ocr = ARGS["input_ocr"]
    # TODO
    # fail-fast: check if given input path is valid
    del ARGS["input_ocr"]
    
    global VERBOSITY
    VERBOSITY = ARGS["verbosity"]
    del ARGS["verbosity"]

    _main(path_input_ocr, **ARGS)


if __name__ == "__main__":
    start()
