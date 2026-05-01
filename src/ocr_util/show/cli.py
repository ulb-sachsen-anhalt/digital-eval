# -*- coding: utf-8 -*-
"""Show CLI - OCR Visualization"""

import os
import typing

from ocr_util.show.ocr_show_segmentation import _convert, _visualize


def start_show(parse_args: typing.Dict):
    """Start show processing with parsed arguments from parent CLI"""
    path_img = parse_args["image"]
    path_ocr = parse_args["ocr"]
    output_dir = parse_args.get("output_dir", os.getcwd())
    verbose = parse_args.get("verbose", False)

    if verbose:
        print('[INFO] call with args ' + str(parse_args))
        print('[INFO] output directory: {}'.format(output_dir))

    if os.path.exists(path_img) and os.path.exists(path_ocr):
        converted_tmp = _convert(path_img, output_dir)
        final_output = _visualize(converted_tmp, path_ocr)
        print('[INFO] visualization complete. Final output: {}'.format(final_output))
    else:
        print('[ERROR] {} or {} not existing!'.format(path_img, path_ocr))
