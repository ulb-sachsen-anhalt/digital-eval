# -*- coding: utf-8 -*-
"""ULB DD/IT OCR Segmentation Visualization"""

import argparse
import os
from typing import Any, cast

import cv2 as _cv2

# cv2 bindings are generated dynamically and frequently confuse static analyzers.
cv2 = cast(Any, _cv2)

# silence pylint false positives for OpenCV C-extension bindings
# pylint:disable=no-member
# pylint:disable=c-extension-no-member

from ocr_util.eval.model.common import DigitalObjectLevel
from ocr_util.eval.model.digital_object_util import DigitalObjectUtil
from ocr_util.eval.model.main import to_digital_object


def _convert(path_img):
    img = cv2.imread(path_img, cv2.IMREAD_COLOR)
    filename = os.path.basename(path_img).split('.')[0]
    directory = os.path.dirname(path_img)
    full_conversion_path = os.path.join(directory, filename+'.png')
    msg_convert = 'convert {} to {}'.format(path_img, full_conversion_path)
    print('[INFO] ' + msg_convert)
    cv2.imwrite(full_conversion_path, img)
    return full_conversion_path


def _digo_to_box_data(digo):
    """Convert a DigitalObjectTree node into a (id, x1, y1, x2, y2, text) tuple.
    Returns None when no bounding box is available."""
    box = digo.as_box()
    if box is None:
        return None
    x1 = int(box[0][0])
    y1 = int(box[0][1])
    x2 = int(box[2][0])
    y2 = int(box[2][1])
    try:
        text = digo.transcription
    except RuntimeError:
        text = ''
    box_data = (digo.id, x1, y1, x2, y2, text)
    if VERBOSE_LEVEL1:
        print('[DEBUG] read data {}'.format(box_data))
    return box_data


def _visualize(path_img, path_ocr, display=False):
    start_msg = 'merge img {} with ocr data {}'.format(path_img, path_ocr)
    print('[INFO] ' + start_msg)
    root_digo = to_digital_object(path_ocr)
    all_digos = DigitalObjectUtil.flatten(root_digo)

    regions = [_digo_to_box_data(d) for d in all_digos if d.level == DigitalObjectLevel.REGION]
    lines   = [_digo_to_box_data(d) for d in all_digos if d.level == DigitalObjectLevel.LINE]
    words   = [_digo_to_box_data(d) for d in all_digos if d.level == DigitalObjectLevel.WORD]

    regions = [r for r in regions if r is not None]
    lines   = [l for l in lines   if l is not None]
    words   = [w for w in words   if w is not None]

    print('[INFO] read {} regions'.format(len(regions)))
    print('[INFO] read {} lines'.format(len(lines)))
    print('[INFO] read {} words'.format(len(words)))

    if regions:
        _render_bb(path_img, regions, (0, 196, 0), -1, (0, 255, 0), display=display)
    if lines:
        _render_bb(path_img, lines, (0, 0, 196), 8, (0, 0, 255), display=display)
    if words:
        _add_contents(path_img, words, (255, 0, 0), display=display)

def _render_bb(path_img, els, color, thickness, text_color, display=False):
    msg_render = 'render {} elements on img {}'.format(len(els), path_img)
    print('[INFO] ' + msg_render)
    img = cv2.imread(path_img)
    overlay_box = img.copy()
    overlay_text = img.copy()
    for el in els:
        x1 = el[1]
        y1 = el[2]
        x2 = el[3]
        y2 = el[4]
        #msg_rect = 'rectangle bb {}x{}:{}x{}'.format(x1, y1, x2, y2)
        #print('[DEBUG] ' + msg_rect)
        margin_left = 10
        margin_top = 30
        cv2.putText(overlay_text, "ID: {}".format(el[0]), (x1+margin_left, y1+margin_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
        cv2.rectangle(overlay_box, (x1, y1), (x2, y2), color, thickness)

    cv2.addWeighted(overlay_box, 0.7, img, 0.3, 0, img)
    cv2.addWeighted(overlay_text, 0.8, img, 0.2, 0, img)
    if display:
        try:
            cv2.imshow('render result', img)
            cv2.waitKey(0)
        except cv2.error as e:
            print('[WARNING] Display not available (headless environment): {}'.format(str(e)))
    cv2.imwrite(path_img, img)

def _add_contents(path_img, els, text_color, display=False):
    msg_render = 'render {} elements on img {}'.format(len(els), path_img)
    print('[INFO] ' + msg_render)
    img = cv2.imread(path_img)
    #overlay_box = img.copy()
    overlay_text = img.copy()
    for el in els:
        x1 = el[1]
        y1 = el[2]
        #x2 = el[3]
        #y2 = el[4]
        #msg_rect = 'rectangle bb {}x{}:{}x{}'.format(x1, y1, x2, y2)
        #print('[DEBUG] ' + msg_rect)
        margin_left = 10
        margin_top = 30
        render_text = "{}".format(el[5])
        if not render_text:
            render_text = '?'
        cv2.putText(overlay_text, render_text, (x1+margin_left, y1+margin_top), cv2.FONT_HERSHEY_COMPLEX, 1.0, text_color, 3)
        #cv2.rectangle(overlay_box, (x1, y1), (x2, y2), color, thickness)

    cv2.addWeighted(overlay_text, 0.8, img, 0.2, 0, img)
    if display:
        try:
            cv2.imshow('render result', img)
            cv2.waitKey(0)
        except cv2.error as e:
            print('[WARNING] Display not available (headless environment): {}'.format(str(e)))
    cv2.imwrite(path_img, img)


########
# MAIN #
########
APP_ARGUMENTS = argparse.ArgumentParser()
APP_ARGUMENTS.add_argument("-i", "--image", required=True, help="path image file (TIF)")
APP_ARGUMENTS.add_argument("-o", "--ocr", required=True, help="path OCR file (ALTO or PAGE XML)")
APP_ARGUMENTS.add_argument("-v", "--verbose", required=False, help="output info")
APP_ARGUMENTS.add_argument("-d", "--display", required=False, action="store_true", help="display result interactively (requires GUI support)")

ARGS = vars(APP_ARGUMENTS.parse_args())
PATH_IMG = ARGS["image"]
PATH_OCR = ARGS["ocr"]
VERBOSE_LEVEL1 = ARGS["verbose"]
DISPLAY = ARGS["display"]

if VERBOSE_LEVEL1:
    print('[INFO] call with args ' + str(ARGS))

if os.path.exists(PATH_IMG) and os.path.exists(PATH_OCR):
    CONVERTED_TMP = _convert(PATH_IMG)
    _visualize(CONVERTED_TMP, PATH_OCR, display=DISPLAY)
else:
    print('[ERROR] {} or {} not existing!'.format(PATH_IMG, PATH_OCR))
