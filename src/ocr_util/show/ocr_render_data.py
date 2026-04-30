# -*- coding: utf-8 -*-
"""ULB DD/IT OCR Segementation Visualization"""

import argparse
import os
import xml.dom.minidom

from cv2 import cv2


def _convert(path_img):
    img = cv2.imread(path_img, cv2.IMREAD_COLOR)
    filename = os.path.basename(path_img).split('.')[0]
    directory = os.path.dirname(path_img)
    full_conversion_path = os.path.join(directory, filename+'.png')
    msg_convert = 'convert {} to {}'.format(path_img, full_conversion_path)
    print('[INFO] ' + msg_convert)
    cv2.imwrite(full_conversion_path, img)
    return full_conversion_path


def _visualize(path_img, path_alto):
    start_msg = 'merge img {} with alto {}'.format(path_img, path_alto)
    print('[INFO] ' + start_msg)
    root_el = _parse_dom(path_alto)
    textblock_els = _load_children(root_el, 'TextBlock')
    line_els = _load_children(root_el, 'TextLine')
    string_els = _load_children(root_el, 'String')
    els = _transform(textblock_els)
    lines = _transform(line_els)
    contents = _transform(string_els)
    if els:
        _render_bb(path_img, els, (0, 196, 0), -1, (0, 255, 0))
    if lines:
        _render_bb(path_img, lines, (0, 0, 196), 8, (0, 0, 255))
    if contents:
        _add_contents(path_img, contents, (255, 0, 0))


def _parse_dom(file_path):
    dom_tree = xml.dom.minidom.parse(file_path)
    return dom_tree.documentElement


def _load_children(element, children):
    return element.getElementsByTagName(children)

def _transform(xml_blocks):
    transformed = [(_get_element_data(block)) for block in xml_blocks]
    print('[INFO] read ' + str(len(transformed)) + ' entries')
    return transformed

def _get_element_data(el):
    box_id = el.getAttribute('ID')
    x1 = int(el.getAttribute('HPOS'))
    y1 = int(el.getAttribute('VPOS'))
    x2 = x1 + int(el.getAttribute('WIDTH'))
    y2 = y1 + int(el.getAttribute('HEIGHT'))
    text = el.getAttribute('CONTENT')
    box_data = (box_id, x1, y1, x2, y2, text)
    if VERBOSE_LEVEL1:
        msg_box = 'read data {}'.format(box_data)
        print('[DEBUG] ' + msg_box)
    return (box_id, x1, y1, x2, y2, text)

def _render_bb(path_img, els, color, thickness, text_color):
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
    cv2.imshow('render result', img)
    cv2.waitKey(0)
    cv2.imwrite(path_img, img)

def _add_contents(path_img, els, text_color):
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

    #cv2.addWeighted(overlay_box, 0.7, img, 0.3, 0, img)
    cv2.addWeighted(overlay_text, 0.8, img, 0.2, 0, img)
    cv2.imshow('render result', img)
    cv2.waitKey(0)
    cv2.imwrite(path_img, img)


########
# MAIN #
########
APP_ARGUMENTS = argparse.ArgumentParser()
APP_ARGUMENTS.add_argument("-i", "--image", required=True, help="path image file (TIF)")
APP_ARGUMENTS.add_argument("-a", "--alto", required=True, help="path alto file")
APP_ARGUMENTS.add_argument("-v", "--verbose", required=False, help="output info")

ARGS = vars(APP_ARGUMENTS.parse_args())
PATH_IMG = ARGS["image"]
PATH_ALTO = ARGS["alto"]
VERBOSE_LEVEL1 = ARGS["verbose"]

if VERBOSE_LEVEL1:
    print('[INFO] call with args ' + str(ARGS))

if os.path.exists(PATH_IMG) and os.path.exists(PATH_ALTO):
    CONVERTED_TMP = _convert(PATH_IMG)
    _visualize(CONVERTED_TMP, PATH_ALTO)
else:
    print('[ERROR] {} or {} not existing!'.format(PATH_IMG, PATH_ALTO))
