# -*- coding: utf-8 -*-
"""OCR Model Test Module"""

from os.path import (
    join,
)

import shutil

import pytest

from shapely.geometry import (
    Polygon
)

from digital_eval.evaluation import (
    OCRData,
    get_bbox_data,
)

from .conftest import (
    TEST_RES_DIR,
)


@pytest.fixture(name='page_gt_type_art_filename')
def create_alto_gt_type_article(tmp_path):
    original_file = f'{TEST_RES_DIR}/groundtruth/page/1681877805_J_0075_0001.art.gt.xml'
    tmp_filename = '1681877805_J_0075_0001.gt.art1.xml'
    tmp_alto = tmp_path / 'alto'
    tmp_alto.mkdir()
    path = tmp_alto / tmp_filename
    shutil.copyfile(original_file, path)
    return str(path)


@pytest.fixture(name='page_gt_type_ann_filename')
def create_alto_gt_type_announcement(tmp_path):
    original_file = f'{TEST_RES_DIR}/groundtruth/page/1681877805_J_0075_0001.art.gt.xml'
    tmp_filename = '1681877805_J_0075_0001.gt.annx.xml'
    tmp_alto = tmp_path / 'alto'
    tmp_alto.mkdir()
    path = tmp_alto / tmp_filename
    shutil.copyfile(original_file, path)
    return str(path)


def test_groundtruth_type_from_file_with_art1_in_name(
        page_gt_type_art_filename):
    '''check that gt-type "article" can be extracted from file with "art1" in name'''

    ocr_data = OCRData(page_gt_type_art_filename)

    assert 'n.a.' != ocr_data.get_type_groundtruth()
    assert ocr_data.get_type_groundtruth().startswith('art')


def test_groundtruth_type_from_file_with_annx_in_name(
        page_gt_type_ann_filename):
    '''check that gt-type "announcement" can be extracted from file with "annx" in name'''

    ocr_data = OCRData(page_gt_type_ann_filename)

    assert 'n.a.' != ocr_data.get_type_groundtruth()
    assert ocr_data.get_type_groundtruth().startswith('ann')


def test_get_bbox_from_filename():
    file_path = join(TEST_RES_DIR, 'groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')
    actual_bbox = get_bbox_data(file_path)
    assert ((375, 2050), (2325, 9550)) == actual_bbox


def test_get_bbox_from_string_data():
    file_path = f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0001_0768.xml'
    actual_bbox = get_bbox_data(file_path)
    assert ((61, 151), (7395, 10305)) == actual_bbox


def test_get_bbox_from_ocrd_page():
    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml'

    # act
    (p1, p2) = get_bbox_data(ocr_path)

    # assert
    assert p1[0] == 220
    assert p1[1] == 240
    assert p2[0] == 1048
    assert p2[1] == 1646


def test_get_bbox_from_page2019():
    """Ensure other PAGE formats than Transcribus 2013
    can be used as GT-Input
    """

    # arrange
    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/page01.gt.xml'

    # act
    (p1, p2) = get_bbox_data(ocr_path)

    # assert
    assert p1[0] == 667
    assert p1[1] == 595
    assert p2[0] == 2317
    assert p2[1] == 2900


def test_get_bbox_fails_file_missing():
    file_path = f'{TEST_RES_DIR}/alto/gt/1667522809_J_0073_0002.xml'
    with pytest.raises(IOError) as exc:
        get_bbox_data(file_path)
    assert "not existing" in str(exc)
