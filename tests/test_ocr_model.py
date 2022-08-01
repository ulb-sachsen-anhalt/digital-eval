# -*- coding: utf-8 -*-
"""OCR Model Test Module"""

from os.path import (
    join,
)

import shutil

import pytest

from digital_eval.evaluation import (
    OCRData,
    get_bbox_data,
    review
)

from digital_eval.model import (
    to_pieces,
    PieceType
)

from .conftest import (
    TEST_RES_DIR,
)


@pytest.fixture(name='page_gt_type_art_filename')
def create_alto_gt_type_article(tmp_path):
    original_file = './tests/resources/groundtruth/page/1681877805_J_0075_0001.art.gt.xml'
    tmp_filename = '1681877805_J_0075_0001.gt.art1.xml'
    tmp_alto = tmp_path / 'alto'
    tmp_alto.mkdir()
    path = tmp_alto / tmp_filename
    shutil.copyfile(original_file, path)
    return str(path)


@pytest.fixture(name='page_gt_type_ann_filename')
def create_alto_gt_type_announcement(tmp_path):
    original_file = './tests/resources/groundtruth/page/1681877805_J_0075_0001.art.gt.xml'
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
    '''check that gt-type "annoucement" can be extracted from file with "annx" in name'''

    ocr_data = OCRData(page_gt_type_ann_filename)

    assert 'n.a.' != ocr_data.get_type_groundtruth()
    assert ocr_data.get_type_groundtruth().startswith('ann')


def test_get_bbox_from_filename():
    file_path = join(TEST_RES_DIR, 'groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')
    actual_bbox = get_bbox_data(file_path)
    assert ((375, 2050), (2325, 9550)) == actual_bbox


def test_get_bbox_from_string_data():
    file_path = './tests/resources/candidate/frk_alto/1667522809_J_0001_0768.xml'
    actual_bbox = get_bbox_data(file_path)
    assert ((61, 151), (7395, 10305)) == actual_bbox


def test_get_bbox_from_ocrd_page():
    ocr_path = './tests/resources/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml'

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
    ocr_path = './tests/resources/groundtruth/page/page01.gt.xml'

    # act
    (p1, p2) = get_bbox_data(ocr_path)

    # assert
    assert p1[0] == 667
    assert p1[1] == 595
    assert p2[0] == 2317
    assert p2[1] == 2900


def test_get_bbox_fails_file_missing():
    file_path = './tests/resources/alto/gt/1667522809_J_0073_0002.xml'
    with pytest.raises(IOError) as exc:
        get_bbox_data(file_path)
    assert "not existing" in str(exc)


def test_review_without_gt_coords():
    """Check Behavior when GT is just plain text an cannot provide any kind of GT-Frame"""

    # arrange
    file_gt = './tests/resources/groundtruth/txt/217745.gt.txt'
    file_cn = './tests/resources/candidate/ara_alto/217745.xml'

    # act I
    coords = get_bbox_data(file_gt)

    # assert
    assert coords is None

    # act II
    result = review(file_cn, coords, oneliner=True)

    # assert
    assert result


def test_to_pieces_page_odem_transkribus_gt():
    """Ensure PAGE 2013 Transcribus Groundtruth works"""

    ocr_path = './tests/resources/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml'

    # act
    page_piece = to_pieces(ocr_path)

    # assert
    # 1 region
    assert len(page_piece.pieces) == 1
    # 23 lines
    assert len(page_piece.pieces[0].pieces) == 23
    # first line has 7 words
    assert len(page_piece.pieces[0].pieces[0].pieces) == 7
    # last line has 7 words
    assert len(page_piece.pieces[0].pieces[22].pieces) == 4

    # first line textual content
    line1_text = page_piece.pieces[0].pieces[0].transcription
    assert line1_text == 'und erklaͤret die Schrift nicht nur al⸗'
    assert len(line1_text) == 39
    # first region textual content ...
    region_text = page_piece.pieces[0].transcription
    assert len(region_text) == 829
    # ... which, in this case (only one single region),
    # is equals the complete top_piece textual content
    assert page_piece.transcription == region_text


def test_to_pieces_altov3():
    """Ensure old ZD1 groundtruth in ALTO V3 can still be computed"""

    ocr_path = join(TEST_RES_DIR, 'groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')

    # act
    page_piece = to_pieces(ocr_path)

    # assert
    # 10 regions
    assert len(page_piece.pieces) == 10
    # region IDa
    assert page_piece.pieces[0].id == 'block_27'
    assert page_piece.pieces[1].id == 'block_28'

    # region 2 has 2 lines
    assert len(page_piece.pieces[1].pieces) == 2
    # line 1 of region 1 has 2 words
    assert len(page_piece.pieces[0].pieces[0].pieces) == 2
    # line 2 of region 2 has 5 words
    assert len(page_piece.pieces[1].pieces[1].pieces) == 5

    # first line textual content
    line1_text = page_piece.pieces[0].pieces[0].transcription
    assert line1_text == 'Neueſte Ereigniſſe.'
    # first region has just one single line,
    # therefore both must match
    region1_text = page_piece.pieces[0].transcription
    assert region1_text == line1_text
    assert page_piece.transcription.startswith(region1_text)


def test_to_pieces_page_odem():
    """Ensure PAGE 2019 straight from OCR-D ODEM is usable"""

    ocr_path = './tests/resources/candidate/frk_page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.xml'

    # act
    page_piece = to_pieces(ocr_path)

    # assert
    # 3 regions
    assert len(page_piece.pieces) == 3
    # 23 lines
    assert len(page_piece.pieces[0].pieces) == 1
    # first line in region 3 has 7 words
    assert len(page_piece.pieces[2].pieces[0].pieces) == 7

    # first line textual content
    line1_text = page_piece.pieces[2].pieces[0].transcription
    assert line1_text == 'und erklaͤret die Schrift nicht nur al⸗'
    assert page_piece.pieces[2].pieces[0].transcription == line1_text

@pytest.mark.skip("disabled")
def test_is_in_basics():
    """Check basic behavior:
    * a piece shall contain all its pieces
    * a piece from different region shall not contain 
      pieces from other super-piece
    """

    ocr_path = './tests/resources/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.xml'

    # act
    page_piece = to_pieces(ocr_path)

    # assert
    piece_region1 = page_piece.pieces[0]
    piece_region3 = page_piece.pieces[2]

    # regions 1 and region 3 are contained in page_piece
    assert piece_region1 in page_piece
    assert piece_region3 in page_piece

    # region 1 is not in region 3 and vice versa
    assert piece_region1 not in piece_region3
    assert piece_region3 not in piece_region1

@pytest.mark.skip("disabled")
def test_dimensions_alto_and_contains_relation():
    """Check basic dimensions
    * an ALTO page piece spans complete PrintSpace
    * a region piece has also dimension and is contained in page
    * a line piece has dimensions and is contained in region
    * a word piece has dimensions and is contained in line + region
    """

    ocr_path = './tests/resources/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml'

    # act
    page_piece = to_pieces(ocr_path)

    # assert
    assert page_piece.dimensions == [
        [0, 0], [6633, 0], [6633, 9944], [0, 9944]]
    region1 = page_piece.pieces[0]
    line1 = region1.pieces[0]
    word1 = line1.pieces[0]

    assert region1.dimensions == [
        [2100, 802], [3199, 802], [3199, 925], [2100, 925]]
    assert line1.dimensions == [[2100, 802], [
        3199, 802], [3199, 925], [2100, 925]]
    # even in this case line1 covers region1 completely
    assert region1.dimensions == line1.dimensions
    assert word1.dimensions == [[2101, 802], [
        2545, 802], [2545, 920], [2101, 920]]


def test_pieces_type_with_contains_relation():
    """Ensure PieceTypes and contains relations"""

    ocr_path = join(TEST_RES_DIR, 'groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')

    # act
    page_piece = to_pieces(ocr_path)
    region1 = page_piece.pieces[0]
    line1 = region1.pieces[0]
    word1 = line1.pieces[0]

    # assert
    assert region1 in page_piece
    assert region1.type == PieceType.REGION
    assert line1 in region1
    assert line1.type == PieceType.LINE
    assert word1 in line1 and word1 in region1
    assert word1.type == PieceType.WORD
