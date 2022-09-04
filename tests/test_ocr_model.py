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

from digital_eval.model import (
    to_pieces,
    PieceStage,
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
    '''check that gt-type "announcement" can be extracted from file with "annx" in name'''

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

# @pytest.mark.skip("disabled")
@pytest.fixture(name='odem01', scope='module')
def _fixture_odem01():
    """Check basic behavior:
    * a piece shall contain all its pieces
    * a piece from different region shall not contain 
      pieces from other super-piece
    """

    ocr_path = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml'
    page_piece = to_pieces(ocr_path)
    yield page_piece


def test_pieces_odem01_region_and_page(odem01):
    """basics: sub-piece one is contained
    in super-piece of total page"""

    # we have only one sub-piece
    # a single text region
    assert len(odem01.pieces) == 1
    piece_region1 = odem01.pieces[0]

    # region 1 contained in page_piece
    assert piece_region1 in odem01


@pytest.mark.skip(reason='needs discussion')
def test_pieces_odem01_page_not_in_region(odem01):
    """Evidently can't a super-structure
    *not* be contained in it's own child"""
    
    # top level pageregion 1 is not in region 3 and vice versa
    assert odem01 not in odem01.pieces[0]


@pytest.fixture(name="zd101")
def _fixture_zd101():
    ocr_path = './tests/resources/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml'
    page_piece = to_pieces(ocr_path)
    yield page_piece


def test_pieces_zd101_page_piece_dimension(zd101):
    """Check ALTO page piece spans only space
    from all child pieces, i.e. the regions
    """

    # explore dimensions
    # 10 text_regions, 4 corners => 40 points
    assert len(zd101.dimensions) == 40
    # top left 1st region
    assert [802, 2100] in zd101.dimensions
    # top left 2nd region
    assert [403, 2252] in zd101.dimensions
    # bottom right latest region
    assert [2313, 9545] in zd101.dimensions


def test_pieces_zd101_page_bounding_box_dimension(zd101):
    """check if bounding box is reasonable
    for single column
    
    * top_left: 401,2100
    * bottom_right: 2380, 9545
    """

    _polygon = Polygon(zd101.dimensions)

    # all regions contained in this box
    assert _polygon.bounds == (401.0, 2100.0, 2380.0, 9545.0)


def test_pieces_zd101_region01_dimensions(zd101):
    """
    * a region piece has also dimension and is contained in page
    * a line piece has dimensions and is contained in region
    * a word piece has dimensions and is contained in line + region
    """

    assert len(zd101.pieces) == 10
    region1 = zd101.pieces[0]
    line1 = region1.pieces[0]
    assert region1.dimensions == [
        [802, 2100], [1901,2100], [1901,2223], [802,2223]]

    # this region has only one line
    assert len(region1.pieces) == 1
    # which is completely same as region
    assert line1.dimensions == [[802, 2100], 
        [1901,2100], [1901, 2223], [802,2223]]
    assert region1.dimensions == line1.dimensions
    # this line has two words
    assert len(line1.pieces) == 2
    word1 = line1.pieces[0]
    # coords for word01 differ sligthly from line
    assert word1.dimensions == [[802, 2101], 
        [1246, 2101], [1246, 2219], [802, 2219]]
    assert word1 in line1 and line1 in region1


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
    assert region1.type == PieceStage.REGION
    assert line1 in region1
    assert line1.type == PieceStage.LINE
    assert word1 in line1 and word1 in region1
    assert word1.type == PieceStage.WORD
