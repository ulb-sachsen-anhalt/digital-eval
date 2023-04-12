# handle possibly missing tif data
import os
import shutil
import xml.dom.minidom as md
from pathlib import Path, PurePath
from typing import Dict, List

import pytest
from shapely import Polygon

from digital_eval import Piece
from digital_eval.model import from_pieces, PieceUtil
from ocr_util import PolygonFrameFilterUtil, PolygonFrameFilter

RES_ROOT = os.path.join('tests', 'resources', 'frames')
RES_ALTO = os.path.join(RES_ROOT, 'alto')


@pytest.fixture(name='xml_fixture')
def _create_filter_fixture(tmp_path) -> Dict[str, str]:
    tmp_sub = tmp_path / 'xml'
    tmp_sub.mkdir()
    path_0768 = tmp_sub / '1667522809_J_0001_0768.xml'
    shutil.copyfile(os.path.join(RES_ALTO, '1667522809_J_0001_0768.xml'), path_0768)
    path_0768_2022 = tmp_sub / '1667522809_J_0001_0768.2022.xml'
    shutil.copyfile(os.path.join(RES_ALTO, '1667522809_J_0001_0768.2022.xml'), path_0768_2022)
    path_0260 = tmp_sub / '1681877805_J_0001_0260.xml'
    shutil.copyfile(os.path.join(RES_ALTO, '1681877805_J_0001_0260.xml'), path_0260)
    path_0512 = tmp_sub / '1667524704_J_0125_0512.xml'
    shutil.copyfile(os.path.join(RES_ALTO, '1667524704_J_0125_0512.gt.xml'), path_0512)
    path_1208 = tmp_sub / '1667524704_J_0125_1208.gt.xml'
    shutil.copyfile(os.path.join(RES_ALTO, '1667524704_J_0125_1208.gt.xml'), path_1208)

    return {'0768': str(path_0768),
            '0768_22': str(path_0768_2022),
            '0260': str(path_0260),
            '0512': str(path_0512),
            '1208': str(path_1208)}


def test_poly(xml_fixture):
    # arrange
    points: str = '550,700 2700,700 2700,4350 550,4350'
    alto_in_path: str = xml_fixture['0768']
    filter_ocr = PolygonFrameFilter(alto_in_path, points)

    # act
    piece_result: Piece = filter_ocr.process()

    # assert
    assert piece_result
    assert isinstance(piece_result, Piece)
    assert isinstance(filter_ocr.polygon, Polygon)
    assert isinstance(filter_ocr.ocr_file_path, Path)

    # assert
    poly: Polygon = PolygonFrameFilterUtil.str_to_polygon(points)
    pieces: List[Piece] = PieceUtil.flatten(piece_result)
    pieces.remove(piece_result)
    for piece in pieces:
        assert piece.is_in_polygon(poly)


def test_poly_legacy(xml_fixture):
    # arrange
    points: str = '550,700 2700,4350'
    alto_in_path: str = xml_fixture['0768']
    filter_ocr = PolygonFrameFilter(alto_in_path, points)

    # act
    piece_result: Piece = filter_ocr.process()

    # assert
    assert piece_result
    assert isinstance(piece_result, Piece)
    assert isinstance(filter_ocr.polygon, Polygon)
    assert isinstance(filter_ocr.ocr_file_path, Path)

    # assert
    poly: Polygon = PolygonFrameFilterUtil.str_to_polygon(points)
    pieces: List[Piece] = PieceUtil.flatten(piece_result)
    pieces.remove(piece_result)
    for piece in pieces:
        assert piece.is_in_polygon(poly)

def test_filter_0001_0768_2020(xml_fixture):
    """Check result file exists and contains CONTENT"""

    # arrange
    points: str = '550,700 2700,700 2700,4350 550,4350'
    alto_in_path: str = xml_fixture['0768']
    filter_ocr = PolygonFrameFilter(alto_in_path, points)

    # act
    piece_result: Piece = filter_ocr.process()
    file_out_path: PurePath = from_pieces(piece_result)

    # assert
    assert os.path.exists(file_out_path)
    path_result_expect: str = os.path.join(
        os.path.dirname(xml_fixture['0768']),
        '1667522809_J_0001_0768.gt.xml')
    assert path_result_expect == str(file_out_path)

    # assert
    tmp_xml: md.Document = md.parse(str(file_out_path))
    string_elements = tmp_xml.getElementsByTagName('String')
    assert len(string_elements) < 7744
    for string_element in string_elements:
        assert string_element.tagName == 'String'
        assert string_element.getAttribute('CONTENT') != ''


def test_filter_0001_0768_2022(xml_fixture):
    """Change in format: now unwrap from ContentBlocks first"""

    # arrange
    points: str = '525,825 2725,825 2725,7125 525,7125'
    alto_in_path: str = xml_fixture['0768_22']
    filter_ocr = PolygonFrameFilter(alto_in_path, points)

    # act
    piece_result: Piece = filter_ocr.process()
    file_out_path: PurePath = from_pieces(piece_result)

    # assert
    assert os.path.exists(file_out_path)
    path_result_expect: str = os.path.join(
        os.path.dirname(
            xml_fixture['0768_22']),
        '1667522809_J_0001_0768.2022.gt.xml')
    assert str(file_out_path) == path_result_expect

    # assert
    tmp_xml: md.Document = md.parse(str(file_out_path))
    string_elements = tmp_xml.getElementsByTagName('String')
    assert len(string_elements) > 960
    for string_element in string_elements:
        assert string_element.tagName == 'String'
        assert string_element.getAttribute('CONTENT') != ''


def test_filter_0001_0260(xml_fixture):
    """check filtered structure from new tesseract ALTO format with ComposedBlock"""

    # arrange
    points: str = '550,700 2700,700 2700,4350 550,4350'
    alto_in_path: str = xml_fixture['0260']
    filter_ocr = PolygonFrameFilter(alto_in_path, points)

    # act
    piece_result: Piece = filter_ocr.process()
    file_out_path: PurePath = from_pieces(piece_result)

    # assert
    assert os.path.exists(file_out_path)
    tmp_xml: md.Document = md.parse(str(file_out_path))
    text_blocks = tmp_xml.getElementsByTagName('TextBlock')
    assert len(text_blocks) == 1
    assert len(tmp_xml.getElementsByTagName('ComposedBlock')) == 1

    # ensure proper double quotes - not direct possible, but see if it gets
    # parsed
    assert len(text_blocks[0].getElementsByTagName('String')) == 98
