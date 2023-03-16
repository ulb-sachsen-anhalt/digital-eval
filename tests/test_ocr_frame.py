# handle possibly missing tif data
import os
import shutil
import xml.dom.minidom as md
from pathlib import Path
from typing import Dict

import pytest
from shapely import Polygon

from digital_eval import Piece
from src.ocr_util import FrameFilterAltoV3, Point2D, PolygonFrameFilter

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


def test_filter_0001_0768_2020(xml_fixture):
    """Check result file exists and contains CONTENT"""

    # arrange
    # filter_ocr = OCRQAFilter(xml_fixture['0768'], "550x700", "2700x4350")
    filter_ocr = FrameFilterAltoV3(xml_fixture['0768'], [Point2D(550, 700), Point2D(2700, 4350)])

    # act
    filter_ocr.process()
    path_result = filter_ocr.get_out_path()

    # assert
    assert os.path.exists(path_result)
    path_result_expect = os.path.join(
        os.path.dirname(
            xml_fixture['0768']),
        '1667522809_J_0001_0768.gt.alto.xml')
    assert path_result_expect == path_result

    # assert
    tmp_data = md.parse(path_result)
    string_elements = tmp_data.getElementsByTagName('String')
    assert len(string_elements) > 1100
    for string_element in string_elements:
        assert string_element.tagName == 'String'
        assert string_element.getAttribute('CONTENT') != ''


def test_filter_0001_0768_2022(xml_fixture):
    """Change in format: now unwrap from ContentBlocks first"""

    # arrange
    # filter_ocr = OCRQAFilter(xml_fixture['0768_22'], "525x825", "2725x7125")
    filter_ocr = FrameFilterAltoV3(xml_fixture['0768_22'], [Point2D(525, 825), Point2D(2725, 7125)])

    # act
    filter_ocr.process()
    path_result = filter_ocr.get_out_path()

    # assert
    assert os.path.exists(path_result)
    path_result_expect = os.path.join(
        os.path.dirname(
            xml_fixture['0768']),
        '1667522809_J_0001_0768.gt.alto.xml')
    assert path_result_expect == path_result

    # assert
    tmp_data = md.parse(path_result)
    string_elements = tmp_data.getElementsByTagName('String')
    assert len(string_elements) > 960
    for string_element in string_elements:
        assert string_element.tagName == 'String'
        assert string_element.getAttribute('CONTENT') != ''


def test_filter_0001_0260(xml_fixture):
    """check filtered structure from new tesseract ALTO format with ComposedBlock"""

    # arrange
    # filter_ocr = OCRQAFilter(xml_fixture['0260'], "550x700", "2700x4350")
    filter_ocr = FrameFilterAltoV3(xml_fixture['0260'], [Point2D(550, 700), Point2D(2700, 4350)])

    # act
    filter_ocr.process()
    path_result = filter_ocr.get_out_path()

    # assert
    assert os.path.exists(path_result)
    tmp_data = md.parse(path_result)
    text_blocks = tmp_data.getElementsByTagName('TextBlock')
    assert len(text_blocks) == 1
    assert not tmp_data.getElementsByTagName('ComposedBlock')

    # ensure proper double quotes - not direct possible, but see if it gets
    # parsed
    assert len(text_blocks[0].getElementsByTagName('String')) == 84


def test_polygon_frame_filter(xml_fixture):
    points: str = '550,700 2700,700 2700,4350 550,4350'
    alto_in_path: str = xml_fixture['0768']
    # alto_out_path: str = alto_in_path + '.frame.xml'
    filter_ocr: PolygonFrameFilter = PolygonFrameFilter(alto_in_path, points)

    # act
    piece: Piece = filter_ocr.process()
    print(piece.document.toprettyxml())

    assert isinstance(piece, Piece)
    assert isinstance(filter_ocr.polygon, Polygon)
    assert isinstance(filter_ocr.ocr_file_path, Path)
