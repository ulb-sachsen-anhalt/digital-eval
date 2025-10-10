"""Test specification for representation
of digital assets in OCR PAGE format
"""

import xml.dom.minidom
import xml.parsers.expat
import xml.etree.ElementTree as ET_std

import lxml.etree as ET_lxml

import pytest

from tests.conftest import TEST_RES_DIR


def test_xml_parser_minidom():
    """Ensure PAGE data from OCR4all groundtruth readable
    and respects different text-equiv elements, created during
    the annotation process. Also verifies that reading order is respected.
    """

    # arrange
    ocr_path = f'{TEST_RES_DIR}/candidate/page/1744746265_19330115.xml'

    # act
    with pytest.raises(xml.parsers.expat.ExpatError) as _err:
        xml.dom.minidom.parse(ocr_path)

    # assert
    assert _err.value.args[0] == 'not well-formed (invalid token): line 73, column 33'


def test_xml_parser_std_etree():
    """Compare different XML parsers for handling historical German text"""

    ocr_path = f'{TEST_RES_DIR}/candidate/page/1744746265_19330115.xml'

    # act
    with pytest.raises(ET_std.ParseError) as _err:
        ET_std.parse(ocr_path)

    # assert
    assert _err.value.args[0] == 'not well-formed (invalid token): line 73, column 33'


def test_xml_parser_lxml_etree():
    """Compare different XML parsers for handling historical German text"""

    ocr_path = f'{TEST_RES_DIR}/candidate/page/1744746265_19330115.xml'

    # act
    doc_tree = ET_lxml.parse(ocr_path)

    # assert
    assert doc_tree is not None
