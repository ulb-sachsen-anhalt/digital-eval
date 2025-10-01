"""format_alto_v3_util module"""
from pathlib import PurePath
from xml.dom.minidom import Document, Element, parse

from digital_eval.model.digital_object_model import DigitalObjectTree
from digital_eval.model.common import DigitalObjectLevel, DigitalObjectTreeOCRFileFormat, UNSET


class FormatAltoV3Util:
    """helper methods for parsing Alto V3 XML files"""

    @staticmethod
    def extract_data(path: str) -> DigitalObjectTree:
        document: Document = parse(path)
        doc_root: Element = document.documentElement
        page_one: Element = doc_root.getElementsByTagName('Page')[0]
        _page_width = int(page_one.getAttribute('WIDTH'))
        _page_height = int(page_one.getAttribute('HEIGHT'))
        _dimensions = [[0, 0], [_page_width, 0], [_page_width, _page_height], [0, _page_height]]
        top_digo: DigitalObjectTree = DigitalObjectTree(
            page_one.getAttribute('ID'),
            page_one,
            document=document,
            file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3,
            file_path=PurePath(path)
        )
        top_digo.dimensions = _dimensions
        top_digo.level = DigitalObjectLevel.PAGE
        top_digo.subject = FormatAltoV3Util.__get_piece_subject(doc_root)
        # composed level
        _block_digos = []
        comp_blocks = doc_root.getElementsByTagName('ComposedBlock')
        if len(comp_blocks) > 0:
            for _comp_block in comp_blocks:
                comp_digo: DigitalObjectTree = DigitalObjectTree(
                    _comp_block.getAttribute('ID'),
                    _comp_block,
                    file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3
                )
                comp_digo.level = DigitalObjectLevel.REGION
                comp_digo.parent = top_digo
                comp_digo.dimensions = FormatAltoV3Util.__extract_dimensions(_comp_block)
                text_blocks = _comp_block.getElementsByTagName('TextBlock')
                if len(text_blocks) < 1:
                    raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
                comp_digo.children = FormatAltoV3Util.__read_blocks(text_blocks, comp_digo)
                _block_digos.append(comp_digo)
        else:
            text_blocks = doc_root.getElementsByTagName('TextBlock')
            if len(text_blocks) < 1:
                raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
            _block_digos = FormatAltoV3Util.__read_blocks(text_blocks, top_digo)
        top_digo.children = _block_digos
        return top_digo

    @staticmethod
    def __read_blocks(block_elements, parent):
        _blocks = []
        for _block_el in block_elements:
            _block = DigitalObjectTree(
                _block_el.getAttribute('ID'),
                _block_el,
                file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3
            )
            _block.level = DigitalObjectLevel.REGION
            _lines = _block_el.getElementsByTagName('TextLine')
            if len(_lines) == 0:
                raise RuntimeError(f"TextBlock@ID={_block.id} contains no lines!")
            _block.parent = parent
            _block.children = FormatAltoV3Util.__read_lines(_lines, _block)
            _block.dimensions = FormatAltoV3Util.__extract_dimensions(_block_el)
            _blocks.append(_block)
        return _blocks

    @staticmethod
    def __get_piece_subject(doc_root):
        gt_type_el = doc_root.getElementsByTagName('OtherTag')
        _subject = UNSET
        if gt_type_el and len(gt_type_el) > 0:
            # deprecated
            label = gt_type_el[0].getAttribute('LABEL')
            if label:
                _subject = label
            # new alto way
            else:
                gt_els = [e for e in gt_type_el if e.getAttribute(
                    'ID') == "ulb_groundtruth_type"]
                if len(gt_els) == 1:
                    value = gt_els[0].getAttribute('VALUE')
                    if value:
                        _subject = value
        return _subject

    @staticmethod
    def __read_lines(the_lines, parent):
        _lines = []
        for _text_line in the_lines:
            _id = _text_line.getAttribute('ID')
            line = DigitalObjectTree(
                _id,
                _text_line,
                file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3
            )
            line.level = DigitalObjectLevel.LINE
            text_strings = _text_line.getElementsByTagName('String')
            if len(text_strings) < 1:
                raise RuntimeError(f"No words in line {_id}!")
            line.children = FormatAltoV3Util.__read_words(text_strings, line)
            line.parent = parent
            line.dimensions = FormatAltoV3Util.__extract_dimensions(_text_line)
            _lines.append(line)
        return _lines

    @staticmethod
    def __read_words(text_strings, parent):
        _words = []
        for _text_string in text_strings:
            _id = _text_string.getAttribute('ID')
            word = DigitalObjectTree(
                _id,
                _text_string,
                file_format=DigitalObjectTreeOCRFileFormat.ALTO_V3
            )
            word.level = DigitalObjectLevel.WORD
            _content = _text_string.getAttribute('CONTENT')
            if not _content.strip():
                continue
            word.transcription = _content
            word.dimensions = FormatAltoV3Util.__extract_dimensions(_text_string)
            word.parent = parent
            _words.append(word)
        return _words

    # does not repspect ALTO shapes
    @staticmethod
    def __extract_dimensions(el):
        _left = int(el.getAttribute('HPOS'))
        _top = int(el.getAttribute('VPOS'))
        _height = int(el.getAttribute('HEIGHT'))
        _width = int(el.getAttribute('WIDTH'))
        return [[_left, _top], [_left + _width, _top],
                [_left + _width, _top + _height], [_left, _top + _height]]
