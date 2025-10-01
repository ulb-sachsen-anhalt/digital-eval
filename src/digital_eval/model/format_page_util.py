"""format_page_util module"""
from pathlib import PurePath
from typing import List, Tuple
from xml.dom.minidom import Document, Element, parse

from digital_eval.model.digital_object_model import DigitalObjectTree
from digital_eval.model.common import DigitalObjectException, DigitalObjectLevel, \
    DigitalObjectTreeOCRFileFormat


class FormatPageUtil:
    """helper methods for parsing Page XML files"""

    @staticmethod
    def extract_data(path: str) -> DigitalObjectTree:
        document: Document = parse(path)
        doc_root: Element = document.documentElement
        ns = doc_root.namespaceURI
        pages = doc_root.getElementsByTagNameNS(ns, 'Page')
        if len(pages) != 1:
            raise DigitalObjectException(f"No unique page for {path}")
        page_one = pages[0]
        page_width = int(page_one.getAttribute('imageWidth'))
        page_height = int(page_one.getAttribute('imageHeight'))
        top_piece = DigitalObjectTree(
            page_one.getAttribute('imageFilename'),
            page_one,
            document=document,
            file_format=DigitalObjectTreeOCRFileFormat.PAGE,
            file_path=PurePath(path)
        )
        top_piece.level = DigitalObjectLevel.PAGE
        top_piece.dimensions = [[0, 0], [page_width, 0],
                                [page_width, page_height], [0, page_height]]
        regions = doc_root.getElementsByTagNameNS(ns, 'TextRegion')
        regions.extend(doc_root.getElementsByTagNameNS(ns, 'TableCell'))
        # no regions are considered to be reasonable
        # don't raise exception, it's an empty page
        if len(regions) < 1:
            return top_piece

        # inspect *all* regions
        region_pieces: List[DigitalObjectTree] = []
        for region in regions:
            _piece = FormatPageUtil.__from_text_element(region, top_piece, ns)
            # go into details
            page_lines = region.getElementsByTagNameNS(ns, 'TextLine')
            if len(page_lines) > 0:
                _piece.children = FormatPageUtil.__read_lines(page_lines, _piece, ns)
            _piece.parent = top_piece
            region_pieces.append(_piece)
        top_piece.children = region_pieces
        _all_points = [point for reg in region_pieces for point in reg.dimensions]
        top_piece.dimensions = _all_points
        return top_piece

    @staticmethod
    def __read_lines(page_lines: List[Element], parent, ns) -> List[DigitalObjectTree]:
        line_pieces = []
        for page_line in page_lines:
            line_piece = FormatPageUtil.__from_text_element(page_line, parent, ns)
            word_tokens = page_line.getElementsByTagNameNS(ns, 'Word')
            line_piece.parent = parent
            # inspect PAGE on word level, if set
            if len(word_tokens) > 0:
                try:
                    word_pieces = [FormatPageUtil.__from_text_element(el, line_piece, ns)
                                   for el in word_tokens]
                    if not word_pieces:
                        raise RuntimeError(f"No words in line {line_piece.id}!")
                    # remove line content in favour of words content
                    line_piece.transcriptions = []
                    line_piece.children = word_pieces
                except DigitalObjectException as _pex:
                    raise RuntimeError(_pex.args) from _pex
            line_pieces.append(line_piece)
        return line_pieces

    @staticmethod
    def __from_text_element(element:Element, parent, ns) -> DigitalObjectTree:
        """transformation from PAGE XML textual nodes
        to generic pieces with specific transkription.

        If on PAGE level Word creates word pieces, and
        also inspects textual contents.

        catches several *bad* data flavours regarding
        coordinates, points and text content

        * missing Coord node
        * Coord exists, but misses attribute "points"
        * Coord exists, attribute "points" exists, but
          contains less than 3 point-pairs, thous only
          forms a line and not even a triangle
        * missing TextEquiv node
        * TextEquiv exists, but no Unicode child
        * Unicode exists, but lacking any text content

        """
        _id = element.getAttribute('id')
        _type, _local = FormatPageUtil.__map_piece_type(element)
        _piece = DigitalObjectTree(
            _id,
            element,
            file_format=DigitalObjectTreeOCRFileFormat.PAGE
        )
        _piece.level = _type
        _piece.parent = parent

        # inspect geometry
        _coords = [n for n in element.childNodes if n.localName == 'Coords']
        if len(_coords) < 1 or 'points' not in _coords[0].attributes:
            raise DigitalObjectException(f"{_local}@ID={_id} invalid coordinate data")
        _points = _coords[0].getAttribute('points').split()
        # invariant: at least want 3 points, otherwise polygon area == Zero
        if len(_points) < 3:
            raise DigitalObjectException(f"{_local}@ID={_id} way too few points {_points}")
        try:
            _piece.dimensions = [[int(_point.split(',')[0]), int(_point.split(',')[1])]
                                 for _point in _points]
        except ValueError as _val_err:
            raise DigitalObjectException(f"{_local}@ID={_id} invalid points {_points}") from _val_err

        # replace current text with next order children text
        _txt_eqs = [n for n in element.childNodes if n.localName == 'TextEquiv']
        if _txt_eqs:
            _first_text: Element = _txt_eqs[0]
            _unicodes = _first_text.getElementsByTagNameNS(ns, 'Unicode')
            if len(_unicodes) < 1:
                raise DigitalObjectException(f"{_local}@ID={_id} text missing unicode")
            _first_unicode = _unicodes[0]
            if _first_unicode.firstChild:
                # replace linebreak if text only at region level
                _content = _first_unicode.firstChild.nodeValue.replace('\n', ' ')
                if _content:
                    _piece.transcription = _content
                    # overthrow existing parent transcription
                    if _piece.parent.transcriptions:
                        _piece.parent.transcriptions = []
        return _piece

    @staticmethod
    def __map_piece_type(element) -> Tuple[DigitalObjectLevel, str]:
        _local = element.localName
        _name = DigitalObjectLevel.UNKNOWN
        if _local == 'Word':
            _name = DigitalObjectLevel.WORD
        elif _local == 'TextLine':
            _name = DigitalObjectLevel.LINE
        elif _local == 'TextRegion':
            _name = DigitalObjectLevel.REGION
        elif _local == 'TableRegion':
            _name = DigitalObjectLevel.TABLE
        elif _local == 'TableCell':
            _name = DigitalObjectLevel.TABLE_CELL
        return _name, _local
