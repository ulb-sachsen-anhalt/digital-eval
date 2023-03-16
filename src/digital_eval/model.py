# -*- coding: utf-8 -*-
"""Model Module"""
from __future__ import annotations

import xml.dom.minidom
import xml.dom.minidom as md
from copy import copy
from enum import (
    Enum
)
from pathlib import PurePath
from typing import (
    List, Optional, Dict, Tuple,
)
from xml.dom import pulldom

from shapely.geometry import (
    Polygon
)

PAGE_2013 = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
XML_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#',
          'pg2013': PAGE_2013}

# mark information as 'not available'
# which *might* be set later on
UNSET: str = 'n.a.'


class PieceLevel(Enum):
    # more hierarchically
    UNKNOWN = 0
    GLYPH = 1
    WORD = 2
    LINE = 3
    TABLE_CELL = 4
    REGION = 5
    TABLE = 6
    PAGE = 7
    SECTION = 8

    def __lt__(self, other_lvl):
        if not isinstance(other_lvl, PieceLevel):
            return False
        return self.value < other_lvl.value

    def __gt__(self, other_lvl):
        if not isinstance(other_lvl, PieceLevel):
            return False
        return self.value > other_lvl.value

    def __eq__(self, other_lvl):
        if not isinstance(other_lvl, PieceLevel):
            return False
        return self.value == other_lvl.value


class PieceContent(Enum):
    # more layout
    UNKNOWN = 0
    PARAGRAPH = 1
    HEADING = 2
    COLUMN = 3
    TABLE = 4
    # more semantically
    IMAGE = 5
    ARTICLE = 6
    ANNOUNCEMENT = 7
    ADVERTISEMENT = 8


class PieceTranscription:

    def __init__(self):
        self.language = UNSET
        self.text = ''
        self.confidence = 0.0


class PieceData:

    def __init__(self):
        self.mime_type = UNSET
        self.data = None


class OcrFileFormat(Enum):
    UNKNOWN = 0
    ALTO_V3 = 1
    PAGE = 2


PieceDimensions = List[List[float]]


class Piece:
    """Piece base composition for analytical purposes"""

    def __init__(self, identifier: str = UNSET, xml_element: md.Element = None, document: md.Document = None):
        self.id: str = identifier
        self.level: PieceLevel = PieceLevel.PAGE
        self.subject: PieceContent = PieceContent.UNKNOWN
        self.data: PieceData = Optional[None]
        self.parent: Piece = Optional[None]
        self.custom: Dict = {}
        self._transcriptions: List = []
        self.__file_path: Optional[PurePath] = None
        self.__dimensions: PieceDimensions = []
        self.__pieces: List[Piece] = []
        self.__xml_element: Optional[md.Element] = xml_element
        self.__document: Optional[md.Document] = document
        self.__ocr_file_format: Optional[OcrFileFormat] = None

    @property
    def ocr_file_format(self) -> OcrFileFormat:
        return self.__ocr_file_format

    @ocr_file_format.setter
    def ocr_file_format(self, off: OcrFileFormat) -> None:
        self.__ocr_file_format = off
        for child in self.pieces:
            child.ocr_file_format = off

    @property
    def dimensions(self) -> PieceDimensions:
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dims: PieceDimensions) -> None:
        self.__dimensions = dims

    @property
    def xml_element(self) -> md.Element:
        return self.__xml_element

    @property
    def document(self) -> md.Document:
        return self.__document

    @document.setter
    def document(self, doc: md.Document) -> None:
        self.__document = doc
        for child in self.pieces:
            child.document = doc

    @property
    def file_path(self) -> PurePath:
        return self.__file_path

    @file_path.setter
    def file_path(self, fp: PurePath) -> None:
        self.__file_path = fp
        for child in self.pieces:
            child.file_path = fp

    @property
    def pieces(self) -> List[Piece]:
        return copy(self.__pieces)

    @pieces.setter
    def pieces(self, pcs: List[Piece]) -> None:
        self.__pieces = pcs
        self.__pass_props()

    def add_pieces(self, *pieces: Piece) -> List[Piece]:
        self.__pieces.extend(*pieces)
        self.__pass_props()
        return self.pieces

    def remove_pieces(self, *pieces: Piece) -> List[Piece]:
        for piece in pieces:
            self.__pieces.remove(piece)
        return self.pieces

    @property
    def transcription(self) -> str:
        """Get textual content as sequential textual string,
        with the order corresponding to it's _previous and
        next properties.

        Text will come without sanitized linebreaks, but
        includes a whitespace between single lines and words.
        """
        if self._transcriptions:
            return self._transcriptions[0].text
        elif not self._transcriptions and self.__is_superstruct():
            return ' '.join([_p.transcription
                             for _p in self.pieces])
        raise RuntimeError(f"ID={self.id}: Can't get text_content for {self.id}!")

    @transcription.setter
    def transcription(self, transcription: str) -> None:
        """Set textual transcription representing this piece"""
        _transcription = PieceTranscription()
        if transcription is not None and len(transcription.strip()) > 0:
            _transcription.text = transcription
        self._transcriptions.append(_transcription)

    def __repr__(self) -> str:
        return f"{self.id}:{self.transcription}"

    def __contains__(self, other_piece) -> bool:
        """Test for membership of other_piece
        Precondition: other_piece is logical 
            child/anchestor of current piece.level
        Calculate hull for self and centroid for
        other_pieces (to catch corner cases, too)
        """

        if not self.dimensions:
            raise RuntimeError(f"ID={self.id}: self has invalid dimensions!")
        if not other_piece.dimensions:
            raise RuntimeError(f"{other_piece.id}: other has invalid dimensions!")
        # check order invariant
        if self.level < other_piece.level or self.level == other_piece.level:
            raise RuntimeError(f"other {other_piece.id} is higher/equal level than {self.id}!")
        # go for centriod for real life 
        # cases where word bounds
        # scratch over region borders
        self_hull = Polygon(self.dimensions).convex_hull
        other_shape = Polygon(other_piece.dimensions)
        return self_hull.contains(other_shape.centroid)

    def __is_superstruct(self):
        return self.level in [
            PieceLevel.PAGE,
            PieceLevel.REGION,
            PieceLevel.LINE,
            PieceLevel.TABLE,
            PieceLevel.TABLE_CELL,
        ]

    def __pass_props(self):
        for child in self.__pieces:
            child.document = self.document
            child.file_path = self.file_path
            child.ocr_file_format = self.ocr_file_format


class PieceUtil:

    @staticmethod
    def to_pieces(path_in: str) -> Piece:
        """Transform given input in various formats
        into internal Piece-Representation"""
        return PieceUtil.__read_data(path_in)

    @staticmethod
    def __read_data(path_in: str) -> Piece:
        try:
            document: md.Document = md.parse(path_in)
            doc_root: md.Element = document.documentElement
        except Exception as _exc:
            raise RuntimeError(f"corrupt XML '{path_in}!")
        if doc_root is None:
            raise RuntimeError('invalid document root')
        name_space = doc_root.getAttribute('xmlns')
        piece: Optional[Piece]
        ocr_file_format: OcrFileFormat = OcrFileFormat.UNKNOWN
        if doc_root.localName == 'alto':
            piece = PieceAltoV3Util.extract_data(doc_root)
            ocr_file_format = OcrFileFormat.ALTO_V3
        elif name_space == PAGE_2013:
            piece = PiecePageUtil.extract_data(doc_root)
            ocr_file_format = OcrFileFormat.PAGE
        elif doc_root.localName == 'PcGts':
            piece = PiecePageUtil.extract_data(doc_root, ns='pc:')
            ocr_file_format = OcrFileFormat.PAGE
        else:
            raise RuntimeError(
                'Unknown Data-Format "{}" in "{}"'.format(doc_root.localName, path_in))
        piece.file_path = PurePath(path_in)
        piece.document = document
        piece.ocr_file_format = ocr_file_format

        return piece


class PieceAltoV3Util:
    @staticmethod
    def extract_data(doc_root) -> Piece:
        page_one: md.Element = doc_root.getElementsByTagName('Page')[0]
        _page_width = int(page_one.getAttribute('WIDTH'))
        _page_height = int(page_one.getAttribute('HEIGHT'))
        _dimensions = [[0, 0], [_page_width, 0], [_page_width, _page_height], [0, _page_height]]
        top_piece: Piece = Piece(page_one.getAttribute('ID'), page_one)
        top_piece.dimensions = _dimensions
        top_piece.level = PieceLevel.PAGE
        top_piece.subject = PieceAltoV3Util.__get_piece_subject(doc_root)
        # composed level
        _block_pieces = []
        comp_blocks = doc_root.getElementsByTagName('ComposedBlock')
        if len(comp_blocks) > 0:
            for _comp_block in comp_blocks:
                comp_piece: Piece = Piece(_comp_block.getAttribute('ID'), _comp_block)
                comp_piece.level = PieceLevel.REGION
                comp_piece.parent = top_piece
                comp_piece.dimensions = PieceAltoV3Util.__extract_dimensions(_comp_block)
                text_blocks = _comp_block.getElementsByTagName('TextBlock')
                if len(text_blocks) < 1:
                    raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
                comp_piece.pieces = PieceAltoV3Util.__read_blocks(text_blocks, comp_piece)
                _block_pieces.append(comp_piece)
        else:
            text_blocks = doc_root.getElementsByTagName('TextBlock')
            if len(text_blocks) < 1:
                raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
            _block_pieces = PieceAltoV3Util.__read_blocks(text_blocks, top_piece)
        top_piece.pieces = _block_pieces
        _all_points = [point for _block in _block_pieces for point in _block.dimensions]
        top_piece.dimensions = _all_points
        return top_piece

    @staticmethod
    def __read_blocks(block_elements, parent):
        _block_pieces = []
        for _block in block_elements:
            _block_piece = Piece(_block.getAttribute('ID'), _block)
            _block_piece.level = PieceLevel.REGION
            _lines = _block.getElementsByTagName('TextLine')
            if len(_lines) == 0:
                raise RuntimeError(f"TextBlock@ID={_block_piece.id} contains no lines!")
            _block_piece.parent = parent
            _block_piece.pieces = PieceAltoV3Util.__read_lines(_lines, _block_piece)
            _block_piece.dimensions = PieceAltoV3Util.__extract_dimensions(_block)
            _block_pieces.append(_block_piece)
        return _block_pieces

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
            line_piece = Piece(_id, _text_line)
            line_piece.level = PieceLevel.LINE
            text_strings = _text_line.getElementsByTagName('String')
            if len(text_strings) < 1:
                raise RuntimeError(f"No words in line {_id}!")
            line_piece.pieces = PieceAltoV3Util.__read_words(text_strings, line_piece)
            line_piece.parent = parent
            line_piece.dimensions = PieceAltoV3Util.__extract_dimensions(_text_line)
            _lines.append(line_piece)
        return _lines

    @staticmethod
    def __read_words(text_strings, parent):
        _words = []
        for _text_string in text_strings:
            _id = _text_string.getAttribute('ID')
            word_piece = Piece(_id, _text_string)
            word_piece.level = PieceLevel.WORD
            _content = _text_string.getAttribute('CONTENT')
            if not _content.strip():
                continue
            word_piece.transcription = _content
            word_piece.dimensions = PieceAltoV3Util.__extract_dimensions(_text_string)
            word_piece.parent = parent
            _words.append(word_piece)
        return _words

    @staticmethod
    def __extract_dimensions(el, prefer_box=True):
        if not prefer_box:
            _shape = [n for n in el.getChildren() if n.localName == 'Shape']
            if len(_shape) == 1:
                # TODO: handle ALTO Shape
                pass
        else:
            _left = int(el.getAttribute('HPOS'))
            _top = int(el.getAttribute('VPOS'))
            _height = int(el.getAttribute('HEIGHT'))
            _width = int(el.getAttribute('WIDTH'))
            return [[_left, _top], [_left + _width, _top],
                    [_left + _width, _top + _height], [_left, _top + _height]]
        raise RuntimeError(f"{el.localName}@ID={el.getAttribute('ID')}: Can't calculate dimensions")


class PiecePageUtil:
    @staticmethod
    def extract_data(doc_root, ns='') -> Piece:
        page_one = doc_root.getElementsByTagName(ns + 'Page')[0]
        page_width = int(page_one.getAttribute('imageWidth'))
        page_height = int(page_one.getAttribute('imageHeight'))
        top_piece = Piece(page_one.getAttribute('imageFilename'), page_one)
        top_piece.level = PieceLevel.PAGE
        top_piece.dimensions = [[0, 0], [page_width, 0],
                                [page_width, page_height], [0, page_height]]
        regions = doc_root.getElementsByTagName(ns + 'TextRegion')
        regions.extend(doc_root.getElementsByTagName(ns + 'TableCell'))
        # no regions is considered to be reasonable
        # don't raise exception, it's an empty page
        if len(regions) < 1:
            return top_piece

        # inspect *all* regions
        region_pieces = []
        for region in regions:
            _piece = PiecePageUtil.__from_text_element(region, top_piece, ns)
            # go into details
            page_lines = region.getElementsByTagName(ns + 'TextLine')
            if len(page_lines) > 0:
                _piece.pieces = PiecePageUtil.__read_lines(page_lines, _piece, ns)
            _piece.parent = top_piece
            region_pieces.append(_piece)
        top_piece.pieces = region_pieces
        _all_points = [point for reg in region_pieces for point in reg.dimensions]
        top_piece.dimensions = _all_points
        return top_piece

    @staticmethod
    def __read_lines(page_lines, parent, ns) -> List[Piece]:
        line_pieces = []
        for page_line in page_lines:
            line_piece = PiecePageUtil.__from_text_element(page_line, parent, ns)
            word_tokens = page_line.getElementsByTagName(ns + 'Word')
            line_piece.parent = parent
            # inspect PAGE on word level, if set
            if len(word_tokens) > 0:
                word_pieces = [PiecePageUtil.__from_text_element(el, line_piece, ns) for el in word_tokens]
                if not word_pieces:
                    raise RuntimeError(f"No words in line {line_piece.id}!")
                # remove line content in favour of words content
                # line_piece.content = None
                line_piece.pieces = word_pieces
            line_pieces.append(line_piece)
        return line_pieces

    @staticmethod
    def __from_text_element(element, parent, ns) -> Piece:
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
        _type, _local = PiecePageUtil.__map_piece_type(element)
        _piece = Piece(_id, element)
        _piece.level = _type
        _piece.parent = parent

        # inspect geometry
        _coords = [n for n in element.childNodes if n.localName == 'Coords']
        if len(_coords) < 1 or 'points' not in _coords[0].attributes:
            raise RuntimeError(f"{_local}@ID={_id} invalid coordinate data")
        _points = _coords[0].getAttribute('points').split()
        # invariant: at least want 3 points, otherwise polygon area == Zero
        if len(_points) < 3:
            raise RuntimeError(f"{_local}@ID={_id} way too few points {_points}")
        _piece.dimensions = [[int(_point.split(',')[0]), int(_point.split(',')[1])]
                             for _point in _points]

        # pick text if on word level
        if _type == PieceLevel.WORD:
            _txt_eqs = [n for n in element.childNodes if n.localName == 'TextEquiv']
            if len(_txt_eqs) != 1:
                raise RuntimeError(f"{_local}@ID={_id} invalid txt node {_txt_eqs}")
            _first_unicode = _txt_eqs[0].getElementsByTagName(ns + 'Unicode')[0]
            if not _first_unicode.firstChild:
                raise RuntimeError(f"{_local}@ID={_id} empty unicode node!")
            _content = _first_unicode.firstChild.nodeValue
            if not _content or not _content.strip():
                raise RuntimeError(f"{_local}@ID={_id} invalid txt content!")
            _piece.transcription = _content

        return _piece

    @staticmethod
    def __map_piece_type(element) -> Tuple[PieceLevel, str]:
        _local = element.localName
        _name = PieceLevel.UNKNOWN
        if _local == 'Word':
            _name = PieceLevel.WORD
        elif _local == 'TextLine':
            _name = PieceLevel.LINE
        elif _local == 'TextRegion':
            _name = PieceLevel.REGION
        elif _local == 'TableRegion':
            _name = PieceLevel.TABLE
        elif _local == 'TableCell':
            _name = PieceLevel.TABLE_CELL
        return _name, _local
