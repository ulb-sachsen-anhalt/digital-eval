# -*- coding: utf-8 -*-
"""Model Module"""
from __future__ import annotations

import xml.dom.minidom as md
from copy import copy
from enum import (
    Enum, IntEnum
)
from pathlib import PurePath
from typing import (
    List, Optional, Dict, Tuple, TextIO, Any,
)
from xml.dom.minidom import Element

from shapely.geometry import (
    Polygon
)

PAGE_2013 = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
XML_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#',
          'pg2013': PAGE_2013}

# mark information as 'not available'
# which *might* be set later on
UNSET: str = 'n.a.'


class PieceLevel(IntEnum):
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


class PieceException(Exception):
    """Mark custom Exception"""


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


class PieceOcrFileFormat(Enum):
    UNKNOWN = "UNKNOWN"
    ALTO_V3 = "ALTO_V3"
    PAGE = "PAGE"


PieceDimensions = List[List[float]]


class PieceChanges:
    removed_elements: List[Element] = []
    resized_elements: List[Element] = []


class Piece:
    """Piece base composition for analytical purposes"""

    def __init__(
            self,
            identifier: str = UNSET,
            xml_element: md.Element = None,
            document: md.Document = None,
            ocr_file_format: PieceOcrFileFormat = PieceOcrFileFormat.UNKNOWN
    ):
        self.id: str = identifier
        self.level: PieceLevel = PieceLevel.PAGE
        self.subject: PieceContent = PieceContent.UNKNOWN
        self.data: Optional[PieceData] = None
        self.parent: Optional[Piece] = None
        self.custom: Dict = {}
        self._transcriptions: List = []
        self.__file_path: Optional[PurePath] = None
        self.__dimensions: PieceDimensions = []
        self.__pieces: List[Piece] = []
        self.__xml_element: md.Element = xml_element
        self.__document: md.Document = document
        self.__ocr_file_format: PieceOcrFileFormat = ocr_file_format

    @property
    def ocr_file_format(self) -> PieceOcrFileFormat:
        return self.__ocr_file_format

    @ocr_file_format.setter
    def ocr_file_format(self, off: PieceOcrFileFormat) -> None:
        self.__ocr_file_format = off
        for child in self.pieces:
            child.ocr_file_format = off

    @property
    def dimensions(self) -> PieceDimensions:
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dims: PieceDimensions) -> None:
        self.__dimensions = dims
        self.__pass_properties_to_child_pieces()
        if self.__ocr_file_format != PieceOcrFileFormat.UNKNOWN:
            if self.__set_dimensions_in_xml(dims):
                PieceChanges.resized_elements.append(self.xml_element)

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
        self.__pass_properties_to_child_pieces()

    def add_pieces(self, *pieces: Piece) -> List[Piece]:
        self.__pieces.extend(*pieces)
        self.__pass_properties_to_child_pieces()
        return self.pieces

    def remove_pieces(self, *pieces: Piece) -> None:
        for piece in pieces:
            self.__pieces.remove(piece)
            element: Element = piece.xml_element
            removable_tags: List[str] = []
            if piece.ocr_file_format == PieceOcrFileFormat.ALTO_V3:
                removable_tags.append('SP')
            removed_elements: List[Element] = _MinidomUtil.remove_element_and_clear_parent(element, removable_tags)
            PieceChanges.removed_elements.extend(removed_elements)

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
    def transcription(self, transscr: str) -> None:
        """Set textual transcription representing this piece"""
        _transcription = PieceTranscription()
        if transscr is not None and len(transscr.strip()) > 0:
            _transcription.text = transscr
        self._transcriptions.append(_transcription)

    def is_in_polygon(self, poly: Polygon) -> bool:
        return PieceUtil.is_piece_in_polygon(self, poly)

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

    def __pass_properties_to_child_pieces(self):
        for child in self.__pieces:
            child.document = self.document
            child.file_path = self.file_path
            child.ocr_file_format = self.ocr_file_format

    def __set_dimensions_in_xml(self, dimensions: PieceDimensions) -> bool:
        xml_element: Element = self.xml_element
        top_left: List[float] = dimensions[0]
        bottom_right: List[float] = dimensions[2]
        x1: float = top_left[0]
        y1: float = top_left[1]
        x2: float = bottom_right[0]
        y2: float = bottom_right[1]
        width: float = x2 - x1
        height: float = y2 - y1
        has_changed: bool = False
        if self.ocr_file_format == PieceOcrFileFormat.ALTO_V3:
            if _MinidomUtil.set_attribute(xml_element, 'HPOS', int(x1)):
                has_changed = True
            if _MinidomUtil.set_attribute(xml_element, 'VPOS', int(y1)):
                has_changed = True
            if _MinidomUtil.set_attribute(xml_element, 'WIDTH', int(width)):
                has_changed = True
            if _MinidomUtil.set_attribute(xml_element, 'HEIGHT', int(height)):
                has_changed = True
        elif self.ocr_file_format == PieceOcrFileFormat.PAGE:
            points: str = _PiecePageUtil.dimensions_to_str(dimensions)
            if _MinidomUtil.set_attribute(xml_element, 'points', points):
                has_changed = True
        else:
            raise NotImplementedError(f'__set_dimensions_in_xml() not implemented for "{self.ocr_file_format}"')
        return has_changed


class PieceUtil:

    @staticmethod
    def to_pieces(path_in: str) -> Piece:
        return PieceUtil.read_data(path_in)

    @staticmethod
    def from_pieces(root_piece: Piece, path_out: PurePath = None) -> PurePath:
        if path_out is None:
            orig_file_path: PurePath = root_piece.file_path
            parent_dir: PurePath = orig_file_path.parent
            basename: str = orig_file_path.name
            suffix: str = orig_file_path.suffix
            new_name: str = basename.replace(f"{suffix}", f".gt{suffix}")
            path_out: PurePath = parent_dir.joinpath(new_name)
        file: TextIO = open(path_out, 'w')
        file.write(PieceUtil.to_xml_str(root_piece.document))
        file.close()
        return path_out

    @staticmethod
    def to_xml_str(node: md.Node) -> str:
        return node.toprettyxml(encoding="utf-8").decode("utf-8")

    @staticmethod
    def read_data(path_in: str) -> Piece:
        try:
            document: md.Document = md.parse(path_in)
            doc_root: md.Element = document.documentElement
        except Exception as _exc:
            raise RuntimeError(f"corrupt XML '{path_in}!")
        if doc_root is None:
            raise RuntimeError('invalid document root')
        name_space = doc_root.getAttribute('xmlns')
        piece: Optional[Piece]
        ocr_file_format: PieceOcrFileFormat = PieceOcrFileFormat.UNKNOWN
        if doc_root.localName == 'alto':
            piece = _PieceAltoV3Util.extract_data(doc_root)
            ocr_file_format = PieceOcrFileFormat.ALTO_V3
        elif name_space == PAGE_2013:
            piece = _PiecePageUtil.extract_data(doc_root)
            ocr_file_format = PieceOcrFileFormat.PAGE
        elif doc_root.localName == 'PcGts':
            piece = _PiecePageUtil.extract_data(doc_root, ns='pc:')
            ocr_file_format = PieceOcrFileFormat.PAGE
        else:
            raise RuntimeError(
                'Unknown Data-Format "{}" in "{}"'.format(doc_root.localName, path_in))
        piece.file_path = PurePath(path_in)
        piece.document = document
        piece.ocr_file_format = ocr_file_format
        return piece

    @staticmethod
    def calulate_dimensions_by_children(piece: Piece) -> PieceDimensions:
        if len(piece.pieces) > 0:
            dims: PieceDimensions = []
            for child_piece in piece.pieces:
                dims.extend(child_piece.dimensions)
            dims = PieceUtil.__calculate_dimensions_rect_bounds(dims)
            return dims
        return piece.dimensions

    @staticmethod
    def is_piece_in_polygon(piece: Piece, polygon: Polygon) -> bool:
        piece_polygon: Polygon = Polygon(piece.dimensions)
        convex_hull: Polygon = polygon.convex_hull
        return convex_hull.contains(piece_polygon.centroid)

    @staticmethod
    def flatten(piece: Piece) -> List[Piece]:

        def flatten_recursive(pc: Piece, pieces: List[Piece] = None) -> List[Piece]:
            if pieces is None:
                pieces = []
            pieces.append(pc)
            for child in pc.pieces:
                flatten_recursive(child, pieces)
            return pieces

        return flatten_recursive(piece)

    @staticmethod
    def __calculate_dimensions_rect_bounds(dimensions: PieceDimensions) -> PieceDimensions:
        min_x: Optional[float] = None
        min_y: Optional[float] = None
        max_x: Optional[float] = None
        max_y: Optional[float] = None
        for point in dimensions:
            point_x: float = point[0]
            point_y: float = point[1]
            if min_x is None or point_x < min_x:
                min_x = point_x
            if min_y is None or point_y < min_y:
                min_y = point_y
            if max_x is None or point_x > max_x:
                max_x = point_x
            if max_y is None or point_y > max_y:
                max_y = point_y
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]


def to_pieces(path_in: str) -> Piece:
    """Transform given input in various formats
    into internal Piece-Representation"""
    return PieceUtil.to_pieces(path_in)


def from_pieces(root_piece: Piece, path_out: str = None) -> PurePath:
    return PieceUtil.from_pieces(root_piece, path_out)


class _PieceAltoV3Util:
    @staticmethod
    def extract_data(doc_root) -> Piece:
        page_one: md.Element = doc_root.getElementsByTagName('Page')[0]
        _page_width = int(page_one.getAttribute('WIDTH'))
        _page_height = int(page_one.getAttribute('HEIGHT'))
        _dimensions = [[0, 0], [_page_width, 0], [_page_width, _page_height], [0, _page_height]]
        top_piece: Piece = Piece(
            page_one.getAttribute('ID'),
            page_one,
            ocr_file_format=PieceOcrFileFormat.ALTO_V3
        )
        top_piece.dimensions = _dimensions
        top_piece.level = PieceLevel.PAGE
        top_piece.subject = _PieceAltoV3Util.__get_piece_subject(doc_root)
        # composed level
        _block_pieces = []
        comp_blocks = doc_root.getElementsByTagName('ComposedBlock')
        if len(comp_blocks) > 0:
            for _comp_block in comp_blocks:
                comp_piece: Piece = Piece(
                    _comp_block.getAttribute('ID'),
                    _comp_block,
                    ocr_file_format=PieceOcrFileFormat.ALTO_V3
                )
                comp_piece.level = PieceLevel.REGION
                comp_piece.parent = top_piece
                comp_piece.dimensions = _PieceAltoV3Util.__extract_dimensions(_comp_block)
                text_blocks = _comp_block.getElementsByTagName('TextBlock')
                if len(text_blocks) < 1:
                    raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
                comp_piece.pieces = _PieceAltoV3Util.__read_blocks(text_blocks, comp_piece)
                _block_pieces.append(comp_piece)
        else:
            text_blocks = doc_root.getElementsByTagName('TextBlock')
            if len(text_blocks) < 1:
                raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
            _block_pieces = _PieceAltoV3Util.__read_blocks(text_blocks, top_piece)
        top_piece.pieces = _block_pieces
        return top_piece

    @staticmethod
    def __read_blocks(block_elements, parent):
        _block_pieces = []
        for _block in block_elements:
            _block_piece = Piece(
                _block.getAttribute('ID'),
                _block,
                ocr_file_format=PieceOcrFileFormat.ALTO_V3
            )
            _block_piece.level = PieceLevel.REGION
            _lines = _block.getElementsByTagName('TextLine')
            if len(_lines) == 0:
                raise RuntimeError(f"TextBlock@ID={_block_piece.id} contains no lines!")
            _block_piece.parent = parent
            _block_piece.pieces = _PieceAltoV3Util.__read_lines(_lines, _block_piece)
            _block_piece.dimensions = _PieceAltoV3Util.__extract_dimensions(_block)
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
            line_piece = Piece(
                _id,
                _text_line,
                ocr_file_format=PieceOcrFileFormat.ALTO_V3
            )
            line_piece.level = PieceLevel.LINE
            text_strings = _text_line.getElementsByTagName('String')
            if len(text_strings) < 1:
                raise RuntimeError(f"No words in line {_id}!")
            line_piece.pieces = _PieceAltoV3Util.__read_words(text_strings, line_piece)
            line_piece.parent = parent
            line_piece.dimensions = _PieceAltoV3Util.__extract_dimensions(_text_line)
            _lines.append(line_piece)
        return _lines

    @staticmethod
    def __read_words(text_strings, parent):
        _words = []
        for _text_string in text_strings:
            _id = _text_string.getAttribute('ID')
            word_piece = Piece(
                _id,
                _text_string,
                ocr_file_format=PieceOcrFileFormat.ALTO_V3
            )
            word_piece.level = PieceLevel.WORD
            _content = _text_string.getAttribute('CONTENT')
            if not _content.strip():
                continue
            word_piece.transcription = _content
            word_piece.dimensions = _PieceAltoV3Util.__extract_dimensions(_text_string)
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


class _PiecePageUtil:

    @staticmethod
    def dimensions_to_str(dimensions: PieceDimensions) -> str:
        strs: List[str] = list(map(lambda p: f'{int(p[0])},{int(p[1])}', dimensions))
        return ' '.join(strs)

    @staticmethod
    def extract_data(doc_root, ns='') -> Piece:
        page_one = doc_root.getElementsByTagName(ns + 'Page')[0]
        page_width = int(page_one.getAttribute('imageWidth'))
        page_height = int(page_one.getAttribute('imageHeight'))
        top_piece = Piece(
            page_one.getAttribute('imageFilename'),
            page_one,
            ocr_file_format=PieceOcrFileFormat.PAGE
        )
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
            _piece = _PiecePageUtil.__from_text_element(region, top_piece, ns)
            # go into details
            page_lines = region.getElementsByTagName(ns + 'TextLine')
            if len(page_lines) > 0:
                _piece.pieces = _PiecePageUtil.__read_lines(page_lines, _piece, ns)
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
            line_piece = _PiecePageUtil.__from_text_element(page_line, parent, ns)
            word_tokens = page_line.getElementsByTagName(ns + 'Word')
            line_piece.parent = parent
            # inspect PAGE on word level, if set
            if len(word_tokens) > 0:
                try:
                    word_pieces = [_PiecePageUtil.__from_text_element(el, line_piece, ns)
                                for el in word_tokens]
                    if not word_pieces:
                        raise RuntimeError(f"No words in line {line_piece.id}!")
                    # remove line content in favour of words content
                    line_piece._transcriptions = []
                    line_piece.pieces = word_pieces
                except PieceException as _pex:
                    raise RuntimeError(_pex.args) from _pex
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
        _type, _local = _PiecePageUtil.__map_piece_type(element)
        _piece = Piece(
            _id,
            element,
            ocr_file_format=PieceOcrFileFormat.PAGE
        )
        _piece.level = _type
        _piece.parent = parent

        # inspect geometry
        _coords = [n for n in element.childNodes if n.localName == 'Coords']
        if len(_coords) < 1 or 'points' not in _coords[0].attributes:
            raise PieceException(f"{_local}@ID={_id} invalid coordinate data")
        _points = _coords[0].getAttribute('points').split()
        # invariant: at least want 3 points, otherwise polygon area == Zero
        if len(_points) < 3:
            raise PieceException(f"{_local}@ID={_id} way too few points {_points}")
        try:
            _piece.dimensions = [[int(_point.split(',')[0]), int(_point.split(',')[1])]
                                for _point in _points]
        except ValueError as _val_err:
            raise PieceException(f"{_local}@ID={_id} invalid points {_points}") from _val_err

        # replace current text with next order children text
        _txt_eqs = [n for n in element.childNodes if n.localName == 'TextEquiv']
        if _txt_eqs:
            _first_unicode = _txt_eqs[0].getElementsByTagName(ns + 'Unicode')[0]
            if _first_unicode.firstChild:
                # replace linebreak if text only at region level
                _content = _first_unicode.firstChild.nodeValue.replace('\n', ' ')
                if _content:
                    _piece.transcription = _content
                    # overthrow existing parent transcription
                    if _piece.parent._transcriptions:
                        _piece.parent._transcriptions = []
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


class _MinidomUtil:
    @staticmethod
    def remove_element_and_clear_parent(element: Element, removable_tags: List[str] = []) -> List[Element]:
        parent: Element = element.parentNode
        removed_elements: List[Element] = []
        if parent:
            parent.removeChild(element)
            removed_elements.append(element)
            siblings: List[Element] = parent.childNodes
            for sibling in siblings:
                is_text_node: bool = sibling.nodeType == md.Node.TEXT_NODE
                is_removable_tag: bool = sibling.nodeName in removable_tags
                is_removable: bool = is_text_node or is_removable_tag
                if is_removable:
                    parent.removeChild(sibling)
                    if is_removable_tag:
                        removed_elements.append(sibling)
            if len(parent.childNodes) == 0:
                removed_parent_elements = _MinidomUtil.remove_element_and_clear_parent(parent, removable_tags)
                removed_elements.extend(removed_parent_elements)
        return removed_elements

    @staticmethod
    def set_attribute(element: Element, attr_name: str, value: Any) -> bool:
        attr_node: md.Node = element.getAttributeNode(attr_name)
        if attr_node is not None:
            old_value: str = str(attr_node.nodeValue)
            new_value: str = str(value)
            if new_value != old_value:
                attr_node.nodeValue = new_value
                return True
        return False
