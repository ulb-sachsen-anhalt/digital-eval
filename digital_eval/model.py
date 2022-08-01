# -*- coding: utf-8 -*-
"""Model Module"""

from enum import (
    Enum
)
from typing import (
    List, 
)

import xml.dom.minidom

from shapely.geometry import (
    Polygon
)


PAGE_2013 = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
XML_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#',
          'pg2013': PAGE_2013}


UNSET = 'n.a.'

class PieceType(Enum):
    # more hierarchically
    UNKNOWN = 0
    GLYPH = 1
    WORD = 2
    LINE = 3
    REGION = 4
    PAGE = 4

class PieceSubject(Enum):
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

class Piece:
    """Piece base composition for analytical purposes"""

    def __init__(self, id=UNSET):
        self.id = id
        self.type = PieceType.PAGE
        self.subject = PieceSubject.UNKNOWN
        self.data = None
        self._transcriptions = []
        self.parent = None
        self.custom = {}
        self.dimensions = []
        self.pieces = []

    def __repr__(self) -> str:
        return f"{self.id}:{self.transcription}"

    def _is_superstruct(self):
        return self.type in [PieceType.PAGE, PieceType.REGION, PieceType.LINE]

    @property
    def transcription(self):
        """Get textual content as sequential textual string,
        with the order corresponding to it's _previous and
        next properties.

        Text will come without sanitized linebreaks, but
        includes a whitespace between single lines and words.
        """
        if self._transcriptions:
            return self._transcriptions[0].text
        elif not self._transcriptions and self._is_superstruct():
            return ' '.join([_p.transcription
                for _p in self.pieces])
        raise RuntimeError(f"ID={self.id}: Can't get text_content for type {self.type}!")

    @transcription.setter
    def transcription(self, transcription):
        """Set textual transcription representing this piece"""
        _transcription = PieceTranscription()
        _transcription.text = transcription
        self._transcriptions.append(_transcription)

    def __contains__(self, other_piece) -> bool:
        """Test for topological membership of an other_piece"""
        if not self.dimensions:
            raise RuntimeError(f"ID={self.id}: self has invalid dimensions!")
        if not other_piece.dimensions:
            raise RuntimeError(f"{other_piece}: other has invalid dimensions!")
        self_shape = Polygon(self.dimensions)
        other_shape = Polygon(other_piece.dimensions)
        return self_shape.contains(other_shape)


def to_pieces(path_in):
    """Transform given input in various formats 
    into internal Piece-Representation"""

    return _read_data(path_in)


def _read_data(path_in):
    doc_root = xml.dom.minidom.parse(path_in).documentElement
    if doc_root is None:
        raise RuntimeError('invalid document root')
    name_space = doc_root.getAttribute('xmlns')
    if doc_root.localName == 'alto':
        return _extract_alto_data(doc_root)
    elif name_space == PAGE_2013:
       return  _extract_page_data(doc_root)
    elif doc_root.localName == 'PcGts':
        return _extract_page_data(doc_root, ns='pc:')
    else:
        raise RuntimeError(
            'Unknown Data-Format "{}" in "{}"'.format(doc_root.localName, path_in))


def _extract_alto_data(doc_root):
    page_one = doc_root.getElementsByTagName('Page')[0]
    _page_width = int(page_one.getAttribute('WIDTH'))
    _page_height = int(page_one.getAttribute('HEIGHT'))
    _dimensions = [[0, 0], [_page_width, 0], [_page_width, _page_height], [0, _page_height]]
    top_piece = Piece(page_one.getAttribute('ID'))
    top_piece.dimensions = _dimensions
    top_piece.type = PieceType.PAGE
    top_piece.subject = __get_piece_subject_alto(doc_root)
    # composed level
    _block_pieces = []
    comp_blocks = doc_root.getElementsByTagName('ComposedBlock')
    if len(comp_blocks) > 0:
        for _comp_block in comp_blocks:
            comp_piece = Piece(_comp_block.getAttribute('ID'))
            comp_piece.type = PieceType.REGION
            comp_piece.parent = top_piece
            comp_piece.dimensions = __extract_alto_dimensions(_comp_block)
            text_blocks = _comp_block.getElementsByTagName('TextBlock')
            if len(text_blocks) < 1:
                raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")       
            comp_piece.pieces = _read_alto_blocks(text_blocks, comp_piece)
            _block_pieces.append(comp_piece)
    else:
        text_blocks = doc_root.getElementsByTagName('TextBlock')
        if len(text_blocks) < 1:
            raise RuntimeError(f"Empty ALTO {doc_root} - no blocks!")
        _block_pieces = _read_alto_blocks(text_blocks, top_piece)
    top_piece.pieces = _block_pieces
    _all_points = [point for _block in _block_pieces for point in _block.dimensions]
    top_piece.dimensions = _all_points
    return top_piece


def _read_alto_blocks(block_elements, parent):
    _block_pieces = []
    for _block in block_elements:
        _block_piece = Piece(_block.getAttribute('ID'))
        _block_piece.type = PieceType.REGION
        _lines = _block.getElementsByTagName('TextLine')
        if len(_lines) == 0:
            raise RuntimeError(f"TextBlock@ID={_block_piece.id} contains no lines!")
        _block_piece.parent = parent
        _block_piece.pieces = _read_lines_alto(_lines, _block_piece)
        _block_piece.dimensions = __extract_alto_dimensions(_block)
        _block_pieces.append(_block_piece)
    return _block_pieces


def __get_piece_subject_alto(doc_root):
    gt_type_el = doc_root.getElementsByTagName('OtherTag')
    _subject = UNSET
    if gt_type_el and len(gt_type_el) > 0:
        # deprecated
        label = gt_type_el[0].getAttribute('LABEL')
        if label:
            _subject = label
        # new alto way
        elif _subject is None:
            gt_els = [e for e in gt_type_el if e.getAttribute(
                'ID') == "ulb_groundtruth_type"]
            if len(gt_els) == 1:
                value = gt_els[0].getAttribute('VALUE')
                if value:
                    _subject = value
    return _subject


def _read_lines_alto(the_lines, parent):
    _lines = []
    for _text_line in the_lines:
        _id = _text_line.getAttribute('ID')
        line_piece = Piece(_id)
        line_piece.type = PieceType.LINE
        text_strings = _text_line.getElementsByTagName('String')
        if len(text_strings) < 1:
            raise RuntimeError(f"No words in line {_id}!")
        line_piece.pieces = _read_words_alto(text_strings, line_piece)
        line_piece.parent = parent
        line_piece.dimensions = __extract_alto_dimensions(_text_line)
        _lines.append(line_piece)
    return _lines


def _read_words_alto(text_strings, parent):
    _words = []
    for _text_string in text_strings:
        _id = _text_string.getAttribute('ID')
        word_piece = Piece(_id)
        word_piece.type = PieceType.WORD
        _content = _text_string.getAttribute('CONTENT')
        if not _content.strip():
            continue
        word_piece.transcription = _content
        word_piece.dimensions = __extract_alto_dimensions(_text_string)
        word_piece.parent = parent
        _words.append(word_piece)
    return _words


def __extract_alto_dimensions(el, prefer_box=True):
    if not prefer_box:
        _shape = [ n for n in el.getChildren() if n.localName == 'Shape']
        if len(_shape) == 1:
            pass
    else:
        _left = int(el.getAttribute('HPOS'))
        _top = int(el.getAttribute('VPOS'))
        _height = int(el.getAttribute('HEIGHT'))
        _width = int(el.getAttribute('WIDTH'))
        return [[_left,_top], [_left + _width, _top], 
                [_left + _width, _top + _height], [_left, _top + _height]]
    raise RuntimeError(f"{el.localName}@ID={el.getAttribute('ID')}: Can't calculate dimensions")


def _extract_page_data(doc_root, ns=''):
    page_one = doc_root.getElementsByTagName(ns+'Page')[0]
    page_width = int(page_one.getAttribute('imageWidth'))
    page_height = int(page_one.getAttribute('imageHeight'))
    top_piece = Piece(page_one.getAttribute('imageFilename'))
    top_piece.type = PieceType.PAGE
    top_piece.dimensions = [[0,0], [page_width,0], 
        [page_width, page_height], [0, page_height]]
    regions = doc_root.getElementsByTagName(ns+'TextRegion')
    regions.extend(doc_root.getElementsByTagName(ns+'TableRegion')) 
    if len(regions) < 1:
        raise RuntimeError(f"Empty PAGE {doc_root} - no regions!")

    # inspect *all* regions
    region_pieces = []
    for region in regions:
        _piece = __from_page_text_element(region, top_piece, ns)
        # go into details
        page_lines = region.getElementsByTagName(ns+'TextLine')
        if len(page_lines) < 1:
            raise RuntimeError(f"Empty block/region {_piece.id}!")
        _piece.pieces = _read_lines_page(page_lines, _piece, ns)
        _piece.parent = top_piece
        region_pieces.append(_piece)
    top_piece.pieces = region_pieces
    _all_points = [point for reg in region_pieces for point in reg.dimensions]
    top_piece.dimensions = _all_points
    return top_piece


def _read_lines_page(page_lines, parent, ns) -> List:
    line_pieces = []
    for page_line in page_lines:
        line_piece = __from_page_text_element(page_line, parent, ns)
        word_tokens = page_line.getElementsByTagName(ns+'Word')
        line_piece.parent = parent
        # inspect PAGE on word level, if set
        if len(word_tokens) > 0:
            word_pieces = [__from_page_text_element(el, line_piece, ns) for el in word_tokens]
            if not word_pieces:
                raise RuntimeError(f"No words in line {line_piece.id}!")
            # remove line content in favour of words content
            # line_piece.content = None
            line_piece.pieces = word_pieces
        line_pieces.append(line_piece)
    return line_pieces


def __from_page_text_element(element, parent, ns) -> Piece:
    """Most basic transformation from PAGE XML textual nodes"""
    _id = element.getAttribute('id')
    _type, _local = ___map_piece_type(element)
    _piece = Piece(_id)
    _piece.type = _type
    _piece.parent = parent
    # inspect geometry
    _coords = [n for n in element.childNodes if n.localName == 'Coords']
    if len(_coords) < 1 or 'points' not in _coords[0].attributes:
        raise RuntimeError(f"{_local}@ID={_id} invalid coordinate data")
    _points = _coords[0].getAttribute('points').split()
    if len(_points) < 4:
        raise RuntimeError(f"{_local}@ID={_id} way too few points {_points}")
    _piece.dimensions = [[int(_point.split(',')[0]),int(_point.split(',')[1])] 
        for _point in _points]
    # inspect text
    _txt_eqs = [n for n in element.childNodes if n.localName == 'TextEquiv']
    if len(_txt_eqs) != 1:
        raise RuntimeError(f"{_local}@ID={_id} invalid txt node {_txt_eqs}")
    _content = _txt_eqs[0].getElementsByTagName(ns+'Unicode')[0].firstChild.nodeValue
    if not _content or not _content.strip():
        raise RuntimeError(f"{_local}@ID={_id} invalid txt content!")
    # only add content when not top-level piece
    if _type == PieceType.WORD:
        _piece.transcription = _content
    return _piece


def ___map_piece_type(element):
    _local = element.localName
    _name = UNSET
    if _local == 'Word':
        _name = PieceType.WORD
    elif _local == 'TextLine':
        _name =  PieceType.LINE
    elif _local == 'TextRegion':
        _name = PieceType.REGION
    return(_name, _local)



def filter_all(self, coords_start, coords_end):
    all_lines = self.get_lines()
    filter_box = BoundingBox(coords_start, coords_end)

    def centroid(bbox):
        x = bbox.p1[0] + int((bbox.p2[0] - bbox.p1[0]) / 2)
        y = bbox.p1[1] + int((bbox.p2[1] - bbox.p1[1]) / 2)
        return (x, y)

    filter_lines = []
    for line in all_lines:
        new_line = OCRWordLine(line.id)
        for word in line.words:
            c = centroid(word)
            if filter_box.contains(BoundingBox(c, c)):
                new_line.add_word(word)
        if new_line.words:
            filter_lines.append(new_line)
    return filter_lines




##############################################################################################
#############################################################################################
############################################################################################
###########################################################################################
class BoundingBox:

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def intersection(self, other) -> bool:
        '''
        Test if two Rectangles truely intersect (given by Tuples that represent their Points)
        cf. https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles
        '''
        x1 = max(min(self.p1[0], self.p2[0]), min(other.p1[0], other.p2[0]))
        x2 = min(max(self.p1[0], self.p2[0]), max(other.p1[0], other.p2[0]))
        y1 = max(min(self.p1[1], self.p2[1]), min(other.p1[1], other.p2[1]))
        y2 = min(max(self.p1[1], self.p2[1]), max(other.p1[1], other.p2[1]))
        if x1 < x2 and y1 < y2:
            return (x2 - x1) * (y2 - y1)
        else:
            return 0

    def enclose(self, other):
        '''Create new BoundingBox that encapsulates self and other Box'''

        x1 = min(self.p1[0], other.p1[0])
        y1 = min(self.p1[1], other.p1[1])
        x2 = max(self.p2[0], other.p2[0])
        y2 = max(self.p2[1], other.p2[1])
        return BoundingBox((x1, y1), (x2, y2))

    def contains(self, other):
        return self.p1[0] < other.p1[0] and self.p1[1] < other.p1[1] and self.p2[0] > other.p2[0] and self.p2[1] > other.p2[1]


class OCRToken(BoundingBox):
    '''Generic OCR Container that represents Data extracted from ALTO or PAGE'''

    def __init__(self, identifier):
        self.id = identifier
        self.p1 = None
        self.p2 = None
        self.has_text = False

    def get_id(self):
        return self.id

    @staticmethod
    def is_alto(element):
        return element.getAttribute('HPOS')

    @staticmethod
    def is_page(element):
        return element.nodeName.startswith('pc:')

    @staticmethod
    def is_page_without_namespace(element):
        return not ':' in element.nodeName

    def calculate_points(self, element):
        if OCRToken.is_alto(element):
            hpos = int(element.getAttribute('HPOS'))
            vpos = int(element.getAttribute('VPOS'))
            self.p1 = (hpos, vpos)
            _width = int(element.getAttribute('WIDTH'))
            _height = int(element.getAttribute('HEIGHT'))
            self.p2 = (self.p1[0] + _width, self.p1[1] + _height)
        elif OCRToken.is_page(element):
            coords = element.getElementsByTagName('pc:Coords')
            if len(coords) > 0:
                point_data = coords[0].getAttribute('points')
                self.p1 = [int(c) for c in point_data.split(' ')[0].split(',')]
                self.p2 = [int(c) for c in point_data.split(' ')[2].split(',')]
        elif OCRToken.is_page_without_namespace(element):
            coords = element.getElementsByTagName('Coords')
            if len(coords) > 0:
                point_data = coords[0].getAttribute('points')
                if len(point_data.strip()) < 1:
                    bad_id = dict(element.attributes.items())['id']
                    raise RuntimeError(f"{bad_id} has empty Coords!")
                if len(point_data.split(' ')) < 4:
                    raise RuntimeError(f"{self.id} has no enough Coords points: {point_data}")
                self.p1 = [int(c) for c in point_data.split(' ')[0].split(',')]
                self.p2 = [int(c) for c in point_data.split(' ')[2].split(',')]
        else:
            raise RuntimeError('{}: Cannot extract geometric Data from "{}"!'.format(
                element.getAttribute('ID'), self.id))


class OCRWord(OCRToken):
    '''Atomic OCR-Unit representing a word'''

    def __init__(self, identifier, element):
        super().__init__(identifier)
        self.characters = None
        if element.localName == 'String':
            self._read_alto_string(element)
        if element.localName == 'Word':
            self._read_page_word(element)
        self.calculate_points(element)

    def _read_alto_string(self, element):
        self.characters = element.getAttribute('CONTENT')

    def _read_page_word(self, element):
        text_equivs = [node 
                      for node in element.childNodes
                      if node.localName == 'TextEquiv']
        if len(text_equivs) == 1:
            try:
                txt_data = [coded.childNodes[0].data 
                            for coded in text_equivs[0].childNodes
                            if coded.localName == 'Unicode']
                self.characters = txt_data[0]
            except IndexError as exc:
                p_word = text_equivs[0].parentNode
                raise RuntimeError(f"{p_word.getAttribute('id')} misses text: {exc.args[0]}")

    def get_characters(self):
        return self.characters

    def __repr__(self):
        return '{}'.format(self.characters)


class OCRWordLine(OCRToken):
    '''Represents an aligned collection of Words'''

    def __init__(self, identifier, element=None):
        super().__init__(identifier)
        self.words = []
        if element:
            self.calculate_points(element)
            self.has_text = True
            page_txts = None
            if OCRToken.is_page(element):
                page_txts = OCRWordLine.page_txts(element)
            elif OCRToken.is_page_without_namespace(element):
                page_txts = OCRWordLine.page2013_txts(element)
            if not page_txts:
                self.has_text = False
            else:
                self.words = page_txts

    def __repr__(self):
        _width = 0
        _height = 0
        if self.p1 and self.p2:
            _width = abs(self.p2[0] - self.p1[0])
            _height = abs(self.p2[1] - self.p1[1])
        return '[{}][{}:{}]{}-{} "{}"'.format(self.get_id(), _width, _height, self.p1, self.p2, self.get_text())

    @staticmethod
    def page_txts(element):
        unicodes = element.getElementsByTagName('pc:Unicode')
        if not len(unicodes) > 0:
            return False
        children = unicodes[0].childNodes
        if not len(children) > 0:
            return False
        chars = children[0].nodeValue.strip()
        if len(chars)> 0:
            return chars

    @staticmethod
    def page2013_txts(element):
        kids = element.childNodes
        if not len(kids) > 0:
            return False
        text_equivs = [k for k in kids if k.localName == 'TextEquiv']
        if text_equivs and len(text_equivs) > 0:
            unicodes = text_equivs[0].getElementsByTagName('Unicode')
            if unicodes:
                first_node = unicodes[0].firstChild
                if first_node:
                    chars = first_node.nodeValue.strip()
                    if OCRWordLine._contains_at_least_one_alpha(chars):
                        return chars

    @staticmethod
    def _contains_at_least_one_alpha(chars):
        return [c for c in chars if c.isalpha()]

    def contains_text(self):
        return len(self.words) > 0

    def add_word(self, ocr_word: OCRWord):
        if not self.p1:
            self.p1 = ocr_word.p1
        if not self.p2:
            self.p2 = ocr_word.p2

        new_box = self.enclose(ocr_word)
        self.p1 = new_box.p1
        self.p2 = new_box.p2
        self.words.append(ocr_word)

    def get_text(self) -> List[str]:
        line = ' '.join([word.get_characters()
                         for word in self.words if isinstance(word, OCRWord)])
        if not line:
            line = self.words
        return line


class OCRRegion(OCRToken):
    '''Logical Collection of Lines'''

    def __init__(self, identifier, element):
        super().__init__(identifier)
        self.lines = []
        self.calculate_points(element)

    def get_lines(self) -> List[OCRWordLine]:
        return self.lines

    def add_line(self, ocr_line: OCRWordLine):
        self.lines.append(ocr_line)

    def __repr__(self) -> str:
        return '[{}]{}-{} "{}"'.format(self.get_id(), self.p1, self.p2, len(self.get_lines()))
