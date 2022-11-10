# -*- coding: utf-8 -*-
"""Model Module"""

import os
import re

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

# mark unset values as 'not available'
NOT_SET = 'n.a.'

class BoundingBox:
    """Naive implementation of rectangular
    box-like areas"""

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def intersection(self, other) -> bool:
        '''
        Test if two rectangles truly intersect (given by tuples that represent their points)
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
        return ':' not in element.nodeName

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
        if len(unicodes) <= 0:
            return False
        children = unicodes[0].childNodes
        if len(children) <= 0:
            return False
        return children[0].nodeValue

    @staticmethod
    def page2013_txts(element):
        kids = element.childNodes
        if len(kids) <= 0:
            return False
        text_equivs = [k for k in kids if k.localName == 'TextEquiv']
        if text_equivs and len(text_equivs) > 0:
            unicodes = text_equivs[0].getElementsByTagName('Unicode')
            if unicodes:
                first_node = unicodes[0].firstChild
                if first_node:
                    return first_node

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


class OCRData:
    '''OCR Data from both PAGE or ALTO'''

    def __init__(self, path_in):
        self.blocks = []
        self.path_in = path_in
        self.page_dimensions = None
        self.type_data = None
        self.type_groundtruth = NOT_SET
        self._get_groundtruth_from_filename()
        self.log_level = 0
        self._read_data()

    def set_log_level(self, log_level):
        self.log_level = log_level

    def _get_groundtruth_from_filename(self):
        file_name = os.path.basename(self.path_in)
        result = re.match(r'.*gt.(\w{3,}).xml$', file_name)
        if result:
            self.type_groundtruth = result[1]
        else:
            alternative = re.match(r'.*\.(\w{3,})\.gt\.xml$', file_name)
            if alternative:
                self.type_groundtruth = alternative[1]

    def _read_data(self):
        doc_root = xml.dom.minidom.parse(self.path_in).documentElement
        if doc_root is None:
            raise RuntimeError('invalid document root')
        name_space = doc_root.getAttribute('xmlns')
        if doc_root.localName == 'alto':
            self._extract_alto_data(doc_root)
        elif name_space == PAGE_2013:
            self._extract_page_data(doc_root)
        elif doc_root.localName == 'PcGts':
            self._extract_page_data(doc_root, ns='pc:')
        else:
            raise RuntimeError(
                'Unknown Data-Format "{}" in "{}"'.format(doc_root.localName, self.path_in))

    def _extract_alto_data(self, doc_root):
        # handle groundtruth type
        gt_type_el = doc_root.getElementsByTagName('OtherTag')
        if gt_type_el and len(gt_type_el) > 0:
            # deprecated
            label = gt_type_el[0].getAttribute('LABEL')
            if label:
                self.type_groundtruth = label
            # new alto way
            elif self.get_type_groundtruth is None:
                gt_els = [e for e in gt_type_el if e.getAttribute(
                    'ID') == "ulb_groundtruth_type"]
                if len(gt_els) == 1:
                    value = gt_els[0].getAttribute('VALUE')
                    if value:
                        self.type_groundtruth = value

        # handle page dimension
        page_one = doc_root.getElementsByTagName('Page')[0]
        self.page_dimensions = (int(page_one.getAttribute(
            'WIDTH')), int(page_one.getAttribute('HEIGHT')))
        text_blocks = doc_root.getElementsByTagName('TextBlock')

        # read block, lines-n-words
        for text_block in text_blocks:
            block_id = text_block.getAttribute('ID')
            ocr_block = OCRRegion(block_id, text_block)
            cured_lines = text_block.getElementsByTagName('TextLine')
            for text_line in cured_lines:
                line_id = text_line.getAttribute('ID')
                ocr_line = OCRWordLine(line_id, text_line)
                text_strings = text_line.getElementsByTagName('String')
                for text_string in text_strings:
                    word_id = text_string.getAttribute('ID')
                    # word_content = text_string.getAttribute('CONTENT')
                    # if not word_content.strip():
                    #     if self.log_level > 1:
                    #         print('[TRACE]({}) ignore empty word "{}"'.format(
                    #             self.path_in, word_id))
                    #     continue
                    ocr_word = OCRWord(word_id, text_string)
                    ocr_line.add_word(ocr_word)
                if len(ocr_line.words) > 0:
                    ocr_block.add_line(ocr_line)
                else:
                    if self.log_level > 1:
                        print('[TRACE]({}) ignore empty line "{}"'.format(
                            self.path_in, line_id))
            self.blocks.append(ocr_block)

    def _extract_page_data(self, doc_root, ns=''):
        page_one = doc_root.getElementsByTagName(ns+'Page')[0]
        self.page_dimensions = (int(page_one.getAttribute('imageWidth')), int(
            page_one.getAttribute('imageHeight')))
        blocks = doc_root.getElementsByTagName(ns+'TextRegion')
        blocks.extend (doc_root.getElementsByTagName(ns+'TableRegion'))
        for block in blocks:
            block_id = block.getAttribute('id')
            ocr_block = OCRRegion(block_id, block)
            cured_lines = block.getElementsByTagName(ns+'TextLine')
            for text_line in cured_lines:
                line_id = text_line.getAttribute('id')
                word_tokens = text_line.getElementsByTagName(ns+'Word')
                # 1. inspect PAGE on word level
                if len(word_tokens) > 0:
                    ocr_line = OCRWordLine(line_id)
                    for word_token in word_tokens:
                        word_id = word_token.getAttribute('id')
                        ocr_word = OCRWord(word_id, word_token)
                        ocr_line.add_word(ocr_word)
                # 2. inspect PAGE on line level
                else:
                    ocr_line = OCRWordLine(line_id, text_line)
                # final inspection
                # if not ocr_line or not ocr_line.contains_text():
                #     if self.log_level > 1:
                #         print('[TRACE]({}) ignore empty line "{}"'.format(
                #             self.path_in, line_id))
                #     continue
                ocr_block.add_line(ocr_line)
            self.blocks.append(ocr_block)

    def get_lines(self) -> List[OCRWordLine]:
        line_blocks = [block.get_lines() for block in self.blocks]
        return [l for lines in line_blocks for l in lines]

    def get_type_groundtruth(self) -> str:
        return self.type_groundtruth

    def filter_all(self, coords_start, coords_end):
        all_lines = self.get_lines()
        filter_box = BoundingBox(coords_start, coords_end)
        filter_lines = []
        for line in all_lines:
            new_line = OCRWordLine(line.id)
            if not isinstance(line.words, str):
                for _word in line.words:
                    c = centroid(_word)
                    if filter_box.contains(BoundingBox(c, c)):
                        new_line.add_word(_word)
                if len(new_line.words) > 0:
                    filter_lines.append(new_line)
            elif isinstance(line.words, str):
                c = centroid(line)
                if filter_box.contains(BoundingBox(c, c)):
                    filter_lines.append(line)
        return filter_lines

    def get_lines_text(self) -> List[str]:
        the_lines = self.get_lines()
        return [l.get_text() for l in the_lines]

    def get_page_dimensions(self):
        return self.page_dimensions


def centroid(bbox):
    _polygon = Polygon(([bbox.p1[0], bbox.p1[1]],[bbox.p2[0], bbox.p1[1]],[bbox.p2[0], bbox.p2[1]],[bbox.p1[0], bbox.p2[1]]))
    _polygon.centroid
    return (_polygon.centroid.x, _polygon.centroid.y)
