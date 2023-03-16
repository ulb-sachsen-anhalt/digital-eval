import os
import re
from pathlib import Path
from typing import Final, Mapping, Match, Optional
from typing import NamedTuple, Tuple, List
from xml.dom.minidom import Element, Node

import numpy as np
from lxml import etree
from shapely import Polygon
from shapely.geometry import Point

from digital_eval import Piece, PieceLevel, PieceUtil, PieceDimensions

Tuple2D = Tuple[float, float]


class Point2D(NamedTuple):
    x: float
    y: float

    def to_tuple(self) -> Tuple2D:
        return self.x, self.y


Point2DList = List[Point2D]


class Util:
    __POINT_LIST_PATTERN: str = r'^(?:(?:-?\d+(?:\.\d+)?),(?:-?\d+(?:\.\d+)?)\ ?)+$'

    @staticmethod
    def remove_element_and_clear_parent(element: Element) -> None:
        parent: Element = element.parentNode
        if parent:
            parent.removeChild(element)
            siblings: List[Element] = parent.childNodes
            for sibling in siblings:
                is_text_node: bool = sibling.nodeType == Node.TEXT_NODE
                if is_text_node:
                    parent.removeChild(sibling)
            if len(parent.childNodes) == 0:
                Util.remove_element_and_clear_parent(parent)

    @staticmethod
    def str_to_point_2d(point_str: str) -> Point2D:
        x_str: str
        y_str: str
        x_str, y_str = point_str.split(',')
        x = float(x_str)
        y = float(y_str)
        return Point2D(x, y)

    @staticmethod
    def str_to_polygon(points_list: str) -> Polygon:
        match: Match = re.match(Util.__POINT_LIST_PATTERN, points_list)
        points_str: str = match.string
        point_strs_arr: List[str] = points_str.split(' ')
        points_arr: List[Point] = list(map(Util.__str_to_point, point_strs_arr))
        return Polygon(points_arr)

    @staticmethod
    def is_piece_in_polygon(piece: Piece, polygon: Polygon) -> bool:
        piece_polygon: Polygon = Polygon(piece.dimensions)
        convex_hull: Polygon = polygon.convex_hull
        return convex_hull.contains(piece_polygon.centroid)

    @staticmethod
    def calulate_dimensions_by_children(piece: Piece) -> PieceDimensions:
        if len(piece.pieces) > 0:
            dims: PieceDimensions = []
            for child_piece in piece.pieces:
                dims.extend(child_piece.dimensions)
            dims = Util.__calculate_dimensions_rect_bounds(dims)
            return dims
        return piece.dimensions

    @staticmethod
    def __calculate_dimensions_rect_bounds(dimensions: PieceDimensions) -> PieceDimensions:
        min_x: Optional[float] = None
        min_y: Optional[float] = None
        max_x: Optional[float] = None
        max_y: Optional[float] = None
        for point in dimensions:
            point_x: float = point[0]
            point_y: float = point[0]
            if min_x is None or point_x < min_x:
                min_x = point_x
            if min_y is None or point_y < min_y:
                min_y = point_y
            if max_x is None or point_x > max_x:
                max_x = point_x
            if max_y is None or point_y > max_y:
                max_y = point_y
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

    @staticmethod
    def __str_to_point(point_str: str) -> Point:
        x_str: str
        y_str: str
        x_str, y_str = point_str.split(',')
        x = float(x_str)
        y = float(y_str)
        return Point(x, y)

    @staticmethod
    def __set_dimensions_in_xml(piece: Piece):
        pass


class Point2D(NamedTuple):
    x: float
    y: float

    def to_tuple(self) -> Tuple2D:
        return self.x, self.y


class FrameFilterAltoV3:
    __NAMESPACES: Final[Mapping[str, str]] = {"alto": "http://www.loc.gov/standards/alto/ns-v3#"}
    __FILTER_ALTO_ELS: Final[List[str]] = ['TextBlock', 'Illustration', 'GraphicalElement']
    __DELETE_ALTO_ELS: Final[List[str]] = ['SP']

    def __init__(self, path_alto_in: str, points: Point2DList, path_alto_out: str = None, verbosity: int = 0):
        self.verbosity = verbosity
        self.__path_alto_in: str = path_alto_in
        self.__path_alto_out: str = path_alto_out
        self.__points: Point2DList = points
        self.__path_out: str = self.__create_out_path() if path_alto_out is None else path_alto_out
        self.__removals = {}
        self.__resized = {}
        self.__doc_root = None
        start_msg: str = f'filter strs from {path_alto_in} between {points}'
        print('[INFO] ' + start_msg)

    #
    def get_out_path(self) -> str:
        return self.__path_out

    def process(self) -> str:
        return self.__process_rectangle()

    def __process_rectangle(self) -> str:

        xml_tree = etree.parse(self.__path_alto_in)
        self.__doc_root = xml_tree.getroot()
        print_space = self.__doc_root[1][0][0]

        # optional: extract ComposedBlock TextRegions
        composed_blocks = print_space.xpath('alto:ComposedBlock', namespaces=FrameFilterAltoV3.__NAMESPACES)
        # move all composed_block's children
        # straight up to PrintSpace
        # clear empty composed block afterwards
        for composed_block in composed_blocks:
            for sub in composed_block.getchildren():
                print_space.append(sub)
            tag_name = etree.QName(composed_block).localname
            self.__set_removal(tag_name)
            self.__remove_el(composed_block)

        # transform OCR-D output remove GraphicalElement etc.
        top_left: Tuple2D
        bottom_right: Tuple2D
        top_left, bottom_right = FrameFilterAltoV3.__points_to_rect_bounds(self.__points)
        for to_filter in FrameFilterAltoV3.__FILTER_ALTO_ELS:
            els = print_space.xpath(f'alto:{to_filter}', namespaces=FrameFilterAltoV3.__NAMESPACES)
            for _el in els:
                self.__clear_leaves_recursive(
                    _el, top_left, bottom_right
                )

        # remove all now empty elements
        self.__clear(print_space)
        # shrink existing container elemenents to fit word content
        self.__shrink(print_space)

        for k, v in self.__removals.items():
            print(f'[INFO] filtered {v} {k} Elements')

        print(f'[INFO] resized  {self.__resized}')
        actual_words = self.__doc_root.xpath('//alto:String', namespaces=FrameFilterAltoV3.__NAMESPACES)
        n_chars = sum([len(w.attrib['CONTENT']) for w in actual_words])
        print(f'[INFO] frame has {len(actual_words)} words ({n_chars} chars)')

        # serialize
        _xml_tree = etree.ElementTree(self.__doc_root)
        _xml_tree.write(self.__path_out, encoding="utf-8", pretty_print=True)

        return self.__path_out

    def __create_out_path(self) -> str:
        dirname = os.path.dirname(self.__path_alto_in)
        filename = os.path.basename(self.__path_alto_in).split(".")[0]
        path_out = os.path.join(dirname, filename + ".gt.alto.xml")
        if os.path.exists(path_out):
            path_out = path_out + '_tmp'
        return path_out

    def __set_removal(self, tag_name):
        if tag_name not in self.__removals:
            self.__removals.setdefault(tag_name, 1)
        else:
            self.__removals[tag_name] = self.__removals[tag_name] + 1

    def __clear_leaves_recursive(self, el, top_left, btm_right):
        if el.getchildren():
            for kid in el.getchildren():
                tag_name = FrameFilterAltoV3.__get_tag_name(kid)
                # delete element immediately if blacklisted
                if tag_name in FrameFilterAltoV3.__DELETE_ALTO_ELS:
                    FrameFilterAltoV3.__remove_el(kid)
                    self.__set_removal(tag_name)
                else:
                    self.__clear_leaves_recursive(kid, top_left, btm_right)
        else:
            tl, br = FrameFilterAltoV3.__as_box(el)
            if not FrameFilterAltoV3.__is_in(top_left, btm_right, tl, br):
                FrameFilterAltoV3.__remove_el(el)
                tag_name = FrameFilterAltoV3.__get_tag_name(el)
                self.__set_removal(tag_name)
            elif not FrameFilterAltoV3.__get_element_data(el)[3]:
                FrameFilterAltoV3.__remove_el(el)
                tag_name = FrameFilterAltoV3.__get_tag_name(el)
                self.__set_removal(tag_name + '_empty')

    def __clear(self, print_space):
        last_branches = print_space.xpath(
            '//alto:TextLine', namespaces=FrameFilterAltoV3.__NAMESPACES)
        for last in last_branches:
            # if the branch has whithered, remove it
            _the_kids = last.getchildren()
            if len(_the_kids) == 0:
                tag_name = FrameFilterAltoV3.__get_tag_name(last)
                self.__set_removal(tag_name)
                parents = FrameFilterAltoV3.__get_parents_to(last, 'PrintSpace')
                FrameFilterAltoV3.__remove_el(last)
                for parent in parents:
                    if not parent.getchildren():
                        parent_tag_name = self.__get_tag_name(parent)
                        self.__set_removal(parent_tag_name)
                        FrameFilterAltoV3.__remove_el(parent)

    def __shrink(self, print_space):
        """still valid subtree, but propably must shrink
        lines and afterwards regions, too"""
        _regions = print_space.xpath('//alto:TextBlock', namespaces=FrameFilterAltoV3.__NAMESPACES)
        for _region in _regions:
            _lines = print_space.xpath(
                '//alto:TextLine', namespaces=FrameFilterAltoV3.__NAMESPACES)
            for _line in _lines:
                if FrameFilterAltoV3.__resized_element_to_fit_children(_line, 'alto:String'):
                    the_label = etree.QName(_line).localname
                    self.__add_resized(the_label)
            if FrameFilterAltoV3.__resized_element_to_fit_children(_region, 'alto:TextLine'):
                the_label = etree.QName(_region).localname
                self.__add_resized(the_label)

    def __add_resized(self, tag_name):
        just_resized = 1
        if tag_name in self.__resized:
            prev_n = self.__resized[tag_name]
            just_resized = prev_n + 1
        self.__resized[tag_name] = just_resized

    @staticmethod
    def __remove_el(el) -> None:
        """Remove this element from it's parent element"""
        super_el = el.getparent()
        if super_el is not None:
            super_el.remove(el)
        else:
            _id = el.attrib['ID'] if 'ID' in el.attrib else 'n.a'
            raise RuntimeError(f"No parent found for {_id}!")

    @staticmethod
    def __as_box(elem):
        """extract geom info into tuple:
        top-left - bottom-right"""
        try:
            x_1 = int(elem.attrib['HPOS'])
            y_1 = int(elem.attrib['VPOS'])
            w = int(elem.attrib['WIDTH'])
            h = int(elem.attrib['HEIGHT'])
        except KeyError as err:
            print(f"[ERROR] {err} with {elem}")
        y_2 = y_1 + h
        x_2 = x_1 + w
        return (x_1, y_1), (x_2, y_2)

    @staticmethod
    def __get_element_data(elem):
        """Get relevant information from OCR-Token like dimension
        and optional textual content"""

        box_id = elem.attrib['ID']
        _x0 = int(elem.attrib['HPOS'])
        _y0 = int(elem.attrib['VPOS'])
        x = _x0 + int(elem.attrib['WIDTH']) / 2
        y = _y0 + int(elem.attrib['HEIGHT']) / 2
        text = ""
        if 'CONTENT' in elem.attrib and elem.attrib.get('CONTENT').strip():
            text = elem.attrib['CONTENT'].strip()
        element_box = (box_id, x, y, text)
        return element_box

    @staticmethod
    def __is_in(tl_self, br_self, tl_other, br_other) -> bool:
        return tl_self[0] <= tl_other[0] \
            and tl_self[1] <= tl_other[1] \
            and br_other[0] <= br_self[0] \
            and br_other[1] <= br_self[1]

    @staticmethod
    def __get_tag_name(_el):
        """Get Element Tagname without Namespace prefixed"""
        return etree.QName(_el).localname

    @staticmethod
    def __get_parents_to(el, parent_tag):
        """Traverse ancestors tree-up"""
        parent = el.getparent()
        tag_name = FrameFilterAltoV3.__get_tag_name(parent)
        parents = []
        while tag_name != parent_tag:
            parents.append(parent)
            parent = parent.getparent()
            tag_name = FrameFilterAltoV3.__get_tag_name(parent)
        return parents

    @staticmethod
    def __resized_element_to_fit_children(element, sub_element: str):
        """Optional resize element to only span it's current children"""

        kids = element.findall(sub_element, FrameFilterAltoV3.__NAMESPACES)
        _boxes = [FrameFilterAltoV3.__as_box(s) for s in kids]
        mins = np.min(_boxes, axis=0)
        _children_min_x = mins[0][0]
        _children_min_y = mins[0][1]
        resized = False
        # start
        _prev_hpos = int(element.attrib['HPOS'])
        if _prev_hpos != _children_min_x:
            # fit new left border
            element.attrib['HPOS'] = str(_children_min_x)
            resized = True
        _prev_vpos = int(element.attrib['VPOS'])
        if _prev_vpos != _children_min_y:
            # fit new top border
            # since tesseract creates often lines
            # which *do not* enclose their words
            # this is likely to to done each time
            element.attrib['VPOS'] = str(_children_min_y)
            resized = True
        maxs = np.max(_boxes, axis=0)
        _children_max_x = maxs[1][0]
        _children_max_y = maxs[1][1]
        _width = _children_max_x - _children_min_x
        _prev_width = int(element.attrib['WIDTH'])
        if _prev_width != _width:
            # fit new right border
            element.attrib['WIDTH'] = str(_width)
            resized = True
        _height = _children_max_y - _children_min_y
        _prev_height = int(element.attrib['HEIGHT'])
        if _prev_height != _height:
            # fit new bottom
            # for details, see remark on top border
            element.attrib['HEIGHT'] = str(_height)
            resized = True
        return resized

    @staticmethod
    def __points_to_rect_bounds(points: Point2DList) -> Tuple[Tuple2D, Tuple2D]:
        min_x: float = -1
        min_y: float = -1
        max_x: float = -1
        max_y: float = -1
        for point in points:
            point_x: float = point.x
            point_y: float = point.y
            if point_x < min_x or min_x == -1:
                min_x = point_x
            if point_y < min_y or min_y == -1:
                min_y = point_y
            if point_x > max_x or max_x == -1:
                max_x = point_x
            if point_y > max_y or max_y == -1:
                max_y = point_y
        top_left: Tuple2D = (min_x, min_y)
        bottom_right: Tuple2D = (max_x, max_y)
        return top_left, bottom_right


class PolygonFrameFilter:

    def __init__(self, ocr_path_in: str, points_list: str):
        self.__ocr_path_in: Path = Path(ocr_path_in)
        self.__polygon: Polygon = Util.str_to_polygon(points_list)

    @property
    def ocr_file_path(self) -> Path:
        return self.__ocr_path_in

    @property
    def polygon(self) -> Polygon:
        return self.__polygon

    def process(self) -> Optional[Piece]:
        piece_result: Piece = PieceUtil.to_pieces(str(self.__ocr_path_in))
        self.__process_piece(piece_result)
        return piece_result

    def __process_piece(self, piece: Piece) -> bool:
        for child_piece in piece.pieces:
            if not self.__process_piece(child_piece):
                piece.remove_pieces(child_piece)
                element: Element = child_piece.xml_element
                Util.remove_element_and_clear_parent(element)
                # gucke im parent ob da noch nodes
        if piece.level > PieceLevel.WORD:
            if len(piece.pieces) == 0:
                return False
            piece.dimensions = Util.calulate_dimensions_by_children(piece)
            return True
        if piece.level != PieceLevel.WORD:
            raise Exception(f'Unknown Level: {piece.level}')
        return Util.is_piece_in_polygon(piece, self.polygon)
