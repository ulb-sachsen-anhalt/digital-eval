"""filter module"""

import re
from pathlib import Path
from typing import Match, Optional, Dict
from typing import NamedTuple, List

from shapely import Polygon
from shapely.geometry import Point

from digital_eval.model import (
    DigitalObjectTree,
    DigitalObjectLevel,
    DigitalObjectChanges,
    to_digital_object,
)
from digital_eval.model.digital_object_util import (
    DigitalObjectUtil,
)


class PolygonFrameFilterReport(NamedTuple):
    """report container for structual manipulations"""
    removed_elements: Dict[str, int] = {}
    resized_elements: Dict[str, int] = {}


class PolygonFrameFilterUtil:
    """helper methods for the PolygonFameFlter"""
    POINT_LIST_PATTERN: str = r'^(?:(?:-?\d+(?:\.\d+)?),(?:-?\d+(?:\.\d+)?)\ ?)+$'

    @staticmethod
    def str_to_polygon(points_list: str) -> Polygon:
        match: Match = re.match(PolygonFrameFilterUtil.POINT_LIST_PATTERN, points_list)
        points_str: str = match.string
        point_strs_arr: List[str] = points_str.split(' ')
        points_arr: List[Point] = list(map(PolygonFrameFilterUtil.__str_to_point, point_strs_arr))
        if len(points_arr) == 2:
            topleft: Point = points_arr[0]
            bottomright: Point = points_arr[1]
            points_arr = [
                topleft,
                Point(bottomright.x, topleft.y),
                bottomright,
                Point(topleft.x, bottomright.y),
            ]
        return Polygon(points_arr)

    @staticmethod
    def __str_to_point(point_str: str) -> Point:
        x_str: str
        y_str: str
        x_str, y_str = point_str.split(',')
        x = float(x_str)
        y = float(y_str)
        return Point(x, y)


class PolygonFrameFilter:
    """extract structural data from an ocr file within a given polygon"""

    def __init__(self, ocr_path_in: str, points_list: str, verbosity: int):
        self.__verbosity: int = verbosity
        self.__ocr_path_in: Path = Path(ocr_path_in)
        self.__polygon: Polygon = PolygonFrameFilterUtil.str_to_polygon(points_list)
        self.__report: PolygonFrameFilterReport = PolygonFrameFilterReport()
        if self.__verbosity > 0:
            start_msg: str = f'filter strs from {ocr_path_in} between {points_list}'
            print('[INFO ] ' + start_msg)

    @property
    def ocr_file_path(self) -> Path:
        return self.__ocr_path_in

    @property
    def polygon(self) -> Polygon:
        return self.__polygon

    def process(self) -> Optional[DigitalObjectTree]:
        """apply the filter and return resulting structural data"""
        digo_result: DigitalObjectTree = to_digital_object(str(self.__ocr_path_in))
        self.__process_digo(digo_result)
        self.__create_report()
        return digo_result

    def __process_digo(self, digo: DigitalObjectTree) -> bool:
        for child_digo in digo.children:
            keep_piece: bool = self.__process_digo(child_digo)
            if not keep_piece:
                digo.remove_children(child_digo)
        if DigitalObjectLevel.WORD < digo.level < DigitalObjectLevel.PAGE:
            if len(digo.children) == 0:
                return False
            digo.dimensions = DigitalObjectUtil.calulate_dimensions_by_children(digo)
            return True
        if digo.level >= DigitalObjectLevel.PAGE:
            return True
        # Word
        if digo.level != DigitalObjectLevel.WORD:
            raise RuntimeError(f'Unknown Level: {digo.level}')
        return digo.is_in_polygon(self.polygon)

    def __create_report(self):
        for removed_element in DigitalObjectChanges.removed_elements:
            name: str = removed_element.nodeName
            try:
                value: int = self.__report.removed_elements[name]
                self.__report.removed_elements[name] = value + 1
            except KeyError:
                self.__report.removed_elements[name] = 1

        for resized_element in DigitalObjectChanges.resized_elements:
            name: str = resized_element.nodeName
            try:
                value: int = self.__report.resized_elements[name]
                self.__report.resized_elements[name] = value + 1
            except KeyError:
                self.__report.resized_elements[name] = 1

        for k, v in self.__report.removed_elements.items():
            if self.__verbosity > 1:
                print(f'[DEBUG] removed {v} {k} Elements')

        for k, v in self.__report.resized_elements.items():
            if self.__verbosity > 1:
                print(f'[DEBUG] resized {v} {k} Elements')
