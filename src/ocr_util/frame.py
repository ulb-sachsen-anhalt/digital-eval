import re
from pathlib import Path
from typing import Match, Optional, Dict
from typing import NamedTuple, List

from shapely import Polygon
from shapely.geometry import Point

from digital_eval import Piece, PieceLevel
from digital_eval.model import to_pieces, PieceUtil, PieceChanges


class PolygonFrameFilterReport(NamedTuple):
    removed_elements: Dict[str, int] = {}
    resized_elements: Dict[str, int] = {}


class PolygonFrameFilterUtil:
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

    def __init__(self, ocr_path_in: str, points_list: str):
        self.__ocr_path_in: Path = Path(ocr_path_in)
        self.__polygon: Polygon = PolygonFrameFilterUtil.str_to_polygon(points_list)
        self.__report: PolygonFrameFilterReport = PolygonFrameFilterReport()
        start_msg: str = f'filter strs from {ocr_path_in} between {points_list}'
        print('[INFO] ' + start_msg)

    @property
    def ocr_file_path(self) -> Path:
        return self.__ocr_path_in

    @property
    def polygon(self) -> Polygon:
        return self.__polygon

    def process(self) -> Optional[Piece]:
        piece_result: Piece = to_pieces(str(self.__ocr_path_in))
        self.__process_piece(piece_result)
        self.__create_report()
        return piece_result

    def __process_piece(self, piece: Piece) -> bool:
        for child_piece in piece.pieces:
            keep_piece: bool = self.__process_piece(child_piece)
            if not keep_piece:
                piece.remove_pieces(child_piece)
        if PieceLevel.WORD < piece.level < PieceLevel.PAGE:
            if len(piece.pieces) == 0:
                return False
            piece.dimensions = PieceUtil.calulate_dimensions_by_children(piece)
            return True
        elif piece.level >= PieceLevel.PAGE:
            return True
        # Word
        if piece.level != PieceLevel.WORD:
            raise Exception(f'Unknown Level: {piece.level}')
        return piece.is_in_polygon(self.polygon)

    def __create_report(self):
        for removed_element in PieceChanges.removed_elements:
            name: str = removed_element.nodeName
            try:
                value: int = self.__report.removed_elements[name]
                self.__report.removed_elements[name] = value + 1
            except KeyError as err:
                self.__report.removed_elements[name] = 1

        for resized_element in PieceChanges.resized_elements:
            name: str = resized_element.nodeName
            try:
                value: int = self.__report.resized_elements[name]
                self.__report.resized_elements[name] = value + 1
            except KeyError as err:
                self.__report.resized_elements[name] = 1

        for k, v in self.__report.removed_elements.items():
            print(f'[INFO] removed {v} {k} Elements')

        for k, v in self.__report.resized_elements.items():
            print(f'[INFO] resized {v} {k} Elements')
