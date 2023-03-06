from typing import NamedTuple, Tuple, List

Tuple2D = Tuple[float, float]


class Point2D(NamedTuple):
    x: float
    y: float

    def to_tuple(self) -> Tuple2D:
        return self.x, self.y


Point2DList = List[Point2D]
