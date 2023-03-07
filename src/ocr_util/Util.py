from .datatypes import Point2D


class Util:

    @staticmethod
    def str_to_point_2d(point_str: str) -> Point2D:
        x_str: str
        y_str: str
        x_str, y_str = point_str.split(',')
        x = float(x_str)
        y = float(y_str)
        return Point2D(x, y)
