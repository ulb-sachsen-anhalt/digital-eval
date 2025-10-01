"""main module"""

from pathlib import PurePath

from digital_eval.model.digital_object_model import DigitalObjectTree
from digital_eval.model.digital_object_util import DigitalObjectUtil


def to_digital_object(path_in: str) -> DigitalObjectTree:
    """Transform given input from various formats
    into internal Piece-Representation"""
    return DigitalObjectUtil.to_digital_objects(path_in)


def from_digital_object(root_digo: DigitalObjectTree, path_out: str = None) -> PurePath:
    return DigitalObjectUtil.from_digital_objects(root_digo, path_out)
