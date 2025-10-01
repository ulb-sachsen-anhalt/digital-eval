"""Test specification for common aspects of OCR assets"""

from digital_eval.model.common import (
	DigitalObjectLevel,
)


def test_digital_object_hierarchy_bottom_up():
    """Ensure behavior of digital_objects
    hierarchy from bottom up"""

    assert DigitalObjectLevel.WORD < DigitalObjectLevel.LINE
    assert DigitalObjectLevel.LINE < DigitalObjectLevel.REGION
    assert DigitalObjectLevel.WORD < DigitalObjectLevel.REGION
    assert DigitalObjectLevel.REGION < DigitalObjectLevel.PAGE
    assert DigitalObjectLevel.LINE < DigitalObjectLevel.PAGE
    assert DigitalObjectLevel.WORD < DigitalObjectLevel.PAGE


def test_digital_object_hierarchy_top_down():
    """Fix behaviour of DigitalObject Hierarchy"""

    assert DigitalObjectLevel.REGION > DigitalObjectLevel.LINE
    assert DigitalObjectLevel.LINE > DigitalObjectLevel.WORD
    assert DigitalObjectLevel.REGION > DigitalObjectLevel.WORD
    assert DigitalObjectLevel.WORD > DigitalObjectLevel.GLYPH
