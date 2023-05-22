#
# provided API exports
#
from .evaluation import (
    Evaluator,
    find_groundtruth,
    gather_candidates,
    Evaluator,
    report_stdout,
    ocr_to_text,
)

from .metrics import (
    accuracy_for,
    error_for,
    UC_NORMALIZATION_DEFAULT,
    MetricChars,
    MetricLetters,
    MetricWords,
    MetricBoW,
    MetricIRPre,
    MetricIRRec,
    MetricIRFM,
)

from .model import (
    Piece,
    PieceLevel,
    PieceContent,
    PieceUtil,
    PieceDimensions,
    to_pieces,
    from_pieces,
)

from .model_legacy import (
    OCRToken,
    OCRWord,
    OCRWordLine,
    OCRRegion,
    BoundingBox,
)
