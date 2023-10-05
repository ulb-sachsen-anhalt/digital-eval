#
# provided API exports
#
from .evaluation import (
    Evaluator,
    find_groundtruth,
    gather_candidates,
    Evaluator,
    report_stdout,
    ocr_to_text_legacy,
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

