"""digital eval main API"""
__version__ = '1.6.0'
from .evaluation import (
    Evaluator,
    find_groundtruth,
    gather_candidates,
    report_stdout,
)

from .metrics import (
    UC_NORMALIZATION_DEFAULT,
    MetricChars,
    MetricLetters,
    MetricWords,
    MetricBoW,
    MetricIRPre,
    MetricIRRec,
    MetricIRFM,
)
