"""digital eval main API"""
__version__ = '1.6.0'
from .evaluation import (
    Evaluator,
    report_stdout,
)
from .resolve import (
    find_groundtruth,
    gather_candidates,
)
from .metrics import (
    MetricChars,
    MetricLetters,
    MetricWords,
    MetricBoW,
    MetricIRPre,
    MetricIRRec,
)

from .preprocessing import UC_NORMALIZATION_DEFAULT
