"""digital eval main API"""

__version__ = "2.0.0-dev"

from .evaluation import (
    Evaluator,
    EvalEntry,
    report_stdout,
)
from .aggregation import (
    AggregationDimension,
    AggregationStrategy,
    DirectoryHierarchyExtractor,
    TypeExtractor,
    CustomMetadataExtractor,
    FilenamePatternExtractor,
    METSModsExtractor,
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
