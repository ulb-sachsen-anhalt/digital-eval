# -*- coding: utf-8 -*-
"""Metrics Module"""

from __future__ import annotations

from collections import (
    Counter
)
from functools import partial

import string

from typing import (
    List,
    Set, Callable,
)

import unicodedata

from nltk import (
    download,
)
from nltk.corpus import (
    stopwords
)
from nltk.metrics import (
    recall,
    f_measure
)
from nltk.metrics import precision as nltk_precision

from rapidfuzz.distance import (
    Levenshtein
)

# Python3 standard Unicode Normalization
#
UC_NORMALIZATION_DEFAULT = 'NFC'
UC_NORMALIZATION_NFKD = 'NFKD'

# whitespaces
#
# usual spatium and special control sequences
WHITESPACES = string.whitespace

WHITESPACES_EXCLUDING_BLANK_CHARS = WHITESPACES[1:]

# punctuations
#
#   * regular ASCII-punctuations
#   * Dashes        \u2012-2017
#   * Quotations    \u2018-201F
PUNCTUATIONS = string.punctuation + '\u2012' + '\u2013' + '\u2014' + '\u2015' + '\u2016' + \
               '\u2017' + '\u2018' + '\u2019' + '\u201A' + '\u201B' + \
               '\u201C' + '\u201D' + '\u201E' + '\u201F'
# no special line break delimiter
PUNCTUATIONS = PUNCTUATIONS + '\u2E17'  # DOUBLE OBLIQUE HYPHEN
# no spaces
PUNCTUATIONS = PUNCTUATIONS + '\u0020' + '\u00a0' + '\u2000' + \
               '\u2001' + '\u2002' + '\u2003' + '\u2004' + '\u2005' + \
               '\u2006' + '\u2007' + '\u2008' + '\u2009' + \
               '\u200a' + '\u2028' + '\u205f' + '\u3000'

# digits
#
#   * ASCII digits
#   * arabic digits
#   * persian / indic digits
DIGITS = string.digits + '\u0660' + '\u0661' + '\u0662' + '\u0663' + \
         '\u0664' + '\u0665' + '\u0666' + '\u0667' + '\u0668' + '\u0669'
# persian indic digits
DIGITS = DIGITS + '\u06f0' + '\u06f1' + '\u06f2' + '\u06f3' + \
         '\u06f4' + '\u06f5' + '\u06f6' + '\u06f7' + '\u06f8' + '\u06f9'

# filter mechanics
#
# via Python3 string translation maps
WHITESPACE_TRANSLATOR = str.maketrans('', '', WHITESPACES)
WHITESPACE_EXCLUDING_BLANK_CHARS_TRANSLATOR = str.maketrans('', '', WHITESPACES_EXCLUDING_BLANK_CHARS)
PUNCT_TRANLATOR = str.maketrans('', '', PUNCTUATIONS)
DIGIT_TRANSLATOR = str.maketrans('', '', DIGITS)


def _filter_whitespaces(a_str) -> str:
    return a_str.translate(WHITESPACE_TRANSLATOR)


def _filter_whitespaces_excluding_blank_chars(a_str) -> str:
    return a_str.translate(WHITESPACE_EXCLUDING_BLANK_CHARS_TRANSLATOR)


def _filter_puncts(a_str) -> str:
    return a_str.translate(PUNCT_TRANLATOR)


def _filter_digits(a_str) -> str:
    return a_str.translate(DIGIT_TRANSLATOR)


def _tokenize(a_str) -> List[str]:
    return a_str.split() if isinstance(a_str, str) else a_str


def _tokenize_to_sorted_set(a_str) -> Set[str]:
    return set(sorted(_tokenize(a_str)))


#
# information retrieval (nltk)
#
NLTK_STOPWORDS = [
    'german',
    'russian',
    'english',
    'french',
    'greek',
    'arabic',
    'turkish',
    'italian']
STOPWORDS_DEFAULT = ['german', 'english', 'arabic', 'russian']


def get_stopwords(nltk_mappings=NLTK_STOPWORDS, languages=None) -> Set[str]:
    """Helper Function to gather NLTK stopword data
    * ensure stopwords files are locally available
    * extract them as set
    """
    try:
        for mapping in nltk_mappings:
            stopwords.words(mapping)
    except LookupError:
        download('stopwords')
    if languages is None:
        languages = STOPWORDS_DEFAULT
    _stopwords = {_all_words
                  for _lang in languages
                  for _all_words in stopwords.words(_lang)
                  }
    return _stopwords


def _strip_languages_stopwords(tokens, languages):
    return tokens - get_stopwords(languages=languages)


def _strip_stopwords_for(languages):
    return partial(_strip_languages_stopwords, languages=languages)


def normalize_unicode(input_str: str, uc_norm_by=UC_NORMALIZATION_DEFAULT) -> str:
    """Apply basic unicode normalization
    """

    if uc_norm_by is not None:
        input_str = unicodedata.normalize(uc_norm_by, input_str)
    return input_str


def transform_string(the_content):
    """Perform recent character transformations"""

    punct_translator = str.maketrans('', '', PUNCTUATIONS)
    digit_translator = str.maketrans('', '', DIGITS)
    the_content = the_content.translate(punct_translator)
    the_content = the_content.translate(digit_translator)
    return the_content


class DigitalEvalMetricException(Exception):
    """Mark Exception during validation/calculating metrics"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def _inspect_calculation_object(an_object):
    if not isinstance(an_object, OCRDifferenceMetric):
        _msg = f"{an_object} is no instance of {OCRDifferenceMetric}!"
        raise DigitalEvalMetricException(_msg)
    try:
        _diff = an_object.diff
        _ref = an_object.n_ref
        if _diff is None or _diff < 0:
            raise DigitalEvalMetricException(f"invalid diff: {_diff}!")
        if _ref is None or _ref < 0:
            raise DigitalEvalMetricException(f"invalid ref: {_ref}!")
    except AttributeError as _ae:
        raise DigitalEvalMetricException(_ae.args[0]) from _ae


def accuracy_for(metric_obj: OCRDifferenceMetric) -> float:
    """Calculate accuracy as ratio of
    correct items, with correct items
    being expected items minus
    number of differences.

    Respect following corner cases:
    * if less correct items than differences => 0
    * if both correct items and differences eq zero => 1
      means: nothing to find and it did detect nothing
      (i.e. no false-positives)

    Args:
        metric_obj (OCRDifferenceMetric): object containing information
        about reference data and difference

    Returns:
        float: accuracy in range 0.0 - 1.0
    """

    _inspect_calculation_object(metric_obj)
    diffs = metric_obj.diff
    n_refs = len(metric_obj._data_reference)
    return _calculate(lambda d, n: (n - d) / n, diffs, n_refs)


def error_for_bow(metric_obj: OCRDifferenceMetric) -> float:
    _inspect_calculation_object(metric_obj)
    diffs = metric_obj.diff
    n_refs = len(metric_obj._data_reference) + len(metric_obj._data_candidate)
    return _calculate(lambda d, n: d / n, diffs, n_refs)


def accuracy_for_bow(metric_obj: OCRDifferenceMetric) -> float:
    _inspect_calculation_object(metric_obj)
    diffs = metric_obj.diff
    n_refs = len(metric_obj._data_reference) + len(metric_obj._data_candidate)
    return _calculate(lambda d, n: 1 - (d / n), diffs, n_refs)


def error_for(metric_obj: OCRDifferenceMetric) -> float:
    """Calculate error as ratio of
    difference and number of
    expected items.

    Respect following corner cases:
    * if less expected items than differences => 0
    * if both expected items and differences eq zero => 1
      means: nothing to find and detected nothing
      (i.e. no false-positives)

    Args:
        metric_obj (OCRDifferenceMetric): object containing information
        about reference data and difference

    Returns:
        float: error in range 0.0 - 1.0
    """
    _inspect_calculation_object(metric_obj)
    diffs = metric_obj.diff
    n_refs = len(metric_obj._data_reference)
    return _calculate(lambda d, n: d / n, diffs, n_refs)


def _calculate(calculate: Callable[[int, int], float], diffs: int, n_refs: int) -> float:
    if (n_refs - diffs) < 0:
        return 0.0
    if n_refs == 0 and diffs == 0:
        return 1.0
    elif n_refs > 0:
        return calculate(diffs, n_refs)
    else:
        return 0.0


def norm_to_scale(value, scale_by) -> float:
    """Normalize outcome in range 0 - scale_by"""
    return value * scale_by


def norm_percentual(value):
    """Normalize value in between 0 - 100"""
    return partial(norm_to_scale, scale_by=100)(value)


class OCRDifferenceMetric:
    """Basic definition of a OCRDifferenceMetric"""

    def __init__(
            self,
            precision=2,
            normalization=UC_NORMALIZATION_DEFAULT,
            calc_func=accuracy_for,
            preprocessings=None,
            postprocessings=None
    ) -> None:
        self.precision = precision
        self._value = None
        self.diff = None
        self._label = None
        self.unicode_normalization = normalization
        self.preprocessings = []
        if isinstance(preprocessings, list):
            self.preprocessings = preprocessings
        self.calc_func = calc_func
        self.postprocessings = [norm_percentual]
        if isinstance(postprocessings, list):
            self.postprocessings = postprocessings
        self.input_reference = None
        self.input_candidate = None
        self._data_reference = None
        self._data_candidate = None
        self.languages = None

    @property
    def reference(self):
        """Reference/Groundtruth data"""
        return self._data_reference

    @reference.setter
    def reference(self, value):
        self.input_reference = value
        self._data_reference = normalize_unicode(
            value, self.unicode_normalization)
        self._value = None

    @property
    def candidate(self):
        """Candidate data"""
        return self._data_candidate

    @candidate.setter
    def candidate(self, value):
        self.input_candidate = value
        self._data_candidate = normalize_unicode(value)
        self._value = None

    @property
    def label(self):
        """Metric's label"""
        return self._label

    @property
    def n_ref(self) -> int:
        """Number of current reference data"""
        if not hasattr(
                self, '_data_reference') or self._data_reference is None:
            raise DigitalEvalMetricException("invalid reference data!")
        return len(self._data_reference)

    @n_ref.setter
    def n_ref(self, value):
        """Exists only for testing purposes"""

        self._data_reference = 't' * value

    @property
    def value(self):
        """Evaluate each time and round
        with desired precision afterwards"""

        if self._value is None:
            if self.preprocessings:
                for _pre in self.preprocessings:
                    self._data_reference = _pre(self._data_reference)
                    self._data_candidate = _pre(self._data_candidate)
            self._forward()
            if self.calc_func:
                self._value = self.calc_func(self)
            else:
                self._value = self.diff
            if self.postprocessings:
                for _post in self.postprocessings:
                    self._value = _post(self._value)
        return round(self._value, self.precision)

    def _forward(self):
        """Calculate metric's value
        remember this needs further refinement"""
        raise NotImplementedError


class MetricChars(OCRDifferenceMetric):
    """Calculate plain sequent character based metric"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings)
        self._label = 'Cs'
        self.name = 'Characters'
        self.preprocessings = [_filter_whitespaces_excluding_blank_chars]
        self.postprocessings = [norm_percentual]

    def _forward(self):
        self.diff = edit_distance(self._data_reference, self._data_candidate)


class MetricLetters(OCRDifferenceMetric):
    """Calculate metric for only a certain sub-set of
    character sequence"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings)
        self._label = 'Ls'
        self.preprocessings = [
            _filter_whitespaces,
            _filter_puncts,
            _filter_digits]

    def _forward(self):
        self.diff = edit_distance(self._data_reference, self._data_candidate)


class MetricWords(OCRDifferenceMetric):
    """Calculate metric for a sequence of word tokens"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings)
        self._label = 'Ws'
        self.preprocessings = [_tokenize]

    def _forward(self):
        self.diff = edit_distance(self._data_reference, self._data_candidate)


class MetricBoW(OCRDifferenceMetric):
    """Calculate metric for a multiset of word tokens"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings)
        self._label = 'BoWs'
        self.preprocessings = [_tokenize]

    def _forward(self):
        self.diff = bag_of_tokens(self._data_reference, self._data_candidate)


class MetricIR(OCRDifferenceMetric):
    """Calculate information retrival metrics"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings)
        self.languages = languages
        self.preprocessings = [_tokenize_to_sorted_set,
                               _strip_stopwords_for(self.languages)
                               ]
        # no aligning required, we rely on nltk
        self.calc_func = None
        # no percentual value
        self.postprocessings = []

    def _forward(self):
        """to remind that this class needs further refinement"""
        raise NotImplementedError


class MetricIRPre(MetricIR):
    """Calculate precision"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings,
            languages)
        self._label = 'Pre'

    def _forward(self):
        self.diff = ir_precision(self._data_reference, self._data_candidate)


class MetricIRRec(MetricIR):
    "Calculate recall"

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings,
            languages)
        self._label = 'Rec'

    def _forward(self):
        self.diff = ir_recall(self._data_reference, self._data_candidate)


class MetricIRFM(MetricIR):
    """Calculate harmonic mean for precision/recall"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT, calc_func=accuracy_for,
                 preprocessings=None, postprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            calc_func,
            preprocessings,
            postprocessings,
            languages)
        self._label = 'FM'

    def _forward(self):
        self.diff = ir_fmeasure(self._data_reference, self._data_candidate)


def edit_distance(reference_data, candidate_data) -> int:
    """Calculate edit distance with levenshtein-distance.
    as sum of edit operations required to get from
    candidate to reference string / token_list

    Works with characters and word-like tokens, where
    tokens correspond also to:
    * abbreviations  (like "Nr." or "Etg.")
    * numbers/years  (like "1899")
    * split-up words (line endings/beginnings)
    """

    return Levenshtein.distance(reference_data, candidate_data)


def bag_of_tokens(reference_tokens: List[str],
                  candidate_tokens: List[str]) -> int:
    """Calculate intersection/difference
    between reference and candidate token list
    """
    false_negatives: List[str] = _diff(reference_tokens, candidate_tokens)
    false_positives: List[str] = _diff(candidate_tokens, reference_tokens)
    return len(false_negatives) + len(false_positives)


def _diff(gt_tokens, cd_tokens) -> List[str]:
    return list((Counter(gt_tokens) - Counter(cd_tokens)).elements())


def ir_precision(reference_data, candidate_data) -> float:
    """Calculate Precision for given languages"""

    _prec = nltk_precision(reference_data, candidate_data)
    # nltk actually handles this inconsistently ...
    if _prec is None:
        _prec = 0.0
    return _prec


def ir_recall(reference_data, candidate_data) -> float:
    """Calculate Recall for given languages"""

    # here nltk reports 0.0 if nothing recalled
    return recall(reference_data, candidate_data)


def ir_fmeasure(reference_data, candidate_data) -> float:
    """Calculate F-Measure for given languages"""

    _fm = f_measure(reference_data, candidate_data)
    # required since nltk actually handles this inconsistently ...
    if _fm is None:
        _fm = 0.0
    return _fm
