# -*- coding: utf-8 -*-
"""Metrics Module"""

from collections import (
    Counter
)
from functools import partial

import string

from typing import (
    List, 
    Set,
    Tuple, 
)

import unicodedata

from nltk import (
    download,
)
from nltk.corpus import (
    stopwords
)
from nltk.metrics import (
    precision,
    recall,
    f_measure
)
from rapidfuzz.string_metric import (
    levenshtein
)


# Python3 standard Unicode Normalization
# 
UC_NORMALIZATION = 'NFKD'


# whitespaces
#
# usual spatium and special control sequences
WHITESPACES = string.whitespace


# punctuations
#
#   * regular ASCII-punctuations
#   * Dashes        \u2012-2017
#   * Quotations    \u2018-201F
PUNCTUATIONS = string.punctuation + '\u2012' + '\u2013' + '\u2014' + '\u2015' + '\u2016' + '\u2017' + '\u2018' + '\u2019' + '\u201A' + '\u201B' + '\u201C' + '\u201D' + '\u201E' + '\u201F'
# no special line break delimiter
PUNCTUATIONS = PUNCTUATIONS + '\u2E17'  # DOUBLE OBLIQUE HYPHEN
# no spaces
PUNCTUATIONS = PUNCTUATIONS + '\u0020' + '\u00a0' + '\u2000' + '\u2001' + '\u2002' + '\u2003' + '\u2004' + '\u2005' + '\u2006' + '\u2007' + '\u2008' + '\u2009' + '\u200a' + '\u2028' + '\u205f' + '\u3000'


# digits
#
#   * ASCII digits 
#   * arabic digits
#   * persian / indic digits
DIGITS = string.digits + '\u0660' + '\u0661' + '\u0662' + '\u0663' + '\u0664' + '\u0665' + '\u0666' + '\u0667' + '\u0668' + '\u0669' 
# persian indic digits
DIGITS = DIGITS + '\u06f0' + '\u06f1' + '\u06f2' + '\u06f3' + '\u06f4' + '\u06f5' + '\u06f6' + '\u06f7' + '\u06f8' + '\u06f9'


# filter mechanics
#
# via Python3 string translation maps
WHITESPACE_TRANSLATOR =str.maketrans('','', WHITESPACES)
PUNCT_TRANLATOR = str.maketrans('', '', PUNCTUATIONS)
DIGIT_TRANSLATOR = str.maketrans('', '', DIGITS)


def filter_whitespaces(a_str) -> str:
    return a_str.translate(WHITESPACE_TRANSLATOR)


def filter_puncts(a_str) -> str:
    return a_str.translate(PUNCT_TRANLATOR)


def filter_digits(a_str) -> str:
    return a_str.translate(DIGIT_TRANSLATOR)


def tokenize(a_str) -> List[str]:
    return a_str.split() if isinstance(a_str, str) else a_str


def tokenize_to_sorted_set(a_str) -> Set[str]:
    return set(sorted(tokenize(a_str)))


#
# information retrieval (nltk)
#
NLTK_STOPWORDS = ['german', 'russian', 'english', 'french', 'greek', 'arabic', 'turkish', 'italian']
STOPWORDS_DEFAULT = ['german', 'english', 'arabic','russian']
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
    if languages == None:
        languages = STOPWORDS_DEFAULT
    _stopwords = set([_all_words
                      for _lang in languages
                      for _all_words in stopwords.words(_lang)]
                    )  
    return _stopwords


def strip_languages_stopwords(tokens, languages):
    return tokens - get_stopwords(languages=languages)


def strip_stopwords_for(languages):
    return partial(strip_languages_stopwords, languages=languages)


def normalize_unicode(input_str: str, uc_norm_by=UC_NORMALIZATION) -> str: 
    """Apply basic unicode normalization
    """

    if uc_norm_by is not None:
        input_str = unicodedata.normalize(UC_NORMALIZATION, input_str)
    return input_str


def transform_string(the_content):
    """Perform recent character transformations"""

    punct_translator = str.maketrans('', '', PUNCTUATIONS)
    digit_translator = str.maketrans('', '', DIGITS)
    the_content = the_content.translate(punct_translator)
    the_content = the_content.translate(digit_translator)
    return the_content


def align_difference_to_reference(the_obj) -> float:
    """calculate ratio of reference number and metric value
    in range between 0 .. 1.0

    Respect following corner cases:

    * if more errors than references => 0
    * if both references and errors eq zero => 1
      means: nothing to find and it did detect nothing 
      (i.e. no false-positives) 
    * otherwise subtract errors and align diff to reference 
    """

    diffs = the_obj.diff
    n_refs = len(the_obj._data_reference)
    if (n_refs - diffs) < 0:
        return 0
    if n_refs == 0 and diffs == 0:
        return 1.0
    elif n_refs > 0:
        return (n_refs - diffs) / n_refs


def norm_to_scale(value, scale_by) -> float:
    """
    Normalize outcome in range 0 - scale_bound
    """

    return value * scale_by


def norm_percentual(value):
    return partial(norm_to_scale, scale_by=100)(value)


class OCRDifferenceMetric:
    """Basic definition of a OCRDifferenceMetric"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION, 
        preprocessings=None, align_func=align_difference_to_reference,
        postprocessings=None) -> None:
        self.precision = precision
        self._value = None
        self.diff = None
        self._label = None
        self.unicode_normalization = normalization
        self.preprocessings = []
        if isinstance(preprocessings, list):
            self.preprocessings = preprocessings
        self.align_func = align_func
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
        return self._data_reference

    @reference.setter
    def reference(self, value):
        self.input_reference = value
        self._data_reference = normalize_unicode(value, self.unicode_normalization)

    @property
    def candidate(self):
        return self._data_candidate

    @candidate.setter
    def candidate(self, value):
        self.input_candidate = value
        self._data_candidate = normalize_unicode(value)

    @property
    def label(self):
        return self._label

    @property
    def n_ref(self):
        return len(self._data_reference)

    @property
    def value(self):
        """Evaluate each time and round
        with desired precision afterwards"""

        if self._value is None:
            if self.preprocessings:
                for _pre in self.preprocessings:
                    self._data_reference = _pre(self._data_reference)
                    self._data_candidate = _pre(self._data_candidate)
            self._calc()
            if self.align_func:
                self._value = self.align_func(self)
            else:
                self._value = self.diff
            if self.postprocessings:
                for _post in self.postprocessings:
                    self._value = _post(self._value)
        return round(self._value, self.precision)

    def _calc(self):
        """Calculate metric's value
        remember this needs further refinement"""
        raise NotImplementedError


class MetricChars(OCRDifferenceMetric):

    def __init__(self):
        super().__init__()
        self._label = 'Cs'
        self.name = 'Characters'
        self.preprocessings = [filter_whitespaces]
        self.postprocessings = [norm_percentual]

    def _calc(self):
        self.diff = edit_distance(self._data_reference, self._data_candidate)


class MetricLetters(OCRDifferenceMetric):

    def __init__(self):
        super().__init__()
        self._label = 'Ls'
        self.preprocessings = [filter_whitespaces, filter_puncts, filter_digits]

    def _calc(self):
        self.diff = edit_distance(self._data_reference, self._data_candidate)


class MetricWords(OCRDifferenceMetric):

    def __init__(self):
        super().__init__()
        self._label = 'Ws'
        self.preprocessings = [tokenize]
    
    def _calc(self):
        self.diff = edit_distance(self._data_reference, self._data_candidate)


class MetricBoW(OCRDifferenceMetric):

    def __init__(self):
        super().__init__()
        self._label = 'BoWs'
        self.preprocessings = [tokenize]

    def _calc(self):
        self.diff = bag_of_tokens(self._data_reference, self._data_candidate)


class MetricIR(OCRDifferenceMetric):

    def __init__(self, languages=None):
        super().__init__()
        self.languages = languages
        self.preprocessings = [tokenize_to_sorted_set, 
            strip_stopwords_for(self.languages)
        ]
        # no aligning required, we rely on nltk
        self.align_func = None
        # no percentual value
        self.postprocessings = []

    def _calc(self):
        """to remind that this class needs further refinement"""
        raise NotImplementedError


class MetricIRPre(MetricIR):

    def __init__(self, languages=None):
        super().__init__(languages)
        self._label = 'Pre'

    def _calc(self):
        self.diff = ir_precision(self._data_reference, self._data_candidate)


class MetricIRRec(MetricIR):

    def __init__(self, languages=None):
        super().__init__(languages)
        self._label = 'Rec'

    def _calc(self):
        self.diff = ir_recall(self._data_reference, self._data_candidate)


class MetricIRFM(MetricIR):

    def __init__(self, languages=None):
        super().__init__(languages)
        self._label = 'FM'

    def _calc(self):
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

    return levenshtein(reference_data, candidate_data)


def bag_of_tokens(reference_tokens: List[str], candidate_tokens: List[str]) -> int:
    """Calculate intersection/difference
    between reference and candidate token list
    """

    return len(_diff(reference_tokens, candidate_tokens))


def _diff(gt_tokens, cd_tokens) -> List[str]:
    return list((Counter(gt_tokens) - Counter(cd_tokens)).elements())


def ir_precision(reference_data, candidate_data) -> float:
    """Calculate Precision for given languages"""
    
    _prec = precision(reference_data, candidate_data) 
    # nltk actually handles this inconsistently ...
    if _prec == None:
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
    if _fm == None:
        _fm = 0.0
    return _fm
