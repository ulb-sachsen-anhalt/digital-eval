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
TRANSLATE_WHITESPACES = lambda s: s.translate(WHITESPACE_TRANSLATOR)
TRANSLATE_PUNCTS = lambda s : s.translate(PUNCT_TRANLATOR)
TRANSLATE_DIGITS = lambda s : s.translate(DIGIT_TRANSLATOR)
TOKENIZER = lambda s: s.split() if isinstance(s, str) else s
TOKENIZER_SET = lambda s : set(sorted(TOKENIZER(s)))

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


class Metric:
    """Basic definition of a Metric"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION, preprocessings=None) -> None:
        self.precision = precision
        self._value = None
        self.diff = None
        self.n_ref = 0
        self.label = None
        self.name = None
        self.unicode_normalization = normalization
        self.preprocessings = None
        if isinstance(preprocessings, list):
            self.preprocessings = preprocessings
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
    def value(self):
        """Evaluate lazy but only one time
        return cached result and round it
        with desired precision afterwards"""

        if self._value is None:
            if self.preprocessings:
                for _pre in self.preprocessings:
                    self._data_reference = _pre(self._data_reference)
                    self._data_candidate = _pre(self._data_candidate)
            self._calc()
        return round(self._value, self.precision)

    def _calc(self):
        """Calculate metric value
        First, normalize text on UTF-8 level
        """

class MetricCA(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'CCA'
        self.name = 'Character Accuracy'
        self.preprocessings = [TRANSLATE_WHITESPACES]

    def _calc(self):
        self._value, self.diff, self.n_ref = edit_distance(self._data_reference, self._data_candidate)


class MetricLA(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'CLA'
        self.name = 'Letter Accuracy'
        self.preprocessings = [TRANSLATE_WHITESPACES, TRANSLATE_PUNCTS, TRANSLATE_DIGITS]

    def _calc(self):
        self._value, self.diff, self.n_ref = edit_distance(self._data_reference, self._data_candidate)


class MetricWA(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'WWA'
        self.name = 'Word Token Accuracy'
        self.preprocessings = [TOKENIZER]
    
    def _calc(self):
        self._value, self.diff, self.n_ref = edit_distance(self._data_reference, self._data_candidate)


class MetricBoW(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'WBoW'
        self.name = 'Bag of Words'
        self.preprocessings = [TOKENIZER]

    def _calc(self):
        self._value, self.diff, self.n_ref = bag_of_tokens(self._data_reference, self._data_candidate)

class MetricIR(Metric):

    def __init__(self, languages=None):
        super().__init__()
        self.label = 'IRPre'
        self.name = 'IR Precision'
        self.languages = languages
        self.preprocessings = [TOKENIZER_SET, 
            partial(strip_languages_stopwords, languages=self.languages)
        ]

    def _calc(self):
        """to remind that this class needs further refinement"""
        raise NotImplementedError


class MetricPre(MetricIR):

    def __init__(self, languages=None):
        super().__init__(languages)
        self.label = 'IRPre'
        self.name = 'IR Precision'

    def _calc(self):
        self._value, self.n_ref = ir_precision(self._data_reference, self._data_candidate)


class MetricRec(MetricIR):

    def __init__(self, languages=None):
        super().__init__(languages)
        self.label = 'IRRec'
        self.name = 'IR Recall'

    def _calc(self):
        self._value, self.n_ref = ir_recall(self._data_reference, self._data_candidate)


class MetricFM(MetricIR):

    def __init__(self, languages=None):
        super().__init__(languages)
        self.label = 'IRFM'
        self.name = 'IR F-Measure'

    def _calc(self):
        self._value, self.n_ref = ir_fmeasure(self._data_reference, self._data_candidate)


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


def edit_distance(reference_data, candidate_data) -> Tuple[float, int, int]:
    """Calculate edit distance with levenshtein-distance.
    Afterwards, normalize result to reference data
    
    Works with characters and word-like tokens, where
    tokens correspond also to:
    * abbreviations  (like "Nr." or "Etg.")
    * numbers/years  (like "1899")
    * split-up words (line endings/beginnings)

    """

    distance = levenshtein(reference_data, candidate_data)
    _len_ref = len(reference_data)
    _result = _norm(len(reference_data), distance)
    return (_result, distance, _len_ref)


def bag_of_tokens(reference_tokens: List[str], candidate_tokens: List[str]) -> Tuple[float, int, int]:
    """Calculate intersection/difference
    between reference and candidate token list
    """

    n_tokens_gt = len(reference_tokens)
    diff_tokens =_diff(reference_tokens, candidate_tokens)
    n_tokens_missed = len(diff_tokens)
    hit_rate =_norm(n_tokens_gt, len(diff_tokens))
    _len_ref = len(reference_tokens)
    return (hit_rate, n_tokens_missed, _len_ref)


def _diff(gt_tokens, cd_tokens) -> List[str]:
    return list((Counter(gt_tokens) - Counter(cd_tokens)).elements())


def ir_precision(reference_data, candidate_data) -> Tuple:
    """Calculate Precision for given languages"""
    
    _prec = precision(reference_data, candidate_data) 
    # nltk actually handles this inconsistently ...
    if _prec == None:
        _prec = 0.0
    return (_prec, len(reference_data))


def ir_recall(reference_data, candidate_data) -> Tuple:
    """Calculate Recall for given languages"""
    
    # here nltk reports 0.0 if nothing recalled
    _rec = recall(reference_data, candidate_data)
    return (_rec, len(reference_data))


def ir_fmeasure(reference_data, candidate_data) -> Tuple:
    """Calculate F-Measure for given languages"""
    
    _fm = f_measure(reference_data, candidate_data)
    # required since nltk actually handles this inconsistently ...
    if _fm == None:
        _fm = 0.0
    return (_fm, len(reference_data))


def _norm(reference, errs, scale_by=100) -> float:
    """
    Normalize outcome in range 0 - 100

    * if more differences than actual len reference => 0
    * if both len reference and errs eq zero => 100
      there was nothing to find and it did detect nothing 
      (i.e. no false-positive for an image page) 
    * otherwise align to len reference
    """

    if (reference - errs) < 0:
        return 0
    if reference == 0 and errs == 0:
        return 100
    return scale_by * ((reference - errs) / reference)
