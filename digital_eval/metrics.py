# -*- coding: utf-8 -*-
"""Metrics Module"""

from collections import (
    Counter
)

import string

from typing import (
    List, 
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

# unicode normalization
UC_NORMALIZATION = 'NFKD'

# punctuations to take into account
# includes
#   * regular ASCII-punctuations
#   * Dashes        \u2012-2017
#   * Quotations    \u2018-201F
PUNCTUATIONS = string.punctuation + '\u2012' + '\u2013' + '\u2014' + '\u2015' + '\u2016' + '\u2017' + '\u2018' + '\u2019' + '\u201A' + '\u201B' + '\u201C' + '\u201D' + '\u201E' + '\u201F'
# no spaces
PUNCTUATIONS = PUNCTUATIONS + '\u0020' + '\u00a0' + '\u2000' + '\u2001' + '\u2002' + '\u2003' + '\u2004' + '\u2005' + '\u2006' + '\u2007' + '\u2008' + '\u2009' + '\u200a' + '\u2028' + '\u205f' + '\u3000'
# arabib indic digits
DIGITS = string.digits + '\u0660' + '\u0661' + '\u0662' + '\u0663' + '\u0664' + '\u0665' + '\u0666' + '\u0667' + '\u0668' + '\u0669' 
# persian indic digits
DIGITS = DIGITS + '\u06f0' + '\u06f1' + '\u06f2' + '\u06f3' + '\u06f4' + '\u06f5' + '\u06f6' + '\u06f7' + '\u06f8' + '\u06f9'

# information retrival (nltk)
STOPWORDS = ['german', 'russian', 'english', 'french', 'greek', 'arabic', 'turkish', 'italian']
STOPWORDS_DEFAULT = ['german', 'english', 'arabic','russian']


class Metric:
    """Basic definition of a Metric"""

    def __init__(self, precision=2) -> None:
        self.precision = precision
        self.value = None
        self.diff = None
        self.n_ref = 0
        self.label = None
        self.name = None
        self.input_reference = None
        self.input_candidate = None
        self.data_reference = None
        self.data_candidate = None
        self.languages = None

    def calc(self):
        """Calculate metric value
        First, normalize text on UTF-8 level
        """

        self.data_reference = unicodedata.normalize(UC_NORMALIZATION, self.input_reference)
        self.data_candidate = unicodedata.normalize(UC_NORMALIZATION, self.input_candidate)


class MetricCA(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'CCA'
        self.name = 'Character Accuracy'

    def calc(self):
        super().calc()
        self.value, self.diff, _n_ref = character_accuracy(self.data_reference, self.data_candidate)
        self.n_ref = round(_n_ref, self.precision)


class MetricLA(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'CLA'
        self.name = 'Letter Accuracy'

    def calc(self):
        super().calc()
        self.data_reference = transform_string(self.data_reference)
        self.data_candidate = transform_string(self.data_candidate)
        self.value, self.diff, _n_ref = calculate_lar(self.data_reference, self.data_candidate)
        self.n_ref = round(_n_ref, self.precision)


class MetricWA(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'WWA'
        self.name = 'Word Token Accuracy'
    
    def calc(self):
        super().calc()
        self.data_reference = self.data_reference.split()
        self.data_candidate = self.data_candidate.split()
        self.value, self.diff, _n_ref = token_based(self.data_reference, self.data_candidate)
        self.n_ref = round(_n_ref, self.precision)


class MetricBoW(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'WBoW'
        self.name = 'Bag of Words'

    def calc(self):
        super().calc()
        self.data_reference = self.data_reference.split()
        self.data_candidate = self.data_candidate.split()
        self.value, self.diff, _n_ref = bag_of_tokens(self.data_reference, self.data_candidate)
        self.n_ref = round(_n_ref, self.precision)


class MetricPre(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'IRPre'
        self.name = 'IR Precision'
        self.languages = None

    def calc(self):
        super().calc()
        self.data_reference, self.data_candidate = _ir_preprocess(self.data_reference, self.data_candidate, self.languages)
        self.value, _n_ref = ir_precision(self.data_reference, self.data_candidate)
        self.n_ref = round(_n_ref, self.precision)
        self.data_reference = sorted(self.data_reference)
        self.data_candidate = sorted(self.data_candidate)


class MetricRec(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'IRRec'
        self.name = 'IR Recall'
        self.languages = None

    def calc(self):
        super().calc()
        self.data_reference, self.data_candidate = _ir_preprocess(self.data_reference, self.data_candidate, self.languages)
        self.value, _n_ref = ir_recall(self.data_reference, self.data_candidate)
        self.n_ref = round(_n_ref, self.precision)
        self.data_reference = sorted(self.data_reference)
        self.data_candidate = sorted(self.data_candidate)


class MetricFM(Metric):

    def __init__(self):
        super().__init__()
        self.label = 'IRFM'
        self.name = 'IR F-Measure'
        self.languages = None

    def calc(self):
        super().calc()
        self.data_reference, self.data_candidate = _ir_preprocess(self.data_reference, self.data_candidate, self.languages)
        self.value, _n_ref = ir_fmeasure(self.data_reference, self.data_candidate)
        self.n_ref = round(_n_ref, self.precision)
        self.data_reference = sorted(self.data_reference)
        self.data_candidate = sorted(self.data_candidate)


def transform_string(the_content):
    """Perform recent character transformations"""

    punct_translator = str.maketrans('', '', PUNCTUATIONS)
    digit_translator = str.maketrans('', '', DIGITS)
    the_content = the_content.translate(punct_translator)
    the_content = the_content.translate(digit_translator)
    return the_content


def character_accuracy(gt_str, test_str) -> Tuple[float, int, int]:
    """Calculate common levenshtein-distance"""

    distance = levenshtein(gt_str, test_str)
    _len_ref = len(gt_str)
    _result = _norm(len(gt_str), distance)
    return (_result, distance, _len_ref)


def calculate_lar(reference: str, candidate: str) -> Tuple[float, int, int]:
    """Apply additional preprocess to both datasets"""
    
    distance = levenshtein(reference, candidate)
    _len_ref = len(reference)
    _result = _norm(_len_ref, distance)
    return (_result, distance, _len_ref)


def token_based(reference_tokens: List[str], candidate_tokens: List[str]) -> Tuple[float, int, int]:
    """Levenshtein on word-level
    Ratio of token misses of two texts.
    Tokens correspond often words, but also to:
    * abbreviations (like "Nr." or "Etg.")
    * numbers/years (like "1899")
    * splitted words (line endings/beginnings)
    """

    _len_ref = len(reference_tokens)
    distance = levenshtein(reference_tokens, candidate_tokens)
    return (_norm(_len_ref, distance), distance, _len_ref)


def bag_of_tokens(reference_tokens: List[str], candidate_tokens: List[str]) -> Tuple[float, int, int]:
    """Calculate intersection/difference
    between GT and Candidate Text"""

    n_tokens_gt = len(reference_tokens)
    diff_tokens =_diff(reference_tokens, candidate_tokens)
    n_tokens_missed = len(diff_tokens)
    hit_rate = 100 * (n_tokens_gt - len(diff_tokens)) / n_tokens_gt
    _len_ref = len(reference_tokens)
    return (hit_rate, n_tokens_missed, _len_ref)


def _diff(gt_tokens, cd_tokens) -> List[str]:
    return list((Counter(gt_tokens) - Counter(cd_tokens)).elements())


def _setup_stopwords(word_mappings=STOPWORDS):
    """Helper Function to ensure NLTK stopword data available
    """
    try:
        for mapping in word_mappings: 
            stopwords.words(mapping)
    except LookupError:
        download('stopwords')


def _ir_preprocess(gt_data, test_data, languages):
    """Common Preprocessing for Information Retrival Metrics"""
    _setup_stopwords()
    if languages == None:
        languages = STOPWORDS_DEFAULT
    _stopwords = set([_all_words
                      for _lang in languages
                      for _all_words in stopwords.words(_lang)]
                    )  
    # propably feed with list strings
    if isinstance(gt_data, list):
        gt_data = ' '.join(gt_data)
    if isinstance(test_data, list):
        test_data = ' '.join(test_data)
    gt_tokens = set(gt_data.split()) - _stopwords
    test_tokens = set(test_data.split()) - _stopwords
    return (gt_tokens, test_tokens)


def ir_precision(refrence_data, candidate_data) -> Tuple:
    """Calculate Precision for given languages"""
    
    _prec = precision(refrence_data, candidate_data) 
    # nltk actually handles this inconsistently ...
    if _prec == None:
        _prec = 0.0
    return (_prec, len(refrence_data))


def ir_recall(refrence_data, candidate_data) -> Tuple:
    """Calculate Recall for given languages"""
    
    # here nltk reports 0.0 if nothing recalled
    _rec = recall(refrence_data, candidate_data)
    return (_rec, len(refrence_data))


def ir_fmeasure(refrence_data, candidate_data) -> Tuple:
    """Calculate F-Measure for given languages"""
    
    _fm = f_measure(refrence_data, candidate_data)
    # required since nltk actually handles this inconsistently ...
    if _fm == None:
        _fm = 0.0
    return (_fm, len(refrence_data))


def _norm(reference, errs, scale_by=100) -> float:
    '''Normalize outcome based on specific reference into range 0 - 100'''
    if (reference - errs) < 0:
        return 0
    return scale_by * ((reference - errs) / reference)
