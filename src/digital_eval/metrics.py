# -*- coding: utf-8 -*-
"""Metrics Module"""

from __future__ import annotations

import collections
import functools
import string
import typing
import unicodedata
import xml.dom.minidom

import nltk
import nltk.corpus as nltk_corp
from nltk.metrics import precision as nltk_precision
from nltk.metrics import (
     recall,
     f_measure
 )
import rapidfuzz.distance.Levenshtein as rfls

import digital_object as do

from digital_eval.dictionary_metrics.common import LANGUAGE_KEY_DEFAULT
from digital_eval.dictionary_metrics.language_tool.LanguageTool import LanguageTool

# Python3 standard Unicode Normalization
#
UC_NORMALIZATION_DEFAULT = 'NFC'
UC_NORMALIZATION_NFKD = 'NFKD'

# whitespaces
#
# usual spatium and special control sequences
WHITESPACES = string.whitespace

WHITESPACES_EXCL_BLANK_CHARS = WHITESPACES[1:]

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
WHITESPACE_TRNSL = str.maketrans('', '', WHITESPACES)
WHITESPACE_EXCL_BLANK_CHARS_TRNSL = str.maketrans('', '', WHITESPACES_EXCL_BLANK_CHARS)
PUNCT_TRNSL = str.maketrans('', '', PUNCTUATIONS)
DIGIT_TRNSL = str.maketrans('', '', DIGITS)


def _filter_whitespaces(a_str) -> str:
    return a_str.translate(WHITESPACE_TRNSL)


def _filter_whitespaces_excluding_blank_chars(a_str) -> str:
    return a_str.translate(WHITESPACE_EXCL_BLANK_CHARS_TRNSL)


def _filter_puncts(a_str) -> str:
    return a_str.translate(PUNCT_TRNSL)


def _filter_digits(a_str) -> str:
    return a_str.translate(DIGIT_TRNSL)


def _tokenize(a_str) -> typing.List[str]:
    return a_str.split() if isinstance(a_str, str) else a_str


def _tokenize_to_sorted_set(a_str) -> typing.Set[str]:
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


def get_stopwords(nltk_mappings=None, languages=None) -> typing.Set[str]:
    """Helper Function to gather NLTK stopword data
    * ensure stopwords files are locally available
    * extract them as set
    """
    if nltk_mappings is None:
        nltk_mappings = NLTK_STOPWORDS
    try:
        for mapping in nltk_mappings:
            nltk_corp.stopwords.words(mapping)
    except LookupError:
        nltk.download('stopwords')
    if languages is None:
        languages = STOPWORDS_DEFAULT
    _stopwords = {_all_words
                  for _lang in languages
                  for _all_words in nltk_corp.stopwords.words(_lang)
                  }
    return _stopwords


def _strip_languages_stopwords(tokens, languages):
    return tokens - get_stopwords(languages=languages)


def _strip_stopwords_for(languages):
    return functools.partial(_strip_languages_stopwords, languages=languages)


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


def digital_object_to_dict_text(file_path: str, frame=None, oneliner=False) -> typing.Tuple:
    line_texts: typing.List[str]
    len_lines: int
    line_texts, len_lines = digital_object_to_text(file_path=file_path, frame=frame, oneliner=False)
    non_empty_lines: typing.List[str] = [line_text for line_text in line_texts if len(line_text) > 0]
    lines_sanitized_wraps: typing.List[str] = _sanitize_wraps(non_empty_lines)
    lines_sanitized_chars: typing.List[str] = _sanitize_chars(lines_sanitized_wraps)
    text = ' '.join(lines_sanitized_chars) if oneliner else lines_sanitized_chars
    return text, len_lines


def digital_object_to_text(file_path, frame=None, oneliner=True) -> typing.Tuple:
    """Wrap OCR-Data Comparison"""

    try:
        top_digo: do.DigitalObjectTree = do.to_digital_object(file_path)
        # explicit filter frame?
        if not frame:
            frame = top_digo.dimensions
        elif len(frame) == 2:
            frame = [[frame[0][0], frame[0][1]],
                     [frame[1][0], frame[0][1]],
                     [frame[1][0], frame[1][1]],
                     [frame[0][0], frame[1][1]]]
        frame_digo = do.DigitalObjectTree()
        frame_digo.dimensions = frame
        filter_word_pieces(frame_digo, top_digo)
        the_lines = _get_line_digos_from_digo(top_digo)
        if oneliner:
            return top_digo.transcription, len(the_lines)
        else:
            return [line.transcription for line in the_lines], len(the_lines)
    except xml.parsers.expat.ExpatError as _:
        with open(file_path, mode='r', encoding='utf-8') as fhandle:
            text_lines = fhandle.readlines()
            if oneliner:
                text_lines = ' '.join([l.strip() for l in text_lines])
            return text_lines, len(text_lines)
    except RuntimeError as exc:
        raise RuntimeError(f"{file_path}: {exc}") from exc


def filter_word_pieces(frame, current) -> int:
    _filtered = 0
    _tmp_stack = []
    _total_stack = []
    # stack all items
    _total_stack.append(current)
    _tmp_stack.append(current)
    while _tmp_stack:
        _current: do.DigitalObjectTree = _tmp_stack.pop()
        if _current.children:
            _tmp_stack += _current.children
            _total_stack += _current.children
    # now pick words
    _words = [_p for _p in _total_stack if _p.level == do.DigitalObjectLevel.WORD]

    # check for each word piece
    for _word in _words:
        if _word not in frame:
            _filtered += 1
            _uplete(_word)
    return _filtered


def _uplete(curr: do.DigitalObjectTree):
    if len(curr.children) == 0 and curr.level < do.DigitalObjectLevel.PAGE:
        _pa: do.DigitalObjectTree = curr.parent
        _pa.remove_children(curr)
        _uplete(_pa)


def _get_line_digos_from_digo(digo: do.DigitalObjectTree, lines: typing.List = None) -> typing.List[do.DigitalObjectTree]:
    if lines is None:
        lines = []
    if digo.level == do.DigitalObjectLevel.LINE and digo.transcription:
        lines.append(digo)
        return lines
    for child in digo.children:
        _get_line_digos_from_digo(child, lines)
    return lines


_HYPHENS: typing.List[str] = [
    "⸗",
    "-",
    "—",
]


def _sanitize_wraps(lines: typing.List[str]) -> typing.List[str]:
    """Sanitize word wraps if
    * last word token ends with '-', "⸗" or "—"
    * another line following
    * following line not empty
    """

    normalized_lines: typing.List[str] = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            for hyphen in _HYPHENS:
                if line.endswith(hyphen):
                    next_line = lines[i + 1]
                    if len(next_line.strip()) == 0:
                        # encountered empty next line, no merge possible
                        continue
                    next_line_tokens = next_line.split()
                    nextline_first_token = next_line_tokens.pop(0)
                    # join the rest of valid next line
                    lines[i + 1] = ' '.join(next_line_tokens)
                    line = line[:-1] + nextline_first_token
                    break
        normalized_lines.append(line)
    return normalized_lines


def _sanitize_chars(lines: typing.List[str]) -> typing.List[str]:
    """Replace or remove nonrelevant chars for current german word error rate"""

    sanitized: typing.List[str] = []
    for line in lines:
        text = line.strip()
        bad_chars = '0123456789“„"\'?!*.;:-=[]()|'
        text = ''.join([c for c in text if c not in bad_chars])
        if '..' in text:
            text = text.replace('..', '')
        if '  ' in text:
            text = text.replace('  ', ' ')
        text = ' '.join([t for t in text.split() if len(t) > 1])
        sanitized.append(text)

    return sanitized


class DigitalEvalMetricException(Exception):
    """Mark Exception during validation/calculating metrics"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SimilarityMetric:
    """Basic definition of OCR Similarity Metric,
    expressed in percent (0 - 100)
    """

    def __init__(
            self,
            precision=2,
            normalization=UC_NORMALIZATION_DEFAULT,
            preprocessings=None,
            to_text_func=digital_object_to_text,
    ) -> None:
        self.to_text_func: typing.Optional[typing.Callable] = to_text_func
        self.precision = precision
        self._value = None
        self._label = None
        self.unicode_normalization = normalization
        self.preprocessings = []
        if isinstance(preprocessings, list):
            self.preprocessings = preprocessings
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
        self._data_reference = normalize_unicode(value, self.unicode_normalization)
        self._value = None

    @property
    def candidate(self):
        """Candidate data"""
        return self._data_candidate

    @candidate.setter
    def candidate(self, value):
        self.input_candidate = value
        self._data_candidate = normalize_unicode(value, self.unicode_normalization)
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
            self._value *= 100
        return round(self._value, self.precision)

    def _forward(self):
        self._value = levenshtein_norm(self._data_reference, self._data_candidate)


class MetricChars(SimilarityMetric):
    """Calculate plain sequent character based metric"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None):
        super().__init__(
            precision,
            normalization,
            preprocessings)
        self._label = 'Cs'
        self.name = 'Characters'
        self.preprocessings = [_filter_whitespaces_excluding_blank_chars]


class MetricLetters(SimilarityMetric):
    """Calculate metric for only a certain sub-set of
    character sequence"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None):
        super().__init__(
            precision,
            normalization,
            preprocessings)
        self._label = 'Ls'
        self.preprocessings = [
            _filter_whitespaces,
            _filter_puncts,
            _filter_digits]


class MetricWords(SimilarityMetric):
    """Calculate metric for a sequence of word tokens"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None):
        super().__init__(
            precision,
            normalization,
            preprocessings)
        self._label = 'Ws'
        self.preprocessings = [_tokenize]


class MetricDictionary(SimilarityMetric):
    """Calculate metric for a multiset of word tokens"""

    LANGUAGE: str = LANGUAGE_KEY_DEFAULT

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None, to_text_func=digital_object_to_dict_text):
        super().__init__(
            precision=precision,
            normalization=normalization,
            preprocessings=preprocessings,
            to_text_func=to_text_func,
        )


class MetricDictionaryLangTool(MetricDictionary):
    """Calculate metric for a multiset of word tokens"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_NFKD,
                 preprocessings=None):
        if not isinstance(preprocessings, list):
            preprocessings = [_normalize_vocal_ligatures]
        super().__init__(
            precision=precision,
            normalization=normalization,
            preprocessings=preprocessings,
        )
        self._label = 'DictLT'
        self.diff = 0

    def _forward(self):
        text: str = self._data_candidate
        text_list: typing.List[str] = self._data_candidate.split()
        self._data_reference = text_list
        num_words: int = len(text_list)
        lt_response_data: typing.Dict = LanguageTool.check(text, MetricDictionary.LANGUAGE)
        total_matches = lt_response_data['matches'] if 'matches' in lt_response_data else 0
        typo_errors = len(total_matches)
        self.diff = typo_errors if typo_errors <= num_words else num_words


class MetricBoW(SimilarityMetric):
    """Calculate metric for a multiset of word tokens"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None):
        super().__init__(
            precision,
            normalization,
            preprocessings)
        self._label = 'BoWs'
        self.preprocessings = [_tokenize]

    def _forward(self):
        self._value = bag_of_tokens(self._data_reference, self._data_candidate)


class MetricIR(SimilarityMetric):
    """Calculate information retrival metrics"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            preprocessings)
        self.languages = languages
        self.preprocessings = [_tokenize_to_sorted_set,
                               _strip_stopwords_for(self.languages)
                               ]
        # no aligning required, we rely on nltk
        self.calc_func = None

    def _forward(self):
        """to remind that this class needs further refinement"""
        raise NotImplementedError


class MetricIRPre(MetricIR):
    """Calculate precision"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            preprocessings,
            languages)
        self._label = 'Pre'

    def _forward(self):
        self._value = ir_precision(self._data_reference, self._data_candidate)


class MetricIRRec(MetricIR):
    "Calculate recall"

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            preprocessings,
            languages)
        self._label = 'Rec'

    def _forward(self):
        self._value = ir_recall(self._data_reference, self._data_candidate)


class MetricIRFM(MetricIR):
    """Calculate harmonic mean for precision/recall"""

    def __init__(self, precision=2, normalization=UC_NORMALIZATION_DEFAULT,
                 preprocessings=None, languages=None):
        super().__init__(
            precision,
            normalization,
            preprocessings,
            languages)
        self._label = 'FM'

    def _forward(self):
        self._value = ir_fmeasure(self._data_reference, self._data_candidate)


def levenshtein_norm(reference_data, candidate_data, inverse=False) -> int:
    """Calculate levenshtein metric as ration of sum of edit operations 
    normalized to sum of edit and equal operations.

    Works with characters and word-like tokens, where
    tokens correspond also to:
    * abbreviations  (like "Nr." or "Etg.")
    * numbers/years  (like "1899")
    * split-up words (line endings/beginnings)
    """
    if inverse:
        return rfls.normalized_distance(reference_data, candidate_data)
    return rfls.normalized_similarity(reference_data, candidate_data)


def bag_of_tokens(reference_tokens: typing.List[str],
                  candidate_tokens: typing.List[str]) -> int:
    """Calculate difference between reference and candidate token list
    """
    false_negatives: typing.List[str] = _diff(reference_tokens, candidate_tokens)
    false_positives: typing.List[str] = _diff(candidate_tokens, reference_tokens)
    delta = len(false_negatives) + len(false_positives)
    total = len(reference_tokens) + len(candidate_tokens)
    subtrahend = (delta / total) if total > 0 else 0
    ratio = 1 - subtrahend
    return ratio


def _diff(gt_tokens, cd_tokens) -> typing.List[str]:
    return list((collections.Counter(gt_tokens) - collections.Counter(cd_tokens)).elements())


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


# diacritica to take care of
_COMBINING_SMALL_E = '\u0364'

def _normalize_vocal_ligatures(a_string) -> str:
    """Replace vocal ligatures, which otherwise
    may confuse the index component workflow,
    especially COMBINING SMALL LETTER E : \u0364

    a^e, o^e, u^e => (u0364) => ä, ö, ü
    """

    _out = []
    for i, _c in enumerate(a_string):
        if _c == _COMBINING_SMALL_E:
            _preceeding_vocal = _out[i - 1]
            _vocal_name = unicodedata.name(_preceeding_vocal)
            _replacement = ''
            if 'LETTER A' in _vocal_name:
                _replacement = 'ä'
            elif 'LETTER O' in _vocal_name:
                _replacement = 'ö'
            elif 'LETTER U' in _vocal_name:
                _replacement = 'ü'
            else:
                _msg = f"No conversion for {_preceeding_vocal} ('{a_string}')!"
                raise DigitalEvalMetricException(f"normalize vocal ligatures: {_msg}")
            _out[i - 1] = _replacement
        _out.append(_c)

    # strip all combining e's anyway
    return ''.join(_out).replace(_COMBINING_SMALL_E, '')
