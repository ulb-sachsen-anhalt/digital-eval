"""Preprocessings for metric input"""

import unicodedata
import string
import typing
import xml.parsers

from pathlib import Path

import nltk
import nltk.corpus as nltk_corp

import digital_eval.model as dimo

# Python3 standard Unicode Normalization
#
UC_NORMALIZATION_DEFAULT = 'NFC'
UC_NORMALIZATION_NFKD = 'NFKD'

# whitespaces
#
# usual spatium and special control sequences
WHITESPACES = string.whitespace

WHITESPACES_EXCL_SPACE_CHARS = WHITESPACES[1:]

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

WHITESPACE_TRNSL = str.maketrans('', '', WHITESPACES)
PUNCT_TRNSL = str.maketrans('', '', PUNCTUATIONS)
DIGIT_TRNSL = str.maketrans('', '', DIGITS)

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



class PreprocessingException(Exception):
    """Mark problems concerning preprocessing"""


class Preprocessor:
    """Wrapper for different preprocessing
    implementations concerning textual
    OCR metric input data"""

    def __init__(self, input_data):
        self._input = input_data

    def run(self):
        """Wrap required preprocessings"""

    @property
    def result(self):
        """get processed data"""
        return self._input


class TextPreprocessor(Preprocessor):
    """Common flavor to handle plain text strings"""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.code_norm = UC_NORMALIZATION_DEFAULT
        self.frame = None
        self.one_liner = True

    def normalize_encoding(self):
        """Convert code points to 
        required unicode form"""

        self._input = unicodedata.normalize(self.code_norm, self._input)

    def run(self):
        if isinstance(self._input, Path):
            self._input, _ = file_to_text(self._input,
                                         self.frame,
                                         self.one_liner)
        self.normalize_encoding()


class LetterPreprocessor(TextPreprocessor):
    """Strip any non-letter characters
    from input data"""

    def strip_chars(self):
        """remove non-letter characters"""

        self._input = self._input.translate(WHITESPACE_TRNSL)
        self._input = self._input.translate(PUNCT_TRNSL)
        self._input = self._input.translate(DIGIT_TRNSL)

    def run(self):
        super().run()
        self.strip_chars()


class SimpleTokenizer(TextPreprocessor):
    """Tokenize input string"""

    def tokenize(self):
        """make string list"""
        self._input = self._input.split() if isinstance(self._input, str) else self._input

    def run(self):
        super().run()
        self.tokenize()


class LanguageAwareTokenizer(SimpleTokenizer):
    """Tokenize to set"""

    def __init__(self, input_data, languages=None):
        super().__init__(input_data)
        if languages is None:
            languages = STOPWORDS_DEFAULT
        self.languages = languages

    def tokenize_to_sorted_set(self):
        """enhanced tokenizing"""
        self.tokenize()
        self._input = set(sorted(self._input))

    def strip_stopwords(self):
        """remove some tokens"""
        self._input = self._input - LanguageAwareTokenizer._get_stopwords(languages=self.languages)

    def run(self):
        if isinstance(self._input, Path):
            self._input, _ = file_to_text(self._input,
                                         self.frame,
                                         self.one_liner)
        self.normalize_encoding()
        self.tokenize_to_sorted_set()
        self.strip_stopwords()

    @classmethod
    def _get_stopwords(cls, nltk_mappings=None, languages=None) -> typing.Set[str]:
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


class DictionaryTextPreprocessor(TextPreprocessor):
    """Prepare input data to be comparable
    with dictionary entries or spellcheckers
    """
    # diacritica to take care of
    _COMBINING_SMALL_E = u'\u0364'

    def __init__(self, input_data):
        if isinstance(input_data, Path):
            input_data, _ = file_to_dict_text(input_data, oneliner=True)
        super().__init__(input_data)

    def normalize_vocal_ligatures(self) -> str:
        """Replace vocal ligatures, which otherwise
        may confuse the index component workflow,
        especially COMBINING SMALL LETTER E : \u0364

        a^e, o^e, u^e => (u0364) => ä, ö, ü
        """

        _out = []
        for i, _c in enumerate(self._input):
            if _c == DictionaryTextPreprocessor._COMBINING_SMALL_E:
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
                    _msg = f"No conversion for {_preceeding_vocal} ('{self._input}')!"
                    raise PreprocessingException(f"normalize vocal ligatures: {_msg}")
                _out[i - 1] = _replacement
            _out.append(_c)

        # strip all combining e's anyway
        self._input = ''.join(_out).replace(DictionaryTextPreprocessor._COMBINING_SMALL_E, '')

    def run(self):
        self.normalize_encoding()
        self.normalize_vocal_ligatures()


def file_to_dict_text(file_path: str, frame=None, oneliner=False) -> typing.Tuple:
    """Convert file data into sanitized text
    usable for spellcheckings or dictionairies"""

    line_texts: typing.List[str]
    len_lines: int
    line_texts, len_lines = file_to_text(file_path=file_path, frame=frame, oneliner=False)
    non_empty_lines: typing.List[str] = [line_text for line_text in line_texts if len(line_text) > 0]
    lines_sanitized_wraps: typing.List[str] = _sanitize_wraps(non_empty_lines)
    lines_sanitized_chars: typing.List[str] = _sanitize_chars(lines_sanitized_wraps)
    text = ' '.join(lines_sanitized_chars) if oneliner else lines_sanitized_chars
    return text, len_lines


def file_to_text(file_path, frame=None, oneliner=True) -> typing.Tuple:
    """Convert file data into plain text"""

    try:
        top_digo: dimo.DigitalObjectTree = dimo.to_digital_object(str(file_path))
        # explicit filter frame?
        if not frame:
            frame = top_digo.dimensions
        elif len(frame) == 2:
            frame = [[frame[0][0], frame[0][1]],
                     [frame[1][0], frame[0][1]],
                     [frame[1][0], frame[1][1]],
                     [frame[0][0], frame[1][1]]]
        frame_digo = dimo.DigitalObjectTree()
        frame_digo.dimensions = frame
        filter_word_pieces(frame_digo, top_digo)
        the_lines = _get_digos_from_digo(top_digo)
        if oneliner:
            return top_digo.transcription, len(the_lines)
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
    """respect frame for current digital object
    return number of filtered elements
    """
    _filtered = 0
    _tmp_stack = []
    _total_stack = []
    # stack all items
    _total_stack.append(current)
    _tmp_stack.append(current)
    while _tmp_stack:
        _current: dimo.DigitalObjectTree = _tmp_stack.pop()
        if _current.children:
            _tmp_stack += _current.children
            _total_stack += _current.children
    # now pick words
    _words = [_p for _p in _total_stack if _p.level == dimo.DigitalObjectLevel.WORD]

    # check for each word piece
    for _word in _words:
        if _word not in frame:
            _filtered += 1
            _uplete(_word)
    return _filtered


def _uplete(curr: dimo.DigitalObjectTree):
    if len(curr.children) == 0 and curr.level < dimo.DigitalObjectLevel.PAGE:
        _pa: dimo.DigitalObjectTree = curr.parent
        _pa.remove_children(curr)
        _uplete(_pa)


def _get_digos_from_digo(digo: dimo.DigitalObjectTree, lines=None) -> typing.List:
    if lines is None:
        lines = []
    if digo.level == dimo.DigitalObjectLevel.LINE and digo.transcription:
        lines.append(digo)
        return lines
    for child in digo.children:
        _get_digos_from_digo(child, lines)
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
