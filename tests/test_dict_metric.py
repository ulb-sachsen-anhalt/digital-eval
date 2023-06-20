from digital_eval.evaluation import (
    piece_to_dict_text,
    piece_to_text
)
from digital_eval.metrics import normalize_unicode, UC_NORMALIZATION_NFKD, normalize_vocal_ligatures
from .conftest import TEST_RES_DIR


def test_piece_to_dict_text_alto():
    alto_path = f'{TEST_RES_DIR}/dict_metric/alto.xml'

    # act
    alto_text_no_sanit, _ = piece_to_text(alto_path, oneliner=True)
    alto_words_no_sanit = alto_text_no_sanit.split()
    alto_text, _ = piece_to_dict_text(alto_path, oneliner=True)
    alto_lines, alto_num_lines = piece_to_dict_text(alto_path, oneliner=False)
    alto_lines_norm_vocal_ligatures = [normalize_vocal_ligatures(line) for line in alto_lines]
    alto_lines_norm = [normalize_unicode(line, UC_NORMALIZATION_NFKD) for line in alto_lines_norm_vocal_ligatures]
    alto_text_norm = " ".join(alto_lines_norm)
    alto_words = alto_text_norm.split()

    # assert
    assert alto_text_no_sanit == "Dieſe uͤberfruͤhte An⸗ kunft des hailigen Raimarſ. ſachſen- ſtolz, aͤhnlich"
    assert 10 == len(alto_words_no_sanit)
    assert alto_text == "Dieſe uͤberfruͤhte Ankunft des hailigen Raimarſ ſachſenſtolz, aͤhnlich"
    assert alto_text_norm == "Diese überfrühte Ankunft des hailigen Raimars sachsenstolz, ähnlich"
    assert 8 == len(alto_words)


def test_piece_to_dict_text_page2019():
    page_path = f'{TEST_RES_DIR}/dict_metric/page2019.xml'

    # act
    page_text_no_sanit, _ = piece_to_text(page_path, oneliner=True)
    page_words_no_sanit = page_text_no_sanit.split()
    page_text, _ = piece_to_dict_text(page_path, oneliner=True)
    page_lines, alto_num_lines = piece_to_dict_text(page_path, oneliner=False)
    page_lines_norm_vocal_ligatures = [normalize_vocal_ligatures(line) for line in page_lines]
    page_lines_norm = [normalize_unicode(line, UC_NORMALIZATION_NFKD) for line in page_lines_norm_vocal_ligatures]
    page_text_norm = " ".join(page_lines_norm)
    page_words = page_text_norm.split()

    # assert
    assert page_text_no_sanit == "Dieſe uͤberfruͤhte An⸗ kunft des hailigen Raimarſ. ſachſen- ſtolz, aͤhnlich"
    assert 10 == len(page_words_no_sanit)
    assert page_text == "Dieſe uͤberfruͤhte Ankunft des hailigen Raimarſ ſachſenſtolz, aͤhnlich"
    assert page_text_norm == "Diese überfrühte Ankunft des hailigen Raimars sachsenstolz, ähnlich"
    assert 8 == len(page_words)


def test_piece_to_dict_text_page2013():
    page_path = f'{TEST_RES_DIR}/dict_metric/page2013.xml'

    # act
    page_text_no_sanit, _ = piece_to_text(page_path, oneliner=True)
    page_words_no_sanit = page_text_no_sanit.split()
    page_text, _ = piece_to_dict_text(page_path, oneliner=True)
    page_lines, alto_num_lines = piece_to_dict_text(page_path, oneliner=False)
    page_lines_norm_vocal_ligatures = [normalize_vocal_ligatures(line) for line in page_lines]
    page_lines_norm = [normalize_unicode(line, UC_NORMALIZATION_NFKD) for line in page_lines_norm_vocal_ligatures]
    page_text_norm = " ".join(page_lines_norm)
    page_words = page_text_norm.split()

    # assert
    assert page_text_no_sanit == "Dieſe uͤberfruͤhte An⸗ kunft des hailigen Raimarſ. ſachſen- ſtolz, aͤhnlich"
    assert 10 == len(page_words_no_sanit)
    assert page_text == "Dieſe uͤberfruͤhte Ankunft des hailigen Raimarſ ſachſenſtolz, aͤhnlich"
    assert page_text_norm == "Diese überfrühte Ankunft des hailigen Raimars sachsenstolz, ähnlich"
    assert 8 == len(page_words)
