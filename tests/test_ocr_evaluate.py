# -*- coding: utf-8 -*-
"""OCR Evaluation Test Module"""

import os
import shutil

from xml.dom.minidom import (
    parseString,
)

from xml.etree.ElementTree import (
    ParseError
)

import pytest

from pytest import (
    approx
)

from digital_eval.evaluation import (
    EvalEntry,
    Evaluator,
    OCRData,
    match_candidates,
    ocr_to_text,
    piece_to_text,
    filter_word_pieces,
    get_bbox_data,
)
from digital_eval.metrics import MetricIRFM, MetricIRPre, MetricIRRec, MetricChars

from digital_eval.model import (
    PieceLevel,
)

from digital_eval.model_legacy import (
    BoundingBox,
    OCRWord,
    OCRWordLine,
)

from .conftest import (
    TEST_RES_DIR
)


def test_match_candidates_alto_candidate_with_coords():
    actual_matches = match_candidates(f'{TEST_RES_DIR}/candidate/frk_alto',
                                      f'{TEST_RES_DIR}/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')
    assert f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0001_part.xml' == actual_matches[0]


def test_match_candidates_both_txt_files():
    path_candidates = f'{TEST_RES_DIR}/candidate/txt'
    path_gt = f'{TEST_RES_DIR}/groundtruth/txt/1246734.gt.txt'
    actual_matches = match_candidates(path_candidates, path_gt)
    assert f'{TEST_RES_DIR}/candidate/txt/OCR-Fraktur_1246734.txt' == actual_matches[0]


def test_match_candidates_fails_no_groundtruth():
    with pytest.raises(IOError) as exc:
        match_candidates(
            f'{TEST_RES_DIR}/candidate/txt',
            './test/sresources/txt/no_gt.txt')
    assert "invalid groundtruth data path" in str(exc)


def test_match_candidates_fails_no_candidates():
    with pytest.raises(IOError) as exc:
        match_candidates(
            './text/no_results',
            f'{TEST_RES_DIR}/txt/gt/1246734.txt')
    assert "invalid ocr result path" in str(exc)


def test_match_candidates_groundtruth_txt_candidate_alto():
    path_cd = f'{TEST_RES_DIR}/candidate/ara_alto'
    path_gt = f'{TEST_RES_DIR}/groundtruth/txt/217745.gt.txt'

    # act
    actual_matches = match_candidates(path_cd, path_gt)

    # assert
    assert actual_matches[0] == f'{TEST_RES_DIR}/candidate/ara_alto/217745.xml'


def test_ocr_to_text_alto_candidate_with_coords():
    """Check lines from regular ALTO candidate"""

    alto_path = f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0512_01.xml'
    p1 = (300, 375)
    p2 = (6200, 3425)

    # act
    result = ocr_to_text(alto_path, coords=(p1, p2))

    # assert
    assert result is not None
    assert 'n.a.' == result[0]
    lines = result[1]
    # subject to switch dependend on
    # handling of rather empty lines
    assert 166 == len(lines) or 169 == len(lines)


def test_piece_to_text_alto_candidate_with_coords():
    """Check lines from ALTO candidate"""

    alto_path = f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0512_01.xml'
    p1 = (300, 375)
    p2 = (6200, 3425)

    # act
    _gt_type, _as_lines, _ = piece_to_text(alto_path, frame=(p1, p2), oneliner=False)

    # assert
    assert _gt_type == 'n.a.'
    # subject to switch dependend on
    # handling of rather empty lines
    assert 166 == len(_as_lines) or 169 == len(_as_lines)


def test_piece_to_oneliner_page_groundtruth():
    """Check output for page newspaper groundtruth"""

    _path = f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0002.art.gt.xml'
    p1 = (2839, 2468)
    p2 = (7309, 6876)

    # act
    _gt_01, _ocr_01, _n_lines01 = ocr_to_text(_path, coords=(p1, p2), oneliner=True)
    _gt_02, _ocr_02, _n_lines02 = piece_to_text(_path, frame=(p1, p2), oneliner=True)

    # assert
    assert _gt_01 == 'art'
    assert 97 == _n_lines01
    assert 5385 == len(_ocr_01)
    assert _gt_02 == 'art'
    assert 97 == _n_lines02
    # they shall match
    assert _ocr_01 == _ocr_02
    assert 5385 == len(_ocr_02)


def test_ocr_to_text_text_data_without_coords():
    """Also data sets without coordinates can be
    processed and transformed into text/textlines"""

    text_path = f'{TEST_RES_DIR}/groundtruth/txt/1246734.gt.txt'

    # act
    result = ocr_to_text(text_path, coords=None, oneliner=True)

    # assert
    assert result is not None
    assert 'n.a.' == result[0]
    one_liner = result[1]
    assert 650 == len(one_liner)


def test_ocr_to_text_groundtruth_odem_ocrd_page_2019():
    text_path = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml'

    # act
    result = ocr_to_text(text_path, oneliner=True)

    # assert
    assert result is not None
    assert 'n.a.' == result[0]
    text_as_one_liner = result[1]
    # we have a groundtruth text spanning 458 chars in the rectangular area
    assert 829 == len(text_as_one_liner)


def test_ocr_to_text_candidate_odem_ocrd_page_2019():
    text_path = f'{TEST_RES_DIR}/candidate/frk_page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.xml'

    # act
    result = ocr_to_text(text_path, coords=((216, 240), (1050, 1640)), oneliner=True)

    # assert
    assert result is not None
    assert 'n.a.' == result[0]
    text_as_one_liner = result[1]
    # we have a candidate text spanning 824 chars in the rectangular area
    assert 824 == len(text_as_one_liner)


def test_evaluate_single_alto_candidate_with_page_groundtruth(tmp_path):
    """Illustrate, evaluation of single candidate
    with proper organization of groundtruth data"""

    # arrange
    eval_domain = tmp_path / 'candidate' / '1667522809_J_0001'
    eval_domain.mkdir(parents=True)
    gt_domain = tmp_path / 'groundtruth' / '1667522809_J_0001'
    gt_domain.mkdir(parents=True)
    evaluator = Evaluator(eval_domain)
    evaluator.metrics = [MetricChars()]
    # required for directory-like aggregation
    evaluator.domain_reference = gt_domain
    _candidate_src = os.path.join(f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0001_0002.xml')
    shutil.copy(_candidate_src, eval_domain)
    _gt_src = os.path.join(f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0002.art.gt.xml')
    _gt_dst = str(gt_domain / '1667522809_J_0001_0002.art.gt.xml')
    shutil.copy(_gt_src, _gt_dst)

    # act
    eval_entry = EvalEntry(str(eval_domain / '1667522809_J_0001_0002.xml'))
    eval_entry.path_g = _gt_dst
    evaluator.eval_all([eval_entry], sequential=True)
    evaluator.aggregate(by_type=True)
    evaluator.eval_map()
    result = evaluator.get_results()[0]
    defaults = result.get_defaults()

    # assert
    assert 5 == len(defaults)
    # metric label
    assert 'Cs@1667522809_J_0001' == defaults[0]
    assert 1 == defaults[1]  # number of data points
    # metric raw
    assert 37.07 == pytest.approx(defaults[2], rel=1e-3)
    # metric with stripped outliers (no outlier, of course!)
    assert 0 == result.n_outlier
    assert not result.cleared_result
    # reference size chars
    assert 4607 == defaults[4]


def test_evaluate_page_groundtruth_with_itself(tmp_path):
    """Use Groundtruth as candidate and evaluate it
    with itself as reference data shall yield
    accuracy of nearly 100 percent"""

    # arrange
    eval_domain = tmp_path / 'candidate' / '1667522809_J_0001'
    eval_domain.mkdir(parents=True)
    gt_domain = tmp_path / 'groundtruth' / '1667522809_J_0001'
    gt_domain.mkdir(parents=True)
    evaluator = Evaluator(eval_domain)
    evaluator.metrics = [MetricChars()]
    evaluator.domain_reference = gt_domain
    _candidate_src = os.path.join(f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0002.art.gt.xml')
    _candidate_dst = str(eval_domain / '1667522809_J_0001_0002.xml')
    shutil.copy(_candidate_src, _candidate_dst)
    _gt_src = os.path.join(f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0002.art.gt.xml')
    _gt_dst = str(gt_domain / '1667522809_J_0001_0002.art.gt.xml')
    shutil.copy(_gt_src, _gt_dst)

    # act
    eval_entry = EvalEntry(str(eval_domain / '1667522809_J_0001_0002.xml'))
    eval_entry.path_g = _gt_dst
    evaluator.eval_all([eval_entry], sequential=True)
    evaluator.aggregate(by_type=True)
    evaluator.eval_map()
    result = evaluator.get_results()[0]
    defaults = result.get_defaults()

    # assert
    assert 5 == len(defaults)
    # metric label
    assert 'Cs@1667522809_J_0001' == defaults[0]
    assert 1 == defaults[1]  # number of data points
    assert 100.00 == pytest.approx(defaults[2], rel=1e-3)
    # reference size chars
    assert 4607 == defaults[4]


def test_evaluate_set_with_5_entries(tmp_path):
    """Simulate evaluation with 5 data sets
    * sub_dir 'eng' having 1 pair
    * sub_dir 'ger' having 4 pairs

    Hereby it is required to have *real* paths to make
    this work properly.

    please note:
        in tests it turned out it needs more than 5 data points / aggregation stage
        to measure the impact of the outlier (here: 1/6 with CA 86,447) 

    also note the somehow hacky way of setting the numbers of the reference data
    by injecting wacky test strings - only size matters
    """

    # arrange
    path_dir_gt = tmp_path / 'odem'
    path_dir_gt.mkdir()
    path_dir_c = tmp_path / 'media' / 'jpg' / 'odem'
    path_dir_c.mkdir(parents=True)
    evaluator = Evaluator(path_dir_c)
    evaluator.domain_reference = path_dir_gt
    _metric_ca1 = MetricChars()
    _metric_ca1._value = 95.70
    _metric_ca1._data_reference = 't' * 810
    _metric_ca2 = MetricChars()
    _metric_ca2._value = 96.53
    _metric_ca2._data_reference = 't' * 675
    _metric_ca3 = MetricChars()
    _metric_ca3._value = 94.91
    _metric_ca3._data_reference = 't' * 1395
    _metric_ca4 = MetricChars()
    _metric_ca4._value = 94.40
    _metric_ca4._data_reference = 't' * 1466
    # outlier !
    _metric_ca5 = MetricChars()
    _metric_ca5._value = 86.44
    _metric_ca5._data_reference = 't' * 1520
    _metric_ca6 = MetricChars()
    _metric_ca6._value = 93.44
    _metric_ca6._data_reference = 't' * 1520

    entry1 = EvalEntry(path_dir_c / 'eng' / 'urn+nbn+de+gbv+3+1-135654-p0403-5_eng.xml')
    entry1.path_g = str(path_dir_gt / 'eng' / 'urn+nbn+de+gbv+3+1-135654-p0403-5_eng.gt.xml')
    entry1.metrics = [_metric_ca1]
    entry2 = EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-816198-p0493-2_ger.xml')
    entry2.path_g = str('/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-816198-p0493-2_ger.gt.xml')
    entry2.metrics = [_metric_ca2]
    entry3 = EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-818383-p0034-5_ger.xml')
    entry3.path_g = '/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-818383-p0034-5_ger.gt.xml'
    entry3.metrics = [_metric_ca3]
    entry4 = EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-822479-p1119-4_ger.xml')
    entry4.path_g = '/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-822479-p1119-4_ger.gt.xml'
    entry4.metrics = [_metric_ca4]
    entry5 = EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-828020-p0173-6_ger.xml')
    entry5.path_g = '/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-828020-p0173-6_ger.gt.xml'
    entry5.metrics = [_metric_ca5]
    entry6 = EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-125584-p0314-6_ger.xml')
    entry6.path_g = '/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-125584-p0314-6_ger.gt.xml'
    entry6.metrics = [_metric_ca6]
    evaluator.evaluation_entries = [entry1, entry2, entry3, entry4, entry5, entry6]

    # act
    evaluator.aggregate(by_metrics=[0])
    evaluator.eval_map()
    results = evaluator.get_results()

    # assert
    # ensure standard deviation decreases by dropping outlier
    assert 3 == len(results)
    assert results[0].std == approx(3.33, abs=1e-2)
    assert results[0].cleared_result.std < results[0].std
    assert results[0].cleared_result.std == approx(1.06, abs=1e-2)


@pytest.mark.parametrize("b1,b2,expected", [
    (BoundingBox((100, 100), (200, 200)), BoundingBox((0, 0), (6000, 8000)), 10000),
    (BoundingBox((100, 100), (200, 200)), BoundingBox((0, 500), (6000, 8000)), 0),
    (BoundingBox((100, 100), (200, 200)), BoundingBox((0, 199), (6000, 8000)), 100)])
def test_they_intersect(b1, b2, expected):
    """Test different intersection results:
    * regular intersection
    * no intersection
    * small intersection

    Args:
        b1 (BoundingBox):
        b2 (BoundingBox):
        expected (int)  :
    """
    assert expected == b1.intersection(b2)


def test_get_groundtruth_type_legacy():
    """Ensure GT-Annotations from ZD1 are still to be found"""

    file_path = f'{TEST_RES_DIR}/groundtruth/page/1681877805_J_0075_0001.art.gt.xml'
    actual = OCRData(file_path)
    assert 'art' == actual.get_type_groundtruth()


def test_get_groundtruth_type_alternative():
    """Ensure new flavour of GT-Type annotations are processed properly"""

    file_path = f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0938.ann.gt.xml'
    actual = OCRData(file_path)
    assert 'ann' == actual.get_type_groundtruth()


def test_read_alto():
    ocr_data = OCRData(f"{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0001_part.xml")
    gt_lines = ocr_data.get_lines_text()
    assert 5 == len(gt_lines)
    assert 'großen Wahlrehisdemonſtratio jede, daß ſie keine Freunde' == gt_lines[0]
    assert 'un  Digzipli -zu wahren:pexſtehen allerdings richtig, laſſen.' == gt_lines[4]


def test_read_alto_no_groundtype():
    ocr_data = OCRData(f"{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0001_part.xml")
    assert 'n.a.' == ocr_data.get_type_groundtruth()


def test_read_alto_groundtype_exists():
    ocr_data = OCRData(f"{TEST_RES_DIR}/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml")
    assert 'article' == ocr_data.get_type_groundtruth()


def test_alto_page_dimensions():
    ocr_data = OCRData(f"{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0001_part.xml")
    page_dim = ocr_data.get_page_dimensions()
    assert (6712, 9944) == page_dim


@pytest.mark.parametrize(["path_data", "coords_start", "coords_end", "n_lines"],
                         [
                             (f"{TEST_RES_DIR}/groundtruth/page/page01.gt.xml", (667, 595), (2317, 2900), 29),
                             (f"{TEST_RES_DIR}/candidate/page_lines/page01.xml", (667, 595), (2317, 2900), 29),
                             (f"{TEST_RES_DIR}/groundtruth/page/1681877805_J_0075_0001.art.gt.xml", None, None, 101)
                         ])
def test_get_line_data(path_data, coords_start, coords_end, n_lines):
    """Ensure different formats will be properly processd,
    i.e. with frame information lines and words are filtered,
    and with missing frame data all lines are being read
    """
    # arrange
    ocr_data = OCRData(path_data)
    txt_data = []

    # act
    if not coords_start:
        txt_data = ocr_data.get_lines_text()
    else:
        txt_data = ocr_data.filter_all(coords_start, coords_end)

    assert n_lines == len(txt_data)
    for _line in txt_data:
        if isinstance(_line, str):
            assert len(_line) > 0
        else:
            assert len(_line.get_text()) > 0


def test_page_data():
    ocr_data = OCRData(f"{TEST_RES_DIR}/candidate/frk_page/1667522809_J_0001_0512.xml")
    gt_lines = ocr_data.get_lines_text()
    assert [] != gt_lines
    # 519 lines when only respect lines 
    # which contain at least 2 ("two")
    # alphabetical characters, 532 otherwise
    # 532 lines when only respect lines
    # which are *not* rather empty, 536 otherwise
    assert 536 == len(gt_lines)
    # specific textual content check
    # line 12 if dropped empty lines, 14 otherwise
    assert 'Seite 4 Sonnabend' == gt_lines[13]


def test_read_page_2013_data():
    ocr_data = OCRData(f"{TEST_RES_DIR}/groundtruth/page/1681877805_J_0075_0001.art.gt.xml")
    gt_lines = ocr_data.get_lines_text()
    assert gt_lines
    assert [] != gt_lines
    assert 101 == len(gt_lines)
    assert 'vie kleine Excellenz von Meppen nicht mehr unter den Lebenden' == gt_lines[2]


def test_page_page_dimensions():
    ocr_data = OCRData(f"{TEST_RES_DIR}/candidate/frk_page/1667522809_J_0001_0512.xml")
    assert (7295, 10584) == ocr_data.get_page_dimensions()


def test_raise_exception_for_empty_words():
    """Ensure inconsistent data is being hunted down:
    raise RuntimeError when encountered PAGE XML
    with empty Word boxes, i.e. TextLine with Words
    but Words has empty TextEquiv/Unicode elements
    
    origin: 1667522809_J_0001_0768.gt.art1
    (produced by Transcribus Workflows)
    """

    file_path = f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0768.art.gt.xml'

    # act
    with pytest.raises(RuntimeError) as err:
        OCRData(file_path)

    # assert
    assert 'word_1641981922406_2298 misses text: list index out of range' in err.value.args[0]


def test_line_repr():
    """Inject minidom-compatible test-data 
    to check str-serialization of OCR Token
    """
    line = OCRWordLine('test_line')
    assert not line.p2
    the_xml = """
    <alto xmlns="http://www.loc.gov/standards/alto/ns-v3#" 
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"  xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v3# http://www.loc.gov/alto/v3/alto-3-0.xsd">
        <String ID="string_0" HPOS="100" VPOS="100" WIDTH="500" HEIGHT="100" WC="0.0" CONTENT="test"/>
    </alto>
    """
    the_dom = parseString(the_xml)
    the_word_el = the_dom.getElementsByTagName('String')[0]
    ocr_word = OCRWord('w_01', the_word_el)
    line.add_word(ocr_word)
    assert line.__repr__().startswith('[test_line][500:100]')


def test_no_groundtruth_at_all(tmp_path):
    """
    Behavior if no groundtruth found:
    Raise RuntimeError and exit, since evaluation
    doesn't make any sense so far
    """

    evaluator = Evaluator(tmp_path)
    evaluator.eval_all([])

    with pytest.raises(RuntimeError) as err:
        evaluator.aggregate()

    assert 'missing evaluation data' in str(err.value)


def test_handle_exception_invalid_literal_for_int():
    """Handle evaluation exception: 
        invalid literal for int() with base 10: ''

        Please note, that this exception results from 
        inconsistencies in the groundtruth data, therefore 
        only use a dummy placeholder, no *real* OCR candidate,
        originating from missing Coords
    """

    # arrange
    path_gt = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-792101-p0667-5_ger.gt.xml'
    eval_entry = EvalEntry('dummy_candidate')
    eval_entry.path_g = path_gt

    # act
    evaluator = Evaluator('dummy_path')
    with pytest.raises(RuntimeError) as err:
        evaluator.eval_entry(eval_entry)

    # assert
    assert 'urn+nbn+de+gbv+3+1-792101-p0667-5_ger.gt.xml' in err.value.args[0]
    assert 'too few points' in err.value.args[0] or 'empty Coords' in err.value.args[0]


def test_handle_empty_candidate_information_retrival():
    """Handle evaluation exception: 
        unsupported format string passed to NoneType.__format__
        results from complete failing candidate text
        and nltk behaving inconsistent
    """

    # arrange
    path_gt = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-138193-p0904-0_ger.gt.xml'
    path_cd = f'{TEST_RES_DIR}/candidate/frk_page/urn+nbn+de+gbv+3+1-138193-p0904-0_ger.xml'
    eval_entry = EvalEntry(path_cd)
    eval_entry.path_g = path_gt
    evaluator = Evaluator('/data')
    evaluator.metrics = [MetricIRPre(), MetricIRRec(), MetricIRFM()]
    evaluator.verbosity = 1

    # act
    evaluator.eval_entry(eval_entry)

    # assert
    assert eval_entry.metrics[0].label == 'Pre'
    assert eval_entry.metrics[0].value == 0.0
    assert eval_entry.metrics[1].label == 'Rec'
    assert eval_entry.metrics[1].value == 0.0
    assert eval_entry.metrics[2].label == 'FM'
    assert eval_entry.metrics[2].value == 0.0


def test_handle_table_text_groundtruth():
    """Handle evaluation properly with very
    poor candidate data from ocr-ing a table
     
    legacy: "missing gt text from urn+nbn+de+gbv+3+1-126343-p0285-7_ger.gt.xml"
    """

    # arrange
    path_gt = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-126343-p0285-7_ger.gt.xml'
    path_cd = f'{TEST_RES_DIR}/candidate/frk_page/urn+nbn+de+gbv+3+1-126343-p0285-7_ger.xml'
    eval_entry = EvalEntry(path_cd)
    eval_entry.path_g = path_gt

    # act
    evaluator = Evaluator('/data')
    evaluator.metrics = [MetricChars()]
    evaluator._wrap_eval_entry(eval_entry)

    # assert / legacy: 5.825 , actual 4.0
    _result_cca = eval_entry.metrics[0].value
    assert _result_cca > 3.9 and _result_cca < 4.1


def test_get_box_from_empty_page():
    """How to deal with empty PAGE"""

    # arrange
    _path_gt = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-201080-p0034-8_ger.gt.xml'

    # act
    _p1, _p2 = get_bbox_data(_path_gt)

    # assert 
    assert _p1 == (77, 58)
    assert _p2 == (2012, 2506)


def test_handle_exception_invalid_alto_xml():
    """Handle invalid XML data

    ALTO data got corrupted by copying it around
    (the latest rows simply missing)
    """

    # arrange
    path_gt = f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0001_0256_corrupt.xml'
    eval_entry = EvalEntry('dummy_candidate')
    eval_entry.path_g = path_gt

    # act
    evaluator = Evaluator('dummy_path')
    with pytest.raises(ParseError) as err:
        evaluator.eval_entry(eval_entry)

    # assert
    assert 'no element found' in err.value.args[0]
