# -*- coding: utf-8 -*-
"""OCR Evaluation Test Module"""

import os
import shutil

from pathlib import Path
from xml.etree.ElementTree import (
    ParseError
)

import pytest
from pytest import (
    approx
)

import digital_eval.evaluation as digev
import digital_eval.metrics as digem
import digital_eval.preprocessing as dipre
import digital_eval.resolve as dire

from .conftest import (
    TEST_RES_DIR
)


def test_match_candidates_alto_candidate_with_coords():
    actual_matches = dire.match_candidates(f'{TEST_RES_DIR}/candidate/frk_alto',
                                      f'{TEST_RES_DIR}/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml')
    assert f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0001_part.xml' == actual_matches[0]


def test_match_candidates_both_txt_files():
    path_candidates = f'{TEST_RES_DIR}/candidate/txt'
    path_gt = f'{TEST_RES_DIR}/groundtruth/txt/1246734.gt.txt'
    actual_matches = dire.match_candidates(path_candidates, path_gt)
    assert f'{TEST_RES_DIR}/candidate/txt/OCR-Fraktur_1246734.txt' == actual_matches[0]


def test_match_candidates_fails_no_groundtruth():
    with pytest.raises(IOError) as exc:
        dire.match_candidates(
            f'{TEST_RES_DIR}/candidate/txt',
            './test/sresources/txt/no_gt.txt')
    assert "invalid groundtruth data path" in str(exc)


def test_match_candidates_fails_no_candidates():
    with pytest.raises(IOError) as exc:
        dire.match_candidates(
            './text/no_results',
            f'{TEST_RES_DIR}/txt/gt/1246734.txt')
    assert "invalid ocr result path" in str(exc)


def test_match_candidates_groundtruth_txt_candidate_alto():
    path_cd = f'{TEST_RES_DIR}/candidate/ara_alto'
    path_gt = f'{TEST_RES_DIR}/groundtruth/txt/217745.gt.txt'

    # act
    actual_matches = dire.match_candidates(path_cd, path_gt)

    # assert
    assert actual_matches[0] == f'{TEST_RES_DIR}/candidate/ara_alto/217745.xml'


def test_piece_to_text_alto_candidate_with_coords():
    """Check lines from ALTO candidate"""

    alto_path = f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0073_0512_01.xml'
    p1 = (300, 375)
    p2 = (6200, 3425)

    # act
    _as_lines, _ = dipre.file_to_text(alto_path, frame=(p1, p2), oneliner=False)
    _gt_type = digev._get_groundtruth_from_filename(alto_path)

    # assert
    assert _gt_type == 'n.a.'
    assert 166 == len(_as_lines)


def test_evaluate_single_alto_candidate_with_page_groundtruth(tmp_path):
    """Illustrate evaluation of single candidate
    with proper organization of groundtruth data"""

    # arrange
    eval_domain = tmp_path / 'candidate' / '1667522809_J_0001'
    eval_domain.mkdir(parents=True)
    gt_domain = tmp_path / 'groundtruth' / '1667522809_J_0001'
    gt_domain.mkdir(parents=True)
    evaluator = digev.Evaluator(eval_domain)
    evaluator.metrics = [digem.MetricChars()]
    # required for directory-like aggregation
    evaluator.domain_reference = gt_domain
    _candidate_src = os.path.join(f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0001_0002.xml')
    shutil.copy(_candidate_src, eval_domain)
    _gt_src = os.path.join(f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0002.art.gt.xml')
    _gt_dst = str(gt_domain / '1667522809_J_0001_0002.art.gt.xml')
    shutil.copy(_gt_src, _gt_dst)

    # act
    eval_entry = digev.EvalEntry(str(eval_domain / '1667522809_J_0001_0002.xml'))
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
    assert 39.19 == pytest.approx(defaults[2], rel=1e-3)
    # metric with stripped outliers (no outlier, of course!)
    assert 0 == result.n_outlier
    assert not result.cleared_result
    # reference size chars
    # changed from 5385 to 5309 since
    # children text wins over parent text
    assert 5309 == defaults[4]


def test_evaluate_page_groundtruth_with_itself(tmp_path):
    """Use Groundtruth as candidate and evaluate it
    with itself as reference data shall yield
    accuracy of nearly 100 percent"""

    # arrange
    eval_domain = tmp_path / 'candidate' / '1667522809_J_0001'
    eval_domain.mkdir(parents=True)
    gt_domain = tmp_path / 'groundtruth' / '1667522809_J_0001'
    gt_domain.mkdir(parents=True)
    evaluator = digev.Evaluator(eval_domain)
    evaluator.metrics = [digem.MetricChars()]
    evaluator.domain_reference = gt_domain
    _candidate_src = os.path.join(f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0002.art.gt.xml')
    _candidate_dst = str(eval_domain / '1667522809_J_0001_0002.xml')
    shutil.copy(_candidate_src, _candidate_dst)
    _gt_src = os.path.join(f'{TEST_RES_DIR}/groundtruth/page/1667522809_J_0001_0002.art.gt.xml')
    _gt_dst = str(gt_domain / '1667522809_J_0001_0002.art.gt.xml')
    shutil.copy(_gt_src, _gt_dst)

    # act
    eval_entry = digev.EvalEntry(str(eval_domain / '1667522809_J_0001_0002.xml'))
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
    # changed from 5385 to 5309 since
    # children text wins over parent text
    assert 5309 == defaults[4]


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
    evaluator = digev.Evaluator(path_dir_c)
    evaluator.domain_reference = path_dir_gt
    _metric_ca1 = digem.MetricChars()
    _metric_ca1._value = 95.70
    _metric_ca1.data_reference = 't' * 810
    _metric_ca2 = digem.MetricChars()
    _metric_ca2._value = 96.53
    _metric_ca2.data_reference = 't' * 675
    _metric_ca3 = digem.MetricChars()
    _metric_ca3._value = 94.91
    _metric_ca3.data_reference = 't' * 1395
    _metric_ca4 = digem.MetricChars()
    _metric_ca4._value = 94.40
    _metric_ca4.data_reference = 't' * 1466
    # outlier !
    _metric_ca5 = digem.MetricChars()
    _metric_ca5._value = 86.44
    _metric_ca5.data_reference = 't' * 1520
    _metric_ca6 = digem.MetricChars()
    _metric_ca6._value = 93.44
    _metric_ca6.data_reference = 't' * 1520

    entry1 = digev.EvalEntry(path_dir_c / 'eng' / 'urn+nbn+de+gbv+3+1-135654-p0403-5_eng.xml')
    entry1.path_g = str(path_dir_gt / 'eng' / 'urn+nbn+de+gbv+3+1-135654-p0403-5_eng.gt.xml')
    entry1.metrics = [_metric_ca1]
    entry2 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-816198-p0493-2_ger.xml')
    entry2.path_g = str('/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-816198-p0493-2_ger.gt.xml')
    entry2.metrics = [_metric_ca2]
    entry3 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-818383-p0034-5_ger.xml')
    entry3.path_g = '/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-818383-p0034-5_ger.gt.xml'
    entry3.metrics = [_metric_ca3]
    entry4 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-822479-p1119-4_ger.xml')
    entry4.path_g = '/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-822479-p1119-4_ger.gt.xml'
    entry4.metrics = [_metric_ca4]
    entry5 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-828020-p0173-6_ger.xml')
    entry5.path_g = '/data/ocr/groundtruth/odem/ger/urn+nbn+de+gbv+3+1-828020-p0173-6_ger.gt.xml'
    entry5.metrics = [_metric_ca5]
    entry6 = digev.EvalEntry(path_dir_c / 'ger' / 'urn+nbn+de+gbv+3+1-125584-p0314-6_ger.xml')
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


def test_no_groundtruth_at_all(tmp_path):
    """
    Behavior if no groundtruth found:
    Raise RuntimeError and exit, since evaluation
    doesn't make any sense so far
    """

    evaluator = digev.Evaluator(tmp_path)
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
    path_gt = TEST_RES_DIR / 'groundtruth' / 'page' / 'urn+nbn+de+gbv+3+1-792101-p0667-5_ger.gt.xml'
    assert path_gt.exists()
    eval_entry = digev.EvalEntry(path_gt)
    eval_entry.path_g = Path(path_gt).absolute()

    # act
    evaluator = digev.Evaluator('dummy_path')
    evaluator.metrics = [digem.SimilarityMetric()]
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
    eval_entry = digev.EvalEntry(path_cd)
    eval_entry.path_g = Path(path_gt).absolute()
    evaluator = digev.Evaluator('/data')
    evaluator.metrics = [digem.MetricIRPre(), digem.MetricIRRec()]
    evaluator.verbosity = 1

    # act
    evaluator.eval_entry(eval_entry)

    # assert
    assert eval_entry.metrics[0].label == 'Pre'
    assert eval_entry.metrics[0].value == 0.0
    assert eval_entry.metrics[1].label == 'Rec'
    assert eval_entry.metrics[1].value == 0.0


def test_handle_table_text_groundtruth():
    """Handle evaluation properly with very
    poor candidate data from ocr-ing a table
     
    legacy: "missing gt text from urn+nbn+de+gbv+3+1-126343-p0285-7_ger.gt.xml"
    """

    # arrange
    path_gt = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-126343-p0285-7_ger.gt.xml'
    path_cd = f'{TEST_RES_DIR}/candidate/frk_page/urn+nbn+de+gbv+3+1-126343-p0285-7_ger.xml'
    eval_entry = digev.EvalEntry(path_cd)
    eval_entry.path_g = path_gt

    # act
    evaluator = digev.Evaluator('/data')
    evaluator.metrics = [digem.MetricChars()]
    evaluator._wrap_eval_entry(eval_entry)

    # assert / legacy 1: 5.825 / legacy2: 4.0
    _result_cca = eval_entry.metrics[0].value
    assert 5.9 < _result_cca < 6.1


def test_get_box_from_empty_page():
    """How to deal with empty PAGE"""

    # arrange
    _path_gt = f'{TEST_RES_DIR}/groundtruth/page/urn+nbn+de+gbv+3+1-201080-p0034-8_ger.gt.xml'

    # act
    _p1, _p2 = digev.get_bounding_box(_path_gt)

    # assert 
    assert _p1 == (77, 58)
    assert _p2 == (2012, 2506)


def test_get_box_when_line_points_messy():
    """Ensure to get valid geometric data
    even if Coords@points have trailing
    whitespaces"""

    # arrange
    _path_gt = f'{TEST_RES_DIR}/groundtruth/page/rahbar-1771946695-00000040.xml'

    # act
    _p1, _p2 = digev.get_bounding_box(_path_gt)

    # assert
    assert _p1 == (368, 619)
    assert _p2 == (1298, 2314)


def test_handle_exception_invalid_alto_xml():
    """Handle invalid XML data

    ALTO data got corrupted by copying it around
    (the latest rows simply missing)
    """

    # arrange
    path_gt = f'{TEST_RES_DIR}/candidate/frk_alto/1667522809_J_0001_0256_corrupt.xml'
    eval_entry = digev.EvalEntry('dummy_candidate')
    eval_entry.path_g = path_gt

    # act
    evaluator = digev.Evaluator('dummy_path')
    evaluator.metrics = [digem.SimilarityMetric()]
    with pytest.raises(ParseError) as err:
        evaluator.eval_entry(eval_entry)

    # assert
    assert 'no element found' in err.value.args[0]
