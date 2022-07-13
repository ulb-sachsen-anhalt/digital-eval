# -*- coding: utf-8 -*-
"""OCR Metric Test Module"""

import random

import unicodedata

import pytest

from digital_eval.metrics import (
    character_accuracy,
    bag_of_tokens,
    ir_fmeasure,
    ir_recall,
    ir_precision,
    token_based,
)

def test_metric_normalization():
    """Normalization required
    raw1 has "á" as {U+00E0} 
    str2 has "á" as {U+0061}+{U+0301} 
    """
    
    # arrange
    raw1 = 'the á lazy brown fox jumps over the hump'
    raw2 = 'the á lazy brown fox jumps over the hump'
    norm1 = unicodedata.normalize('NFKD', raw1)
    norm2 = unicodedata.normalize('NFKD', raw2)

    # act
    (ccr, distance, ref) = character_accuracy(norm1, norm2)

    # assert
    assert 0 == distance
    assert 100 == pytest.approx(ccr, 0.001)
    assert ref == 41
    # the "á" char from raw string gets 
    # decomposed into {U+0061}+{U+0301}
    # by normalization with de-composition
    # therefore normalised str is 
    # one char longer
    assert len(raw1) + 1 == len(norm1)


def test_metric_calculate_correctness():
    """explore edit-distance"""
    str1 = 'sthe lazy brown fox jumps overthe hump'
    str2 = 'fthe lazy brown fox jumps ouer the hump'
    (ccr, distance, _) = character_accuracy(str1, str2)
    assert 3 == distance
    assert 92.10 == pytest.approx(ccr, 0.001)


def test_metric_bot_ident():
    """BOW with identical tokens"""

    gt = "the lazy brown fox jumps over the hump again and again three times the dude"
    l2 = list(gt.split())
    random.shuffle(l2)
    s2 = ' '.join(l2)

    (hit_rate, _, _) = bag_of_tokens(gt, s2)
    assert 100.0 == pytest.approx(hit_rate, 0.001)


def test_metric_bot_more_candidates():
    """
    Behaviour of BOW with multiple identical entries
    """

    gt = "the dizzy brown fox jumps"
    s2 = "the dizzy brown fox fox fox jumps"

    (hit_rate, _, _) = bag_of_tokens(gt, s2)
    assert 100.0 == pytest.approx(hit_rate, 0.001)


def test_metric_bot_miss_tokens():
    """BOW with missing tokens in candidate text"""

    gt = "the lazy brown fox jumps"
    s2 = "the brown fux jumps"

    (hit_rate, _, _) = bag_of_tokens(gt, s2)
    assert 60.0 == pytest.approx(hit_rate, 0.001)


def test_ir_metrics_basics():
    """Some explorative tests with IR metrics"""
    
    gt = "the lazy brown fox jumps"
    td = "the lazy brown fox fox fox jumps"

    _pre, _n_ref1 = ir_precision(gt, td, ['german'])
    _rec, _n_ref2 = ir_recall(gt, td, ['german'])
    _fm, _n_ref3 = ir_fmeasure(gt, td, ['german'])

    assert _pre == 1.0
    assert _rec == 1.0
    assert _fm == 1.0
    assert _n_ref1 == 5
    assert _n_ref1 == _n_ref2
    assert _n_ref2 == _n_ref3


def test_ir_metrics_bad():
    gt = "the lazy brown fox jumps over"
    td = "the red fox"

    _pre, _n_ref1 = ir_precision(gt, td, ['german'])
    _rec, _n_ref2 = ir_recall(gt, td, ['german'])
    _fm, _n_ref3 = ir_fmeasure(gt, td, ['german'])

    assert 0.66 == pytest.approx(_pre , 0.01)
    assert 0.33 == pytest.approx(_rec , 0.01)
    assert 0.44 == pytest.approx(_fm , 0.01)
    assert _n_ref1 == 6


def test_ir_metrics_german_stopwords():
    gt = "dieser faule Fuchs springt die Hecke"
    td = "faule Fuchs springt Hecke"

    _pre, _n_ref1 = ir_precision(gt, td, ['german'])
    _rec, _n_ref2 = ir_recall(gt, td, ['german'])
    _fm, _n_ref3 = ir_fmeasure(gt, td, ['german'])

    assert _pre == 1.0
    assert _rec == 1.0
    assert _fm == 1.0


def test_metrics_token_based_more_gt_than_tc():
    """1 fits for text with 7 tokens = 14.28"""

    # arrange
    gt = "der faulte Fuchs springt über die Hecke"
    tc = "faule springt Fuchs Hecke"

    # act
    ratio, diff, ref = token_based(gt, tc)
    
    # sert
    assert diff == 5
    assert 28.56 == pytest.approx(ratio, rel=1e-2)
    assert ref == len(gt.split())


def test_metrics_token_based_more_tc_than_gt():
    """test candidate is longer than groundtruth 
    would allow => 0.0"""

    # arrange
    gt = "der faule Fuchs"
    tc = "der faule Fuchs springt über die"

    # actsert
    assert 0.0 == pytest.approx(token_based(gt, tc)[0])


def test_metrics_token_based_equal():
    """No missmatch => 100.00"""

    # arrange
    gt = "der fahle Fuchs springt über die Hecke"
    tc = "der fahle Fuchs springt über die Hecke"

    # act
    ratio, diff, _ = token_based(gt, tc)

    # assert
    assert 100.00 == pytest.approx(ratio)
    assert 0 == diff


def test_metrics_token_based_same_length_but_final_differ():
    """One miss between two lists => 80.00"""

    # arrange
    gt = "faule Fuchs springt die Hecke"
    tc = "faule Fuchs springt über Hecke"

    # act
    normed, diff, _ = token_based(gt, tc)
    
    # assert
    assert 80.00 == pytest.approx(normed)
    assert 1 == diff


def test_metrics_token_based_no_test_candidate():
    """Empty test candidate => 0.0"""

    # arrange
    gt = "ein Dachs springt die Hecke"
    tc = ""

    # act
    ratio, diff, _ = token_based(gt, tc)

    # assert
    assert 0.0 == pytest.approx(ratio)
    assert diff == 5
