# -*- coding: utf-8 -*-
"""OCR Metric Test Module"""

import random

import unicodedata

import pytest

from digital_eval.metrics import (
    MetricCA,
    MetricFM,
    MetricLA,
    MetricWA,
    MetricBoW,
    MetricPre,
    MetricRec,
    UC_NORMALIZATION,
    normalize_unicode,
    edit_distance,
    bag_of_tokens,
)


# default reference
THE_COMBINED_A_FOX = 'the á lazy brown fox jumps over the hump'
THE_LAZY_FOX = 'the lazy brown fox jumps over the hump'
THE_FOX_LAZY = 'the fox lazy brown jumps over the hump'
THE_FOX_INPUT_IR = 'the hump lazy brown fox fox fox jumps'


def test_metric_unicode_normalization_happens():
    """Normalization required and effects examined
    raw1 has "á" as {U+00E0} => gets canonical decomposed
    raw2 has "á" as {U+0061}+{U+0301}
    """
    
    # arrange
    raw1 = 'the á lazy brown fox jumps over the hump'
    raw2 = THE_COMBINED_A_FOX
    norm1 = normalize_unicode(raw1)
    norm2 = normalize_unicode(raw2)

    # act
    (ccr, distance, ref) = edit_distance(norm1, norm2)

    # assert
    # although both raw string look similar, they differ in fact
    assert raw1 != raw2
    # after normalization, they *are* similar
    assert norm1 == norm2
    assert 0 == distance
    assert 100 == pytest.approx(ccr, 0.001)
    assert ref == 41
    # the "á" char from raw1 string gets 
    # decomposed into {U+0061}+{U+0301}
    # by normalization with de-composition
    # therefore normalised str is 
    # one char longer
    assert len(raw1) + 1 == len(norm1)


def test_metric_unicode_normalization_not_happens():
    """Normalization has no effect since
    the letters "a" and "á" are still different
    after normalization, they just stay 
    {U+0061} and {U+0061}+{U+0301}

    =>

    results in this case in a *raw* edit distance 
    of "3" since the "á" gets decomposed into 2 
    unicode points and still there is one whitespace 
    too much left
    """
    
    # arrange
    raw1 = THE_LAZY_FOX
    raw2 = THE_COMBINED_A_FOX
    norm1 = unicodedata.normalize(UC_NORMALIZATION, raw1)
    norm2 = unicodedata.normalize(UC_NORMALIZATION, raw2)

    # act
    (_, distance, _) = edit_distance(norm1, norm2)

    # assert
    assert 3 == distance


def test_metric_unicode_normalization_textual_metric():
    """Normalization has no effect too, 
    but since the whitespaces get dropped, 
    the distance calculation just
    only concerns the *real* characters

    => distance decreases to "2"
    """
    
    # arrange
    c = MetricCA()
    c.reference = THE_LAZY_FOX
    c.candidate = THE_COMBINED_A_FOX

    # act
    c.value

    # assert
    assert 2 == c.diff


def test_metric_calculate_correctness():
    """explore edit-distance"""
    str1 = 'sthe lazy brown fox jumps overthe hump'
    str2 = 'fthe lazy brown fox jumps ouer the hump'
    (ccr, distance, _) = edit_distance(str1, str2)
    assert 3 == distance
    assert 92.10 == pytest.approx(ccr, 0.001)


def test_metric_characters_from_empty_gt():
    """What happens when there's something reported"""

    # arrange
    _metric = MetricCA()
    _metric.reference = ''
    _metric.candidate = 'fthe lazy brown fox jumps ouer the hump'

    # assert
    assert 0 == _metric.value
    assert 32 == _metric.diff
    # ensure whitespaces being dropped
    assert 'fthelazybrownfoxjumpsouerthehump' == _metric._data_candidate


def test_metric_letter_from_empty_gt_and_empty_candidate():
    """explore edit-distance"""

    # arrange
    _metric = MetricLA()
    _metric.reference = ''
    _metric.candidate = ''

    # assert
    assert 100 == _metric.value
    assert 0 == _metric.diff


def test_metric_wa_with_only_slight_difference():
    """simple word accurracy test"""

    # arrange
    _metric = MetricWA()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_FOX_LAZY

    # act
    _actual = _metric.value

    # assert
    # string has 38 characters, but tokens are only 8 present
    assert len(_metric.input_reference) == 38
    assert len(_metric.reference) == 8
    assert 2 == _metric.diff
    assert 75.0 == _actual


def test_metric_wa_with_identical_data():
    """simple word accurracy test"""

    # arrange
    _metric = MetricWA()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_LAZY_FOX

    # act
    _actual = _metric.value

    # assert
    assert 0 == _metric.diff
    assert 100 == _actual


def test_metric_bow_from_reasonable_input():
    """simple bag of words test"""

    # arrange
    _metric = MetricBoW()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_FOX_LAZY

    # act
    _actual = _metric.value

    # assert
    assert 0 == _metric.diff
    assert 100 == _actual


def test_metric_bow_from_empty_gt_and_empty_candidate():
    """how to handle empty data - means: no errors"""

    # arrange
    _metric = MetricBoW()
    _metric.reference = ''
    _metric.candidate = ''

    # act
    _actual = _metric.value

    # assert
    assert 0 == _metric.diff
    assert 100 == _actual


def test_metric_character_accuracy():
    """simple usage of MetricsCA"""

    str1 = 'sthe lazy brown fox jumps overthe hump'
    str2 = 'fthe lazy brown fox jumps ouer the hump'

    # arrange
    m = MetricCA()
    m.reference = str1
    m.candidate = str2

    # assert
    assert 93.75 == pytest.approx(m.value, rel=0.001, abs=0.001)


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

    (hit_rate, _, _) = bag_of_tokens(gt.split(), s2.split())
    assert 60.0 == pytest.approx(hit_rate, 0.001)


def test_ir_metric_precision_fox():
    """Basic test IR Precision with candidate 
    having all tokens included (minus stopwords)"""
    
    # arrange
    m_prec = MetricPre()
    m_prec.reference = THE_LAZY_FOX
    m_prec.candidate = THE_FOX_INPUT_IR

    # act
    actual = m_prec.value

    # assert
    assert actual == 1.0
    assert m_prec._data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}


def test_ir_metric_recall_fox():
    """Basic test IR Recall - everthing has been found
    (minus stoppwords)"""
    
    # arrange
    m_prec = MetricRec()
    m_prec.reference = THE_LAZY_FOX
    m_prec.candidate = THE_FOX_INPUT_IR

    # act
    actual = m_prec.value

    # assert
    assert actual == 1.0
    assert m_prec._data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}


IR_CANDIDATE_TEXT = 'the red fox'
def test_ir_metrics_precision_english_poor_candidate():
    """Example with all IR-Metrics and 
    a rather poor candidate"""

    # arrange
    pre = MetricPre()
    pre.reference = THE_LAZY_FOX
    pre.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 0.50 == pytest.approx(pre.value, 0.01)
    assert pre._data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}
    assert pre._data_candidate == {'red', 'fox'}


def test_ir_metrics_recall_english_poor_candidate():
    """Example with all IR-Metrics and 
    a rather poor candidate"""

    # arrange
    rec = MetricRec()
    rec.reference = THE_LAZY_FOX
    rec.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 0.20 == pytest.approx(rec.value , 0.01)


def test_ir_metrics_fmeasure_english_poor_candidate():
    """Example with all IR-Metrics and 
    a rather poor candidate"""

    # arrange
    fm = MetricFM()
    fm.reference = THE_LAZY_FOX
    fm.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 0.29 == pytest.approx(fm.value , 0.01)


IR_CANDIDATE_TEXT_GERMAN = 'dieser faule Fuchs springt die Hecke'
IR_REFERENCE_TEXT_GERMAN = 'Fuchs springt faule Hecke'
IR_REFERENCE_TEXT_GERMAN_POOR = 'forsche Fuchs hopst'
def test_ir_metrics_precision_german():
    """Candidate with german phrase
    and very nice candidate precision"""

    # arrange
    prec = MetricPre(['german'])
    prec.reference = IR_REFERENCE_TEXT_GERMAN
    prec.candidate = IR_CANDIDATE_TEXT_GERMAN

    # act
    assert prec.value == 1.0


def test_ir_metrics_recall_german():
    """Candidate with german phrase
    and very nice candidate recall"""

    # arrange
    rec = MetricRec(['german'])
    rec.reference = IR_REFERENCE_TEXT_GERMAN
    rec.candidate = IR_CANDIDATE_TEXT_GERMAN

    # act
    assert rec.value == 1.0


def test_ir_metrics_precision_german_poor_candidate():
    """Candidate with german phrase
    and rather poor candidate"""

    # arrange
    p = MetricPre(['german'])
    p.reference = IR_CANDIDATE_TEXT_GERMAN
    p.candidate = IR_REFERENCE_TEXT_GERMAN_POOR

    # assert
    assert p.value == pytest.approx(0.33)


def test_ir_metrics_recall_german_poor_candidate():
    """Candidate with german phrase
    and rather poor candidate"""

    # arrange
    r = MetricRec(['german'])
    r.reference = IR_CANDIDATE_TEXT_GERMAN
    r.candidate = IR_REFERENCE_TEXT_GERMAN_POOR

    # assert
    assert r.value == 0.25


def test_metrics_token_based_more_gt_than_tc():
    """1 fits for text with 7 tokens = 14.28"""

    # arrange
    gt = "der faulte Fuchs springt über die Hecke"
    tc = "faule springt Fuchs Hecke"

    # act
    ratio, diff, ref = edit_distance(gt.split(), tc.split())
    
    # assert
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
    assert 0.0 == pytest.approx(edit_distance(gt.split(), tc.split())[0])


def test_metrics_token_based_equal():
    """No mismatch => 100.00"""

    # arrange
    gt = "der fahle Fuchs springt über die Hecke"
    tc = "der fahle Fuchs springt über die Hecke"

    # act
    ratio, diff, _ = edit_distance(gt.split(), tc.split())

    # assert
    assert 100.00 == pytest.approx(ratio)
    assert 0 == diff


def test_metrics_token_based_same_length_but_final_differ():
    """One miss between two lists => 80.00"""

    # arrange
    gt = "faule Fuchs springt die Hecke"
    tc = "faule Fuchs springt über Hecke"

    # act
    normed, diff, _ = edit_distance(gt.split(), tc.split())
    
    # assert
    assert 80.00 == pytest.approx(normed)
    assert 1 == diff


def test_metrics_token_based_no_test_candidate():
    """Empty candidate yields word accuracy 0.0"""

    # arrange
    gt = "ein Dachs springt die Hecke"

    # act
    ratio, diff, _ = edit_distance(gt.split(), [])

    # assert
    assert 0.0 == pytest.approx(ratio)
    assert diff == 5
