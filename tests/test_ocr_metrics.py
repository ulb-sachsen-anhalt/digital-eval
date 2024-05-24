# -*- coding: utf-8 -*-
"""OCR Metric Test Module"""

import random

import pytest

import digital_eval.metrics as digem

# default reference
THE_COMBINED_A_FOX = 'the á lazy brown fox jumps over the hump'
THE_LAZY_FOX = 'the lazy brown fox jumps over the hump'
THE_FOX_LAZY = 'the fox lazy brown jumps over the hump'
THE_FOX_INPUT_IR = 'the hump lazy brown fox fox fox jumps'


def test_metric_unicode_normalization_textual_metric():
    """default OCR-D compliant UTF-8 normalization 
    yield similarity of 95%"
    """

    # arrange
    char_metric = digem.MetricChars()
    char_metric.reference = THE_LAZY_FOX
    char_metric.candidate = THE_COMBINED_A_FOX

    # actsert
    assert 95.0 == char_metric.value


def test_metric_characters_from_empty_gt():
    """Total un-similarity"""

    # arrange
    _metric = digem.MetricChars()
    # _metric.preprocessings = [_filter_whitespaces]
    _metric.reference = ''
    _metric.candidate = THE_LAZY_FOX

    # assert
    assert 0.0 == _metric.value


def test_metric_letter_from_empty_gt_and_empty_candidate():
    """Behavor: Similarity of empty strings"""

    # arrange
    _metric = digem.MetricLetters()
    _metric.reference = ''
    _metric.candidate = ''

    # assert
    assert 100.0 == _metric.value


def test_metric_words_with_only_slight_difference():
    """simple word accurracy test"""

    # arrange
    _metric = digem.MetricWords()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_FOX_LAZY

    # act
    _actual = _metric.value

    # assert
    # string has 38 characters, but tokens are only 8 present
    assert len(_metric.input_reference) == 38
    assert len(_metric.reference) == 8
    assert 75.0 == _actual


def test_metric_wa_with_identical_data():
    """simple word similarity for similar inputs"""

    # arrange
    _metric = digem.MetricWords()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_LAZY_FOX

    # assert
    assert 100 == _metric.value


def test_metric_bow_from_reasonable_input():
    """simple bag of words test"""

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = THE_LAZY_FOX
    _metric.candidate = THE_FOX_LAZY

    # act
    assert 100 == _metric.value


def test_metric_bow_from_empty_gt_and_empty_candidate():
    """how to handle empty data - means: no errors"""

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = ''
    _metric.candidate = ''

    # act
    _actual = _metric.value

    # assert
    assert 100 == _actual


def test_bow_ocrd_similarity_rate():
    """
    Behavior for bow similarity rate of two strings
    OCR-D spec: https://github.com/OCR-D/spec/blob/master/ocrd_eval.md#bag-of-words-error-rate
    """

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = "der Mann steht an der Ampel"
    _metric.candidate = "cer Mann fteht an der Ampel"

    # act
    _actual = _metric.value

    # assert
    assert 66.66 == pytest.approx(_actual, rel=1e-2)


def test_bow_ocrd_spec_similarity_rate_ref_contains_more_data():
    """
    Behavior for bow error rate of two strings
    OCR-D spec: https://github.com/OCR-D/spec/blob/master/ocrd_eval.md#bag-of-words-error-rate
    """

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = "der Mann steht an der roten Ampel"
    _metric.candidate = "cer Mann fteht an der Ampel"

    # act
    _actual = _metric.value

    # assert
    assert 61.54 == pytest.approx(_actual, rel=1e-2)


def test_bow_ocrd_spec_similarity_rate_ref_contains_less_data():
    """
    Behavior for bow error rate of two strings
    OCR-D spec: https://github.com/OCR-D/spec/blob/master/ocrd_eval.md#bag-of-words-error-rate
    """

    # arrange
    _metric = digem.MetricBoW()
    _metric.reference = "der Mann steht an der Ampel"
    _metric.candidate = "cer Mann fteht an der schönen roten Ampel"

    # act
    _actual = _metric.value

    # assert
    assert 57.14 == pytest.approx(_actual, rel=1e-2)


def test_metric_character_accuracy():
    """simple usage of MetricsCA"""

    str1 = 'sthe lazy brown fox jumps overthe hump'
    str2 = 'fthe lazy brown fox jumps ouer the hump'

    # arrange
    char_metric = digem.MetricChars()
    char_metric.reference = str1
    char_metric.candidate = str2

    # assert
    assert 92.31 == pytest.approx(char_metric.value, rel=0.001, abs=0.001)


def test_metric_bot_ident():
    """BOW with identical tokens"""

    gt1 = "the lazy brown fox jumps over the hump again and again three times the dude"
    list2 = list(gt1.split())
    random.shuffle(list2)
    str2 = ' '.join(list2)

    result = digem.bag_of_tokens(gt1.split(), str2.split())
    assert result == 1.0
    assert len(gt1.split()) == len(str2.split())


def test_metric_bot_candidate_with_only_repetitions():
    """
    Behaviour of BOW with multiple identical entries
    """

    gt1  = "the dizzy brown fox jumps"
    str2 = "the dizzy brown fox fox fox jumps"

    # actsert
    assert 0.83 == pytest.approx(digem.bag_of_tokens(gt1.split(), str2.split()), abs=1e-2)


def test_metric_bot_miss_tokens():
    """BOW with missing tokens in candidate text"""

    gt1 = "the lazy brown fox jumps"
    str2 = "the brown fux jumps"

    # acsert
    assert 0.66 == pytest.approx(digem.bag_of_tokens(gt1.split(), str2.split()), abs=1e-2)


def test_ir_metric_precision_fox():
    """Basic test IR Precision with candidate
    having all tokens included (minus stopwords)"""

    # arrange
    m_prec = digem.MetricIRPre()
    m_prec.reference = THE_LAZY_FOX
    m_prec.candidate = THE_FOX_INPUT_IR

    # act
    actual = m_prec.value

    # assert
    assert actual == 100.0
    assert m_prec._data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}


def test_ir_metric_recall_fox():
    """Basic test IR Recall - everthing has been found
    (minus stoppwords)"""

    # arrange
    m_prec = digem.MetricIRRec()
    m_prec.reference = THE_LAZY_FOX
    m_prec.candidate = THE_FOX_INPUT_IR

    # act
    actual = m_prec.value

    # assert
    assert actual == 100.0
    assert m_prec._data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}


IR_CANDIDATE_TEXT = 'the red fox'


def test_ir_metrics_precision_english_poor_candidate():
    """Example with all IR-Metrics and
    a rather poor candidate"""

    # arrange
    pre = digem.MetricIRPre()
    pre.reference = THE_LAZY_FOX
    pre.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 50.0 == pytest.approx(pre.value, 0.01)
    assert pre._data_reference == {'brown', 'fox', 'jumps', 'lazy', 'hump'}
    assert pre._data_candidate == {'red', 'fox'}


def test_ir_metrics_recall_english_poor_candidate():
    """Example with all IR-Metrics and
    a rather poor candidate"""

    # arrange
    rec = digem.MetricIRRec()
    rec.reference = THE_LAZY_FOX
    rec.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 20.0 == pytest.approx(rec.value, 0.01)


def test_ir_metrics_fmeasure_english_poor_candidate():
    """Example with all IR-Metrics and
    a rather poor candidate"""

    # arrange
    metric_fm = digem.MetricIRFM()
    metric_fm.reference = THE_LAZY_FOX
    metric_fm.candidate = IR_CANDIDATE_TEXT

    # assert
    assert 28.57 == pytest.approx(metric_fm.value, 0.01)


IR_CANDIDATE_TEXT_GERMAN = 'dieser faule Fuchs springt die Hecke'
IR_REFERENCE_TEXT_GERMAN = 'Fuchs springt faule Hecke'
IR_REFERENCE_TEXT_GERMAN_POOR = 'forsche Fuchs hopst'


def test_ir_metrics_precision_german():
    """Candidate with german phrase
    and very nice candidate precision"""

    # arrange
    prec = digem.MetricIRPre(languages=['german'])
    prec.reference = IR_REFERENCE_TEXT_GERMAN
    prec.candidate = IR_CANDIDATE_TEXT_GERMAN

    # act
    assert prec.value == 100.0


def test_ir_metrics_recall_german():
    """Candidate with german phrase
    and very nice candidate recall"""

    # arrange
    rec = digem.MetricIRRec(languages=['german'])
    rec.reference = IR_REFERENCE_TEXT_GERMAN
    rec.candidate = IR_CANDIDATE_TEXT_GERMAN

    # act
    assert rec.value == 100.0


def test_ir_metrics_precision_german_poor_candidate():
    """Candidate with german phrase
    and rather poor candidate"""

    # arrange
    metric_pre = digem.MetricIRPre(languages=['german'])
    metric_pre.reference = IR_CANDIDATE_TEXT_GERMAN
    metric_pre.candidate = IR_REFERENCE_TEXT_GERMAN_POOR

    # assert
    assert metric_pre.value == pytest.approx(33.33, 1e-2)


def test_ir_metrics_recall_german_poor_candidate():
    """Candidate with german phrase
    and rather poor candidate"""

    # arrange
    metric_rec = digem.MetricIRRec(languages=['german'])
    metric_rec.reference = IR_CANDIDATE_TEXT_GERMAN
    metric_rec.candidate = IR_REFERENCE_TEXT_GERMAN_POOR

    # assert
    assert metric_rec.value == 25.0


def test_metrics_token_based_more_gt_than_tc():
    """token edit distance with
    * 2 exchanges (first 2 tokens), followed by
    * 3 insertions ('springt', 'über', 'die')
    => total 5 edit operations required

    aligned means: (7 - 5) / 7
    => 0.2857

    as percents
    => 28.57 %
    """

    # arrange
    gt1 = "der faulte Fuchs springt über die Hecke".split()
    cand = "faule springt Fuchs Hecke".split()

    # act
    m_word = digem.MetricWords()
    m_word._data_reference = gt1
    m_word._data_candidate = cand

    # assert
    assert 28.57 == pytest.approx(m_word.value, rel=1e-2)
    assert len(cand) + 3 == len(gt1)



########################################################### OCR-Pipeline-Tests

# def test_step_estimateocr_textline_conversions():
#     """Test functional behavior for valid ALTO-output"""
#
#     test_data = os.path.join('tests', 'resources', '500_gray00003.xml')
#
#     # act
#     # pylint: disable=protected-access
#     xml_data = ET.parse(test_data)
#     lines = get_lines(xml_data)
#     (_, n_lines, _, _, n_lines_out) = textlines2data(lines)
#
#     assert n_lines == 360
#     assert n_lines_out == 346
#
# @mock.patch("requests.post")
# def test_step_estimateocr_lines_and_tokens_err_ratio(mock_requests):
#     """Test behavior of for valid ALTO-output"""
#
#     # arrange
#     test_data = os.path.join(PROJECT_ROOT_DIR,
#                              'tests', 'resources', '500_gray00003.xml')
#     mock_requests.side_effect = _fixture_languagetool
#     params = {'service_url': 'http://localhost:8010/v2/check',
#               'language': 'de-DE',
#               'enabled_rules': 'GERMAN_SPELLER_RULE'
#               }
#     step = StepEstimateOCR(params)
#     step.path_in = test_data
#
#     # act
#     step.execute()
#
#     assert step.statistics
#     assert mock_requests.called == 1
#     assert step.n_errs == 548
#     assert step.n_words == 2636
#     assert step.statistics[0] == pytest.approx(79.211, rel=1e-3)
#
# @mock.patch("requests.post")
# def test_step_estimateocr_lines_and_tokens_hit_ratio(mock_requests):
#     """Test behavior of for valid ALTO-output"""
#
#     # arrange
#     test_data = os.path.join(PROJECT_ROOT_DIR,
#                              'tests', 'resources', '500_gray00003.xml')
#     mock_requests.side_effect = _fixture_languagetool
#     params = {'service_url': 'http://localhost:8010/v2/check',
#               'language': 'de-DE',
#               'enabled_rules': 'GERMAN_SPELLER_RULE'
#               }
#     step = StepEstimateOCR(params)
#     step.path_in = test_data
#
#     # act
#     step.execute()
#
#     assert mock_requests.called == 1
#     err_ratio = (step.n_errs / step.n_words) * 100
#     assert err_ratio == pytest.approx(20.789, rel=1e-3)
#
#     # revert metric to represent hits
#     # to hit into positive compliance
#     hits = (step.n_words - step.n_errs) / step.n_words * 100
#     assert hits == pytest.approx(79.21, rel=1e-3)
#
#     # holds this condition?
#     assert hits == pytest.approx(100 - err_ratio, rel=1e-9)