# -*- coding: utf-8 -*-
"""OCR Metric Test Module"""

import random

import pytest

import digital_eval.metrics as digem


def test_metric_calculate_character_edit_distance():
    """explore edit-distance"""
    str1 = 'sthe lazy brown fox jumps overthe hump'
    str2 = 'fthe lazy brown fox jumps ouer the hump'
    distance = digem.levenshtein_norm(str1, str2)
    assert 0.923 == pytest.approx(distance, 1e-4)


def test_metric_bot_ident():
    """BOW with identical tokens"""

    gt1 = "the lazy brown fox jumps over the hump again and again three times the dude"
    list2 = list(gt1.split())
    random.shuffle(list2)
    str2 = ' '.join(list2)

    similarity = digem.bag_of_tokens(gt1.split(), str2.split())
    assert similarity == 1.0
    assert len(gt1.split()) == len(str2.split())


def test_metric_bot_candidate_with_only_repetitions():
    """
    Behaviour of BOW with multiple identical entries
    """

    gt1 = "the dizzy brown fox jumps"
    str2 = "the dizzy brown fox fox fox jumps"

    # actsert
    assert 0.833 == pytest.approx(digem.bag_of_tokens(gt1.split(), str2.split()), 1e-3)


def test_metric_bot_miss_tokens():
    """BOW with missing tokens in candidate text"""

    gt1 = "the lazy brown fox jumps"
    str2 = "the brown fux jumps"

    # acsert
    assert 0.66 == pytest.approx(digem.bag_of_tokens(gt1.split(), str2.split()), abs=1e-2)


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
    result = digem.levenshtein_norm(gt1, cand)

    # assert
    assert 0.2857 == pytest.approx(result, rel=1e-4)
    assert len(cand) + 3 == len(gt1)


def test_metrics_token_based_equal():
    """No mismatch => 100.00"""

    # arrange
    gt1  = "der fahle Fuchs springt über die Hecke"
    cand = "der fahle Fuchs springt über die Hecke"

    # act
    sim = digem.levenshtein_norm(gt1.split(), cand.split())

    # assert
    assert 1.0 == sim


def test_metrics_token_based_no_test_candidate():
    """Empty candidate yields if result is
    inverted total distance of 1.0
    """

    # arrange
    gt1 = "ein Dachs springt die Hecke"

    # act
    diff = digem.levenshtein_norm(gt1.split(), [], inverse=True)

    # assert
    assert diff == 1.0
