# -*- coding: utf-8 -*-
"""OCR Metric Test Module"""

import random

import pytest

import digital_eval.metrics as deme

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
    norm1 = deme.normalize_unicode(raw1, uc_norm_by=deme.UC_NORMALIZATION_NFKD)
    norm2 = deme.normalize_unicode(raw2, uc_norm_by=deme.UC_NORMALIZATION_NFKD)

    # act
    similarity = deme.levenshtein_norm(norm1, norm2)
    assert 1.0 == pytest.approx(similarity, abs=1e-6)

    # assert
    # although both raw string look similar, they differ in fact
    assert raw1 != raw2
    # after normalization, they *are* similar
    assert norm1 == norm2
    assert len(norm1) == 41
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
    {U+0061} and {U+00e1} for NFC and {U+0061}+{U+0301} for NFKD
    """

    # arrange
    raw1 = THE_LAZY_FOX
    raw2 = THE_COMBINED_A_FOX
    norm1_nfc = deme.normalize_unicode(raw1, uc_norm_by=deme.UC_NORMALIZATION_DEFAULT)
    norm1_nfkd = deme.normalize_unicode(raw1, uc_norm_by=deme.UC_NORMALIZATION_NFKD)
    norm2_nfc = deme.normalize_unicode(raw2, uc_norm_by=deme.UC_NORMALIZATION_DEFAULT)
    norm2_nfkd = deme.normalize_unicode(raw2, uc_norm_by=deme.UC_NORMALIZATION_NFKD)

    # act
    sim_nfc = deme.levenshtein_norm(norm1_nfc, norm2_nfc)
    sim_nfkd = deme.levenshtein_norm(norm1_nfkd, norm2_nfkd)

    # assert
    assert 0.95 == sim_nfc
    assert 0.92 == pytest.approx(sim_nfkd, 1e-2)


def test_metric_calculate_character_edit_distance():
    """explore edit-distance"""
    str1 = 'sthe lazy brown fox jumps overthe hump'
    str2 = 'fthe lazy brown fox jumps ouer the hump'
    distance = deme.levenshtein_norm(str1, str2)
    assert 0.923 == pytest.approx(distance, 1e-4)


def test_metric_bot_ident():
    """BOW with identical tokens"""

    gt1 = "the lazy brown fox jumps over the hump again and again three times the dude"
    list2 = list(gt1.split())
    random.shuffle(list2)
    str2 = ' '.join(list2)

    similarity = deme.bag_of_tokens(gt1.split(), str2.split())
    assert similarity == 1.0
    assert len(gt1.split()) == len(str2.split())


def test_metric_bot_candidate_with_only_repetitions():
    """
    Behaviour of BOW with multiple identical entries
    """

    gt1 = "the dizzy brown fox jumps"
    str2 = "the dizzy brown fox fox fox jumps"

    # actsert
    assert 0.833 == pytest.approx(deme.bag_of_tokens(gt1.split(), str2.split()), 1e-3)


def test_metric_bot_miss_tokens():
    """BOW with missing tokens in candidate text"""

    gt1 = "the lazy brown fox jumps"
    str2 = "the brown fux jumps"

    # acsert
    assert 0.66 == pytest.approx(deme.bag_of_tokens(gt1.split(), str2.split()), abs=1e-2)


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
    result = deme.levenshtein_norm(gt1, cand)

    # assert
    assert 0.2857 == pytest.approx(result, rel=1e-4)
    assert len(cand) + 3 == len(gt1)


def test_metrics_token_based_equal():
    """No mismatch => 100.00"""

    # arrange
    gt1  = "der fahle Fuchs springt über die Hecke"
    cand = "der fahle Fuchs springt über die Hecke"

    # act
    sim = deme.levenshtein_norm(gt1.split(), cand.split())

    # assert
    assert 1.0 == sim


def test_metrics_token_based_no_test_candidate():
    """Empty candidate yields if result is
    inverted total distance of 1.0
    """

    # arrange
    gt1 = "ein Dachs springt die Hecke"

    # act
    diff = deme.levenshtein_norm(gt1.split(), [], inverse=True)

    # assert
    assert diff == 1.0
