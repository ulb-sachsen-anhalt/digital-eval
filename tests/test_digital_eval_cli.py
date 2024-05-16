# -*- coding: utf-8 -*-
"""OCR Evaluation Test Module"""

import shutil

import digital_eval.cli as dival

from .conftest import TEST_RES_DIR


_DOMAIN_LABEL = 'ger_frk'


def test_mwe_cli(tmp_path, capsys):
    """Minimum working example CLI 
    to fix *real* outcomes when playing with
    metrics implementations

    Match five candidates from subdir 'ger_frk' with
    total 13 references of according gt-subdir to 
    creates 4 default evaluation results (Ls,Cs)
    (no reference for candiate 1667522809_J_0001_0256_corrupt.xml)
    """

    # arrange
    dival.VERBOSITY = 1
    src_candidates = TEST_RES_DIR / 'candidate' / 'frk_alto'
    src_reference = TEST_RES_DIR / 'groundtruth' / 'page'
    dst_candidates = tmp_path / 'candidate' / _DOMAIN_LABEL
    dst_reference = tmp_path / 'reference' / _DOMAIN_LABEL
    tmp_candidate = shutil.copytree(src_candidates, dst_candidates)
    tmp_reference = shutil.copytree(src_reference, dst_reference)

    # assert final path segments do match by name frk_alto == frk_alto
    assert _DOMAIN_LABEL == tmp_candidate.name
    assert _DOMAIN_LABEL == tmp_reference.name

    # act
    _results = dival._main(dst_candidates, dst_reference,
                           dival.DEFAULT_OCR_METRICS, dival.DEFAULT_UTF8_NORM,
                           dival.DEFAULT_CALCULCATION, None)

    # assert
    captured = capsys.readouterr().out
    assert captured.startswith("[DEBUG] text normalized using 'NFC'")
    assert len(captured) == 1045
    std_lines = captured.split('\n')
    assert len(std_lines) == 11
    assert std_lines[1].startswith('[DEBUG] from "5" filtered "3" candidates')
    assert std_lines[4] == "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.20(5309), Ls:38.54(4383)(- 0.66)]"
    assert len(_results) == 4
    assert _results[0]
