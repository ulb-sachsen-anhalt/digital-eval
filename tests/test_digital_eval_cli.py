# -*- coding: utf-8 -*-
"""OCR Evaluation Test Module"""

import shutil

from pathlib import Path

import digital_eval.cli as dig
import digital_eval.preprocessing as dipre

from .conftest import TEST_RES_DIR


_DOMAIN_LABEL = 'ger_frk'


def test_mwe_cli_defaults(tmp_path, capsys):
    """Minimum working example CLI 
    to fix *real* outcomes when playing with
    metrics implementations

    Match five candidates from subdir 'ger_frk' with
    total 13 references of according gt-subdir to 
    creates 4 default evaluation results (Ls,Cs)
    (no reference for candiate 1667522809_J_0001_0256_corrupt.xml)
    """

    # arrange
    dig.VERBOSITY = 1
    src_candidates = TEST_RES_DIR / 'candidate' / 'frk_alto'
    src_reference = TEST_RES_DIR / 'groundtruth' / 'page'
    dst_candidates = tmp_path / 'candidate' / _DOMAIN_LABEL
    dst_reference = tmp_path / 'reference' / _DOMAIN_LABEL
    tmp_candidate: Path = shutil.copytree(src_candidates, dst_candidates)
    tmp_reference: Path = shutil.copytree(src_reference, dst_reference)

    # assert final path segments do match by name frk_alto == frk_alto
    assert _DOMAIN_LABEL == tmp_candidate.name
    assert _DOMAIN_LABEL == tmp_reference.name

    # act
    cli_args = {"candidates": dst_candidates, "reference": dst_reference,
                "metrics": dig.DEFAULT_OCR_METRICS,
                "verbosity": 1,
                "utf8": dig.DEFAULT_UTF8_NORM,
                "sequential": True}
    eval_results = dig.start_evaluation(cli_args)

    # assert
    assert len(eval_results) == 4
    captured = capsys.readouterr().out
    std_lines = captured.split('\n')
    assert len(std_lines) == 11
    assert std_lines[0] == "[DEBUG] text normalized using 'NFC' code points for 'Cs,Ls'"
    assert str(std_lines[1]).startswith('[DEBUG] from "5" filtered "3" candidates')
    assert std_lines[4] == "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.20(5309), Ls:38.54(4383)(- 0.66)]"


def test_mwe_cli_utf8_nfkd(tmp_path, capsys):
    """Minimum working example CLI 
    to fix *real* outcomes when playing with
    metrics implementations

    Match five candidates from subdir 'ger_frk' with
    total 13 references of according gt-subdir to 
    creates 4 default evaluation results (Ls,Cs)
    (no reference for candiate 1667522809_J_0001_0256_corrupt.xml)
    """

    # arrange
    dig.VERBOSITY = 1
    src_candidates = TEST_RES_DIR / 'candidate' / 'frk_alto'
    src_reference = TEST_RES_DIR / 'groundtruth' / 'page'
    dst_candidates = tmp_path / 'candidate' / _DOMAIN_LABEL
    dst_reference = tmp_path / 'reference' / _DOMAIN_LABEL
    tmp_candidate: Path = shutil.copytree(src_candidates, dst_candidates)
    tmp_reference: Path = shutil.copytree(src_reference, dst_reference)

    # assert final path segments do match by name frk_alto == frk_alto
    assert _DOMAIN_LABEL == tmp_candidate.name
    assert _DOMAIN_LABEL == tmp_reference.name

    # act
    cli_args = {"candidates": dst_candidates, "reference": dst_reference,
                "metrics": dig.DEFAULT_OCR_METRICS,
                "verbosity": 1,
                "utf8": dipre.UC_NORMALIZATION_NFKD,
                "sequential": True}
    eval_results = dig.start_evaluation(cli_args)

    # assert
    assert len(eval_results) == 4
    captured = capsys.readouterr().out
    std_lines = captured.split('\n')
    assert len(std_lines) == 11
    assert std_lines[0] == "[DEBUG] text normalized using 'NFKD' code points for 'Cs,Ls'"
    assert str(std_lines[1]).startswith('[DEBUG] from "5" filtered "3" candidates')
    assert std_lines[4] == "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.10(5363), Ls:38.52(4437)(- 0.58)]"


def test_mwe_cli_data_resolving(tmp_path, capsys):
    """Minimum working example CLI 
    to inspect behavior for intermediate missmatches
    => OCR-D GT-PAGE directory
    """

    # arrange
    dig.VERBOSITY = 1
    src_candidates = TEST_RES_DIR / 'candidate' / 'frk_alto'
    src_reference = TEST_RES_DIR / 'groundtruth' / 'page'
    dst_candidates = tmp_path / 'candidate' / _DOMAIN_LABEL
    dst_reference = tmp_path / 'reference' / _DOMAIN_LABEL /'GT-PAGE'
    tmp_candidate: Path = shutil.copytree(src_candidates, dst_candidates)
    tmp_reference: Path = shutil.copytree(src_reference, dst_reference)

    # assert final path segments do match by name frk_alto == frk_alto
    assert _DOMAIN_LABEL == tmp_candidate.name

    # act
    cli_args = {"candidates": dst_candidates, "reference": dst_reference,
                "metrics": dig.DEFAULT_OCR_METRICS,
                "verbosity": 1,
                "utf8": dipre.UC_NORMALIZATION_NFKD,
                "sequential": True}
    eval_results = dig.start_evaluation(cli_args)

    # assert
    assert len(eval_results) == 4
    captured = capsys.readouterr().out
    std_lines = captured.split('\n')
    assert len(std_lines) == 12
    assert std_lines[0] == "[WARN ] base domains 'ger_frk' and 'GT-PAGE' mismatch, aggregation might fail!"
    assert std_lines[1] == "[DEBUG] text normalized using 'NFKD' code points for 'Cs,Ls'"
    assert std_lines[5] == "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.10(5363), Ls:38.52(4437)(- 0.58)]"
