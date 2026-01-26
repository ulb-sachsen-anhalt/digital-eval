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
    assert std_lines[0] == "[WARN ] base 'ger_frk' and 'GT-PAGE' mismatch, aggregation might be inaccurate!"
    assert std_lines[1] == "[DEBUG] text normalized using 'NFKD' code points for 'Cs,Ls'"
    assert std_lines[5] == "[DEBUG] [1667522809_J_0001_0002](art) [Cs:39.10(5363), Ls:38.52(4437)(- 0.58)]"


def test_single_candidate_file_cli(tmp_path, capsys):
    """Test CLI with a single candidate file as argument
    
    Ensures that a single candidate file can be passed directly
    instead of a directory and is processed correctly.
    """

    # arrange
    dig.VERBOSITY = 1
    src_candidate_file = TEST_RES_DIR / 'candidate' / 'frk_alto' / '1667522809_J_0001_0002.xml'
    src_reference = TEST_RES_DIR / 'groundtruth' / 'page'
    # Place the file in a subdirectory to simulate normal structure
    dst_candidate_dir = tmp_path / 'candidate' / 'ger_frk'
    dst_candidate_file = dst_candidate_dir / '1667522809_J_0001_0002.xml'
    # Match the reference directory name to candidate directory name
    dst_reference = tmp_path / 'reference' / 'ger_frk'
    
    # create directories and copy single file
    dst_candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_candidate_file, dst_candidate_file)
    tmp_reference: Path = shutil.copytree(src_reference, dst_reference)

    # assert file exists
    assert dst_candidate_file.is_file()
    assert tmp_reference.is_dir()

    # act - pass single file as candidates argument
    cli_args = {"candidates": dst_candidate_file, "reference": tmp_reference,
                "metrics": dig.DEFAULT_OCR_METRICS,
                "verbosity": 1,
                "utf8": dig.DEFAULT_UTF8_NORM,
                "sequential": True}
    eval_results = dig.start_evaluation(cli_args)

    # assert - should process single file and match with reference
    # Note: aggregate() creates multiple results: one per metric per domain/type
    # With 2 metrics (Cs, Ls) and by_type=True, we get multiple aggregation results
    assert len(eval_results) > 0
    # Check that the specific file was processed by checking eval_keys
    eval_keys = [result.eval_key for result in eval_results]
    assert any('ger_frk' in key or '1667522809_J_0001_0002' in key for key in eval_keys)
    captured = capsys.readouterr().out
    std_lines = captured.split('\n')
    assert std_lines[0] == "[DEBUG] text normalized using 'NFC' code points for 'Cs,Ls'"
    # Verify the specific file appears in the output
    assert any('1667522809_J_0001_0002' in line for line in std_lines)
