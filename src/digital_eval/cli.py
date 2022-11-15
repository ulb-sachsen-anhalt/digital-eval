# -*- coding: utf-8 -*-
"""OCR QA Evaluation CLI"""

import argparse
import os
import sys

import datetime as dt

from typing import (
    List
)

from digital_eval import (
    find_groundtruth,
    gather_candidates,
    Evaluator,
    report_stdout,
    ocr_to_text,
    UC_NORMALIZATION,
    accuracy_for,
    error_for,
    MetricChars,
    MetricLetters,
    MetricWords,
    MetricBoW,
    MetricIRPre,
    MetricIRRec,
    MetricIRFM
)

# script constants
DEFAULT_VERBOSITY = 0
VERBOSITY = DEFAULT_VERBOSITY
EVAL_VERBOSITY = DEFAULT_VERBOSITY

# calculations
DEFAULT_CALCULCATION = 'acc'
CALC_DICT = {
    'acc' : accuracy_for,
    'accuracy' : accuracy_for,
    'err' : error_for, 
    'error' : error_for,
}
DEFAULT_UTF8_NORM = UC_NORMALIZATION

# metrics
DEFAULT_OCR_METRICS = 'Cs,Ls'
DEFAULT_OCR_METRIC_PREPROCESSINGS = ''
DEFAULT_OCR_METRIC_POSTPROCESSINGS = ''
METRIC_DICT = {
    'Cs' : MetricChars,
    'Characters' : MetricChars,
    'Ls' : MetricLetters,
    'Letters' : MetricLetters,
    'Ws' : MetricWords,
    'Words' : MetricWords,
    'BoWs' : MetricBoW,
    'BagOfWords' : MetricBoW,
    'IRPre' : MetricIRPre,
    'Pre': MetricIRPre,
    'Precision' : MetricIRPre,
    'IRRec' : MetricIRRec,
    'Rec': MetricIRRec,
    'IRFMeasure' : MetricIRFM,
    'FM' : MetricIRFM,
}


def _get_info():
    here = os.path.abspath(os.path.dirname(__file__))
    _v = ''
    _t = ''
    _fp = os.path.join(here, 'VERSION')
    with open(_fp) as fp:
        _v = fp.read()
    _t = dt.datetime.fromtimestamp(os.stat(_fp).st_mtime).strftime("%Y-%m-%d")
    return f'v{_v}/{_t}'


def _initialize_metrics(the_metrics, norm, calc) -> List:
    _tokens = the_metrics.split(',')
    try:
        return [METRIC_DICT[m](normalization=norm, calc_func=CALC_DICT[calc]) 
                for m in _tokens]
    except KeyError as _err:
        _keys = ','.join(METRIC_DICT.keys()) + ','.join(CALC_DICT.keys())
        _msg = f"Unknown: '{_err.args[0]}'.\nPlease use one of the following keys: '{_keys}'."
        print(_msg)
        sys.exit(1)


########
# MAIN #
########
def _main(path_candidates, path_reference, metrics, utf8norm, calc, xtra, is_legacy=False, is_sequential=False):

    # create basic evaluator instance
    evaluator = Evaluator(path_candidates, VERBOSITY, xtra)
    evaluator.metrics = _initialize_metrics(metrics, norm=utf8norm, calc=calc)
    evaluator.calc = calc
    if VERBOSITY >= 1:
        print(f"[DEBUG] text normalized using '{utf8norm}' calculate '{calc}' metric values for '{metrics}'")
    
    if is_legacy:
        evaluator.to_text_func = ocr_to_text
    evaluator.is_sequential = is_sequential
    evaluator.domain_reference = path_reference

    # gather structure information
    candidates = gather_candidates(path_candidates)
    if not candidates:
        print(f"[WARN ] no ocr data (.*xml) in any dir starting from '{path_candidates}'! exit.")
        sys.exit(0)

    # match groundtruth
    for entry in candidates:
        gt = find_groundtruth(entry.path_c, path_candidates, path_reference)
        if gt:
            entry.path_g = gt

    # remove all paths where no groundtruth exists
    gt_entries = [c for c in candidates if c.path_g]
    n_entries = len(candidates)
    n_diff = n_entries - len(gt_entries)
    gt_missing = set(gt_entries) ^ set (candidates)
    rnd_str = f" ({gt_missing})" if gt_missing else ""
    if VERBOSITY >= 1:
        print(f'[DEBUG] from "{n_entries}" filtered "{n_diff}" candidates missing groundtruth{rnd_str}')

    # trigger actual evaluation
    evaluator.eval_all(gt_entries)

    # aggregate
    evaluator.aggregate(by_type=True)

    # evaluator.evaluate()
    evaluator.eval_map()

    # get results
    # results = evaluator.get_results()

    # serialize stdout report
    if VERBOSITY >= 0:
        report_stdout(evaluator, VERBOSITY)


def start():
    PARSER = argparse.ArgumentParser(description=f"""
        Evaluate Mass Digital Data. ({_get_info()})
        """)
    PARSER.add_argument(
        "candidates", 
        help="Root Directory to inspect"
        )
    PARSER.add_argument("-ref", "--reference", 
        required=False,
        help="Root Reference directory for Groundtruth or alike (optional)"
        )
    PARSER.add_argument("-v", "--VERBOSITY", 
        action='count', 
        default=DEFAULT_VERBOSITY,
        required=False, 
        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')"
        )
    PARSER.add_argument("--calc", 
        default=DEFAULT_CALCULCATION,
        required=False,
        help=f"Calculation to perform (optional; default: '{DEFAULT_CALCULCATION}'; available: '{','.join(CALC_DICT.keys())}')"
        )
    # metrics
    PARSER.add_argument("--metrics",
        default=DEFAULT_OCR_METRICS,
        required=False, 
        help=f"List of metrics to use (optional, default: '{DEFAULT_OCR_METRICS}'; available: '{','.join(METRIC_DICT.keys())}')"
        )
    PARSER.add_argument("--legacy", 
        action='store_true',
        required=False,
        help="legacy evaluation with naive rectangular geometry (optional; default: 'False')", 
        )
    PARSER.add_argument("--utf8",
        default=DEFAULT_UTF8_NORM,
        required=False,
        help=f"UTF-8 Unicode Python Normalization (optional; default: '{DEFAULT_UTF8_NORM}'; available: 'NFC','NFKC','NFD','NFKD')",
        )
    PARSER.add_argument("--sequential", 
        action='store_true',
        required=False,
        help="Execute calculations sequentially (optional; default: 'False')", 
        )
    PARSER.add_argument("-x", "--extra", 
        required=False, 
        help="pass additional information to evaluation, like 'ignore_geometry' (compare only text, ignore coords)"
        )
    PARSER.set_defaults(legacy=False)
    PARSER.set_defaults(sequential=False)

    ARGS = vars(PARSER.parse_args())
    path_candidates = ARGS["candidates"]
    path_reference = ARGS["reference"]
    global VERBOSITY
    VERBOSITY = ARGS["VERBOSITY"]
    IS_LEGACY = ARGS["legacy"]
    IS_SEQUENTIAL = ARGS["sequential"]
    xtra = ARGS["extra"]
    metrics = ARGS["metrics"]
    calc = ARGS["calc"]
    utf8norm = ARGS["utf8"]

    # go on
    # basic validation
    if not os.path.isdir(path_candidates):
        print(f'[ERROR] input "{path_candidates}": invalid directory! exit!')
        sys.exit(1)
    if path_reference and not os.path.isdir(path_reference):
        print(f'[ERROR] reference "{path_reference}": invalid directory! exit!')
        sys.exit(1)

    # sanitize trailing slash
    path_candidates = path_candidates[:-1] if path_candidates.endswith('/') else path_candidates
    path_reference = path_reference[:-1] if path_reference.endswith('/') else path_reference

    # if candidates and both reference provided: do domains match?
    if path_candidates and path_reference:
        _base_can = os.path.basename(path_candidates)
        _base_ref = os.path.basename(path_reference)
        if _base_can != _base_ref:
            print(f"[WARN ] start domains '{_base_can}' and '{_base_ref}' mismatch, summary might be inaccurate!")

    # some diagnostics
    if VERBOSITY >= 2:
        args = f"{path_candidates}, {path_reference}, {VERBOSITY}, {xtra}"
        print(f'[DEBUG] called with {args}')
    
    # here we go
    _main(path_candidates, path_reference, metrics, utf8norm, calc, xtra, is_legacy=IS_LEGACY, is_sequential=IS_SEQUENTIAL)


if __name__ == "__main__":
    start()
