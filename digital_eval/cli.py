# -*- coding: utf-8 -*-
"""OCR QA Evaluation CLI"""

import argparse
import os
import sys

from typing import (
    List
)

from digital_eval import (
    find_groundtruth,
    gather_candidates,
    Evaluator,
    report_stdout,
    ocr_to_text,
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
EVAL_VERBOSITY = DEFAULT_VERBOSITY
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


def _initialize_metrics(the_metrics) -> List:
    _tokens = the_metrics.split(',')
    try:
        return [METRIC_DICT[m]() for m in _tokens]
    except KeyError as _err:
        _msg = f"Unknown Metric with label '{_err.args[0]}' requested.\nPlease use one of the following labels: '{','.join(METRIC_DICT.keys())}'."
        print(_msg)
        sys.exit(1)


########
# MAIN #
########
def main():

    # create basic evaluator instance
    evaluator = Evaluator(path_candidates, VERBOSITY, xtra)
    evaluator.metrics = _initialize_metrics(metrics)
    
    if IS_LEGACY:
        evaluator.to_text_func = ocr_to_text
    if IS_SEQUENTIAL:
        evaluator.IS_SEQUENTIAL = True
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
    report_stdout(evaluator)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="""
        Evaluate large amounts of Digital Data, 
        organized in directory structures.
        """)
    PARSER.add_argument(
        "candidates", 
        help="Root Directory to inspect")
    PARSER.add_argument("-ref", "--reference", 
        required=False,
        help="Root Reference directory for Groundtruth or alike (optional)")
    PARSER.add_argument("-cfg", "--configuration-file", 
        required=False,
        help="Path to configuration INI-file (optional)")
    PARSER.add_argument("-v", "--VERBOSITY", 
        action='count', 
        default=DEFAULT_VERBOSITY,
        required=False, help="""
            Verbosity. 
            To increase, append multiple 'v's (optional, default: '')
            """)
    _METRICS_KEYS = ','.join(METRIC_DICT.keys())
    PARSER.add_argument("-m", "--metrics",
        default=DEFAULT_OCR_METRICS,
        required=False, help=f"comma separated list of metrics, for example \"-m Ls,Ws\" for Letter and Words related metrics.\nDefaults: {DEFAULT_OCR_METRICS}.\nSee: {_METRICS_KEYS} for further information")
    PARSER.add_argument("--legacy", 
        help="enable legacy text functionality with custom, simple geometry", 
        action='store_true',
        required=False)
    PARSER.add_argument("--sequential", 
        help="execute calculation sequentially (optional, default: False)", 
        action='store_true',
        required=False)
    PARSER.add_argument("-x", "--extra", 
        required=False, 
        help="""
            pass additional information to evaluation, like
            * 'ignore_geometry' 
            compare only textual contents without respect to coords
            """)
    PARSER.set_defaults(legacy=False)
    PARSER.set_defaults(sequential=False)

    ARGS = vars(PARSER.parse_args())
    path_candidates = ARGS["candidates"]
    path_reference = ARGS["reference"]
    PATH_CONFIG = ARGS["configuration_file"]
    VERBOSITY = ARGS["VERBOSITY"]
    IS_LEGACY = ARGS["legacy"]
    IS_SEQUENTIAL = ARGS["sequential"]
    xtra = ARGS["extra"]
    metrics = ARGS["metrics"]

    # go on
    # basic validation
    if not os.path.isdir(path_candidates):
        print(f'[ERROR] input "{path_candidates}": invalid directory! exit!')
        sys.exit(1)
    if path_reference and not os.path.isdir(path_reference):
        print(f'[ERROR] reference "{path_reference}": invalid directory! exit!')
        sys.exit(1)
    if PATH_CONFIG and not os.path.isfile(PATH_CONFIG):
        print(f'[ERROR] configuration file invalid: "{PATH_CONFIG}"! exit!')
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
    main()
