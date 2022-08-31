# -*- coding: utf-8 -*-
"""OCR QA Evaluation CLI"""

import argparse
import os
import sys

from digital_eval import (
    find_groundtruth,
    gather_candidates,
    Evaluator,
    report_stdout,
)

DEFAULT_VERBOSITY = 0
EVAL_VERBOSITY = DEFAULT_VERBOSITY


def _main(path_candidates, path_reference, verbosity, xtra):

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
    if verbosity >= 2:
        args = f"{path_candidates}, {path_reference}, {verbosity}, {xtra}"
        print(f'[DEBUG] called with {args}')

    # create basic evaluator instance
    evaluator = Evaluator(path_candidates, verbosity, xtra)
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
    if verbosity >= 1:
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


########
# MAIN #
########
def main():
    PARSER = argparse.ArgumentParser(description="""
        Evaluate large amounts of Digital Data, 
        organized in directory structures.
        """)
    PARSER.add_argument(
                        "candidates", help="Root Directory to inspect")
    PARSER.add_argument("-ref", "--reference", required=False,
                        help="Root Reference directory for Groundtruth or alike (optional)")
    PARSER.add_argument("-v", "--verbosity", action='count', default=DEFAULT_VERBOSITY,
                        required=False, help="""
                        Verbosity. 
                        To increase, append multiple 'v's (optional, default: '')
                        """)
    PARSER.add_argument("-x", "--extra", required=False, 
                        help="""
                        pass additional information to evaluation, like
                        * 'ignore_geometry' 
                        compare only textual contents without respect to coords
                        """)

    ARGS = vars(PARSER.parse_args())
    path_candidates = ARGS["candidates"]
    path_ref = ARGS["reference"]
    verbosity = ARGS["verbosity"]
    xtra = ARGS["extra"]
    _main(path_candidates, path_ref, verbosity, xtra)


if __name__ == "__main__":
    main()
