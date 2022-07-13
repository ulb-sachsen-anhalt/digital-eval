# -*- coding: utf-8 -*-
"""OCR QA Evaluation CLI"""

import argparse
import sys

from datetime import date

from digital_eval import (
    find_groundtruth,
    gather_candidates,
    validate_paths,
    Evaluator,
)

DEFAULT_VERBOSITY = 0
EVAL_VERBOSITY = DEFAULT_VERBOSITY


def _main(pcandidates, preference, verbosity, xtra):
    (path_candidates, path_ref) = validate_paths(pcandidates, preference)

    if verbosity >= 2:
        args = f"{path_candidates}, {path_ref}, {verbosity}"
        print(f'[DEBUG] call with {args}')

    evaluator = Evaluator(path_ref, verbosity, xtra)

    # gather structure information
    candidates = gather_candidates(path_candidates)
    if not candidates:
        print(f"[WARN] no ocr data (.*xml) in any dir starting from '{path_candidates}' ! exit.")
        sys.exit(0)

    # match groundtruth
    for entry in candidates:
        gt = find_groundtruth(entry.path_c, path_candidates, path_ref)
        if gt:
            entry.path_g = gt

    # remove all paths where no groundtruth exists
    gt_entries = [c for c in candidates if c.path_g]
    n_entries = len(candidates)
    n_diff = n_entries - len(gt_entries)
    gt_missing = set(gt_entries) ^ set (candidates)
    rnd_str = f" ({gt_missing})" if gt_missing else ""
    print(f'[INFO ] from "{n_entries}" filtered "{n_diff}" candidates missing groundtruth{rnd_str}')

    # trigger actual evaluation
    evaluator.eval_all(gt_entries)

    # aggregate
    evaluator.aggregate(by_type=True)

    # evaluator.evaluate()
    evaluator.eval_map()

    # get results
    results = evaluator.get_results()
    evaluation_date = date.today().isoformat()

    print(f'[INFO ] Evaluation Summary for "{path_candidates}" vs. "{path_ref} ({evaluation_date})')
    for result in results:
        (gt_type, n_total, mean_total, med, _n_refs) = result.get_defaults()
        add_stats = f', std: {result.std:.2f}, median: {med:.2f}' if n_total > 1 else ''
        print(f'[INFO ] "{gt_type}"\t∅: {mean_total:.2f}\t{n_total} items, {_n_refs} refs{add_stats}')
        if result.cleared_result:
            (_, n_t2, mean2, med2, n_c2) = result.cleared_result.get_defaults()
            ccr_std = result.cleared_result.std
            drops = n_total - n_t2
            if drops > 0:
                print(f'[INFO ] "{gt_type}"\t∅: {mean2:.2f}\t{n_t2} items (-{drops}), {n_c2} refs, std: {ccr_std:.2f}, median: {med2:.2f}')


########
# MAIN #
########
def main():
    PARSER = argparse.ArgumentParser(description="Evaluate Digital Data")
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
