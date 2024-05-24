# -*- coding: utf-8 -*-
"""OCR QA Evaluation CLI"""

import argparse
import os
import sys
import typing

import digital_eval as digev
import digital_eval.dictionary_metrics.common as digev_cm
import digital_eval.metrics as digem
from digital_eval.dictionary_metrics.language_tool.LanguageTool import LanguageTool


# script constants
DEFAULT_VERBOSITY = 0
VERBOSITY = DEFAULT_VERBOSITY
EVAL_VERBOSITY = DEFAULT_VERBOSITY
DEFAULT_UTF8_NORM = digev.UC_NORMALIZATION_DEFAULT

# metrics
DEFAULT_OCR_METRICS = 'Cs,Ls'
DEFAULT_OCR_METRIC_PREPROCESSINGS = ''
DEFAULT_OCR_METRIC_POSTPROCESSINGS = ''
METRIC_DICT = {
    'Cs': digev.MetricChars,
    'Characters': digev.MetricChars,
    'Ls': digev.MetricLetters,
    'Letters': digev.MetricLetters,
    'Ws': digev.MetricWords,
    'Words': digev.MetricWords,
    'BoWs': digev.MetricBoW,
    'BagOfWords': digev.MetricBoW,
    'IRPre': digev.MetricIRPre,
    'Pre': digev.MetricIRPre,
    'Precision': digev.MetricIRPre,
    'IRRec': digev.MetricIRRec,
    'Rec': digev.MetricIRRec,
    'IRFMeasure': digev.MetricIRFM,
    'FM': digev.MetricIRFM,
    'DictLT': digem.MetricDictionaryLangTool,
    'DictionaryLangTool': digem.MetricDictionaryLangTool,
}


def _initialize_metrics(
        the_metrics,
        norm,
) -> typing.List[digem.SimilarityMetric]:
    _tokens = the_metrics.split(',')
    try:
        metric_objects: typing.List[digem.SimilarityMetric] = []
        for m in _tokens:
            clazz: typing.Type[digem.SimilarityMetric] = METRIC_DICT[m]
            if 'Dict' in m:
                norm = digem.UC_NORMALIZATION_NFKD
            metric_inst: digem.SimilarityMetric = clazz(normalization=norm)
            metric_objects.append(metric_inst)
        return metric_objects
    except KeyError as _err:
        _keys = ','.join(METRIC_DICT.keys())
        _msg = f"Unknown: '{_err.args[0]}'.\nPlease use one of the following keys: '{_keys}'."
        print(_msg)
        sys.exit(1)


#########
# START #
#########
def start_evaluation(parse_args: typing.Dict):
    """Main workflow"""

    path_candidates = parse_args["candidates"]
    path_reference = parse_args["reference"]
    metrics: str = parse_args["metrics"]
    utf8norm = parse_args["utf8"]
    verbosity = parse_args["verbosity"]
    is_seq = parse_args["sequential"] if "sequential" in parse_args else False
    xtra = parse_args["extra"] if "extra" in parse_args else None

    if "language" in parse_args:
        digem.MetricDictionary.LANGUAGE = parse_args["language"]
    uses_lang_tool: bool = 'DictLT' in metrics or "DictionaryLangTool" in metrics
    if uses_lang_tool:
        lt_url: str = parse_args["lt_api_url"] if "lp_api_url" in parse_args else LanguageTool.DEFAULT_URL
        LanguageTool.initialize(lt_url)

    # go on with basic validation
    if not os.path.isdir(path_candidates):
        print(f'[ERROR] input "{path_candidates}": invalid directory! exit!')
        sys.exit(1)
    if path_reference and not os.path.isdir(path_reference):
        print(f'[ERROR] reference "{path_reference}": invalid directory! exit!')
        sys.exit(1)

    # sanitize trailing slash
    if not isinstance(path_candidates, str):
        path_candidates = str(path_candidates)
    if not isinstance(path_reference, str):
        path_reference = str(path_reference)
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
    evaluator = digev.Evaluator(
        path_candidates,
        verbosity=verbosity,
        extras=xtra,
    )
    evaluator.metrics = _initialize_metrics(metrics, norm=utf8norm)#, calc=calc)
    if verbosity >= 1:
        print(f"[DEBUG] text normalized using '{utf8norm}' code points for '{metrics}'")

    evaluator.is_sequential = is_seq
    evaluator.domain_reference = path_reference

    # gather structure information
    candidates = digev.gather_candidates(path_candidates)
    if not candidates:
        print(f"[WARN ] no ocr data (.*xml) in any dir starting from '{path_candidates}'! exit.")
        sys.exit(0)

    # match groundtruth
    for entry in candidates:
        gt = digev.find_groundtruth(entry.path_c, path_candidates, path_reference)
        if gt:
            entry.path_g = gt

    # remove all paths where no groundtruth exists
    gt_entries = [c for c in candidates if c.path_g]
    n_entries = len(candidates)
    n_diff = n_entries - len(gt_entries)
    gt_missing = set(gt_entries) ^ set(candidates)
    rnd_str = f" ({gt_missing})" if gt_missing else ""
    if verbosity >= 1:
        print(f'[DEBUG] from "{n_entries}" filtered "{n_diff}" candidates missing groundtruth{rnd_str}')

    # trigger actual evaluation
    evaluator.eval_all(gt_entries)

    # aggregate
    evaluator.aggregate(by_type=True)

    # evaluator.evaluate()
    evaluator.eval_map()

    # serialize stdout report
    if verbosity >= 0:
        digev.report_stdout(evaluator, verbosity)

    # for testing purposes
    eval_results =  evaluator.get_results()

    # final clean-up
    if uses_lang_tool:
        LanguageTool.deinitialize()

    return eval_results


def start():
    """Wrap argparsing"""
    parser = argparse.ArgumentParser(description=f"Evaluate Mass Digitalization Data {digev.__version__}")
    parser.add_argument("candidates",
                        help="Root Directory for evaluation candidates"
                        )
    parser.add_argument("-ref", "--reference",
                        required=False,
                        help="Root directory for Reference/Groundtruth data (optional, but necessary for most metrics)"
                        )
    parser.add_argument("-v", "--verbosity",
                        action='count',
                        default=DEFAULT_VERBOSITY,
                        required=False,
                        help=f"Verbosity flag. To increase, append multiple 'v's (optional; default: '{DEFAULT_VERBOSITY}')"
                        )
    parser.add_argument("--metrics",
                        default=DEFAULT_OCR_METRICS,
                        required=False,
                        help=f"List of metrics to use (optional, default: '{DEFAULT_OCR_METRICS}'; available: '{','.join(METRIC_DICT.keys())}')"
                        )
    parser.add_argument("--utf8",
                        default=DEFAULT_UTF8_NORM,
                        required=False,
                        help=f"UTF-8 Unicode Python Normalization (optional; default: '{DEFAULT_UTF8_NORM}'; available: 'NFC','NFKC','NFD','NFKD')",
                        )
    parser.add_argument("-s", "--sequential",
                        action='store_true',
                        required=False,
                        help="Execute calculations sequentially (optional; default: 'False')",
                        )
    parser.add_argument("-x", "--extra",
                        required=False,
                        help="pass additional information to evaluation, like 'ignore_geometry' (compare only text, ignore coords)"
                        )
    parser.add_argument('-l', "--language",
                        default=digev_cm.LANGUAGE_KEY_DEFAULT,
                        choices=digev_cm.LANGUAGE_KEYS,
                        required=False,
                        help=f"Language code for LanguagTool according to ISO 639-2 (optional; default: '{digev_cm.LANGUAGE_KEY_DEFAULT}')",
                        )
    parser.add_argument('-u', "--lt-api-url",
                        default=LanguageTool.DEFAULT_URL,
                        required=False,
                        help=f"Language Tool Api URL (optional; default: '{LanguageTool.DEFAULT_URL}')",
                        )
    main_args = vars(parser.parse_args())
    start_evaluation(main_args)


if __name__ == "__main__":
    start()
