# -*- coding: utf-8 -*-
"""OCR Evaluation Module"""
from __future__ import annotations

import concurrent.futures
import copy
import os
import re
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET
from datetime import (
    date
)
from math import (
    floor
)
from multiprocessing import (
    cpu_count
)
from pathlib import (
    Path
)
from typing import (
    List,
    Tuple,
)

import numpy as np

from digital_object import ( 
	DigitalObjectTree, 
	DigitalObjectLevel, 
	to_digital_object,
)

PAGE_2013 = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
XML_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#',
          'pg2013': PAGE_2013}

# just use textual information for evaluation
# do *not* respect any geometrics
EVAL_EXTRA_IGNORE_GEOMETRY = 'ignore_geometry'

# mark unset values as 'not available'
NOT_SET = 'n.a.'

# how long evaluation shall take maximal
# where "None" means "no timeout"
EVAL_TIMEOUT = None


def strip_outliers_from(data_tuples, fence_ratio=1.5):
    """Determine a data set's outliers by interquartile range (IQR)
    
    calculate data points 
     * below median of quartile 1 (lower fence), and
     * above median of quartile 3 (upper fence)
    """

    data_points = [e[1] for e in data_tuples]
    median = np.median(data_points)
    Q1 = np.median([v for v in data_points if v < median])
    Q3 = np.median([v for v in data_points if v > median])
    regulars = [data
                for data in data_tuples
                if data[1] >= (Q1 - fence_ratio * (Q3 - Q1)) and data[1] <= (Q1 + fence_ratio * (Q3 - Q1))]
    return (regulars, Q1, Q3)


def get_statistics(data_points):
    """Get common statistics like mean, median and std for data_points"""

    the_mean = np.mean(data_points)
    the_deviation = np.std(data_points)
    the_median = np.median(data_points)
    return (the_mean, the_deviation, the_median)


def gather_candidates(start_path) -> List:
    candidates = []
    if os.path.isdir(start_path):
        for curr_dir, _, files in os.walk(start_path):
            xml_files = [f for f in files if str(f).endswith('.xml')]
            if xml_files:
                for xml_file in xml_files:
                    rel_path = os.path.join(curr_dir, xml_file)
                    entry = (EvalEntry(os.path.abspath(rel_path)))
                    candidates.append(entry)
    else:
        candidates.append(EvalEntry(start_path))

    candidates.sort(key=lambda e: e.path_c)
    return candidates


def find_groundtruth(path_candidate, root_candidates, root_groundtruth):
    file_name = os.path.basename(path_candidate)
    file_dir = os.path.dirname(path_candidate)
    path_segmts = file_dir.split(os.sep)
    candidate_root_dir = os.path.basename(root_candidates) if os.path.isdir(
        root_candidates) else os.path.dirname(root_candidates)
    _segm_cand = path_segmts.pop()
    _segm_gt = [os.path.splitext(file_name)[0]]
    while candidate_root_dir != _segm_cand:
        _segm_gt.append(_segm_cand)
        _segm_cand = path_segmts.pop()
    _segm_gt.reverse()
    _gt_path = str(os.sep).join(_segm_gt)
    groundtruth_filepath = os.path.join(root_groundtruth, _gt_path)
    groundtruth_filepath_parent = os.path.dirname(groundtruth_filepath)
    if os.path.exists(groundtruth_filepath_parent):
        path_groundtruth = match_candidate(groundtruth_filepath)
        return path_groundtruth


def match_candidates(path_candidates, path_gt_file):
    '''Find candidates that match groundtruth'''

    if not os.path.isdir(path_candidates):
        raise IOError('invalid ocr result path "{}"'.format(path_candidates))
    if not os.path.exists(path_gt_file):
        raise IOError(
            'invalid groundtruth data path "{}"'.format(path_gt_file))

    gt_filename = os.path.basename(path_gt_file)

    # 0: assume groundtruth is xml data
    cleared_name = ''
    if gt_filename.endswith('.xml'):
        # 1: get image name from metadata
        doc_root = ET.parse(path_gt_file).getroot()
        if 'alto' in doc_root.tag:
            filename_el = doc_root.find(
                './/alto:sourceImageInformation/alto:fileName', XML_NS)
            if filename_el is not None:
                filename_text = filename_el.text
                if filename_text:
                    cleared_name = os.path.splitext(filename_text.strip())[0]

        # 2: 2nd try: calculate cleared_name by matching 1st 6 chars as digits from file_name
        if cleared_name == '' and re.match(r'^[\d{6,}].*', gt_filename):
            file_name_tokens = gt_filename.split("_")
            tokens = []
            if len(file_name_tokens) > 4:
                tokens = file_name_tokens[:4]
            elif len(file_name_tokens) == 4:
                tokens = file_name_tokens[:3]
                if ".xml" in file_name_tokens[3]:
                    last_token = file_name_tokens[3].split('.')[0]
                    tokens = tokens + [last_token]
            cleared_name = "_".join(tokens)

        matches = [f for f in os.listdir(
            path_candidates) if names_match(cleared_name, f)]
        if matches:
            return [os.path.join(path_candidates, m) for m in matches]

    # 3: assume gt is textfile and name is contained in results data
    elif re.match(r'^[\d{5,}].*\.txt$', gt_filename):
        cleared_name = os.path.splitext(gt_filename)[0]
        matches = [f
                   for f in os.listdir(path_candidates)
                   if names_match(cleared_name, f)]
        if matches:
            return [os.path.join(path_candidates, m)
                    for m in matches]

    return []


def match_candidate(path_gt_file_pattern):
    '''Find candidates that match groundtruth'''

    gt_filename = os.path.basename(path_gt_file_pattern)

    # 1: assume groundtruth is straight name like xml data
    gt_path_xml = path_gt_file_pattern + '.xml'
    if os.path.exists(gt_path_xml):
        return gt_path_xml

    # inspect all files in given directory if it fits anyway
    # assume groundtruth starts with same tokens
    gt_dir = os.path.dirname(path_gt_file_pattern)
    gt_files = [f
                for f in os.listdir(gt_dir)
                if f.endswith(".xml") or f.endswith(".txt")]
    for _file in gt_files:
        if _file.startswith(gt_filename):
            return os.path.join(gt_dir, _file)


def names_match(name_groundtruth, name_candidate):
    if '.gt' in name_groundtruth:
        name_groundtruth = name_groundtruth.replace('.gt', '')
    if name_groundtruth in name_candidate:
        candidate_ext = os.path.splitext(name_candidate)[1]
        if candidate_ext == '.txt' or candidate_ext == '.xml':
            return True

    return False


def get_bbox_data(file_path):
    '''Get Bounding Box Data from given resource, if any exists'''

    if not os.path.exists(file_path):
        raise IOError('{} not existing!'.format(file_path))

    # 1: inspect filename
    file_name = os.path.basename(file_path)
    result = re.match(r'.*_(\d{2,})x(\d{2,})_(\d{2,})x(\d{2,})', file_name)
    if result:
        groups = result.groups()
        x0 = int(groups[0])
        x1 = int(groups[2])
        y1 = int(groups[3])
        y0 = int(groups[1])
        return ((x0, y0), (x1, y1))

    with open(file_path, encoding='utf-8') as _handle:
        # rather brute force approach
        # to recognize OCR formats inside
        start_token = _handle.read(128)
        _frame_points = None

        # switch by estimated ocr format
        if 'alto' in start_token:
            # legacy: read from custom ALTO meta data
            root_element = ET.parse(file_path).getroot()
            element = root_element.find(
                './/alto:Tags/alto:OtherTag[@ID="ulb_groundtruth_points"]', XML_NS)
            if element is not None:
                points = element.attrib['VALUE'].split(' ')
                _p1 = points[0].split(',')
                p1 = (int(_p1[0]), int(_p1[1]))
                _p2 = points[2].split(',')
                p2 = (int(_p2[0]), int(_p2[1]))
                return (p1, p2)

            # read from given alto coordinates
            raw_elements = root_element.findall('.//alto:String', XML_NS)
            non_empty = [s for s in raw_elements if s.attrib['CONTENT'].strip(
            ) and re.match(r'[^\d]', s.attrib['CONTENT'])]
            return calculate_bounding_box(non_empty, _map_alto)

        elif 'PcGts' in start_token:
            # read from given page coordinates
            doc_root = xml.dom.minidom.parse(file_path).documentElement
            name_space = doc_root.namespaceURI
            root_element = ET.parse(file_path).getroot()
            # step one: read PAGE border coords
            _xpr_page_borders = f'{{{name_space}}}Page/{{{name_space}}}Border/{{{name_space}}}Coords'
            _page_coords = root_element.findall(_xpr_page_borders)
            if len(_page_coords) > 0:
                _frame_points = calculate_bounding_box(_page_coords, _map_page2013)
            # step two: if possible, go for sub-part geometry
            _xpr_line_coords = f'.//{{{name_space}}}TextLine/{{{name_space}}}Coords'
            _line_coords = root_element.findall(_xpr_line_coords)
            if len(_line_coords) > 0:
                _frame_points = calculate_bounding_box(_line_coords, _map_page2013)
            if _frame_points:
                return _frame_points
            else:
                raise RuntimeError(f"{file_path} missing page/line coords!")
    return None


def _map_alto(e: ET.Element) -> Tuple[str, int, int, int, int]:
    i = e.attrib['ID']
    x0 = int(e.attrib['HPOS'])
    y0 = int(e.attrib['VPOS'])
    x1 = x0 + int(e.attrib['WIDTH'])
    y1 = y0 + int(e.attrib['HEIGHT'])
    return (i, x0, y0, x1, y1)


def _map_page2013(elem: ET.Element) -> Tuple[str, int, int, int, int]:
    points = elem.attrib['points'].strip().split(' ')
    _xs = [int(p.split(',')[0]) for p in points]
    _ys = [int(p.split(',')[1]) for p in points]
    return (NOT_SET, min(_xs), min(_ys), max(_xs), max(_ys))


def _get_line_digos_from_digo(digo: DigitalObjectTree, lines: List[DigitalObjectTree] = None) -> List[DigitalObjectTree]:
    if lines is None:
        lines = []
    if digo.level == DigitalObjectLevel.LINE and digo.transcription:
        lines.append(digo)
        return lines
    for child in digo.children:
        _get_line_digos_from_digo(child, lines)
    return lines


def calculate_bounding_box(elements: List[ET.Element], map_func) -> Tuple[int, int, int, int]:
    """Review element's points to get points for
    minimum (top-left) and maximum (bottom-right)"""

    all_points = [map_func(e) for e in elements]
    all_x1 = [p[1] for p in all_points]
    all_y1 = [p[2] for p in all_points]
    all_x2 = [p[3] for p in all_points]
    all_y2 = [p[4] for p in all_points]
    return ((min(all_x1), min(all_y1)), (max(all_x2), max(all_y2)))


def digital_object_to_text(file_path, frame=None, oneliner=True) -> Tuple[str | List[str], int]:
    """Wrap OCR-Data Comparison"""

    try:
        top_digo: DigitalObjectTree = to_digital_object(file_path)
        # explicit filter frame?
        if not frame:
            frame = top_digo.dimensions
        elif len(frame) == 2:
            frame = [[frame[0][0], frame[0][1]],
                     [frame[1][0], frame[0][1]],
                     [frame[1][0], frame[1][1]],
                     [frame[0][0], frame[1][1]]]
        frame_digo = DigitalObjectTree()
        frame_digo.dimensions = frame
        filter_word_pieces(frame_digo, top_digo)
        the_lines = _get_line_digos_from_digo(top_digo)
        if oneliner:
            return top_digo.transcription, len(the_lines)
        else:
            return [line.transcription for line in the_lines], len(the_lines)
    except xml.parsers.expat.ExpatError as _:
        with open(file_path, mode='r', encoding='utf-8') as fhandle:
            text_lines = fhandle.readlines()
            if oneliner:
                text_lines = ' '.join([l.strip() for l in text_lines])
            return text_lines, len(text_lines)
    except RuntimeError as exc:
        raise RuntimeError(f"{file_path}: {exc}") from exc


def digital_object_to_dict_text(file_path: str, frame=None, oneliner=False) -> Tuple[str | List[str], int]:
    line_texts: List[str]
    len_lines: int
    line_texts, len_lines = digital_object_to_text(file_path=file_path, frame=frame, oneliner=False)
    non_empty_lines: List[str] = [line_text for line_text in line_texts if len(line_text) > 0]
    lines_sanitized_wraps: List[str] = _sanitize_wraps(non_empty_lines)
    lines_sanitized_chars: List[str] = _sanitize_chars(lines_sanitized_wraps)
    text = ' '.join(lines_sanitized_chars) if oneliner else lines_sanitized_chars
    return text, len_lines


_HYPHENS: List[str] = [
    "⸗",
    "-",
    "—",
]


def _sanitize_wraps(lines: List[str]) -> List[str]:
    """Sanitize word wraps if
    * last word token ends with '-', "⸗" or "—"
    * another line following
    * following line not empty
    """

    normalized_lines: List[str] = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            for hyphen in _HYPHENS:
                if line.endswith(hyphen):
                    next_line = lines[i + 1]
                    if len(next_line.strip()) == 0:
                        # encountered empty next line, no merge possible
                        continue
                    next_line_tokens = next_line.split()
                    nextline_first_token = next_line_tokens.pop(0)
                    # join the rest of valid next line
                    lines[i + 1] = ' '.join(next_line_tokens)
                    line = line[:-1] + nextline_first_token
                    break
        normalized_lines.append(line)
    return normalized_lines


def _sanitize_chars(lines: List[str]) -> List[str]:
    """Replace or remove nonrelevant chars for current german word error rate"""

    sanitized: List[str] = []
    for line in lines:
        text = line.strip()
        bad_chars = '0123456789“„"\'?!*.;:-=[]()|'
        text = ''.join([c for c in text if c not in bad_chars])
        if '..' in text:
            text = text.replace('..', '')
        if '  ' in text:
            text = text.replace('  ', ' ')
        text = ' '.join([t for t in text.split() if len(t) > 1])
        sanitized.append(text)

    return sanitized


def _get_groundtruth_from_filename(file_path) -> str:
    _file_name = os.path.basename(file_path)
    result = re.match(r'.*gt.(\w{3,}).xml$', _file_name)
    if result:
        return result[1]
    else:
        alternative = re.match(r'.*\.(\w{3,})\.gt\.xml$', _file_name)
        if alternative:
            return alternative[1]
        else:
            return NOT_SET


def filter_word_pieces(frame, current) -> int:
    _filtered = 0
    _tmp_stack = []
    _total_stack = []
    # stack all items
    _total_stack.append(current)
    _tmp_stack.append(current)
    while _tmp_stack:
        _current: DigitalObjectTree = _tmp_stack.pop()
        if _current.children:
            _tmp_stack += _current.children
            _total_stack += _current.children
    # now pick words
    _words = [_p for _p in _total_stack if _p.level == DigitalObjectLevel.WORD]

    # check for each word piece
    for _word in _words:
        if _word not in frame:
            _filtered += 1
            _uplete(_word)
    return _filtered


def _uplete(curr: DigitalObjectTree):
    if len(curr.children) == 0 and curr.level < DigitalObjectLevel.PAGE:
        _pa: DigitalObjectTree = curr.parent
        _pa.remove_children(curr)
        _uplete(_pa)


def _normalize_gt_type(label) -> str:
    if label.startswith('art'):
        return 'article'
    elif label.startswith('ann'):
        return 'announcement'
    else:
        return NOT_SET


class EvaluationResult:
    '''
    Wrap statistical information
    for groundtruth Evaluation
    regarding a specific set
    = a directory, which's name serves as eval_key

    optional:
        enclose EvaluationResult with outliers removed
    '''

    def __init__(self, eval_key: str, n_total: int = 1, n_chars=0, n_lines=0):
        self.eval_key = eval_key
        self.total_mean = 0.0
        self.n_total = n_total
        self.n_outlier = 0
        self.n_chars = n_chars
        self.n_lines = n_lines
        self.mean = 0.0
        self.std = 0.0
        self.median = 0.0
        # set special descendant from same type
        # to hold optional metrics regarding
        # removed outliers
        self.cleared_result = None

    def get_defaults(self):
        """Provide default data (eval_key, number of elements, mean) that must be available"""

        return self.eval_key, self.n_total, self.mean, self.median, self.n_chars


class EvalEntry:
    """Container to transform evaluation results into
    string representation"""

    def __init__(self, path):
        self.path_c = path
        self.path_g = None
        self.gt_type = NOT_SET
        self.metrics = []

    def __str__(self) -> str:
        """Dependency between metrics 
        * 0=CA => 1=LA 
        * 2=WA => 3=BOT
        """
        _pres = [0, 2]
        _accs = [1, 3]
        _raws = []
        _pre_v = None
        for i, m in enumerate(self.metrics):
            _val = m.value
            _ref = m.n_ref
            if _ref > 10000:
                _ref_fmt = f'{(floor(float(m.n_ref) / 1000)):>2}K+'
            else:
                _ref_fmt = f'{m.n_ref:>4}'
            _raw = f'{m.label}:{_val:>5.2f}({_ref_fmt})'
            if i in _pres:
                _pre_v = _val
            if i in _accs and _pre_v is not None:
                diff = round(_val, 3) - round(_pre_v, 3)
                _raw += f'(+{diff:>5.2f})' if diff > 0 else f'(-{abs(diff):>5.2f})'
                _pre_v = None
            _raws.append(_raw)
        return ', '.join(_raws)

    def __repr__(self) -> str:
        return '{} {}'.format(self.gt_type, self.path_c)


class Evaluator:
    """Wrapper for Evaluation given candidates versus reference data

    Raises:
        RuntimeError: if candidates or reference data missing
    """

    def __init__(
            self,
            root_candidates,
            verbosity=0,
            extras=None,

    ):
        """initiate new Evaluator

        Args:
            root_candidates (string|Path): Root domain/path to search for candidates
            verbosity (int, optional): Level of verbosity Defaults to 0.
            extras (_type_, optional): Implementation dependend. Defaults to None.
        """
        self.domain_candidate = root_candidates
        self.domain_reference = None
        self.evaluation_entries = []
        self.verbosity = verbosity
        self.evaluation_data = {}
        self.evaluation_results = []
        self.evaluation_map = {}
        self.text_mode = extras == EVAL_EXTRA_IGNORE_GEOMETRY
        self.is_sequential = False
        self.metrics: List = []
        self.evaluation_report = {}

    def eval_all(self, entries: List[EvalEntry], sequential=False) -> None:
        """evaluate all pairs groundtruth-candidate"""

        _entries = []
        if sequential or self.is_sequential:
            _entries = [self._wrap_eval_entry(e) for e in entries]
        else:
            cpus = cpu_count()
            n_executors = cpus - 1 if cpus > 3 else 1
            if self.verbosity == 1:
                print(f"[DEBUG] use {n_executors} executors ({cpus}) to create evaluation data")
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_executors) as executor:
                try:
                    _entries = list(
                        executor.map(self._wrap_eval_entry, entries, timeout=EVAL_TIMEOUT))
                except concurrent.futures.TimeoutError:
                    print(f"[ERROR] takes longer than {EVAL_TIMEOUT}s to evaluate {len(entries)} entries!")
                    sys.exit(1)
                except Exception as err:
                    print(f"[ERROR] '{err}' creating evaluation data!")
                    sys.exit(1)

        # review evaluation results
        if _entries:
            _not_nones = [e for e in _entries if e is not None]
            if self.verbosity == 1:
                print(f"[DEBUG] processed {len(_entries)}, omitted {len(_entries) - len(_not_nones)} empty results")
            self.evaluation_entries = _not_nones

        self.evaluation_entries = sorted(self.evaluation_entries, key=lambda e: e.path_c)
        # detail report
        self.evaluation_report['candidates'] = [self._generate_report_candidate(e)
                                                for e in self.evaluation_entries]

    def _wrap_eval_entry(self, entry: EvalEntry):
        """Wrapper for creation of evaluation data
        to be used in common process-pooling"""

        if entry.path_g:
            try:
                return self.eval_entry(entry)
            except Exception as exc:
                print(f"[WARN ][{entry.path_g}] _wrap {exc}")

    def eval_entry(self, entry: EvalEntry) -> EvalEntry:
        """Create evaluation entry for matching pair of 
        groundtruth and candidate data"""

        # evaluate metric copies
        _current_metrics = []

        for _m in self.metrics:

            path_g = entry.path_g
            path_c = entry.path_c

            # read coordinate information (if any provided)
            # to create frame for candidate data
            coords = get_bbox_data(path_g)
            if coords is not None and self.verbosity >= 2:
                print(f"[TRACE] token coordinates {coords[0]}, {coords[1]}")

            to_text_func = _m.to_text_func

            # load ground-thruth text
            (txt_gt, _) = to_text_func(path_g, oneliner=True)

            if not txt_gt:
                print(f"[WARN ] groundtrooth '{path_g}' contains no text")

            # if text mode is enforced
            # forget groundtruth coordinates
            coords = None if self.text_mode else coords

            # read candidate data as text
            (txt_c, _) = to_text_func(path_c, coords, oneliner=True)

            if not txt_c:
                print(f"[WARN ] candidate '{path_c}' contains no text")

            if self.verbosity >= 2:
                _label_ref = os.path.basename(path_g)
                _label_can = os.path.basename(path_c)
                print(f'[TRACE][{_label_ref}] RAW GROUNDTRUTH :: "{txt_gt}"')
                print(f'[TRACE][{_label_can}] RAW CANDIDATE   :: "{txt_c}"')

            _curr = copy.copy(_m)
            _curr.reference = txt_gt
            _curr.candidate = txt_c
            # ATTENZIONE! inital access to this attribute
            # triggers preprocessing and calculation!
            _curr.value
            _current_metrics.append(_curr)
            if self.verbosity >= 2:
                _label_ref = os.path.basename(path_g)
                _label_can = os.path.basename(path_c)
                print(f'[TRACE][{_label_ref}][{_curr.label}] REFERENCE :: "{_curr._data_reference}"')
                print(f'[TRACE][{_label_can}][{_curr.label}] CANDIDATE :: "{_curr._data_candidate}"')

        # enrich entry with metrics and
        # normalize data type (i.e., art or ann or ...)
        _normed_gt_type = _normalize_gt_type(_get_groundtruth_from_filename(entry.path_g))
        entry.gt_type = _normed_gt_type
        entry.metrics = _current_metrics
        return entry

    def _generate_report_candidate(self, the_entry):
        try:
            image_name = os.path.basename(the_entry.path_c)
            _type = the_entry.gt_type
            if '+' in image_name and '_' in image_name:
                _tkns = image_name.split('_')
                image_name = _tkns[0].replace('+', ':') + '_' + _tkns[1]
            if '.xml' in image_name:
                image_name = image_name.replace('.xml', '')
            gt_label = f"({_type[:3]})" if _type and _type != NOT_SET else ''
            return f'[{image_name}]{gt_label} [{the_entry}]'
        except Exception as exc:
            print(f'[WARN ] {exc}')

    def _add(self, evaluation_result: EvaluationResult):
        self.evaluation_results.append(evaluation_result)

    def eval_map(self):
        for k, data_tuples in self.evaluation_map.items():
            n_total = len(data_tuples)
            data_points = [e[1] for e in data_tuples]
            n_chars = sum([e[2] for e in data_tuples])

            # set initial result level values
            evaluation_result = EvaluationResult(k, n_total, n_chars=n_chars)
            evaluation_result.mean = data_points[0]
            evaluation_result.median = data_points[0]

            # if more than one single evaluation item
            # calculate additional statistics to reflect
            # impact of outlying data sets
            # take CA and number of GT into account
            # also calculate statistics (mean, std)
            if len(data_points) > 1:
                (mean, std, median) = get_statistics(data_points)
                evaluation_result.mean = mean
                evaluation_result.median = median
                evaluation_result.std = std
                if std >= 1.0:
                    (regulars, _, _) = strip_outliers_from(data_tuples)
                    regulars_data_points = [e[1] for e in regulars]
                    clear_result = EvaluationResult(k, len(regulars))
                    (mean2, std2, med2) = get_statistics(regulars_data_points)
                    clear_result.mean = mean2
                    clear_result.std = std2
                    clear_result.median = med2
                    clear_result.n_chars = sum([e[2] for e in regulars])
                    # set as child component
                    evaluation_result.cleared_result = clear_result
            self._add(evaluation_result)
            # re-order
            self.evaluation_results = sorted(self.evaluation_results, key=lambda e: e.eval_key)

    def aggregate(self, by_type=False, by_metrics=[0, 1, 2, 3]):

        # precheck - having root dir
        self._check_aggregate_preconditions()

        root_base = Path(self.domain_reference).parts[-1]

        # aggregate on each directory
        for _metrics_index in by_metrics:
            for ee in self.evaluation_entries:
                # if we do not have all these different metrics set,
                # do of course not aggregate by non-existing index!
                if _metrics_index >= len(self.evaluation_entries[0].metrics):
                    continue
                path_key = f"{ee.metrics[_metrics_index].label}@{root_base}"
                # ATTENZIONE! works only when forehand
                # the *real* attribute has been accessed
                # *at least one time*
                # kept this way for testing reasons
                metric_value = ee.metrics[_metrics_index].value
                metric_gt_refs = ee.metrics[_metrics_index].n_ref
                dir_o = os.path.dirname(ee.path_c)
                ocr_parts = Path(dir_o).parts
                if root_base in ocr_parts:
                    tokens = list(ocr_parts[ocr_parts.index(root_base):])
                    if tokens:
                        # store at top-level
                        if path_key not in self.evaluation_map:
                            self.evaluation_map[path_key] = []
                        self.evaluation_map[path_key].append((ee.path_c, metric_value, metric_gt_refs))
                        # if by_type, aggregate type at top level
                        if by_type and ee.gt_type and ee.gt_type != NOT_SET:
                            type_key = path_key + '@' + ee.gt_type
                            if type_key not in self.evaluation_map:
                                self.evaluation_map[type_key] = []
                            self.evaluation_map[type_key].append((ee.path_c, metric_value, metric_gt_refs))
                        tokens.pop(0)
                        # store at any sub-level
                        curr = path_key
                        while tokens:
                            token = tokens.pop(0)
                            curr = curr + os.sep + token
                            if curr not in self.evaluation_map:
                                self.evaluation_map[curr] = []
                            self.evaluation_map[curr].append((ee.path_c, metric_value, metric_gt_refs))

    def _check_aggregate_preconditions(self):
        if not self.evaluation_entries:
            raise RuntimeError("missing evaluation data")
        if not Path(self.domain_candidate).is_dir():
            raise RuntimeError("no candidate root dir to aggregate data from")

    def get_results(self):
        return self.evaluation_results


def report_stdout(evaluator: Evaluator, verbosity):
    """Generate report data on stdout"""

    if verbosity >= 1:
        if 'candidates' in evaluator.evaluation_report:
            for _c in evaluator.evaluation_report['candidates']:
                print(f'[DEBUG] {_c}')
    results = evaluator.get_results()
    _path_can = evaluator.domain_candidate
    _path_ref = evaluator.domain_reference
    evaluation_date = date.today().isoformat()
    print(f'[INFO ] Evaluation Summary (candidates: "{_path_can}" vs. reference: "{_path_ref}" ({evaluation_date})')
    for result in results:
        (gt_type, n_total, mean_total, med, _n_refs) = result.get_defaults()
        add_stats = f', std: {result.std:.2f}, median: {med:.2f}' if n_total > 1 else ''
        print(f'[INFO ] "{gt_type}"\t∅: {mean_total:.2f}\t{n_total} items, {_n_refs} refs{add_stats}')
        if result.cleared_result:
            (_, n_t2, mean2, med2, n_c2) = result.cleared_result.get_defaults()
            ccr_std = result.cleared_result.std
            drops = n_total - n_t2
            if drops > 0:
                print(
                    f'[INFO ] "{gt_type}"\t∅: {mean2:.2f}\t{n_t2} items (-{drops}), {n_c2} refs, std: {ccr_std:.2f}, median: {med2:.2f}')
