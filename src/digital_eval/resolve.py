"""Search and find data pairs
candidates and corresponding reference/groundtruth
"""

import os
import re
import typing

import xml.etree.ElementTree as ET

from .evaluation import EvalEntry


_XML_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#',
          'pg2013': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}


def gather_candidates(start_path, file_ext='.xml') -> typing.List[EvalEntry]:
    """gather all files from start_path, by default
    XML-like (ALTO, PAGE)"""
    candidates = []
    if os.path.isdir(start_path):
        for curr_dir, _, files in os.walk(start_path):
            xml_files = [f for f in files if str(f).endswith(file_ext)]
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
    """Find correspondig groundtruth file for
    given candidate by domain_name
    """
    candidate_name = os.path.basename(path_candidate)
    candidate_dir = os.path.dirname(path_candidate)
    cand_path_segmts = candidate_dir.split(os.sep)
    candidate_root_dir = os.path.basename(root_candidates) if os.path.isdir(
        root_candidates) else os.path.dirname(root_candidates)
    _segm_cand = cand_path_segmts.pop()
    _segm_gt = [os.path.splitext(candidate_name)[0]]
    while candidate_root_dir != _segm_cand:
        _segm_gt.append(_segm_cand)
        _segm_cand = cand_path_segmts.pop()
    _segm_gt.reverse()
    _gt_path = str(os.sep).join(_segm_gt)
    groundtruth_filepath = os.path.join(root_groundtruth, _gt_path)
    groundtruth_filepath_parent = os.path.dirname(groundtruth_filepath)
    if os.path.exists(groundtruth_filepath_parent):
        path_groundtruth = _match_candidate(groundtruth_filepath)
        return path_groundtruth
    return None


def _match_candidate(path_gt_file_pattern):
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


def match_candidates(path_candidates, path_gt_file):
    '''Find candidates that match groundtruth'''

    if not os.path.isdir(path_candidates):
        raise IOError(f'invalid ocr result path "{path_candidates}"')
    if not os.path.exists(path_gt_file):
        raise IOError(f'invalid groundtruth data path "{path_gt_file}"')

    gt_filename = os.path.basename(path_gt_file)

    # 0: assume groundtruth is xml data
    cleared_name = ''
    if gt_filename.endswith('.xml'):
        # 1: get image name from metadata
        doc_root = ET.parse(path_gt_file).getroot()
        if 'alto' in doc_root.tag:
            filename_el = doc_root.find(
                './/alto:sourceImageInformation/alto:fileName', _XML_NS)
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
            path_candidates) if _names_match(cleared_name, f)]
        if matches:
            return [os.path.join(path_candidates, m) for m in matches]

    # 3: assume gt is textfile and name is contained in results data
    elif re.match(r'^[\d{5,}].*\.txt$', gt_filename):
        cleared_name = os.path.splitext(gt_filename)[0]
        matches = [f
                   for f in os.listdir(path_candidates)
                   if _names_match(cleared_name, f)]
        if matches:
            return [os.path.join(path_candidates, m)
                    for m in matches]
    return []


def _names_match(name_groundtruth, name_candidate):
    if '.gt' in name_groundtruth:
        name_groundtruth = name_groundtruth.replace('.gt', '')
    if name_groundtruth in name_candidate:
        candidate_ext = os.path.splitext(name_candidate)[1]
        if candidate_ext == '.txt' or candidate_ext == '.xml':
            return True

    return False
