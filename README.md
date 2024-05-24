# digital eval

![example workflow](https://github.com/ulb-sachsen-anhalt/digital-eval/actions/workflows/python-app.yml/badge.svg)
[![PyPi version](https://badgen.net/pypi/v/digital-eval/)](https://pypi.org/project/digital-eval) ![PyPI - Downloads](https://img.shields.io/pypi/dm/digital-eval) ![PyPI - License](https://img.shields.io/pypi/l/digital-eval) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/digital-eval)

Python3 Tool to report evaluation outcomes from mass digitalization workflows.

## Features

* [OCR-D compliant](https://ocr-d.de/en/spec/ocrd_eval#character-error-rate-cer) normalized similarity for edit-distance based metrics based on characters, letters and words
* choose from textual metrics based on characters or words plus common Information Retrieval
* choose from different UTF-8 Python norms
* match groundtruth (i.e. reference data) and candidates by filename start
* use geometric information to evaluate only specific frame (i.e. specific column or region from large page) of
  candidates (requires ALTO or PAGE format)
* aggregate evaluation outcomes on domain range (with multiple subdomains) according to folder layout
* formats: ALTO, PAGE or plain text for both groundtruth and candidates
* speedup with parallel execution
* additional OCR util:
  * filter custom areas of single OCR files of ALTO files

## Installation

```bash
pip install digital-eval
```

## Usage

### Metrics

#### Edit-Distance similarity

Calculate string similarity for each single reference/groundtruth and test/candidate item.
Complete haracter-based text string (`Cs`, `Characters`) or Letter-based (`Ls`, `Letters`) minus whitespaces,
punctuation and common digits (arabic, persian). 
Word/Token-based edit-distance of single tokens identified by markup elements or whitespaces, depending on data.

#### Set based

Calculate union of sets of tokens/words (`BoW`, `BagOfWords`).
Operate on sets of tokens/words with respect to language specific stopwords using [nltk](https://www.nltk.org/)
-framework for:

* Precision (`IRPre`, `Pre`, `Precision`): How many tokens from candidate are in groundtruth reference?
* Recall (`IRRec`, `Rec`, `Recall`): How many tokens from groundtruth reference should candidate include?
* F-Measure (`IRFMeasure`, `FM`): weighted ratio Precision / Recall

### UTF-8 Normalization

Use standard Python Implementation of UTF-8 normalizations; default: `NFC` (cf.:[OCR-D spec](https://ocr-d.de/en/spec/ocrd_eval#unicode-normalization)).

### Statistics

Statistics calculated by [numpy](https://numpy.org/) include arithmetic mean, median. Additionally includes an outlier detection with
interquartile range and are shown in relation to the total amount of specific groundtruth/reference (ref) for each metric, i.e. char, letters or tokens/words.

### Evaluate treelike structures

To evaluate OCR-candidate-data batch-like versus existing groundtruth, please make sure that your structures fit this layout:

```bash
groundtruth root/
├── <domain>/ 
│    └── <subdomain>/
│         └── <page-01>.gt.xml
candidate root/
├── <domain>/ 
│    └── <subdomain>/
│         └── <page-01>.xml
```

Now call:

```bash
digital-eval <path-candidate-root>/domain/ -ref <path-groundtruth>/domain/
```

for an aggregated overview on stdout. Increase verbosity with `-v` (or even `-vv`) to get detailed
information about each single data item evaluated.

Structured OCR is considered to contain valid geometrical and textual data on word level, even though for recent PAGE also line level is possible.

### Data problems

Inconsistent OCR Groundtruth with empty texts (ALTO String elements missing CONTENT or PAGE without TextEquiv) or invalid geometrical coordinates (less than 3 points or even empty) will lead to evaluation errors if geometry must be respected.

Erroneous data files will be reported and excluded from evaluation.

## Additional OCR Utils

### Filter Area (ALTO)

You can filter a custom area of a page of an ALTO file by providing the points of an arbitrary shape.
The format of the `-p, --points` argument is `<pt_1_x>,<pt_1_y> <pt_2_x>,<pt_2_y> <pt_3_x>,<pt_3_y> ... <pt_n_x>,<pt_n_y>` . For simple rectangular areas this can be expressed also with two points, with first point as top left and second point as bottom right: `<pt_top_left_x>,<pt_top_left_y> <pt_bottom_right_x>,<pt_bottom_right_y>`.

The following example filters a rectangular area of 600x400 pixels of a page, which is described by an input ALTO file and saves the result to an output ALTO file

```bash
ocr-util frame -i page_1.alto.xml -p "0,0 600,0 600,400 0,400" -o page_1_area.alto.xml
```

For plain rectangles exists a short form with only two points, top left and bottom right:

```bash
ocr-util frame -i page_1.alto.xml -p "0,0 600,400" -o page_1_area.alto.xml
```

## Development

Plattform: Intel(R) Core(TM) i5-6500 CPU@3.20GHz, 16GB RAM, Ubuntu 22.04 LTS, Python 3.8+

```bash
# clone local
git clone <repository-url> <local-dir>
cd <local-dir>

# enable virtual python 3 environment (linux)
# and update pip itself
python3 -m venv venv
. venv/bin/activate
python -m pip install -U pip

# install
python -m pip install -e .

# install additional development dependencies
python -m pip install -r tests/test_requirements.txt

# run tests
python -m pytest -v
```

## Contribute

Contributions, suggestions and proposals welcome!

## License

Under terms of the [MIT license](https://opensource.org/licenses/MIT).

**NOTE**: This software depends on packages that _might_ be licensed under different terms.
