# digital eval

![example workflow](https://github.com/ulb-sachsen-anhalt/digital-eval/actions/workflows/python-app.yml/badge.svg)

Python3 Tool to report evaluation outcomes from mass digitalization workflows.

## Features

* match automatically groundtruth (i.e. reference data) and candidates by filename
* use geometric information to evaluate only specific frame (i.e. specific column or region from large page) of candidates (requires ALTO or PAGE format)
* aggregate evaluation outcome on domain range (with multiple subdomains)
* choose from textual metrics based on characters or words plus common Information Retrieval
* choose between accuracy / error rate and different UTF-8 Python norms
* formats: ALTO, PAGE or plain text for both groundtruth and candidates
* speedup with parallel execution

## Installation

```bash
pip install digital-eval
```

## Usage

### Metrics

Calculate similarity (`acc`) or difference (`err`) ratios between single reference/groundtruth and test/candidate item.  

#### Edit-Distance based

Character-based text string minus whitechars (`Cs`, `Characters`) or Letter-based (`Ls`, `Letters`) minus whites, punctuation and digits.
Word/Token-based edit-distance of single tokens identified by whitespaces.

#### Set based

Calculate union of sets of tokens/words (`BoW`, `BagOfWords`).
Operate on sets of tokens/words with respect to language specific stopwords using [nltk](https://www.nltk.org/)-framework for:

* Precision (`IRPre`, `Pre`, `Precision`): How many tokens from candidate are in groundtruth reference?
* Recall (`IRRec`, `Rec`, `Recall`): How many tokens from groundtruth reference should candidate include?
* F-Measure (`IRFMeasure`, `FM`): weighted ratio Precision / Recall

### UTF-8 Normalisations

Use standard Python Implementation of UTF-8 normalizations; default: `NFKD`.

### Statistics

Statistics calculated via [numpy](https://numpy.org/) include arithmetic mean, median and outlier detection with interquartile range and are based on the specific groundtruth/reference (ref) for each metric, i.e. char, letters or tokens.

### Evaluate treelike structures

To evaluate OCR-candidate-data batch-like versus existing Groundtruth, please make sure that your structures fit this way:

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

Now call via:

```bash
digital-eval <path-candidate-root>/domain/ -ref <path-groundtruth>/domain/
```

for an aggregated overview on stdout. Feel free to increase verbosity via `-v` (or even `-vv`) to get detailed information about each single data set which was evaluated.

Structured OCR is considered to contain valid geometrical and textual data on word level, even though for recent PAGE also line level is possible.

### Data problems  

Inconsistent OCR Groundtruth with empty texts (ALTO String elements missing CONTENT or PAGE without TextEquiv) or invalid geometrical coordinates (less than 3 points or even empty) will lead to evaluation errors if geometry must be respected.

## Development

Plattform: Intel(R) Core(TM) i5-6500 CPU@3.20GHz, 16GB RAM, Ubuntu 20.04 LTS, Python 3.8.

```bash
# clone local
git clone <repository-url> <local-dir>
cd <local-dir>

# enable virtual python environment (linux)
# and install libraries
python3 -m venv venv
. venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# install
pip install .

# optional:
# install additional development dependencies
pip install -r tests/test_requirements.txt
pytest -v

# run
digital-eval --help
```

## Contribute

Contributions, suggestions and proposals welcome!

## Licence

Under terms of the [MIT license](https://opensource.org/licenses/MIT).

**NOTE**: This software depends on other packages that _may_ be licensed under different open source licenses.
