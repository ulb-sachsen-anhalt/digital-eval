# digital eval

![example workflow](https://github.com/ulb-sachsen-anhalt/digital-eval/actions/workflows/python-app.yml/badge.svg)

Python3 Tool to evaluate outcomes from mass digitalization workflows.

## Features

* textual metrics based on characters or words plus common Information Retrieval
* groundtruth formats: ALTO, PAGE or plain text
* candidate formats: ALTO, PAGE or plain text
* match groundtruth and candidates by filename beginnings
* sum up evaluation on domain range (with multiple subdomains)
* speedup with parallel execution depending on available cores
* use geometric information to evaluate only specific frame (i.e. specific column or region from large page) of candidate (required ALTO or PAGE format)

## Usage

## Installation

```bash
# clone local
git clone <repository-url> <local-dir>

# enable virtual python environment (linux)
# and install libraries
python3 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# install and run
pip install .
digital-eval --help
```

### Metrics

#### Character based Metrics

* Character Accuracy (CCA): calculate similarity of candidate and reference groundtruth with edit-distance
* Letter Accuracy (CLA): respect only alphabetical characters, i.e. no whitespaces, interpunctuations or numbers (arabic and persian)

#### Word based Metrics

* Word Accuracy (WWA): like CCA, but on "word"-level ("word" means each whitespace separated token, therefore also numbers, abbreviations, etc.)
* Bag of Words (WBoW): no respect to "word" order, calculate intersection between candidate and groundtruth reference

#### Information Retrieval

Operate on sets of "words" with respect to language specific stopwords using [nltk](https://www.nltk.org/)-framework.

* Precision (IRPre): How many tokens from candidate are also in groundtruth reference?
* Recall (IRRec): How many tokens from groundtruth reference should candidate include?
* F-Measure (IRFM): weighted mean Precision / Recall

### Statistics

Statistics are calculated via [numpy](https://numpy.org/) and include arithmetic mean, median and an outlier detection using interquartile range.

For each metric statistics are calculated based on the specific groundtruth reference (ref) for each metric, i.e. characters, letters, tokens or a set of tokens.

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

## Contribute

Contributions, suggestions and proposals welcome!

### Development

```bash
# install additional development dependencies
pip install -r tests/test_requirements.txt

# install in editable mode
pip install -e .

# run tests
pytest -v
```

## Licence

Under terms of the [MIT license](https://opensource.org/licenses/MIT).

**NOTE**: This software depends on other packages that _may_ be licensed under different open source licenses.
