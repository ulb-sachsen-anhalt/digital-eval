# digital eval

![example workflow](https://github.com/ulb-sachsen-anhalt/digital-eval/actions/workflows/python-app.yml/badge.svg)

Evaluate data from mass digitalization workflows

## Requirements

* Python 3.6+
* pip

## Installation

```bash
# clone local
git clone <repository-url> <local-dir>

# enable virtual python enviroment (linux)
# and install libraries
python3 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# install and run
pip install .
digital-eval --help
```

### Development

```bash
# dev dependencies
pip install -r tests/test_requirements.txt

# install with link
pip install -e .
```

## Evaluate OCR-Data Batch

### Features

* Formats: ALTO, PAGE or plain text
* match groundtruth and candiates by filename beginnings
* speedup with parallel executions
* use geometric information to evaluate only specific frames (ALTO or PAGE)

### Usage

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

Inconsistent OCR Groundtruth with empty texts (ALTO String elements missing CONTENT or PAGE without TextEquiv) or invalid geometrical coordinates (less than 3 points or even empty) will lead to evaluation errors.


# Contribute

Contributions welcome!

# Licence

Under terms of the [MIT license](https://opensource.org/licenses/MIT).

**NOTE**: This software depends on other packages that may be licensed under different open source licenses.
