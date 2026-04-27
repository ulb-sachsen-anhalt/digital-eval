# OCR Util

![example workflow](https://github.com/ulb-sachsen-anhalt/ocr-util/actions/workflows/python-app.yml/badge.svg)
[![PyPi version](https://badgen.net/pypi/v/digital-eval/)](https://pypi.org/project/digital-eval) ![PyPI - Downloads](https://img.shields.io/pypi/dm/digital-eval) ![PyPI - License](https://img.shields.io/pypi/l/digital-eval) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/digital-eval)

Collection of utils to 
* evaluation of OCR data for the masses
* generation of extended OCR-Evaluation Corpora
* generation of pair-wise Trainingdata for OCR-Backends

## Requirements

* recent *nix-OS
* Python3.10+ Environment

## Local installation

```bash
pip install - e.
```

## Usage

```bash
# evaluation
ocr eval --help

# corpus management
ocr generate-corpus --help
```

### Evaluation data problems

Inconsistent OCR Groundtruth with empty texts (ALTO String elements missing CONTENT or PAGE without TextEquiv) or invalid geometrical coordinates (less than 3 points or even empty) will lead to evaluation errors if geometry must be respected.

_Please note_:  
Erroneous or otherwise insufficient data files are excluded from evaluation.

## Development

Plattform: Intel(R) Core(TM) i5-6500 CPU@3.20GHz, 16GB RAM, Ubuntu 22.04 LTS, Python 3.10+

```bash
# clone local
git clone <repository-url> <local-dir>
cd <local-dir>

# enable virtual python 3 environment (linux)
# and update pip itself
python3.10 -m venv venv
. venv/bin/activate
python -m pip install -U pip

# install with dev dependencies
python -m pip install -e ".[dev,test]"

# run tests with coverage
python -m pytest --cov=src

# run tests faster (parallel, auto worker count)
python -m pytest -q -n auto
```

## Contribute

Contributions, suggestions and proposals welcome!

## License

Under terms of the [MIT license](https://opensource.org/licenses/MIT).

**NOTE**: This software depends on packages that _might_ be licensed under different terms.
