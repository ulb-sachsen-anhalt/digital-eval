# digital eval

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
