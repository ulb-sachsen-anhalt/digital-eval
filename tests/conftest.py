# -*- coding: utf-8 -*-
"""Shared test functionalities"""

from os.path import (
    join
)

from pathlib import (
    Path
)


PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_RES_DIR = join(PROJECT_ROOT_DIR, 'resources')
TEST_RES_DIR = join(PROJECT_ROOT_DIR, 'tests', 'resources')
