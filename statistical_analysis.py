"""
Statistical Analysis of Portfolio Performance

Tests statistical significance of differences between models.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config

