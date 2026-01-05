"""
Robustness Checks

Test sensitivity to parameters and assumptions.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config
from portfolio_models import *
from backtesting import Backtester
