"""
Visualise Backtest Results

Create comprehensive visualisations for thesis.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
import config

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figures directory
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
