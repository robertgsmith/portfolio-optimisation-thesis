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

def plot_cumulative_returns():
    """Plot cumulative returns for all models."""
    
    returns = pd.read_csv(config.RESULTS_DIR / "backtest_returns.csv", 
                         index_col=0, parse_dates=True)
    
    cum_returns = (1 + returns).cumprod() - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in cum_returns.columns:
        ax.plot(cum_returns.index, cum_returns[col], label=col, linewidth=2)
    
    ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'cumulative_returns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cumulative_returns.png")
    plt.close()
