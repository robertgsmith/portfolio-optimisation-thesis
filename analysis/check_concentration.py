"""
Quick Check: Portfolio Concentration Analysis

Check if your portfolios are too concentrated.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config


def analyse_concentration():
    """Check portfolio concentration."""
    
    print("\n" + "="*70)
    print("PORTFOLIO CONCENTRATION ANALYSIS")
    print("="*70)
    
    weights_dir = config.RESULTS_DIR / "weights"
    
    for model_file in sorted(weights_dir.glob("weights_*.csv")):
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        
        # Calculate metrics for each rebalancing
        herfindahl_series = (weights ** 2).sum(axis=1)
        n_positions_series = (weights > 0.01).sum(axis=1)  # Positions > 1%
        top10_weight_series = weights.apply(lambda x: x.nlargest(10).sum(), axis=1)
        
        print(f"\n{model_name}:")
        print(f"  Average Herfindahl Index: {herfindahl_series.mean():.4f}")
        print(f"  Average # positions >1%: {n_positions_series.mean():.1f}")
        print(f"  Average top 10 weight: {top10_weight_series.mean():.1%}")
        
        # Industry benchmarks
        print(f"\n  Comparison to industry standards:")
        
        # Herfindahl
        if herfindahl_series.mean() > 0.05:
            print(f"    !!!  Herfindahl ({herfindahl_series.mean():.4f}) > 0.05 (concentrated!)")
        else:
            print(f"    ++ Herfindahl ({herfindahl_series.mean():.4f}) <= 0.05 (diversified)")
        
        # Number of positions
        if n_positions_series.mean() < 30:
            print(f"    !!!  Avg positions ({n_positions_series.mean():.1f}) < 30 (concentrated!)")
        else:
            print(f"    ++ Avg positions ({n_positions_series.mean():.1f}) >= 30 (diversified)")
        
        # Top 10 concentration
        if top10_weight_series.mean() > 0.60:
            print(f"    !!!  Top 10 weight ({top10_weight_series.mean():.1%}) > 60% (concentrated!)")
        else:
            print(f"    ++ Top 10 weight ({top10_weight_series.mean():.1%}) <= 60% (diversified)")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nIf you see !!! warnings:")
    print("  → Portfolios ARE concentrated")
    print("  → This explains similar turnover across methods")
    print("  → Consider adding further diversification constraints")
    print("\nIf you see ++ checks:")
    print("  → Portfolios are reasonably diversified")
    print("  → Basic constraints are sufficient")
    print("  → Similar turnover has other explanations")
    print("="*70 + "\n")


if __name__ == "__main__":
    analyse_concentration()

