# Compare results with vs without diversification constraints (requires copying results/ to results_baseline/)

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compare_results():
    """Compare baseline vs constrained results."""
    
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT DIVERSIFICATION CONSTRAINTS")
    print("="*70)
    
    # Load baseline results (without constraints)
    try:
        baseline_metrics = pd.read_csv('results_baseline/backtest_metrics.csv', index_col=0)
        print("\n>> Loaded baseline results (no constraints)")
    except:
        print("\n!!!  Baseline results not found. Did you run with ENABLE_DIVERSIFICATION = False ? (Found in config.py in diversification section)")
        print("   Run: python run_backtest.py with ENABLE_DIVERSIFICATION = False")
        print("   Then copy results/ to results_baseline/ and then rerun with ENABLE_DIVERSIFICATION = True")
        return
    
    # Load constrained results (with constraints)
    try:
        constrained_metrics = pd.read_csv('results/backtest_metrics.csv', index_col=0)
        print(">> Loaded constrained results (with constraints)")
    except:
        print("\n!!!  Constrained results not found. Run: python run_backtest.py")
        return
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Baseline (No Constraints)': baseline_metrics['sharpe_ratio'],
        'Constrained (Min 20 Assets)': constrained_metrics['sharpe_ratio'],
        'Difference': constrained_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
    })
    
    print("\n" + "-"*70)
    print("SHARPE RATIO COMPARISON")
    print("-"*70)
    print(comparison.round(4))
    
    # Turnover comparison
    turnover_comparison = pd.DataFrame({
        'Baseline Turnover': baseline_metrics['avg_turnover'],
        'Constrained Turnover': constrained_metrics['avg_turnover'],
        'Difference': constrained_metrics['avg_turnover'] - baseline_metrics['avg_turnover']
    })
    
    print("\n" + "-"*70)
    print("TURNOVER COMPARISON")
    print("-"*70)
    print(turnover_comparison.round(4))
    
    # Load concentration data
    try:
        # Run check_concentration on both
        print("\n" + "-"*70)
        print("CONCENTRATION COMPARISON")
        print("-"*70)
        print("\nBaseline (No Constraints):")
        print("  Should show: Herfindahl ~0.099, ~10 effective assets")
        print("\nConstrained (Min 20 Assets):")
        print("  Should show: Herfindahl ~0.05, ~20 effective assets")
        print("\nRun check_concentration.py on both results folders to verify.")
    except:
        pass
    
    # Save comparison
    comparison.to_csv('results/comparison_sharpe.csv')
    turnover_comparison.to_csv('results/comparison_turnover.csv')
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Check if methods now differ
    sharpe_std = comparison['Constrained (Min 20 Assets)'].std()
    if sharpe_std > 0.01:
        print("\n++ Robust methods NOW show differentiation!")
        print(f"   Sharpe ratio std: {sharpe_std:.4f}")
    else:
        print("\n!!!  Methods still similar (might need stronger constraints)")
    
    print("\n>> Saved: results/comparison_sharpe.csv")
    print(">> Saved: results/comparison_turnover.csv")
    print("="*70 + "\n")

if __name__ == "__main__":
    compare_results()