"""
Run Full Analysis Pipeline

Simple master script to execute complete analysis.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import subprocess
import sys
from pathlib import Path

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_script(script_name, description):
    """Run a Python script."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode == 0:
        print(f"! {description} completed")
        return True
    else:
        print(f"!!! {description} failed")
        return False

def main():
    """Run complete pipeline."""
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMISATION - COMPLETE ANALYSIS")
    print("="*70)
    print("\nThis will run:")
    print("  1. Portfolio backtesting")
    print("  2. Visualisations")
    print("  3. Statistical tests")
    print("  4. Robustness checks")
    print("\nEstimated time: ~10 minutes")
    print("="*70)
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Backtesting
    if not run_script("scripts/run_backtest.py", "Portfolio Backtesting"):
        print("\n !!! Backtesting failed, but continuing...")
    
    # Step 2: Visualisations
    run_script("analysis/visualise_results.py", "Visualisations")
    
    # Step 3: Statistical Tests
    run_script("analysis/statistical_analysis.py", "Statistical Tests")
    
    # Step 4: Robustness Checks
    run_script("analysis/robustness_checks.py", "Robustness Checks")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults in: {Path('results').absolute()}")
    print("\nKey outputs:")
    print("  - Figures: results/figures/")
    print("  - Tables: results/tables/")
    print("  - Metrics: results/backtest_metrics.csv")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()