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


def check_transaction_costs():
    """Test different transaction cost levels."""
    
    print("\n" + "="*70)
    print("ROBUSTNESS CHECK: Transaction Costs")
    print("="*70)
    
    returns = pd.read_csv(config.get_data_path("log_returns.csv", "processed"),
                         index_col=0, parse_dates=True)
    
    models = {
        'Mean-Variance': MeanVariancePortfolio(),
        'Shrinkage': ShrinkagePortfolio(),
        'Bayesian': BayesianPortfolio(),
        'Robust': RobustPortfolio(),
        'Equal Weight': EqualWeightPortfolio()
    }
    
    cost_scenarios = [0.0005, 0.001, 0.0015, 0.0025]  # 5, 10, 15, 25 bps
    
    results = []
    
    for cost in cost_scenarios:
        print(f"\nTesting transaction cost: {cost*10000:.0f} bps")
        
        backtester = Backtester(
            returns=returns,
            models=models,
            transaction_cost=cost
        )
        
        backtester.run_backtest(verbose=False)
        metrics = backtester.calculate_metrics()
        
        for model_name in models.keys():
            results.append({
                'Model': model_name,
                'Transaction Cost (bps)': cost * 10000,
                'Sharpe Ratio': metrics.loc[model_name, 'sharpe_ratio'],
                'Annual Return': metrics.loc[model_name, 'annual_return'],
                'Turnover': metrics.loc[model_name, 'avg_turnover']
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'robustness_transaction_costs.csv', index=False)
    print(f"\n✓ Saved: robustness_transaction_costs.csv")
