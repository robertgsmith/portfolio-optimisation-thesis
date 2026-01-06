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


def check_estimation_windows():
    """Test different estimation window sizes."""
    
    print("\n" + "="*70)
    print("ROBUSTNESS CHECK: Estimation Windows")
    print("="*70)
    
    returns = pd.read_csv(config.get_data_path("log_returns.csv", "processed"),
                         index_col=0, parse_dates=True)
    
    models = {
        'Mean-Variance': MeanVariancePortfolio(),
        'Shrinkage': ShrinkagePortfolio(),
        'Bayesian': BayesianPortfolio(),
        'Robust': RobustPortfolio()
    }
    
    windows = [126, 252, 504]  # 6 months, 1 year, 2 years
    
    results = []
    
    for window in windows:
        print(f"\nTesting estimation window: {window} days")
        
        backtester = Backtester(
            returns=returns,
            models=models,
            estimation_window=window
        )
        
        backtester.run_backtest(verbose=False)
        metrics = backtester.calculate_metrics()
        
        for model_name in models.keys():
            results.append({
                'Model': model_name,
                'Estimation Window': window,
                'Sharpe Ratio': metrics.loc[model_name, 'sharpe_ratio'],
                'Annual Return': metrics.loc[model_name, 'annual_return'],
                'Annual Volatility': metrics.loc[model_name, 'annual_volatility']
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'robustness_estimation_windows.csv', index=False)
    print(f"\n✓ Saved: robustness_estimation_windows.csv")


def check_subperiods():
    """Test performance in different market regimes."""
    
    print("\n" + "="*70)
    print("ROBUSTNESS CHECK: Sub-periods")
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
    
    # Define sub-periods
    periods = {
        'Pre-COVID (2010-2019)': ('2010-01-01', '2019-12-31'),
        'COVID Era (2020-2021)': ('2020-01-01', '2021-12-31'),
        'Post-COVID (2022-2024)': ('2022-01-01', '2024-12-31')
    }
    
    results = []
    
    for period_name, (start, end) in periods.items():
        print(f"\nTesting period: {period_name}")
        
        period_returns = returns.loc[start:end]
        
        if len(period_returns) < config.ESTIMATION_WINDOW + 252:
            print(f"  Skipping (insufficient data)")
            continue
        
        backtester = Backtester(
            returns=period_returns,
            models=models
        )
        
        backtester.run_backtest(verbose=False)
        metrics = backtester.calculate_metrics()
        
        for model_name in models.keys():
            results.append({
                'Model': model_name,
                'Period': period_name,
                'Sharpe Ratio': metrics.loc[model_name, 'sharpe_ratio'],
                'Annual Return': metrics.loc[model_name, 'annual_return'],
                'Max Drawdown': metrics.loc[model_name, 'max_drawdown']
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'robustness_subperiods.csv', index=False)
    print(f"\n✓ Saved: robustness_subperiods.csv")