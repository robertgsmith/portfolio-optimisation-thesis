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
    
    returns = pd.read_csv(config.RESULTS_DIR / "backtest_returns.csv",
                         index_col=0, parse_dates=True)
    
    # Get actual date range from data
    actual_start = returns.index.min()
    actual_end = returns.index.max()
    
    print(f"Data available from {actual_start.date()} to {actual_end.date()}")
    
    models = {
        'Mean-Variance': MeanVariancePortfolio(),
        'Shrinkage': ShrinkagePortfolio(),
        'Bayesian': BayesianPortfolio(),
        'Robust': RobustPortfolio(),
        'Equal Weight': EqualWeightPortfolio()
    }
    
    # Define sub-periods - use same timezone as data
    # Extract timezone from data if present
    tz = actual_start.tz if hasattr(actual_start, 'tz') else None
    
    def make_timestamp(date_str):
        """Create timestamp with same timezone as data."""
        ts = pd.Timestamp(date_str)
        if tz is not None:
            ts = ts.tz_localize(tz)
        return ts
    
    # Define sub-periods based on actual data availability
    periods = {}
    
    # Pre-COVID period
    pre_covid_end = make_timestamp('2019-12-31')
    if actual_start <= pre_covid_end and actual_end >= actual_start:
        pre_covid_start = max(actual_start, make_timestamp('2012-01-01'))
        if (pre_covid_end - pre_covid_start).days > config.ESTIMATION_WINDOW:
            periods['Pre-COVID (2012-2019)'] = (pre_covid_start, pre_covid_end)
    
    # COVID period
    covid_start = make_timestamp('2020-01-01')
    covid_end = make_timestamp('2021-12-31')
    if actual_start <= covid_end and actual_end >= covid_start:
        period_start = max(actual_start, covid_start)
        period_end = min(actual_end, covid_end)
        if (period_end - period_start).days > config.ESTIMATION_WINDOW:
            periods['COVID Era (2020-2021)'] = (period_start, period_end)
    
    # Post-COVID period
    post_covid_start = make_timestamp('2022-01-01')
    if actual_end >= post_covid_start:
        period_start = max(actual_start, post_covid_start)
        if (actual_end - period_start).days > config.ESTIMATION_WINDOW:
            periods['Post-COVID (2022-2024)'] = (period_start, actual_end)
    
    if not periods:
        print("⚠️  No valid sub-periods found in data")
        return
    
    print(f"\nTesting {len(periods)} sub-periods...")
    
    results = []
    
    for period_name, (start_ts, end_ts) in periods.items():
        print(f"\nTesting period: {period_name}")
        print(f"  Date range: {start_ts.date()} to {end_ts.date()}")
        
        # Filter returns for this period
        period_returns = returns.loc[start_ts:end_ts]
        
        print(f"  Trading days: {len(period_returns)}")
        
        # Check if we have enough data
        min_required = config.ESTIMATION_WINDOW + 252
        if len(period_returns) < min_required:
            print(f"  ⚠️  Skipping (need {min_required} days, have {len(period_returns)})")
            continue
        
        try:
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
                    'Start Date': start_ts.strftime('%Y-%m-%d'),
                    'End Date': end_ts.strftime('%Y-%m-%d'),
                    'Trading Days': len(period_returns),
                    'Sharpe Ratio': metrics.loc[model_name, 'sharpe_ratio'],
                    'Annual Return': metrics.loc[model_name, 'annual_return'],
                    'Annual Volatility': metrics.loc[model_name, 'annual_volatility'],
                    'Max Drawdown': metrics.loc[model_name, 'max_drawdown']
                })
            
            print(f"  ✓ Completed successfully")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(config.RESULTS_DIR / 'robustness_subperiods.csv', index=False)
        print(f"\n✓ Saved: robustness_subperiods.csv")
        
        # Display summary
        print("\n" + "-"*70)
        print("SUMMARY BY PERIOD")
        print("-"*70)
        for period_name in [p for p in periods.keys()]:
            period_data = results_df[results_df['Period'] == period_name]
            if not period_data.empty:
                print(f"\n{period_name}:")
                summary = period_data[['Model', 'Sharpe Ratio', 'Annual Return', 'Max Drawdown']]
                print(summary.to_string(index=False))
    else:
        print("\n⚠️  No results generated")


def main():
    """Run all robustness checks."""
    
    print("\n" + "="*70)
    print("RUNNING ROBUSTNESS CHECKS")
    print("="*70)
    
    check_transaction_costs()
    check_estimation_windows()
    check_subperiods()
    
    print("\n" + "="*70)
    print("ALL ROBUSTNESS CHECKS COMPLETED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
