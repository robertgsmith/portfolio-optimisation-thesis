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
    
    # Load INDIVIDUAL ASSET returns
    returns_path = config.get_data_path("log_returns.csv", "processed")
    
    if not returns_path.exists():
        print("⚠️  Asset returns not found. Run data pipeline first.")
        return
    
    # Read CSV
    # try re-reading with explicit parse_dates for the first column
    returns = pd.read_csv(
        returns_path,
        index_col=0,
        parse_dates=[0]
    )

    idx = returns.index

    # If parse_dates produced something other than DatetimeIndex, try coercion
    if not isinstance(idx, pd.DatetimeIndex):
        # Coerce strings -> datetimes (use utc to correctly parse offsets like -05:00)
        coerced = pd.to_datetime(idx, errors="coerce", utc=True)

        # If coercion worked, coerced is a DatetimeIndex (with tz=UTC)
        if coerced.isnull().any():
            # show problematic rows and raise to let you inspect the CSV
            bad = returns.index[coerced.isnull()]
            print("Unparseable index entries (first 10):", list(bad[:10]))
            raise ValueError("Some index rows could not be parsed as datetimes. Check CSV index column.")
        # convert to tz-naive local time (optional)
        coerced = coerced.tz_convert(None)
        returns.index = coerced
    else:
        # If already DatetimeIndex with tz info, remove tz (convert safely)
        if returns.index.tz is not None:
            returns.index = returns.index.tz_convert(None)


    print(f"Loaded individual asset returns")
    print(f"  Assets: {len(returns.columns)}")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Select top 30 most liquid assets
    print("\nSelecting top 30 most liquid assets for sub-period analysis...")
    liquidity = returns.abs().mean().sort_values(ascending=False)
    top_assets = liquidity.head(30).index
    returns_subset = returns[top_assets]
    
    print(f"Selected assets: {', '.join(list(top_assets[:5]))} ...")
    
    # Models with relaxed constraints
    models = {
        'Mean-Variance': MeanVariancePortfolio(max_weight=0.25, risk_aversion=2.0),
        'Shrinkage': ShrinkagePortfolio(max_weight=0.25, risk_aversion=2.0),
        'Bayesian': BayesianPortfolio(max_weight=0.25, risk_aversion=2.0),
        'Robust': RobustPortfolio(max_weight=0.25, epsilon=0.2, risk_aversion=2.0),
        'Equal Weight': EqualWeightPortfolio(max_weight=0.25)
    }
    
    # Get actual date range
    actual_start = returns_subset.index.min()
    actual_end = returns_subset.index.max()
    
    # Define periods as timestamps
    periods = {}
    
    # Pre-COVID
    pre_start = max(actual_start, pd.Timestamp('2015-01-01'))
    pre_end = pd.Timestamp('2019-12-31')
    if pre_start < pre_end and actual_start <= pre_end:
        periods['Pre-COVID (2015-2019)'] = (pre_start, pre_end)
    
    # COVID
    covid_start = pd.Timestamp('2020-01-01')
    covid_end = pd.Timestamp('2021-12-31')
    if actual_start <= covid_end and actual_end >= covid_start:
        periods['COVID Era (2020-2021)'] = (
            max(actual_start, covid_start),
            min(actual_end, covid_end)
        )
    
    # Post-COVID
    post_start = pd.Timestamp('2022-01-01')
    if actual_end >= post_start:
        periods['Post-COVID (2022-2024)'] = (
            max(actual_start, post_start),
            actual_end
        )
    
    if not periods:
        print("⚠️  No valid periods found in data")
        return
    
    print(f"\nTesting {len(periods)} periods...")
    
    results = []
    
    for period_name, (start_ts, end_ts) in periods.items():
        print(f"\n{period_name}")
        print(f"  Dates: {start_ts.date()} to {end_ts.date()}")
        
        period_returns = returns_subset.loc[start_ts:end_ts]
        print(f"  Trading days: {len(period_returns)}")
        
        if len(period_returns) < 300:
            print(f"  ⚠️  Skipping (insufficient data)")
            continue
        
        try:
            backtester = Backtester(
                returns=period_returns,
                models=models,
                estimation_window=126,
                rebalancing_freq=63,
                transaction_cost=0.001
            )
            
            backtester.run_backtest(verbose=False)
            metrics = backtester.calculate_metrics()
            
            # Check variation
            sharpes = metrics['sharpe_ratio'].values
            sharpe_range = sharpes.max() - sharpes.min()
            print(f"  Sharpe range: {sharpe_range:.4f}", end="")
            
            if sharpe_range < 0.01:
                print(" ⚠️  (minimal variation)")
            else:
                print(" ✓")
            
            for model_name in models.keys():
                results.append({
                    'Model': model_name,
                    'Period': period_name,
                    'Start': start_ts.strftime('%Y-%m-%d'),
                    'End': end_ts.strftime('%Y-%m-%d'),
                    'Days': len(period_returns),
                    'Sharpe': metrics.loc[model_name, 'sharpe_ratio'],
                    'Return': metrics.loc[model_name, 'annual_return'],
                    'Volatility': metrics.loc[model_name, 'annual_volatility'],
                    'Max DD': metrics.loc[model_name, 'max_drawdown'],
                    'Turnover': metrics.loc[model_name, 'avg_turnover']
                })
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(config.RESULTS_DIR / 'robustness_subperiods.csv', index=False)
        print(f"\n✓ Saved: robustness_subperiods.csv")
        
        # Summary
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        for period in periods.keys():
            pdata = results_df[results_df['Period'] == period]
            if not pdata.empty:
                print(f"\n{period}:")
                print(pdata[['Model', 'Sharpe', 'Return', 'Volatility', 'Turnover']].to_string(index=False))
    else:
        print("\n⚠️  No results - skipping sub-period analysis")


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
