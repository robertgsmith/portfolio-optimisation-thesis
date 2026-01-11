"""
Statistical Analysis of Portfolio Performance

Tests statistical significance of differences between models.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config


def bootstrap_sharpe_difference(returns1, returns2, n_bootstrap=10000):
    """
    Bootstrap test for difference in Sharpe ratios.
    
    Parameters
    ----------
    returns1 : pd.Series
        Returns for model 1
    returns2 : pd.Series
        Returns for model 2
    n_bootstrap : int
        Number of bootstrap samples
    
    Returns
    -------
    p_value : float
        P-value for H0: Sharpe1 = Sharpe2
    sharpe_diff : float
        Actual difference in Sharpe ratios
    confidence_interval : tuple
        95% confidence interval
    """
    # Actual Sharpe ratios
    sharpe1 = returns1.mean() / returns1.std() * np.sqrt(252)
    sharpe2 = returns2.mean() / returns2.std() * np.sqrt(252)
    actual_diff = sharpe1 - sharpe2
    
    # Bootstrap
    sharpe_diffs = []
    n = len(returns1)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        sample1 = returns1.iloc[idx].values
        sample2 = returns2.iloc[idx].values
        
        # Calculate Sharpe ratios
        s1 = sample1.mean() / sample1.std() * np.sqrt(252) if sample1.std() > 0 else 0
        s2 = sample2.mean() / sample2.std() * np.sqrt(252) if sample2.std() > 0 else 0
        
        sharpe_diffs.append(s1 - s2)
    
    sharpe_diffs = np.array(sharpe_diffs)
    
    # P-value (two-tailed)
    p_value = 2 * min(
        np.mean(sharpe_diffs >= 0),
        np.mean(sharpe_diffs <= 0)
    )
    
    # 95% confidence interval
    ci_lower = np.percentile(sharpe_diffs, 2.5)
    ci_upper = np.percentile(sharpe_diffs, 97.5)
    
    return p_value, actual_diff, (ci_lower, ci_upper)


def test_mean_returns():
    """Test if mean returns differ significantly between models."""
    
    print("\n" + "="*70)
    print("TEST 1: Difference in Mean Returns")
    print("="*70)
    
    returns = pd.read_csv(config.RESULTS_DIR / "backtest_returns.csv",
                         index_col=0, parse_dates=True)
    
    results = []
    
    # Compare robust methods vs Mean-Variance
    baseline = 'Mean-Variance'
    comparisons = ['Shrinkage', 'Bayesian', 'Robust', 'Equal Weight']
    
    for model in comparisons:
        # Paired t-test (same dates)
        t_stat, p_value = ttest_rel(returns[baseline], returns[model])
        
        mean_diff = returns[baseline].mean() - returns[model].mean()
        
        results.append({
            'Comparison': f'{baseline} vs {model}',
            'Mean Difference (daily)': mean_diff,
            'Mean Difference (annual)': mean_diff * 252,
            't-statistic': t_stat,
            'p-value': p_value,
            'Significant (5%)': 'Yes' if p_value < 0.05 else 'No'
        })
        
        print(f"\n{baseline} vs {model}:")
        print(f"  Mean difference: {mean_diff*252:.4f} (annualised)")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'test_mean_returns.csv', index=False)
    print(f"\n✓ Saved: test_mean_returns.csv")


def test_sharpe_ratios():
    """Test if Sharpe ratios differ significantly between models."""
    
    print("\n" + "="*70)
    print("TEST 2: Difference in Sharpe Ratios (Bootstrap)")
    print("="*70)
    
    returns = pd.read_csv(config.RESULTS_DIR / "backtest_returns.csv",
                         index_col=0, parse_dates=True)
    
    results = []
    
    baseline = 'Mean-Variance'
    comparisons = ['Shrinkage', 'Bayesian', 'Robust', 'Equal Weight']
    
    for model in comparisons:
        p_value, actual_diff, (ci_lower, ci_upper) = bootstrap_sharpe_difference(
            returns[baseline],
            returns[model],
            n_bootstrap=10000
        )
        
        results.append({
            'Comparison': f'{baseline} vs {model}',
            'Sharpe Difference': actual_diff,
            'CI Lower (95%)': ci_lower,
            'CI Upper (95%)': ci_upper,
            'p-value': p_value,
            'Significant (5%)': 'Yes' if p_value < 0.05 else 'No'
        })
        
        print(f"\n{baseline} vs {model}:")
        print(f"  Sharpe difference: {actual_diff:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'test_sharpe_ratios.csv', index=False)
    print(f"\n✓ Saved: test_sharpe_ratios.csv")


def test_volatility():
    """Test if volatility differs significantly between models."""
    
    print("\n" + "="*70)
    print("TEST 3: Difference in Volatility")
    print("="*70)
    
    returns = pd.read_csv(config.RESULTS_DIR / "backtest_returns.csv",
                         index_col=0, parse_dates=True)
    
    results = []
    
    baseline = 'Mean-Variance'
    comparisons = ['Shrinkage', 'Bayesian', 'Robust', 'Equal Weight']
    
    for model in comparisons:
        # Levene's test for equality of variances
        stat, p_value = stats.levene(returns[baseline], returns[model])
        
        vol_baseline = returns[baseline].std() * np.sqrt(252)
        vol_model = returns[model].std() * np.sqrt(252)
        vol_diff = vol_baseline - vol_model
        
        results.append({
            'Comparison': f'{baseline} vs {model}',
            f'{baseline} Vol': vol_baseline,
            f'{model} Vol': vol_model,
            'Volatility Difference': vol_diff,
            'Levene Statistic': stat,
            'p-value': p_value,
            'Significant (5%)': 'Yes' if p_value < 0.05 else 'No'
        })
        
        print(f"\n{baseline} vs {model}:")
        print(f"  {baseline} volatility: {vol_baseline:.4f}")
        print(f"  {model} volatility: {vol_model:.4f}")
        print(f"  Difference: {vol_diff:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'test_volatility.csv', index=False)
    print(f"\n✓ Saved: test_volatility.csv")


def test_turnover():
    """Test if turnover differs significantly between models."""
    
    print("\n" + "="*70)
    print("TEST 4: Difference in Portfolio Turnover")
    print("="*70)
    
    # Load weights for each model
    weights_dir = config.RESULTS_DIR / "weights"
    
    turnovers = {}
    
    for model_file in weights_dir.glob("weights_*.csv"):
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        
        # Calculate turnover at each rebalancing
        turnover_series = weights.diff().abs().sum(axis=1).dropna()
        turnovers[model_name] = turnover_series
    
    # Compare turnover distributions
    results = []
    
    for model_name, turnover in turnovers.items():
        results.append({
            'Model': model_name,
            'Mean Turnover': turnover.mean(),
            'Median Turnover': turnover.median(),
            'Std Turnover': turnover.std(),
            'Max Turnover': turnover.max()
        })
        
        print(f"\n{model_name}:")
        print(f"  Mean turnover: {turnover.mean():.4f}")
        print(f"  Median turnover: {turnover.median():.4f}")
        print(f"  Std: {turnover.std():.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'test_turnover.csv', index=False)
    print(f"\n✓ Saved: test_turnover.csv")


def test_drawdowns():
    """Test if maximum drawdowns differ significantly."""
    
    print("\n" + "="*70)
    print("TEST 5: Maximum Drawdown Comparison")
    print("="*70)
    
    drawdowns = pd.read_csv(config.RESULTS_DIR / "backtest_drawdowns.csv",
                           index_col=0, parse_dates=True)
    
    results = []
    
    for model in drawdowns.columns:
        max_dd = drawdowns[model].min()
        
        # Recovery time (days from max DD to recovery)
        dd_series = drawdowns[model]
        max_dd_date = dd_series.idxmin()
        recovery_dates = dd_series[max_dd_date:][dd_series >= 0]
        
        if len(recovery_dates) > 0:
            recovery_time = (recovery_dates.index[0] - max_dd_date).days
        else:
            recovery_time = np.nan  # Not yet recovered
        
        results.append({
            'Model': model,
            'Max Drawdown': max_dd,
            'Max DD Date': max_dd_date.strftime('%Y-%m-%d'),
            'Recovery Time (days)': recovery_time
        })
        
        print(f"\n{model}:")
        print(f"  Max drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
        print(f"  Date: {max_dd_date.strftime('%Y-%m-%d')}")
        if not np.isnan(recovery_time):
            print(f"  Recovery time: {recovery_time:.0f} days")
        else:
            print(f"  Recovery: Not yet recovered")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'test_drawdowns.csv', index=False)
    print(f"\n✓ Saved: test_drawdowns.csv")


def create_summary_table():
    """Create comprehensive summary table of all tests."""
    
    print("\n" + "="*70)
    print("CREATING SUMMARY TABLE")
    print("="*70)
    
    # Load all test results
    mean_test = pd.read_csv(config.RESULTS_DIR / 'test_mean_returns.csv')
    sharpe_test = pd.read_csv(config.RESULTS_DIR / 'test_sharpe_ratios.csv')
    vol_test = pd.read_csv(config.RESULTS_DIR / 'test_volatility.csv')
    
    # Create summary
    summary = pd.DataFrame({
        'Test': ['Mean Return', 'Sharpe Ratio', 'Volatility'],
        'Robust vs MV Significant': [
            mean_test[mean_test['Comparison'].str.contains('Robust')]['Significant (5%)'].values[0],
            sharpe_test[sharpe_test['Comparison'].str.contains('Robust')]['Significant (5%)'].values[0],
            vol_test[vol_test['Comparison'].str.contains('Robust')]['Significant (5%)'].values[0]
        ]
    })
    
    print("\n", summary)
    
    summary.to_csv(config.RESULTS_DIR / 'statistical_tests_summary.csv', index=False)
    print(f"\n✓ Saved: statistical_tests_summary.csv")


def main():
    """Run all statistical tests."""
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS OF PORTFOLIO PERFORMANCE")
    print("="*70)
    
    test_mean_returns()
    test_sharpe_ratios()
    test_volatility()
    test_turnover()
    test_drawdowns()
    create_summary_table()
    
    print("\n" + "="*70)
    print("ALL STATISTICAL TESTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nKey findings will help answer your research questions:")
    print("  1. Are robust methods significantly better than MV?")
    print("  2. Do they provide more stable portfolios?")
    print("  3. Are differences statistically significant?")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
