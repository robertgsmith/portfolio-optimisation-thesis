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
