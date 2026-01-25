"""
Analyse Expected Return Estimates

Check if historical means produce extreme values.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config


def analyse_expected_returns():
    """Analyse expected return estimates for extreme values."""
    
    print("\n" + "="*70)
    print("EXPECTED RETURN ANALYSIS")
    print("="*70)
    
    # Load returns
    returns_path = config.get_data_path("log_returns.csv", "processed")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    # Calculate rolling expected returns (as used in backtesting)
    window = config.ESTIMATION_WINDOW
    
    rolling_means = returns.rolling(window=window).mean() * config.TRADING_DAYS_PER_YEAR
    
    print(f"\nRolling Expected Returns (annualised, {window}-day window):")
    print(f"  Mean: {rolling_means.mean().mean():.4f}")
    print(f"  Median: {rolling_means.median().median():.4f}")
    print(f"  Min: {rolling_means.min().min():.4f}")
    print(f"  Max: {rolling_means.max().max():.4f}")
    print(f"  Std: {rolling_means.std().mean():.4f}")
    
    # Check for extreme values
    print("\n" + "-"*70)
    print("EXTREME VALUE DETECTION")
    print("-"*70)
    
    # Find instances where expected return > 100% or < -50%
    extreme_high = (rolling_means > 1.0).sum().sum()
    extreme_low = (rolling_means < -0.5).sum().sum()
    
    print(f"\nInstances where expected return > 100%: {extreme_high}")
    print(f"Instances where expected return < -50%: {extreme_low}")
    
    if extreme_high > 0 or extreme_low > 0:
        print("\n!!!  WARNING: Extreme expected returns detected!")
        print("   This can cause concentrated portfolios")
    else:
        print("\n++ No extreme expected returns detected")
    
    # Cross-sectional dispersion over time
    cross_sectional_std = rolling_means.std(axis=1)
    
    print(f"\nCross-sectional dispersion (std of expected returns across assets):")
    print(f"  Mean: {cross_sectional_std.mean():.4f}")
    print(f"  Max: {cross_sectional_std.max():.4f}")
    print(f"  95th percentile: {cross_sectional_std.quantile(0.95):.4f}")
    
    if cross_sectional_std.mean() > 0.3:
        print("\n!!!  High dispersion detected (mean > 30%)")
        print("   This suggests widely varying expected returns")
        print("   → Can lead to concentrated portfolios")
    else:
        print("\n++ Reasonable dispersion")
    
    # Plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution of expected returns (all time periods)
    ax1 = axes[0, 0]
    all_expected_returns = rolling_means.values.flatten()
    all_expected_returns = all_expected_returns[~np.isnan(all_expected_returns)]
    
    ax1.hist(all_expected_returns, bins=100, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', label='Zero')
    ax1.set_title('Distribution of Expected Returns (All Periods)', fontweight='bold')
    ax1.set_xlabel('Annualised Expected Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series of cross-sectional dispersion
    ax2 = axes[0, 1]
    ax2.plot(cross_sectional_std.index, cross_sectional_std.values)
    ax2.axhline(0.3, color='red', linestyle='--', label='High dispersion threshold')
    ax2.set_title('Cross-Sectional Dispersion Over Time', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Std of Expected Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Top 10 vs Bottom 10 expected returns over time
    ax3 = axes[1, 0]
    top10_mean = rolling_means.apply(lambda x: x.nlargest(10).mean(), axis=1)
    bottom10_mean = rolling_means.apply(lambda x: x.nsmallest(10).mean(), axis=1)
    
    ax3.plot(top10_mean.index, top10_mean.values, label='Top 10 assets', linewidth=2)
    ax3.plot(bottom10_mean.index, bottom10_mean.values, label='Bottom 10 assets', linewidth=2)
    ax3.fill_between(top10_mean.index, top10_mean.values, bottom10_mean.values, alpha=0.3)
    ax3.set_title('Expected Return Gap (Top 10 vs Bottom 10)', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Expected Return')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Boxplot of expected returns by year
    ax4 = axes[1, 1]

    idx = rolling_means.index

    if not isinstance(idx, pd.DatetimeIndex):
        # Coerce to datetimes, parsing tz offsets robustly by using utc=True
        coerced = pd.to_datetime(idx, errors="coerce", utc=True)

        # If coercion failed for any entries, show examples and raise so you can inspect source
        if coerced.isnull().any():
            bad_examples = list(idx[coerced.isnull()][:10])
            raise ValueError(
                "Some index values could not be parsed as datetimes. "
                f"Examples: {bad_examples}"
            )

        # coerced is tz-aware (UTC). Convert to tz-naive (drop tz info)
        rolling_means.index = coerced.tz_convert(None)
    else:
        # Already a DatetimeIndex: if tz-aware, convert to tz-naive
        if rolling_means.index.tz is not None:
            rolling_means.index = rolling_means.index.tz_convert(None)

    # Extracting years and building yearly_data and matching labels in lockstep
    years = rolling_means.index.year
    years_unique = sorted(set(years))
    yearly_data = []
    year_labels = []

    for year in years_unique:
        mask = (years == year)
        values = rolling_means[mask].values.flatten()
        values = values[~np.isnan(values)]
        if len(values) > 0:
            yearly_data.append(values)
            year_labels.append(str(year))   # convert to strings for plotting

    # Calling boxplot with matching labels
    if len(yearly_data) > 0:
        ax4.boxplot(yearly_data, tick_labels=year_labels)
    else:
        ax4.text(0.5, 0.5, "No yearly data available", ha='center')

    ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax4.set_title('Expected Returns by Year', fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Expected Return')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'expected_returns_analysis.png', dpi=300)
    print(f"\n>> Saved: expected_returns_analysis.png")
    plt.close()
    
    # Save summary
    summary = pd.DataFrame({
        'Metric': [
            'Mean Expected Return',
            'Median Expected Return',
            'Min Expected Return',
            'Max Expected Return',
            'Cross-Sectional Std (Mean)',
            'Extreme Highs (>100%)',
            'Extreme Lows (<-50%)',
        ],
        'Value': [
            rolling_means.mean().mean(),
            rolling_means.median().median(),
            rolling_means.min().min(),
            rolling_means.max().max(),
            cross_sectional_std.mean(),
            extreme_high,
            extreme_low,
        ]
    })
    
    summary.to_csv(config.RESULTS_DIR / 'expected_returns_summary.csv', index=False)
    print(">> Saved: expected_returns_summary.csv")
    
    return summary


def compare_bayesian_vs_historical():
    """Compare Bayesian-shrunk returns vs raw historical means."""
    
    print("\n" + "="*70)
    print("BAYESIAN SHRINKAGE EFFECT")
    print("="*70)
    
    returns_path = config.get_data_path("log_returns.csv", "processed")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    # Take a sample period
    sample_returns = returns.iloc[-252:]  # Last year
    
    # Historical mean
    hist_mean = sample_returns.mean().values * config.TRADING_DAYS_PER_YEAR
    
    # Grand mean (market)
    grand_mean = hist_mean.mean()
    
    # Bayesian shrinkage
    shrinkage = 0.2  # Typical value
    bayesian_mean = (1 - shrinkage) * hist_mean + shrinkage * grand_mean
    
    # Compare
    comparison = pd.DataFrame({
        'Asset': returns.columns,
        'Historical': hist_mean,
        'Bayesian': bayesian_mean,
        'Difference': hist_mean - bayesian_mean,
        'Pct_Change': (bayesian_mean - hist_mean) / np.abs(hist_mean) * 100
    })
    
    comparison = comparison.sort_values('Historical', ascending=False)
    
    print("\nTop 10 assets by historical expected return:")
    print(comparison.head(10)[['Asset', 'Historical', 'Bayesian', 'Difference']].to_string(index=False))
    
    print("\nEffect of Bayesian shrinkage:")
    print(f"  Historical mean range: [{hist_mean.min():.4f}, {hist_mean.max():.4f}]")
    print(f"  Bayesian mean range: [{bayesian_mean.min():.4f}, {bayesian_mean.max():.4f}]")
    print(f"  Range reduction: {(1 - (bayesian_mean.max() - bayesian_mean.min()) / (hist_mean.max() - hist_mean.min())) * 100:.1f}%")
    
    if bayesian_mean.max() - bayesian_mean.min() < hist_mean.max() - hist_mean.min():
        print("\n++ Bayesian shrinkage IS reducing extreme values")
        print("   Bayesian model should be more stable!")


def main():
    """Run expected returns analysis."""
    
    summary = analyse_expected_returns()
    compare_bayesian_vs_historical()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    extreme_returns = summary[summary['Metric'].str.contains('Extreme')]['Value'].sum()
    high_dispersion = summary[summary['Metric'] == 'Cross-Sectional Std (Mean)']['Value'].values[0] > 0.3
    
    if extreme_returns > 0 or high_dispersion:
        print("\n!!!  Issues detected: Extreme expected returns detected!")
        print("       This can cause concentrated portfolios")
    else:
        print("\n++ No major issues:")
        print("   'Historical mean seems to be in the normal (no extreme results).")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()