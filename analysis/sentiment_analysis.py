# Sentiment analysis evaluate impact on portfolio performance

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def compare_sentiment_vs_baseline():
    """Compare sentiment portfolio vs baseline."""
    
    print("\n" + "="*70)
    print("SENTIMENT PORTFOLIO ANALYSIS")
    print("="*70)
    
    # Load results
    returns = pd.read_csv(config.RESULTS_DIR / 'backtest_returns.csv', 
                         index_col=0, parse_dates=True)
    metrics = pd.read_csv(config.RESULTS_DIR / 'backtest_metrics.csv',
                         index_col=0)
    
    # Check if sentiment portfolio exists
    if 'Sentiment Risk' not in returns.columns:
        print("\n!!!  Sentiment Risk portfolio not found in results")
        print("   Did you run backtest with sentiment model?")
        return
    
    # Compare metrics
    print("\n" + "-"*70)
    print("PERFORMANCE COMPARISON")
    print("-"*70)
    
    comparison = pd.DataFrame({
        'Mean-Variance': metrics.loc['Mean-Variance'],
        'Sentiment Risk': metrics.loc['Sentiment Risk'],
        'Difference': metrics.loc['Sentiment Risk'] - metrics.loc['Mean-Variance']
    }).T
    
    key_metrics = ['sharpe_ratio', 'annual_return', 'annual_volatility', 
                   'max_drawdown', 'avg_turnover']
    
    print(comparison[key_metrics].round(4))
    
    # Statistical test
    print("\n" + "-"*70)
    print("STATISTICAL SIGNIFICANCE")
    print("-"*70)
    
    mv_returns = returns['Mean-Variance']
    sent_returns = returns['Sentiment Risk']
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(sent_returns, mv_returns)
    
    print(f"\nMean return difference:")
    print(f"  Sentiment Risk: {sent_returns.mean()*252:.4f}")
    print(f"  Mean-Variance: {mv_returns.mean()*252:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  ++ Statistically significant at 5% level")
    else:
        print("  !!! Not statistically significant")
    
    # Bootstrap Sharpe ratio test
    print("\nBootstrap Sharpe ratio test (10,000 iterations)...")
    
    sharpe_diffs = []
    n = len(mv_returns)
    
    for _ in range(10000):
        idx = np.random.choice(n, size=n, replace=True)
        s1 = sent_returns.iloc[idx]
        s2 = mv_returns.iloc[idx]
        
        sharpe1 = s1.mean() / s1.std() * np.sqrt(252) if s1.std() > 0 else 0
        sharpe2 = s2.mean() / s2.std() * np.sqrt(252) if s2.std() > 0 else 0
        
        sharpe_diffs.append(sharpe1 - sharpe2)
    
    sharpe_diffs = np.array(sharpe_diffs)
    p_value_bootstrap = 2 * min(np.mean(sharpe_diffs >= 0), np.mean(sharpe_diffs <= 0))
    
    print(f"  95% CI: [{np.percentile(sharpe_diffs, 2.5):.4f}, {np.percentile(sharpe_diffs, 97.5):.4f}]")
    print(f"  p-value: {p_value_bootstrap:.4f}")
    
    if p_value_bootstrap < 0.05:
        print("  ++ Statistically significant at 5% level")
    else:
        print("  !!! Not statistically significant")


def analyse_by_regime():
    """Analyse performance by Fed policy regime."""
    
    print("\n" + "="*70)
    print("PERFORMANCE BY POLICY REGIME")
    print("="*70)
    
    # Load data
    returns = pd.read_csv(config.RESULTS_DIR / 'backtest_returns.csv',
                         index_col=0, parse_dates=True)
    
    if 'Sentiment Risk' not in returns.columns:
        print("\n!!!  Sentiment Risk not found. Skipping regime analysis.")
        return
    
    # Load policy factor
    policy_path = config.SENTIMENT_DIR / 'monetary_policy_factor.csv'
    
    if not policy_path.exists():
        print("\n!!!  Policy factor not found. Skipping regime analysis.")
        return
    
    policy = pd.read_csv(policy_path, index_col=0, parse_dates=True)
    
    # Remove timezone from both if present
    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    if hasattr(policy.index, 'tz') and policy.index.tz is not None:
        policy.index = policy.index.tz_localize(None)
    
    # Merge on date
    combined = returns.join(policy['Policy_Tone'], how='inner')
    
    if len(combined) == 0:
        print("\n!!!  No overlapping dates between returns and policy data")
        return
    
    print(f"\nMerged {len(combined)} days of data")
    
    # Define regimes
    combined['Regime'] = 'Neutral'
    combined.loc[combined['Policy_Tone'] > 0.3, 'Regime'] = 'Dovish'
    combined.loc[combined['Policy_Tone'] < -0.3, 'Regime'] = 'Hawkish'
    
    # Calculate metrics by regime
    results = []
    
    for regime in ['Dovish', 'Neutral', 'Hawkish']:
        regime_data = combined[combined['Regime'] == regime]
        
        if len(regime_data) == 0:
            continue
        
        for model in ['Mean-Variance', 'Sentiment Risk']:
            if model not in regime_data.columns:
                continue
            
            model_returns = regime_data[model]
            
            sharpe = 0
            if model_returns.std() > 0:
                sharpe = (model_returns.mean() / model_returns.std() * np.sqrt(252))
            
            results.append({
                'Regime': regime,
                'Model': model,
                'Days': len(model_returns),
                'Ann_Return': model_returns.mean() * 252,
                'Ann_Vol': model_returns.std() * np.sqrt(252),
                'Sharpe': sharpe
            })
    
    if not results:
        print("\n!!!  No results to display")
        return
    
    results_df = pd.DataFrame(results)
    
    print("\nSharpe Ratios by Regime:")
    pivot = results_df.pivot(index='Model', columns='Regime', values='Sharpe')
    print(pivot.round(3))
    
    print("\nDays in Each Regime:")
    regime_counts = combined['Regime'].value_counts()
    print(regime_counts)
    
    # Save
    results_df.to_csv(config.RESULTS_DIR / 'sentiment_by_regime.csv', index=False)
    print(f"\n>> Saved: sentiment_by_regime.csv")

###


def plot_cumulative_comparison():
    """Plot cumulative returns comparison."""
    
    returns = pd.read_csv(config.RESULTS_DIR / 'backtest_returns.csv',
                         index_col=0, parse_dates=True)
    
    if 'Sentiment Risk' not in returns.columns:
        return
    
    # Cumulative returns
    cum_returns = (1 + returns[['Mean-Variance', 'Sentiment Risk']]).cumprod()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(cum_returns.index, cum_returns['Mean-Variance'], 
            label='Mean-Variance (Baseline)', linewidth=2, color='steelblue')
    ax.plot(cum_returns.index, cum_returns['Sentiment Risk'], 
            label='Sentiment Risk Management', linewidth=2, color='darkgreen')
    
    ax.set_title('Cumulative Performance: Sentiment vs Baseline', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'sentiment_comparison.png', dpi=300)
    print(">> Saved: sentiment_comparison.png")
    plt.close()


def main():
    """Run sentiment analysis."""
    
    compare_sentiment_vs_baseline()
    analyse_by_regime()
    plot_cumulative_comparison()
    
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Questions:")
    print("  1. Does sentiment improve Sharpe ratio? (Statistical test)")
    print("  2. Does it work better in certain regimes? (Regime analysis)")
    print("  3. Is the improvement economically significant? (Returns plot)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()