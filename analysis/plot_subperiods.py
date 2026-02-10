# Plot sub-period performance

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Consistent color scheme
COLORS = {
    'Mean-Variance': '#1f77b4',
    'Shrinkage': '#ff7f0e',
    'Bayesian': '#2ca02c',
    'Robust': '#d62728',
    'Equal Weight': '#7f7f7f'
}

OPT_METHODS = ['Mean-Variance', 'Shrinkage', 'Bayesian', 'Robust']


def plot_all_subperiods():
    """Plot sub-period performance - 3x4 grid (3 periods × 4 comparisons)."""
    
    print("\n" + "="*70)
    print("ALL SUB-PERIODS ANALYSIS")
    print("="*70)
    
    # Load returns
    returns = pd.read_csv(
        config.RESULTS_DIR / "backtest_returns.csv",
        index_col=0,
        parse_dates=[0]
    )

    # Coerce to datetime, set UTC, then drop tz
    returns.index = pd.to_datetime(returns.index, errors="raise", utc=True).tz_convert(None)
    
    # Define periods
    periods = [
        ('Pre-COVID (2015-2019)', '2015-01-01', '2019-12-31'),
        ('COVID Era (2020-2021)', '2020-01-01', '2021-12-31'),
        ('Post-COVID (2022-2024)', '2022-01-01', '2024-12-31')
    ]
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    for row_idx, (period_name, start, end) in enumerate(periods):
        try:
            period_returns = returns.loc[start:end]
            
            if len(period_returns) == 0:
                print(f"!!!  No data for {period_name}")
                continue
            
            # Cumulative returns
            cum_returns = (1 + period_returns).cumprod()
            
            # Plot each comparison in columns
            for col_idx, method in enumerate(OPT_METHODS):
                ax = axes[row_idx, col_idx]
                
                # Plot optimisation method
                ax.plot(cum_returns.index, cum_returns[method], 
                       label=method, linewidth=2, color=COLORS[method])
                
                # Plot Equal Weight benchmark
                ax.plot(cum_returns.index, cum_returns['Equal Weight'], 
                       label='Equal Weight', linewidth=2, color=COLORS['Equal Weight'],
                       linestyle='--', alpha=0.7)
                
                # Title only on top row
                if row_idx == 0:
                    ax.set_title(f'{method} vs Equal Weight', fontsize=11, fontweight='bold')
                
                # Period label only on left column
                if col_idx == 0:
                    ax.set_ylabel(f'{period_name}\nCumulative Return', fontsize=9)
                else:
                    ax.set_ylabel('')
                
                # X-label only on bottom row
                if row_idx == len(periods) - 1:
                    ax.set_xlabel('Date', fontsize=9)
                else:
                    ax.set_xlabel('')
                
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
            
            print(f">> {period_name}: {len(period_returns)} days")
            
        except Exception as e:
            print(f"!!!  Error plotting {period_name}: {e}")
    
    plt.suptitle('Sub-Period Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'subperiod_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\n>> Saved: subperiod_performance_comparison.png")
    plt.close()


def calculate_subperiod_metrics():
    """Calculate metrics for each sub-period."""
    
    print("\n" + "="*70)
    print("SUB-PERIOD METRICS")
    print("="*70)
    
    # Load returns
    returns = pd.read_csv(
        config.RESULTS_DIR / "backtest_returns.csv",
        index_col=0,
        parse_dates=[0]
    )

    # Coerce to datetime (parse strings/objects), set UTC, then drop tz
    returns.index = pd.to_datetime(returns.index, errors="raise", utc=True).tz_convert(None)
    
    # Define periods
    periods = {
        'Pre-COVID': ('2015-01-01', '2019-12-31'),
        'COVID': ('2020-01-01', '2021-12-31'),
        'Post-COVID': ('2022-01-01', '2024-12-31'),
        'Full Period': (returns.index[0].strftime('%Y-%m-%d'), 
                       returns.index[-1].strftime('%Y-%m-%d'))
    }
    
    results = []
    
    for period_name, (start, end) in periods.items():
        try:
            period_returns = returns.loc[start:end]
            
            if len(period_returns) == 0:
                continue
            
            for model in returns.columns:
                model_returns = period_returns[model]
                
                # Calculate metrics
                ann_return = model_returns.mean() * 252
                ann_vol = model_returns.std() * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                
                # Max drawdown
                cum_returns = (1 + model_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                max_dd = drawdown.min()
                
                results.append({
                    'Period': period_name,
                    'Model': model,
                    'Annual Return': ann_return,
                    'Annual Volatility': ann_vol,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown': max_dd,
                    'Days': len(model_returns)
                })
        
        except Exception as e:
            print(f"!!!  Error calculating {period_name}: {e}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(config.RESULTS_DIR / 'subperiod_metrics.csv', index=False)
        print("\n>> Saved: subperiod_metrics.csv")
        
        # Display summary
        print("\nSharpe Ratios by Period:")
        pivot = results_df.pivot(index='Model', columns='Period', values='Sharpe Ratio')
        print(pivot.round(3))
    
    return results_df if results else None


def main():
    """Run all sub-period analyses."""
    
    plot_all_subperiods()
    metrics = calculate_subperiod_metrics()
    
    print("\n" + "="*70)
    print("SUB-PERIOD ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()