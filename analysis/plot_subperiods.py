"""
Plot Sub-Period Performance

Visualise performance during different market regimes.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_all_subperiods():
    """Plot performance across all sub-periods."""
    
    print("\n" + "="*70)
    print("ALL SUB-PERIODS ANALYSIS")
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
        'Pre-COVID (2015-2019)': ('2015-01-01', '2019-12-31'),
        'COVID Era (2020-2021)': ('2020-01-01', '2021-12-31'),
        'Post-COVID (2022-2024)': ('2022-01-01', '2024-12-31')
    }
    
    # Create subplots
    fig, axes = plt.subplots(len(periods), 1, figsize=(14, 12))
    
    for idx, (period_name, (start, end)) in enumerate(periods.items()):
        try:
            period_returns = returns.loc[start:end]
            
            if len(period_returns) == 0:
                print(f"!!!  No data for {period_name}")
                continue
            
            # Cumulative returns
            cum_returns = (1 + period_returns).cumprod()
            
            # Plot
            ax = axes[idx] if len(periods) > 1 else axes
            
            for col in cum_returns.columns:
                ax.plot(cum_returns.index, cum_returns[col], label=col, linewidth=2)
            
            ax.set_title(f'{period_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            print(f">> {period_name}: {len(period_returns)} days")
            
        except Exception as e:
            print(f"!!!  Error plotting {period_name}: {e}")
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'subperiod_performance.png', dpi=300)
    print("\n>> Saved: subperiod_performance.png")
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