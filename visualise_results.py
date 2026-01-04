"""
Visualise Backtest Results

Create comprehensive visualisations for thesis.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
import config

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figures directory
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_cumulative_returns():
    """Plot cumulative returns for all models."""
    
    returns = pd.read_csv(config.RESULTS_DIR / "backtest_returns.csv", 
                         index_col=0, parse_dates=True)
    
    cum_returns = (1 + returns).cumprod() - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in cum_returns.columns:
        ax.plot(cum_returns.index, cum_returns[col], label=col, linewidth=2)
    
    ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'cumulative_returns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cumulative_returns.png")
    plt.close()


def plot_drawdowns():
    """Plot drawdowns for all models."""
    
    drawdowns = pd.read_csv(config.RESULTS_DIR / "backtest_drawdowns.csv",
                           index_col=0, parse_dates=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[col] * 100, label=col, linewidth=2)
    
    ax.fill_between(drawdowns.index, 0, drawdowns.min(axis=1) * 100, 
                     alpha=0.3, color='red')
    
    ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'drawdowns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: drawdowns.png")
    plt.close()


def plot_rolling_sharpe():
    """Plot rolling Sharpe ratios."""
    
    returns = pd.read_csv(config.RESULTS_DIR / "backtest_returns.csv",
                         index_col=0, parse_dates=True)
    
    # Calculate rolling Sharpe (252-day window)
    rolling_sharpe = returns.rolling(window=252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in rolling_sharpe.columns:
        ax.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Rolling Sharpe Ratio (252-day)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'rolling_sharpe.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: rolling_sharpe.png")
    plt.close()


def plot_performance_metrics():
    """Create bar chart of key performance metrics."""
    
    metrics = pd.read_csv(config.RESULTS_DIR / "backtest_metrics.csv", index_col=0)
    
    key_metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(key_metrics):
        metrics[metric].plot(kind='bar', ax=axes[i], color='steelblue')
        axes[i].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Value', fontsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_metrics.png")
    plt.close()


def plot_risk_return_scatter():
    """Create risk-return scatter plot."""
    
    metrics = pd.read_csv(config.RESULTS_DIR / "backtest_metrics.csv", index_col=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(metrics['annual_volatility'] * 100, 
                        metrics['annual_return'] * 100,
                        s=200, alpha=0.6, c=metrics['sharpe_ratio'],
                        cmap='RdYlGn', edgecolors='black', linewidth=1.5)
    
    # Add labels
    for idx, row in metrics.iterrows():
        ax.annotate(idx, 
                   (row['annual_volatility'] * 100, row['annual_return'] * 100),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=10)
    
    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annual Volatility (%)', fontsize=12)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: risk_return_scatter.png")
    plt.close()


def plot_weight_evolution():
    """Plot weight evolution over time for one model."""
    
    # Use Mean-Variance as example
    weights = pd.read_csv(config.RESULTS_DIR / "weights" / "weights_mean-variance.csv",
                         index_col=0, parse_dates=True)
    
    # Get top 10 assets by average weight
    avg_weights = weights.mean().sort_values(ascending=False).head(10)
    top_assets = avg_weights.index
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weights[top_assets].plot(ax=ax, linewidth=1.5)
    
    ax.set_title('Portfolio Weight Evolution - Mean-Variance (Top 10 Assets)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'weight_evolution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: weight_evolution.png")
    plt.close()


def plot_turnover_comparison():
    """Compare average turnover across models."""
    
    metrics = pd.read_csv(config.RESULTS_DIR / "backtest_metrics.csv", index_col=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics['avg_turnover'].plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Average Portfolio Turnover', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Turnover per Rebalancing', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'turnover_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: turnover_comparison.png")
    plt.close()
