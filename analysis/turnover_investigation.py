"""
Enhanced Turnover Analysis

Investigate why models show identical/similar turnover.

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


def analyse_turnover_deeply():
    """Deep dive into turnover characteristics."""
    
    print("\n" + "="*70)
    print("ENHANCED TURNOVER ANALYSIS")
    print("="*70)
    
    weights_dir = config.RESULTS_DIR / "weights"
    
    results = []
    
    for model_file in weights_dir.glob("weights_*.csv"):
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        
        # Calculate weight changes
        weight_changes = weights.diff().abs()
        
        # Turnover at each rebalancing
        turnover_series = weight_changes.sum(axis=1).dropna()
        
        # Weight stability metrics
        weight_std = weights.std(axis=0).mean()  # Average std across assets
        weight_range = (weights.max(axis=0) - weights.min(axis=0)).mean()
        
        # Concentration over time
        herfindahl = (weights ** 2).sum(axis=1)
        
        # Number of position changes (weights that changed by > 1%)
        significant_changes = (weight_changes > 0.01).sum(axis=1).mean()
        
        results.append({
            'Model': model_name,
            'Mean Turnover': turnover_series.mean(),
            'Median Turnover': turnover_series.median(),
            'Std Turnover': turnover_series.std(),
            'Max Turnover': turnover_series.max(),
            'Min Turnover': turnover_series.min(),
            'Weight Volatility': weight_std,
            'Weight Range': weight_range,
            'Avg Herfindahl': herfindahl.mean(),
            'Avg Assets Changed >1%': significant_changes
        })
        
        print(f"\n{model_name}:")
        print(f"  Mean turnover: {turnover_series.mean():.4f}")
        print(f"  Weight volatility (avg std): {weight_std:.4f}")
        print(f"  Avg assets with >1% change: {significant_changes:.1f}")
        print(f"  Concentration (Herfindahl): {herfindahl.mean():.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_DIR / 'turnover_detailed.csv', index=False)
    print(f"\n>> Saved: turnover_detailed.csv")
    
    return results_df


def plot_turnover_time_series():
    """Plot turnover over time for each model."""
    
    weights_dir = config.RESULTS_DIR / "weights"
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Turnover over time
    ax1 = axes[0]
    
    for model_file in weights_dir.glob("weights_*.csv"):
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        
        turnover = weights.diff().abs().sum(axis=1).dropna()
        
        if model_name != 'Equal Weight':  # Skip equal weight (0 turnover)
            ax1.plot(turnover.index, turnover.values, label=model_name, alpha=0.7)
    
    ax1.set_title('Portfolio Turnover Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Turnover')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight stability (Herfindahl index)
    ax2 = axes[1]
    
    for model_file in weights_dir.glob("weights_*.csv"):
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        
        herfindahl = (weights ** 2).sum(axis=1)
        ax2.plot(herfindahl.index, herfindahl.values, label=model_name, alpha=0.7)
    
    ax2.set_title('Portfolio Concentration (Herfindahl Index)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Herfindahl Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'turnover_and_concentration.png', dpi=300)
    print(">> Saved: turnover_and_concentration.png")
    plt.close()


def compare_weight_distributions():
    """Compare weight distributions across models."""
    
    weights_dir = config.RESULTS_DIR / "weights"
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, model_file in enumerate(weights_dir.glob("weights_*.csv")):
        if idx >= 6:
            break
            
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        
        # Flatten all weights
        all_weights = weights.values.flatten()
        all_weights = all_weights[all_weights > 0.001]  # Remove near-zero weights
        
        axes[idx].hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].axvline(config.MAX_WEIGHT, color='red', linestyle='--', 
                         label=f'Max ({config.MAX_WEIGHT:.0%})')
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Weight')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(list(weights_dir.glob("weights_*.csv"))), 6):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'weight_distributions.png', dpi=300)
    print(">> Saved: weight_distributions.png")
    plt.close()


def main():
    """Run enhanced turnover analysis."""
    
    print("\n" + "="*70)
    print("INVESTIGATING TURNOVER PATTERNS")
    print("="*70)
    
    results = analyse_turnover_deeply()
    
    print("\n" + "="*70)
    print("CREATING VISUALISATIONS")
    print("="*70)
    
    plot_turnover_time_series()
    compare_weight_distributions()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Questions to Answer:")
    print("  1. Do models show different turnover patterns over time?")
    print("  2. Do weight distributions differ across models?")
    print("  3. Is concentration (Herfindahl) different?")
    print("\nIf turnover is still identical, possible explanations:")
    print("  - Constraints are binding (10% max weight)")
    print("  - S&P 100 is highly correlated (limited diversification)")
    print("  - Monthly rebalancing dominates model differences")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()