"""
Create Monetary Policy Factor

Download and process interest rate data as Fed policy proxy.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
import yfinance as yf
import pandas as pd
import numpy as np


def download_treasury_yields():
    """
    Download 10-Year Treasury yields as monetary policy proxy.
    
    Returns
    -------
    yields : pd.DataFrame
        Treasury yield data
    """
    print("\n" + "="*70)
    print("DOWNLOADING MONETARY POLICY DATA")
    print("="*70)
    
    print("\nDownloading 10-Year Treasury Yield (^TNX)...")
    
    # Download treasury yield
    # ^TNX = CBOE 10-Year Treasury Yield Index
    tnx = yf.download("^TNX", start="2010-01-01", end="2024-12-31", progress=False)
    
    print(f">> Downloaded {len(tnx)} days of data")
    print(f"  Date range: {tnx.index[0].date()} to {tnx.index[-1].date()}")

    close = tnx['Close']
    if hasattr(close, "iloc") and close.ndim > 1:
        close = close.iloc[:, 0]

    min_yield = float(close.min())
    max_yield = float(close.max())

    print(f"  Yield range: {min_yield:.2f}% to {max_yield:.2f}%")
    
    return tnx


def calculate_policy_tone(yields):
    """
    Calculate policy tone from yield changes.
    
    Interpretation:
    - Rising yields (positive change) = Hawkish policy = Negative for stocks
    - Falling yields (negative change) = Dovish policy = Positive for stocks
    
    Parameters
    ----------
    yields : pd.DataFrame
        Treasury yield data
    
    Returns
    -------
    policy : pd.DataFrame
        Policy tone indicators
    """
    print("\n" + "-"*70)
    print("CALCULATING POLICY TONE")
    print("-"*70)
    
    policy = pd.DataFrame(index=yields.index)
    
    # Method 1: Simple change (daily)
    policy['Yield_Level'] = yields['Close']
    policy['Yield_Change'] = yields['Close'].diff()
    
    # Policy tone: negative of yield change
    # (rising yields = tightening = negative tone)
    policy['Policy_Tone_Raw'] = -policy['Yield_Change']
    
    # Method 2: Smoothed change (21-day moving average)
    policy['Yield_Change_MA21'] = policy['Yield_Change'].rolling(21).mean()
    policy['Policy_Tone_Smoothed'] = -policy['Yield_Change_MA21']
    
    # Method 3: Standardized (z-score)
    # This makes it easier to interpret: -1 to +1 range
    rolling_mean = policy['Yield_Change'].rolling(252).mean()
    rolling_std = policy['Yield_Change'].rolling(252).std()
    policy['Policy_Tone_Zscore'] = -(policy['Yield_Change'] - rolling_mean) / rolling_std
    
    # Clip z-score to [-3, 3] to avoid extreme values
    policy['Policy_Tone_Zscore'] = policy['Policy_Tone_Zscore'].clip(-3, 3)
    
    # Normalize to [-1, 1] for easier interpretation
    policy['Policy_Tone'] = policy['Policy_Tone_Zscore'] / 3.0
    
    print("\nPolicy Tone Statistics:")
    print(f"  Raw tone (bp/day): mean={policy['Policy_Tone_Raw'].mean():.4f}, std={policy['Policy_Tone_Raw'].std():.4f}")
    print(f"  Smoothed tone: mean={policy['Policy_Tone_Smoothed'].mean():.4f}, std={policy['Policy_Tone_Smoothed'].std():.4f}")
    print(f"  Normalized tone: mean={policy['Policy_Tone'].mean():.4f}, std={policy['Policy_Tone'].std():.4f}")
    print(f"  Range: [{policy['Policy_Tone'].min():.2f}, {policy['Policy_Tone'].max():.2f}]")
    
    # Add regime indicator
    policy['Regime'] = 'Neutral'
    policy.loc[policy['Policy_Tone'] > 0.3, 'Regime'] = 'Dovish'
    policy.loc[policy['Policy_Tone'] < -0.3, 'Regime'] = 'Hawkish'
    
    print("\nPolicy Regime Distribution:")
    print(policy['Regime'].value_counts())
    
    return policy


def save_policy_factor(policy):
    """Save policy factor to CSV."""
    
    output_path = config.SENTIMENT_DIR / 'monetary_policy_factor.csv'
    config.SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    policy.to_csv(output_path)
    
    print("\n" + "="*70)
    print("SAVED MONETARY POLICY FACTOR")
    print("="*70)
    print(f"Location: {output_path}")
    print("\nColumns:")
    for col in policy.columns:
        print(f"  - {col}")
    
    print("\nUsage in portfolio optimization:")
    print("  Policy_Tone > 0: Dovish (accommodative) → Risk-on")
    print("  Policy_Tone < 0: Hawkish (restrictive) → Risk-off")
    print("="*70 + "\n")


def plot_policy_factor(policy):
    """Create visualization of policy factor."""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Yield levels
        ax1 = axes[0]
        ax1.plot(policy.index, policy['Yield_Level'], linewidth=1.5, color='steelblue')
        ax1.set_title('10-Year Treasury Yield', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Yield (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Policy tone (raw)
        ax2 = axes[1]
        ax2.plot(policy.index, policy['Policy_Tone_Raw'], linewidth=0.8, alpha=0.5, color='gray', label='Daily')
        ax2.plot(policy.index, policy['Policy_Tone_Smoothed'], linewidth=2, color='orange', label='21-day MA')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Policy Tone (Raw)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Tone (bp change)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Normalized policy tone
        ax3 = axes[2]
        ax3.plot(policy.index, policy['Policy_Tone'], linewidth=1.5, color='darkgreen')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.axhline(y=0.3, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Dovish threshold')
        ax3.axhline(y=-0.3, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Hawkish threshold')
        ax3.fill_between(policy.index, 0, policy['Policy_Tone'], 
                         where=policy['Policy_Tone']>0, alpha=0.3, color='green', label='Dovish')
        ax3.fill_between(policy.index, 0, policy['Policy_Tone'], 
                         where=policy['Policy_Tone']<0, alpha=0.3, color='red', label='Hawkish')
        ax3.set_title('Normalized Policy Tone', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Tone [-1, 1]')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_path = config.FIGURES_DIR / 'monetary_policy_factor.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f">> Saved visualization: {fig_path}")
        plt.close()
        
    except Exception as e:
        print(f"!!!  Could not create visualization: {e}")


def main():
    """Create monetary policy factor."""
    
    # Download data
    yields = download_treasury_yields()
    
    # Calculate policy tone
    policy = calculate_policy_tone(yields)
    
    # Save
    save_policy_factor(policy)
    
    # Visualize
    plot_policy_factor(policy)
    
    print("\n>> Monetary policy factor ready for use in portfolio optimisation")


if __name__ == "__main__":
    main()