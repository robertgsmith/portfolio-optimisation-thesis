# Portfolio weight correlation analysis between portfolios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config


def calculate_weight_correlations():
    """Calculate correlations between model weights."""
    
    print("\n" + "="*70)
    print("PORTFOLIO WEIGHT CORRELATION ANALYSIS")
    print("="*70)
    
    weights_dir = config.RESULTS_DIR / "weights"
    
    # Load all weight matrices
    weight_matrices = {}
    
    for model_file in weights_dir.glob("weights_*.csv"):
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        weight_matrices[model_name] = weights
    
    # Calculate average weight correlation across time
    models = list(weight_matrices.keys())
    n_models = len(models)
    
    corr_matrix = np.zeros((n_models, n_models))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            weights1 = weight_matrices[model1].values.flatten()
            weights2 = weight_matrices[model2].values.flatten()
            
            corr_matrix[i, j] = np.corrcoef(weights1, weights2)[0, 1]
    
    corr_df = pd.DataFrame(corr_matrix, index=models, columns=models)
    
    print("\nWeight Correlation Matrix:")
    print(corr_df.round(4))
    
    # Save
    corr_df.to_csv(config.RESULTS_DIR / 'weight_correlations.csv')
    print(f"\n>> Saved: weight_correlations.csv")
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                vmin=0, vmax=1, square=True, linewidths=1, ax=ax,
                cbar_kws={'label': 'Correlation'})
    
    ax.set_title('Portfolio Weight Correlations Between Models', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'weight_correlation_heatmap.png', dpi=300)
    print(">> Saved: weight_correlation_heatmap.png")
    plt.close()
    
    return corr_df


def analyse_weight_differences():
    """Analyse typical differences in weights."""
    
    print("\n" + "="*70)
    print("WEIGHT DIFFERENCE ANALYSIS")
    print("="*70)
    
    weights_dir = config.RESULTS_DIR / "weights"
    
    # Load weights
    weight_matrices = {}
    for model_file in weights_dir.glob("weights_*.csv"):
        model_name = model_file.stem.replace('weights_', '').replace('_', ' ').title()
        weights = pd.read_csv(model_file, index_col=0, parse_dates=True)
        weight_matrices[model_name] = weights
    
    # Compare robust methods vs Mean-Variance
    if 'Mean-Variance' in weight_matrices:
        baseline_weights = weight_matrices['Mean-Variance']
        
        comparisons = ['Shrinkage', 'Bayesian', 'Robust']
        
        results = []
        
        for model in comparisons:
            if model in weight_matrices:
                model_weights = weight_matrices[model]
                
                # Calculate differences
                weight_diff = (baseline_weights - model_weights).abs()
                
                results.append({
                    'Comparison': f'MV vs {model}',
                    'Mean Absolute Diff': weight_diff.values.mean(),
                    'Median Absolute Diff': np.median(weight_diff.values),
                    'Max Absolute Diff': weight_diff.values.max(),
                    '% Rebalances with >5% diff': (weight_diff.max(axis=1) > 0.05).sum() / len(weight_diff) * 100
                })
                
                print(f"\nMean-Variance vs {model}:")
                print(f"  Mean absolute difference: {weight_diff.values.mean():.4f}")
                print(f"  Max difference: {weight_diff.values.max():.4f}")
                print(f"  % rebalances with >5% difference: {(weight_diff.max(axis=1) > 0.05).sum() / len(weight_diff) * 100:.1f}%")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(config.RESULTS_DIR / 'weight_differences.csv', index=False)
        print(f"\n>> Saved: weight_differences.csv")


def main():
    """Run weight correlation analysis."""
    
    corr_df = calculate_weight_correlations()
    analyse_weight_differences()
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("\nWeight Correlations:")
    print("  > 0.95: Models are nearly identical (problematic)")
    print("  0.80-0.95: High similarity (expected for similar methods)")
    print("  < 0.80: Distinct portfolios (good differentiation)")
    print("\nIf correlations > 0.95:")
    print("  - Models may be converging to similar solutions")
    print("  - This explains identical turnover")
    print("  - Still valid, but limits contribution of robustness methods")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()