"""
Run Backtest

Execute complete backtest of all portfolio models.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import models and backtester
from portfolio_models import (
    MeanVariancePortfolio,
    ShrinkagePortfolio,
    BayesianPortfolio,
    RobustPortfolio,
    EqualWeightPortfolio
    # # Sentiment Risk Portfolio (not used in final thesis results)
    # SentimentRiskPortfolio
)
from backtesting import Backtester


def main():
    """Run complete backtest."""
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMISATION BACKTEST")
    print("="*70)
    
    # Load data
    logger.info("Loading return data...")
    returns_path = config.get_data_path("log_returns.csv", "processed")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    logger.info(f"Data shape: {returns.shape}")
    logger.info(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Initialise models
    logger.info("\nInitialising portfolio models...")
    
    models = {
        'Mean-Variance': MeanVariancePortfolio(
            risk_aversion=config.RISK_AVERSION_DEFAULT
        ),
        'Shrinkage': ShrinkagePortfolio(
            risk_aversion=config.RISK_AVERSION_DEFAULT
        ),
        'Bayesian': BayesianPortfolio(
            risk_aversion=config.RISK_AVERSION_DEFAULT
        ),
        'Robust': RobustPortfolio(
            epsilon=config.ROBUST_EPSILON,
            risk_aversion=config.RISK_AVERSION_DEFAULT
        ),
        'Equal Weight': EqualWeightPortfolio()
        # # Sentiment Risk Portfolio (not used in final thesis results)
        # 'Sentiment Risk': SentimentRiskPortfolio(
        # base_risk_aversion=config.RISK_AVERSION_DEFAULT,
        # sentiment_sensitivity=1.5  
        # )
    }
    
    for name in models.keys():
        logger.info(f"  >> {name}")
    
    # Initialise backtester
    logger.info("\nInitialising backtester...")
    backtester = Backtester(
        returns=returns,
        models=models,
        estimation_window=config.ESTIMATION_WINDOW,
        rebalancing_freq=config.REBALANCING_FREQUENCY,
        transaction_cost=config.TRANSACTION_COST,
        initial_cash=1000000.0
    )
    
    # Run backtest
    print("\n" + "-"*70)
    print("RUNNING BACKTEST")
    print("-"*70)
    
    results = backtester.run_backtest(verbose=True)
    
    # Calculate and display metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    metrics = backtester.calculate_metrics()
    
    # Display key metrics
    key_metrics = [
        'annual_return',
        'annual_volatility',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'calmar_ratio',
        'avg_turnover'
    ]
    
    print("\n", metrics[key_metrics].round(4))
    
    # Display additional info
    print("\n" + "-"*70)
    print("ADDITIONAL STATISTICS")
    print("-"*70)
    
    for model_name in models.keys():
        print(f"\n{model_name}:")
        rets = results['returns'][model_name]
        print(f"  Total Return:    {((1 + rets).prod() - 1) * 100:.2f}%")
        print(f"  Win Rate:        {(rets > 0).sum() / len(rets) * 100:.2f}%")
        print(f"  Best Day:        {rets.max() * 100:.2f}%")
        print(f"  Worst Day:       {rets.min() * 100:.2f}%")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    backtester.save_results()
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()