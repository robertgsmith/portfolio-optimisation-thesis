"""
DATA PIPELINE FOR S&P 100 PORTFOLIO OPTIMIZATION
================================================================

This file contains the the run commands the entire pipeline.

Authors: Robert George Smith & Joaquin Rodriguez
Project: Robust Portfolio Optimisation Under Parameter Uncertainty
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import warnings

from data_components.data_downloader import DataDownloader
from data_components.data_processor import DataProcessor
from data_components.feature_engineer import FeatureEngineer
from data_components.summary_statistics import SummaryStatistics

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline():
    """Execute the complete data preparation pipeline."""
    
    logger.info("="*70)
    logger.info("STARTING COMPLETE DATA PIPELINE")
    logger.info("="*70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Download and consolidate data
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 1: DOWNLOADING DATA ###")
    
    downloader = DataDownloader(start_date="2010-01-01", end_date="2024-12-31")
    prices_df, volume_df, date_coverage = downloader.download_all_tickers()
    
    # Determine optimal date range
    optimal_start, available_tickers = downloader.determine_common_date_range(
        date_coverage,
        min_coverage_threshold=0.90
    )
    
    # Filter and save
    filtered_prices = downloader.filter_and_save(
        prices_df,
        volume_df,
        optimal_start,
        available_tickers
    )
    
    # -------------------------------------------------------------------------
    # STEP 2: Process data
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 2: PROCESSING DATA ###")
    
    processor = DataProcessor()
    
    # Compute returns
    log_returns = processor.compute_returns(filtered_prices, return_type="log")
    simple_returns = processor.compute_returns(filtered_prices, return_type="simple")
    
    processor.save_processed_data(log_returns, "log_returns.csv")
    processor.save_processed_data(simple_returns, "simple_returns.csv")
    
    # Compute rolling statistics
    rolling_stats = processor.compute_rolling_statistics(log_returns)
    for name, data in rolling_stats.items():
        processor.save_processed_data(data, f"{name}.csv")
    
    # Compute momentum signals
    momentum_signals = processor.compute_momentum_signals(filtered_prices)
    for name, data in momentum_signals.items():
        processor.save_processed_data(data, f"{name}.csv")
    
    # Compute covariance matrices
    cov_matrices = processor.compute_covariance_matrices(log_returns)
    for name, data in cov_matrices.items():
        processor.save_processed_data(data, f"{name}.csv")

    # -------------------------------------------------------------------------
    # STEP 3: Feature engineering
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 3: FEATURE ENGINEERING ###")
    
    engineer = FeatureEngineer()
    
    # Create expected return estimates
    return_estimates = engineer.create_expected_return_estimates(
        log_returns,
        filtered_prices
    )
    for name, data in return_estimates.items():
        engineer.save_features(data, f"expected_returns_{name}.csv")
    
    # Create market features
    market_features = engineer.create_market_features(log_returns, filtered_prices)
    engineer.save_features(market_features, "market_features.csv")
    
    # -------------------------------------------------------------------------
    # STEP 4: Summary statistics
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 4: COMPUTING SUMMARY STATISTICS ###")
    
    summariser = SummaryStatistics()
    
    # Return statistics
    return_stats = summariser.compute_return_statistics(log_returns)
    summariser.save_statistics(return_stats, "return_statistics.csv")
    
    # Correlation analysis
    corr_matrix, cov_matrix = summariser.compute_correlation_analysis(log_returns)
    summariser.save_statistics(corr_matrix, "correlation_matrix.csv")
    summariser.save_statistics(cov_matrix, "covariance_matrix.csv")
    
    # -------------------------------------------------------------------------
    # DATA PREPARATION PIPELINE COMPLETE
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70)
    logger.info("DATA PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Final dataset shape: {filtered_prices.shape}")
    print(f"Number of assets: {len(filtered_prices.columns)}")
    print(f"Date range: {filtered_prices.index[0].date()} to {filtered_prices.index[-1].date()}")
    print(f"Trading days: {len(filtered_prices)}")
    print(f"\nTop 10 assets by Sharpe ratio:")
    print(return_stats.nlargest(10, 'sharpe_ratio')[['ann_mean', 'ann_vol', 'sharpe_ratio']])
    print("\nData saved to:")
    print("  - data/raw/")
    print("  - data/processed/")
    print("  - data/features/")
    print("  - data/analysis/")
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_complete_pipeline()