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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_complete_pipeline()