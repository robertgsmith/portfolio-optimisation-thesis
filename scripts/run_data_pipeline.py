# Run data pipeline (download and prepare data)

import logging
import sys
from pathlib import Path

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

# Configure logging from config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

from data_pipeline import (
    DataDownloader,
    DataProcessor,
    FeatureEngineer,
    SummaryStatistics
)

def run_complete_pipeline():
    """Execute the complete data preparation pipeline."""
    
    # Print configuration summary
    config.print_config_summary()

    logger.info("="*70)
    logger.info("STARTING COMPLETE DATA PIPELINE")
    logger.info("="*70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Download and consolidate data
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 1: DOWNLOADING DATA ###")
    
    downloader = DataDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )   
    prices_df, volume_df, date_coverage = downloader.download_all_tickers()
    
    # Determine optimal date range
    optimal_start, available_tickers = downloader.determine_common_date_range(
        date_coverage,
        min_coverage_threshold=config.MIN_COVERAGE_THRESHOLD
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
    summariser = SummaryStatistics()

    # Compute returns
    # log_returns = processor.compute_returns(filtered_prices, return_type="log")
    log_returns = processor.compute_returns(filtered_prices, return_type=config.RETURN_TYPE)
    simple_returns = processor.compute_returns(filtered_prices, return_type="simple")

    processor.save_processed_data(log_returns, "log_returns.csv")
    processor.save_processed_data(simple_returns, "simple_returns.csv")

    # Display dataset characteristics
    summariser.return_further_data_info(log_returns)

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
    
    # Return statistics (summariser created earlier)
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
    asset_count = len(filtered_prices.columns)
    day_count = len(filtered_prices)
    first_date = filtered_prices.index[0].date()
    last_date = filtered_prices.index[-1].date()

    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Final dataset shape: {filtered_prices.shape}")
    print(f"Number of assets: {asset_count}")
    print(f"Date range: {first_date} to {last_date}")
    print(f"Trading days: {day_count}")
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