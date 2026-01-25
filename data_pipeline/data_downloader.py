"""
Data Downloader Module

Downloads and consolidates S&P 100 stock data from Yahoo Finance.
"""

# Standard library imports
import sys
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple

# Third-party imports
import yfinance as yf
import pandas as pd

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

# Configure logging from config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Download and consolidate S&P 100 stock data from Yahoo Finance."""
    
    # S&P 100 tickers from config
    SP100_TICKERS = config.SP100_TICKERS
    
    def __init__(
        self,
        start_date: str = config.START_DATE,
        end_date: str = config.END_DATE
    ):
        """
        Initialise the data downloader.
        
        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = config.DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialised downloader for period {start_date} to {end_date}")
    
    def download_all_tickers(
        self,
        tickers: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
        """
        Download all ticker data and consolidate into single DataFrames.
        
        Parameters
        ----------
        tickers : List[str], optional
            List of tickers to download. Uses SP100_TICKERS if None.
        
        Returns
        -------
        prices_df : pd.DataFrame
            Adjusted close prices for all tickers
        volume_df : pd.DataFrame
            Trading volumes for all tickers
        date_coverage : Dict[str, pd.Timestamp]
            First available date for each ticker
        """
        if tickers is None:
            tickers = self.SP100_TICKERS
        
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        
        all_prices = {}
        all_volumes = {}
        date_coverage = {}
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"[{i}/{len(tickers)}] Downloading {ticker}...")
                
                # Use yfinance Ticker object for better compatibility
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True  # This gives us adjusted prices
                )
                
                if not data.empty and 'Close' in data.columns:
                    all_prices[ticker] = data['Close']
                    all_volumes[ticker] = data['Volume']
                    date_coverage[ticker] = data.index[0]
                    logger.info(f"  >> {ticker}: {len(data)} days, from {data.index[0].date()}")
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"  ✗ {ticker}: No data available")
                    
            except Exception as e:
                failed_tickers.append(ticker)
                logger.error(f"  ✗ {ticker}: {str(e)}")
        
        if failed_tickers:
            logger.warning(f"\nFailed to download: {', '.join(failed_tickers)}")
        
        # Create consolidated DataFrames
        prices_df = pd.DataFrame(all_prices)
        volume_df = pd.DataFrame(all_volumes)
        
        logger.info(f"\nSuccessfully downloaded {len(prices_df.columns)} tickers")
        return prices_df, volume_df, date_coverage
    
    def determine_common_date_range(
        self,
        date_coverage: Dict[str, pd.Timestamp],
        min_coverage_threshold: float = 0.90
    ) -> Tuple[pd.Timestamp, List[str]]:
        """
        Determine optimal start date based on ticker coverage.
        
        Parameters
        ----------
        date_coverage : Dict[str, pd.Timestamp]
            First available date for each ticker
        min_coverage_threshold : float
            Minimum fraction of tickers that must be available
        
        Returns
        -------
        optimal_start_date : pd.Timestamp
            Recommended start date for analysis
        available_tickers : List[str]
            Tickers available from this date
        """
        if not date_coverage:
            raise ValueError("No tickers were successfully downloaded")
        
        # Sort tickers by first available date
        sorted_dates = sorted(date_coverage.items(), key=lambda x: x[1])
        
        n_tickers = len(sorted_dates)
        min_tickers = int(n_tickers * min_coverage_threshold)
        
        # Find date where min_coverage_threshold of tickers are available
        # Index should be: n_tickers - min_tickers
        target_index = max(0, n_tickers - min_tickers)
        optimal_date = sorted_dates[target_index][1]
        
        available_tickers = [
            ticker for ticker, date in date_coverage.items()
            if date <= optimal_date
        ]
        
        logger.info(f"\nDate Coverage Analysis:")
        logger.info(f"  Earliest available: {sorted_dates[0][1].date()} ({sorted_dates[0][0]})")
        logger.info(f"  Latest available: {sorted_dates[-1][1].date()} ({sorted_dates[-1][0]})")
        logger.info(f"  Optimal start date: {optimal_date.date()}")
        logger.info(f"  Tickers available: {len(available_tickers)}/{n_tickers} ({len(available_tickers)/n_tickers*100:.1f}%)")
        
        # Print some examples of excluded tickers if any
        excluded_tickers = [ticker for ticker, date in date_coverage.items() if date > optimal_date]
        if excluded_tickers:
            logger.info(f"  Excluded tickers (insufficient history): {', '.join(excluded_tickers[:5])}")
        
        return optimal_date, available_tickers
    
    def filter_and_save(
        self,
        prices_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        start_date: pd.Timestamp,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Filter data by date range and tickers, then save to CSV.
        
        Parameters
        ----------
        prices_df : pd.DataFrame
            Price data
        volume_df : pd.DataFrame
            Volume data
        start_date : pd.Timestamp
            Start date for filtering
        tickers : List[str]
            Tickers to include
        
        Returns
        -------
        filtered_prices : pd.DataFrame
            Filtered price data
        """
        # Filter by date and tickers
        filtered_prices = prices_df.loc[start_date:, tickers].copy()
        filtered_volumes = volume_df.loc[start_date:, tickers].copy()
        
        # Check for missing data
        missing_pct = filtered_prices.isnull().sum() / len(filtered_prices) * 100
        if missing_pct.max() > 0:
            logger.warning(f"\nMissing data detected:")
            for ticker in missing_pct[missing_pct > 0].index:
                logger.warning(f"  {ticker}: {missing_pct[ticker]:.2f}%")
        
        # Forward fill missing values (up to 5 days)
        filtered_prices = filtered_prices.ffill(limit=5)
        filtered_volumes = filtered_volumes.ffill(limit=5)
        
        # Drop any remaining rows with missing data
        initial_rows = len(filtered_prices)
        filtered_prices = filtered_prices.dropna()
        filtered_volumes = filtered_volumes.dropna()
        dropped_rows = initial_rows - len(filtered_prices)
        
        if dropped_rows > 0:
            logger.info(f"\nDropped {dropped_rows} rows with missing data")
        
        # Save to CSV
        raw_dir = config.RAW_DATA_DIR
        raw_dir.mkdir(exist_ok=True)
        
        prices_path = raw_dir / "sp100_prices.csv"
        volumes_path = raw_dir / "sp100_volumes.csv"
        
        filtered_prices.to_csv(prices_path)
        filtered_volumes.to_csv(volumes_path)
        
        logger.info(f"\nSaved consolidated data:")
        logger.info(f"  Prices: {prices_path}")
        logger.info(f"  Volumes: {volumes_path}")
        logger.info(f"  Shape: {filtered_prices.shape}")
        logger.info(f"  Date range: {filtered_prices.index[0].date()} to {filtered_prices.index[-1].date()}")
        
        return filtered_prices

