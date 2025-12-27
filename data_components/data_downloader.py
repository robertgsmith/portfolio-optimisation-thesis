import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Download and consolidate S&P 100 stock data from Yahoo Finance."""

    SP100_TICKERS = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT',
        'AMZN', 'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY',
        'BRK-B', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST',
        'CRM', 'CSCO', 'CVS', 'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR',
        'EXC', 'F', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS',
        'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT',
        'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK',
        'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE',
        'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG',
        'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS',
        'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
    ]

    def __init__(self, start_date: str = "2010-01-01", end_date: str = "2024-12-31"):
        """
        Initialize downloader with date range.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

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

                data = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    show_errors=False
                )

                if not data.empty and 'Adj Close' in data.columns:
                    all_prices[ticker] = data['Adj Close']
                    all_volumes[ticker] = data['Volume']
                    date_coverage[ticker] = data.index[0]
                    logger.info(f"  ✓ {ticker}: {len(data)} days, from {data.index[0].date()}")
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
        # Sort tickers by first available date
        sorted_dates = sorted(date_coverage.items(), key=lambda x: x[1])

        n_tickers = len(sorted_dates)
        min_tickers = int(n_tickers * min_coverage_threshold)

        # Find date where min_coverage_threshold of tickers are available
        optimal_date = sorted_dates[n_tickers - min_tickers][1]

        available_tickers = [
            ticker for ticker, date in date_coverage.items()
            if date <= optimal_date
        ]

        logger.info(f"\nDate Coverage Analysis:")
        logger.info(f"  Earliest available: {sorted_dates[0][1].date()} ({sorted_dates[0][0]})")
        logger.info(f"  Latest available: {sorted_dates[-1][1].date()} ({sorted_dates[-1][0]})")
        logger.info(f"  Optimal start date: {optimal_date.date()}")
        logger.info(f"  Tickers available: {len(available_tickers)}/{n_tickers} ({len(available_tickers)/n_tickers*100:.1f}%)")

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
        filtered_prices = filtered_prices.fillna(method='ffill', limit=5)
        filtered_volumes = filtered_volumes.fillna(method='ffill', limit=5)

        # Drop any remaining rows with missing data
        initial_rows = len(filtered_prices)
        filtered_prices = filtered_prices.dropna()
        filtered_volumes = filtered_volumes.dropna()
        dropped_rows = initial_rows - len(filtered_prices)

        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with missing data")

        # Save to CSV
        raw_dir = self.data_dir / "raw"
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
