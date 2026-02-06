"""
Data Processor Module

Processes raw price data for portfolio optimisation.
"""

# Standard library imports
import sys
from pathlib import Path
import logging
from typing import List, Dict

# Third-party imports
import pandas as pd
import numpy as np

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


class DataProcessor:
    """Process raw price data for portfolio optimisation."""
    
    def __init__(self, data_dir: Path = config.DATA_DIR):
        """Initialise processor with data directory."""
        self.data_dir = data_dir
        self.processed_dir = config.PROCESSED_DATA_DIR
    
    def load_raw_prices(self) -> pd.DataFrame:
        """Load raw price data from CSV."""
        prices_path = config.get_data_path("sp100_prices.csv", "raw")
        raw_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded raw prices: {raw_prices.shape}")
        return raw_prices
    
    def compute_returns(
        self,
        prices: pd.DataFrame,
        return_type: str = "log"
    ) -> pd.DataFrame:
        """
        Compute returns from prices.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        return_type : str
            Type of returns: 'log' or 'simple'
        
        Returns
        -------
        return_data : pd.DataFrame
            Return data
        """
        log_return = return_type == "log"
        simple_return = return_type == "simple"

        if log_return:
            return_data = np.log(prices / prices.shift(1))
        elif simple_return:
            return_data = prices.pct_change()
        else:
            raise ValueError("return_type must be 'log' or 'simple'")
        
        return_data = return_data.dropna()
        logger.info(f"Computed {return_type} returns: {return_data.shape}")
        return return_data
    
    def compute_rolling_statistics(
        self,
        returns: pd.DataFrame,
        windows: List[int] = config.ROLLING_WINDOWS
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling mean and volatility.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data
        windows : List[int]
            Rolling window sizes in trading days
        
        Returns
        -------
        stats : Dict[str, pd.DataFrame]
            Dictionary of rolling statistics
        """
        stock_stats = {}
        
        for window in windows:
            # Rolling mean (annualised)
            rolling_mean = (
                returns.rolling(window=window).mean()
                * config.TRADING_DAYS_PER_YEAR
            )
            stock_stats[f'rolling_mean_{window}d'] = rolling_mean
            
            # Rolling volatility (annualised)
            rolling_vol = (
                returns.rolling(window=window).std()
                * np.sqrt(config.TRADING_DAYS_PER_YEAR)
            )
            stock_stats[f'rolling_vol_{window}d'] = rolling_vol
            
            logger.info(f"Computed {window}-day rolling statistics")
        
        return stock_stats
    
    def compute_momentum_signals(
        self,
        prices: pd.DataFrame,
        lookback_periods: List[int] = config.MOMENTUM_PERIODS
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute momentum signals (total returns over lookback period).
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        lookback_periods : List[int]
            Lookback periods in trading days
        
        Returns
        -------
        momentum : Dict[str, pd.DataFrame]
            Dictionary of momentum signals
        """
        momentum = {}
        
        for period in lookback_periods:
            # Total return over period
            signal_at_moment = prices / prices.shift(period) - 1
            momentum[f'momentum_{period}d'] = signal_at_moment
            logger.info(f"Computed {period}-day momentum")
        
        return momentum
    
    def compute_covariance_matrices(
        self,
        returns: pd.DataFrame,
        estimation_window: int = config.ESTIMATION_WINDOW
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling covariance matrices.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data
        estimation_window : int
            Window size for covariance estimation
        
        Returns
        -------
        cov_matrices : Dict[str, pd.DataFrame]
            Dictionary with sample covariance at different points
        """
        logger.info(f"Computing rolling covariance matrices (window={estimation_window})...")
        
        # For now, compute full-sample covariance as baseline
        sample_covariance = returns.cov()
        
        logger.info(f"Computed sample covariance matrix: {sample_covariance.shape}")
        
        return {'sample_covariance': sample_covariance}
    
    def save_processed_data(
        self,
        data: pd.DataFrame,
        filename: str
    ) -> None:
        """Save processed data to CSV."""
        filepath = self.processed_dir / filename
        data.to_csv(filepath)
        logger.info(f"Saved: {filepath}")

    