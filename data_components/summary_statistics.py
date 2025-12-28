"""
Summary Statistics Module

Computes and saves summary statistics.
"""

# Standard library imports
from pathlib import Path
import logging
from typing import Tuple

# Third-party imports
import pandas as pd
import numpy as np

# Add parent directory to path and import config.py
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Configure logging from config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


class SummaryStatistics:
    """Compute and save summary statistics."""
    
    def __init__(self, data_dir: Path = config.DATA_DIR):
        """Initialize summary statistics generator."""
        self.data_dir = data_dir
        self.analysis_dir = config.ANALYSIS_DIR
        self.analysis_dir.mkdir(exist_ok=True)

    def compute_return_statistics(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute comprehensive return statistics.
        
        Parameters

        ----------
        returns : pd.DataFrame
            Return data
        
        Returns
        -------
        return_stats : pd.DataFrame
            Summary statistics
        """
        return_stats = pd.DataFrame()
        
        # Annualised mean return
        return_stats['ann_mean'] = returns.mean() * config.TRADING_DAYS_PER_YEAR
        
        # Annualised volatility
        return_stats['ann_vol'] = returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        return_stats['sharpe_ratio'] = return_stats['ann_mean'] / return_stats['ann_vol']
        
        # Downside deviation (annualised)
        downside_returns = returns[returns < 0]
        return_stats['downside_vol'] = downside_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        
        # Sortino ratio
        return_stats['sortino_ratio'] = return_stats['ann_mean'] / return_stats['downside_vol']
        
        # Skewness and kurtosis
        return_stats['skewness'] = returns.skew()
        return_stats['kurtosis'] = returns.kurtosis()
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return_stats['max_drawdown'] = drawdown.min()
        
        # Min and max returns
        return_stats['min_return'] = returns.min()
        return_stats['max_return'] = returns.max()
        
        logger.info("Computed return statistics")
        return return_stats
    
    def compute_correlation_analysis(
        self,
        returns: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlation and covariance matrices.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data
        
        Returns
        -------
        corr_matrix : pd.DataFrame
            Correlation matrix
        cov_matrix : pd.DataFrame
            Covariance matrix (annualised)
        """
        corr_matrix = returns.corr()
        cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR  # Annualised
        
        logger.info(f"Computed correlation matrix: {corr_matrix.shape}")
        return corr_matrix, cov_matrix
    
    def save_statistics(
        self,
        data: pd.DataFrame,
        filename: str
    ) -> None:
        """Save statistics to CSV."""
        filepath = self.analysis_dir / filename
        data.to_csv(filepath)
        logger.info(f"Saved: {filepath}")

    def return_further_data_info(self, returns: pd.DataFrame) -> None:
        """
        Print additional dataset characteristics.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data
        """
        first_date = returns.index[0]
        last_date = returns.index[-1]

        day_count = len(returns)
        year_count = (last_date - first_date).days / 365.25
        average_days_per_year = day_count / year_count

        asset_count = len(returns.columns) # same result as returns.shape[1]
        expected_tickers = len(config.SP100_TICKERS)
        
        print(f"\nDataset Characteristics:")
        print(f"  Sample period: {first_date.date()} to {last_date.date()}")
        print(f"  Calendar years: {year_count:.1f}")
        print(f"  Trading days: {day_count}")
        print(f"  Average trading days per year: {average_days_per_year:.1f}")
        print(f"  Assets: {asset_count}")
        print(f"  Coverage of S&P 100: {asset_count / expected_tickers * 100:.1f}%")
        print(f"  Total observations: {day_count * asset_count:,}")