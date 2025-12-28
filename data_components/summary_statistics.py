import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
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

class SummaryStatistics:
    """Compute and save summary statistics."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialise summary statistics generator."""
        self.data_dir = Path(data_dir)
        self.analysis_dir = self.data_dir / "analysis"
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
        TRADING_DAYS_PER_YEAR = 252
        return_stats = pd.DataFrame()
        
        # Annualised mean return
        return_stats['ann_mean'] = returns.mean() * TRADING_DAYS_PER_YEAR
        
        # Annualised volatility
        return_stats['ann_vol'] = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        return_stats['sharpe_ratio'] = return_stats['ann_mean'] / return_stats['ann_vol']
        
        # Downside deviation (annualised)
        downside_returns = returns[returns < 0]
        return_stats['downside_vol'] = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
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
        TRADING_DAYS_PER_YEAR = 252
        corr_matrix = returns.corr()
        cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR  # Annualised
        
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
