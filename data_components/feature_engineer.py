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


class FeatureEngineer:
    """Create features for portfolio optimization models."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize feature engineer."""
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)

    def create_expected_return_estimates(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        methods: List[str] = ['historical_mean', 'momentum', 'combined']
    ) -> Dict[str, pd.DataFrame]:
        """
        Create different expected return estimates.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
        prices : pd.DataFrame
            Price data
        methods : List[str]
            Estimation methods to use
        
        Returns
        -------
        expected_return_estimates : Dict[str, pd.DataFrame]
            Dictionary of return estimates
        """
        TRADING_DAYS_PER_YEAR = 252
        expected_return_estimates = {}
        
        selected_historical_mean_method = 'historical_mean' in methods
        selected_momentum_method = 'momentum' in methods
        selected_combined_method = 'combined' in methods

        combined_method_is_possible = (
            'historical_mean' in expected_return_estimates
            and 'momentum' in expected_return_estimates
        )
        historical_mean_method_weight = 0.5
        momentum__method_weight = 0.5
        
        if selected_historical_mean_method:
            # Simple historical mean (expanding window)
            hist_mean = returns.expanding(min_periods=TRADING_DAYS_PER_YEAR).mean()
            expected_return_estimates['historical_mean'] = hist_mean
            logger.info("Created historical mean estimates")
        
        if selected_momentum_method:
            # Momentum-based estimate (12-month momentum)
            momentum = prices / prices.shift(TRADING_DAYS_PER_YEAR) - 1
            expected_return_estimates['momentum'] = momentum
            logger.info("Created momentum-based estimates")
        
        if selected_combined_method and combined_method_is_possible:
            # Weighted combination
            combined_estimates = (
                historical_mean_method_weight * expected_return_estimates['historical_mean']
                + momentum__method_weight * expected_return_estimates['momentum']
            )
            expected_return_estimates['combined'] = combined_estimates
            logger.info("Created combined estimates")
        
        return expected_return_estimates
    