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


class DataProcessor:
    """Process raw price data for portfolio optimization."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize processor with data directory."""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
    
    def load_raw_prices(self) -> pd.DataFrame:
        """Load raw price data from CSV."""
        prices_path = self.data_dir / "raw" / "sp100_prices.csv"
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
    