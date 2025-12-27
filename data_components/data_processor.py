import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import warnings

from data_components.data_downloader import DataDownloader

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
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded raw prices: {prices.shape}")
        return prices