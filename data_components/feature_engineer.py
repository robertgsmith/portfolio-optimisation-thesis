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

    