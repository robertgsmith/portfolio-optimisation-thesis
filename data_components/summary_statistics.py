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
        """Initialize summary statistics generator."""
        self.data_dir = Path(data_dir)
        self.analysis_dir = self.data_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)