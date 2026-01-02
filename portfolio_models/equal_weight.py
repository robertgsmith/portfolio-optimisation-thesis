"""
Equal Weight Portfolio

Simple 1/N equal weighting strategy (naive diversification).

Authors: Robert George Smith & Joaquin Rodriguez
Reference: DeMiguel et al. (2009). Optimal Versus Naive Diversification.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from .base_portfolio import BasePortfolio

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class EqualWeightPortfolio(BasePortfolio):
    """
    Equal weight (1/N) portfolio.
    
    Assigns equal weight to all assets, ignoring expected returns
    and covariances. Despite its simplicity, DeMiguel et al. (2009)
    showed this often outperforms optimised portfolios out-of-sample
    due to reduced estimation error.
    """
    
    def __init__(
        self,
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """
        Initialise equal weight portfolio.
        
        Parameters
        ----------
        max_weight : float
            Maximum weight per asset (may affect equal weighting if < 1/N)
        min_weight : float
            Minimum weight per asset
        risk_free_rate : float
            Risk-free rate
        """
        super().__init__(max_weight, min_weight, risk_free_rate)
        self.model_name = "Equal Weight (1/N)"
