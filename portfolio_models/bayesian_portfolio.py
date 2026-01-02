"""
Bayesian Portfolio Optimisation

Portfolio optimisation with Bayes-Stein estimator for expected returns.

Authors: Robert George Smith & Joaquin Rodriguez
Reference: Jorion (1986). Bayes-Stein Estimation for Portfolio Analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from mean_variance import MeanVariancePortfolio

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class BayesianPortfolio(MeanVariancePortfolio):
    """
    Bayesian portfolio with Bayes-Stein estimator.
    
    Shrinks expected returns toward the grand mean:
        μ_bayesian = (1-w) * μ_sample + w * μ_grand
    
    Where:
        μ_sample = historical mean returns
        μ_grand = grand mean (equal-weighted market)
        w = shrinkage weight
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        shrinkage_intensity: Optional[float] = None,
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """
        Initialise Bayesian portfolio.
        
        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter
        shrinkage_intensity : float, optional
            Manual shrinkage intensity (0 to 1)
            If None, uses Jorion's formula to estimate
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        risk_free_rate : float
            Risk-free rate
        """
        super().__init__(risk_aversion, max_weight, min_weight, risk_free_rate)
        self.shrinkage_intensity = shrinkage_intensity
        self.model_name = "Bayesian (Jorion)"
        self.estimated_shrinkage_ = None