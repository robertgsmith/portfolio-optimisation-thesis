"""
Shrinkage Covariance Portfolio

Mean-variance optimisation with Ledoit-Wolf shrinkage covariance estimator.

Authors: Robert George Smith & Joaquin Rodriguez
Reference: Ledoit & Wolf (2003). Improved Estimation of the Covariance Matrix.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import Optional
import logging

from mean_variance import MeanVariancePortfolio

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class ShrinkagePortfolio(MeanVariancePortfolio):
    """
    Portfolio optimisation with Ledoit-Wolf shrinkage covariance.
    
    Uses shrinkage estimator to stabilise the covariance matrix:
        Σ_shrunk = δ * F + (1-δ) * Σ_sample
    
    Where:
        F = structured estimator (constant correlation)
        δ = shrinkage intensity (automatically determined)
        Σ_sample = sample covariance
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        shrinkage_target: str = 'auto',
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """
        Initialise shrinkage portfolio.
        
        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter
        shrinkage_target : str
            Target for shrinkage: 'auto' uses Ledoit-Wolf optimal
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        risk_free_rate : float
            Risk-free rate
        """
        super().__init__(risk_aversion, max_weight, min_weight, risk_free_rate)
        self.shrinkage_target = shrinkage_target
        self.model_name = "Shrinkage (Ledoit-Wolf)"
        self.shrinkage_intensity_ = None
    
    def estimate_shrinkage_covariance(
        self,
        returns: pd.DataFrame
    ) -> tuple:
        """
        Estimate covariance matrix using Ledoit-Wolf shrinkage.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
        
        Returns
        -------
        cov_shrunk : np.ndarray
            Shrinkage covariance matrix (annualised)
        shrinkage : float
            Shrinkage intensity used
        """
        # Fit Ledoit-Wolf estimator
        ledoit_wolf = LedoitWolf(store_precision=False, assume_centered=False)
        ledoit_wolf.fit(returns.values)
        
        # Get shrinkage covariance (annualised)
        cov_shrunk = ledoit_wolf.covariance_ * config.TRADING_DAYS_PER_YEAR
        
        # Get shrinkage intensity
        shrinkage = ledoit_wolf.shrinkage_
        
        logger.info(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.4f}")
        
        return cov_shrunk, shrinkage
    
    def optimise(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimise portfolio using shrinkage covariance.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
        expected_returns : np.ndarray, optional
            Expected returns (if None, uses historical mean)
        cov_matrix : np.ndarray, optional
            Ignored - shrinkage covariance is estimated from returns
        
        Returns
        -------
        weights : np.ndarray
            Optimal portfolio weights
        """
        # Estimate shrinkage covariance
        cov_shrunk, shrinkage = self.estimate_shrinkage_covariance(returns)
        self.shrinkage_intensity_ = shrinkage
        
        # Use parent class optimisation with shrunk covariance
        weights = super().optimise(
            returns=returns,
            expected_returns=expected_returns,
            cov_matrix=cov_shrunk
        )
        
        return weights
    