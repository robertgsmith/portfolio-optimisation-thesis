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

    def optimise(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute equal weights for all assets.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (only used for dimensionality)
        expected_returns : np.ndarray, optional
            Ignored - equal weights don't use expected returns
        cov_matrix : np.ndarray, optional
            Ignored - equal weights don't use covariances
        
        Returns
        -------
        weights : np.ndarray
            Equal portfolio weights (1/N for each asset)
        """
        n_assets = returns.shape[1]
        
        # Equal weights
        equal_weight = 1.0 / n_assets
        
        # Check if equal weight violates max_weight constraint
        if equal_weight > self.max_weight:
            logger.warning(
                f"Equal weight {equal_weight:.4f} exceeds max_weight {self.max_weight:.4f}. "
                f"Capping at max_weight."
            )
            # Cap at max_weight and distribute remainder
            weights = np.full(n_assets, self.max_weight)
            weights = weights / weights.sum()  # Normalise
        else:
            weights = np.full(n_assets, equal_weight)
        
        # Store results
        self.weights_ = weights
        
        # Compute statistics if data provided
        if expected_returns is not None:
            if cov_matrix is None:
                cov_matrix = returns.cov().values * config.TRADING_DAYS_PER_YEAR
            
            self.expected_return_ = self.compute_expected_return(weights, expected_returns)
            self.volatility_ = self.compute_volatility(weights, cov_matrix)
            self.sharpe_ratio_ = self.compute_sharpe_ratio(
                self.expected_return_,
                self.volatility_
            )
            
            logger.info(f"Equal weight portfolio: Sharpe={self.sharpe_ratio_:.4f}")
        
        return weights
    