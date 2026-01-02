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

    def estimate_bayesian_returns(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Estimate expected returns using Bayes-Stein estimator.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
        cov_matrix : np.ndarray, optional
            Covariance matrix (if None, uses sample)
        
        Returns
        -------
        bayesian_returns : np.ndarray
            Bayes-Stein expected returns (annualised)
        shrinkage : float
            Shrinkage intensity used
        """
        n_observations, n_assets = returns.shape
        
        # Sample mean (annualised)
        sample_mean = returns.mean().values * config.TRADING_DAYS_PER_YEAR
        
        # Grand mean (equal-weighted market return) - this is a SCALAR
        grand_mean = sample_mean.mean()
        
        # Covariance matrix (annualised)
        if cov_matrix is None:
            cov_matrix = returns.cov().values * config.TRADING_DAYS_PER_YEAR
        
        # Estimate shrinkage intensity using Jorion's formula
        if self.shrinkage_intensity is None:
            try:
                # Deviation vector from grand mean
                deviation = sample_mean - grand_mean
                
                # Compute shrinkage using Jorion (1986) formula
                # w = (N + 2) / [(N + 2) + T * deviation^T * Σ^-1 * deviation]
                
                # Inverse covariance (with regularisation for numerical stability)
                try:
                    inv_cov = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    # Add small ridge if not invertible
                    inv_cov = np.linalg.inv(cov_matrix + 1e-8 * np.eye(n_assets))
                
                # Numerator: N + 2
                numerator = n_assets + 2
                
                # Denominator: (N + 2) + T * deviation^T * Σ^-1 * deviation
                quadratic_form = deviation @ inv_cov @ deviation
                denominator = numerator + n_observations * quadratic_form
                
                # Shrinkage intensity
                shrinkage = numerator / denominator
                shrinkage = np.clip(shrinkage, 0, 1)  # Ensure [0, 1]
                
            except Exception as e:
                logger.warning(f"Shrinkage calculation failed: {e}. Using default 0.2")
                shrinkage = 0.2
        else:
            shrinkage = self.shrinkage_intensity
        
        # Bayes-Stein estimator
        # μ_BS = (1 - w) * μ_sample + w * μ_market
        bayesian_returns = (1 - shrinkage) * sample_mean + shrinkage * grand_mean
        
        logger.info(f"Bayes-Stein shrinkage intensity: {shrinkage:.4f}")
        logger.info(f"Grand mean: {grand_mean:.4f}")
        logger.info(f"Sample mean range: [{sample_mean.min():.4f}, {sample_mean.max():.4f}]")
        logger.info(f"Bayesian mean range: [{bayesian_returns.min():.4f}, {bayesian_returns.max():.4f}]")
        
        return bayesian_returns, shrinkage
        
    