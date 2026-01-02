"""
Mean-Variance Portfolio Optimisation

Classical Markowitz (1952) mean-variance optimisation.

Authors: Robert George Smith & Joaquin Rodriguez
Reference: Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional
import logging

from .base_portfolio import BasePortfolio

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class MeanVariancePortfolio(BasePortfolio):
    """
    Mean-variance portfolio optimisation (Markowitz 1952).
    
    Solves the optimisation problem:
        maximize: w^T μ - (λ/2) w^T Σ w
        subject to: w^T 1 = 1
                    min_weight ≤ w_i ≤ max_weight
    
    Where:
        w = portfolio weights
        μ = expected returns
        Σ = covariance matrix
        λ = risk aversion parameter
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """
        Initialise mean-variance portfolio.
        
        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter (λ). Higher values = more risk averse.
            λ = 0: Maximum return (ignores risk)
            λ → ∞: Minimum variance portfolio
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        risk_free_rate : float
            Risk-free rate
        """
        super().__init__(max_weight, min_weight, risk_free_rate)
        self.risk_aversion = risk_aversion
        self.model_name = "Mean-Variance (Markowitz)"
    
    def optimise(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute optimal mean-variance portfolio weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (used if expected_returns/cov not provided)
        expected_returns : np.ndarray, optional
            Expected returns for each asset (annualised)
            If None, uses historical mean
        cov_matrix : np.ndarray, optional
            Covariance matrix (annualised)
            If None, uses sample covariance
        
        Returns
        -------
        weights : np.ndarray
            Optimal portfolio weights
        """
        number_of_assets = returns.shape[1]
        
        # Compute expected returns if not provided
        if expected_returns is None:
            expected_returns = returns.mean().values * config.TRADING_DAYS_PER_YEAR
        
        # Compute covariance matrix if not provided
        if cov_matrix is None:
            cov_matrix = returns.cov().values * config.TRADING_DAYS_PER_YEAR
        
        # Define optimisation variables
        optimisation_variable = cp.Variable(number_of_assets)
        
        # Expected return
        portfolio_return = expected_returns @ optimisation_variable  # matrix multiplication
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(optimisation_variable, cov_matrix)
        
        # Objective: maximise return - (risk_aversion/2) * variance
        objective = cp.Maximize(
            portfolio_return - (self.risk_aversion / 2) * portfolio_variance
        )
        
        # Constraints
        constraints = [
            cp.sum(optimisation_variable) == 1,  # Weights sum to 1
            optimisation_variable >= self.min_weight,  # Minimum weight
            optimisation_variable <= self.max_weight   # Maximum weight
        ]
        
        # Solve optimisation problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"Optimisation status: {problem.status}")
                # Fall back to equal weights
                weights = np.ones(number_of_assets) / number_of_assets
                return weights
            
            weights = optimisation_variable.value
            
            # Validate and store results
            if self.validate_weights(weights):
                self.weights_ = weights
                self.expected_return_ = self.compute_expected_return(weights, expected_returns)
                self.volatility_ = self.compute_volatility(weights, cov_matrix)
                self.sharpe_ratio_ = self.compute_sharpe_ratio(
                    self.expected_return_,
                    self.volatility_
                )
                
                logger.info(f"MV optimisation successful: Sharpe={self.sharpe_ratio_:.4f}")
                return weights
            else:
                logger.warning("Weight validation failed, applying constraints")
                return self.apply_weight_constraints(weights)
        
        except Exception as e:
            logger.error(f"Optimisation failed: {str(e)}")
            # Return equal weights as fallback
            weights = np.ones(number_of_assets) / number_of_assets
            return weights