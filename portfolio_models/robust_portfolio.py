"""
Robust Portfolio Optimisation

Portfolio optimisation with uncertainty sets for parameter uncertainty.

Authors: Robert George Smith & Joaquin Rodriguez
Reference: Bertsimas & Sim (2004). The Price of Robustness.
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


class RobustPortfolio(BasePortfolio):
    """
    Robust portfolio optimisation with uncertainty sets.
    
    Handles parameter uncertainty by optimizing for worst-case scenario
    within an uncertainty set:
    
        minimize: max_{Σ ∈ U} w^T Σ w
        subject to: w^T μ ≥ r_target
                    w^T 1 = 1
                    min_weight ≤ w_i ≤ max_weight
    
    Where U is an uncertainty set for the covariance matrix.
    """
    
    def __init__(
        self,
        epsilon: float = config.UNCERTAINTY_SET_SIZE,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """
        Initialise robust portfolio.
        
        Parameters
        ----------
        epsilon : float
            Size of uncertainty set (higher = more conservative)
        target_return : float, optional
            Target return constraint (if None, uses risk-return tradeoff)
        risk_aversion : float
            Risk aversion parameter (used if no target return)
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        risk_free_rate : float
            Risk-free rate
        """
        super().__init__(max_weight, min_weight, risk_free_rate)
        self.epsilon = epsilon
        self.target_return = target_return
        self.risk_aversion = risk_aversion
        self.model_name = "Robust Optimisation (Bertsimas-Sim)"

    def optimise(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute robust optimal portfolio weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
        expected_returns : np.ndarray, optional
            Expected returns (if None, uses historical mean)
        cov_matrix : np.ndarray, optional
            Covariance matrix (if None, uses sample)
        
        Returns
        -------
        weights : np.ndarray
            Optimal robust portfolio weights
        """
        n_assets = returns.shape[1]
        
        # Compute expected returns if not provided
        if expected_returns is None:
            expected_returns = returns.mean().values * config.TRADING_DAYS_PER_YEAR
        
        # Compute covariance matrix if not provided
        if cov_matrix is None:
            cov_matrix = returns.cov().values * config.TRADING_DAYS_PER_YEAR
        
        # Estimate uncertainty in covariance matrix
        # Use standard errors of covariance estimates
        n_obs = len(returns)
        cov_std = np.sqrt(2 / n_obs) * np.abs(cov_matrix)  # Approximation
        
        # Define optimisation variables
        w = cp.Variable(n_assets)
        
        # Portfolio return
        portfolio_return = expected_returns @ w
        
        # Worst-case variance (conservative)
        # Increase diagonal elements by epsilon * std
        cov_robust = cov_matrix + self.epsilon * np.diag(np.diag(cov_std))
        portfolio_variance = cp.quad_form(w, cov_robust)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight
        ]
        
        # Add target return constraint if specified
        if self.target_return is not None:
            constraints.append(portfolio_return >= self.target_return)
            # Objective: minimise worst-case risk
            objective = cp.Minimize(portfolio_variance)
        else:
            # Objective: risk-return tradeoff
            objective = cp.Maximize(
                portfolio_return - (self.risk_aversion / 2) * portfolio_variance
            )
        
        # Solve optimisation problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.SCS, verbose=False)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"Robust optimisation status: {problem.status}")
                return np.ones(n_assets) / n_assets
            
            weights = w.value
            
            # Validate and store results
            if self.validate_weights(weights):
                self.weights_ = weights
                # Use nominal covariance for statistics (not robust)
                self.expected_return_ = self.compute_expected_return(weights, expected_returns)
                self.volatility_ = self.compute_volatility(weights, cov_matrix)
                self.sharpe_ratio_ = self.compute_sharpe_ratio(
                    self.expected_return_,
                    self.volatility_
                )
                
                logger.info(f"Robust optimisation successful: Sharpe={self.sharpe_ratio_:.4f}")
                logger.info(f"Uncertainty adjustment (epsilon): {self.epsilon:.4f}")
                return weights
            else:
                logger.warning("Weight validation failed, applying constraints")
                return self.apply_weight_constraints(weights)
        
        except Exception as e:
            logger.error(f"Robust optimisation failed: {str(e)}")
            return np.ones(n_assets) / n_assets
