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
        maximise: w^T μ - (λ/2) w^T Σ w
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
        
        if expected_returns is None:
            expected_returns = returns.mean().values * config.TRADING_DAYS_PER_YEAR
            
        # Winsorize extreme expected returns (addresses extreme values issue)
        if getattr(config, 'WINSORIZE_EXPECTED_RETURNS', False):
            lower_pct = getattr(config, 'WINSORIZE_LOWER_PERCENTILE', 0.05)
            upper_pct = getattr(config, 'WINSORIZE_UPPER_PERCENTILE', 0.95)
            lower_bound = np.percentile(expected_returns, lower_pct * 100)
            upper_bound = np.percentile(expected_returns, upper_pct * 100)
            expected_returns = np.clip(expected_returns, lower_bound, upper_bound)
            logger.info(f"Winsorized expected returns: [{lower_bound:.2%}, {upper_bound:.2%}]")

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

        # Add diversification constraint (prevents concentration)
        if getattr(config, 'ENABLE_DIVERSIFICATION', False):
            min_eff = getattr(config, 'MIN_EFFECTIVE_ASSETS', 20)
            max_herfindahl = 1.0 / min_eff
            constraints.append(cp.sum_squares(optimisation_variable) <= max_herfindahl)
            logger.info(f"Applied diversification: min {min_eff} effective assets (Herfindahl <= {max_herfindahl:.4f})")
        
        # Solve optimisation problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.SCS, verbose=False)
            
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
        
    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Compute the efficient frontier.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
        n_points : int
            Number of points on the frontier
        expected_returns : np.ndarray, optional
            Expected returns
        cov_matrix : np.ndarray, optional
            Covariance matrix
        
        Returns
        -------
        frontier_returns : np.ndarray
            Expected returns along frontier
        frontier_volatilities : np.ndarray
            Volatilities along frontier
        frontier_weights : np.ndarray
            Weights for each frontier portfolio
        """
        number_of_assets = returns.shape[1]
        optimisation_variable = cp.Variable(number_of_assets)
        
        if expected_returns is None:
            expected_returns = returns.mean().values * config.TRADING_DAYS_PER_YEAR
            
        # Winsorize extreme expected returns (addresses extreme values issue)
        if getattr(config, 'WINSORIZE_EXPECTED_RETURNS', False):
            lower_pct = getattr(config, 'WINSORIZE_LOWER_PERCENTILE', 0.05)
            upper_pct = getattr(config, 'WINSORIZE_UPPER_PERCENTILE', 0.95)
            lower_bound = np.percentile(expected_returns, lower_pct * 100)
            upper_bound = np.percentile(expected_returns, upper_pct * 100)
            expected_returns = np.clip(expected_returns, lower_bound, upper_bound)
            logger.info(f"Winsorized expected returns: [{lower_bound:.2%}, {upper_bound:.2%}]")
        
        if cov_matrix is None:
            cov_matrix = returns.cov().values * config.TRADING_DAYS_PER_YEAR
        
        # Find minimum and maximum return portfolios
        weights = cp.Variable(number_of_assets)
        portfolio_return = expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.min_weight,
            weights <= self.max_weight
        ]

        # Add diversification constraint (prevents concentration)
        if getattr(config, 'ENABLE_DIVERSIFICATION', False):
            min_eff = getattr(config, 'MIN_EFFECTIVE_ASSETS', 20)
            max_herfindahl = 1.0 / min_eff
            constraints.append(cp.sum_squares(optimisation_variable) <= max_herfindahl)
            logger.info(f"Applied diversification: min {min_eff} effective assets (Herfindahl <= {max_herfindahl:.4f})")
        
        # Minimum variance portfolio
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        min_return = expected_returns @ weights.value
        
        # Maximum return portfolio
        problem = cp.Problem(cp.Maximize(portfolio_return), constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        max_return = expected_returns @ weights.value
        
        # Generate points along frontier
        target_returns = np.linspace(min_return, max_return, n_points)
        frontier_volatilities = []
        frontier_weights = []
        
        for target in target_returns:
            # Minimise variance subject to target return
            constraints_with_target = constraints + [portfolio_return >= target]
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_with_target)
            problem.solve(solver=cp.SCS, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                vol = np.sqrt(problem.value)
                frontier_volatilities.append(vol)
                frontier_weights.append(weights.value)
            else:
                target_returns = target_returns[:len(frontier_volatilities)]
                break
        
        frontier_return_array = np.array(target_returns[:len(frontier_volatilities)])
        frontier_volatility_array = np.array(frontier_volatilities)
        frontier_weight_array = np.array(frontier_weights)

        return_arrays = (
            frontier_return_array,
            frontier_volatility_array,
            frontier_weight_array
        )
        return return_arrays