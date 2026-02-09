# Base portfolio class (abstract base class for all portfolio optimisation models)

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict
import logging

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class BasePortfolio(ABC):
    """
    Abstract base class for portfolio optimisation models.
    
    All portfolio models must inherit from this class and implement
    the optimise() method.
    """

    
    def __init__(
        self,
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """
        Initialise base portfolio.
        
        Parameters
        ----------
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset (0 = no short selling)
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
        """
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate
        
        # Storage for latest optimization results
        self.weights_ = None
        self.expected_return_ = None
        self.volatility_ = None
        self.sharpe_ratio_ = None

    @abstractmethod
    def optimise(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Compute optimal portfolio weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns data
        **kwargs : dict
            Additional model-specific parameters
        
        Returns
        -------
        weights : np.ndarray
            Optimal portfolio weights (sum to 1)
        """
        pass

    def compute_expected_return(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray
    ) -> float:
        """
        Compute expected portfolio return.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        expected_returns : np.ndarray
            Expected returns for each asset
        
        Returns
        -------
        float
            Expected portfolio return (annualized)
        """
        returns_dot_product = np.dot(weights, expected_returns)
        return returns_dot_product
    
    def compute_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        Compute portfolio volatility (standard deviation).
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        cov_matrix : np.ndarray
            Covariance matrix
        
        Returns
        -------
        float
            Portfolio volatility (annualized)
        """
        variance = np.dot(weights, np.dot(cov_matrix, weights))
        sqrt_variance = np.sqrt(variance)
        return sqrt_variance

    def compute_sharpe_ratio(
        self,
        expected_return: float,
        volatility: float
    ) -> float:
        """
        Compute Sharpe ratio.
        
        Parameters
        ----------
        expected_return : float
            Expected portfolio return
        volatility : float
            Portfolio volatility
        
        Returns
        -------
        float
            Sharpe ratio
        """
        sharpe = 0.0
        has_volatility = volatility != 0

        if has_volatility:
            risk_premium = expected_return - self.risk_free_rate
            sharpe = risk_premium / volatility
        return sharpe
    
    def get_portfolio_statistics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive portfolio statistics.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        expected_returns : np.ndarray
            Expected returns
        cov_matrix : np.ndarray
            Covariance matrix
        
        Returns
        -------
        dict
            Portfolio statistics
        """
        exp_return = self.compute_expected_return(weights, expected_returns)
        volatility = self.compute_volatility(weights, cov_matrix)
        sharpe = self.compute_sharpe_ratio(exp_return, volatility)
        
        # Weight concentration (Herfindahl index)
        concentration = np.sum(weights ** 2)
        
        # Effective number of assets
        effective_assets = 0
        concentration_greater_than_zero = concentration > 0
        if concentration_greater_than_zero:
            effective_assets = 1 / concentration

        portfolio_statistics = {
            'expected_return': exp_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'concentration': concentration,
            'effective_n_assets': effective_assets,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'sum_weights': np.sum(weights)
        }
        
        return portfolio_statistics

    def validate_weights(self, weights: np.ndarray) -> bool:
        """
        Validate that weights satisfy constraints.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights to validate
        
        Returns
        -------
        bool
            True if weights are valid
        """
        # More lenient tolerance for numerical precision
        tolerance = 1e-4
        
        # Check sum to 1 (with tolerance)
        if not np.isclose(np.sum(weights), 1.0, atol=tolerance):
            logger.warning(f"Weights sum to {np.sum(weights):.6f}, not 1.0")
            return False
        
        # Check bounds with tolerance
        if np.any(weights < self.min_weight - tolerance):
            logger.warning(f"Some weights below minimum: {np.min(weights):.9f}")
            return False
        
        if np.any(weights > self.max_weight + tolerance):
            logger.warning(f"Some weights above maximum: {np.max(weights):.9f}")
            return False
        
        return True
    
    def apply_weight_constraints(
        self,
        weights: np.ndarray,
        method: str = 'rescale'
    ) -> np.ndarray:
        """
        Apply weight constraints and normalization.
        
        Parameters
        ----------
        weights : np.ndarray
            Unconstrained weights
        method : str
            Method to apply constraints: 'rescale' or 'project'
        
        Returns
        -------
        np.ndarray
            Constrained and normalized weights
        """
        if method == 'rescale':
            # Clip to bounds with small buffer for numerical precision
            weights = np.clip(weights, self.min_weight - 1e-8, self.max_weight + 1e-8)
            
            # Ensure strictly within bounds after clipping
            weights = np.maximum(weights, self.min_weight)
            weights = np.minimum(weights, self.max_weight)
            
            # Normalize to sum to 1
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                # Fallback to equal weights
                weights = np.ones(len(weights)) / len(weights)
            
            return weights
        
        elif method == 'project':
            # More sophisticated projection (would need optimization)
            # For now, use simple rescale
            return self.apply_weight_constraints(weights, method='rescale')
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def __repr__(self) -> str:
        """String representation of portfolio."""
        portfolio_representation = f"{self.__class__.__name__}(max_weight={self.max_weight}, min_weight={self.min_weight})"
        return portfolio_representation