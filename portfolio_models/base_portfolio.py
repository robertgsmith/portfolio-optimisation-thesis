"""
Base Portfolio Class

Abstract base class for all portfolio optimisation models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
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
    
    