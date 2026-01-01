"""
Base Portfolio Class

Abstract base class for all portfolio optimisation models.
"""

from abc import ABC, abstractmethod
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