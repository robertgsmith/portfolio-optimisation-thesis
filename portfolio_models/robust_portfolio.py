"""
Robust Portfolio Optimisation

Portfolio optimisation with uncertainty sets for parameter uncertainty.

Authors: Robert George Smith & Joaquin Rodriguez
Reference: Bertsimas & Sim (2004). The Price of Robustness.
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
        self.model_name = "Robust Optimization (Bertsimas-Sim)"
