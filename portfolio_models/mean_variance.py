"""
Mean-Variance Portfolio Optimisation

Classical Markowitz (1952) mean-variance optimisation.

Authors: Robert George Smith & Joaquin Rodriguez
Reference: Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
"""

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