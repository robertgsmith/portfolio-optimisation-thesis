"""
Backtester

Rolling window backtesting engine for portfolio optimisation models.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

from portfolio_models.base_portfolio import BasePortfolio
from .performance_metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class Backtester:
    """
    Rolling window backtesting framework for portfolio models.
    
    Implements out-of-sample testing with:
    - Rolling estimation window
    - Periodic rebalancing
    - Transaction costs
    - Performance tracking
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        models: Dict[str, BasePortfolio],
        estimation_window: int = config.ESTIMATION_WINDOW,
        rebalancing_freq: int = config.REBALANCING_FREQUENCY,
        transaction_cost: float = config.TRANSACTION_COST,
        initial_cash: float = 1000000.0
    ):
        """
        Initialise backtester.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns data (dates x assets)
        models : Dict[str, BasePortfolio]
            Dictionary of portfolio models to test
            Format: {'model_name': model_instance}
        estimation_window : int
            Number of periods for estimation window
        rebalancing_freq : int
            Rebalancing frequency in periods (21 = monthly)
        transaction_cost : float
            Transaction cost as fraction (0.001 = 10 bps)
        initial_cash : float
            Initial portfolio value
        """
        self.returns = returns
        self.models = models
        self.estimation_window = estimation_window
        self.rebalancing_freq = rebalancing_freq
        self.transaction_cost = transaction_cost
        self.initial_cash = initial_cash

        # Results storage
        self.results = {}
        self.weights_history = {}
        self.portfolio_returns = {}
        self.portfolio_values = {}
        
        logger.info(f"Initialised backtester with {len(models)} models")
        logger.info(f"Estimation window: {estimation_window} periods")
        logger.info(f"Rebalancing frequency: {rebalancing_freq} periods")
        logger.info(f"Transaction cost: {transaction_cost*10000:.1f} bps")
    