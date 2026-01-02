"""
Performance Metrics

Functions to calculate portfolio performance metrics.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualised Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns (daily)
    risk_free_rate : float
        Risk-free rate (annualised)
    periods_per_year : int
        Number of periods per year
    
    Returns
    -------
    float
        Annualised Sharpe ratio
    """
    has_no_returns_data = len(returns) == 0
    has_no_volatility_data = returns.std() == 0
    if has_no_returns_data or has_no_volatility_data:
        return 0.0
    
    risk_premium = returns - risk_free_rate
    excess_returns = risk_premium / periods_per_year
    
    mean_excess_return = excess_returns.mean()
    volatility = returns.std()
    annualisation = np.sqrt(periods_per_year)

    sharpe = mean_excess_return / volatility * annualisation
    
    return sharpe
