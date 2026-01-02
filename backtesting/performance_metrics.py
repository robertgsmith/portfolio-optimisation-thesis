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


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualised Sortino ratio (uses downside deviation).
    
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
        Annualised Sortino ratio
    """
    has_no_returns_data = len(returns) == 0
    if has_no_returns_data:
        return 0.0
    
    risk_premium = returns - risk_free_rate
    excess_returns = risk_premium / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    has_no_downside_returns_data = len(downside_returns) == 0
    has_no_downside_volatility_data = downside_returns.std() == 0
    if has_no_downside_returns_data or has_no_downside_volatility_data:
        return 0.0
    
    downside_std = downside_returns.std()
    mean_excess_return = excess_returns.mean()
    annualisation = np.sqrt(periods_per_year)

    sortino = mean_excess_return / downside_std * annualisation
    
    return sortino


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    
    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    has_no_returns_data = len(returns) == 0
    if has_no_returns_data:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    maximum_drawdown = drawdown.min()  # returns most negative value

    return maximum_drawdown


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Calmar ratio (annualised return / max drawdown).
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    periods_per_year : int
        Number of periods per year
    
    Returns
    -------
    float
        Calmar ratio
    """
    has_no_returns_data = len(returns) == 0
    if has_no_returns_data:
        return 0.0
    
    annualised_return = returns.mean() * periods_per_year
    max_drawdown = calculate_max_drawdown(returns)
    
    has_no_max_drawdown = max_drawdown == 0
    if has_no_max_drawdown:
        return 0.0
    
    calmar = annualised_return / abs(max_drawdown)
    return calmar


def calculate_portfolio_turnover(weights_history: pd.DataFrame) -> float:
    """
    Calculate average portfolio turnover.
    
    Parameters
    ----------
    weights_history : pd.DataFrame
        DataFrame of weights over time (dates x assets)
    
    Returns
    -------
    float
        Average turnover per rebalancing period
    """
    has_enough_weights_history = len(weights_history) > 1
    if not has_enough_weights_history:
        return 0.0
    
    weight_changes = weights_history.diff().abs().sum(axis=1)  # Calculate weight changes
    average_turnover = weight_changes.iloc[1:].mean()  # Average turnover (excluding first period)
    
    return average_turnover


def calculate_weight_concentration(weights: np.ndarray) -> float:
    """
    Calculate Herfindahl index (weight concentration).
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    
    Returns
    -------
    float
        Herfindahl index (0 = perfectly diversified, 1 = concentrated)
    """
    squared_weights = weights ** 2
    concentration_index = np.sum(squared_weights)
    return concentration_index
