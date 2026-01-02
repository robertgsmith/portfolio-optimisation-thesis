"""
Backtesting Framework

Rolling window backtesting for portfolio optimization models.

Authors: Robert George Smith & Joaquin Rodriguez
"""

from .backtester import Backtester
from .performance_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_portfolio_turnover,
    calculate_weight_concentration,
    calculate_all_metrics
)

__all__ = [
    'Backtester',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_portfolio_turnover',
    'calculate_weight_concentration',
    'calculate_all_metrics'
]