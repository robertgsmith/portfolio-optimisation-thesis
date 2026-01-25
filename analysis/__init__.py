"""
Analysis Package

Post-backtest analysis modules for thesis.

Authors: Robert George Smith & Joaquin Rodriguez
"""

# main scripts
from .visualise_results import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_rolling_sharpe,
    plot_performance_metrics,
    plot_risk_return_scatter,
    plot_weight_evolution,
    plot_turnover_comparison,
    create_summary_table as create_visualisation_summary_table
)
from .statistical_analysis import (
    bootstrap_sharpe_difference,
    test_mean_returns,
    test_sharpe_ratios,
    test_volatility,
    test_turnover,
    test_drawdowns,
    create_summary_table
)
from .robustness_checks import (
    check_transaction_costs,
    check_estimation_windows,
    check_subperiods
)

# additional scripts
from .check_concentration import analyse_concentration
from .check_expected_return import (
    analyse_expected_returns,
    compare_bayesian_vs_historical
)
from .turnover_investigation import (
    analyse_turnover_deeply,
    plot_turnover_time_series,
    compare_weight_distributions
)
from .weight_correlation import (
    calculate_weight_correlations,
    analyse_weight_differences
)
# from .plot_periods import 

# all function imports
__all__ = [
    'plot_cumulative_returns',
    'plot_drawdowns',
    'plot_rolling_sharpe',
    'plot_performance_metrics',
    'plot_risk_return_scatter',
    'plot_weight_evolution',
    'plot_turnover_comparison',
    'create_visualisation_summary_table',
    'bootstrap_sharpe_difference',
    'test_mean_returns',
    'test_sharpe_ratios',
    'test_volatility',
    'test_turnover',
    'test_drawdowns',
    'create_summary_table',
    'check_transaction_costs',
    'check_estimation_windows',
    'check_subperiods',
    'analyse_concentration',
    'analyse_expected_returns',
    'compare_bayesian_vs_historical',
    'analyse_turnover_deeply',
    'plot_turnover_time_series',
    'compare_weight_distributions',
    'calculate_weight_correlations',
    'analyse_weight_differences'
]