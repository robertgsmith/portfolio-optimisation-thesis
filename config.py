"""
Configuration File for Portfolio Optimisation Thesis

This file contains all configuration parameters, constants, and settings
for the robust portfolio optimisation project.

Authors: Robert George Smith & Joaquin Rodriguez
Project: Robust Portfolio Optimisation Under Parameter Uncertainty
"""

from pathlib import Path
from typing import List

# ============================================================================
# PROJECT METADATA
# ============================================================================

PROJECT_NAME = "Robust Portfolio Optimisation"
AUTHORS = ["Robert George Smith", "Joaquin Rodriguez"]
INSTITUTION = "Frankfurt School of Finance & Management"
PROGRAM = "BSc Computational Business Analytics"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Date Range
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# Asset Universe
ASSET_UNIVERSE = "S&P 100"  # OEX Index
MIN_COVERAGE_THRESHOLD = 0.90  # Minimum 90% of tickers must have data

# S&P 100 Tickers (OEX components)
SP100_TICKERS = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT',
    'AMZN', 'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY',
    'BRK-B', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST',
    'CRM', 'CSCO', 'CVS', 'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR',
    'EXC', 'F', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS',
    'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT',
    'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK',
    'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE',
    'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG',
    'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS',
    'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
]

# Data Quality
MAX_MISSING_DATA_PCT = 0.05  # Maximum 5% missing data per ticker
FORWARD_FILL_LIMIT = 5  # Maximum days to forward fill missing data

# ============================================================================
# RETURN CALCULATION
# ============================================================================

# Return Type
RETURN_TYPE = "log"  # Options: "log" or "simple"

# Annualisation
TRADING_DAYS_PER_YEAR = 252
TRADING_WEEKS_PER_YEAR = 52
TRADING_MONTHS_PER_YEAR = 12

# Combined Expected Return Method weights
HISTORICAL_MEAN_METHOD_WEIGHT= 0.5
MOMENTUM_METHOD_WEIGHT= 0.5

# ============================================================================
# PORTFOLIO OPTIMISATION PARAMETERS
# ============================================================================

# Estimation Window
ESTIMATION_WINDOW = 252  # 1 year of daily data
MIN_ESTIMATION_WINDOW = 126  # Minimum 6 months for estimation

# Rebalancing
REBALANCING_FREQUENCY = 21  # Monthly rebalancing (21 trading days)

# Position Limits
MAX_WEIGHT = 0.10  # Maximum 10% per asset
MIN_WEIGHT = 0.00  # No short selling
MAX_SECTOR_WEIGHT = 0.30  # Maximum 30% per sector (if sector data available)

# Target Portfolio Statistics
TARGET_RETURN = 0.12  # 12% annualised return target (if using target return optimisation)
MAX_VOLATILITY = 0.20  # 20% maximum portfolio volatility

# ============================================================================
# RISK PARAMETERS
# ============================================================================

# Risk-Free Rate
RISK_FREE_RATE = 0.00  # Assuming 0% for Sharpe ratio calculation
# Note: Could use 3-month T-bill rate for more realistic analysis

# Confidence Levels
CONFIDENCE_LEVEL_VAR = 0.95  # 95% confidence for VaR
CONFIDENCE_LEVEL_CVAR = 0.95  # 95% confidence for CVaR

# Robust Optimisation Uncertainty Parameters
UNCERTAINTY_SET_SIZE = 0.5  # Epsilon for robust optimisation (Bertsimas & Sim)
COVARIANCE_UNCERTAINTY = 0.1  # 10% uncertainty in covariance estimates

# ============================================================================
# DIVERSIFICATION CONSTRAINTS (Industry Best Practice)
# ============================================================================

# Minimum Effective Number of Assets
MIN_EFFECTIVE_ASSETS = 40          # Minimum effective number of assets
ENABLE_DIVERSIFICATION = True      # Set to False to disable for comparison

# Expected Return Treatment
WINSORIZE_EXPECTED_RETURNS = True  # Cap extreme values
WINSORIZE_LOWER_PERCENTILE = 0.05  # 5th percentile
WINSORIZE_UPPER_PERCENTILE = 0.95  # 95th percentile

# ============================================================================
# PORTFOLIO MODEL SPECIFIC PARAMETERS (Add this entire new section)
# ============================================================================

# Mean-Variance Optimisation
RISK_AVERSION_DEFAULT = 1.0  # Default risk aversion parameter (λ)
RISK_AVERSION_RANGE = [0.5, 1.0, 2.0, 5.0]  # For sensitivity analysis

# Shrinkage Portfolio (Ledoit-Wolf)
SHRINKAGE_TARGET = 'auto'  # Options: 'auto', 'constant_correlation', 'constant_variance'

# Bayesian Portfolio (Jorion)
BAYESIAN_SHRINKAGE_INTENSITY = None  # None = auto-estimate, or float [0, 1]
BAYESIAN_PRIOR_WEIGHT = 0.1  # Weight given to market prior
TAU = 0.05  # Uncertainty in prior

# Robust Optimisation (Bertsimas & Sim)
ROBUST_EPSILON = 0.5  # Size of uncertainty set (0 = no uncertainty, 1 = high uncertainty)
ROBUST_GAMMA = 0.5  # Budget of uncertainty
ROBUST_TARGET_RETURN = None  # None = use risk-return tradeoff, or float for target return

# ============================================================================
# OPTIMISATION SOLVER SETTINGS (Add this entire new section)
# ============================================================================

# CVXPY solver preferences
CVXPY_SOLVER = "SCS"  # Primary solver (Options: ECOS, SCS, OSQP, CVXOPT)
CVXPY_SOLVER_FALLBACK = "OSQP"  # Fallback if primary fails
CVXPY_VERBOSE = False  # Set to True for debugging optimization issues
CVXPY_MAX_ITER = 10000  # Maximum iterations for solver
CVXPY_ABSTOL = 1e-7  # Absolute tolerance
CVXPY_RELTOL = 1e-6  # Relative tolerance
CVXPY_FEASTOL = 1e-7  # Feasibility tolerance

# Solver-specific settings
SOLVER_SETTINGS = {
    'ECOS': {
        'max_iters': CVXPY_MAX_ITER,
        'abstol': CVXPY_ABSTOL,
        'reltol': CVXPY_RELTOL,
        'feastol': CVXPY_FEASTOL
    },
    'SCS': {
        'max_iters': CVXPY_MAX_ITER,
        'eps': CVXPY_ABSTOL,
        'verbose': CVXPY_VERBOSE
    },
    'OSQP': {
        'max_iter': CVXPY_MAX_ITER,
        'eps_abs': CVXPY_ABSTOL,
        'eps_rel': CVXPY_RELTOL,
        'verbose': CVXPY_VERBOSE
    }
}

# ============================================================================
# TRANSACTION COSTS
# ============================================================================

# Base transaction cost
TRANSACTION_COST = 0.0010  # 10 basis points (0.1%)

# Transaction cost scenarios for robustness checks
TRANSACTION_COSTS_SCENARIOS = {
    'low': 0.0005,      # 5 basis points
    'base': 0.0010,     # 10 basis points
    'medium': 0.0015,   # 15 basis points
    'high': 0.0025      # 25 basis points
}

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

# Momentum Lookback Periods (in trading days)
MOMENTUM_PERIODS = [21, 63, 126, 252]  # 1M, 3M, 6M, 12M

# Rolling Statistics Windows (in trading days)
ROLLING_WINDOWS = [21, 63, 126, 252]  # 1M, 3M, 6M, 12M

# Expected Return Estimation Methods
EXPECTED_RETURN_METHODS = ['historical_mean', 'momentum', 'combined']

# Covariance Estimation Methods
COVARIANCE_METHODS = ['sample', 'ledoit_wolf', 'robust']

# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================

# Initial Portfolio Value
INITIAL_PORTFOLIO_VALUE = 1000000.0  # $1 million

# Backtest Period (optional - can override in backtest)
BACKTEST_START_DATE = None  # None = use all available data
BACKTEST_END_DATE = None     # None = use all available data

# Performance Metrics to Calculate
PERFORMANCE_METRICS = [
    'annual_return',
    'annual_volatility',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'total_return',
    'avg_turnover',
    'avg_concentration',
    'win_rate'
]

# ============================================================================
# SENTIMENT ANALYSIS (Extension)
# ============================================================================

# Sentiment portfolio parameters
SENTIMENT_WEIGHT = 0.1              # Weight for return adjustment method
SENTIMENT_SENSITIVITY = 0.5         # Sensitivity for risk management method
SENTIMENT_ENABLE = True             # Toggle sentiment extension

# ============================================================================
# DIRECTORIES
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
ANALYSIS_DIR = DATA_DIR / "analysis"

# Sentiment data
SENTIMENT_DIR = DATA_DIR / "sentiment"

# Create directories if they don't exist
for directory in [
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR,
    ANALYSIS_DIR, SENTIMENT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PORTFOLIO MODELS CONFIGURATION
# ============================================================================

PORTFOLIO_MODELS = {
    'mean_variance': {
        'name': 'Mean-Variance (Markowitz)',
        'enabled': True,
        'parameters': {}
    },
    'shrinkage': {
        'name': 'Shrinkage Covariance (Ledoit-Wolf)',
        'enabled': True,
        'parameters': {
            'shrinkage_target': 'constant_correlation',
            'shrinkage_intensity': None  # Auto-determined
        }
    },
    'bayesian': {
        'name': 'Bayesian (Jorion)',
        'enabled': True,
        'parameters': {
            'prior_weight': 0.1,  # Weight given to market prior
            'tau': 0.05  # Uncertainty in prior
        }
    },
    'robust': {
        'name': 'Robust Optimisation (Bertsimas & Sim)',
        'enabled': True,
        'parameters': {
            'epsilon': UNCERTAINTY_SET_SIZE,
            'gamma': 0.5  # Budget of uncertainty
        }
    },
    'equal_weight': {
        'name': '1/N Equal Weight (Benchmark)',
        'enabled': True,
        'parameters': {}
    }
}

# ============================================================================
# VISUALISATION SETTINGS
# ============================================================================

# Plot Style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Color Palette
COLOR_PALETTE = 'Set2'
COLORS = {
    'mean_variance': '#1f77b4',
    'shrinkage': '#ff7f0e',
    'bayesian': '#2ca02c',
    'robust': '#d62728',
    'equal_weight': '#9467bd'
}

# Figure Sizes
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (15, 6)
FIGSIZE_LARGE = (12, 8)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# VALIDATION & CHECKS
# ============================================================================

def validate_config():
    """Validate configuration parameters."""
    
    # Check date range
    from datetime import datetime
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    assert end > start, "END_DATE must be after START_DATE"
    
    # Check weight constraints
    assert 0 <= MIN_WEIGHT <= MAX_WEIGHT <= 1, "Invalid weight constraints"
    assert MAX_WEIGHT * len(SP100_TICKERS) >= 1, "MAX_WEIGHT too restrictive"
    
    # Check estimation window
    assert ESTIMATION_WINDOW >= MIN_ESTIMATION_WINDOW, "ESTIMATION_WINDOW too small"
    
    # Check transaction costs
    assert 0 <= TRANSACTION_COST <= 0.01, "Transaction cost should be between 0 and 1%"
    
    print(">> Configuration validated successfully")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_data_path(filename: str, subdir: str = "raw") -> Path:
    """
    Get full path for a data file.
    
    Parameters
    ----------
    filename : str
        Name of the file
    subdir : str
        Subdirectory ('raw', 'processed', 'features', 'analysis')
    
    Returns
    -------
    Path
        Full path to the file
    """
    subdir_map = {
        'raw': RAW_DATA_DIR,
        'processed': PROCESSED_DATA_DIR,
        'features': FEATURES_DIR,
        'analysis': ANALYSIS_DIR,
        'sentiment': SENTIMENT_DIR
    }
    return subdir_map[subdir] / filename


def get_results_path(filename: str, subdir: str = "tables") -> Path:
    """
    Get full path for a results file.
    
    Parameters
    ----------
    filename : str
        Name of the file
    subdir : str
        Subdirectory ('figures' or 'tables')
    
    Returns
    -------
    Path
        Full path to the file
    """
    if subdir == 'figures':
        return FIGURES_DIR / filename
    elif subdir == 'tables':
        return TABLES_DIR / filename
    else:
        return RESULTS_DIR / filename


def print_config_summary():
    """Print a summary of key configuration parameters."""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Project: {PROJECT_NAME}")
    print(f"Authors: {', '.join(AUTHORS)}")
    print(f"\nData Configuration:")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Universe: {ASSET_UNIVERSE} ({len(SP100_TICKERS)} tickers)")
    print(f"  Estimation Window: {ESTIMATION_WINDOW} days")
    print(f"  Rebalancing: Every {REBALANCING_FREQUENCY} days")
    print(f"\nPortfolio Constraints:")
    print(f"  Max Weight: {MAX_WEIGHT*100:.1f}%")
    print(f"  Min Weight: {MIN_WEIGHT*100:.1f}%")
    print(f"  Transaction Cost: {TRANSACTION_COST*10000:.1f} bps")
    print(f"\nModels Enabled:")
    for model_key, model_config in PORTFOLIO_MODELS.items():
        if model_config['enabled']:
            print(f"  >> {model_config['name']}")
    print("="*70 + "\n")


# ============================================================================
# RUN VALIDATION ON IMPORT
# ============================================================================

if __name__ == "__main__":
    validate_config()
    print_config_summary()
else:
    # Silent validation when imported
    try:
        validate_config()
    except AssertionError as e:
        print(f"!!!  Configuration Error: {e}")