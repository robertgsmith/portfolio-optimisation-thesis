"""
Backtester

Rolling window backtesting engine for portfolio optimisation models.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

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
    
    def run_backtest(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run rolling window backtest for all models.
        
        Parameters
        ----------
        start_date : pd.Timestamp, optional
            Start date for backtest (if None, uses first valid date)
        end_date : pd.Timestamp, optional
            End date for backtest (if None, uses last date)
        verbose : bool
            Show progress bar
        
        Returns
        -------
        dict
            Dictionary with results for each model
        """
        logger.info("Starting backtest...")
        
        # Determine backtest period
        if start_date is None:
            start_index = self.estimation_window
        else:
            try:
                start_index = self.returns.index.get_loc(start_date)
            except KeyError:
                # choose nearest later date
                start_index = self.returns.index.searchsorted(start_date)
                logger.warning("start_date not in index, using next available date %s", self.returns.index[start_index])
        # ensure enough history for full estimation window
        if start_index < self.estimation_window:
            logger.warning("start_date has less than estimation_window history; using start at estimation_window index")
            start_index = self.estimation_window
        
        if end_date is None:
            end_index = len(self.returns)
        else:
            end_index = self.returns.index.get_loc(end_date) + 1
        
        # Get rebalancing dates
        rebalancing_dates = self.returns.index[start_index:end_index:self.rebalancing_freq]
        
        logger.info(f"Backtest period: {self.returns.index[start_index]} to {self.returns.index[end_index-1]}")
        logger.info(f"Number of rebalancing periods: {len(rebalancing_dates)}")
        
        # Initialize results for each model
        for model_name in self.models.keys():
            self.weights_history[model_name] = []
            self.portfolio_returns[model_name] = []
            self.portfolio_values[model_name] = [self.initial_cash]
        
        # Rolling window backtest
        if verbose:
            iterator = tqdm(rebalancing_dates, desc="Backtesting")
        else:
            iterator = rebalancing_dates
        
        for instance, rebalancing_date in enumerate(iterator):
            rebalancing_index = self.returns.index.get_loc(rebalancing_date)
            
            # Get estimation window
            estimation_start_index = max(0, rebalancing_index - self.estimation_window)
            estimation_end_index = rebalancing_index
            estimation_returns = self.returns.iloc[estimation_start_index:estimation_end_index]
            
            # Determine holding period (until next rebalancing or end)
            if instance < len(rebalancing_dates) - 1:
                next_rebalancing_index = self.returns.index.get_loc(rebalancing_dates[instance + 1])
            else:
                next_rebalancing_index = end_index
            
            holding_returns = self.returns.iloc[rebalancing_index:next_rebalancing_index]
            
            # Optimise and hold for each model
            for model_name, model in self.models.items():
                try:
                    # Compute expected returns and covariance for this window
                    expected_returns = estimation_returns.mean().values * config.TRADING_DAYS_PER_YEAR
                    cov_matrix = estimation_returns.cov().values * config.TRADING_DAYS_PER_YEAR
                    
                    # Optimise portfolio
                    weights = model.optimise(
                        returns=estimation_returns,
                        expected_returns=expected_returns,
                        cov_matrix=cov_matrix
                    )
                    
                    # Store weights
                    self.weights_history[model_name].append({
                        'date': rebalancing_date,
                        'weights': weights
                    })
                    
                    # Calculate portfolio returns for holding period
                    portfolio_period_return = (holding_returns.values @ weights).flatten()
                    
                    # Apply transaction costs if not first period
                    if instance > 0:
                        prev_weights = self.weights_history[model_name][-2]['weights']
                        turnover = np.sum(np.abs(weights - prev_weights))
                        transaction_costs = turnover * self.transaction_cost
                        
                        # Subtract transaction cost from first period return
                        if len(portfolio_period_return) > 0:
                            portfolio_period_return[0] -= transaction_costs
                    
                    # Store returns
                    self.portfolio_returns[model_name].extend(portfolio_period_return)
                    
                    # Update portfolio value
                    for ret in portfolio_period_return:
                        new_value = self.portfolio_values[model_name][-1] * (1 + ret)
                        self.portfolio_values[model_name].append(new_value)
                
                except Exception as e:
                    logger.error(f"Error optimizing {model_name} at {rebalancing_date}: {str(e)}")
                    # Use previous weights or equal weights as fallback
                    if instance > 0:
                        weights = self.weights_history[model_name][-1]['weights']
                    else:
                        weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
                    
                    self.weights_history[model_name].append({
                        'date': rebalancing_date,
                        'weights': weights
                    })
                    portfolio_period_return = (holding_returns.values @ weights).flatten()
                    self.portfolio_returns[model_name].extend(portfolio_period_return)
                    
                    for ret in portfolio_period_return:
                        new_value = self.portfolio_values[model_name][-1] * (1 + ret)
                        self.portfolio_values[model_name].append(new_value)
        
        logger.info("Backtest completed successfully")
        
        # Convert results to DataFrames
        self._format_results()
        
        return self.results
    