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
        
        # Initialise results for each model
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

                    # AFTER: (Add current_date parameter if model supports it)
                    # Check if model has optimise method that accepts current_date
                    import inspect
                    sig = inspect.signature(model.optimise)

                    # # Sentiment Risk Portfolio (not used in final thesis results)
                    # if 'current_date' in sig.parameters:
                    #     # Sentiment portfolio - pass current date
                    #     weights = model.optimise(
                    #         returns=estimation_returns,
                    #         expected_returns=None,
                    #         cov_matrix=None,
                    #         current_date=estimation_returns.index[-1]  # Last date in training window
                    #     )
                    # else:
                    # Regular portfolio - don't pass date
                    weights = model.optimise(
                        returns=estimation_returns,
                        expected_returns=None,
                        cov_matrix=None
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
    
    def _format_results(self):
        """Format backtest results into DataFrames."""
        
        # Get test period dates
        test_dates = []
        for model_name in self.models.keys():
            test_dates = self.returns.index[-len(self.portfolio_returns[model_name]):]
            break
        
        # Portfolio returns DataFrame
        returns_df = pd.DataFrame(
            {name: rets for name, rets in self.portfolio_returns.items()},
            index=test_dates
        )
        
        # Portfolio values DataFrame
        values_df = pd.DataFrame(
            {name: vals[1:] for name, vals in self.portfolio_values.items()},  # Exclude initial value
            index=test_dates
        )
        
        # Weights history DataFrames
        weights_dfs = {}
        for model_name, weights_list in self.weights_history.items():
            weights_data = []
            dates = []
            for entry in weights_list:
                dates.append(entry['date'])
                weights_data.append(entry['weights'])
            
            weights_dfs[model_name] = pd.DataFrame(
                weights_data,
                index=dates,
                columns=self.returns.columns
            )
        
        self.results = {
            'returns': returns_df,
            'values': values_df,
            'weights': weights_dfs
        }
    
    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate performance metrics for all models.
        
        Returns
        -------
        pd.DataFrame
            Performance metrics for each model
        """
        if not self.results:
            raise ValueError("No backtest results available. Run run_backtest() first.")
        
        metrics_dict = {}
        
        for model_name in self.models.keys():
            returns = self.results['returns'][model_name]
            weights = self.results['weights'][model_name]
            
            metrics = calculate_all_metrics(
                returns=returns,
                weights_history=weights,
                risk_free_rate=config.RISK_FREE_RATE,
                periods_per_year=config.TRADING_DAYS_PER_YEAR
            )
            
            metrics_dict[model_name] = metrics
        
        metrics_df = pd.DataFrame(metrics_dict).T
        
        logger.info("Performance metrics calculated")
        return metrics_df
    
    def get_cumulative_returns(self) -> pd.DataFrame:
        """
        Get cumulative returns for all models.
        
        Returns
        -------
        pd.DataFrame
            Cumulative returns over time
        """
        if not self.results:
            raise ValueError("No backtest results available. Run run_backtest() first.")
        
        cum_returns = (1 + self.results['returns']).cumprod() - 1
        return cum_returns
    
    def get_drawdowns(self) -> pd.DataFrame:
        """
        Get drawdowns for all models.
        
        Returns
        -------
        pd.DataFrame
            Drawdowns over time
        """
        if not self.results:
            raise ValueError("No backtest results available. Run run_backtest() first.")
        
        cumulative = (1 + self.results['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        return drawdowns
    
    def save_results(self, output_dir: Optional[Path] = None):
        """
        Save backtest results to CSV files.
        
        Parameters
        ----------
        output_dir : Path, optional
            Output directory (default: config.RESULTS_DIR)
        """
        if not self.results:
            raise ValueError("No backtest results available. Run run_backtest() first.")
        
        if output_dir is None:
            output_dir = config.RESULTS_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save returns
        returns_path = output_dir / "backtest_returns.csv"
        self.results['returns'].to_csv(returns_path)
        logger.info(f"Saved returns to {returns_path}")
        
        # Save portfolio values
        values_path = output_dir / "backtest_values.csv"
        self.results['values'].to_csv(values_path)
        logger.info(f"Saved values to {values_path}")
        
        # Save weights for each model
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        for model_name, weights_df in self.results['weights'].items():
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            weights_path = weights_dir / f"weights_{safe_name}.csv"
            weights_df.to_csv(weights_path)
            logger.info(f"Saved {model_name} weights to {weights_path}")
        
        # Save metrics
        metrics = self.calculate_metrics()
        metrics_path = output_dir / "backtest_metrics.csv"
        metrics.to_csv(metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save cumulative returns
        cum_returns = self.get_cumulative_returns()
        cum_path = output_dir / "backtest_cumulative_returns.csv"
        cum_returns.to_csv(cum_path)
        logger.info(f"Saved cumulative returns to {cum_path}")
        
        # Save drawdowns
        drawdowns = self.get_drawdowns()
        drawdowns_path = output_dir / "backtest_drawdowns.csv"
        drawdowns.to_csv(drawdowns_path)
        logger.info(f"Saved drawdowns to {drawdowns_path}")
        
        print(f"\n>> All results saved to {output_dir}")