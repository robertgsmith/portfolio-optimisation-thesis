"""
Sentiment-Enhanced Portfolio with Risk Management (not used in final thesis results)

Adjusts portfolio risk appetite based on Federal Reserve policy tone.

Authors: Robert George Smith & Joaquin Rodriguez
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

# Import config
import sys
from pathlib import Path

# Get the project root directory (parent of this script's directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from .mean_variance import MeanVariancePortfolio
import config

logger = logging.getLogger(__name__)


class SentimentRiskPortfolio(MeanVariancePortfolio):
    """
    Portfolio with dynamic risk management based on Fed policy tone.
    
    Methodology:
    1. Load monetary policy tone indicator
    2. Adjust risk aversion parameter based on current policy tone:
       - Dovish policy (tone > 0) → Lower risk aversion → Risk-on
       - Hawkish policy (tone < 0) → Higher risk aversion → Risk-off
    3. Optimize portfolio with adjusted risk aversion
    
    Behavioral Interpretation:
    - When Fed is accommodative (dovish), investors take more risk
    - When Fed is restrictive (hawkish), investors reduce risk exposure
    
    Reference:
    - Bernanke & Kuttner (2005): Fed policy affects stock prices
    - Tetlock (2007): News sentiment predicts returns
    """
    
    def __init__(
        self,
        base_risk_aversion: float = 1.0,
        sentiment_sensitivity: float = 0.5,
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """
        Initialise sentiment risk management portfolio.
        
        Parameters
        ----------
        base_risk_aversion : float
            Baseline risk aversion (when sentiment is neutral)
            Default 1.0
        sentiment_sensitivity : float
            How much to adjust risk aversion based on sentiment
            Range: 0 (no adjustment) to 1 (full adjustment)
            Default 0.5 = moderate sensitivity
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        risk_free_rate : float
            Risk-free rate
        """
        super().__init__(base_risk_aversion, max_weight, min_weight, risk_free_rate)
        
        self.base_risk_aversion = base_risk_aversion
        self.sentiment_sensitivity = sentiment_sensitivity
        self.model_name = "Sentiment Risk Management"
        
        # Load sentiment data
        self.sentiment_data = self._load_sentiment()
        self.current_sentiment = 0.0  # Track current sentiment
    
    def _load_sentiment(self):
        """Load Fed sentiment/policy data."""
        try:
            sentiment_path = config.SENTIMENT_DIR / 'monetary_policy_factor.csv'
            
            if sentiment_path.exists():
                sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded policy factor: {len(sentiment)} observations")
                logger.info(f"  Date range: {sentiment.index[0].date()} to {sentiment.index[-1].date()}")
                return sentiment
            else:
                logger.warning(f"Policy factor not found at {sentiment_path}")
                logger.warning("Run: python sentiment/create_policy_factor.py")
                return None
        
        except Exception as e:
            logger.error(f"Could not load policy factor: {e}")
            return None

    def get_policy_tone(self, date):
        """
        Get policy tone for a given date.
        
        Parameters
        ----------
        date : pd.Timestamp
            Date to get policy tone for
        
        Returns
        -------
        tone : float
            Policy tone score
            Positive = dovish (accommodative)
            Negative = hawkish (restrictive)
            Range: approximately [-1, 1]
        """
        if self.sentiment_data is None:
            return 0.0
        
        try:
            # Remove timezone info if present
            if hasattr(date, 'tz') and date.tz is not None:
                date = date.tz_localize(None)
            
            # Get most recent policy tone on or before this date
            available_dates = self.sentiment_data.index[self.sentiment_data.index <= date]
            
            if len(available_dates) == 0:
                logger.warning(f"No sentiment data before {date}")
                return 0.0
            
            most_recent_date = available_dates[-1]
            tone = self.sentiment_data.loc[most_recent_date, 'Policy_Tone']
            
            # Handle NaN
            if pd.isna(tone):
                return 0.0
            
            # Debug logging
            logger.info(f"Date: {date.date()}, Sentiment: {tone:.3f}")
            
            return float(tone)
        
        except Exception as e:
            logger.error(f"Error getting policy tone for {date}: {e}")
            return 0.0
    
    def adjust_risk_aversion(self, policy_tone: float) -> float:
        """
        Adjust risk aversion based on policy tone.
        
        Formula:
        λ_adjusted = λ_base * (1 - sensitivity * tone)
        
        Where:
        - tone ∈ [-1, 1]
        - sensitivity ∈ [0, 1]
        
        Examples (with base_risk_aversion=1.0, sensitivity=0.5):
        - Dovish (tone=+1.0): λ = 1.0 * (1 - 0.5*1.0) = 0.5 (lower risk aversion → risk-on)
        - Neutral (tone=0.0): λ = 1.0 * (1 - 0.5*0.0) = 1.0 (no change)
        - Hawkish (tone=-1.0): λ = 1.0 * (1 - 0.5*(-1.0)) = 1.5 (higher risk aversion → risk-off)
        
        Parameters
        ----------
        policy_tone : float
            Policy tone indicator
        
        Returns
        -------
        adjusted_risk_aversion : float
            Adjusted risk aversion parameter
        """
        # Calculate adjustment
        adjustment_factor = 1.0 - (self.sentiment_sensitivity * policy_tone)
        
        adjusted = self.base_risk_aversion * adjustment_factor
        
        # Clip to reasonable range [0.1, 5.0]
        # Prevents extreme values that could cause numerical issues
        adjusted = np.clip(adjusted, 0.1, 5.0)
        
        logger.debug(f"Risk aversion: {self.base_risk_aversion:.2f} → {adjusted:.2f} (tone={policy_tone:.3f})")
        
        return adjusted
    
    def optimise(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimize portfolio with sentiment-adjusted risk aversion.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (last date used for sentiment lookup)
        expected_returns : np.ndarray, optional
            Expected returns (if None, uses historical mean)
        cov_matrix : np.ndarray, optional
            Covariance matrix (if None, uses sample)
        
        Returns
        -------
        weights : np.ndarray
            Optimal portfolio weights
        """
        # Get current date (last date in returns data)
        current_date = returns.index[-1]
        
        # Get policy tone
        policy_tone = self.get_policy_tone(current_date)
        self.current_sentiment = policy_tone  # Store for reporting
        
        # Adjust risk aversion
        adjusted_risk_aversion = self.adjust_risk_aversion(policy_tone)
        
        # Temporarily update risk aversion
        original_risk_aversion = self.risk_aversion
        self.risk_aversion = adjusted_risk_aversion
        
        # Log the adjustment
        regime = "DOVISH" if policy_tone > 0.3 else "HAWKISH" if policy_tone < -0.3 else "NEUTRAL"
        logger.info(f"Date: {current_date.date()}, Policy: {regime} ({policy_tone:.3f}), λ: {adjusted_risk_aversion:.3f}")
        
        # Use parent class optimization with adjusted risk aversion
        weights = super().optimise(
            returns=returns,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix
        )
        
        # Restore original risk aversion
        self.risk_aversion = original_risk_aversion
        
        return weights
    
    def get_statistics(self) -> dict:
        """Get portfolio statistics including sentiment info."""
        stats = super().get_portfolio_statistics(
            self.weights_,
            np.zeros(len(self.weights_)),  # Placeholder
            np.eye(len(self.weights_))     # Placeholder
        )
        
        stats['current_sentiment'] = self.current_sentiment
        stats['sentiment_regime'] = (
            'Dovish' if self.current_sentiment > 0.3 
            else 'Hawkish' if self.current_sentiment < -0.3 
            else 'Neutral'
        )
        
        return stats


# ============================================================================
# Alternative: Sentiment-Adjusted Expected Returns
# ============================================================================

class SentimentReturnPortfolio(MeanVariancePortfolio):
    """
    Adjust expected returns based on policy tone.
    
    Simpler than risk management approach.
    """
    
    def __init__(
        self,
        sentiment_weight: float = 0.1,
        risk_aversion: float = 1.0,
        max_weight: float = config.MAX_WEIGHT,
        min_weight: float = config.MIN_WEIGHT,
        risk_free_rate: float = config.RISK_FREE_RATE
    ):
        """Initialise sentiment return portfolio."""
        super().__init__(risk_aversion, max_weight, min_weight, risk_free_rate)
        self.sentiment_weight = sentiment_weight
        self.model_name = "Sentiment-Adjusted Returns"
        self.sentiment_data = self._load_sentiment()
    
    def _load_sentiment(self):
        """Load sentiment data."""
        try:
            sentiment_path = config.SENTIMENT_DIR / 'monetary_policy_factor.csv'
            if sentiment_path.exists():
                return pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        except:
            pass
        return None
    
    def optimise(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Optimise with sentiment-adjusted returns."""
        
        # Get policy tone
        current_date = returns.index[-1]
        policy_tone = 0.0
        
        if self.sentiment_data is not None:
            available = self.sentiment_data.index[self.sentiment_data.index <= current_date]
            if len(available) > 0:
                policy_tone = self.sentiment_data.loc[available[-1], 'Policy_Tone']
                if pd.isna(policy_tone):
                    policy_tone = 0.0
        
        # Calculate base returns
        if expected_returns is None:
            expected_returns = returns.mean().values * config.TRADING_DAYS_PER_YEAR
        
        # Adjust returns: dovish → boost returns, hawkish → reduce returns
        adjustment = self.sentiment_weight * policy_tone * 0.05  # 5% max adjustment
        adjusted_returns = expected_returns * (1 + adjustment)
        
        # Optimize with adjusted returns
        return super().optimise(
            returns=returns,
            expected_returns=adjusted_returns,
            cov_matrix=cov_matrix
        )


if __name__ == "__main__":
    print("""
    Sentiment Risk Management Portfolio
    
    Usage:
    ------
    1. Create monetary policy factor:
       python sentiment/create_policy_factor.py
    
    2. Add to backtesting models:
       from portfolio_models.sentiment_portfolio import SentimentRiskPortfolio
       
       models = {
           'Sentiment Risk': SentimentRiskPortfolio(
               base_risk_aversion=1.0,
               sentiment_sensitivity=1.5
           )
       }
    
    3. Run backtest as normal
    
    Parameters:
    -----------
    - base_risk_aversion: Baseline risk aversion (1.0 = moderate)
    - sentiment_sensitivity: How much to adjust (0.5 = moderate, 1.0 = aggressive)
    
    Interpretation:
    ---------------
    - Dovish Fed → Lower risk aversion → More equity exposure
    - Hawkish Fed → Higher risk aversion → Less equity exposure
    """)