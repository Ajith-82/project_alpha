"""
Trend Analyzer Module

Provides statistical analysis functions for stock price trends,
including log-price statistics, scoring, and rating.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .VolatileConfig import RatingThresholds


logger = logging.getLogger(__name__)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation: smooth approximation to ReLU.
    Maps real numbers to positive numbers.
    
    Args:
        x: Input array
        
    Returns:
        Positive values: log(1 + exp(x))
    """
    # Numerically stable softplus:
    # For large x: softplus(x) ≈ x
    # For small x: softplus(x) ≈ exp(x)
    # Use log1p for stability
    return np.where(
        x > 20,
        x,  # For large x, softplus(x) ≈ x
        np.log1p(np.exp(np.clip(x, -500, 20)))
    )



def estimate_logprice_statistics(
    mu: np.ndarray, sigma: np.ndarray, tt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate mean and standard deviation of log-prices.
    
    Args:
        mu: Polynomial regression coefficients (shape: [num_stocks, order+1])
        sigma: Standard deviation parameters
        tt: Time array (shape: [order+1, num_timesteps])
        
    Returns:
        Tuple of (mean_logp, std_logp)
    """
    mean_logp = np.dot(mu, tt)
    std_logp = softplus(sigma)
    return mean_logp, std_logp


def estimate_price_statistics(
    mu: np.ndarray, sigma: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert log-price statistics to price statistics.
    
    Uses log-normal distribution properties:
    - mean = exp(μ + σ²/2)
    - std = sqrt(exp(2μ + σ²) * (exp(σ²) - 1))
    
    Args:
        mu: Mean of log-prices
        sigma: Standard deviation of log-prices
        
    Returns:
        Tuple of (mean_price, std_price)
    """
    mean_price = np.exp(mu + sigma**2 / 2)
    var_factor = np.exp(sigma**2) - 1
    std_price = np.sqrt(np.exp(2 * mu + sigma**2) * var_factor)
    return mean_price, std_price


@dataclass
class TrendResult:
    """Results from trend analysis."""
    scores: np.ndarray
    growth: np.ndarray
    volatility: np.ndarray
    rates: List[str]
    
    # Estimates at different levels
    logp_est: np.ndarray
    std_logp_est: np.ndarray
    p_est: np.ndarray
    std_p_est: np.ndarray
    
    # Predictions
    logp_pred: np.ndarray
    std_logp_pred: np.ndarray
    p_pred: np.ndarray
    std_p_pred: np.ndarray


class TrendAnalyzer:
    """
    Analyzes stock price trends using polynomial regression.
    
    Computes:
    - Z-scores comparing predicted vs current prices
    - Growth rates from trend derivatives
    - Volatility from price standard deviations
    - Categorical ratings based on score thresholds
    """
    
    def __init__(self, thresholds: Optional[RatingThresholds] = None):
        """
        Initialize TrendAnalyzer.
        
        Args:
            thresholds: Rating thresholds configuration
        """
        self.thresholds = thresholds or RatingThresholds()
    
    def compute_scores(
        self,
        logp_pred: np.ndarray,
        logp_current: np.ndarray,
        std_logp_pred: np.ndarray,
        horizon: int = 5,
    ) -> np.ndarray:
        """
        Compute z-scores for each stock.
        
        Score = (predicted_logp - current_logp) / std_pred
        
        Positive score → price below trend (potential buy)
        Negative score → price above trend (potential sell)
        
        Args:
            logp_pred: Predicted log-prices at horizon
            logp_current: Current log-prices
            std_logp_pred: Standard deviation of predictions
            horizon: Prediction horizon (for indexing)
            
        Returns:
            Z-scores array
        """
        return (logp_pred[:, horizon] - logp_current[:, -1]) / std_logp_pred.squeeze()
    
    def compute_growth(
        self, phi: np.ndarray, order: int, num_timesteps: int
    ) -> np.ndarray:
        """
        Compute daily growth rate from polynomial coefficients.
        
        Growth = d/dt(polynomial) evaluated at t=1
        
        Args:
            phi: Polynomial coefficients (shape: [num_stocks, order+1])
            order: Polynomial order
            num_timesteps: Number of time steps (for normalization)
            
        Returns:
            Growth rates (% per day)
        """
        # Derivative coefficients: multiply by power index
        derivative_weights = np.arange(1, order + 1)
        return np.dot(phi[:, 1:], derivative_weights) / num_timesteps
    
    def compute_volatility(
        self, std_price: np.ndarray, current_price: np.ndarray
    ) -> np.ndarray:
        """
        Compute relative volatility.
        
        Volatility = std(price) / current_price
        
        Args:
            std_price: Standard deviation of estimated prices
            current_price: Current prices
            
        Returns:
            Relative volatility array
        """
        return std_price[:, -1] / current_price[:, -1]
    
    def rate_stocks(self, scores: np.ndarray) -> List[str]:
        """
        Assign categorical ratings based on scores.
        
        Ratings:
        - HIGHLY BELOW TREND (score > 3): Strong buy signal
        - BELOW TREND (score > 2): Buy signal
        - ALONG TREND (-2 < score < 2): Neutral
        - ABOVE TREND (score < -2): Sell signal
        - HIGHLY ABOVE TREND (score < -3): Strong sell signal
        
        Args:
            scores: Z-scores array
            
        Returns:
            List of rating strings
        """
        thresholds = self.thresholds.to_dict()
        ratings = []
        
        for score in scores:
            if score > thresholds["HIGHLY BELOW TREND"]:
                ratings.append("HIGHLY BELOW TREND")
            elif score > thresholds["BELOW TREND"]:
                ratings.append("BELOW TREND")
            elif score > thresholds["ALONG TREND"]:
                ratings.append("ALONG TREND")
            elif score > thresholds["ABOVE TREND"]:
                ratings.append("ABOVE TREND")
            else:
                ratings.append("HIGHLY ABOVE TREND")
        
        return ratings
    
    def analyze(
        self,
        phi: np.ndarray,
        psi: np.ndarray,
        logp: np.ndarray,
        price: np.ndarray,
        tt: np.ndarray,
        order: int,
        horizon: int = 5,
    ) -> TrendResult:
        """
        Perform complete trend analysis.
        
        Args:
            phi: Mean polynomial parameters
            psi: Volatility parameters
            logp: Log-prices array
            price: Raw prices array
            tt: Time array
            order: Polynomial order
            horizon: Prediction horizon in days
            
        Returns:
            TrendResult with all computed metrics
        """
        num_stocks, num_timesteps = logp.shape
        
        # Compute log-price estimates
        logp_est, std_logp_est = estimate_logprice_statistics(phi, psi, tt)
        
        # Compute predictions at horizon
        tt_pred = (
            (1 + (np.arange(1 + horizon) / num_timesteps))
            ** np.arange(order + 1).reshape(-1, 1)
        ).astype("float32")
        logp_pred, std_logp_pred = estimate_logprice_statistics(phi, psi, tt_pred)
        
        # Convert to price statistics
        p_est, std_p_est = estimate_price_statistics(logp_est, std_logp_est)
        p_pred, std_p_pred = estimate_price_statistics(logp_pred, std_logp_pred)
        
        # Compute metrics
        scores = self.compute_scores(logp_pred, logp, std_logp_pred, horizon)
        growth = self.compute_growth(phi, order, num_timesteps)
        volatility = self.compute_volatility(std_p_est, price)
        ratings = self.rate_stocks(scores)
        
        return TrendResult(
            scores=scores,
            growth=growth,
            volatility=volatility,
            rates=ratings,
            logp_est=logp_est,
            std_logp_est=std_logp_est,
            p_est=p_est,
            std_p_est=std_p_est,
            logp_pred=logp_pred,
            std_logp_pred=std_logp_pred,
            p_pred=p_pred,
            std_p_pred=std_p_pred,
        )
