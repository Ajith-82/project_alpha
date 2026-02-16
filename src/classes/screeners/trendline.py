"""
Trendline Screener

Screens stocks based on price trend slope analysis.
"""

import numpy as np
import pandas as pd

from config.settings import settings
from .base import BaseScreener, ScreenerResult, Signal


class TrendlineScreener(BaseScreener):
    """
    Trendline Screener.
    
    Analyzes price trends using linear regression slope:
    - Strong Up: angle >= 60°
    - Weak Up: 30° <= angle < 60°
    - Sideways: -30° < angle < 30°
    - Weak Down: -60° < angle <= -30°
    - Strong Down: angle <= -60°
    """
    
    name = "trendline"
    description = "Price trend slope analysis"
    
    def __init__(self, lookback_days: int = settings.trend_lookback_days):
        """
        Initialize trendline screener.
        
        Args:
            lookback_days: Days to analyze for trend
        """
        self.lookback_days = lookback_days
    
    def screen(self, ticker: str, data: pd.DataFrame) -> ScreenerResult:
        """
        Screen stock based on trend slope.
        
        Args:
            ticker: Stock symbol
            data: Price data with Close column
            
        Returns:
            ScreenerResult with trend classification
        """
        if "Close" not in data.columns:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": "Close column not found"},
            )
        
        if len(data) < self.lookback_days:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": f"Need at least {self.lookback_days} days of data"},
            )
        
        try:
            # Get recent data
            trend_data = data.tail(self.lookback_days).copy()
            trend_data = trend_data.reset_index(drop=True)
            trend_data = trend_data.fillna(0)
            trend_data = trend_data.replace([np.inf, -np.inf], 0)
            
            # Calculate slope using linear regression
            x = trend_data.index.values
            y = trend_data["Close"].values
            
            if len(x) < 2:
                return ScreenerResult(
                    ticker=ticker,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    details={"error": "Insufficient data points"},
                )
            
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate angle in degrees
            angle = np.rad2deg(np.arctan(slope))
            
            # Classify trend
            if angle == 0:
                trend = "Unknown"
                signal = Signal.HOLD
                confidence = 0.0
            elif -30 <= angle <= 30:
                trend = "Sideways"
                signal = Signal.HOLD
                confidence = 0.5
            elif 30 < angle < 60:
                trend = "Weak Up"
                signal = Signal.BUY
                confidence = 0.6
            elif angle >= 60:
                trend = "Strong Up"
                signal = Signal.STRONG_BUY
                confidence = 0.8
            elif -60 < angle < -30:
                trend = "Weak Down"
                signal = Signal.SELL
                confidence = 0.6
            elif angle <= -60:
                trend = "Strong Down"
                signal = Signal.STRONG_SELL
                confidence = 0.8
            else:
                trend = "Unknown"
                signal = Signal.HOLD
                confidence = 0.0
            
            return ScreenerResult(
                ticker=ticker,
                signal=signal,
                confidence=confidence,
                details={
                    "trend": trend,
                    "angle": round(angle, 2),
                    "slope": round(slope, 4),
                    "start_price": round(y[0], 2),
                    "end_price": round(y[-1], 2),
                },
            )
            
        except Exception as e:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": str(e)},
            )
