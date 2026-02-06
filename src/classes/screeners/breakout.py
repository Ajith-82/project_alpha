"""
Breakout Screener

Screens stocks for breakout patterns based on volume and price action.
"""

import numpy as np
import pandas as pd

from .base import BaseScreener, ScreenerResult, Signal


class BreakoutScreener(BaseScreener):
    """
    Breakout Pattern Screener.
    
    Identifies stocks with breakout candles in recent trading:
    - Large bullish candles (> 100% of 20-day average O-C range)
    - Low selling pressure (< 40% wick to body ratio)
    - Volume spike (> 50% above 20-day average)
    """
    
    name = "breakout"
    description = "Breakout pattern detection"
    
    def __init__(
        self,
        lookback_days: int = 5,
        min_avg_volume: int = 100000,
        oc_threshold: float = 100.0,
        volume_threshold: float = 50.0,
        selling_pressure_max: float = 0.40,
    ):
        """
        Initialize breakout screener.
        
        Args:
            lookback_days: Days to search for breakout
            min_avg_volume: Minimum average volume required
            oc_threshold: Min % above 20-day O-C average
            volume_threshold: Min % above 20-day volume average
            selling_pressure_max: Max upper wick to body ratio
        """
        self.lookback_days = lookback_days
        self.min_avg_volume = min_avg_volume
        self.oc_threshold = oc_threshold
        self.volume_threshold = volume_threshold
        self.selling_pressure_max = selling_pressure_max
    
    def screen(self, ticker: str, data: pd.DataFrame) -> ScreenerResult:
        """
        Screen stock for breakout pattern.
        
        Args:
            ticker: Stock symbol
            data: Price data with OHLCV columns
            
        Returns:
            ScreenerResult with BUY signal if breakout found
        """
        try:
            # Need at least 25 candles for analysis
            if len(data) < 25:
                return ScreenerResult(
                    ticker=ticker,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    details={"error": "Insufficient data"},
                )
            
            # Get last 25 candles for analysis
            df = data.tail(25).copy()
            
            # Calculate metrics
            df["SellingPressure"] = df["High"] - df["Close"]
            df["O_to_C"] = df["Close"] - df["Open"]
            df["OC_20D_Mean"] = df["O_to_C"].rolling(20).mean()
            df["OC_perc_from_20D_Mean"] = (
                100 * (df["O_to_C"] - df["OC_20D_Mean"]) / df["OC_20D_Mean"].abs().replace(0, 1)
            )
            df["MaxOC_Prev10"] = df["O_to_C"].rolling(10).max()
            df["Volume_20D_Mean"] = df["Volume"].rolling(20).mean()
            df["Volume_perc_from_20D_Mean"] = (
                100 * (df["Volume"] - df["Volume_20D_Mean"]) / df["Volume_20D_Mean"].replace(0, 1)
            )
            
            # Get last N candles for screening
            latest = df.tail(self.lookback_days)
            
            # Breakout conditions
            condition = (
                (latest["O_to_C"] >= 0.0) &  # Bullish candle
                (latest["O_to_C"] == latest["MaxOC_Prev10"]) &  # Largest body in 10 days
                (latest["SellingPressure"] / latest["O_to_C"].replace(0, np.inf) <= self.selling_pressure_max) &
                (latest["OC_perc_from_20D_Mean"] >= self.oc_threshold) &
                (latest["Volume_perc_from_20D_Mean"] >= self.volume_threshold)
            )
            
            breakouts = latest[condition]
            avg_volume = latest["Volume"].mean()
            
            # Check if breakout found with sufficient volume
            if not breakouts.empty and avg_volume >= self.min_avg_volume:
                # Calculate confidence based on how strong the breakout is
                latest_breakout = breakouts.iloc[-1]
                oc_strength = min(latest_breakout["OC_perc_from_20D_Mean"] / 200, 1.0)
                vol_strength = min(latest_breakout["Volume_perc_from_20D_Mean"] / 100, 1.0)
                confidence = (oc_strength + vol_strength) / 2
                
                return ScreenerResult(
                    ticker=ticker,
                    signal=Signal.BUY,
                    confidence=confidence,
                    details={
                        "breakout_count": len(breakouts),
                        "avg_volume": int(avg_volume),
                        "oc_strength": round(oc_strength * 100, 1),
                        "volume_strength": round(vol_strength * 100, 1),
                    },
                )
            else:
                return ScreenerResult(
                    ticker=ticker,
                    signal=Signal.HOLD,
                    confidence=0.3,
                    details={
                        "avg_volume": int(avg_volume),
                        "reason": "No breakout pattern found",
                    },
                )
                
        except Exception as e:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": str(e)},
            )
