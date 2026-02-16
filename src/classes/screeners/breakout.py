"""
Breakout Screener

Screens stocks for breakout patterns based on volume and price action.
"""

import numpy as np
import pandas as pd

from config.settings import settings
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
        min_avg_volume: int = settings.min_volume,
        oc_threshold: float = settings.breakout_oc_threshold,
        volume_threshold: float = settings.breakout_volume_threshold,
        selling_pressure_max: float = settings.breakout_selling_pressure_max,
        min_adx: float = settings.breakout_adx_min,
        atr_expansion_factor: float = settings.breakout_atr_expansion_factor,
    ):
        """
        Initialize breakout screener.
        
        Args:
            lookback_days: Days to search for breakout
            min_avg_volume: Minimum average volume required
            oc_threshold: Min % above 20-day O-C average
            volume_threshold: Min % above 20-day volume average
            selling_pressure_max: Max upper wick to body ratio
            min_adx: Minimum ADX for trend strength
            atr_expansion_factor: Min ATR expansion vs 20-day average
        """
        self.lookback_days = lookback_days
        self.min_avg_volume = min_avg_volume
        self.oc_threshold = oc_threshold
        self.volume_threshold = volume_threshold
        self.selling_pressure_max = selling_pressure_max
        self.min_adx = min_adx
        self.atr_expansion_factor = atr_expansion_factor
    
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
            # Volume and Confirmation Metrics
            df["Volume_20D_Mean"] = df["Volume"].rolling(20).mean()
            df["Volume_perc_from_20D_Mean"] = (
                100 * (df["Volume"] - df["Volume_20D_Mean"]) / df["Volume_20D_Mean"].replace(0, 1)
            )
            
            # Check for confirmation indicators (ADX/ATR)
            # These should have been added by add_indicators()
            has_adx = "ADX" in df.columns
            has_atr = "ATR" in df.columns
            
            if has_atr:
                df["ATR_20D_Mean"] = df["ATR"].rolling(20).mean()
            
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
            
            # Add Confirmation Filters
            if has_adx:
                condition &= (latest["ADX"] >= self.min_adx)
                
            if has_atr:
                # ATR Expansion check
                condition &= (latest["ATR"] >= latest["ATR_20D_Mean"] * self.atr_expansion_factor)
            
            breakouts = latest[condition]
            avg_volume = latest["Volume"].mean()
            
            # Check if breakout found with sufficient volume
            if not breakouts.empty and avg_volume >= self.min_avg_volume:
                # Calculate confidence based on how strong the breakout is
                latest_breakout = breakouts.iloc[-1]
                oc_strength = min(latest_breakout["OC_perc_from_20D_Mean"] / 200, 1.0)
                vol_strength = min(latest_breakout["Volume_perc_from_20D_Mean"] / 100, 1.0)
                confidence = (oc_strength + vol_strength) / 2
                
                details = {
                    "breakout_count": len(breakouts),
                    "avg_volume": int(avg_volume),
                    "oc_strength": round(oc_strength * 100, 1),
                    "volume_strength": round(vol_strength * 100, 1),
                }
                
                if has_adx:
                    details["adx"] = round(latest_breakout["ADX"], 1)
                if has_atr:
                    details["atr_expansion"] = round(latest_breakout["ATR"] / latest_breakout["ATR_20D_Mean"], 2) if latest_breakout["ATR_20D_Mean"] > 0 else 0
                
                return ScreenerResult(
                    ticker=ticker,
                    signal=Signal.BUY,
                    confidence=confidence,
                    details=details,
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
