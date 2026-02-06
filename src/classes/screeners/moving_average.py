"""
Moving Average Screener

Screens stocks based on moving average crossovers and positions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from .base import BaseScreener, ScreenerResult, Signal


class MovingAverageScreener(BaseScreener):
    """
    Moving Average Screener.
    
    Uses multiple MA crossover strategies:
    - SMA 10 > SMA 30 crossover
    - MACD signal crossover
    - Price above 50/200 SMA
    """
    
    name = "moving_average"
    description = "Moving average crossover signals"
    
    def __init__(self, lookback_days: int = 5):
        """
        Initialize MA screener.
        
        Args:
            lookback_days: Days to look back for signals
        """
        self.lookback_days = lookback_days
        
        # Strategy definitions
        self.strategies = pd.DataFrame({
            "10_cross_30": [0, 0, 1, 1, 1],
            "MACD_Signal_MACD": [1, 1, 1, 0, 0],
            "MACD_lim": [0, 0, 0, 1, 1],
            "abv_50": [1, 1, 1, 0, 0],
            "abv_200": [0, 1, 0, 0, 1],
            "strategy": [1, 2, 3, 4, 5],
        })
    
    def _add_signal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add signal indicator columns."""
        df = df.copy()
        
        # SMA crossovers
        if "SMA_10" in df.columns and "SMA_30" in df.columns:
            df["10_cross_30"] = np.where(df["SMA_10"] > df["SMA_30"], 1, 0)
        else:
            df["10_cross_30"] = 0
        
        # MACD signal cross
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            df["MACD_Signal_MACD"] = np.where(df["MACD_signal"] < df["MACD"], 1, 0)
            df["MACD_lim"] = np.where(df["MACD"] > 0, 1, 0)
        else:
            df["MACD_Signal_MACD"] = 0
            df["MACD_lim"] = 0
        
        # Above longer SMAs
        if all(col in df.columns for col in ["SMA_10", "SMA_30", "SMA_50"]):
            df["abv_50"] = np.where(
                (df["SMA_30"] > df["SMA_50"]) & (df["SMA_10"] > df["SMA_50"]),
                1, 0
            )
        else:
            df["abv_50"] = 0
        
        if all(col in df.columns for col in ["SMA_10", "SMA_30", "SMA_50", "SMA_200"]):
            df["abv_200"] = np.where(
                (df["SMA_30"] > df["SMA_200"]) &
                (df["SMA_10"] > df["SMA_200"]) &
                (df["SMA_50"] > df["SMA_200"]),
                1, 0
            )
        else:
            df["abv_200"] = 0
        
        return df
    
    def screen(self, ticker: str, data: pd.DataFrame) -> ScreenerResult:
        """
        Screen stock using moving average strategies.
        
        Args:
            ticker: Stock symbol
            data: Price data with SMA/MACD columns
            
        Returns:
            ScreenerResult with matched strategies
        """
        try:
            df = self._add_signal_indicators(data)
            
            recent = df.tail(self.lookback_days).reset_index(drop=True)
            
            if len(recent) == 0:
                return ScreenerResult(
                    ticker=ticker,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    details={"error": "No data"},
                )
            
            # Find matching strategies
            matched_strategies = []
            
            for _, row in recent.iterrows():
                try:
                    row_df = pd.DataFrame([row])
                    strategy_cols = list(self.strategies.columns[:-1])
                    
                    for col in strategy_cols:
                        if col not in row_df.columns:
                            row_df[col] = 0
                    
                    merged = row_df.merge(
                        self.strategies,
                        on=strategy_cols,
                        how="inner"
                    )
                    
                    if len(merged) > 0:
                        matched_strategies.extend(merged["strategy"].tolist())
                        
                except Exception:
                    pass
            
            # Determine signal
            if matched_strategies:
                # More matches = higher confidence
                confidence = min(len(matched_strategies) / 5, 1.0)
                signal = Signal.BUY
            else:
                confidence = 0.3
                signal = Signal.HOLD
            
            # Get current MA values for details
            details = {
                "matched_strategies": list(set(matched_strategies)),
                "strategy_count": len(set(matched_strategies)),
            }
            
            if "SMA_10" in df.columns:
                details["sma_10"] = round(df["SMA_10"].iloc[-1], 2)
            if "SMA_50" in df.columns:
                details["sma_50"] = round(df["SMA_50"].iloc[-1], 2)
            if "SMA_200" in df.columns:
                details["sma_200"] = round(df["SMA_200"].iloc[-1], 2)
            
            return ScreenerResult(
                ticker=ticker,
                signal=signal,
                confidence=confidence,
                details=details,
            )
            
        except Exception as e:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": str(e)},
            )
