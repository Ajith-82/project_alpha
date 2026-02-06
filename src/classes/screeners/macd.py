"""
MACD Screener

Screens stocks based on MACD crossover signals.
"""

import numpy as np
import pandas as pd

from .base import BaseScreener, ScreenerResult, Signal


class MACDScreener(BaseScreener):
    """
    MACD (Moving Average Convergence Divergence) Screener.
    
    Generates buy signals when MACD crosses above signal line,
    and sell signals when MACD crosses below signal line.
    """
    
    name = "macd"
    description = "MACD crossover signals"
    
    def __init__(self, screening_period: int = 5):
        """
        Initialize MACD screener.
        
        Args:
            screening_period: Days to look back for signals
        """
        self.screening_period = screening_period
    
    def screen(self, ticker: str, data: pd.DataFrame) -> ScreenerResult:
        """
        Screen stock using MACD crossover.
        
        Args:
            ticker: Stock symbol
            data: Price data with MACD and MACD_signal columns
            
        Returns:
            ScreenerResult with BUY/SELL/HOLD signal
        """
        if "MACD" not in data.columns or "MACD_signal" not in data.columns:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": "MACD columns not found"},
            )
        
        try:
            # Calculate position: 1 = MACD > Signal, -1 = MACD < Signal
            data = data.copy()
            data["position"] = np.where(
                data["MACD"] > data["MACD_signal"],
                1,
                np.where(data["MACD"] < data["MACD_signal"], -1, 0),
            )
            
            # Get signals for screening period
            signals = data["position"].tail(self.screening_period)
            current_signal = signals.iloc[-1] if len(signals) > 0 else 0
            
            # Calculate confidence based on consistency
            if len(signals) > 1:
                consistency = abs(signals.sum()) / len(signals)
            else:
                consistency = 0.5
            
            # Determine signal
            if current_signal == 1:
                signal = Signal.BUY
            elif current_signal == -1:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
            
            # Get MACD values for details
            latest_macd = data["MACD"].iloc[-1] if len(data) > 0 else 0
            latest_signal_line = data["MACD_signal"].iloc[-1] if len(data) > 0 else 0
            
            return ScreenerResult(
                ticker=ticker,
                signal=signal,
                confidence=consistency,
                details={
                    "macd": round(latest_macd, 4),
                    "signal_line": round(latest_signal_line, 4),
                    "histogram": round(latest_macd - latest_signal_line, 4),
                },
            )
            
        except Exception as e:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": str(e)},
            )
