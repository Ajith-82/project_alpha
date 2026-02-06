"""
Donchian Channel Screener

Screens stocks based on Donchian channel breakouts.
"""

import numpy as np
import pandas as pd

try:
    from ta.volatility import DonchianChannel
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from .base import BaseScreener, ScreenerResult, Signal


class DonchianScreener(BaseScreener):
    """
    Donchian Channel Screener.
    
    Generates signals based on Donchian channel bands:
    - BUY when price touches/breaks lower band
    - SELL when price touches/breaks upper band
    """
    
    name = "donchian"
    description = "Donchian channel breakout signals"
    
    def __init__(
        self,
        window: int = 20,
        screening_period: int = 5,
    ):
        """
        Initialize Donchian screener.
        
        Args:
            window: Donchian channel period
            screening_period: Days to look back for signals
        """
        self.window = window
        self.screening_period = screening_period
    
    def screen(self, ticker: str, data: pd.DataFrame) -> ScreenerResult:
        """
        Screen stock using Donchian channels.
        
        Args:
            ticker: Stock symbol
            data: Price data with OHLCV columns
            
        Returns:
            ScreenerResult with BUY/SELL/HOLD signal
        """
        if not TA_AVAILABLE:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": "ta library not available"},
            )
        
        required = {"High", "Low", "Close"}
        if not required.issubset(set(data.columns)):
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": "Missing required columns"},
            )
        
        try:
            df = data.copy()
            
            # Calculate Donchian channels
            dc = DonchianChannel(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                window=self.window,
            )
            
            df["don_low"] = dc.donchian_channel_lband()
            df["don_mid"] = dc.donchian_channel_mband()
            df["don_high"] = dc.donchian_channel_hband()
            
            # Generate signals
            # 1 = at lower band (buy), -1 = at upper band (sell)
            df["position"] = np.where(
                (df["Close"] == df["don_low"]) | (df["Low"] == df["don_low"]),
                1,
                np.where(
                    (df["Close"] == df["don_high"]) | (df["High"] == df["don_high"]),
                    -1,
                    0,
                ),
            )
            
            # Get recent signals
            signals = df["position"].tail(self.screening_period)
            current_signal = signals.iloc[-1] if len(signals) > 0 else 0
            
            # Calculate confidence
            if len(signals) > 1:
                consistency = abs(signals.sum()) / len(signals)
            else:
                consistency = 0.5
            
            # Get channel details
            latest_close = df["Close"].iloc[-1]
            latest_low = df["don_low"].iloc[-1]
            latest_mid = df["don_mid"].iloc[-1]
            latest_high = df["don_high"].iloc[-1]
            
            # Position in channel (0 = at low, 1 = at high)
            channel_width = latest_high - latest_low
            if channel_width > 0:
                position_in_channel = (latest_close - latest_low) / channel_width
            else:
                position_in_channel = 0.5
            
            if current_signal == 1:
                signal = Signal.BUY
            elif current_signal == -1:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
            
            return ScreenerResult(
                ticker=ticker,
                signal=signal,
                confidence=consistency,
                details={
                    "channel_low": round(latest_low, 2),
                    "channel_mid": round(latest_mid, 2),
                    "channel_high": round(latest_high, 2),
                    "position_in_channel": round(position_in_channel, 2),
                },
            )
            
        except Exception as e:
            return ScreenerResult(
                ticker=ticker,
                signal=Signal.HOLD,
                confidence=0.0,
                details={"error": str(e)},
            )
