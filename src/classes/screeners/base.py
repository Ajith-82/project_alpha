"""
Screener Base Classes

Provides abstract base class and result types for all screeners.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import pandas as pd


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class ScreenerResult:
    """
    Result from a single stock screening.
    
    Attributes:
        ticker: Stock symbol
        signal: Trading signal (BUY/SELL/HOLD)
        confidence: Confidence score (0.0 - 1.0)
        details: Additional screening details
        screener_name: Name of the screener that produced this result
    """
    ticker: str
    signal: Signal
    confidence: float = 0.5
    details: Dict[str, Any] = field(default_factory=dict)
    screener_name: str = ""
    
    def __post_init__(self):
        """Validate confidence is in range."""
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.signal in (Signal.BUY, Signal.STRONG_BUY)
    
    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.signal in (Signal.SELL, Signal.STRONG_SELL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "screener": self.screener_name,
            **self.details,
        }


@dataclass
class BatchScreenerResult:
    """Results from batch screening multiple stocks."""
    results: List[ScreenerResult]
    screener_name: str
    
    @property
    def buys(self) -> List[ScreenerResult]:
        """Get all buy signals."""
        return [r for r in self.results if r.is_bullish]
    
    @property
    def sells(self) -> List[ScreenerResult]:
        """Get all sell signals."""
        return [r for r in self.results if r.is_bearish]
    
    @property
    def holds(self) -> List[ScreenerResult]:
        """Get all hold signals."""
        return [r for r in self.results if r.signal == Signal.HOLD]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])


class BaseScreener(ABC):
    """
    Abstract base class for all screeners.
    
    Subclasses must implement:
    - name: Unique screener identifier
    - description: Human-readable description
    - screen(): Screen a single stock
    """
    
    name: str = "base"
    description: str = "Base screener"
    
    @abstractmethod
    def screen(self, ticker: str, data: pd.DataFrame) -> ScreenerResult:
        """
        Screen a single stock.
        
        Args:
            ticker: Stock symbol
            data: Price data DataFrame with OHLCV columns
            
        Returns:
            ScreenerResult with signal and details
        """
        pass
    
    def screen_batch(
        self,
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
    ) -> BatchScreenerResult:
        """
        Screen multiple stocks.
        
        Args:
            tickers: List of stock symbols
            price_data: Dict mapping ticker → DataFrame
            
        Returns:
            BatchScreenerResult with all results
        """
        results = []
        for ticker in tickers:
            if ticker in price_data:
                try:
                    result = self.screen(ticker, price_data[ticker])
                    result.screener_name = self.name
                    results.append(result)
                except Exception as e:
                    # Log error but continue
                    results.append(ScreenerResult(
                        ticker=ticker,
                        signal=Signal.HOLD,
                        confidence=0.0,
                        details={"error": str(e)},
                        screener_name=self.name,
                    ))
        
        return BatchScreenerResult(results=results, screener_name=self.name)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            True if valid
        """
        required = {"Open", "High", "Low", "Close", "Volume"}
        return required.issubset(set(data.columns))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
