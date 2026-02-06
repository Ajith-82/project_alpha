"""
Screeners Package

Provides modular, extensible stock screening functionality.
"""

from .base import (
    BaseScreener,
    ScreenerResult,
    BatchScreenerResult,
    Signal,
)
from .registry import (
    ScreenerRegistry,
    get_registry,
    register_screener,
)
from .macd import MACDScreener
from .breakout import BreakoutScreener
from .donchian import DonchianScreener
from .trendline import TrendlineScreener
from .moving_average import MovingAverageScreener


# Register all built-in screeners
_registry = get_registry()
_registry.register(MACDScreener())
_registry.register(BreakoutScreener())
_registry.register(DonchianScreener())
_registry.register(TrendlineScreener())
_registry.register(MovingAverageScreener())


__all__ = [
    # Base
    "BaseScreener",
    "ScreenerResult",
    "BatchScreenerResult",
    "Signal",
    # Registry
    "ScreenerRegistry",
    "get_registry",
    "register_screener",
    # Screeners
    "MACDScreener",
    "BreakoutScreener",
    "DonchianScreener",
    "TrendlineScreener",
    "MovingAverageScreener",
]
