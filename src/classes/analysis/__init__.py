"""
Analysis Package

Provides modular components for stock volatility analysis.
"""

from .VolatileConfig import VolatileConfig, TrainingConfig, RatingThresholds
from .TrendAnalyzer import (
    TrendAnalyzer,
    TrendResult,
    softplus,
    estimate_logprice_statistics,
    estimate_price_statistics,
)
from .CorrelationAnalyzer import (
    CorrelationAnalyzer,
    CorrelationResult,
    MatchResult,
)
from .VolatileAnalyzer import VolatileAnalyzer, AnalysisResult


__all__ = [
    # Config
    "VolatileConfig",
    "TrainingConfig",
    "RatingThresholds",
    # Trend
    "TrendAnalyzer",
    "TrendResult",
    "softplus",
    "estimate_logprice_statistics",
    "estimate_price_statistics",
    # Correlation
    "CorrelationAnalyzer",
    "CorrelationResult",
    "MatchResult",
    # Main
    "VolatileAnalyzer",
    "AnalysisResult",
]
