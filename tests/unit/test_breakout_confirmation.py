import pytest
import pandas as pd
import numpy as np
from src.classes.screeners.breakout import BreakoutScreener
from src.classes.screeners.base import Signal

@pytest.fixture
def breakout_data():
    """Create synthetic data with a breakout pattern."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="B")
    data = pd.DataFrame({
        "Open": np.linspace(100, 105, 50),
        "High": np.linspace(101, 106, 50),
        "Low": np.linspace(99, 104, 50),
        "Close": np.linspace(100.5, 105.5, 50),
        "Volume": np.full(50, 1000000),
        "Adj Close": np.linspace(100.5, 105.5, 50)
    }, index=dates)
    
    # Create valid breakout on the last day
    # 1. Bullish candle: Close > Open
    # 2. Max body in 10 days: Large body
    # 3. Low selling pressure: High ~= Close
    # 4. OC > threshold: Large move
    # 5. Volume > threshold: High volume
    
    idx = -1
    data.iloc[idx, data.columns.get_loc("Open")] = 100.0
    data.iloc[idx, data.columns.get_loc("Close")] = 110.0  # +10% move
    data.iloc[idx, data.columns.get_loc("High")] = 110.1   # Tiny wick
    data.iloc[idx, data.columns.get_loc("Low")] = 99.0
    data.iloc[idx, data.columns.get_loc("Volume")] = 2000000 # 2x avg volume
    
    return data

@pytest.fixture
def screener():
    return BreakoutScreener(
        min_adx=20.0,
        atr_expansion_factor=1.5
    )

def test_breakout_confirmed(screener, breakout_data):
    """Test valid breakout with strong ADX and expanding ATR."""
    df = breakout_data.copy()
    
    # Add ADX > 20
    df["ADX"] = 25.0
    
    # Add ATR expansion
    # Mean ATR = 1.0, Current ATR = 2.0 ( > 1.5x)
    df["ATR"] = 1.0
    df.iloc[-1, df.columns.get_loc("ATR")] = 2.0 
    
    result = screener.screen("TEST", df)
    
    assert result.signal == Signal.BUY
    assert result.details["breakout_count"] >= 1
    assert result.details["adx"] == 25.0
    # Mean of 19 ones and 1 two is 1.05. Expansion = 2.0 / 1.05 ~= 1.90
    assert result.details["atr_expansion"] == pytest.approx(1.9, rel=0.1)

def test_breakout_rejected_low_adx(screener, breakout_data):
    """Test breakout rejected due to low ADX (weak trend)."""
    df = breakout_data.copy()
    
    # Weak trend
    df["ADX"] = 15.0
    
    # Valid ATR
    df["ATR"] = 1.0
    df.iloc[-1, df.columns.get_loc("ATR")] = 2.0
    
    result = screener.screen("TEST", df)
    
    # Should hold because ADX < 20
    assert result.signal == Signal.HOLD
    assert "No breakout pattern found" in result.details.get("reason", "")

def test_breakout_rejected_low_atr_expansion(screener, breakout_data):
    """Test breakout rejected due to lack of volatility expansion."""
    df = breakout_data.copy()
    
    # Strong trend
    df["ADX"] = 25.0
    
    # No ATR expansion
    df["ATR"] = 1.0
    df.iloc[-1, df.columns.get_loc("ATR")] = 1.2 # < 1.5x
    
    result = screener.screen("TEST", df)
    
    # Should hold because ATR expansion < 1.5
    assert result.signal == Signal.HOLD

def test_missing_indicators_fallback(screener, breakout_data):
    """Test fallback when ADX/ATR columns are missing (should skip checks)."""
    df = breakout_data.copy()
    # No ADX or ATR columns
    
    result = screener.screen("TEST", df)
    
    # Should BUY because indicators are missing, so checks are skipped
    assert result.signal == Signal.BUY
    assert "adx" not in result.details
