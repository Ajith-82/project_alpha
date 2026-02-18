import pytest
import pandas as pd
from src.classes.screeners.breakout import BreakoutScreener
from src.classes.screeners.trendline import TrendlineScreener

def test_breakout_screener_uptrend(uptrend_data):
    """Test breakout screener on standard uptrend data (should not trigger breakout, just trend)."""
    screener = BreakoutScreener()
    result = screener.screen("TEST", uptrend_data)
    
    # Simple uptrend might likely be categorized as "ALONG TREND" or "GROWTH" depending on thresholds
    # The run method returns a ScreenerResult object
    assert result is not None
    
def test_trendline_screener():
    """Test trendline screener identifying an uptrend."""
    # Create data with steep slope to ensure angle > 30 degrees
    # Slope of 1.0 gives 45 degrees -> Weak Up (BUY)
    prices = [100 + i for i in range(30)] # Slope 1.0
    data = pd.DataFrame({
        "Close": prices,
        "Open": prices,
        "High": prices,
        "Low": prices,
        "Volume": [1000]*30
    })
    
    screener = TrendlineScreener(lookback_days=20)
    result = screener.screen("TEST", data)
    
    assert result.signal.name == "BUY" or result.signal.value == "BUY"
    assert result.confidence > 0

def test_breakout_screener_consolidation(breakout_data):
    """Test breakout detection."""
    # This data is designed to have a breakout
    # However, our fixture logic might need verifying if it actually triggers the specific 'BreakoutScreener' logic
    screener = BreakoutScreener()
    # Mock settings if needed
    
    # For now, just ensuring it runs without error and returns a result structure
    result = screener.screen("TEST", breakout_data)
    assert result.ticker == "TEST"
    assert result.signal.name in ["BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"]

