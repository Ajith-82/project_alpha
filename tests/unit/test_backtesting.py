import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
from unittest.mock import MagicMock

from src.classes.backtesting.engine import BacktestEngine, ProjectAlphaStrategy
from src.classes.screeners.breakout import BreakoutScreener
from src.classes.screeners.trendline import TrendlineScreener
from src.classes.backtesting.performance import BacktestPerformance

# Mock Data Fixture
@pytest.fixture
def mock_ohlcv_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    df = pd.DataFrame({
        'Open': np.linspace(100, 200, 100),
        'High': np.linspace(105, 205, 100),
        'Low': np.linspace(95, 195, 100),
        'Close': np.linspace(100, 200, 100), # Exact uptrend
        'Volume': np.random.randint(1000, 2000, 100)
    }, index=dates)
    return df

def test_backtest_engine_run(mock_ohlcv_data):
    # Test execution with Breakout screener logic
    engine = BacktestEngine(mock_ohlcv_data, initial_capital=10000)
    bt, stats = engine.run(strategy_class=ProjectAlphaStrategy, screener_cls=BreakoutScreener)
    
    assert stats is not None
    assert 'Start' in stats
    assert 'End' in stats
    assert 'Return [%]' in stats

def test_backtest_performance_metrics(mock_ohlcv_data):
    engine = BacktestEngine(mock_ohlcv_data)
    bt, stats = engine.run(strategy_class=ProjectAlphaStrategy, screener_cls=BreakoutScreener)
    
    result = BacktestPerformance.extract_metrics(stats, "MOCK", "Breakout")
    
    assert result.ticker == "MOCK"
    assert result.strategy == "Breakout"
    assert isinstance(result.return_pct, float)
    assert isinstance(result.sharpe_ratio, float)

def test_vectorized_signals_uptrend(mock_ohlcv_data):
    # In uptrend mock data, Close > SMA20 should trigger signals
    # We test via the strategy indirect execution or checking adapter logic directly
    from src.classes.backtesting.adapter import ScreenerSignalAdapter
    
    adapter = ScreenerSignalAdapter(BreakoutScreener())
    # Breakout logic requires Close > SMA20 AND Volume > VolSMA20 etc.
    # Our mock data is perfectly linear uptrend so Close > SMA20 is true after window
    # Volume is random though.
    
    # Let's adjust volume to be increasing to force signal
    mock_ohlcv_data['Volume'] = np.linspace(1000, 2000, 100)
    
    signals = adapter.compute_signal_vectorized_breakout(mock_ohlcv_data)
    
    # Should be 0 for first 20 days (NaN SMA), then 1 because Close > SMA and Vol > SMA
    assert signals.iloc[0] == 0

    # Ensure Day 30 meets all conditions:
    # 1. Close > SMA20 (Already true due to uptrend)
    # 2. Volume > SMA20_Volume (We set increasing volume, so latest > average)
    # 3. Close > Open (Green candle)
    # In our mock data: Open=130.3, Close=130.3 -> Doji. This might fail Close > Open check.
    # Let's force a green candle on day 30
    mock_ohlcv_data.iloc[30, mock_ohlcv_data.columns.get_loc('Close')] = 135
    mock_ohlcv_data.iloc[30, mock_ohlcv_data.columns.get_loc('Open')] = 130
    
    signals = adapter.compute_signal_vectorized_breakout(mock_ohlcv_data)
    assert signals.iloc[30] == 1 # Check a point well into trend
