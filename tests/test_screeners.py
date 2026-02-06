"""
Unit tests for the Screeners package.

Tests BaseScreener, ScreenerRegistry, and individual screener implementations.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classes.screeners.base import BaseScreener, ScreenerResult, BatchScreenerResult, Signal
from classes.screeners.registry import ScreenerRegistry, get_registry
from classes.screeners.macd import MACDScreener
from classes.screeners.breakout import BreakoutScreener
from classes.screeners.trendline import TrendlineScreener
from classes.screeners.moving_average import MovingAverageScreener


class TestSignal(unittest.TestCase):
    """Test cases for Signal enum."""
    
    def test_signal_values(self):
        """Test signal enum values."""
        self.assertEqual(Signal.BUY.value, "BUY")
        self.assertEqual(Signal.SELL.value, "SELL")
        self.assertEqual(Signal.HOLD.value, "HOLD")
        self.assertEqual(Signal.STRONG_BUY.value, "STRONG_BUY")
        self.assertEqual(Signal.STRONG_SELL.value, "STRONG_SELL")


class TestScreenerResult(unittest.TestCase):
    """Test cases for ScreenerResult."""
    
    def test_basic_result(self):
        """Test basic result creation."""
        result = ScreenerResult(
            ticker="AAPL",
            signal=Signal.BUY,
            confidence=0.8,
        )
        
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.signal, Signal.BUY)
        self.assertEqual(result.confidence, 0.8)
    
    def test_confidence_clamping(self):
        """Test confidence is clamped to [0, 1]."""
        result = ScreenerResult(ticker="AAPL", signal=Signal.BUY, confidence=1.5)
        self.assertEqual(result.confidence, 1.0)
        
        result = ScreenerResult(ticker="AAPL", signal=Signal.BUY, confidence=-0.5)
        self.assertEqual(result.confidence, 0.0)
    
    def test_is_bullish(self):
        """Test bullish classification."""
        result = ScreenerResult(ticker="AAPL", signal=Signal.BUY)
        self.assertTrue(result.is_bullish)
        
        result = ScreenerResult(ticker="AAPL", signal=Signal.STRONG_BUY)
        self.assertTrue(result.is_bullish)
        
        result = ScreenerResult(ticker="AAPL", signal=Signal.HOLD)
        self.assertFalse(result.is_bullish)
    
    def test_is_bearish(self):
        """Test bearish classification."""
        result = ScreenerResult(ticker="AAPL", signal=Signal.SELL)
        self.assertTrue(result.is_bearish)
        
        result = ScreenerResult(ticker="AAPL", signal=Signal.STRONG_SELL)
        self.assertTrue(result.is_bearish)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ScreenerResult(
            ticker="AAPL",
            signal=Signal.BUY,
            confidence=0.8,
            details={"extra": "value"},
            screener_name="test",
        )
        
        d = result.to_dict()
        self.assertEqual(d["ticker"], "AAPL")
        self.assertEqual(d["signal"], "BUY")
        self.assertEqual(d["confidence"], 0.8)
        self.assertEqual(d["screener"], "test")
        self.assertEqual(d["extra"], "value")


class TestBatchScreenerResult(unittest.TestCase):
    """Test cases for BatchScreenerResult."""
    
    def setUp(self):
        """Create sample results."""
        self.results = [
            ScreenerResult("AAPL", Signal.BUY, 0.8),
            ScreenerResult("MSFT", Signal.SELL, 0.7),
            ScreenerResult("GOOGL", Signal.HOLD, 0.5),
            ScreenerResult("AMZN", Signal.STRONG_BUY, 0.9),
        ]
        self.batch = BatchScreenerResult(self.results, "test")
    
    def test_buys(self):
        """Test buy signal filtering."""
        buys = self.batch.buys
        self.assertEqual(len(buys), 2)  # AAPL, AMZN
    
    def test_sells(self):
        """Test sell signal filtering."""
        sells = self.batch.sells
        self.assertEqual(len(sells), 1)  # MSFT
    
    def test_holds(self):
        """Test hold signal filtering."""
        holds = self.batch.holds
        self.assertEqual(len(holds), 1)  # GOOGL
    
    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        df = self.batch.to_dataframe()
        self.assertEqual(len(df), 4)
        self.assertIn("ticker", df.columns)
        self.assertIn("signal", df.columns)


class TestScreenerRegistry(unittest.TestCase):
    """Test cases for ScreenerRegistry."""
    
    def setUp(self):
        """Create fresh registry."""
        self.registry = ScreenerRegistry()
        self.registry.clear()
    
    def tearDown(self):
        """Clean up."""
        self.registry.clear()
    
    def test_register_and_get(self):
        """Test registering and retrieving screeners."""
        screener = MACDScreener()
        self.registry.register(screener)
        
        retrieved = self.registry.get("macd")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "macd")
    
    def test_list_available(self):
        """Test listing available screeners."""
        self.registry.register(MACDScreener())
        self.registry.register(BreakoutScreener())
        
        available = self.registry.list_available()
        self.assertIn("macd", available)
        self.assertIn("breakout", available)
    
    def test_contains(self):
        """Test containment check."""
        self.registry.register(MACDScreener())
        
        self.assertIn("macd", self.registry)
        self.assertNotIn("unknown", self.registry)


class TestMACDScreener(unittest.TestCase):
    """Test cases for MACDScreener."""
    
    def setUp(self):
        """Create sample data."""
        self.screener = MACDScreener(screening_period=5)
        
        # Create sample MACD data
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        self.data = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 20),
            "High": np.random.uniform(105, 115, 20),
            "Low": np.random.uniform(95, 105, 20),
            "Close": np.random.uniform(100, 110, 20),
            "Volume": np.random.uniform(1000000, 2000000, 20),
            "MACD": np.linspace(-1, 1, 20),  # Crossing from negative to positive
            "MACD_signal": np.zeros(20),  # Zero line
        }, index=dates)
    
    def test_screen_buy_signal(self):
        """Test buy signal generation."""
        # MACD above signal line
        self.data["MACD"] = np.ones(20)
        self.data["MACD_signal"] = np.zeros(20)
        
        result = self.screener.screen("AAPL", self.data)
        
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.signal, Signal.BUY)
    
    def test_screen_sell_signal(self):
        """Test sell signal generation."""
        # MACD below signal line
        self.data["MACD"] = -np.ones(20)
        self.data["MACD_signal"] = np.zeros(20)
        
        result = self.screener.screen("AAPL", self.data)
        
        self.assertEqual(result.signal, Signal.SELL)
    
    def test_missing_columns(self):
        """Test handling of missing MACD columns."""
        data = pd.DataFrame({"Close": [100, 101, 102]})
        result = self.screener.screen("AAPL", data)
        
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertIn("error", result.details)


class TestBreakoutScreener(unittest.TestCase):
    """Test cases for BreakoutScreener."""
    
    def setUp(self):
        """Create sample data."""
        self.screener = BreakoutScreener(min_avg_volume=10000)
        
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        self.data = pd.DataFrame({
            "Open": np.full(30, 100.0),
            "High": np.full(30, 105.0),
            "Low": np.full(30, 95.0),
            "Close": np.full(30, 102.0),
            "Volume": np.full(30, 100000),
        }, index=dates)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        short_data = self.data.head(10)
        result = self.screener.screen("AAPL", short_data)
        
        self.assertEqual(result.signal, Signal.HOLD)
    
    def test_no_breakout(self):
        """Test no breakout pattern."""
        result = self.screener.screen("AAPL", self.data)
        
        # With flat data, no breakout should be detected
        self.assertEqual(result.signal, Signal.HOLD)


class TestTrendlineScreener(unittest.TestCase):
    """Test cases for TrendlineScreener."""
    
    def setUp(self):
        """Create sample data."""
        self.screener = TrendlineScreener(lookback_days=5)
    
    def test_strong_uptrend(self):
        """Test strong uptrend detection."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "Close": [100, 120, 140, 160, 180, 200, 220, 240, 260, 280],
        }, index=dates)
        
        result = self.screener.screen("AAPL", data)
        
        self.assertIn(result.signal, [Signal.BUY, Signal.STRONG_BUY])
        self.assertEqual(result.details["trend"], "Strong Up")
    
    def test_strong_downtrend(self):
        """Test strong downtrend detection."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "Close": [280, 260, 240, 220, 200, 180, 160, 140, 120, 100],
        }, index=dates)
        
        result = self.screener.screen("AAPL", data)
        
        self.assertIn(result.signal, [Signal.SELL, Signal.STRONG_SELL])
        self.assertEqual(result.details["trend"], "Strong Down")
    
    def test_sideways(self):
        """Test sideways trend detection."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "Close": [100, 101, 100, 101, 100, 101, 100, 101, 100, 101],
        }, index=dates)
        
        result = self.screener.screen("AAPL", data)
        
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertEqual(result.details["trend"], "Sideways")
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        data = pd.DataFrame({"Close": [100, 101]})
        result = self.screener.screen("AAPL", data)
        
        self.assertEqual(result.signal, Signal.HOLD)


class TestMovingAverageScreener(unittest.TestCase):
    """Test cases for MovingAverageScreener."""
    
    def setUp(self):
        """Create sample data."""
        self.screener = MovingAverageScreener(lookback_days=5)
        
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        self.data = pd.DataFrame({
            "Close": np.linspace(100, 110, 10),
            "SMA_10": np.full(10, 105),
            "SMA_30": np.full(10, 100),
            "SMA_50": np.full(10, 95),
            "SMA_200": np.full(10, 90),
            "MACD": np.ones(10),
            "MACD_signal": np.zeros(10),
        }, index=dates)
    
    def test_screen_with_indicators(self):
        """Test screening with all indicators present."""
        result = self.screener.screen("AAPL", self.data)
        
        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(result.ticker, "AAPL")
    
    def test_screen_without_indicators(self):
        """Test screening without SMA columns."""
        data = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})
        result = self.screener.screen("AAPL", data)
        
        self.assertEqual(result.signal, Signal.HOLD)


class TestBaseScreenerBatch(unittest.TestCase):
    """Test base screener batch processing."""
    
    def test_batch_screening(self):
        """Test batch screening of multiple stocks."""
        screener = MACDScreener()
        
        tickers = ["AAPL", "MSFT", "GOOGL"]
        price_data = {}
        
        for ticker in tickers:
            dates = pd.date_range("2024-01-01", periods=20, freq="D")
            price_data[ticker] = pd.DataFrame({
                "Close": np.random.uniform(100, 110, 20),
                "MACD": np.random.uniform(-1, 1, 20),
                "MACD_signal": np.zeros(20),
            }, index=dates)
        
        batch_result = screener.screen_batch(tickers, price_data)
        
        self.assertEqual(len(batch_result.results), 3)
        self.assertEqual(batch_result.screener_name, "macd")


if __name__ == "__main__":
    unittest.main()
