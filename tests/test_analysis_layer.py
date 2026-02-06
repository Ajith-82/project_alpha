"""
Unit tests for the Analysis Layer modules.

Tests TrendAnalyzer, CorrelationAnalyzer, and configuration classes.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classes.analysis.VolatileConfig import VolatileConfig, TrainingConfig, RatingThresholds
from classes.analysis.TrendAnalyzer import (
    TrendAnalyzer,
    softplus,
    estimate_logprice_statistics,
    estimate_price_statistics,
)
from classes.analysis.CorrelationAnalyzer import CorrelationAnalyzer, MatchResult


class TestVolatileConfig(unittest.TestCase):
    """Test cases for VolatileConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VolatileConfig()
        
        self.assertEqual(config.horizon, 5)
        self.assertFalse(config.plot_losses)
        self.assertTrue(config.verbose)
        self.assertIsNone(config.save_model)
        self.assertIsNone(config.load_model)
    
    def test_training_config_defaults(self):
        """Test training config defaults."""
        config = VolatileConfig()
        
        self.assertEqual(config.training.learning_rate, 0.01)
        self.assertEqual(config.training.correlation_steps, 50000)
        self.assertEqual(config.training.trend_steps, 10000)
        self.assertEqual(config.training.order_correlation, 52)
        self.assertEqual(config.training.order_trend, 2)
    
    def test_rating_thresholds(self):
        """Test rating thresholds configuration."""
        thresholds = RatingThresholds()
        
        self.assertEqual(thresholds.highly_below_trend, 3.0)
        self.assertEqual(thresholds.below_trend, 2.0)
        self.assertEqual(thresholds.along_trend, -2.0)
        self.assertEqual(thresholds.above_trend, -3.0)
    
    def test_thresholds_to_dict(self):
        """Test thresholds conversion to dict."""
        thresholds = RatingThresholds()
        d = thresholds.to_dict()
        
        self.assertEqual(d["HIGHLY BELOW TREND"], 3.0)
        self.assertEqual(d["BELOW TREND"], 2.0)
        self.assertEqual(d["ALONG TREND"], -2.0)
        self.assertEqual(d["ABOVE TREND"], -3.0)
    
    def test_from_args(self):
        """Test config creation from args."""
        args = Mock()
        args.plot_losses = True
        args.save_model = "model.pkl"
        args.load_model = None
        args.verbose = False
        
        config = VolatileConfig.from_args(args)
        
        self.assertTrue(config.plot_losses)
        self.assertEqual(config.save_model, "model.pkl")
        self.assertFalse(config.verbose)


class TestSoftplus(unittest.TestCase):
    """Test cases for softplus function."""
    
    def test_positive_input(self):
        """Test softplus with positive input."""
        x = np.array([1.0, 2.0, 3.0])
        result = softplus(x)
        
        # softplus(x) ≈ x for large x
        self.assertTrue(np.all(result > x * 0.9))
    
    def test_zero_input(self):
        """Test softplus at zero."""
        x = np.array([0.0])
        result = softplus(x)
        
        # softplus(0) = log(2) ≈ 0.693
        np.testing.assert_almost_equal(result[0], np.log(2), decimal=5)
    
    def test_negative_input(self):
        """Test softplus with negative input."""
        x = np.array([-10.0])
        result = softplus(x)
        
        # softplus(x) ≈ 0 for very negative x
        self.assertTrue(result[0] < 0.001)
    
    def test_always_positive(self):
        """Test softplus always returns positive."""
        x = np.array([-100, -10, -1, 0, 1, 10, 100])
        result = softplus(x)
        
        self.assertTrue(np.all(result > 0))


class TestEstimateStatistics(unittest.TestCase):
    """Test cases for log-price and price statistics."""
    
    def test_logprice_statistics(self):
        """Test log-price mean and std estimation."""
        mu = np.array([[1.0, 0.1, 0.01]])  # 1 stock, order 2
        sigma = np.array([0.5])  # Volatility param
        tt = np.array([[1], [0.5], [0.25]])  # Time array
        
        mean, std = estimate_logprice_statistics(mu, sigma, tt)
        
        self.assertEqual(mean.shape, (1, 1))
        self.assertEqual(std.shape, (1,))
        self.assertTrue(std[0] > 0)
    
    def test_price_statistics(self):
        """Test price mean and std estimation."""
        logp_mean = np.array([[5.0]])  # log(price) ≈ 5 → price ≈ 148
        logp_std = np.array([0.1])
        
        mean, std = estimate_price_statistics(logp_mean, logp_std)
        
        self.assertTrue(mean[0, 0] > 100)  # exp(5) ≈ 148
        self.assertTrue(std[0, 0] > 0)


class TestTrendAnalyzer(unittest.TestCase):
    """Test cases for TrendAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TrendAnalyzer()
        self.num_stocks = 5
        self.num_timesteps = 100
        
        # Generate sample data
        np.random.seed(42)
        self.logp = np.random.randn(self.num_stocks, self.num_timesteps)
        self.price = np.exp(self.logp)
    
    def test_compute_scores(self):
        """Test z-score computation."""
        logp_pred = np.random.randn(self.num_stocks, 6)
        logp_current = np.random.randn(self.num_stocks, self.num_timesteps)
        std_pred = np.abs(np.random.randn(self.num_stocks)) + 0.1
        
        scores = self.analyzer.compute_scores(
            logp_pred, logp_current, std_pred, horizon=5
        )
        
        self.assertEqual(scores.shape, (self.num_stocks,))
    
    def test_compute_growth(self):
        """Test growth rate computation."""
        phi = np.random.randn(self.num_stocks, 3)  # order 2
        
        growth = self.analyzer.compute_growth(phi, order=2, num_timesteps=100)
        
        self.assertEqual(growth.shape, (self.num_stocks,))
    
    def test_compute_volatility(self):
        """Test volatility computation."""
        std_price = np.abs(np.random.randn(self.num_stocks, self.num_timesteps)) + 1
        
        volatility = self.analyzer.compute_volatility(std_price, self.price)
        
        self.assertEqual(volatility.shape, (self.num_stocks,))
        self.assertTrue(np.all(volatility > 0))
    
    def test_rate_stocks_highly_below(self):
        """Test rating for highly below trend."""
        scores = np.array([4.0])
        ratings = self.analyzer.rate_stocks(scores)
        
        self.assertEqual(ratings[0], "HIGHLY BELOW TREND")
    
    def test_rate_stocks_below(self):
        """Test rating for below trend."""
        scores = np.array([2.5])
        ratings = self.analyzer.rate_stocks(scores)
        
        self.assertEqual(ratings[0], "BELOW TREND")
    
    def test_rate_stocks_along(self):
        """Test rating for along trend."""
        scores = np.array([0.0])
        ratings = self.analyzer.rate_stocks(scores)
        
        self.assertEqual(ratings[0], "ALONG TREND")
    
    def test_rate_stocks_above(self):
        """Test rating for above trend."""
        scores = np.array([-2.5])
        ratings = self.analyzer.rate_stocks(scores)
        
        self.assertEqual(ratings[0], "ABOVE TREND")
    
    def test_rate_stocks_highly_above(self):
        """Test rating for highly above trend."""
        scores = np.array([-4.0])
        ratings = self.analyzer.rate_stocks(scores)
        
        self.assertEqual(ratings[0], "HIGHLY ABOVE TREND")


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test cases for CorrelationAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CorrelationAnalyzer()
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        self.num_stocks = len(self.tickers)
    
    def test_estimate_matches_basic(self):
        """Test basic match estimation."""
        np.random.seed(42)
        mu = np.random.randn(self.num_stocks, 53)  # order 52
        tt = np.linspace(0.01, 1, 100) ** np.arange(53).reshape(-1, 1)
        tt = tt.astype("float32")
        
        matches = self.analyzer.estimate_matches(self.tickers, mu, tt)
        
        self.assertEqual(len(matches), self.num_stocks)
        for ticker in self.tickers:
            self.assertIn(ticker, matches)
            self.assertIsInstance(matches[ticker], MatchResult)
            self.assertIn(matches[ticker].match, self.tickers)
            self.assertNotEqual(matches[ticker].match, ticker)  # Not self-matched
    
    def test_match_result_fields(self):
        """Test MatchResult dataclass fields."""
        result = MatchResult(match="MSFT", index=1, distance=0.5)
        
        self.assertEqual(result.match, "MSFT")
        self.assertEqual(result.index, 1)
        self.assertEqual(result.distance, 0.5)
    
    def test_estimate_clusters(self):
        """Test cluster estimation."""
        np.random.seed(42)
        mu = np.random.randn(self.num_stocks, 53)
        tt = np.linspace(0.01, 1, 100) ** np.arange(53).reshape(-1, 1)
        tt = tt.astype("float32")
        
        clusters = self.analyzer.estimate_clusters(self.tickers, mu, tt)
        
        self.assertEqual(len(clusters), self.num_stocks)
        # Each cluster index should be a non-negative integer
        self.assertTrue(all(c >= 0 for c in clusters))


class TestAnalysisResultDataFrame(unittest.TestCase):
    """Test cases for AnalysisResult DataFrame conversion."""
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        from classes.analysis.VolatileAnalyzer import AnalysisResult
        
        result = AnalysisResult(
            tickers=["AAPL", "MSFT", "GOOGL"],
            scores=np.array([2.5, 0.0, -2.5]),
            growth=np.array([0.01, 0.02, -0.01]),
            volatility=np.array([0.1, 0.2, 0.15]),
            rates=["BELOW TREND", "ALONG TREND", "ABOVE TREND"],
            matches={
                "AAPL": MatchResult("MSFT", 1, 0.5),
                "MSFT": MatchResult("GOOGL", 2, 0.6),
                "GOOGL": MatchResult("AAPL", 0, 0.7),
            },
        )
        
        data = {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "price": np.array([[100, 101, 102], [200, 201, 202], [300, 301, 302]]),
            "currencies": ["USD", "USD", "USD"],
            "sectors": {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech"},
            "industries": {"AAPL": "Hardware", "MSFT": "Software", "GOOGL": "Internet"},
        }
        
        df = result.to_dataframe(data)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("SYMBOL", df.columns)
        self.assertIn("RATE", df.columns)
        self.assertIn("MATCH", df.columns)


if __name__ == "__main__":
    unittest.main()
