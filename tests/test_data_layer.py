"""
Unit tests for the Data Layer modules.

Tests DataFetcher, DataCache, and DataValidator functionality.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classes.data.DataValidator import DataValidator, ValidationResult
from classes.data.DataCache import CacheManager
from classes.data.DataFetcher import StockFetcher, FetchResult


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        
        # Create valid sample DataFrame
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        self.valid_df = pd.DataFrame({
            "Open": np.random.uniform(100, 200, 100),
            "High": np.random.uniform(100, 200, 100),
            "Low": np.random.uniform(100, 200, 100),
            "Close": np.random.uniform(100, 200, 100),
            "Adj Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
        }, index=dates.strftime("%Y-%m-%d"))
    
    def test_validate_valid_data(self):
        """Test validation passes for valid data."""
        result = self.validator.validate_price_data(self.valid_df, "AAPL")
        
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.stats["rows"], 100)
    
    def test_validate_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        result = self.validator.validate_price_data(pd.DataFrame(), "AAPL")
        
        self.assertFalse(result.valid)
        self.assertIn("empty", result.errors[0].lower())
    
    def test_validate_none_data(self):
        """Test validation fails for None data."""
        result = self.validator.validate_price_data(None, "AAPL")
        
        self.assertFalse(result.valid)
        self.assertIn("None", result.errors[0])
    
    def test_validate_missing_columns(self):
        """Test validation fails for missing required columns."""
        df = self.valid_df.drop(columns=["Open", "Close"])
        result = self.validator.validate_price_data(df, "AAPL")
        
        self.assertFalse(result.valid)
        self.assertIn("Missing required columns", result.errors[0])
    
    def test_validate_negative_prices(self):
        """Test validation fails for negative prices."""
        df = self.valid_df.copy()
        df.iloc[0, 0] = -100  # Set negative Open
        result = self.validator.validate_price_data(df, "AAPL")
        
        self.assertFalse(result.valid)
        self.assertTrue(any("Negative prices" in e for e in result.errors))
    
    def test_validate_insufficient_rows(self):
        """Test validation fails for insufficient data rows."""
        df = self.valid_df.head(3)  # Only 3 rows, minimum is 5
        result = self.validator.validate_price_data(df, "AAPL")
        
        self.assertFalse(result.valid)
        self.assertIn("Insufficient data", result.errors[0])
    
    def test_validate_excessive_nan(self):
        """Test validation warns for excessive NaN values."""
        df = self.valid_df.copy()
        df.iloc[0:50, :5] = np.nan  # 50% NaN
        result = self.validator.validate_price_data(df, "AAPL")
        
        # Should warn but not fail (unless strict mode)
        self.assertTrue(any("Excessive NaN" in w for w in result.warnings))
    
    def test_validate_company_info(self):
        """Test company info validation."""
        valid_info = {"sector": "Technology", "industry": "Software"}
        result = self.validator.validate_company_info(valid_info, "AAPL")
        
        self.assertTrue(result.valid)
        self.assertTrue(result.stats["has_sector"])
        self.assertTrue(result.stats["has_industry"])
    
    def test_validate_company_info_missing_fields(self):
        """Test company info validation warns for missing fields."""
        info = {"name": "Apple Inc"}  # Missing sector/industry
        result = self.validator.validate_company_info(info, "AAPL")
        
        self.assertTrue(result.valid)  # Still valid but with warnings
        self.assertTrue(any("Missing important fields" in w for w in result.warnings))
    
    def test_filter_valid(self):
        """Test filtering valid data from batch."""
        invalid_df = pd.DataFrame()  # Empty = invalid
        
        price_data = {
            "AAPL": self.valid_df,
            "INVALID": invalid_df,
        }
        
        valid_price, valid_info, removed = self.validator.filter_valid(price_data)
        
        self.assertIn("AAPL", valid_price)
        self.assertNotIn("INVALID", valid_price)
        self.assertIn("INVALID", removed)
    
    def test_get_summary(self):
        """Test summary generation."""
        results = {
            "AAPL": ValidationResult(valid=True),
            "GOOG": ValidationResult(valid=True),
            "BAD": ValidationResult(valid=False, errors=["Error"]),
        }
        
        summary = self.validator.get_summary(results)
        
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["valid"], 2)
        self.assertEqual(summary["invalid"], 1)


class TestCacheManager(unittest.TestCase):
    """Test cases for CacheManager class."""
    
    def setUp(self):
        """Set up test fixtures with temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_set_and_get(self):
        """Test basic set and get operations."""
        data = {"test": "data", "numbers": [1, 2, 3]}
        
        self.assertTrue(self.cache.set("test_key", data))
        
        loaded = self.cache.get("test_key")
        self.assertEqual(loaded["test"], "data")
        self.assertEqual(loaded["numbers"], [1, 2, 3])
    
    def test_get_nonexistent(self):
        """Test get returns None for non-existent key."""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)
    
    def test_is_fresh(self):
        """Test cache freshness check."""
        data = {"test": "data"}
        self.cache.set("fresh_key", data)
        
        self.assertTrue(self.cache.is_fresh("fresh_key"))
        self.assertFalse(self.cache.is_fresh("nonexistent"))
    
    def test_invalidate(self):
        """Test cache invalidation."""
        data = {"test": "data"}
        self.cache.set("invalidate_key", data)
        
        removed = self.cache.invalidate("invalidate_key")
        self.assertEqual(removed, 1)
        
        # Should be None after invalidation
        self.assertIsNone(self.cache.get("invalidate_key"))
    
    def test_market_subdirectory(self):
        """Test cache with market subdirectory."""
        data = {"market": "us"}
        
        self.assertTrue(self.cache.set("sp500", data, market="us"))
        
        loaded = self.cache.get("sp500", market="us")
        self.assertEqual(loaded["market"], "us")
        
        # Should not find in different market
        self.assertIsNone(self.cache.get("sp500", market="india"))


class TestStockFetcher(unittest.TestCase):
    """Test cases for StockFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = StockFetcher(max_retries=2, verbose=False)
    
    def test_format_ticker_india(self):
        """Test ticker formatting for India market."""
        result = self.fetcher._format_ticker("RELIANCE", "india")
        self.assertEqual(result, "RELIANCE.NS")
        
        # Already formatted
        result = self.fetcher._format_ticker("RELIANCE.NS", "india")
        self.assertEqual(result, "RELIANCE.NS")
    
    def test_format_ticker_us(self):
        """Test ticker formatting for US market."""
        result = self.fetcher._format_ticker("aapl", "us")
        self.assertEqual(result, "AAPL")
    
    def test_unformat_ticker(self):
        """Test ticker unformatting."""
        result = self.fetcher._unformat_ticker("RELIANCE.NS", "india")
        self.assertEqual(result, "RELIANCE")
        
        result = self.fetcher._unformat_ticker("AAPL", "us")
        self.assertEqual(result, "AAPL")
    
    @patch("classes.data.DataFetcher.yf.Ticker")
    def test_fetch_one_success(self, mock_ticker):
        """Test successful fetch with mocked yfinance."""
        # Create mock data
        mock_history = pd.DataFrame({
            "Open": [100, 101],
            "High": [105, 106],
            "Low": [99, 100],
            "Close": [104, 105],
            "Volume": [1000000, 1100000],
            "Dividends": [0, 0],
            "Stock Splits": [0, 0],
        }, index=pd.date_range("2024-01-01", periods=2))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker_instance.info = {"sector": "Technology"}
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.fetcher.fetch_one("AAPL", "us", "2024-01-01", "2024-01-02")
        
        self.assertTrue(result.success)
        self.assertEqual(result.ticker, "AAPL")
        self.assertIsNotNone(result.price_data)
        self.assertEqual(result.company_info["sector"], "Technology")
    
    @patch("classes.data.DataFetcher.yf.Ticker")
    def test_fetch_one_empty_data(self, mock_ticker):
        """Test fetch handling empty data."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.fetcher.fetch_one("INVALID", "us")
        
        self.assertFalse(result.success)
        self.assertIn("No data", result.error)
    
    @patch("classes.data.DataFetcher.yf.Ticker")
    def test_fetch_one_retry_on_error(self, mock_ticker):
        """Test retry logic on fetch error."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_ticker_instance
        
        # Use short delays for testing
        fetcher = StockFetcher(max_retries=2, retry_delays=[0.01, 0.02])
        result = fetcher.fetch_one("AAPL", "us")
        
        self.assertFalse(result.success)
        self.assertEqual(result.retries, 2)  # Should have retried twice
    
    def test_results_to_dict(self):
        """Test converting results to legacy format."""
        # Create mock results
        mock_df = pd.DataFrame({"Close": [100, 101]})
        results = {
            "AAPL": FetchResult(
                ticker="AAPL",
                price_data=mock_df,
                company_info={"sector": "Tech"},
                success=True,
            ),
            "FAIL": FetchResult(
                ticker="FAIL",
                price_data=None,
                company_info=None,
                success=False,
                error="Failed",
            ),
        }
        
        converted = self.fetcher.results_to_dict(results)
        
        self.assertIn("AAPL", converted["tickers"])
        self.assertNotIn("FAIL", converted["tickers"])
        self.assertIn("AAPL", converted["price_data"])
        self.assertIn("AAPL", converted["company_info"])
    
    def test_progress_callback(self):
        """Test progress callback is called."""
        callback_calls = []
        
        def callback(ticker, completed, total):
            callback_calls.append((ticker, completed, total))
        
        fetcher = StockFetcher(progress_callback=callback)
        
        # Verify callback is set
        self.assertIsNotNone(fetcher.progress_callback)


class TestFetchResult(unittest.TestCase):
    """Test cases for FetchResult dataclass."""
    
    def test_fetch_result_success(self):
        """Test FetchResult for successful fetch."""
        df = pd.DataFrame({"Close": [100]})
        result = FetchResult(
            ticker="AAPL",
            price_data=df,
            company_info={"sector": "Tech"},
            success=True,
        )
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.retries, 0)
    
    def test_fetch_result_failure(self):
        """Test FetchResult for failed fetch."""
        result = FetchResult(
            ticker="FAIL",
            price_data=None,
            company_info=None,
            success=False,
            error="Network timeout",
            retries=3,
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Network timeout")
        self.assertEqual(result.retries, 3)


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult dataclass."""
    
    def test_validation_result_valid(self):
        """Test ValidationResult for valid data."""
        result = ValidationResult(valid=True)
        
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)
    
    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            valid=False,
            errors=["Error 1", "Error 2"],
        )
        
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 2)


class TestDataTransformer(unittest.TestCase):
    """Test cases for DataTransformer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from classes.data.DataTransformer import DataTransformer, TransformConfig
        self.transformer = DataTransformer()
        
        # Create sample price DataFrame
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        self.sample_df = pd.DataFrame({
            "Open": np.random.uniform(100, 200, 100),
            "High": np.random.uniform(100, 200, 100),
            "Low": np.random.uniform(100, 200, 100),
            "Close": np.random.uniform(100, 200, 100),
            "Adj Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
        }, index=dates.strftime("%Y-%m-%d"))
    
    def test_add_sma(self):
        """Test SMA indicator addition."""
        result = self.transformer.add_sma(self.sample_df, windows=[10, 30])
        
        self.assertIn("SMA_10", result.columns)
        self.assertIn("SMA_30", result.columns)
        # First 9 values should be NaN for SMA_10
        self.assertTrue(pd.isna(result["SMA_10"].iloc[0]))
        # 10th value should be valid
        self.assertFalse(pd.isna(result["SMA_10"].iloc[9]))
    
    def test_add_macd(self):
        """Test MACD indicator addition."""
        result = self.transformer.add_macd(self.sample_df)
        
        self.assertIn("MACD", result.columns)
        self.assertIn("MACD_signal", result.columns)
        self.assertIn("MACD_hist", result.columns)
    
    def test_add_rsi(self):
        """Test RSI indicator addition."""
        result = self.transformer.add_rsi(self.sample_df)
        
        self.assertIn("RSI", result.columns)
    
    def test_add_all_indicators(self):
        """Test adding all indicators."""
        result = self.transformer.add_all_indicators(self.sample_df)
        
        self.assertIn("SMA_10", result.columns)
        self.assertIn("MACD", result.columns)
        self.assertIn("RSI", result.columns)
    
    def test_clean_price_data(self):
        """Test data cleaning."""
        df = self.sample_df.copy()
        df.iloc[0, 0] = np.nan  # Add a NaN
        
        result = self.transformer.clean_price_data(df)
        
        # Should have no NaN values after cleaning
        self.assertEqual(result.isnull().sum().sum(), 0)
    
    def test_remove_high_nan_columns(self):
        """Test removal of high-NaN columns."""
        df = self.sample_df.copy()
        df["BadColumn"] = np.nan  # 100% NaN
        
        result, removed = self.transformer.remove_high_nan_columns(df, threshold=0.5)
        
        self.assertIn("BadColumn", removed)
        self.assertNotIn("BadColumn", result.columns)
    
    def test_extract_volatile_columns(self):
        """Test extracting specific columns."""
        price_data = {
            "AAPL": self.sample_df,
            "MSFT": self.sample_df.copy(),
        }
        
        result = self.transformer.extract_volatile_columns(price_data)
        
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertIn("Adj Close", result["AAPL"].columns)
        self.assertIn("Volume", result["AAPL"].columns)
        self.assertNotIn("Open", result["AAPL"].columns)
    
    def test_combine_price_data(self):
        """Test combining multiple DataFrames."""
        price_data = {
            "AAPL": self.sample_df[["Adj Close", "Volume"]],
            "MSFT": self.sample_df[["Adj Close", "Volume"]],
        }
        
        result = self.transformer.combine_price_data(price_data)
        
        # Should have MultiIndex columns
        self.assertEqual(result.columns.nlevels, 2)
        self.assertIn("AAPL", result.columns.get_level_values(0))
        self.assertIn("MSFT", result.columns.get_level_values(0))
    
    def test_prepare_volatile_data(self):
        """Test full volatile data preparation."""
        price_data = {
            "AAPL": self.sample_df,
            "MSFT": self.sample_df.copy(),
        }
        company_info = {
            "AAPL": {"sector": "Technology", "industry": "Consumer Electronics"},
            "MSFT": {"sector": "Technology", "industry": "Software"},
        }
        
        result = self.transformer.prepare_volatile_data(price_data, company_info, "us")
        
        self.assertIn("tickers", result)
        self.assertIn("price", result)
        self.assertIn("volume", result)
        self.assertIn("sectors", result)
        self.assertEqual(result["default_currency"], "USD")
    
    def test_normalize_prices_minmax(self):
        """Test min-max normalization."""
        result = self.transformer.normalize_prices(self.sample_df, method="minmax")
        
        # Values should be between 0 and 1
        for col in ["Open", "High", "Low", "Close"]:
            self.assertGreaterEqual(result[col].min(), 0)
            self.assertLessEqual(result[col].max(), 1)
    
    def test_calculate_returns(self):
        """Test return calculation."""
        result = self.transformer.calculate_returns(self.sample_df)
        
        self.assertIn("Returns", result.columns)
        self.assertIn("Log_Returns", result.columns)


if __name__ == "__main__":
    unittest.main()

