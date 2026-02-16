import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.classes.data.DataFetcher import StockFetcher, FetchResult
from src.classes.Download import download

@pytest.fixture
def mock_yf_ticker():
    with patch("src.classes.data.DataFetcher.yf.Ticker") as mock:
        yield mock

@pytest.fixture
def sample_price_data():
    dates = pd.date_range("2023-01-01", periods=10)
    # Distinct values to survive drop_duplicates
    prices = [100.0 + i for i in range(10)]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 5 for p in prices],
        "Low": [p - 5 for p in prices],
        "Close": [p + 2 for p in prices],
        "Volume": [10000 + i*100 for i in range(10)]
    }, index=dates)
    return df

def test_fetch_one_success(mock_yf_ticker, sample_price_data):
    """Test successful data fetch."""
    # Setup mock
    mock_instance = mock_yf_ticker.return_value
    mock_instance.history.return_value = sample_price_data
    mock_instance.info = {"sector": "Technology"}
    
    fetcher = StockFetcher(verbose=True)
    result = fetcher.fetch_one("TEST", "us")
    
    assert result.success is True
    assert result.ticker == "TEST"
    assert result.price_data is not None
    assert len(result.price_data) == 10
    assert result.company_info == {"sector": "Technology"}
    assert result.retries == 0

def test_fetch_one_retry_success(mock_yf_ticker, sample_price_data):
    """Test fetch succeeds after retry."""
    mock_instance = mock_yf_ticker.return_value
    # First call raises Exception, second returns data
    mock_instance.history.side_effect = [Exception("Network Error"), sample_price_data]
    mock_instance.info = {}
    
    fetcher = StockFetcher(max_retries=2, retry_delays=[0.1, 0.1])
    result = fetcher.fetch_one("TEST", "us")
    
    assert result.success is True
    assert result.retries == 1
    assert result.price_data is not None

def test_fetch_one_retry_exhaustion(mock_yf_ticker):
    """Test fetch fails after max retries."""
    mock_instance = mock_yf_ticker.return_value
    mock_instance.history.side_effect = Exception("Persistent Error")
    
    fetcher = StockFetcher(max_retries=2, retry_delays=[0.1, 0.1])
    result = fetcher.fetch_one("TEST", "us")
    
    assert result.success is False
    assert result.retries == 2
    assert "Persistent Error" in result.error

def test_download_integration_batch(mock_yf_ticker, sample_price_data):
    """Test high-level download function with batch processing."""
    mock_instance = mock_yf_ticker.return_value
    mock_instance.history.return_value = sample_price_data
    mock_instance.info = {}
    
    # download() calls StockFetcher internally
    # Validation logic in download() requires checks
    # Validation in Download.py: validator.validate_price_data
    # nan_threshold=0.33, min_rows=5
    # sample_data has 10 rows, no NaNs
    
    tickers = ["AAPL", "GOOGL"]
    
    # Mocking DatabaseManager saving to avoid DB requirement
    with patch("src.classes.Download._save_to_db") as mock_db:
         results = download(market="us", tickers=tickers, use_rich_progress=False)
         
    assert len(results) > 0 # Should contain 'price_data' etc
    assert "AAPL" in results["price_data"]
    assert "GOOGL" in results["price_data"]
