import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.classes.data.DataFetcher import StockFetcher, FetchResult
from src.classes.Download import download

@pytest.fixture
def sample_price_data():
    dates = pd.date_range("2023-01-01", periods=50)
    # Distinct values to survive drop_duplicates
    prices = [100.0 + i for i in range(50)]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 5 for p in prices],
        "Low": [p - 5 for p in prices],
        "Close": [p + 2 for p in prices],
        "Volume": [10000 + i*100 for i in range(50)]
    }, index=dates)
    return df

def test_fetch_one_success(sample_price_data):
    """Test successful data fetch."""
    fetcher = StockFetcher(verbose=True)
    fetcher.provider = MagicMock()
    fetcher.provider.fetch_data.return_value = sample_price_data
    fetcher.provider.get_company_info.return_value = {"sector": "Technology"}
    
    result = fetcher.fetch_one("TEST", "us")
    
    assert result.success is True
    assert result.ticker == "TEST"
    assert result.price_data is not None
    assert len(result.price_data) == 50
    assert result.company_info == {"sector": "Technology"}
    assert result.retries == 0

def test_fetch_one_retry_success(sample_price_data):
    """Test fetch succeeds after retry."""
    fetcher = StockFetcher(max_retries=2, retry_delays=[0.01, 0.01])
    fetcher.provider = MagicMock()
    # First call raises Exception, second returns data
    fetcher.provider.fetch_data.side_effect = [Exception("Network Error"), sample_price_data]
    fetcher.provider.get_company_info.return_value = {}
    
    result = fetcher.fetch_one("TEST", "us")
    
    assert result.success is True
    assert result.retries == 1
    assert result.price_data is not None

def test_fetch_one_retry_exhaustion():
    """Test fetch fails after max retries."""
    fetcher = StockFetcher(max_retries=2, retry_delays=[0.01, 0.01])
    fetcher.provider = MagicMock()
    fetcher.provider.fetch_data.side_effect = Exception("Persistent Error")
    
    result = fetcher.fetch_one("TEST", "us")
    
    assert result.success is False
    assert result.retries == 2
    assert "Persistent Error" in result.error

def test_download_integration_batch(sample_price_data):
    """Test high-level download function with batch processing."""
    tickers = ["AAPL", "GOOGL"]
    
    # Mock the provider at the StockFetcher level
    with patch("src.classes.Download.StockFetcher") as MockFetcherClass:
        mock_fetcher = MockFetcherClass.return_value
        
        # Mock fetch_batch to return FetchResult objects
        mock_fetcher.fetch_batch.return_value = {
            "AAPL": FetchResult(
                ticker="AAPL",
                price_data=sample_price_data.copy(),
                company_info={"sector": "Technology"},
                success=True,
            ),
            "GOOGL": FetchResult(
                ticker="GOOGL",
                price_data=sample_price_data.copy(),
                company_info={"sector": "Communication"},
                success=True,
            ),
        }
        
        with patch("src.classes.Download._save_to_db"):
            results = download(market="us", tickers=tickers, use_rich_progress=False)
         
    assert len(results) > 0
    assert "AAPL" in results["price_data"]
    assert "GOOGL" in results["price_data"]
