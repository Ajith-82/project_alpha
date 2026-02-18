import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
from datetime import datetime
import pandas as pd
from unittest.mock import patch, MagicMock
from src.classes.data.provider_chain import YFinanceProvider, PolygonProvider

class TestYFinanceProvider:
    @pytest.fixture
    def provider(self):
        return YFinanceProvider()

    @pytest.fixture
    def mock_yf_download(self):
        with patch('yfinance.download') as mock:
            yield mock

    def test_fetch_data_success(self, provider, mock_yf_download):
        # Setup mock data
        mock_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [99.0, 100.0],
            'Close': [102.0, 103.0],
            'Volume': [1000, 1100],
            'Adj Close': [102.0, 103.0] # Should be ignored or handled
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        
        mock_yf_download.return_value = mock_df

        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 3)
        result = provider.fetch_data('AAPL', start, end)

        # Assertions
        assert not result.empty
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert len(result) == 2
        assert result.index.freq is None # or whatever pandas infers

        mock_yf_download.assert_called_once_with(
            'AAPL', start=start, end=end, progress=False, auto_adjust=False
        )

    def test_fetch_data_returns_empty(self, provider, mock_yf_download):
        mock_yf_download.return_value = pd.DataFrame()
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        result = provider.fetch_data('INVALID', start, end)
        
        assert result.empty

    def test_fetch_data_handles_multiindex_columns(self, provider, mock_yf_download):
        # yfinance sometimes returns MultiIndex columns (Price, Ticker)
        mock_df = pd.DataFrame({
            ('Open', 'AAPL'): [100.0],
            ('High', 'AAPL'): [105.0],
            ('Low', 'AAPL'): [99.0],
            ('Close', 'AAPL'): [102.0],
            ('Volume', 'AAPL'): [1000]
        }, index=pd.to_datetime(['2023-01-01']))
        
        mock_yf_download.return_value = mock_df

        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        result = provider.fetch_data('AAPL', start, end)

        assert not result.empty
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    def test_fetch_data_missing_columns(self, provider, mock_yf_download):
        # Case where returned data is missing required columns
        mock_df = pd.DataFrame({
            'Open': [100.0],
            # Missing High, Low, Close, Volume
        }, index=pd.to_datetime(['2023-01-01']))
        
        mock_yf_download.return_value = mock_df
        
    
        with pytest.raises(KeyError):
             provider.fetch_data('AAPL', datetime(2023, 1, 1), datetime(2023, 1, 2))

    def test_check_health_success(self, provider):
        # Mock yf.Ticker(...).history(...)
        with patch('yfinance.Ticker') as mock_ticker_cls:
            mock_ticker = mock_ticker_cls.return_value
            # Non-empty dataframe check
            mock_ticker.history.return_value = pd.DataFrame({'Close': [100]})
            
            assert provider.check_health() is True
            mock_ticker.history.assert_called_with(period="1d")

    def test_check_health_failure(self, provider):
        with patch('yfinance.Ticker') as mock_ticker_cls:
            mock_ticker = mock_ticker_cls.return_value
            # Empty dataframe check
            mock_ticker.history.return_value = pd.DataFrame()
            
            assert provider.check_health() is False
        
        # Exception check
        with patch('yfinance.Ticker') as mock_ticker_cls:
            mock_ticker_cls.side_effect = Exception("API error")
            assert provider.check_health() is False

    def test_get_company_info(self, provider):
        with patch('yfinance.Ticker') as mock_ticker_cls:
            mock_info = {"sector": "Technology", "symbol": "AAPL"}
            mock_ticker_cls.return_value.info = mock_info
            
            info = provider.get_company_info("AAPL")
            assert info == mock_info
            mock_ticker_cls.assert_called_with("AAPL")

    def test_get_company_info_failure(self, provider):
        with patch('yfinance.Ticker') as mock_ticker_cls:
            mock_ticker_cls.side_effect = Exception("API Error")
            assert provider.get_company_info("AAPL") == {}



class TestPolygonProvider:
    @pytest.fixture
    def mock_client(self):
        with patch('src.classes.data.provider_chain.RESTClient') as mock:
            yield mock.return_value

    @pytest.fixture
    def provider(self, mock_client):
        return PolygonProvider(api_key="fake_key")

    def test_fetch_data_success(self, provider, mock_client):
        # Mock Agg objects
        agg1 = MagicMock(timestamp=1672531200000, open=100, high=105, low=99, close=102, volume=1000)
        agg2 = MagicMock(timestamp=1672617600000, open=101, high=106, low=100, close=103, volume=1100)
        
        mock_client.list_aggs.return_value = [agg1, agg2]

        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 3)
        result = provider.fetch_data('AAPL', start, end)

        assert not result.empty
        assert len(result) == 2
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert result.index[0] == datetime.fromtimestamp(1672531200)

        mock_client.list_aggs.assert_called_once()
        args, kwargs = mock_client.list_aggs.call_args
        assert args[0] == 'AAPL'
        assert kwargs['limit'] == 50000

    def test_fetch_data_empty(self, provider, mock_client):
        mock_client.list_aggs.return_value = []
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        result = provider.fetch_data('AAPL', start, end)

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_check_health_success(self, provider, mock_client):
        # Mock get_last_trade returning something valid
        mock_client.get_last_trade.return_value = MagicMock()
        
        assert provider.check_health() is True
        mock_client.get_last_trade.assert_called_with(ticker="SPY")

    def test_check_health_failure(self, provider, mock_client):
        # Mock failure (returns None or raises)
        mock_client.get_last_trade.return_value = None
        assert provider.check_health() is False

        # Exception
        mock_client.get_last_trade.side_effect = Exception("API Error")
        assert provider.check_health() is False

    def test_get_company_info(self, provider, mock_client):
        # Mock details object
        mock_details = MagicMock()
        mock_details.sic_description = "Technology"
        mock_details.market_cap = 1000000
        
        mock_client.get_ticker_details.return_value = mock_details
        
        info = provider.get_company_info("AAPL")
        assert info["sector"] == "Technology"
        assert info["marketCap"] == 1000000
        
        mock_client.get_ticker_details.assert_called_with("AAPL")

    def test_get_company_info_failure(self, provider, mock_client):
        mock_client.get_ticker_details.side_effect = Exception("API Error")
        assert provider.get_company_info("AAPL") == {}



