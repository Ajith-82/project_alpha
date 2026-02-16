import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.classes import Download
from requests.models import Response

@pytest.fixture
def mock_response():
    resp = Response()
    resp.status_code = 200
    resp._content = b'{"chart":{"result":[{"meta":{"currency":"USD","symbol":"AAPL","exchangeName":"NMS","instrumentType":"EQUITY","firstTradeDate":345479400,"regularMarketTime":1616184000,"gmtoffset":-14400,"timezone":"EDT","exchangeTimezoneName":"America/New_York","regularMarketPrice":119.99,"chartPreviousClose":120.53,"priceHint":2,"currentTradingPeriod":{"pre":{"timezone":"EDT","end":1616164200,"start":1616140800,"gmtoffset":-14400},"regular":{"timezone":"EDT","end":1616187600,"start":1616164200,"gmtoffset":-14400},"post":{"timezone":"EDT","end":1616202000,"start":1616187600,"gmtoffset":-14400}},"dataGranularity":"1d","range":"","validRanges":["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]},"timestamp":[1616164200],"indicators":{"quote":[{"low":[119.0],"open":[120.0],"high":[121.0],"volume":[1000],"close":[119.9]}],"adjclose":[{"adjclose":[119.9]}]}}],"error":null}}'
    return resp

def test_download_one_success(mock_response):
    with patch("requests.get", return_value=mock_response) as mock_get:
        data = Download._download_one("AAPL", 1616164200, 1616187600)
        assert data["chart"]["result"][0]["meta"]["symbol"] == "AAPL"
        mock_get.assert_called_once()

def test_parse_quotes():
    data = {
        "timestamp": [1616164200, 1616250600],
        "indicators": {
            "quote": [{
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [95.0, 96.0],
                "close": [102.0, 103.0],
                "volume": [1000, 1200]
            }],
            "adjclose": [{
                "adjclose": [102.0, 103.0]
            }]
        }
    }
    
    df = Download._parse_quotes(data)
    assert not df.empty
    assert len(df) == 2
    assert "Adj Close" in df.columns
    assert "Volume" in df.columns
    # Ensure correct index type (pandas converts to datetime)
    assert isinstance(df.index, pd.Index)

def test_handle_start_end_dates():
    start, end = Download._handle_start_end_dates("2023-01-01", "2023-01-31")
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert start < end

