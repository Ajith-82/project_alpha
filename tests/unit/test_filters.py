import pytest
from unittest.mock import MagicMock, patch
from src.classes.filters.fundamental_filter import FundamentalFilter
from src.classes.filters.sentiment_filter import SentimentFilter

# --- FundamentalFilter Tests ---

def test_fundamental_filter_no_key():
    """Test safe fallback when no API key is provided."""
    f = FundamentalFilter(api_key=None)
    result = f.check_health("TEST")
    assert result["passed"] is True
    assert result["reason"] == "No API access"

@patch("finnhub.Client")
def test_fundamental_filter_pass(mock_client_cls):
    """Test passing fundamental check."""
    mock_client = MagicMock()
    mock_client.company_basic_financials.return_value = {
        "metric": {
            "totalDebtToEquity": 50.0,  # Healthy
            "netProfitMarginTTM": 10.0
        }
    }
    mock_client_cls.return_value = mock_client
    
    f = FundamentalFilter(api_key="fake_key")
    result = f.check_health("AAPL")
    
    assert result["passed"] is True
    assert result["reason"] == "Fundamentals OK"

@patch("finnhub.Client")
def test_fundamental_filter_fail(mock_client_cls):
    """Test failing fundamental check (High Debt)."""
    mock_client = MagicMock()
    mock_client.company_basic_financials.return_value = {
        "metric": {
            "totalDebtToEquity": 300.0, # > 200 threshold
            "netProfitMarginTTM": 5.0
        }
    }
    mock_client_cls.return_value = mock_client
    
    f = FundamentalFilter(api_key="fake_key")
    result = f.check_health("BAD_STOCK")
    
    assert result["passed"] is False
    assert "High Debt" in result["reason"]

# --- SentimentFilter Tests ---

@patch("src.classes.filters.sentiment_filter.pipeline")
def test_sentiment_filter_positive(mock_pipeline):
    """Test positive sentiment aggregation."""
    mock_classifier = MagicMock()
    mock_classifier.return_value = [
        {"label": "positive", "score": 0.9},
        {"label": "neutral", "score": 0.5}, # 0.0 value
    ]
    mock_pipeline.return_value = mock_classifier
    
    s = SentimentFilter()
    headlines = ["Great earnings", "Meeting tomorrow"]
    result = s.analyze_sentiment(headlines)
    
    # Avg: (0.9 + 0.0) / 2 = 0.45 > 0.15
    assert result["label"] == "positive"
    assert result["score"] == 0.45

@patch("src.classes.filters.sentiment_filter.pipeline")
def test_sentiment_filter_negative(mock_pipeline):
    """Test negative sentiment aggregation."""
    mock_classifier = MagicMock()
    mock_classifier.return_value = [
        {"label": "negative", "score": 0.8},
        {"label": "negative", "score": 0.4},
    ]
    mock_pipeline.return_value = mock_classifier
    
    s = SentimentFilter()
    headlines = ["Lawsuit filed", "CEO resigns"]
    result = s.analyze_sentiment(headlines)
    
    # Avg: (-0.8 + -0.4) / 2 = -0.6 < -0.15
    assert result["label"] == "negative"
    assert result["score"] == -0.6

@patch("src.classes.filters.sentiment_filter.pipeline")
def test_sentiment_filter_error_handling(mock_pipeline):
    """Test robust error handling."""
    mock_pipeline.side_effect = Exception("Model failed")
    
    s = SentimentFilter()
    assert s.classifier is None
    
    result = s.analyze_sentiment(["Some headline"])
    assert result["label"] == "neutral"
    assert result["score"] == 0.0
