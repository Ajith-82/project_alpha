import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.project_alpha import run_screening

@pytest.fixture
def mock_args():
    args = MagicMock()
    # Default args
    args.market = "us"
    args.symbols = None
    args.cache = False
    args.no_plots = True
    args.rank = "growth"
    args.backtest = False
    args.quiet = True
    # args.db_path = None # Optional
    # args.load_model = None
    # args.save_model = None
    return args

@pytest.fixture
def sample_market_data():
    # Create valid data structure for load_data return
    dates = pd.date_range("2023-01-01", periods=30)
    prices = [100 + i for i in range(30)]
    df = pd.DataFrame({
        "Close": prices,
        "Open": prices,
        "High": prices,
        "Low": prices,
        "Volume": [1000]*30,
        "Adj Close": prices
    }, index=dates)
    
    return {
        "tickers": ["TEST"],
        "price_data": {"TEST": df},
        "company_info": {"TEST": {"sector": "Tech", "industry": "Software"}}
    }

def test_pipeline_execution(mock_args, sample_market_data):
    """Test full pipeline execution with mocked data."""
    
    # load_data is imported directly in project_alpha.py
    with patch("src.project_alpha.load_data", return_value=sample_market_data) as mock_load:
        with patch("src.project_alpha.settings") as mock_settings:
            # Setup settings to run a specific screener
            mock_settings.screeners = ["trendline"]
            mock_settings.trend_lookback_days = 20
            
            # Mock console to avoid clutter
            with patch("src.project_alpha.console") as mock_console:
                # tools is imported as 'tools', so we patch tools.save_dict_with_timestamp
                # checking if tools is imported or specific function
                # The code has 'import classes.Tools as tools'
                with patch("src.project_alpha.tools.save_dict_with_timestamp") as mock_save:
                    
                    # Run the function
                    run_screening(mock_args)
                    
                    # Assertions
                    mock_load.assert_called_once()
                    pass
