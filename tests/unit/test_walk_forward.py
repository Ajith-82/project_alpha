import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from classes.backtesting.walk_forward import WalkForwardValidator
from classes.screeners.base import BaseScreener

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2020-01-01", end="2022-01-01", freq="D")
    df = pd.DataFrame({
        "Open": 100,
        "High": 105,
        "Low": 95,
        "Close": 100,
        "Volume": 1000,
        "Dividends": 0.0,
        "Stock Splits": 0.0
    }, index=dates)
    return df

class MockScreener(BaseScreener):
    def run(self, data):
        return pd.Series(0, index=data.index) # Neutral signal

class TestWalkForwardValidator:
    
    def test_initialization(self, sample_data):
        validator = WalkForwardValidator(sample_data, train_period_days=365, test_period_days=90)
        assert validator.train_period.days == 365
        assert validator.test_period.days == 90
        
    def test_generate_windows(self, sample_data):
        # 2 years of data (~730 days)
        # Train 365, Test 90
        # Window 1: Train [0, 365), Test [365, 455)
        # Window 2: Train [0, 455), Test [455, 545)
        # ...
        
        validator = WalkForwardValidator(sample_data, train_period_days=365, test_period_days=90)
        windows = list(validator.generate_windows())
        
        assert len(windows) > 0
        
        # Check first window
        w1_train, w1_test, (t1_start, t1_end) = windows[0]
        assert len(w1_train) >= 365 # Approx due to freq
        # Test start should be train end
        assert t1_start == w1_train.index[-1] + pd.Timedelta(days=1)
        # Or reasonably close depending on slicing logic. 
        # Logic: train < current_train_end, test >= test_start (which is current_train_end)
        # So test starts exactly where train ends (exclusive).
        
        # Check expansion
        w2_train, w2_test, _ = windows[1]
        assert len(w2_train) > len(w1_train) # Expanding window
        
    def test_validate_flow(self, sample_data):
        validator = WalkForwardValidator(sample_data, train_period_days=300, test_period_days=50)
        
        # Mock BacktestEngine to avoid running actual backtests (slow/complex)
        with patch("classes.backtesting.walk_forward.BacktestEngine") as MockEngine:
            # Setup mock stats
            mock_stats = {
                "Return [%]": 10.0,
                "Sharpe Ratio": 1.5,
                "Max. Drawdown [%]": 5.0
            }
            MockEngine.return_value.run.return_value = (None, mock_stats)
            
            results = validator.validate(MockScreener)
            
            assert len(results) > 0
            assert "IS_Sharpe" in results[0]
            assert "OOS_Sharpe" in results[0]
            assert results[0]["IS_Sharpe"] == 1.5
            
    def test_get_summary(self, sample_data):
        validator = WalkForwardValidator(sample_data)
        validator.results = [
            {"IS_Sharpe": 2.0, "OOS_Sharpe": 1.0}, # Deg = 0.5
            {"IS_Sharpe": 2.0, "OOS_Sharpe": 0.5}, # Deg = 0.25 (Overfit)
        ]
        
        summary = validator.get_summary()
        assert not summary.empty
        assert "Sharpe_Degradation" in summary.columns
        assert "Overfit_Warning" in summary.columns
        
        assert summary.iloc[0]["Overfit_Warning"] == False # 0.5 is threshold, maybe inclusive?
        # Logic: degradation < 0.5 -> True
        # 0.5 < 0.5 is False
        
        assert summary.iloc[1]["Overfit_Warning"] == True
