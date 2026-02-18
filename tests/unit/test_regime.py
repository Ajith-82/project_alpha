import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from classes.analysis.regime import RegimeDetector


@pytest.fixture
def sample_data():
    """Create synthetic price data with clear trends."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    
    # Create 3 segments: Bull, Bear, Sideways
    # Bull: +1% daily
    bull = np.cumprod(np.ones(30) * 1.01)
    
    # Sideways: Oscillation
    sideways = np.ones(40) * bull[-1] + np.sin(np.arange(40)) * 2
    
    # Bear: -1% daily
    bear = np.cumprod(np.ones(30) * 0.99) * sideways[-1]
    
    prices = np.concatenate([bull, sideways, bear])
    return pd.DataFrame({"Close": prices}, index=dates)


class TestRegimeDetector:

    def test_initialization(self):
        detector = RegimeDetector(n_components=3)
        assert detector.n_components == 3
        assert detector.state_map == {}

    def test_prepare_features(self, sample_data):
        detector = RegimeDetector()
        features = detector.prepare_features(sample_data)
        
        # Original 100 points
        # log returns needs 1 shift -> 99
        # rolling 20 needs 19 more -> 80 valid points
        # actually rolling is inclusive but first 19 are NaN
        # so 100 - 1 - 19 = 80 ? 
        # computed on log_ret which has 99 points.
        # log_ret[0] is NaN. 
        # rolling starts at index 0 of log_ret? no.
        # Let's check logic:
        # log_ret has first NaN.
        # rolling(20) on log_ret... first 20 items will include that NaN?
        # Actually standard Pandas rolling behaviour.
        
        # Expected shape: (n_samples, 2)
        assert features.shape[1] == 2
        assert len(features) > 0

    def test_fit_logic(self, sample_data):
        detector = RegimeDetector()
        
        # Use a mock for GaussianHMM to avoid complex fitting in unit test
        # but we also want to test the mapping logic which depends on means_
        
        # Let's try fitting real small data, HMM is fast enough for 100 points
        detector.fit(sample_data)
        
        assert len(detector.state_map) == 3
        assert "Bull" in detector.state_map.values()
        assert "Bear" in detector.state_map.values()
        assert "Sideways" in detector.state_map.values()

    def test_mapping_correctness(self):
        """Verify that highest return state is mapped to Bull."""
        detector = RegimeDetector()
        
        # Mock the model and its attributes
        detector.model.fit = MagicMock()
        # 3 states with returns: -0.01 (Bear), 0.0 (Sideways), 0.01 (Bull)
        # means_ shape is (n_components, n_features)
        detector.model.means_ = np.array([
            [-0.01, 0.005], # State 0: Negative return
            [0.01, 0.005],  # State 1: Positive return
            [0.00, 0.002]   # State 2: Zero return
        ])
        
        # Manually trigger the mapping logic by calling fit with mocked helper
        # Since fit calls prepare_features and then model.fit, we can't easily partially mock
        # unless we extract mapping logic or mock everything.
        # Let's just mock prepare_features to return dummy X
        
        with patch.object(detector, 'prepare_features', return_value=np.zeros((10, 2))):
            detector.fit(pd.DataFrame({'Close': range(25)}))
            
            # State 1 has highest return (0.01) -> Bull
            # State 0 has lowest return (-0.01) -> Bear
            # State 2 is middle -> Sideways
            
            assert detector.state_map[1] == "Bull"
            assert detector.state_map[0] == "Bear"
            assert detector.state_map[2] == "Sideways"

    def test_predict_alignment(self, sample_data):
        detector = RegimeDetector()
        detector.fit(sample_data)
        
        result = detector.predict(sample_data)
        
        # Result should have same index as input?
        # Our logic returns df with new columns, but NaNs at start might be handled
        # `prepare_features` drops NaNs. 
        # If we just return the tail, alignment is preserved for those dates.
        # The rows with NaNs (first 20) should have NaN in Regime columns or be missing?
        
        # Current logic: `df.copy()` then assign `regime_series` which is indexed by valid dates.
        # So rows 0-19 should have NaN in Regime_State/Regime.
        
        assert len(result) == len(sample_data)
        assert "Regime" in result.columns
        
        # First 20 rows should be NaN because of rolling window
        assert pd.isna(result["Regime"].iloc[0])
        assert pd.isna(result["Regime"].iloc[18])
        
        # Later rows should be populated
        assert not pd.isna(result["Regime"].iloc[-1])
