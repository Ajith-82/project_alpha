import pytest
import numpy as np
import tensorflow as tf
from src.classes import Models

# Only run if TensorFlow is installed
try:
    import tensorflow_probability as tfp
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow Probability not installed")
def test_s_model_creation():
    """Test creation of s_model Distribution."""
    # order_scale has len 1 -> order = 0.
    # phi shape: (num_stocks, order+1) -> (2, 1)
    # tt shape: (order+1, T) -> (1, 10). tensordot axes=1 sums dim 1 of phi and dim 0 of tt.
    info = {
        "tt": np.random.randn(1, 10).astype(np.float32), # Corrected shape (1, 10)
        "order_scale": np.array([1.0], dtype=np.float32), 
        "num_stocks": 2
    }
    
    model = Models.s_model(info)
    assert model is not None
    # Check if we can sample
    sample = model.sample()
    assert len(sample) == 3 # phi, psi, y

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow Probability not installed")
def test_conj_lin_model():
    """Test conjugate linear model logic."""
    # Create random log-price data: (2 stocks, 10 days)
    logp = np.random.randn(2, 10)
    
    preds = Models.conj_lin_model(logp)
    
    # Should return predictions for T-1 steps
    assert preds.shape == (2, 9)
