import pytest
import pandas as pd
from tests.fixtures import sample_data

@pytest.fixture
def uptrend_data():
    """Returns a DataFrame with a 60-day uptrend."""
    return sample_data.make_uptrend()

@pytest.fixture
def downtrend_data():
    """Returns a DataFrame with a 60-day downtrend."""
    return sample_data.make_downtrend()

@pytest.fixture
def sideways_data():
    """Returns a DataFrame with 60 days of sideways price action."""
    return sample_data.make_sideways()

@pytest.fixture
def breakout_data():
    """Returns a DataFrame with a consolidation followed by a breakout."""
    return sample_data.make_breakout()
