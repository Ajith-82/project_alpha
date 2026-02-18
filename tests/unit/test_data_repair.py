import pytest
import pandas as pd
import numpy as np
from classes.data.validators import repair_data

def test_repair_negative_volume():
    df = pd.DataFrame({
        "Volume": [100, -50, 200, -10],
        "Open": [10, 11, 12, 13],
        "High": [10, 11, 12, 13],
        "Low": [10, 11, 12, 13],
        "Close": [10, 11, 12, 13]
    })
    df_repaired = repair_data(df, "TEST")
    assert df_repaired["Volume"].min() >= 0
    assert df_repaired["Volume"].iloc[1] == 0
    assert df_repaired["Volume"].iloc[3] == 0

def test_repair_missing_prices():
    df = pd.DataFrame({
        "Close": [10.0, np.nan, 12.0, 13.0],
        "Open": [10.0, 11.0, 12.0, 13.0],
        "High": [10.0, 11.0, 12.0, 13.0],
        "Low": [10.0, 11.0, 12.0, 13.0],
        "Volume": [100, 100, 100, 100]
    })
    # Add datetime index for interpolation
    df.index = pd.date_range("2023-01-01", periods=4)
    
    df_repaired = repair_data(df, "TEST")
    assert not df_repaired["Close"].isnull().any()
    # Interpolated value should be 11.0
    assert df_repaired["Close"].iloc[1] == 11.0

def test_repair_drop_unrepairable():
    df = pd.DataFrame({
        "Close": [10.0, np.nan, np.nan, np.nan, np.nan, 15.0], # Too many NaNs > limit=3
        "Open": [10.0, 10.0, 10.0, 10.0, 10.0, 15.0],
        "High": [10.0, 10.0, 10.0, 10.0, 10.0, 15.0],
        "Low": [10.0, 10.0, 10.0, 10.0, 10.0, 15.0],
        "Volume": [100, 100, 100, 100, 100, 100]
    })
    df.index = pd.date_range("2023-01-01", periods=6)
    
    df_repaired = repair_data(df, "TEST")
    # Should drop the NaNs if interpolation failed or limit exceeded?
    # limit=3 means max 3 consecutive NaNs are filled.
    # Here we have 4. So some might remain and then be dropped.
    assert not df_repaired.isnull().any().any()
    assert len(df_repaired) < 6
