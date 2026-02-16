import pytest
import sqlite3
import pandas as pd
import json
from src.classes import DatabaseManager

@pytest.fixture
def db_connection():
    """Create an in-memory database connection for testing."""
    conn = sqlite3.connect(":memory:")
    DatabaseManager.create_tables(conn)
    yield conn
    conn.close()

def test_create_tables(db_connection):
    """Test table creation."""
    cursor = db_connection.cursor()
    
    # Check if price_data table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
    assert cursor.fetchone() is not None
    
    # Check if company_info table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='company_info'")
    assert cursor.fetchone() is not None

def test_insert_and_get_price_rows(db_connection):
    """Test inserting and retrieving price data."""
    symbol = "TEST"
    dates = pd.date_range("2023-01-01", periods=5)
    data = {
        "Open": [100.0] * 5,
        "High": [105.0] * 5,
        "Low": [95.0] * 5,
        "Close": [102.0] * 5,
        "Adj Close": [102.0] * 5,
        "Volume": [1000] * 5,
        "Dividends": [0.0] * 5,
        "Stock Splits": [0.0] * 5
    }
    df = pd.DataFrame(data, index=dates)
    
    DatabaseManager.insert_price_rows(db_connection, symbol, df)
    
    # Verify data retrieval
    retrieved_df = DatabaseManager.get_price_dataframe(db_connection, symbol)
    assert not retrieved_df.empty
    assert len(retrieved_df) == 5
    assert "Close" in retrieved_df.columns
    assert retrieved_df.iloc[0]["Close"] == 102.0

def test_insert_and_get_company_info(db_connection):
    """Test inserting and retrieving company info."""
    symbol = "TEST"
    info = {"sector": "Technology", "industry": "Software"}
    
    DatabaseManager.insert_company_info(db_connection, symbol, info)
    
    retrieved_info = DatabaseManager.get_company_info(db_connection, symbol)
    assert retrieved_info == info

def test_get_last_date(db_connection):
    """Test retrieving last date."""
    symbol = "TEST"
    dates = pd.date_range("2023-01-01", periods=2)
    data = {
        "Open": [100.0] * 2,
        "High": [105.0] * 2,
        "Low": [95.0] * 2,
        "Close": [102.0] * 2,
        "Adj Close": [102.0] * 2,
        "Volume": [1000] * 2,
        "Dividends": [0.0] * 2,
        "Stock Splits": [0.0] * 2
    }
    df = pd.DataFrame(data, index=dates)
    DatabaseManager.insert_price_rows(db_connection, symbol, df)
    
    last_date = DatabaseManager.get_last_date(db_connection, symbol)
    # Sqlite might store as string without time if we converted it, or with time.
    # Just check the date part.
    assert str(last_date).startswith("2023-01-02")
