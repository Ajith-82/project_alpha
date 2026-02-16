import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def make_ohlcv(days=60, start_price=100.0, trend=0.0, volatility=0.02, start_date="2023-01-01"):
    """
    Generate synthetic OHLCV data.
    
    Args:
        days: Number of trading days
        start_price: Starting price
        trend: Daily drift (e.g., 0.001 for 0.1% daily up)
        volatility: Daily volatility (sigma)
        start_date: Start date string (YYYY-MM-DD)
    """
    dates = pd.date_range(start=start_date, periods=days, freq="B") # Business days
    
    close_prices = [start_price]
    for _ in range(1, days):
        change = np.random.normal(trend, volatility)
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)
    
    data = []
    for date, close in zip(dates, close_prices):
        # Generate H/L/O around Close
        high = close * (1 + abs(np.random.normal(0, volatility/2)))
        low = close * (1 - abs(np.random.normal(0, volatility/2)))
        open_ = (high + low) / 2 + np.random.normal(0, volatility/4)
        volume = int(np.random.normal(1000000, 200000))
        
        data.append({
            "Date": date,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": max(volume, 1000)
        })
        
    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)
    return df

def make_uptrend(days=60, start_price=100, daily_return=0.005):
    """Generate synthetic uptrend data."""
    return make_ohlcv(days=days, start_price=start_price, trend=daily_return, volatility=0.01)

def make_downtrend(days=60, start_price=100, daily_return=-0.005):
    """Generate synthetic downtrend data."""
    return make_ohlcv(days=days, start_price=start_price, trend=daily_return, volatility=0.01)

def make_sideways(days=60, start_price=100, volatility=0.01):
    """Generate synthetic sideways data."""
    return make_ohlcv(days=days, start_price=start_price, trend=0.0, volatility=volatility)

def make_breakout(days=60, consolidation_days=40, breakout_magnitude=0.05):
    """
    Generate consolidation followed by a breakout.
    First part: sideways
    Second part: sharp move up
    """
    consolidation = make_sideways(days=consolidation_days, start_price=100, volatility=0.005)
    
    last_price = consolidation["Close"].iloc[-1]
    last_date = consolidation.index[-1]
    
    # Breakout candle
    breakout_date = last_date + timedelta(days=1)
    breakout_open = last_price
    breakout_close = last_price * (1 + breakout_magnitude)
    breakout_high = breakout_close * 1.01
    breakout_low = breakout_open * 0.99
    breakout_vol = consolidation["Volume"].mean() * 3  # High volume
    
    breakout_row = pd.DataFrame([{
        "Open": breakout_open,
        "High": breakout_high,
        "Low": breakout_low,
        "Close": breakout_close,
        "Adj Close": breakout_close,
        "Volume": int(breakout_vol)
    }], index=[breakout_date])
    breakout_row.index.name = "Date"
    
    # Post-breakout continuation
    continuation = make_uptrend(days=days-consolidation_days-1, start_price=breakout_close, daily_return=0.002)
    # Adjust dates
    continuation.index = pd.date_range(start=breakout_date + timedelta(days=1), periods=len(continuation), freq="B")
    
    return pd.concat([consolidation, breakout_row, continuation])
