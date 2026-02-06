from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

def add_sma_indicator(df):
    # Add SMA indicators to data
    df["SMA_10"] = SMAIndicator(close=df["Adj Close"], window=10).sma_indicator()
    df["SMA_30"] = SMAIndicator(close=df["Adj Close"], window=30).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=df["Adj Close"], window=50).sma_indicator()
    df["SMA_200"] = SMAIndicator(close=df["Adj Close"], window=200).sma_indicator()

    return df

def add_macd_indicator(df):
    # Add MACD indicator to data
    macd = MACD(close=df["Adj Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    return df

def add_rsi_indicator(df):
    # Add RSI indicator to data
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    return df

def add_indicators(df):
    df = add_sma_indicator(df)
    df = add_macd_indicator(df)
    df = add_rsi_indicator(df)

    return df