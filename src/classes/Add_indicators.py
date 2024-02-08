import pandas_ta as ta

def add_sma_indicator(df):
    # Add SMA indicators to data
    df["SMA_10"] = ta.sma(df["Adj Close"], length=10)
    df["SMA_30"] = ta.sma(df["Adj Close"], length=30)
    df["SMA_50"] = ta.sma(df["Adj Close"], length=50)
    df["SMA_200"] = ta.sma(df["Adj Close"], length=200)

    return df

def add_macd_indicator(df):
    # Add MACD indicator to data
    macd = ta.macd(df["Adj Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]

    return df

def add_rsi_indicator(df):
    # Add RSI indicator to data
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

def add_indicators(df):
    df = add_sma_indicator(df)
    df = add_macd_indicator(df)
    df = add_rsi_indicator(df)

    return df