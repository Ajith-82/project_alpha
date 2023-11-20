from matplotlib import ticker
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as delta
import numpy as np
import seaborn as sb
import os
from classes.Plot_indicators import plot_strategy_multiple


def add_signal_indicators(df):
    # Add SMA indicators to data
    df["SMA_10"] = ta.sma(df["Adj Close"], length=10)
    df["SMA_30"] = ta.sma(df["Adj Close"], length=30)
    df["SMA_50"] = ta.sma(df["Adj Close"], length=50)
    df["SMA_200"] = ta.sma(df["Adj Close"], length=200)

    # Add MACD indicator to data
    macd = ta.macd(df["Adj Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]

    # Add RSI indicator to data
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["10_cross_30"] = np.where(df["SMA_10"] > df["SMA_30"], 1, 0)

    df["MACD_Signal_MACD"] = np.where(df["MACD_signal"] < df["MACD"], 1, 0)

    df["MACD_lim"] = np.where(df["MACD"] > 0, 1, 0)

    df["abv_50"] = np.where(
        (df["SMA_30"] > df["SMA_50"]) & (df["SMA_10"] > df["SMA_50"]), 1, 0
    )

    df["abv_200"] = np.where(
        (df["SMA_30"] > df["SMA_200"])
        & (df["SMA_10"] > df["SMA_200"])
        & (df["SMA_50"] > df["SMA_200"]),
        1,
        0,
    )

    return df


def backtest_signals(row):
    signals = pd.DataFrame(
        {
            "10_cross_30": [0, 0, 1, 1, 1],
            "MACD_Signal_MACD": [1, 1, 1, 0, 0],
            "MACD_lim": [0, 0, 0, 1, 1],
            "abv_50": [1, 1, 1, 0, 0],
            "abv_200": [0, 1, 0, 0, 1],
            "strategy": [1, 2, 3, 4, 5],
        }
    )
    print(signals)

    trades = []
    holding_period = 14

    if trade_in_progress:
        _data = trades[-1]
        # time to sell after n holding days
        if row["index"] == _data["sell_index"]:
            _data["sell_price"] = round(row["Adj Close"], 2)
            _data["sell_date"] = str(pd.to_datetime(row["Date"]).date())
            _data["returns"] = round(
                (_data["sell_price"] - _data["buy_price"]) / _data["buy_price"] * 100, 3
            )
            trades[-1] = _data
            trade_in_progress = False

    else:
        _r = pd.DataFrame([row])
        _r = _r.merge(signals, on=list(signals.columns[:-1]), how="inner")
        strategy = _r.shape[0]

        if strategy > 0:
            trade_in_progress = True
            trades.append(
                {
                    "strategy": _r["strategy"].values[0],
                    "buy_date": str(pd.to_datetime(row["Date"]).date()),
                    "buy_index": row["index"],
                    "sell_date": "",
                    "sell_index": row["index"] + holding_period,
                    "buy_price": round(row["Adj Close"], 2),
                    "sell_price": "",
                    "returns": 0,
                    "stock": row["stock"],
                }
            )


def find_stocks_meeting_conditions(data, signals, lookback_days):
    matching_stocks = []

    for stock in data["stock"].unique():
        stock_data = data[data["stock"] == stock].tail(lookback_days)

        for _, row in stock_data.iterrows():
            _r = pd.DataFrame([row])
            _r = _r.merge(signals, on=list(signals.columns[:-1]), how="inner")
            strategy = _r.shape[0]

            if strategy > 0:
                matching_stocks.append(stock)
                break  # Break if any condition is met in the last 5 days

    return matching_stocks


def stock_meets_conditions(ticker, data, signals, lookback_days):
    matching_trades = []
    data_one = data.reset_index()
    for _, row in data_one.tail(lookback_days).iterrows():
        try:
            _r = pd.DataFrame([row])
            _r = _r.merge(signals, on=list(signals.columns[:-1]), how="inner")
            # strategies = _r['strategy'].tolist()
            strategy = _r.shape[0]
            if strategy > 0:
                matching_trades.append(
                    {
                        "stock": ticker,
                        "strategy": _r["strategy"].values[0],
                        "buy_date": str(pd.to_datetime(row["Date"]).date()),
                        "buy_price": round(row["Adj Close"], 2),
                    }
                )
        except Exception as e:
            # print(f"Error: {e}")
            pass

    return matching_trades


def screener_one(data, look_back_days=5):
    tickers = data["tickers"]
    price_data = data["price_data"]
    signals = pd.DataFrame(
        {
            "10_cross_30": [0, 0, 1, 1, 1],
            "MACD_Signal_MACD": [1, 1, 1, 0, 0],
            "MACD_lim": [0, 0, 0, 1, 1],
            "abv_50": [1, 1, 1, 0, 0],
            "abv_200": [0, 1, 0, 0, 1],
            "strategy": [1, 2, 3, 4, 5],
        }
    )

    plot_tickers = []
    plot_data = {}

    for ticker in tickers:
        _df = price_data[ticker]
        _df = add_signal_indicators(_df)

        matching_trades = stock_meets_conditions(ticker, _df, signals, look_back_days)
        if len(matching_trades) > 0:
            for trade in matching_trades:
                print(
                    f"The {trade['stock']} meets strategy {trade['strategy']} on {trade['buy_date']}"
                )
            plot_tickers.append(ticker)
            plot_data[ticker] = _df

    plot_strategy_multiple(plot_tickers, plot_data)
