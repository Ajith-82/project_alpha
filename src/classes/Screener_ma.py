import pandas as pd
import numpy as np
import concurrent.futures


def add_signal_indicators(df):
    df.loc[:, "10_cross_30"] = np.where(df["SMA_10"] > df["SMA_30"], 1, 0)

    df.loc[:, "MACD_Signal_MACD"] = np.where(df["MACD_signal"] < df["MACD"], 1, 0)

    df.loc[:, "MACD_lim"] = np.where(df["MACD"] > 0, 1, 0)

    df.loc[:, "abv_50"] = np.where(
        (df["SMA_30"] > df["SMA_50"]) & (df["SMA_10"] > df["SMA_50"]), 1, 0
    )

    df.loc[:, "abv_200"] = np.where(
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


def find_matching_trades(ticker, data, signals, lookback_days):
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


def screener_ma(data, look_back_days=5):
    """
    Generate a moving average screener for a given set of stocks based on various signal indicators.
    
    Args:
        data (dict): A dictionary containing "tickers" and "price_data".
        look_back_days (int): The number of days to look back for calculating the moving averages. Default is 5.
        
    Returns:
        dict: A dictionary containing the matching trades for each stock based on the specified conditions.
    """
    tickers = data["tickers"]
    price_data = data["price_data"]
    signals = pd.DataFrame({
        "10_cross_30": [0, 0, 1, 1, 1],
        "MACD_Signal_MACD": [1, 1, 1, 0, 0],
        "MACD_lim": [0, 0, 0, 1, 1],
        "abv_50": [1, 1, 1, 0, 0],
        "abv_200": [0, 1, 0, 0, 1],
        "strategy": [1, 2, 3, 4, 5],
    })

    screener_data = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = {executor.submit(process_stock, ticker, price_data[ticker], signals, look_back_days): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(results):
            ticker = results[future]
            try:
                matching_trades = future.result()
                if matching_trades:
                    screener_data[ticker] = matching_trades
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    return screener_data

def process_stock(ticker, stock_data, signals, look_back_days):
    stock_data = add_signal_indicators(stock_data)
    return find_matching_trades(ticker, stock_data, signals, look_back_days)