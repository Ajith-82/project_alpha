# import yfinance as yf
import requests
import datetime as dt
import pandas as pd
import numpy as np
from src.classes.Send_email import EmailServer


def _download_one(
    market: str, ticker: str, start: str, end: str, interval: str = "1d"
) -> dict:
    """
    Download historical data for a single ticker.

    Parameters
    ----------
    ticker: str
        Ticker for which to download historical information.
    start: int
        Start download data from this timestamp date.
    end: int
        End download data at this timestamp date.
    interval: str
        Frequency between data.

    Returns
    -------
    data: dict
        Scraped dictionary of information.
    """
    if market == "india":
        ticker = f"{ticker}.NS"
    base_url = "https://query1.finance.yahoo.com"
    params = dict(
        period1=start, period2=end, interval=interval.lower(), includePrePost=False
    )
    url = "{}/v8/finance/chart/{}".format(base_url, ticker)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
    }
    data = requests.get(url=url, params=params, headers=headers)
    if "Will be right back" in data.text:
        raise RuntimeError("*** YAHOO! FINANCE is currently down! ***\n")
    data = data.json()
    return data


def _parse_quotes(data: dict, parse_volume: bool = True) -> pd.DataFrame:
    """
    It creates a data frame of adjusted closing prices, and, if `parse_volume=True`, volumes. If no adjusted closing
    price is available, it sets it equal to closing price.

    Parameters
    ----------
    data: dict
        Data containing historical information of corresponding stock.
    parse_volume: bool
        Include or not volume information in the data frame.
    """
    timestamps = data["timestamp"]
    ohlc = data["indicators"]["quote"][0]
    closes = ohlc["close"]
    if parse_volume:
        volumes = ohlc["volume"]
    try:
        adjclose = data["indicators"]["adjclose"][0]["adjclose"]
    except:
        adjclose = closes

    # fix NaNs in the second-last entry of adjusted closing prices
    if adjclose[-2] is None:
        adjclose[-2] = adjclose[-1]

    assert (np.array(adjclose) > 0).all()

    quotes = {"Adj Close": adjclose}
    if parse_volume:
        quotes["Volume"] = volumes
    quotes = pd.DataFrame(quotes)
    quotes.index = pd.to_datetime(timestamps, unit="s").date
    quotes.sort_index(inplace=True)
    quotes = quotes.loc[~quotes.index.duplicated(keep="first")]

    return quotes


def main_old():
    msft = yf.Ticker("MSFT")
    # quotes = yf.download("AAPL", period="1y")
    prices = msft.history(timeout=10)
    # print(quotes)
    print(prices)

    data = {}
    market = "us"
    ticker = "MSFT"
    end = int(dt.datetime.timestamp(dt.datetime.today()))
    start = int(dt.datetime.timestamp(dt.datetime.today() - dt.timedelta(365)))
    data_one = _download_one(market, ticker, start, end)

    data_one = data_one["chart"]["result"][0]
    print(data_one)
    data[ticker] = _parse_quotes(data_one)
    print(data[ticker])


def main():
    email_server = EmailServer("email_config.json")
    email_server.send_svg_attachment(
        "Test Subject",
        "Test Message",
        svg_folder="data/processed_data/plot_indicators",
        mock=False,
    )


if __name__ == "__main__":
    main()
