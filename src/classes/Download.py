#!/usr/bin/env python
import os
import sys
import pickle
import csv
import requests
import numpy as np
from typing import Dict, List, Union
from datetime import datetime, timedelta
import multitasking
import pandas as pd
import yfinance as yf
from typing import Union
from classes.Add_indicators import add_indicators
from classes.DatabaseManager import (
    connect_db,
    create_tables,
    get_last_date,
    insert_price_rows,
    insert_company_info,
    get_price_dataframe,
    get_company_info,
)
from classes.Tools import ProgressBar, save_dict_with_timestamp
from classes.DataSourceManager import get_manager


def _handle_start_end_dates(start, end):
    if end is None:
        end = int(datetime.timestamp(datetime.today()))
    elif isinstance(end, str):
        end = int(datetime.timestamp(datetime.strptime(end, "%Y-%m-%d")))
    if start is None:
        start = int(datetime.timestamp(datetime.today() - timedelta(730)))
    elif isinstance(start, str):
        start = int(datetime.timestamp(datetime.strptime(start, "%Y-%m-%d")))
    return start, end


def load_cache(file_prefix, source_dir="."):
    """
    Load a dictionary from a file.

    Parameters:
    - file_prefix: str, the prefix for the file (without extension).
    - source_dir: str, the source directory for the file (default is the current directory).

    Returns:
    - dict or None: The loaded dictionary or None if the file doesn't exist.
    """
    # Generate timestamp in YYMMDD format
    timestamp = datetime.now().strftime("%y%m%d")

    # Create a new file name with the timestamp
    file_path = os.path.join(source_dir, f"{file_prefix}_{timestamp}.pkl")

    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as file:
                loaded_dict = pickle.load(file)
                print(f"Using data saved to {file.name}.")
                return loaded_dict
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return {}
    else:
        print(f"File not found: {file_path}")
        return {}


def download(
    market: str,
    tickers: List[str],
    start: Union[str, int] = None,
    end: Union[str, int] = None,
    interval: str = "1d",
    db_path: str = None,
) -> Dict[str, Union[List[str], Dict[str, pd.DataFrame], Dict[str, str]]]:
    """
    Download historical data for tickers in the list.

    Parameters
    ----------
    tickers: list
        Tickers for which to download historical information.
    start: str or int
        Start download data from this date.
    end: str or int
        End download data at this date.
    interval: str
        Frequency between data.

    Returns
    -------
    data: dict
        Dictionary including the following keys:
        - tickers: list of tickers
        - price_data: dictionary of pandas dataframes with stock prices
        - company_info: dictionary of company information
    """
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").split()
    )
    tickers = [ticker.upper() for ticker in tickers if ticker is not None]
    tickers = list(set(tickers))

    price_data = {}
    company_info = {}

    start, end = _handle_start_end_dates(start, end)

    @multitasking.task
    def download_one_threaded(
        market: str,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        db_path: str = None,
    ):
        """
        Download historical data for a single ticker with multithreading.
        Plus, it scrapes missing stock information.

        Parameters
        ----------
        ticker: str
            Ticker for which to download historical information.
        interval: str
            Frequency between data.
        start: str
            Start download data from this date.
        end: str
            End download data at this date.
        """
        ticker_data = download_stock_data(
            market, ticker, start, end, interval, db_path=db_path
        )
        data = ticker_data["price_data"]
        info = ticker_data["company_info"]

        if isinstance(data, pd.DataFrame):
            price_data[ticker] = data
        if info:
            company_info[ticker] = info

        progress.animate()

    num_threads = min([len(tickers), multitasking.cpu_count() * 2])
    multitasking.set_max_threads(num_threads)

    progress = ProgressBar(len(tickers), "completed")

    for ticker in tickers:
        download_one_threaded(market, ticker, start, end, interval, db_path)
    multitasking.wait_for_tasks()

    progress.completed()

    # Print data source statistics
    try:
        manager = get_manager(verbose=False)
        manager.print_statistics()
    except:
        pass  # Skip if manager not initialized

    if len(price_data) == 0:
        raise Exception("No symbol with full information is available.")

    return dict(
        tickers=list(price_data.keys()),
        price_data=price_data,
        company_info=company_info,
    )


def download_stock_data(
    market: str,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    db_path: str = None,
) -> dict:
    """
    Download historical price and ticker info data for a single stock.

    Parameters:
    - market (str): Market name ("india" for NSE, otherwise ignored).
    - ticker (str): Ticker symbol of the stock.
    - start_date (str): Start date for data download.
    - end_date (str): End date for data download.
    - interval (str): Frequency of data (default is "1d" for daily).

    Returns:
    - data (dict): A dictionary containing historical price data and stock information.
    """

    conn = None
    if db_path:
        conn = connect_db(db_path)
        create_tables(conn)
        last = get_last_date(conn, ticker)
        if last:
            nd = (pd.to_datetime(last) + timedelta(days=1)).strftime("%Y-%m-%d")
            if start_date is None or pd.to_datetime(nd) > pd.to_datetime(start_date):
                start_date = nd
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                info = get_company_info(conn, ticker)
                conn.close()
                return {"price_data": pd.DataFrame(), "company_info": info}

    try:
        # Use DataSourceManager for multi-source fallback
        manager = get_manager(verbose=False)

        # Fetch data with automatic fallback
        price_data, company_info = manager.fetch_stock_data(
            market=market,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

        if price_data is None or price_data.empty:
            if conn:
                conn.close()
            return {"price_data": None, "company_info": None}

        # Save to database if db_path is provided
        if conn:
            # For Indian market, add .NS suffix for database storage
            db_ticker = f"{ticker}.NS" if market == "india" else ticker
            insert_price_rows(conn, db_ticker, price_data)
            insert_company_info(conn, db_ticker, company_info)
            conn.close()
    except Exception as e:
        print(f"Error in data download for {ticker}: {e}")
        if conn:
            conn.close()
        return {"price_data": None, "company_info": None}

    # Return a dictionary containing price data and stock information
    return {"price_data": price_data, "company_info": company_info}


def load_data(
    cache: str,
    symbols: List[str] = None,
    market: str = "",
    file_prefix: str = "",
    data_dir: str = "",
    db_path: str = None,
) -> dict:
    """
    Load historical data from cache or download it if not available.

    Args:
        cache (str): Cache indicator. If not empty, the function will try to load data from cache.
        symbols (List[str], optional): List of symbols to download data for. Defaults to None.
        market (str, optional): Market information. Defaults to "".
        file_prefix (str, optional): Prefix for the file to save the data. Defaults to "".
        data_dir (str, optional): Directory to save the data. Defaults to "".

    Returns:
        dict: Dictionary containing the historical data.
    """
    while cache:
        print("\nLoading historical data...")
        data = load_cache(file_prefix, data_dir)
        if data:
            return data
        else:
            cache = ""

    if symbols is None:
        symbols_file_path = "symbols_list.txt"
        if os.path.exists(symbols_file_path):
            with open(symbols_file_path, "r") as my_file:
                symbols = my_file.readline().split(" ")
        else:
            print("No symbols information to download data. Exit script.")
            sys.exit()

    print("\nDownloading historical data...")
    data = download(market, symbols, db_path=db_path)
    if db_path:
        conn = connect_db(db_path)
        price_data = {}
        for sym in symbols:
            df = get_price_dataframe(conn, sym)
            if not df.empty:
                price_data[sym] = df
        data["price_data"] = price_data
        conn.close()
    save_dict_with_timestamp(data, file_prefix, data_dir)

    return data


def load_volatile_data(
    market: str,
    data: Dict,
    start: Union[str, int] = None,
    end: Union[str, int] = None,
    interval: str = "1d",
) -> Dict:
    """
    Load volatile data for the given market and data, within the specified time interval.
    :param market: str - the market for which the data is being loaded
    :param data: Dict - the data to be loaded
    :param start: Union[str, int] - the start date or index for the data
    :param end: Union[str, int] - the end date or index for the data
    :param interval: str - the time interval for the data
    :return: Dict - the loaded volatile data
    """

    start, end = _handle_start_end_dates(start, end)

    tickers = data["tickers"]
    si_columns = ["SYMBOL", "CURRENCY", "SECTOR", "INDUSTRY"]
    si_filename = (
        f"data/historic_data/{'india' if market == 'india' else 'us'}/stock_info.csv"
    )
    if not os.path.exists(si_filename):
        with open(si_filename, "w") as file:
            wr = csv.writer(file)
            wr.writerow(si_columns)

    si = pd.read_csv(si_filename)
    missing_tickers = [
        ticker for ticker in tickers if ticker not in si["SYMBOL"].values
    ]
    missing_si = {}
    currencies = {}
    volatile_data = {}

    for ticker in tickers:
        data_one = data["price_data"][ticker]
        columns_to_copy = ["Adj Close", "Volume"]
        data_one = data_one[columns_to_copy]
        volatile_data[ticker] = data_one

        if ticker in missing_tickers:
            currencies[ticker] = "INR" if market == "india" else "USD"
            sector_value = data["company_info"][ticker]["sector"]
            if isinstance(sector_value, list):
                sector_value = sector_value[0]
            industry_value = data["company_info"][ticker]["industry"]
            if isinstance(industry_value, list):
                industry_value = industry_value[0]

            if sector_value and industry_value:
                missing_si[ticker] = {
                    "sector": sector_value,
                    "industry": industry_value,
                }

    if not volatile_data:
        raise Exception("No symbol with full information is available.")

    volatile_data = pd.concat(volatile_data.values(), keys=volatile_data.keys(), axis=1, sort=True)
    volatile_data.drop(
        columns=volatile_data.columns[volatile_data.isnull().sum(0) > 0.33 * volatile_data.shape[0]], inplace=True
    )
    volatile_data = volatile_data.ffill().bfill().drop_duplicates()

    info = zip(
        list(missing_si.keys()),
        [currencies[ticker] for ticker in missing_si.keys()],
        [v["sector"] for v in missing_si.values()],
        [v["industry"] for v in missing_si.values()],
    )
    with open(si_filename, "a+", newline="") as file:
        wr = csv.writer(file)
        for row in info:
            wr.writerow(row)

    si = pd.read_csv(si_filename).set_index("SYMBOL").to_dict(orient="index")

    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in volatile_data.columns.get_level_values(0)[::2].tolist()
    ]
    tickers = volatile_data.columns.get_level_values(0)[::2].tolist()

    if missing_tickers:
        print(
            "\nRemoving {} from list of symbols because we could not collect full information.".format(
                missing_tickers
            )
        )

    currencies = [
        si.get(ticker, {}).get("CURRENCY", currencies.get(ticker)) for ticker in tickers
    ]
    ucurrencies, counts = np.unique(currencies, return_counts=True)
    default_currency = ucurrencies[np.argmax(counts)]
    xrates = get_exchange_rates(
        currencies, default_currency, volatile_data.index, start, end, interval
    )

    return {
        "tickers": tickers,
        "dates": pd.to_datetime(volatile_data.index),
        "price": volatile_data.xs("Adj Close", level=1, axis=1).to_numpy().T,
        "volume": volatile_data.xs("Volume", level=1, axis=1).to_numpy().T,
        "currencies": currencies,
        "exchange_rates": xrates,
        "default_currency": default_currency,
        "sectors": {
            ticker: si.get(ticker, {}).get("SECTOR", "NA_" + ticker)
            for ticker in tickers
        },
        "industries": {
            ticker: si.get(ticker, {}).get("INDUSTRY", "NA_" + ticker)
            for ticker in tickers
        },
    }


def get_exchange_rates(
    from_currencies: list,
    to_currency: str,
    dates: pd.Index,
    start: Union[str, int] = None,
    end: Union[str, int] = None,
    interval: str = "1d",
) -> dict:
    """
    It finds the most common currency and set it as default one. For any other currency, it downloads exchange rate
    closing prices to the default currency and return them as data frame.

    Parameters
    ----------
    from_currencies: list
        A list of currencies to convert.
    to_currency: str
        Currency to convert to.
    dates: date
        Dates for which exchange rates should be available.
    start: str or int
        Start download data from this timestamp date.
    end: str or int
        End download data at this timestamp date.
    interval: str
        Frequency between data.

    Returns
    -------
    xrates: dict
        A dictionary with currencies as keys and list of exchange rates at desired dates as values.
    """
    start, end = _handle_start_end_dates(start, end)
    ucurrencies, counts = np.unique(from_currencies, return_counts=True)
    xrates = _process_exchange_rates(
        ucurrencies, to_currency, dates, start, end, interval
    )
    return xrates


def _process_exchange_rates(ucurrencies, to_currency, dates, start, end, interval):
    """
    Process exchange rates for given currencies and time period.

    Args:
        ucurrencies (list): List of currencies to process.
        to_currency (str): The target currency to convert to.
        dates (list): List of dates for the exchange rates.
        start (str): Start date for the exchange rates.
        end (str): End date for the exchange rates.
        interval (str): Interval for the exchange rates.

    Returns:
        pandas.DataFrame: Processed exchange rates.
    """
    tmp = {}
    if to_currency not in ucurrencies or len(ucurrencies) > 1:
        for curr in ucurrencies:
            if curr != to_currency:
                tmp[curr] = _download_one(
                    curr + to_currency + "=x", start, end, interval
                )
                tmp[curr] = _parse_quotes(
                    tmp[curr]["chart"]["result"][0], parse_volume=False
                )["Adj Close"]
        tmp = pd.concat(tmp.values(), keys=tmp.keys(), axis=1, sort=True)
        xrates = pd.DataFrame(index=dates, columns=tmp.columns)
        xrates.loc[xrates.index.isin(tmp.index)] = tmp
        xrates = xrates.ffill().bfill()
        xrates.to_dict(orient="list")
    else:
        xrates = tmp
    return xrates


def _download_one(ticker: str, start: int, end: int, interval: str = "1d") -> dict:
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
    base_url = "https://query1.finance.yahoo.com"
    params = {
        "period1": start,
        "period2": end,
        "interval": interval.lower(),
        "includePrePost": False,
    }
    url = f"{base_url}/v8/finance/chart/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    response = requests.get(url, params=params, headers=headers)
    if "Will be right back" in response.text:
        raise RuntimeError("*** YAHOO! FINANCE is currently down! ***\n")
    return response.json()


def _parse_quotes(data: dict, parse_volume: bool = True) -> pd.DataFrame:
    """
    Creates a data frame of adjusted closing prices, and optionally includes volume information.

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

    volumes = ohlc["volume"] if parse_volume else None
    adjclose = (
        data["indicators"]["adjclose"][0]["adjclose"]
        if "adjclose" in data["indicators"]
        else closes
    )

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
