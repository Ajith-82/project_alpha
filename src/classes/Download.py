#!/usr/bin/env python
"""
Download Module

Orchestrates stock data downloading, caching, and validation.
This is a refactored version that uses the modular data layer components.
"""

import os
import sys
import csv
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from classes.data import StockFetcher, CacheManager, DataValidator
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
import structlog
from exceptions import DataFetchError, ConfigurationError

# Try to import Rich progress components
try:
    from classes.Console import create_download_progress, console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


logger = structlog.get_logger()


def _handle_start_end_dates(start, end):
    """Convert start/end dates to timestamps."""
    if end is None:
        end = int(datetime.timestamp(datetime.today()))
    elif isinstance(end, str):
        end = int(datetime.timestamp(datetime.strptime(end, "%Y-%m-%d")))
    if start is None:
        start = int(datetime.timestamp(datetime.today() - timedelta(730)))
    elif isinstance(start, str):
        start = int(datetime.timestamp(datetime.strptime(start, "%Y-%m-%d")))
    return start, end


def _date_from_timestamp(ts: int) -> str:
    """Convert timestamp to YYYY-MM-DD string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def load_cache(file_prefix: str, source_dir: str = ".") -> Optional[Dict]:
    """
    Load a dictionary from a pickle cache file.
    
    Uses CacheManager for consistent caching behavior.
    
    Args:
        file_prefix: Prefix for the cache file
        source_dir: Directory containing cache files
        
    Returns:
        Cached data or empty dict if not found
    """
    cache_manager = CacheManager(cache_dir=source_dir)
    data = cache_manager.get(file_prefix)
    
    if data:
        return data
    
    logger.info("Cache not found", file_prefix=file_prefix)
    return {}


def download(
    market: str,
    tickers: List[str],
    start: Union[str, int] = None,
    end: Union[str, int] = None,
    interval: str = "1d",
    db_path: str = None,
    use_rich_progress: bool = True,
) -> Dict[str, Union[List[str], Dict[str, pd.DataFrame], Dict[str, str]]]:
    """
    Download historical data for tickers with Rich progress support.
    
    Uses the refactored StockFetcher with retry logic and validation.
    
    Args:
        market: Market identifier ("us" or "india")
        tickers: List of ticker symbols
        start: Start date (str or timestamp)
        end: End date (str or timestamp)
        interval: Data frequency (default "1d")
        db_path: Optional SQLite database path
        use_rich_progress: Use Rich progress bar if available
        
    Returns:
        Dictionary with 'tickers', 'price_data', and 'company_info'
    """
    # Normalize tickers
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").split()
    )
    tickers = list(set(t.upper() for t in tickers if t))
    
    # Handle dates
    start_ts, end_ts = _handle_start_end_dates(start, end)
    start_date = _date_from_timestamp(start_ts)
    end_date = _date_from_timestamp(end_ts)
    
    # Initialize Rich progress if available
    progress = None
    task_id = None
    completed_count = [0]
    
    if use_rich_progress and RICH_AVAILABLE:
        progress = create_download_progress()
        progress.start()
        task_id = progress.add_task(
            f"[cyan]Downloading {market.upper()} stocks...",
            total=len(tickers)
        )
    
    def progress_callback(ticker: str, completed: int, total: int):
        """Update progress bar."""
        completed_count[0] = completed
        if progress and task_id is not None:
            progress.update(task_id, completed=completed)
        elif not progress:
            # Fallback to simple progress
            if completed % 50 == 0 or completed == total:
                logger.info("Download progress", completed=completed, total=total)
    
    # Create fetcher with retry logic
    fetcher = StockFetcher(
        max_retries=3,
        retry_delays=[1, 2, 4],
        progress_callback=progress_callback,
        verbose=False,
    )
    
    # Fetch batch
    results = fetcher.fetch_batch(
        tickers=tickers,
        market=market,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )
    
    # Stop progress bar
    if progress:
        progress.stop()
    
    # Validate results
    validator = DataValidator(nan_threshold=0.33, min_rows=5)
    
    # Extract successful results
    price_data = {}
    company_info = {}
    
    for ticker, result in results.items():
        if result.success and result.price_data is not None:
            # Validate price data
            # Use strict validators from src.classes.data.validators
            try:
                # Basic validation (NaNs, rows)
                validation = validator.validate_price_data(result.price_data, ticker)
                if not validation.valid:
                     logger.warning(f"Basic validation failed for {ticker}: {validation.errors}")
                     continue

                # Strict validation (Sanity, schema)
                from classes.data.validators import validate_data_quality, repair_data
                
                # Proactively repair data
                result.price_data = repair_data(result.price_data, ticker)
                
                result.price_data = validate_data_quality(result.price_data, ticker)
                
                # If we get here, data is valid
                price_data[ticker] = result.price_data
                if result.company_info:
                    company_info[ticker] = result.company_info
                    
                # Save to database if path provided
                if db_path:
                    _save_to_db(db_path, ticker, result.price_data, result.company_info)
                    
            except Exception as e:
                logger.warning(f"Validation failed for {ticker}: {e}")
        else:
            logger.warning(f"Failed to fetch {ticker}: {result.error}")
    
    if len(price_data) == 0:
        raise Exception("No symbol with valid data is available.")
    
    # Log summary
    success_rate = len(price_data) / len(tickers) * 100
    logger.info(f"Successfully downloaded {len(price_data)}/{len(tickers)} stocks ({success_rate:.1f}%)")
    
    return {
        "tickers": list(price_data.keys()),
        "price_data": price_data,
        "company_info": company_info,
    }


def _save_to_db(db_path: str, ticker: str, price_data: pd.DataFrame, company_info: Dict):
    """Save data to SQLite database."""
    try:
        conn = connect_db(db_path)
        create_tables(conn)
        insert_price_rows(conn, ticker, price_data)
        if company_info:
            insert_company_info(conn, ticker, company_info)
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to save {ticker} to database: {e}")


def download_stock_data(
    market: str,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    db_path: str = None,
) -> Dict:
    """
    Download historical data for a single stock.
    
    Wrapper around StockFetcher.fetch_one() for backward compatibility.
    
    Args:
        market: Market identifier
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        interval: Data frequency
        db_path: Optional database path
        
    Returns:
        Dictionary with 'price_data' and 'company_info'
    """
    # Check database for existing data
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
    
    # Fetch using StockFetcher
    fetcher = StockFetcher(max_retries=3)
    result = fetcher.fetch_one(ticker, market, start_date, end_date, interval)
    
    if result.success:
        # Save to database
        if conn and result.price_data is not None:
            insert_price_rows(conn, ticker, result.price_data)
            if result.company_info:
                insert_company_info(conn, ticker, result.company_info)
        
        if conn:
            conn.close()
        
        return {
            "price_data": result.price_data,
            "company_info": result.company_info,
        }
    
    if conn:
        conn.close()
    
    return {"price_data": None, "company_info": None}


def load_data(
    cache: str,
    symbols: List[str] = None,
    market: str = "",
    file_prefix: str = "",
    data_dir: str = "",
    db_path: str = None,
    use_rich_progress: bool = True,
) -> Dict:
    """
    Load historical data from cache or download if not available.
    
    Args:
        cache: If truthy, try to load from cache first
        symbols: List of symbols to download
        market: Market identifier
        file_prefix: Prefix for cache files
        data_dir: Directory for cache files
        db_path: Optional database path
        use_rich_progress: Use Rich progress bar
        
    Returns:
        Dictionary containing historical data
    """
    # Try loading from cache
    if cache:
        logger.info("Loading historical data")
        cache_manager = CacheManager(cache_dir=data_dir)
        data = cache_manager.get(file_prefix)
        if data:
            return data
    
    # Get symbols if not provided
    if symbols is None:
        symbols_file_path = "symbols_list.txt"
        if os.path.exists(symbols_file_path):
            with open(symbols_file_path, "r") as f:
                symbols = f.readline().split(" ")
        else:
            raise ConfigurationError("No symbols information to download data (symbols_list.txt missing)")
    
    # Download data
    logger.info("Downloading historical data")
    data = download(market, symbols, db_path=db_path, use_rich_progress=use_rich_progress)
    
    # Merge with database data if available
    if db_path:
        conn = connect_db(db_path)
        price_data = {}
        for sym in symbols:
            df = get_price_dataframe(conn, sym)
            if not df.empty:
                price_data[sym] = df
        data["price_data"] = price_data
        conn.close()
    
    # Save to cache
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
    Load volatile data for the given market and data.
    
    Transforms raw price data into the format needed for volatility analysis.
    
    Args:
        market: Market identifier
        data: Dictionary with 'tickers', 'price_data', 'company_info'
        start: Start date
        end: End date
        interval: Data interval
        
    Returns:
        Dictionary with transformed volatile data
    """
    start, end = _handle_start_end_dates(start, end)

    tickers = data["tickers"]
    si_columns = ["SYMBOL", "CURRENCY", "SECTOR", "INDUSTRY"]
    si_filename = (
        f"data/historic_data/{'india' if market == 'india' else 'us'}/stock_info.csv"
    )
    
    if not os.path.exists(si_filename):
        os.makedirs(os.path.dirname(si_filename), exist_ok=True)
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
        try:
            data_one = data["price_data"][ticker]
            columns_to_copy = ["Adj Close", "Volume"]
            data_one = data_one[columns_to_copy]
            volatile_data[ticker] = data_one

            if ticker in missing_tickers:
                currencies[ticker] = "INR" if market == "india" else "USD"
                
                # Safely get sector/industry with fallbacks
                company_info = data.get("company_info", {}).get(ticker, {})
                sector_value = company_info.get("sector", "Unknown")
                industry_value = company_info.get("industry", "Unknown")
                
                if isinstance(sector_value, list):
                    sector_value = sector_value[0] if sector_value else "Unknown"
                if isinstance(industry_value, list):
                    industry_value = industry_value[0] if industry_value else "Unknown"
                
                if sector_value and industry_value:
                    missing_si[ticker] = {
                        "sector": sector_value,
                        "industry": industry_value,
                    }
        except KeyError as e:
            logger.warning(f"Missing data for {ticker}: {e}")
            continue

    if not volatile_data:
        raise Exception("No symbol with full information is available.")

    volatile_data = pd.concat(
        volatile_data.values(), keys=volatile_data.keys(), axis=1, sort=True
    )
    volatile_data.drop(
        columns=volatile_data.columns[
            volatile_data.isnull().sum(0) > 0.33 * volatile_data.shape[0]
        ],
        inplace=True,
    )
    volatile_data = volatile_data.ffill().bfill().drop_duplicates()

    # Save missing stock info
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
        logger.warning(
            "Removing symbols due to incomplete data", 
            count=len(missing_tickers), 
            examples=missing_tickers[:5]
        )

    currencies = [
        si.get(ticker, {}).get("CURRENCY", currencies.get(ticker, "USD"))
        for ticker in tickers
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
    Download exchange rates for currency conversion.
    
    Args:
        from_currencies: List of source currencies
        to_currency: Target currency
        dates: Date index for the rates
        start: Start timestamp
        end: End timestamp
        interval: Data interval
        
    Returns:
        Dictionary of exchange rates
    """
    start, end = _handle_start_end_dates(start, end)
    ucurrencies, counts = np.unique(from_currencies, return_counts=True)
    xrates = _process_exchange_rates(
        ucurrencies, to_currency, dates, start, end, interval
    )
    return xrates


def _process_exchange_rates(ucurrencies, to_currency, dates, start, end, interval):
    """Process exchange rates for given currencies."""
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
    """Download historical data for a single ticker from Yahoo Finance API."""
    base_url = "https://query1.finance.yahoo.com"
    params = {
        "period1": start,
        "period2": end,
        "interval": interval.lower(),
        "includePrePost": False,
    }
    url = f"{base_url}/v8/finance/chart/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/50.0.2661.102 Safari/537.36"
    }
    response = requests.get(url, params=params, headers=headers)
    if "Will be right back" in response.text:
        raise RuntimeError("*** YAHOO! FINANCE is currently down! ***\n")
    return response.json()


def _parse_quotes(data: dict, parse_volume: bool = True) -> pd.DataFrame:
    """Parse quotes from Yahoo Finance API response."""
    timestamps = data["timestamp"]
    ohlc = data["indicators"]["quote"][0]
    closes = ohlc["close"]

    volumes = ohlc["volume"] if parse_volume else None
    adjclose = (
        data["indicators"]["adjclose"][0]["adjclose"]
        if "adjclose" in data["indicators"]
        else closes
    )

    # Fix NaNs in the second-last entry
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
