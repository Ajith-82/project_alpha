#!/usr/bin/env python
"""
Multi-source data fetcher with automatic fallback.
Manages multiple data sources and provides resilient data fetching.
"""
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from classes.DataSourceConfig import get_config
from classes.TwelveDataFetcher import create_twelve_data_fetcher
from classes.AlphaVantageFetcher import create_alpha_vantage_fetcher
from classes.Add_indicators import add_indicators


class DataSourceManager:
    """Manages multiple data sources with automatic fallback."""

    def __init__(self, verbose: bool = True):
        """
        Initialize the data source manager.

        Args:
            verbose: If True, print detailed information about source attempts.
        """
        self.config = get_config()
        self.verbose = verbose
        self.fetchers = {}
        self.source_stats = {
            "yfinance": {"attempts": 0, "successes": 0, "failures": 0},
            "twelvedata": {"attempts": 0, "successes": 0, "failures": 0},
            "alphavantage": {"attempts": 0, "successes": 0, "failures": 0},
        }

        # Initialize fetchers for available sources
        self._initialize_fetchers()

    def _initialize_fetchers(self):
        """Initialize data source fetchers based on configuration."""
        if self.verbose:
            print("\nInitializing data sources...")

        # yfinance is always available (no API key needed)
        if self.config.is_source_available("yfinance"):
            self.fetchers["yfinance"] = None  # yfinance doesn't need a fetcher object
            if self.verbose:
                print("  ✓ yfinance: Available")

        # Initialize Twelve Data if configured
        if self.config.is_source_available("twelvedata"):
            api_key = self.config.get_api_key("twelvedata")
            fetcher = create_twelve_data_fetcher(api_key)
            if fetcher:
                self.fetchers["twelvedata"] = fetcher
                if self.verbose:
                    print("  ✓ twelvedata: Available")
            elif self.verbose:
                print("  ✗ twelvedata: Failed to initialize")

        # Initialize Alpha Vantage if configured
        if self.config.is_source_available("alphavantage"):
            api_key = self.config.get_api_key("alphavantage")
            fetcher = create_alpha_vantage_fetcher(api_key)
            if fetcher:
                self.fetchers["alphavantage"] = fetcher
                if self.verbose:
                    print("  ✓ alphavantage: Available")
            elif self.verbose:
                print("  ✗ alphavantage: Failed to initialize")

        if not self.fetchers:
            raise Exception("No data sources available! Please configure at least one source.")

        if self.verbose:
            print(f"\nActive sources: {', '.join(self.fetchers.keys())}\n")

    def fetch_stock_data(
        self,
        market: str,
        ticker: str,
        start_date,
        end_date,
        interval: str = "1d",
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Fetch stock data with automatic fallback between sources.

        Args:
            market: Market name ("india" for NSE, otherwise US).
            ticker: Stock ticker symbol.
            start_date: Start date (string or timestamp).
            end_date: End date (string or timestamp).
            interval: Data interval (default: "1d").

        Returns:
            Tuple of (price_data DataFrame, company_info dict), or (None, None) if all sources fail.
        """
        # Get available sources in priority order
        available_sources = [s for s in self.config.source_priority if s in self.fetchers]

        if not available_sources:
            print(f"Error: No data sources available for {ticker}")
            return None, None

        # Try each source in order
        for source in available_sources:
            self.source_stats[source]["attempts"] += 1

            if self.verbose:
                print(f"  Trying {source} for {ticker}...", end=" ")

            try:
                price_data, company_info = self._fetch_from_source(
                    source, market, ticker, start_date, end_date, interval
                )

                if price_data is not None and not price_data.empty:
                    self.source_stats[source]["successes"] += 1
                    if self.verbose:
                        print(f"✓ ({len(price_data)} records)")
                    return price_data, company_info
                else:
                    self.source_stats[source]["failures"] += 1
                    if self.verbose:
                        print("✗ (no data)")

            except Exception as e:
                self.source_stats[source]["failures"] += 1
                if self.verbose:
                    print(f"✗ ({str(e)[:50]})")

        # All sources failed
        print(f"Error: All data sources failed for {ticker}")
        return None, None

    def _fetch_from_source(
        self,
        source: str,
        market: str,
        ticker: str,
        start_date,
        end_date,
        interval: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Fetch data from a specific source.

        Args:
            source: Name of the data source.
            market: Market name.
            ticker: Stock ticker symbol.
            start_date: Start date.
            end_date: End date.
            interval: Data interval.

        Returns:
            Tuple of (price_data, company_info).
        """
        if source == "yfinance":
            return self._fetch_from_yfinance(market, ticker, start_date, end_date, interval)
        elif source == "twelvedata":
            return self._fetch_from_twelvedata(ticker, start_date, end_date, interval)
        elif source == "alphavantage":
            return self._fetch_from_alphavantage(ticker, start_date, end_date, interval)
        else:
            return None, None

    def _fetch_from_yfinance(
        self, market: str, ticker: str, start_date, end_date, interval: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Fetch data from yfinance (existing implementation)."""
        # Append .NS for Indian market
        if market == "india":
            ticker = f"{ticker}.NS"

        ticker_obj = yf.Ticker(ticker)

        # Download historical price data
        price_data = ticker_obj.history(
            interval=interval, start=start_date, end=end_date
        )

        if price_data.empty:
            return None, None

        price_data.index = price_data.index.strftime("%Y-%m-%d")
        price_data.sort_index(inplace=True)

        # Add Adj Close
        price_data["Adj Close"] = price_data["Close"].copy()
        price_data = price_data.reindex(
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]
        )
        price_data = price_data.loc[~price_data.index.duplicated(keep="first")]
        price_data = price_data.ffill().bfill().drop_duplicates()
        price_data = add_indicators(price_data)
        price_data = price_data.fillna(0)

        # Get company info
        company_info = ticker_obj.info

        return price_data, company_info

    def _fetch_from_twelvedata(
        self, ticker: str, start_date, end_date, interval: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Fetch data from Twelve Data."""
        fetcher = self.fetchers.get("twelvedata")
        if not fetcher:
            return None, None

        # Map interval format (yfinance uses "1d", twelvedata uses "1day")
        interval_map = {"1d": "1day", "1wk": "1week", "1mo": "1month"}
        td_interval = interval_map.get(interval, interval)

        price_data = fetcher.fetch_historical_data(ticker, start_date, end_date, td_interval)
        company_info = fetcher.fetch_company_info(ticker) if price_data is not None else None

        if price_data is not None and not price_data.empty:
            price_data = add_indicators(price_data)
            price_data = price_data.fillna(0)

        return price_data, company_info

    def _fetch_from_alphavantage(
        self, ticker: str, start_date, end_date, interval: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Fetch data from Alpha Vantage."""
        fetcher = self.fetchers.get("alphavantage")
        if not fetcher:
            return None, None

        price_data = fetcher.fetch_historical_data(ticker, start_date, end_date)
        company_info = fetcher.fetch_company_info(ticker) if price_data is not None else None

        if price_data is not None and not price_data.empty:
            price_data = add_indicators(price_data)
            price_data = price_data.fillna(0)

        return price_data, company_info

    def get_statistics(self) -> Dict:
        """
        Get statistics about data source usage.

        Returns:
            Dictionary with statistics for each source.
        """
        stats = {}
        for source, data in self.source_stats.items():
            if data["attempts"] > 0:
                success_rate = (data["successes"] / data["attempts"]) * 100
                stats[source] = {
                    "attempts": data["attempts"],
                    "successes": data["successes"],
                    "failures": data["failures"],
                    "success_rate": f"{success_rate:.1f}%",
                }
        return stats

    def print_statistics(self):
        """Print usage statistics for all data sources."""
        print("\n" + "=" * 60)
        print("DATA SOURCE STATISTICS")
        print("=" * 60)

        stats = self.get_statistics()
        if not stats:
            print("No data source usage recorded.")
            return

        for source, data in stats.items():
            print(f"\n{source.upper()}:")
            print(f"  Attempts:     {data['attempts']}")
            print(f"  Successes:    {data['successes']}")
            print(f"  Failures:     {data['failures']}")
            print(f"  Success Rate: {data['success_rate']}")

        print("\n" + "=" * 60 + "\n")


# Global manager instance
_manager_instance: Optional[DataSourceManager] = None


def get_manager(verbose: bool = True) -> DataSourceManager:
    """
    Get global data source manager instance (singleton pattern).

    Args:
        verbose: If True, print detailed information.

    Returns:
        DataSourceManager instance.
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DataSourceManager(verbose=verbose)
    return _manager_instance


def reset_manager():
    """Reset the global manager instance (useful for testing)."""
    global _manager_instance
    _manager_instance = None
