#!/usr/bin/env python
"""
Twelve Data API fetcher for stock market data.
Free tier: 800 requests/day
API Documentation: https://twelvedata.com/docs
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Dict
from twelvedata import TDClient


class TwelveDataFetcher:
    """Fetches stock data from Twelve Data API."""

    def __init__(self, api_key: str):
        """
        Initialize Twelve Data client.

        Args:
            api_key: Twelve Data API key.
        """
        self.client = TDClient(apikey=api_key)
        self.name = "twelvedata"

    def fetch_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1day",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format or timestamp.
            end_date: End date in YYYY-MM-DD format or timestamp.
            interval: Data interval (1day, 1week, 1month, etc.).

        Returns:
            DataFrame with OHLCV data, or None if error.
        """
        try:
            # Convert timestamps to dates if needed
            if isinstance(start_date, int):
                start_date = datetime.fromtimestamp(start_date).strftime("%Y-%m-%d")
            if isinstance(end_date, int):
                end_date = datetime.fromtimestamp(end_date).strftime("%Y-%m-%d")

            # Fetch time series data
            ts = self.client.time_series(
                symbol=ticker,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                outputsize=5000,  # Maximum output size
            )

            # Get data as DataFrame
            df = ts.as_pandas()

            if df is None or df.empty:
                print(f"TwelveData: No data returned for {ticker}")
                return None

            # Rename columns to match yfinance format
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df = df.rename(columns=column_mapping)

            # Add missing columns with default values
            df["Adj Close"] = df["Close"].copy()
            df["Dividends"] = 0
            df["Stock Splits"] = 0

            # Ensure index is string format (YYYY-MM-DD)
            df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")

            # Reorder columns to match expected format
            df = df[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                    "Dividends",
                    "Stock Splits",
                ]
            ]

            # Sort by date
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"TwelveData error fetching {ticker}: {e}")
            return None

    def fetch_company_info(self, ticker: str) -> Optional[Dict]:
        """
        Fetch company information for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dictionary with company info, or None if error.
        """
        try:
            # Get profile information
            profile = self.client.get_quote(symbol=ticker)

            if not profile:
                return None

            # Convert to format similar to yfinance
            company_info = {
                "symbol": profile.get("symbol", ticker),
                "name": profile.get("name", ""),
                "exchange": profile.get("exchange", ""),
                "currency": profile.get("currency", "USD"),
                "type": profile.get("type", ""),
                "country": profile.get("country", ""),
                # Note: Twelve Data free tier has limited fundamental data
                "sector": "",
                "industry": "",
            }

            # Try to get additional info if available
            try:
                logo = self.client.get_logo(symbol=ticker)
                if logo:
                    company_info["logo_url"] = logo.get("url", "")
            except:
                pass

            return company_info

        except Exception as e:
            print(f"TwelveData error fetching company info for {ticker}: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test API connection and key validity.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Try to fetch a simple quote for a major stock
            result = self.client.get_quote(symbol="AAPL")
            return result is not None and "symbol" in result
        except Exception as e:
            print(f"TwelveData connection test failed: {e}")
            return False


def create_twelve_data_fetcher(api_key: str) -> Optional[TwelveDataFetcher]:
    """
    Create and validate a Twelve Data fetcher instance.

    Args:
        api_key: Twelve Data API key.

    Returns:
        TwelveDataFetcher instance if valid, None otherwise.
    """
    if not api_key or api_key == "your_twelvedata_api_key_here":
        print("TwelveData: Invalid or missing API key")
        return None

    fetcher = TwelveDataFetcher(api_key)

    # Test connection
    if not fetcher.test_connection():
        print("TwelveData: Connection test failed")
        return None

    return fetcher
