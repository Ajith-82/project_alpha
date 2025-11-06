#!/usr/bin/env python
"""
Alpha Vantage API fetcher for stock market data.
Free tier: 25 requests/day (5 requests/minute)
API Documentation: https://www.alphavantage.co/documentation/
"""
import pandas as pd
import time
from datetime import datetime
from typing import Optional, Dict
from alpha_vantage.timeseries import TimeSeries


class AlphaVantageFetcher:
    """Fetches stock data from Alpha Vantage API."""

    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key.
        """
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format="pandas")
        self.name = "alphavantage"
        self.rate_limit_delay = 12  # 5 requests per minute = 12 seconds between requests

    def fetch_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "daily",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format or timestamp.
            end_date: End date in YYYY-MM-DD format or timestamp.
            interval: Data interval (not used for Alpha Vantage, always daily for free tier).

        Returns:
            DataFrame with OHLCV data, or None if error.
        """
        try:
            # Convert timestamps to dates if needed
            if isinstance(start_date, int):
                start_date = datetime.fromtimestamp(start_date).strftime("%Y-%m-%d")
            if isinstance(end_date, int):
                end_date = datetime.fromtimestamp(end_date).strftime("%Y-%m-%d")

            # Fetch daily adjusted data (includes splits and dividends)
            # outputsize='full' gets up to 20 years of data
            df, meta_data = self.ts.get_daily_adjusted(
                symbol=ticker, outputsize="full"
            )

            if df is None or df.empty:
                print(f"AlphaVantage: No data returned for {ticker}")
                return None

            # Rename columns to match yfinance format
            column_mapping = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
                "7. dividend amount": "Dividends",
                "8. split coefficient": "Stock Splits",
            }
            df = df.rename(columns=column_mapping)

            # Filter date range
            df.index = pd.to_datetime(df.index)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Convert index to string format (YYYY-MM-DD)
            df.index = df.index.strftime("%Y-%m-%d")

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

            # Convert Volume to numeric (it comes as string)
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

            # Rate limiting - wait before next request
            time.sleep(self.rate_limit_delay)

            return df

        except Exception as e:
            print(f"AlphaVantage error fetching {ticker}: {e}")
            # Check for rate limit error
            if "API call frequency" in str(e) or "rate limit" in str(e).lower():
                print("AlphaVantage: Rate limit reached (25 requests/day or 5 requests/minute)")
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
            # Alpha Vantage has a company overview endpoint
            # Using requests directly since alpha_vantage library might not have it
            import requests

            url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key,
            }

            response = requests.get(url, params=params)
            data = response.json()

            if not data or "Symbol" not in data:
                print(f"AlphaVantage: No company info for {ticker}")
                return None

            # Convert to format similar to yfinance
            company_info = {
                "symbol": data.get("Symbol", ticker),
                "name": data.get("Name", ""),
                "description": data.get("Description", ""),
                "exchange": data.get("Exchange", ""),
                "currency": data.get("Currency", "USD"),
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "country": data.get("Country", ""),
                "marketCapitalization": data.get("MarketCapitalization", ""),
                "peRatio": data.get("PERatio", ""),
                "dividendYield": data.get("DividendYield", ""),
            }

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return company_info

        except Exception as e:
            print(f"AlphaVantage error fetching company info for {ticker}: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test API connection and key validity.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Try to fetch a simple quote for a major stock
            df, _ = self.ts.get_quote_endpoint(symbol="AAPL")
            return df is not None and not df.empty
        except Exception as e:
            print(f"AlphaVantage connection test failed: {e}")
            if "Invalid API call" in str(e) or "invalid" in str(e).lower():
                print("AlphaVantage: Invalid API key")
            return False


def create_alpha_vantage_fetcher(api_key: str) -> Optional[AlphaVantageFetcher]:
    """
    Create and validate an Alpha Vantage fetcher instance.

    Args:
        api_key: Alpha Vantage API key.

    Returns:
        AlphaVantageFetcher instance if valid, None otherwise.
    """
    if not api_key or api_key == "your_alphavantage_api_key_here":
        print("AlphaVantage: Invalid or missing API key")
        return None

    fetcher = AlphaVantageFetcher(api_key)

    # Test connection (optional - costs 1 API call)
    # Commenting out to save API calls, will fail gracefully on first use
    # if not fetcher.test_connection():
    #     print("AlphaVantage: Connection test failed")
    #     return None

    return fetcher
