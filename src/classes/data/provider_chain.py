from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from datetime import datetime
import structlog
from typing import Optional, List, Union
from polygon import RESTClient
from requests.exceptions import HTTPError

logger = structlog.get_logger()

class DataProvider(ABC):
    """
    Abstract base class for market data providers.
    Ensures consistent interface and data format across different sources.
    """
    
    @abstractmethod
    def fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a ticker.
        
        Args:
            ticker: Symbol to fetch (e.g., 'AAPL')
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            pd.DataFrame with columns: [Open, High, Low, Close, Volume]
            Index should be DatetimeIndex.
        """
        pass

    def fetch_batch_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> dict:
        """
        Fetch historical OHLCV data for multiple tickers at once.
        
        Default implementation calls fetch_data() individually.
        Providers should override this for more efficient batch operations.
        
        Args:
            tickers: List of symbols to fetch
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            Dict mapping ticker -> pd.DataFrame with columns [Open, High, Low, Close, Volume]
        """
        results = {}
        for ticker in tickers:
            try:
                df = self.fetch_data(ticker, start_date, end_date)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.warning("batch_fetch_individual_failed", ticker=ticker, error=str(e))
        return results

    @abstractmethod
    def check_health(self) -> bool:
        """
        Check if the provider is reachable and functioning.
        
        Returns:
            bool: True if healthy, False otherwise.
        """
        pass

    @abstractmethod
    def get_company_info(self, ticker: str) -> dict:
        """
        Fetch company metadata (sector, industry, etc.).
        
        Args:
            ticker: Symbol to fetch
            
        Returns:
            Dictionary with company info.
        """
        pass



class YFinanceProvider(DataProvider):
    """
    Data provider implementation using yfinance.
    """

    @staticmethod
    def _normalize_single_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a single-ticker DataFrame to standard OHLCV columns."""
        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex columns (yfinance returns them for multi-ticker downloads)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            df.rename(columns={c: c.capitalize() for c in df.columns}, inplace=True)

        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index)
        return df

    def fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logger.info("fetching_data_yfinance", ticker=ticker, start=start_date, end=end_date)
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            return self._normalize_single_df(df)
        except Exception as e:
            logger.error("data_fetch_failed", ticker=ticker, provider="yfinance", error=str(e))
            raise

    def fetch_batch_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> dict:
        """
        Fetch data for multiple tickers in a single yf.download() call.

        This is dramatically faster than individual downloads because yfinance
        batches the HTTP requests internally. For 500 tickers this typically
        takes 30-60 seconds instead of 15-30 minutes.
        """
        logger.info("batch_fetch_yfinance", count=len(tickers), start=start_date, end=end_date)
        try:
            # yf.download accepts a list of tickers and returns a MultiIndex DataFrame
            # with (Price, Ticker) column levels
            raw = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                threads=True,
                group_by="ticker",
            )

            if raw.empty:
                logger.warning("batch_fetch_empty", provider="yfinance")
                return {}

            results = {}
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

            # If only one ticker, yf.download returns flat columns (not grouped)
            if len(tickers) == 1:
                ticker = tickers[0]
                df = self._normalize_single_df(raw)
                if not df.empty:
                    results[ticker] = df
                return results

            # Multi-ticker: columns are MultiIndex (Ticker, Price)
            for ticker in tickers:
                try:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    df = raw[ticker].copy()
                    # Drop rows that are all NaN (ticker might not have data for full range)
                    df = df.dropna(how="all")
                    if df.empty:
                        continue

                    # Handle column name casing
                    if not all(col in df.columns for col in required_cols):
                        df.rename(columns={c: c.capitalize() for c in df.columns}, inplace=True)

                    if all(col in df.columns for col in required_cols):
                        df = df[required_cols].copy()
                        df.index = pd.to_datetime(df.index)
                        results[ticker] = df
                except Exception as e:
                    logger.warning("batch_ticker_extract_failed", ticker=ticker, error=str(e))

            logger.info("batch_fetch_complete", succeeded=len(results), total=len(tickers))
            return results

        except Exception as e:
            logger.error("batch_fetch_failed", provider="yfinance", error=str(e))
            # Fall back to individual downloads
            logger.info("falling_back_to_individual_downloads")
            return super().fetch_batch_data(tickers, start_date, end_date)

    def check_health(self) -> bool:
        """Check health by fetching 1 day of SPY data."""
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1d")
            return not hist.empty
        except Exception as e:
            logger.error("health_check_failed", provider="yfinance", error=str(e))
            return False

    def get_company_info(self, ticker: str) -> dict:
        """Fetch company info using yfinance."""
        try:
            return yf.Ticker(ticker).info
        except Exception as e:
            logger.error("info_fetch_failed", ticker=ticker, provider="yfinance", error=str(e))
            return {}



class PolygonProvider(DataProvider):
    """
    Data provider implementation using Polygon.io.
    Requires an API key.
    """
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key=api_key)

    def fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        logger.info("fetching_data_polygon", ticker=ticker, start=start_date, end=end_date)
        try:
            # Polygon expects YYYY-MM-DD strings
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Fetch daily aggregates (OHLCV)
            # Default to limit 50000 to get enough history if needed
            aggs = []
            for a in self.client.list_aggs(
                ticker, 
                multiplier=1, 
                timespan="day", 
                from_=start_str, 
                to=end_str, 
                limit=50000
            ):
                aggs.append(a)

            if not aggs:
                logger.warning("empty_data_returned", ticker=ticker, provider="polygon")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for agg in aggs:
                # Polygon returns timestamp in milliseconds
                dt = datetime.fromtimestamp(agg.timestamp / 1000.0)
                data.append({
                    'Date': dt,
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            
            # Ensure index is DatetimeIndex and sorted
            df.sort_index(inplace=True)
            
            # Helper to map columns if needed, but we built it correctly
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols]
            
            return df

        except Exception as e:
             logger.error("data_fetch_failed", ticker=ticker, provider="polygon", error=str(e))
             raise

    def check_health(self) -> bool:
        """
        Check health by fetching last trade for a major ticker.
        """
        try:
             # Use a lightweight call like get_last_trade or similar
             # list_aggs with limit 1 is also fine
             # Let's try to get the last trade for SPY
             last_trade = self.client.get_last_trade(ticker="SPY")
             return last_trade is not None
        except Exception as e:
             logger.error("health_check_failed", provider="polygon", error=str(e))
             return False

    def get_company_info(self, ticker: str) -> dict:
        """
        Fetch company info using Polygon.
        """
        try:
            details = self.client.get_ticker_details(ticker)
            if details:
                # Map Polygon details to expected schema if needed
                # For now just return the dict representation
                # attributes: sector, industry, description, etc.
                # Project Alpha expects: 'sector', 'industry' keys primarily.
                # Polygon uses 'sic_description' or 'classification' usually.
                # Let's try to map best effort.
                info = {
                    "sector": getattr(details, "sic_description", "Unknown"), 
                    "industry": getattr(details, "sic_description", "Unknown"), # Polygon might differ
                    # Add standard keys if available
                }
                # Check for standard keys
                if hasattr(details, "market_cap"): info["marketCap"] = details.market_cap
                # ... other fields
                return info
            return {}
        except Exception as e:
             logger.error("info_fetch_failed", ticker=ticker, provider="polygon", error=str(e))
             return {}



