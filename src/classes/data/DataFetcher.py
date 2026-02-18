"""
Data Fetcher Module

Provides a robust interface to yfinance with retry logic,
progress callbacks, and error handling.
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import pandas as pd
import multitasking

from classes.data.provider_chain import YFinanceProvider, PolygonProvider, DataProvider


from classes.Add_indicators import add_indicators


logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a single stock fetch operation."""
    ticker: str
    price_data: Optional[pd.DataFrame]
    company_info: Optional[Dict]
    success: bool
    error: Optional[str] = None
    retries: int = 0


class StockFetcher:
    """
    Fetches stock data from yfinance with retry logic and progress callbacks.
    
    Features:
    - Exponential backoff retry (configurable attempts)
    - Progress callback for UI integration
    - Multithreaded batch downloads
    - Market-specific ticker formatting (e.g., .NS for India)
    """
    
    DEFAULT_RETRY_DELAYS = [1, 2, 4]  # seconds between retries
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delays: Optional[List[float]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        verbose: bool = False,
        provider_name: str = "yfinance",
        api_key: Optional[str] = None,
    ):

        """
        Initialize the StockFetcher.
        
        Args:
            max_retries: Maximum number of retry attempts per ticker
            retry_delays: List of delays (in seconds) between retries
            progress_callback: Callback function(ticker, completed, total)
            progress_callback: Callback function(ticker, completed, total)
            verbose: Enable verbose logging
            provider: Data provider name ("yfinance", "polygon")
            api_key: API key for the provider (optional)
        """
        self.max_retries = max_retries
        self.retry_delays = retry_delays or self.DEFAULT_RETRY_DELAYS
        self.progress_callback = progress_callback
        self.verbose = verbose
        
        # Initialize provider
        self.provider_name = provider_name
        if provider_name == "polygon":
            if not api_key:
                # Fallback to yfinance if key missing? Or raise?
                # For now let's raise or log warning and fallback
                logging.warning("Polygon API key missing, falling back to YFinance")
                self.provider = YFinanceProvider()
                self.provider_name = "yfinance"
            else:
                self.provider = PolygonProvider(api_key=api_key)
        else:
            self.provider = YFinanceProvider()

        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
    
    def _format_ticker(self, ticker: str, market: str) -> str:
        """Format ticker symbol for the target market."""
        ticker = ticker.upper().strip()
        if market == "india" and not ticker.endswith(".NS"):
            return f"{ticker}.NS"
        return ticker
    
    def _unformat_ticker(self, ticker: str, market: str) -> str:
        """Remove market suffix from ticker."""
        if market == "india" and ticker.endswith(".NS"):
            return ticker[:-3]
        return ticker
    
    def fetch_one(
        self,
        ticker: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> FetchResult:
        """
        Fetch data for a single stock with retry logic.
        
        Args:
            ticker: Stock ticker symbol
            market: Market identifier ("us" or "india")
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            interval: Data interval (default "1d")
            
        Returns:
            FetchResult with price_data, company_info, and status
        """
        formatted_ticker = self._format_ticker(ticker, market)
        original_ticker = ticker.upper().strip()
        
        # Default date range: 2 years
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        last_error = None
        retries = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Use provider to fetch data
                # Fetcher expects datetime objects for provider
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                price_data = self.provider.fetch_data(
                    ticker=formatted_ticker,
                    start_date=start_dt,
                    end_date=end_dt
                )

                
                if price_data.empty:
                    raise ValueError(f"No data returned for {formatted_ticker}")
                
                # Format and clean the data
                price_data = self._process_price_data(price_data)
                
                # Get company info
                company_info = self.provider.get_company_info(formatted_ticker)

                
                logger.debug(f"Successfully fetched {original_ticker} ({len(price_data)} rows)")
                
                return FetchResult(
                    ticker=original_ticker,
                    price_data=price_data,
                    company_info=company_info,
                    success=True,
                    retries=attempt,
                )
                
            except Exception as e:
                last_error = str(e)
                retries = attempt
                
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} for {original_ticker} "
                        f"after {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {original_ticker} after {self.max_retries} retries: {e}")
        
        return FetchResult(
            ticker=original_ticker,
            price_data=None,
            company_info=None,
            success=False,
            error=last_error,
            retries=retries,
        )
    
    def _process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean price data."""
        df = df.copy()
        
        # Format index as date strings
        df.index = df.index.strftime("%Y-%m-%d")
        df.sort_index(inplace=True)
        
        # Add Adj Close if not present
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"].copy()
        
        # Reorder columns
        desired_columns = [
            "Open", "High", "Low", "Close", "Adj Close",
            "Volume", "Dividends", "Stock Splits"
        ]
        available_columns = [c for c in desired_columns if c in df.columns]
        df = df[available_columns]
        
        # Remove duplicates and fill missing values
        df = df.loc[~df.index.duplicated(keep="first")]
        df = df.ffill().bfill().drop_duplicates()
        
        # Add technical indicators
        try:
            df = add_indicators(df)
        except Exception as e:
            logger.warning(f"Failed to add indicators: {e}")
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        return df
    
    def fetch_batch(
        self,
        tickers: List[str],
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        max_threads: Optional[int] = None,
    ) -> Dict[str, FetchResult]:
        """
        Fetch data for multiple stocks using the provider's batch download.
        
        Uses the provider's fetch_batch_data() for a single bulk request
        (e.g. one yf.download call for all tickers), then processes each
        result and fetches company info in parallel.
        
        Args:
            tickers: List of ticker symbols
            market: Market identifier
            start_date: Start date
            end_date: End date
            interval: Data interval
            max_threads: Maximum concurrent threads for company info (default: 8)
            
        Returns:
            Dictionary mapping tickers to FetchResult objects
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Normalize tickers
        tickers = list(set(t.upper().strip() for t in tickers if t))
        total = len(tickers)

        # Default date range: 2 years
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Format tickers for the target market
        formatted_map = {}  # formatted_ticker -> original_ticker
        formatted_tickers = []
        for t in tickers:
            ft = self._format_ticker(t, market)
            formatted_map[ft] = t
            formatted_tickers.append(ft)

        # ---- Stage 1: Batch price download (single HTTP request for all tickers) ----
        try:
            raw_data = self.provider.fetch_batch_data(formatted_tickers, start_dt, end_dt)
        except Exception as e:
            logger.error("batch_download_failed", error=str(e))
            raw_data = {}

        # ---- Stage 2: Process each ticker's data ----
        results: Dict[str, FetchResult] = {}
        successful_formatted = []

        for idx, ft in enumerate(formatted_tickers):
            original = formatted_map[ft]
            if ft in raw_data and raw_data[ft] is not None and not raw_data[ft].empty:
                try:
                    price_data = self._process_price_data(raw_data[ft])
                    results[original] = FetchResult(
                        ticker=original,
                        price_data=price_data,
                        company_info=None,  # filled in Stage 3
                        success=True,
                        retries=0,
                    )
                    successful_formatted.append(ft)
                except Exception as e:
                    logger.warning(f"Processing failed for {original}: {e}")
                    results[original] = FetchResult(
                        ticker=original, price_data=None, company_info=None,
                        success=False, error=str(e), retries=0,
                    )
            else:
                results[original] = FetchResult(
                    ticker=original, price_data=None, company_info=None,
                    success=False, error="No data returned in batch", retries=0,
                )

            # Report progress
            if self.progress_callback:
                self.progress_callback(original, idx + 1, total)

        # ---- Stage 3: Fetch company info in parallel (only for successes) ----
        if successful_formatted:
            info_threads = min(max_threads or 8, len(successful_formatted), 8)

            def _fetch_info(ft: str):
                try:
                    return ft, self.provider.get_company_info(ft)
                except Exception:
                    return ft, {}

            with ThreadPoolExecutor(max_workers=info_threads) as pool:
                futures = {pool.submit(_fetch_info, ft): ft for ft in successful_formatted}
                for future in as_completed(futures):
                    ft, info = future.result()
                    original = formatted_map[ft]
                    if original in results and results[original].success:
                        results[original] = FetchResult(
                            ticker=results[original].ticker,
                            price_data=results[original].price_data,
                            company_info=info,
                            success=True,
                            retries=0,
                        )

        return results
    
    def results_to_dict(
        self, results: Dict[str, FetchResult]
    ) -> Dict[str, Union[List[str], Dict[str, pd.DataFrame], Dict[str, Dict]]]:
        """
        Convert FetchResult dictionary to the legacy format expected by other modules.
        
        Args:
            results: Dictionary of FetchResult objects
            
        Returns:
            Dictionary with 'tickers', 'price_data', and 'company_info' keys
        """
        successful = {k: v for k, v in results.items() if v.success and v.price_data is not None}
        
        return {
            "tickers": list(successful.keys()),
            "price_data": {k: v.price_data for k, v in successful.items()},
            "company_info": {k: v.company_info for k, v in successful.items()},
        }
