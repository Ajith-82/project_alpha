import structlog
from typing import List
from datetime import datetime, timedelta
import yfinance as yf
from config.settings import settings

logger = structlog.get_logger()

class NewsFetcher:
    """
    Fetches news headlines for a given ticker.
    Prioritizes Finnhub API if available, falls back to yfinance.
    """
    def __init__(self):
        self.finnhub_client = None
        if settings.finnhub_api_key:
            try:
                import finnhub
                self.finnhub_client = finnhub.Client(api_key=settings.finnhub_api_key)
            except ImportError:
                logger.warning("finnhub-python not installed. News fetching will use yfinance fallback.")
    
    def fetch_headlines(self, ticker: str, days: int = 3) -> List[str]:
        """
        Fetch news headlines for the last N days.
        """
        headlines = []
        try:
            if self.finnhub_client:
                # Use Finnhub
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                # Finnhub company_news returns list of dicts
                news = self.finnhub_client.company_news(ticker, _from=start_date, to=end_date)
                # Sort by datetime desc just in case
                if news:
                    headlines = [n['headline'] for n in news[:10]] # Limit to recent 10 to reduce noise
            else:
                # Fallback to yfinance
                # Note: yfinance news is often just a few latest items
                stock = yf.Ticker(ticker)
                news = stock.news
                headlines = [n.get('title') for n in news] if news else []
                
        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {e}")
            
        return headlines
