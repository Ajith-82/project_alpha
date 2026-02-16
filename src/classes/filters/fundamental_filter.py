import structlog
from typing import Optional, Dict, Any
from functools import lru_cache
from config.settings import settings

logger = structlog.get_logger()

class FundamentalFilter:
    """
    Filters stocks based on fundamental financial metrics.
    Uses Finnhub API for data, with graceful fallback if key is missing.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.finnhub_api_key
        self.client = None
        
        if self.api_key:
            try:
                import finnhub
                self.client = finnhub.Client(api_key=self.api_key)
            except ImportError:
                logger.warning("finnhub-python not installed. Fundamental filtering disabled.")
        else:
            logger.info("No Finnhub API key provided. Fundamental filtering will be skipped.")

    @lru_cache(maxsize=100)
    def check_health(self, ticker: str) -> Dict[str, Any]:
        """
        Check fundamental health of a ticker.
        
        Returns:
            Dict containing 'passed' (bool) and 'reason' (str)
        """
        if not self.client:
            return {"passed": True, "reason": "No API access", "details": {}}
            
        try:
            # Fetch basic financials
            metrics = self.client.company_basic_financials(ticker, 'all')['metric']
            
            reasons = []
            
            # Rule 1: Debt/Equity < 200%
            total_debt_equity = metrics.get('totalDebtToEquity')
            if total_debt_equity and total_debt_equity > 200: 
                reasons.append(f"High Debt/Equity: {total_debt_equity}%")
                
            # Rule 2: P/E Ratio (Value check, but loose for growth)
            # Avoid extremely high P/E or negative P/E
            pe_ttm = metrics.get('peTTM')
            if pe_ttm:
                if pe_ttm < 0:
                     reasons.append(f"Negative P/E: {pe_ttm}")
                elif pe_ttm > 100: # Very loose cap
                     reasons.append(f"Extremely High P/E: {pe_ttm}")

            # Rule 3: ROE > 0 (Profitability)
            roe = metrics.get('roeTTM')
            if roe and roe < 0:
                reasons.append(f"Negative ROE: {roe}%")

            # Rule 4: Revenue Growth > 0 (Growth)
            rev_growth = metrics.get('revenueGrowthTTMYoy')
            if rev_growth and rev_growth < 0:
                 reasons.append(f"Negative Revenue Growth: {rev_growth}%")
                
            if reasons:
                return {
                    "passed": False, 
                    "reason": "; ".join(reasons),
                    "details": metrics
                }
                
            return {"passed": True, "reason": "Fundamentals OK", "details": metrics}
            
        except Exception as e:
            logger.error("Fundamental check failed", ticker=ticker, error=str(e))
            # Fallback to passing if check fails, to not block trading on API hiccup
            return {"passed": True, "reason": "Check Error", "details": {"error": str(e)}}
