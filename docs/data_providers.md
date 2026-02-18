# Data Provider Chain

The Data Provider Chain is a new abstraction layer designed to unify market data fetching from multiple sources (YFinance, Polygon.io, etc.). It ensures that regardless of the source, the application receives data in a consistent format.

## Architecture

The core is the abstract base class `DataProvider` defined in `src/classes/data/provider_chain.py`. All specific providers inherit from this class and must implement:

1.  `fetch_data(ticker, start_date, end_date) -> pd.DataFrame`
2.  `check_health() -> bool`

### Standardized Output

All providers return a pandas DataFrame with the following columns:
- `Open` (float)
- `High` (float)
- `Low` (float)
- `Close` (float)
- `Volume` (int/float)

The index is always a `DatetimeIndex` sorted in ascending order.

## Supported Providers

### 1. YFinanceProvider (`yfinance`) - Default
- **Source**: Yahoo Finance (via `yfinance` library).
- **Pros**: Free, extensive coverage, no API key required.
- **Cons**: Rate limits can be unpredictable, data quality varies.
- **Health Check**: Fetches 1 day of historical data for `SPY`.

### 2. PolygonProvider (`polygon-api-client`)
- **Source**: Polygon.io API.
- **Requires**: API Key (set via `PA_POLYGON_API_KEY` or `Settings`).
- **Pros**: High quality, official API, faster.
- **Cons**: Free tier has rate limits (5 calls/min).
- **Health Check**: Fetches the last trade for `SPY`.

## Usage Example

```python
from src.classes.data.provider_chain import YFinanceProvider, PolygonProvider
from datetime import datetime, timedelta

# Initialize provider
# provider = YFinanceProvider()
provider = PolygonProvider(api_key="YOUR_KEY")

# Check health
if provider.check_health():
    print("Provider is healthy")

# Fetch data
start = datetime.now() - timedelta(days=30)
end = datetime.now()
df = provider.fetch_data("AAPL", start, end)

print(df.head())
```

## Future Improvements

- **AlphaVantage Support**: Add `AlphaVantageProvider`.
- **Failover Logic**: Implement a `ChainProvider` that tries providers in sequence (e.g., Polygon -> YFinance).
- **Rate Limiting**: Add strict rate limiting decorators to respect API tier limits.
