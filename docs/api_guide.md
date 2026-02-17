# API Developer Guide

## Core Modules

### Consensus Engine
`src.classes.screeners.consensus.ConsensusEngine`

Aggregates signals from various screeners and filters to produce a single confidence score.

**Usage:**
```python
engine = ConsensusEngine()
result = engine.calculate_score(ticker, screener_results, filter_scores)
print(result.score, result.recommendation)
```

**Key Methods:**
- `calculate_score(ticker: str, screener_results: Dict[str, ScreenerResult], filter_results: Optional[Dict[str, float]]) -> ConsensusResult`: Calculates score based on weights from `settings.py`.

### News Fetcher
`src.classes.data.news_fetcher.NewsFetcher`

Retrieves news headlines for a given ticker, prioritizing Finnhub API over yfinance scraping.

**Usage:**
```python
fetcher = NewsFetcher()
headlines = fetcher.fetch_headlines("AAPL", days=3)
```

## Filters

### Fundamental Filter
`src.classes.filters.fundamental_filter.FundamentalFilter`

Evaluates financial health using metrics like Debt/Equity, P/E Ratio, and ROE. Uses standard Finnhub API.

**Key Methods:**
- `check_health(ticker: str) -> Dict[str, Any]`: Returns pass/fail status and reasoning. **Cached (LRU).**

### Sentiment Filter
`src.classes.filters.sentiment_filter.SentimentFilter`

Analyzes news sentiment using a pre-trained FinBERT model.

**Key Methods:**
- `analyze_sentiment(headlines: List[str]) -> Dict[str, Any]`: Returns aggregated sentiment score and label. **Singleton Model.**

## Data Management

### Validators
`src.classes.data.validators`

Ensures data integrity for OHLCV DataFrames.

**Functions:**
- `validate_data_quality(df: pd.DataFrame, ticker: str) -> pd.DataFrame`: Runs all validation checks.
- `repair_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame`: Auto-fixes minor issues like negative volume or missing prices.

### Download
`src.classes.Download`

Handles data fetching and local caching.

## Screeners

### Breakout Screener
`src.classes.screeners.breakout.BreakoutScreener`

Detects consolidation breakouts using ADX and ATR expansion.

### Trendline Screener
`src.classes.screeners.trendline.TrendlineScreener`

Identifies stocks in strong uptrends using linear regression slope analysis.
