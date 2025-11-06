# Multi-Source Data Fetching System

## Overview

Project Alpha now supports multiple data sources with automatic fallback! This ensures reliable data fetching even when one source fails or has rate limits.

## Supported Data Sources

### 1. **Yahoo Finance (yfinance)** - Default
- **Cost**: FREE, unlimited
- **API Key**: Not required
- **Rate Limits**: None (but can be unreliable)
- **Coverage**: Global stocks, indices, ETFs, crypto
- **Status**: Always available by default

### 2. **Twelve Data**
- **Cost**: FREE tier - 800 requests/day
- **API Key**: Required (get from https://twelvedata.com/)
- **Rate Limits**: 800 requests/day, 8 requests/minute
- **Coverage**: 5000+ global exchanges
- **Status**: Optional, enabled when API key provided

### 3. **Alpha Vantage**
- **Cost**: FREE tier - 25 requests/day
- **API Key**: Required (get from https://www.alphavantage.co/)
- **Rate Limits**: 25 requests/day, 5 requests/minute
- **Coverage**: US stocks, forex, crypto, fundamentals
- **Status**: Optional, enabled when API key provided

## Quick Start

### Step 1: Install Dependencies

```bash
# Using Poetry (recommended)
poetry lock
poetry install

# Or using pip
pip install python-dotenv twelvedata alpha-vantage
```

### Step 2: Configure API Keys

1. Copy the example configuration:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
# Twelve Data (optional but recommended)
TWELVE_DATA_API_KEY=your_actual_api_key_here

# Alpha Vantage (optional)
ALPHA_VANTAGE_API_KEY=your_actual_api_key_here

# Configure source priority (optional)
DATA_SOURCE_PRIORITY=yfinance,twelvedata,alphavantage
```

### Step 3: Run Project Alpha

```bash
# US Market
python src/project_alpha.py --market us

# Indian Market
python src/project_alpha.py --market india

# Or use the shell scripts
./run_us_stock_scanner.sh
./run_india_stock_scanner.sh
```

## How It Works

### Automatic Fallback

The system tries data sources in priority order:
1. **Primary**: Tries yfinance first (fast, no limits)
2. **Fallback 1**: If yfinance fails, tries Twelve Data
3. **Fallback 2**: If Twelve Data fails, tries Alpha Vantage

```
Request AAPL data
    ↓
Try yfinance → Success ✓
    ↓
Return data
```

```
Request XYZ data
    ↓
Try yfinance → Failed ✗
    ↓
Try Twelve Data → Success ✓
    ↓
Return data
```

### Statistics Tracking

After downloading data, you'll see statistics like:

```
============================================================
DATA SOURCE STATISTICS
============================================================

YFINANCE:
  Attempts:     500
  Successes:    495
  Failures:     5
  Success Rate: 99.0%

TWELVEDATA:
  Attempts:     5
  Successes:    5
  Failures:     0
  Success Rate: 100.0%

============================================================
```

## Configuration Options

### Environment Variables

Edit `.env` to customize behavior:

```bash
# Source Priority (comma-separated)
DATA_SOURCE_PRIORITY=yfinance,twelvedata,alphavantage

# Enable/Disable specific sources
ENABLE_YFINANCE=true
ENABLE_TWELVE_DATA=true
ENABLE_ALPHA_VANTAGE=true
```

### Priority Examples

**Conservative (minimize API usage):**
```bash
DATA_SOURCE_PRIORITY=yfinance
ENABLE_TWELVE_DATA=false
ENABLE_ALPHA_VANTAGE=false
```

**Aggressive (maximize reliability):**
```bash
DATA_SOURCE_PRIORITY=twelvedata,yfinance,alphavantage
```

**Backup only:**
```bash
DATA_SOURCE_PRIORITY=yfinance,twelvedata
ENABLE_ALPHA_VANTAGE=false
```

## Getting API Keys

### Twelve Data (Recommended)

1. Visit: https://twelvedata.com/
2. Click "Get API Key" or "Sign Up"
3. Choose FREE plan (800 requests/day)
4. Copy API key to `.env`

**Why Twelve Data?**
- Higher rate limits (800 vs 25/day)
- Good data quality
- Fast responses
- Covers global markets

### Alpha Vantage

1. Visit: https://www.alphavantage.co/support/#api-key
2. Enter email, click "GET FREE API KEY"
3. Copy API key to `.env`

**Why Alpha Vantage?**
- Excellent for fundamental data
- High-quality historical data
- Good for low-volume usage
- Strong documentation

## Architecture

### Key Components

```
project_alpha/
├── src/classes/
│   ├── DataSourceConfig.py      # Configuration loader
│   ├── DataSourceManager.py     # Fallback orchestrator
│   ├── TwelveDataFetcher.py     # Twelve Data integration
│   ├── AlphaVantageFetcher.py   # Alpha Vantage integration
│   └── Download.py              # Updated to use manager
├── .env.example                 # Configuration template
└── .env                         # Your API keys (gitignored)
```

### Data Flow

```
User Request
    ↓
Download.py
    ↓
DataSourceManager
    ├→ Try Source 1 (yfinance)
    ├→ Try Source 2 (twelvedata) [if needed]
    └→ Try Source 3 (alphavantage) [if needed]
    ↓
Return Data + Statistics
```

## Troubleshooting

### "No data sources available"

**Problem**: No sources configured or all sources failed to initialize.

**Solution**:
1. Ensure `.env` file exists (copy from `.env.example`)
2. Check API keys are valid (no quotes, no spaces)
3. Test with yfinance only (requires no API key)

### "Rate limit reached"

**Problem**: You've exceeded the daily/minutely rate limit.

**Solution**:
1. Wait for rate limit reset (daily/minutely)
2. Enable additional sources in `.env`
3. Adjust `DATA_SOURCE_PRIORITY` to use different sources first

### "Invalid API key"

**Problem**: API key is incorrect or expired.

**Solution**:
1. Verify key in `.env` matches the one from provider
2. Check for extra spaces or quotes
3. Regenerate key from provider website

### Import errors

**Problem**: Missing dependencies after installation.

**Solution**:
```bash
# Reinstall dependencies
poetry install

# Or manually install
pip install python-dotenv twelvedata alpha-vantage pandas yfinance
```

## Best Practices

### For Production Use

1. **Use multiple sources**: Configure at least 2 sources for redundancy
2. **Monitor statistics**: Check success rates regularly
3. **Respect rate limits**: Don't disable rate limit delays
4. **Cache aggressively**: Use `--cache` flag and database storage

### For Development

1. **Start with yfinance**: Test without API keys first
2. **Add sources gradually**: Enable Twelve Data, then Alpha Vantage
3. **Test with small datasets**: Use `-s` flag to test specific symbols
4. **Check logs**: Review which sources are being used

### Rate Limit Management

```bash
# Daily request limits
Yfinance:      ∞ (unlimited)
Twelve Data:   800 requests
Alpha Vantage: 25 requests

# For 500 stocks, optimal configuration:
# - Enable yfinance (primary)
# - Enable Twelve Data (catches ~5-10 failures)
# - Enable Alpha Vantage (catches rare failures)
```

## Advanced Usage

### Programmatic Access

```python
from classes.DataSourceManager import get_manager

# Get manager instance
manager = get_manager(verbose=True)

# Fetch data with automatic fallback
price_data, company_info = manager.fetch_stock_data(
    market="us",
    ticker="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d"
)

# Check statistics
stats = manager.get_statistics()
print(stats)
```

### Custom Source Priority

```python
from classes.DataSourceConfig import DataSourceConfig

# Override configuration
config = DataSourceConfig()
config.source_priority = ["twelvedata", "yfinance"]
```

## Performance Impact

### Speed Comparison

| Source | Average Response Time | Notes |
|--------|----------------------|-------|
| yfinance | 0.5-1.0s per stock | Fast, but occasional timeouts |
| Twelve Data | 0.3-0.8s per stock | Fast, reliable API |
| Alpha Vantage | 12s per stock | Slow due to rate limiting |

### Recommendations

- **Default setup**: yfinance → twelvedata → alphavantage
- **For speed**: yfinance only (accepts occasional failures)
- **For reliability**: twelvedata → yfinance → alphavantage
- **For fundamentals**: alphavantage → twelvedata → yfinance

## Migration from Old System

The old system used yfinance exclusively. The new system:
- ✓ **Backward compatible**: Works without any configuration changes
- ✓ **Drop-in replacement**: No code changes needed
- ✓ **Optional features**: All new sources are optional
- ✓ **Same output format**: Data format unchanged

To migrate:
1. Update code: `git pull` (already done if reading this)
2. Install deps: `poetry install`
3. Optional: Add API keys to `.env`
4. Run normally: Everything works as before, but better!

## Support

### Getting Help

- **Issues**: https://github.com/Ajith-82/project_alpha/issues
- **Documentation**: This file + README.md
- **API Docs**:
  - Twelve Data: https://twelvedata.com/docs
  - Alpha Vantage: https://www.alphavantage.co/documentation/

### Contributing

Want to add more data sources? The system is modular:

1. Create `YourSourceFetcher.py` in `src/classes/`
2. Implement `fetch_historical_data()` and `fetch_company_info()`
3. Add to `DataSourceManager.py`
4. Update this documentation

## Summary

✅ **Multiple data sources** for reliability
✅ **Automatic fallback** when sources fail
✅ **Zero configuration** for basic usage (yfinance)
✅ **Optional API keys** for premium sources
✅ **Statistics tracking** for monitoring
✅ **Backward compatible** with existing scripts

**Recommended Setup**: Enable Twelve Data (FREE, 800 req/day) for best reliability!
