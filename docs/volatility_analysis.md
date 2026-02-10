# Volatility Analysis Results Guide

This guide explains how to interpret the volatility analysis output from Project Alpha.

## Output Files

When you run with `--screeners volatility`, the following files are generated:

| File | Description |
|------|-------------|
| `data/processed_data/volatile/prediction_table.csv` | Main results table |
| `data/processed_data/volatile_trend_trading_batch_0/` | **Trend/Momentum** candidates |
| `data/processed_data/volatile_value_trading_batch_0/` | **Value/Undervalued** candidates |
| `data/processed_data/volatile_breakout_trading_batch_0/` | **Breakout** candidates |

### Trading Categories (Auto-Filtered)

The system automatically categorizes stocks into three trading strategies:

| Category | Criteria | Charts |
|----------|----------|--------|
| **Trend** | GROWTH > 0.1%, VOLATILITY > 10% | Up to 50 |
| **Value** | RATE = "BELOW TREND" or "HIGHLY BELOW TREND" | Up to 50 |
| **Breakout** | RATE = "ALONG TREND", VOLATILITY < 15%, abs(GROWTH) < 0.1% | Up to 50 |

---

## Prediction Table Columns

### SYMBOL
Stock ticker symbol (e.g., `AAPL`, `MSFT`).

### SECTOR & INDUSTRY
Hierarchical classification from Yahoo Finance. Used internally to discover correlations between similar stocks.

### PRICE
Current price with currency (e.g., `278.12 USD`).

### RATE
Rating based on price deviation from predicted trend:

| Rating | Score Range | Interpretation |
|--------|-------------|----------------|
| **HIGHLY BELOW TREND** | > 3.0 | Price significantly below expected → Strong buy signal |
| **BELOW TREND** | 2.0 to 3.0 | Price below expected → Moderate buy signal |
| **ALONG TREND** | -2.0 to 2.0 | Price tracking prediction → Hold |
| **ABOVE TREND** | -3.0 to -2.0 | Price above expected → Moderate sell signal |
| **HIGHLY ABOVE TREND** | < -3.0 | Price significantly above expected → Strong sell signal |

### GROWTH
Daily expected growth rate (log-scale). Positive = upward trend, Negative = downward trend.

**Examples:**
- `0.008` = ~0.8% daily growth expectation (strong uptrend)
- `-0.002` = ~0.2% daily decline expectation (mild downtrend)

### VOLATILITY
Standard deviation of price predictions. Higher = more uncertainty.

**Interpretation:**
- `< 0.10`: Low volatility, stable price predictions
- `0.10 - 0.30`: Normal volatility
- `> 0.30`: High volatility, wider price swings expected

### MATCH
The most correlated stock (based on price movement patterns). Useful for:
- Finding similar stocks for diversification analysis
- Identifying sector/industry correlations
- Pair trading strategies

---

## How Data Flows to Other Screeners

The volatility analysis filters stocks for subsequent screeners:

```
Volatility Analysis Results
       │
       ├── Top 200 by GROWTH (high-volatility)
       │      └── Used by: TREND screener
       │          (Looking for momentum stocks)
       │
       └── Bottom 200 by GROWTH (low-volatility)
              └── Used by: BREAKOUT screener
                  (Looking for stable stocks breaking out)
```

---

## Trading Signals Interpretation

### For Trend/Momentum Trading
Focus on stocks with:
- High positive **GROWTH** (top of table)
- **RATE** = "BELOW TREND" or "HIGHLY BELOW TREND"
- High **VOLATILITY** (more price movement opportunity)

### For Value/Mean-Reversion
Focus on stocks with:
- **RATE** = "HIGHLY BELOW TREND" (undervalued)
- Low **VOLATILITY** (stable, predictable)
- Check **MATCH** for correlation (avoid correlated positions)

### For Breakout Trading
Focus on stocks with:
- Low or moderate **GROWTH** (consolidating)
- **RATE** = "ALONG TREND" (trading in expected range)
- Recently low **VOLATILITY** (compression before breakout)

---

## Example Analysis

```csv
SYMBOL,RATE,GROWTH,VOLATILITY,MATCH
HOOD,HIGHLY BELOW TREND,0.00546,0.400,PLTR
WDC,ALONG TREND,0.00817,0.139,STX
```

**HOOD (Robinhood):**
- Currently **below predicted trend** → potentially undervalued
- High growth expectation (+0.55%/day)
- High volatility → riskier but more opportunity
- Correlated with PLTR (similar fintech movement)

**WDC (Western Digital):**
- Tracking **along expected trend** → fair value
- Strong growth expectation (+0.82%/day)
- Moderate volatility → stable growth
- Correlated with STX (both storage sector)

---

## Command Examples

```bash
# Run only volatility analysis
python src/project_alpha.py --screeners volatility --cache

# Run all screeners (volatility filters feed into others)
python src/project_alpha.py --screeners all --cache

# Skip plotting to speed up analysis
python src/project_alpha.py --screeners volatility --no-plots --cache
```
