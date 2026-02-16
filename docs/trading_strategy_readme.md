# Project Alpha — Trading Strategy Guide

> **Disclaimer:** Project Alpha is provided for **educational and research purposes only**. It does not constitute financial advice. Always do your own due diligence before trading.

---

## What Does This Tool Do?

Project Alpha scans hundreds of stocks (S&P 500 or NSE 500), trains a probabilistic model on their price history, and categorises them into **actionable trading signals** across three strategies — **Trend**, **Value**, and **Breakout** — supplemented by five technical screeners.

```
Market Data (Yahoo Finance)
        │
        ▼
┌──────────────────────┐
│ Volatility Model     │  ← Bayesian hierarchical model (TensorFlow Probability)
│ (Growth, Volatility, │
│  Trend Rating, Match)│
└────────┬─────────────┘
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
  TREND     VALUE     BREAKOUT       ← Three trading categories
 (Momentum) (Mean-Rev) (Consolidation)
    │         │          │
    └────┬────┴──────────┘
         ▼
  Technical Screeners
  (Breakout · MACD · Moving Average · Trendline · Donchian)
         │
         ▼
  Charts + CSV Reports + Email Alerts
```

---

## The Three Trading Categories

### 1. 📈 Trend / Momentum Trading

**Idea:** Find stocks already moving upward with high energy; ride the wave.

| Filter | Threshold |
|--------|-----------|
| Daily GROWTH rate | > 0.1 % |
| VOLATILITY | > 10 % |

**When it works well:**
- Strong bull markets with clear sector rotations.
- Stocks showing institutional accumulation and rising volume.

**When it fails:**
- In choppy, sideways markets — generates many false "momentum" signals that reverse quickly.
- Late-cycle trends can trigger buys right before a pullback.

---

### 2. 💎 Value / Mean-Reversion Trading

**Idea:** If a stock's current price is significantly **below** where the model predicts it should be, it may be undervalued.

| Filter | Condition |
|--------|-----------|
| RATE | "BELOW TREND" or "HIGHLY BELOW TREND" |

The model assigns a **z-score** to each stock:

| Rating | Z-Score | Signal |
|--------|---------|--------|
| HIGHLY BELOW TREND | > 3.0 | Strong buy |
| BELOW TREND | 2.0 – 3.0 | Moderate buy |
| ALONG TREND | −2.0 – 2.0 | Hold / neutral |
| ABOVE TREND | −3.0 – −2.0 | Moderate sell |
| HIGHLY ABOVE TREND | < −3.0 | Strong sell |

**When it works well:**
- Mature, fundamentally sound companies that dip temporarily (e.g. earnings overreaction).
- When the overall market is stable or recovering.

**When it fails:**
- A stock trending "below trend" may be doing so for good reason (deteriorating fundamentals, sector decline). The model has **no knowledge of news, earnings, or macro events** — it sees only past price.
- In a bear market, "below trend" can stay "below trend" for months.

---

### 3. 🔥 Breakout Trading

**Idea:** Stocks trading in a tight range with low volatility may be **compressing** before a directional move.

| Filter | Threshold |
|--------|-----------|
| RATE | "ALONG TREND" |
| VOLATILITY | < 15 % |
| Absolute daily GROWTH | < 0.1 % |

**When it works well:**
- Classic "cup and handle" or "volatility squeeze" setups.
- When general market confidence is building.

**When it fails:**
- Most breakout attempts actually fail. Without confirmation (e.g. volume surge), many consolidations simply continue sideways.
- The filter has no directional bias — a "breakout" could go **down** just as easily as up.

---

## Technical Screeners (Second Layer)

After the volatility model narrows the universe, additional screeners refine the list:

| Screener | What It Looks For | Buy Signal |
|----------|-------------------|------------|
| **Breakout** | Largest bullish candle in 10 days + volume spike > 50 % above 20-day average + low upper wick | Big, clean green candle on high volume |
| **Trendline** | Slope of recent price action via linear regression | Slope angle ≥ 30° (uptrend) |
| **Moving Average** | Combinations of SMA 10/30/50/200 crossovers and MACD position (5 sub-strategies) | At least one strategy matches |
| **MACD** | MACD line crosses above signal line | Bullish crossover in last 5 days |
| **Donchian** | Price touches 20-day channel lower band | Price at channel low (mean-reversion entry) |

---

## The Underlying Model

The core engine is a **Multi-Stock, Industry, Sector, Market — Conjugate, Sequential (MSIS-MCS)** Bayesian model:

1. **Hierarchical structure** — Learns shared patterns at four levels: Market → Sector → Industry → Stock.
2. **Polynomial regression** on log-prices — Captures the trend direction and curvature.
3. **Uncertainty quantification** — Each prediction comes with a standard deviation, enabling confidence-based ratings.
4. **Correlation analysis** — Identifies the most similar stock for every ticker, useful for pair trading and diversification.

Training uses the **Adam optimiser** on model log-likelihood, progressing from coarse (market-level) to fine (stock-level) granularity.

---

## ✅ Strengths

| Strength | Detail |
|----------|--------|
| **Probabilistic predictions** | Unlike simple indicator tools, the model quantifies *uncertainty* — you know how confident the prediction is. |
| **Hierarchical learning** | Information is shared across market, sector, and industry levels, making individual stock estimates more robust — especially for stocks with limited history. |
| **Multi-strategy approach** | Combining volatility analysis with five independent technical screeners reduces the chance of acting on a single noisy signal. |
| **Automatic categorisation** | Stocks are pre-sorted into Trend / Value / Breakout buckets, saving manual screening work. |
| **Correlation matching** | The "MATCH" column reveals which stock moves most similarly — useful for hedging or avoiding over-concentration. |
| **Multi-market support** | Works out-of-the-box for both US (S&P 500) and Indian (NSE 500) equities. |
| **Caching & incremental training** | Model parameters can be saved and reloaded, enabling daily incremental updates instead of full retraining. |

---

## ⚠️ Weaknesses & Risks

| Weakness | Detail |
|----------|--------|
| **Purely price-based** | The model knows nothing about earnings, revenue, news, macro-economic events, or company fundamentals. A stock can look like a "value buy" while the business is actually deteriorating. |
| **Backward-looking** | All predictions are based on historical patterns. Past trends do not guarantee future performance — regime changes (crashes, rate hikes, black swan events) can invalidate the model entirely. |
| **No stop-loss or risk management** | The system generates entry signals but **does not suggest exit points, position sizes, or stop-losses**. A user acting on these signals without their own risk framework could suffer significant losses. |
| **Overfitting risk** | The high-order polynomial (order 52 for correlation, order 2 for trend) can overfit to noise, especially in shorter time windows or less liquid stocks. |
| **Slow training** | The TensorFlow model requires substantial compute time (50,000+ Adam steps for the correlation pass). Not suitable for real-time intraday decisions. |
| **No backtesting framework** | There is no built-in system to evaluate how these signals have performed historically. You cannot easily measure win rate, drawdown, or Sharpe ratio. |
| **False breakouts** | The breakout screener has no directional filter — it flags consolidation patterns, but many "breakouts" fail or move in the wrong direction. |
| **Screener overlap** | Multiple screeners may flag the same stock, which could create false conviction. Conversely, screeners can contradict each other (e.g. MACD says SELL while trendline says BUY). |
| **Data dependency** | Relies on Yahoo Finance via `yfinance`, which can have delays, missing data, or API rate-limiting — potentially producing stale or incomplete analysis. |
| **No transaction costs** | The model does not account for spreads, commissions, slippage, or taxes, all of which erode real-world returns. |

---

## Quick Start

```bash
# Install dependencies
pip install --user poetry
poetry install
poetry shell

# Scan US market (S&P 500) — all screeners
python src/project_alpha.py --market us

# Scan Indian market (NSE 500) — only volatility + trend
python src/project_alpha.py --market india --screeners volatility,trend

# Fast mode — skip charts
python src/project_alpha.py --market us --no-plots --cache
```

### Key CLI Options

| Option | Description |
|--------|-------------|
| `--market us/india` | Choose market |
| `--screeners volatility,breakout,trend,ma,macd,donchain` | Pick specific screeners |
| `--rank rate/growth/volatility` | How to rank results |
| `--top N` | Limit to top N stocks per screener |
| `--min-price / --max-price` | Filter by price range |
| `--save-model FILE` | Save trained model for reuse |
| `--load-model FILE` | Resume from saved model |
| `--db-path PATH` | Use SQLite for persistent caching |
| `--no-plots` | Skip chart generation (faster) |

### Output

Results are saved to `data/processed_data/`:
- **Charts** (SVG) — Candlestick charts with indicators for each screener.
- **CSV reports** — Full prediction tables and screener hit-lists.
- **Email** — Optional email delivery with charts and PDF attachment.

---

## Bottom Line

Project Alpha is a **powerful screening and analysis toolkit** that combines cutting-edge probabilistic modelling with traditional technical analysis. It excels at narrowing a universe of 500 stocks down to a watchlist of 20–50 candidates worth investigating further.

However, **it is not a trading system** — it tells you *what to look at*, not *what to buy*. Every signal should be validated with fundamental research, risk management rules, and your own investment thesis before placing a trade.
