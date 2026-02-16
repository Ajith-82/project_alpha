# Addressing Strategy Weaknesses — Mitigations & Recommended Tools

This document maps each weakness identified in the [Trading Strategy README](file:///opt/developments/project_alpha/docs/trading_strategy_readme.md) to concrete **mitigation actions** and **open-source GitHub utilities** that can be integrated into Project Alpha.

---

## Summary Table

| # | Weakness | Recommended Tool(s) | Effort |
|---|----------|---------------------|--------|
| 1 | Purely price-based | OpenBB, FundamentalAnalysis, FinBERT | Medium |
| 2 | Backward-looking | Regime detection, walk-forward validation | Medium |
| 3 | No stop-loss / risk management | skfolio, pyfolio, PSCalc | Medium |
| 4 | Overfitting risk | Backtesting.py, VectorBT (walk-forward) | Medium |
| 5 | Slow training | TF SavedModel, ONNX export, batch scheduling | Low |
| 6 | No backtesting framework | Backtesting.py, VectorBT, Backtrader | High |
| 7 | False breakouts | TTM Squeeze, ADX/OBV confirmation | Low |
| 8 | Screener overlap / contradiction | Consensus scoring engine | Medium |
| 9 | Data dependency (yfinance) | Polygon.io, Alpha Vantage, Finnhub | Medium |
| 10 | No transaction costs | Zipline cost models, custom slippage | Low |

---

## 1. Purely Price-Based

**Problem:** The model ignores earnings, revenue, macro events, and company fundamentals.

### Mitigation

Add a **fundamental + sentiment filter** as a pre- or post-processing step so that signals are only acted on when fundamentals support the direction.

### Recommended Tools

| Tool | GitHub | What It Adds |
|------|--------|--------------|
| **OpenBB** | [OpenBB-finance/OpenBBTerminal](https://github.com/OpenBB-finance/OpenBBTerminal) | Full financial research platform — fundamentals, macro, news, estimates. "The Bloomberg Terminal you can fork." |
| **FundamentalAnalysis** | [JerBouma/FundamentalAnalysis](https://github.com/JerBouma/FundamentalAnalysis) | Financial ratios, key metrics, growth rates, DCF via Financial Modeling Prep API. |
| **FinBERT** | [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) | Pre-trained NLP model for financial sentiment analysis on news headlines and filings. |
| **Stocksent** | [Aryagm/Stocksent](https://github.com/Aryagm/Stocksent) | Sentiment analysis from trusted news sources, per ticker. |
| **Finnhub API** | [Finnhub-Stock-API/finnhub-python](https://github.com/Finnhub-Stock-API/finnhub-python) | Real-time fundamentals, analyst estimates, earnings surprises, and news sentiment in one API. |

### Suggested Integration

```python
# Post-filter: suppress BUY signals for stocks with negative earnings growth
from fundamentalanalysis import financial_ratios
ratios = financial_ratios("AAPL", api_key, period="annual")
if ratios.loc["netIncomeGrowth"].iloc[0] < 0:
    signal = Signal.HOLD  # Override model's BUY
```

---

## 2. Backward-Looking (Regime Changes)

**Problem:** Past trends don't predict crashes, rate hikes, or black-swan events.

### Mitigation

- **Regime detection** — Add a market-regime classifier (bull/bear/sideways) that suppresses aggressive signals during detected bear regimes.
- **Walk-forward validation** — Retrain and test on rolling windows to measure how well the model adapts to changing conditions.

### Recommended Tools

| Tool | GitHub | What It Adds |
|------|--------|--------------|
| **hmmlearn** | [hmmlearn/hmmlearn](https://github.com/hmmlearn/hmmlearn) | Hidden Markov Models to detect market regimes (bull/bear/sideways). |
| **VectorBT** | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | Built-in walk-forward optimisation, rolling window analysis, and regime-aware parameter tuning. |

### Suggested Integration

Use a 2-state HMM on VIX or market returns. When the model detects a "high-volatility" regime, automatically tighten the screener thresholds or reduce position sizes.

---

## 3. No Stop-Loss or Risk Management

**Problem:** The system says *what to buy* but not *when to exit*, *how much to risk*, or *how to size positions*.

### Mitigation

Add a **risk management layer** that wraps every screener signal with:
- **Stop-loss** (ATR-based or fixed percentage)
- **Position sizing** (Kelly criterion or fixed-risk-per-trade)
- **Portfolio-level risk** (max drawdown, VaR limits)

### Recommended Tools

| Tool | GitHub | What It Adds |
|------|--------|--------------|
| **skfolio** | [skfolio/skfolio](https://github.com/skfolio/skfolio) | Scikit-learn-style portfolio optimisation — Variance, CVaR, Max Drawdown, VaR constraints. |
| **pyfolio** | [quantopian/pyfolio](https://github.com/quantopian/pyfolio) | Performance and risk analysis — drawdown, rolling Sharpe, exposure analysis. |
| **Riskfolio-Lib** | [dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | Portfolio optimisation with 20+ risk measures and risk-parity constraints. |
| **PSCalc** | [mfat/PSCalc](https://github.com/mfat/PSCalc) | Position size calculator based on account balance, stop-loss, and risk percentage. |

### Suggested Integration

```python
# ATR-based stop-loss: 2 × ATR(14) below entry
import pandas_ta as ta
atr = ta.atr(high, low, close, length=14)
stop_loss = entry_price - 2 * atr.iloc[-1]

# Position sizing: risk 1% of portfolio per trade
risk_per_trade = portfolio_value * 0.01
shares = int(risk_per_trade / (entry_price - stop_loss))
```

---

## 4. Overfitting Risk

**Problem:** High-order polynomials (order 52) can memorise noise.

### Mitigation

- **Walk-forward cross-validation** — Train on window W₁, validate on W₂, slide forward, repeat.
- **Regularisation** — Add L2 penalty to the model's loss function.
- **Out-of-sample testing** — Always reserve the most recent N months as a hold-out test.

### Recommended Tools

| Tool | GitHub | What It Adds |
|------|--------|--------------|
| **Backtesting.py** | [kernc/backtesting.py](https://github.com/kernc/backtesting.py) | Simple walk-forward splits, built-in Sharpe/drawdown metrics. |
| **VectorBT** | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | High-performance parameter space sweep with automatic overfitting detection. |

### Suggested Implementation

The correlation model trains at `order=52`. Consider:
1. Reducing `order_correlation` from 52 to 20–30 and comparing predictive accuracy.
2. Adding a validation loss metric to `train_msis_mcs()` using a held-out time slice.

---

## 5. Slow Training

**Problem:** 50,000+ Adam steps for the correlation pass makes daily runs slow.

### Mitigation

- **Save/reload model** — Already supported via `--save-model` / `--load-model`. Use this for **incremental training** on new data each day.
- **Reduce training steps** — Use saved parameters as warm starts; 5,000–10,000 steps for fine-tuning instead of full convergence.
- **GPU acceleration** — Ensure TensorFlow uses GPU if available (check `tf.config.list_physical_devices('GPU')`).
- **Scheduled runs** — Run training overnight via cron; use cached results during market hours.

### Suggested Change

```python
# In VolatileConfig: reduce default steps when using warm-start
if self.load_model:
    self.training.correlation_steps = 10000  # vs 50000 for cold start
    self.training.trend_steps = 2000         # vs 10000 for cold start
```

---

## 6. No Backtesting Framework

**Problem:** No way to measure historical win rate, drawdown, or Sharpe ratio of the strategy.

### Mitigation

Integrate a backtesting engine that replays historical data through the screeners and measures P&L.

### Recommended Tools

| Tool | GitHub | What It Adds |
|------|--------|--------------|
| **Backtesting.py** | [kernc/backtesting.py](https://github.com/kernc/backtesting.py) | Lightweight, Pythonic. Great for testing individual screener signals. Interactive HTML reports. |
| **VectorBT** | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | Vectorised backtesting on millions of parameter combinations. Advanced analytics. |
| **Backtrader** | [mementum/backtrader](https://github.com/mementum/backtrader) | Full-featured event-driven backtester. Slippage, commission, multi-data support. |
| **Zipline** | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | Maintained Zipline fork. Integrated slippage/commission models. Pandas-native. |
| **QuantConnect (Lean)** | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | Professional-grade, multi-asset backtesting with fee/slippage models. |

### Suggested Architecture

```
project_alpha screener signals
        │
        ▼
  Backtesting.py Strategy class
        │  ← Entry: when screener says BUY
        │  ← Exit:  ATR trailing stop or screener says SELL
        │  ← Sizing: fixed risk per trade
        ▼
  Performance report (Sharpe, drawdown, win rate)
```

> [!IMPORTANT]
> This is the **single highest-impact improvement** you can make. Without backtesting, you cannot know if the strategy actually makes money.

---

## 7. False Breakouts

**Problem:** The breakout screener flags consolidation patterns but many fail or move the wrong direction.

### Mitigation

Add **confirmation filters** before emitting a breakout BUY signal:

| Confirmation Method | Implementation |
|---------------------|----------------|
| **Volume Profile** | Require breakout candle to trade above the Volume Point of Control (VPOC). |
| **ADX (trend strength)** | Only flag breakouts when ADX > 20, confirming directional momentum. |
| **ATR expansion** | Require current ATR > 1.5× recent 20-day ATR average (volatility expanding). |
| **TTM Squeeze** | Use Bollinger Bands inside Keltner Channels to confirm volatility compression before breakout. |
| **Directional filter** | Combine with the trendline screener: only accept breakouts *in the direction of the prevailing trend*. |

### Code Sketch

```python
# In BreakoutScreener.screen(), add confirmation:
atr_current = ta.atr(df["High"], df["Low"], df["Close"], length=14).iloc[-1]
atr_avg = ta.atr(df["High"], df["Low"], df["Close"], length=14).rolling(20).mean().iloc[-1]

adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"].iloc[-1]

if atr_current < 1.5 * atr_avg or adx < 20:
    return ScreenerResult(ticker=ticker, signal=Signal.HOLD, ...)
```

---

## 8. Screener Overlap / Contradiction

**Problem:** Multiple screeners may flag the same stock (false conviction) or contradict each other.

### Mitigation

Build a **consensus scoring engine**:

```python
def compute_consensus(screener_results: Dict[str, ScreenerResult]) -> Signal:
    """
    Weight and combine signals from multiple screeners.
    
    Weight scheme:
      volatility (model-based):  0.35
      breakout:                  0.20
      trendline:                 0.20
      macd:                      0.15
      donchian:                  0.10
    """
    weights = {"volatility": 0.35, "breakout": 0.20, "trendline": 0.20, "macd": 0.15, "donchian": 0.10}
    
    score = 0.0
    for name, result in screener_results.items():
        multiplier = {Signal.STRONG_BUY: 1.0, Signal.BUY: 0.5, Signal.HOLD: 0.0, Signal.SELL: -0.5, Signal.STRONG_SELL: -1.0}
        score += weights.get(name, 0.1) * multiplier[result.signal] * result.confidence
    
    if score > 0.3:   return Signal.BUY
    elif score < -0.3: return Signal.SELL
    else:              return Signal.HOLD
```

This ensures no single screener dominates, and contradictions cancel out.

---

## 9. Data Dependency (yfinance)

**Problem:** `yfinance` is unofficial, can break, rate-limit, or return stale data.

### Mitigation

Add a **fallback data provider chain** — if the primary fails, try the next.

### Recommended Providers

| Provider | GitHub / Link | Free Tier | Coverage |
|----------|---------------|-----------|----------|
| **Polygon.io** | [polygon-io/client-python](https://github.com/polygon-io/client-python) | EOD data, 5 calls/min | US equities, options, forex, crypto |
| **Alpha Vantage** | [RomelTorres/alpha_vantage](https://github.com/RomelTorres/alpha_vantage) | 500 calls/day, 5/min | Global equities, forex, crypto, indicators |
| **Finnhub** | [Finnhub-Stock-API/finnhub-python](https://github.com/Finnhub-Stock-API/finnhub-python) | 60 calls/min | US + international, fundamentals, sentiment |
| **EOD Historical Data** | [EodHistoricalData/EODHD-APIs-Python-Financial-Library](https://github.com/EodHistoricalData/EODHD-APIs-Python-Financial-Library) | Limited free tier | 60+ exchanges worldwide |
| **OpenBB** | [OpenBB-finance/OpenBBTerminal](https://github.com/OpenBB-finance/OpenBBTerminal) | Aggregated multi-source | All of the above in one interface |

### Suggested Architecture

```python
class DataProviderChain:
    """Try providers in order; fallback on failure."""
    providers = ["yfinance", "polygon", "alpha_vantage"]
    
    def fetch(self, symbol, start, end):
        for provider in self.providers:
            try:
                return self._fetch_from(provider, symbol, start, end)
            except Exception:
                continue
        raise DataUnavailableError(f"All providers failed for {symbol}")
```

---

## 10. No Transaction Costs

**Problem:** Model ignores spreads, commissions, slippage, and taxes.

### Mitigation

Apply a **cost deduction** to every simulated trade when backtesting. Even before a full backtester is built, you can add a "realistic return" filter.

### Cost Model

```python
@dataclass
class TransactionCosts:
    commission_per_trade: float = 1.00    # USD flat or per-share
    slippage_bps: float = 5.0             # basis points (0.05%)
    spread_bps: float = 3.0               # typical mid-to-ask spread
    
    def total_cost_pct(self) -> float:
        """Total round-trip cost as a percentage."""
        return 2 * (self.slippage_bps + self.spread_bps) / 10000  # 0.16%
    
    def net_return(self, gross_return: float) -> float:
        return gross_return - self.total_cost_pct()
```

### Integration with Backtesting

Zipline and Backtrader both have built-in commission/slippage models. When building the backtesting layer (weakness #6), configure these directly:

```python
# Backtrader example
cerebro.broker.setcommission(commission=0.001)  # 0.1%
cerebro.broker.set_slippage_perc(perc=0.0005)   # 0.05%
```

---

## Priority Roadmap

Based on impact and effort, here is a suggested implementation order:

| Priority | Weakness | Action |
|----------|----------|--------|
| 🔴 **P0** | #6 No backtesting | Integrate **Backtesting.py** — highest impact, proves whether the strategy works. |
| 🔴 **P0** | #3 No risk management | Add ATR stop-loss + position sizing to the output. |
| 🟠 **P1** | #1 Purely price-based | Add **FundamentalAnalysis** or **Finnhub** as a post-filter on signals. |
| 🟠 **P1** | #7 False breakouts | Add ADX + ATR confirmation to the breakout screener. |
| 🟡 **P2** | #8 Screener overlap | Build consensus scoring engine. |
| 🟡 **P2** | #9 Data dependency | Add **Polygon.io** as a fallback provider. |
| 🟢 **P3** | #10 Transaction costs | Add cost model to backtesting engine. |
| 🟢 **P3** | #4 Overfitting | Add walk-forward validation; consider reducing polynomial order. |
| 🟢 **P3** | #2 Backward-looking | Add HMM regime detection on VIX. |
| 🟢 **P3** | #5 Slow training | Use warm-start defaults when `--load-model` is provided. |

---

> [!CAUTION]
> None of these mitigations make the system a "money machine." Even with backtesting, risk management, and fundamentals, **all trading involves risk.** These improvements raise the quality of the tool but do not guarantee profitable outcomes.
