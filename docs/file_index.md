# Project Alpha - File Index

Updated after migration to modular packages.

---

## Active Files

### Entry Point
| File | Purpose |
|------|---------|
| `src/project_alpha.py` | Main CLI entry point |

### Core Classes (`src/classes/`)

| File | Purpose |
|------|---------|
| `Add_indicators.py` | Technical indicator calculations |
| `DatabaseManager.py` | SQLite database operations |
| `Download.py` | Data downloading and caching |
| `IndexListFetcher.py` | S&P 500 / NSE 500 stock lists |
| `Models.py` | TensorFlow models for predictions |
| `Plotting.py` | Plotting utilities (for Volatile) |
| `ScreenipyTA.py` | TradingView integration |
| `Tools.py` | Utility functions |
| `Volatile.py` | Volatility analysis with TFP |

### Modular Packages

#### `src/classes/analysis/`
| File | Purpose |
|------|---------|
| `CorrelationAnalyzer.py` | Stock correlation analysis |
| `TrendAnalyzer.py` | Trend scoring and analysis |
| `VolatileAnalyzer.py` | Refactored volatility analyzer |
| `VolatileConfig.py` | Configuration dataclass |

#### `src/classes/data/`
| File | Purpose |
|------|---------|
| `CacheManager.py` | Data caching with TTL |
| `DataTransformer.py` | Data transformation utilities |
| `DataValidator.py` | Validation and quality checks |
| `StockFetcher.py` | Stock data fetching |

#### `src/classes/output/`
| File | Purpose |
|------|---------|
| `charts.py` | ChartBuilder + create_batch_charts |
| `console.py` | Console output utilities |
| `email.py` | Email server management |
| `exporters.py` | CSV, JSON, HTML exporters |
| `formatters.py` | Result formatting classes |

#### `src/classes/screeners/`
| File | Purpose |
|------|---------|
| `base.py` | BaseScreener ABC + Signal/Result types |
| `breakout.py` | Breakout screener |
| `donchian.py` | Donchian channel screener |
| `macd.py` | MACD screener |
| `moving_average.py` | Moving average screener |
| `registry.py` | Screener registry |
| `trendline.py` | Trendline screener |

---

## Archived Files (`src/classes/_archive/`)

| File | Reason |
|------|--------|
| `Console.py` | Replaced by `output/console.py` |
| `Evaluation.py` | Unused |
| `Plot_stocks.py` | Replaced by `output/charts.py` |
| `Screener.py` | Replaced by modular screeners |
| `Screener_breakout.py` | Replaced by `screeners/breakout.py` |
| `Screener_donchain.py` | Replaced by `screeners/donchian.py` |
| `Screener_ma.py` | Replaced by `screeners/moving_average.py` |
| `Screener_macd.py` | Replaced by `screeners/macd.py` |
| `Screener_trendline.py` | Replaced by `screeners/trendline.py` |
| `Screener_value.py` | Unused, contains TODOs |
| `Send_email.py` | Replaced by `output/email.py` |

---

## Test Files (`tests/`)

| File | Tests |
|------|-------|
| `test_analysis_layer.py` | 23 |
| `test_data_layer.py` | 39 |
| `test_output.py` | 26 |
| `test_screeners.py` | 25 |

**Total: 113 tests**

---

*Last updated: 2026-02-07*
