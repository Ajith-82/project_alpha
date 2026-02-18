# 🚀 Project Alpha

**Your day-to-day trading companion** — a comprehensive stock market analysis and screening toolkit for US and Indian equities.

Project Alpha fetches historical price data, trains Bayesian volatility models, runs a suite of pluggable technical screeners, and validates strategies through backtesting — all from a single CLI.

---

## ✨ Features

| Category | Capabilities |
|----------|-------------|
| **Volatility Analysis** | Hierarchical Bayesian models (TensorFlow Probability) for trend estimation, growth scoring, and stock clustering |
| **Technical Screeners** | Breakout, Trendline, Moving Average, MACD, Donchian — pluggable via `BaseScreener` ABC & Registry |
| **Consensus Scoring** | Weighted multi-signal aggregation with synergy bonuses across screeners and filters |
| **AI-Powered Filters** | Fundamental health checks (Finnhub API) + NLP sentiment analysis (FinBERT) |
| **Regime Detection** | Hidden Markov Model classifying Bull / Bear / Sideways market states |
| **Backtesting** | Strategy validation with ATR-based risk management, position sizing, and transaction cost modeling |
| **Walk-Forward Validation** | Anchored expanding windows with overfitting detection (Sharpe degradation) |
| **Multi-Market** | US (S&P 500, NASDAQ, Dow Jones) and India (NSE 500/50/100) |
| **Multi-Provider** | Data from yfinance or Polygon.io with SQLite / pickle caching |
| **Rich Output** | Interactive terminal tables, SVG/candlestick charts, CSV/JSON exports, email reports with PDF attachments |

---

## 📦 Installation

### Prerequisites
- Python ≥3.12, <3.14
- [Poetry](https://python-poetry.org/) 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/project-alpha.git
cd project-alpha

# Install dependencies
pip install --user poetry
poetry install

# Activate the virtual environment
poetry shell

# (Optional) Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (Finnhub, Polygon, email settings)
```

---

## 🚀 Quick Start

```bash
# Analyze US market with all screeners (default)
python src/project_alpha.py

# Analyze Indian market
python src/project_alpha.py --market india

# Run specific screeners only
python src/project_alpha.py --screeners breakout,trend

# Top 20 stocks, JSON output
python src/project_alpha.py --top 20 --format json

# Filter by price range
python src/project_alpha.py --min-price 10 --max-price 500

# Analyze specific symbols
python src/project_alpha.py -s AAPL -s MSFT -s GOOGL
```

### Advanced Features

```bash
# Enable AI filters (requires API keys / model download)
python src/project_alpha.py --fundamental --sentiment --consensus

# Market regime detection
python src/project_alpha.py --regime-detection --regime-index SPY

# Backtest a strategy
python src/project_alpha.py --backtest --screeners breakout --initial-capital 50000

# Walk-forward validation (overfitting detection)
python src/project_alpha.py --walk-forward --wf-train-months 12 --wf-test-months 3 -s AAPL

# Customize risk parameters
python src/project_alpha.py --risk-per-trade 0.02 --atr-multiplier 2.5 --max-positions 5

# Verbose debug output
python src/project_alpha.py -v --log-level DEBUG

# Quiet mode with JSON logs (for pipelines)
python src/project_alpha.py -q --json-logs
```

Run `python src/project_alpha.py --help` for the full CLI reference.

---

## 📁 Project Structure

```
project_alpha/
├── src/
│   ├── project_alpha.py           # CLI entry point (rich-click)
│   ├── config/                    # Pydantic Settings + YAML defaults
│   └── classes/
│       ├── analysis/              # VolatileAnalyzer, TrendAnalyzer, RegimeDetector
│       ├── screeners/             # BaseScreener ABC, Registry, ConsensusEngine
│       ├── filters/               # FundamentalFilter, SentimentFilter
│       ├── risk/                  # RiskManager, TransactionCosts
│       ├── backtesting/           # BacktestEngine, WalkForwardValidator
│       ├── output/                # Charts, Formatters, Email, Console
│       ├── data/                  # NewsFetcher
│       ├── Download.py            # Multi-threaded data download
│       ├── DatabaseManager.py     # SQLite persistence
│       └── IndexListFetcher.py    # Market index symbol resolution
├── tests/                         # Unit (15 modules) + Integration (2 modules)
├── scripts/                       # Migration & automation scripts
├── docs/                          # Architecture docs, trading strategy guide, roadmap
├── pyproject.toml                 # Poetry dependencies
├── Dockerfile                     # Container build
└── docker-compose.yml             # Service orchestration
```

See [docs/architecture_documentation.md](docs/architecture_documentation.md) for detailed architecture with diagrams.

---

## 🗄️ Data Storage

### Pickle Cache (default)
Data is cached to `data/historic_data/{market}/` as pickle files, keyed by index name and date.

### SQLite Database (optional)
Persist price data across sessions with `--db-path`:

```bash
python src/project_alpha.py --db-path data/prices.db
```

Migrate existing pickle caches:
```bash
python scripts/migrate_pickle_to_db.py
```

### Output
- **Charts:** `data/processed_data/{screener_name}/*.svg`
- **CSV Reports:** `data/processed_data/screener_{name}/*.csv`
- **Backtest Reports:** Interactive HTML files
- **Logs:** `logs/project_alpha_{market}.log`

---

## ⚙️ Configuration

All settings can be configured via environment variables with the `PA_` prefix:

```bash
export PA_MARKET=us
export PA_FINNHUB_API_KEY=your_key
export PA_POLYGON_API_KEY=your_key
export PA_RISK_PER_TRADE=0.02
export PA_DATA_PROVIDER=polygon
```

Or place them in a `.env` file (see `.env.example`).

Settings are managed through [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) with the following precedence:

**CLI flags → Environment variables → `.env` file → Code defaults**

---

## 🐳 Docker

```bash
# Build and run (displays help)
docker compose up --build

# Run a specific scan
docker compose run --rm app --market us --top 10

# With environment file
docker compose --env-file .env run --rm app --market india --screeners breakout
```

---

## ⏰ Scheduled Scans

Use the included shell scripts for cron-based automation:

```bash
# US market scan (weekdays at 4:30 PM ET)
30 16 * * 1-5 /path/to/run_us_stock_scanner.sh

# India market scan (weekdays at 3:45 PM IST)
45 15 * * 1-5 /path/to/run_india_stock_scanner.sh
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture_documentation.md) | Full system architecture with Mermaid diagrams |
| [Trading Strategies](docs/trading_strategy_readme.md) | Screener strategies and signal logic |
| [Implementation Roadmap](docs/implementation_roadmap.md) | Development phases and progress |
| [Deployment Guide](docs/deployment_guide.md) | Production deployment instructions |
| [Email Setup](docs/email_setup.md) | SMTP configuration for report delivery |
| [API Guide](docs/api_guide.md) | External API integration details |

---

## ⚠️ Disclaimer

Project Alpha is provided for **educational and research purposes only**. It does not constitute financial advice. Always do your own due diligence before making investment decisions.
