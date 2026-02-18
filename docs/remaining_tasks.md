# Remaining Tasks (Phase 3: Infrastructure & Production)

These tasks are extracted from `docs/implementation_roadmap.md` and represent the outstanding work required to complete the project.

## §3.1 Data Provider Chain
- [ ] **3.1.1** Create abstract `DataProvider` base class
- [ ] **3.1.2** Implement `YFinanceProvider` — wrap existing `Download.py` logic
- [ ] **3.1.3** Implement `PolygonProvider`
- [ ] **3.1.4** Implement `AlphaVantageProvider`
- [ ] **3.1.5** Implement column normalisation — all providers return `Open, High, Low, Close, Volume` DatetimeIndex
- [ ] **3.1.6** Integrate `validate_ohlcv()` from §2.1 into the provider chain
- [ ] **3.1.7** Add rate limiting per provider (Polygon: 5/min, Alpha Vantage: 25/day)
- [ ] **3.1.8** Replace direct yfinance calls in `Download.py` with `DataProviderChain`
- [ ] **3.1.9** Add `--data-provider` and API key CLI options (keys from Settings)
- [ ] **3.1.10** Add provider health check on startup
- [ ] **3.1.11** Write integration tests with mocked provider responses

## §3.2 Market Regime Detection
- [ ] **3.2.1** Create `RegimeDetector` with 3-state `GaussianHMM`
- [ ] **3.2.2** Implement feature engineering: log returns + 20-day rolling volatility
- [ ] **3.2.3** Classify states by mean return: Bull (highest) / Bear (lowest) / Neutral
- [ ] **3.2.4** Implement regime-based signal adjustment
- [ ] **3.2.5** Add `--regime-detection` and `--regime-index SPY` options
- [ ] **3.2.6** Generate regime overlay on charts (colour periods by regime)
- [ ] **3.2.7** Write tests with synthetic bull/bear/sideways data
- [ ] **3.2.8** Validate on historical S&P 500 (should detect 2020 crash, 2022 bear)

## §3.3 Walk-Forward Validation
- [ ] **3.3.1** Create anchored expanding-window walk-forward validator
- [ ] **3.3.2** Implement window generation (training starts fixed, end expands, test slides forward)
- [ ] **3.3.3** Reuse `engine.py` from Phase 1 for per-window backtest
- [ ] **3.3.4** Aggregate out-of-sample metrics across all windows
- [ ] **3.3.5** Implement overfitting detection: OOS/IS Sharpe ratio < 0.5 = overfit warning
- [ ] **3.3.6** Add `--walk-forward`, `--wf-train-months`, `--wf-test-months` options
- [ ] **3.3.7** Generate walk-forward report: per-window performance + aggregate metrics
- [ ] **3.3.8** Write tests verifying window arithmetic (no overlaps, no gaps)

## §3.4 Volatility Model Improvements
- [ ] **3.4.1** Warm-start default: reduce steps when `--load-model` is used (50000 → 10000)
- [ ] **3.4.2** Add `--polynomial-order` CLI option to experiment with lower correlation orders
- [ ] **3.4.3** Add validation loss tracking: 80/20 time-series split, log hold-out loss
- [ ] **3.4.4** Log training time and convergence metrics to structured log
- [ ] **3.4.5** Add GPU detection and auto-placement with `tf.config.list_physical_devices('GPU')`

## §3.5 Monitoring & Observability
- [ ] **3.5.1** Create `src/monitoring/metrics.py` with Prometheus metrics
- [ ] **3.5.2** Instrument screeners: increment `SIGNALS_TOTAL` on each signal
- [ ] **3.5.3** Instrument data fetching: observe `DOWNLOAD_DURATION` per request
- [ ] **3.5.4** Instrument model training: set `TRAINING_LOSS` per training step
- [ ] **3.5.5** Instrument caching: track hit/miss ratio
- [ ] **3.5.6** Create Prometheus scrape config (`prometheus.yml`)
- [ ] **3.5.7** Create Grafana dashboard: Screener Signals
- [ ] **3.5.8** Create Grafana dashboard: Data Pipeline
- [ ] **3.5.9** Create alerting rules
- [ ] **3.5.10** Expose `/metrics` endpoint in FastAPI (§3.6)
- [ ] **3.5.11** Write tests verifying metrics are incremented correctly

## §3.6 API Layer (FastAPI)
- [ ] **3.6.1** Create `main.py` — FastAPI app with CORS, lifespan events, and `/metrics` endpoint
- [ ] **3.6.2** Create `schemas/requests.py` and `responses.py` with Pydantic models
- [ ] **3.6.3** Implement endpoints (`/health`, `/screeners`, `/predictions`, `/backtest`, `/symbols`)
- [ ] **3.6.4** Create `dependencies.py` — DI for Settings, database, data providers
- [ ] **3.6.5** Add request/response logging middleware using structlog (§1.3)
- [ ] **3.6.6** Add `/metrics` Prometheus endpoint (§3.5)
- [ ] **3.6.7** Add `uvicorn` startup command
- [ ] **3.6.8** Write API integration tests with `TestClient`
- [ ] **3.6.9** Generate OpenAPI spec (`/docs`) and export `openapi.yaml`

## §3.7 Security
- [ ] **3.7.1** Move email credentials from JSON to `.env` / `Settings` (§1.2)
- [ ] **3.7.2** Create `auth.py` with API key authentication for FastAPI
- [ ] **3.7.3** Add optional JWT authentication for multi-user scenarios
- [ ] **3.7.4** Create `rate_limiting.py` — rate limit API endpoints
- [ ] **3.7.5** Add input validation on all API endpoints
- [ ] **3.7.6** Add audit logging for sensitive operations
- [ ] **3.7.7** Write security tests

## §3.8 Scalability Foundation
- [ ] **3.8.1** Refactor data fetching to use `asyncio` + `aiohttp`
- [ ] **3.8.2** Refactor screener batch execution to use `concurrent.futures.ProcessPoolExecutor`
- [ ] **3.8.3** Add optional Celery task queue for long-running operations
- [ ] **3.8.4** Add `--async` flag for async data fetching
- [ ] **3.8.5** Add `docker-compose.yml` with Redis for Celery broker
- [ ] **3.8.6** Write tests verifying async download produces same results as sync

## §3.9 Documentation
- [ ] **3.9.1** Set up MkDocs with `mkdocs-material` theme
- [ ] **3.9.2** Write `getting-started/` guides
- [ ] **3.9.3** Write `user-guide/`
- [ ] **3.9.4** Write `developer-guide/`
- [ ] **3.9.5** Export and include OpenAPI spec from FastAPI
- [ ] **3.9.6** Write `deployment/`
- [ ] **3.9.7** Add `mkdocs.yml`

## Phase 3 Test Plan
- [ ] **3.T.1** Data provider chain fallback
- [ ] **3.T.2** Regime detector classification
- [ ] **3.T.3** Walk-forward integrity
- [ ] **3.T.4** API endpoints
- [ ] **3.T.5** Security mechanisms
- [ ] **3.T.6** Prometheus metrics
- [ ] **3.T.7** Async results
- [ ] **3.T.8** MkDocs build
