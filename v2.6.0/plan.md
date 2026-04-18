# Si Volatility Analytics — Pre-Production Migration Plan

**Version:** 1.1
**Created:** 2026-04-18
**Source:** `v2.6.0/core/` (Google Colab notebooks)
**Target:** `server_code/` (standalone Python server)
**Environment:** Home server, 1–2 users, web UI, no GPU, minimal budget

---

## 1. EXECUTIVE SUMMARY

### What we're doing
Migrating 10 Colab notebook cells (`01`–`06`) from a `globals()`-based execution model
to a proper Python package with:
- Clean `import` graph (no `exec()`, no shared globals)
- FastAPI JSON API backend
- Lightweight web UI (existing drafts to be integrated later)
- Scheduled pipeline runs via cron/systemd timer
- SQLite for run history & caching MOEX responses

### What we're NOT doing (out of scope)
- HAR Backtest system (`07_*`) — migrated separately in Phase 2
- `predGL.json` / `additional_model.md` forecasting engine — Phase 3 (requires cloud GPU)
- Thursday analysis (`10_*`) — Phase 2
- Multi-user auth — not needed for 1–2 users on home LAN

### Key constraints
| Constraint | Decision |
|---|---|
| No GPU | All computation is NumPy/Pandas — no GPU needed |
| Minimal budget | SQLite (not Postgres), no cloud except MOEX API |
| 1–2 users | Single-process FastAPI + uvicorn, no load balancer |
| Home server | systemd service, Let's Encrypt optional (LAN-only OK) |

---

## 2. CURRENT STATE ANALYSIS

### 2.1 Source files inventory

| Colab Cell | File | Lines | Responsibility | Dependencies |
|---|---|---|---|---|
| Cell 1 | `01_imports_config_utils.py` | ~180 | Config, constants, precision helpers, logging | numpy, pandas, matplotlib, requests |
| Cell 2 | `02_moex_client_contracts.py` | ~100 | HTTP client, ticker logic, contract identification | Cell 1 |
| Cell 3 | `03_volatility_estimators.py` | ~350 | OHLCV loading, YZ/CC vol, gap stats, HAR/EWMA | Cell 1, Cell 2 |
| Cell 4 | `04_analytics_classification.py` | ~400 | Classifiers, Samuelson, gap regime, warnings | Cell 1, Cell 3 |
| Cell 5.1 | `05_1_vol_metrics.py` | ~120 | Vol metrics computation | Cell 1, Cell 3, Cell 4 |
| Cell 5.2 | `05_2_self_tests.py` | ~30 | Output integrity checks | Cell 1 |
| Cell 5.3 | `05_3_dashboard_viz.py` | ~50 | Matplotlib dashboard | Cell 1, matplotlib |
| Cell 5.4 | `05_4_rv_forecast_comp.py` | ~120 | RV forecast (HAR + EWMA) | Cell 1, Cell 3 |
| Cell 5.5 | `05_5_main_pipeline.py` | ~150 | Pipeline orchestrator | All above |
| Cell 6 | `06_data_summary_report.py` | ~180 | Post-run JSON validation report | standalone |
| Schema | `11_pydantic_schemas.py` | ~80 | Pydantic validation models | pydantic |

### 2.2 Critical problems in current code

| # | Problem | Impact | Fix in migration |
|---|---|---|---|
| 1 | **No imports** — all cells use `globals()` via `exec()` | Cannot run outside Colab | Proper `import` statements |
| 2 | **No `__init__.py`** — not a package | Cannot `pip install` or test | Package structure |
| 3 | **Hardcoded paths** (`/content/si_vol_analytics`) | Fails outside Colab | Config-driven paths |
| 4 | **No error recovery** — pipeline is all-or-nothing | Single MOEX timeout kills run | Retry + partial results |
| 5 | **No caching** — every run re-downloads all OHLCV | Slow, hammers MOEX API | SQLite cache layer |
| 6 | **Matplotlib `Agg` hardcoded** in imports | Blocks interactive use | Move to viz module only |
| 7 | **`print()` at module level** | Side effects on import | Remove, use `__main__` guards |
| 8 | **No type checking** — `Optional` used but never enforced | Runtime `None` errors | mypy strict mode |
| 9 | **Mixed concerns in Cell 3** — loading + estimation + forecasting | 350-line monolith | Split into 3 modules |
| 10 | **No tests** — `05_2_self_tests.py` is runtime-only | No CI possible | pytest test suite |
| 11 | **Pydantic schema incomplete** — 9 JSON sections missing from model | Validation passes invalid data | Expand schema to match actual output |

### 2.3 Pydantic schema gaps

The current `11_pydantic_schemas.py` defines `SiVolatilityReport` with only 8 fields.
The actual JSON output (`example_report.json`) has 17 top-level sections.

**Missing from schema:**
- `samuelson_daily_short` — DTE 1-10 vol norms
- `volume_ratio_historical` — F2/F1 volume vs history
- `regime_classification` — level/dynamics/composite
- `semivar_asymmetry` — F1/F2 downside bias
- `rate_of_change` — ROC metrics
- `percentile_ranks` — F1 YZ percentiles
- `distribution_classification` — skew/kurtosis labels
- `gap_regime_analysis` — emerging regime detection
- `volume_analysis` — avg volume, F2 reliability

### 2.4 Dependency graph (current)

```
Cell 01 (config)
  ├── Cell 02 (moex client)  ← uses MoexConfig, ContractMeta, log
  ├── Cell 03 (estimators)   ← uses Cell 01 + Cell 02 types
  ├── Cell 04 (analytics)    ← uses Cell 01 + Cell 03 functions
  └── Cell 05.1-5.5          ← uses everything above
       └── Cell 06 (summary) ← reads JSON output, standalone
Cell 11 (schemas)            ← standalone, validates JSON output
```

---

## 3. TARGET ARCHITECTURE

### 3.1 Package structure

```
server_code/
├── pyproject.toml                  # Package metadata, dependencies
├── README.md
├── si_vol/                         # Main package
│   ├── __init__.py                 # Version, public API
│   ├── config.py                   # ← from 01 (dataclasses, constants)
│   ├── exceptions.py               # ← from 01 (AnalyticsError, DataValidationError)
│   ├── helpers.py                   # ← from 01 (r_vol, null_val, vscalar, etc.)
│   ├── moex/
│   │   ├── __init__.py
│   │   ├── client.py               # ← from 02 (MoexClient)
│   │   ├── contracts.py            # ← from 02 (identify_contracts, ticker logic)
│   │   └── cache.py                # NEW: SQLite OHLCV cache
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # ← from 03 (moex_load_candles, load_all_candles)
│   │   └── validation.py           # ← from 03 (validate_ohlcv)
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── volatility.py           # ← from 03 (yang_zhang_series, cc_vol_series, etc.)
│   │   ├── gaps.py                 # ← from 03 (gap_statistics)
│   │   └── forecast.py             # ← from 03 (calc_har_rv_forecast, calc_session_aligned_rv_forecast)
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── classifiers.py          # ← from 04 (classify_skew, classify_kurtosis, etc.)
│   │   ├── samuelson.py            # ← from 04 (build_samuelson_daily_1_10, calc_samuelson_status)
│   │   ├── volume.py               # ← from 04 (avg_volume_nd, activity_weight, etc.)
│   │   ├── gap_regime.py           # ← from 04 (detect_emerging_gap_regime)
│   │   └── warnings.py             # ← from 04 (build_data_quality_warnings)
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── metrics.py              # ← from 05_1 (compute_vol_metrics, etc.)
│   │   ├── forecasts.py            # ← from 05_4 (compute_rv_forecasts)
│   │   ├── runner.py               # ← from 05_5 (run_pipeline)
│   │   ├── self_tests.py           # ← from 05_2 (run_self_tests)
│   │   └── summary.py              # ← from 06 (summarize_json_data)
│   ├── viz/
│   │   ├── __init__.py
│   │   └── dashboard.py            # ← from 05_3 (plot_analytics_dashboard)
│   └── schemas/
│       ├── __init__.py
│       └── models.py               # ← from 11_pydantic_schemas.py (EXPANDED)
├── api/
│   ├── __init__.py
│   ├── app.py                      # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # POST /api/run, GET /api/results/{id}
│   │   ├── reports.py              # GET /api/reports, GET /api/reports/{date}
│   │   └── health.py               # GET /api/health
│   └── deps.py                     # Dependency injection (config, db)
├── web/                            # Static files for UI (added later)
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Fixtures: mock MOEX responses, sample OHLCV
│   ├── test_config.py
│   ├── test_moex_client.py
│   ├── test_contracts.py
│   ├── test_volatility.py
│   ├── test_gaps.py
│   ├── test_classifiers.py
│   ├── test_samuelson.py
│   ├── test_schemas.py             # Validate example_report.json against expanded schema
│   ├── test_pipeline_integration.py
│   └── test_api.py
├── scripts/
│   ├── run_pipeline.py             # CLI entrypoint
│   └── migrate_cache.py            # One-time: import old JSON results
├── data/                           # Runtime data (gitignored)
│   ├── cache.sqlite                # MOEX OHLCV cache
│   ├── results/                    # Pipeline JSON outputs
│   └── charts/                     # Dashboard PNGs
└── deploy/
    ├── si-vol.service              # systemd unit file
    ├── si-vol-scheduler.timer      # systemd timer (daily run)
    └── .env.example                # Environment variables template
```

### 3.2 Module mapping (source → target)

| Source (Colab) | Target module | What changes |
|---|---|---|
| `01` config section | `si_vol/config.py` | Remove `print()`, add `__all__` |
| `01` exceptions | `si_vol/exceptions.py` | No logic change |
| `01` helpers (r_vol, null_val...) | `si_vol/helpers.py` | No logic change |
| `01` matplotlib setup | `si_vol/viz/__init__.py` | Lazy import, `Agg` only in viz |
| `02` MoexClient | `si_vol/moex/client.py` | Add `__enter__`/`__exit__` context manager |
| `02` ticker/contract logic | `si_vol/moex/contracts.py` | No logic change |
| `03` OHLCV loading | `si_vol/data/loader.py` | Add cache integration |
| `03` validate_ohlcv | `si_vol/data/validation.py` | No logic change |
| `03` yang_zhang, cc_vol, semivar | `si_vol/estimators/volatility.py` | No logic change |
| `03` gap_statistics | `si_vol/estimators/gaps.py` | No logic change |
| `03` HAR/EWMA forecast | `si_vol/estimators/forecast.py` | No logic change |
| `04` classifiers | `si_vol/analytics/classifiers.py` | No logic change |
| `04` Samuelson | `si_vol/analytics/samuelson.py` | No logic change |
| `04` volume helpers | `si_vol/analytics/volume.py` | No logic change |
| `04` gap regime | `si_vol/analytics/gap_regime.py` | No logic change |
| `04` warnings | `si_vol/analytics/warnings.py` | No logic change |
| `05_1` metrics | `si_vol/pipeline/metrics.py` | Add proper imports |
| `05_2` self-tests | `si_vol/pipeline/self_tests.py` | Also create pytest versions |
| `05_3` dashboard | `si_vol/viz/dashboard.py` | No logic change |
| `05_4` forecasts | `si_vol/pipeline/forecasts.py` | No logic change |
| `05_5` pipeline | `si_vol/pipeline/runner.py` | Replace `/content/` paths |
| `06` summary | `si_vol/pipeline/summary.py` | Remove `__main__` file search |
| `11` schemas | `si_vol/schemas/models.py` | Expand to cover all 17 JSON sections |

### 3.3 Technology stack

| Layer | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Match Colab runtime |
| **API** | FastAPI + uvicorn | Async, auto-docs, lightweight |
| **Database** | SQLite (via `sqlite3`) | Zero-config, sufficient for 1–2 users |
| **Cache** | SQLite table `ohlcv_cache` | Avoid re-downloading MOEX data |
| **Scheduler** | systemd timer | No extra dependency (vs celery/cron) |
| **Web UI** | Static files served by FastAPI | User's existing drafts |
| **Charts** | Matplotlib (Agg backend) | Keep existing code, serve PNGs via API |
| **Validation** | Pydantic v2 | Already in project (`11_pydantic_schemas.py`) |
| **Testing** | pytest + httpx (for API) | Standard, fast |
| **Process mgr** | systemd | Already on Linux server |
| **Reverse proxy** | Caddy (optional) | Auto-HTTPS, simpler than nginx |

### 3.4 Data flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  systemd     │────▶│  FastAPI      │────▶│  Pipeline    │
│  timer       │     │  /api/run     │     │  runner.py   │
│  (daily)     │     │              │     │              │
└─────────────┘     └──────────────┘     └──────┬───────┘
                           │                      │
                    ┌──────▼───────┐        ┌─────▼──────┐
                    │  Web UI      │        │ MOEX ISS   │
                    │  /reports    │        │ API        │
                    │  /dashboard  │        └─────┬──────┘
                    └──────────────┘              │
                                           ┌─────▼──────┐
                                           │ SQLite     │
                                           │ ohlcv_cache│
                                           └─────┬──────┘
                                                  │
                                           ┌─────▼──────┐
                                           │ JSON output│
                                           │ + PNG chart │
                                           └────────────┘
```

---

## 4. MIGRATION PHASES

### Phase 1: Core Package (Week 1–2)

**Goal:** All pipeline logic runs via `python -m si_vol.pipeline.runner` with identical JSON output.

| Step | Task | Validation |
|---|---|---|
| 1.1 | Create `pyproject.toml` with dependencies | `pip install -e .` succeeds |
| 1.2 | Extract `config.py`, `exceptions.py`, `helpers.py` from Cell 01 | `from si_vol.config import PipelineConfig` works |
| 1.3 | Extract `moex/client.py`, `moex/contracts.py` from Cell 02 | `from si_vol.moex.client import MoexClient` works |
| 1.4 | Split Cell 03 into `data/`, `estimators/` (3+2 files) | Unit tests pass for each estimator |
| 1.5 | Split Cell 04 into `analytics/` (5 files) | Unit tests pass for classifiers |
| 1.6 | Extract Cell 05.1–05.5 into `pipeline/` | `run_pipeline()` produces JSON |
| 1.7 | Extract Cell 06 into `pipeline/summary.py` | Summary report matches Colab output |
| 1.8 | Expand `si_vol/schemas/models.py` from Cell 11 | All 17 JSON sections validated |
| 1.9 | **Regression test:** compare JSON output vs `example_report.json` | Field-by-field match (float tolerance 1e-6) |

**Acceptance criteria:**
```bash
cd server_code
pip install -e .
python -m si_vol.pipeline.runner
# → produces si_vol_analytics_2.6.0.json identical to Colab output
```

### Phase 2: SQLite Cache + Config (Week 2–3)

**Goal:** MOEX responses cached, paths configurable, no hardcoded `/content/`.

| Step | Task | Validation |
|---|---|---|
| 2.1 | Create `moex/cache.py` — SQLite OHLCV cache | Second run skips MOEX API calls |
| 2.2 | Replace all `/content/` paths with `config.out_dir` | Works from any directory |
| 2.3 | Add `.env` support (`SI_VOL_DATA_DIR`, `SI_VOL_LOG_LEVEL`) | Env vars override defaults |
| 2.4 | Store run results in `data/results/{date}/` | Historical results preserved |

**Cache schema:**
```sql
CREATE TABLE ohlcv_cache (
    ticker    TEXT NOT NULL,
    date_from TEXT NOT NULL,
    date_till TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    data      BLOB NOT NULL,  -- gzip-compressed JSON
    PRIMARY KEY (ticker, date_from, date_till)
);

CREATE TABLE run_history (
    run_id     TEXT PRIMARY KEY,
    query_date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status     TEXT NOT NULL,  -- 'running', 'success', 'failed'
    json_path  TEXT,
    error      TEXT
);
```

### Phase 3: FastAPI Backend (Week 3–4)

**Goal:** HTTP API serves pipeline results and triggers runs.

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server status, last run info |
| `/api/run` | POST | Trigger pipeline run (async background task) |
| `/api/run/{run_id}` | GET | Run status and result |
| `/api/reports` | GET | List all historical reports |
| `/api/reports/{date}` | GET | Full JSON report for date |
| `/api/reports/{date}/summary` | GET | Summary report (Cell 06 output) |
| `/api/reports/{date}/dashboard` | GET | Dashboard PNG image |
| `/api/reports/{date}/analyze` | POST | Send report to LLM with `qwery_for_analyze.json` prompt |
| `/docs` | GET | Auto-generated OpenAPI docs |

**Key design decisions:**
- Pipeline runs in **background thread** (not async — NumPy/Pandas are CPU-bound)
- Only **one run at a time** (mutex lock)
- Results served from filesystem (JSON + PNG)
- No WebSocket needed — UI polls `/api/run/{id}` for status

### Phase 4: Pydantic Validation Integration (Week 4)

**Goal:** Pipeline output validated against `SiVolatilityReport` schema on every run.

The existing schema (`11_pydantic_schemas.py`) defines:
- `NullableValue` — wrapper for missing/stale metrics
- `ContractMeta`, `VolatilityMetrics`, `GapStats`, `HARForecast`, `DataWarning` — sub-models
- `SiVolatilityReport` — top-level report model
- `validate_pipeline_output()` — entry point for strict validation

| Step | Task |
|---|---|
| 4.1 | Move schema to `si_vol/schemas/models.py` (remove `print()` at bottom) |
| 4.2 | Call `validate_pipeline_output(output)` at end of `run_pipeline()` |
| 4.3 | Add missing fields to schema (currently incomplete vs actual JSON output) |
| 4.4 | API returns validation errors as structured response |

**Schema gaps to fix** (actual JSON has fields not in current Pydantic model):
- `samuelson_daily_short` — not in schema
- `volume_ratio_historical` — not in schema
- `regime_classification` — not in schema
- `semivar_asymmetry` — not in schema
- `rate_of_change` — not in schema
- `percentile_ranks` — not in schema
- `distribution_classification` — not in schema
- `gap_regime_analysis` — not in schema
- `volume_analysis` — not in schema

### Phase 5: Web UI Integration (Week 4–5)

**Goal:** Serve user's existing UI drafts via FastAPI static files.

| Step | Task |
|---|---|
| 5.1 | Mount `web/` as static files in FastAPI |
| 5.2 | Integrate user's UI drafts (added later to repo) |
| 5.3 | API proxy for CORS-free frontend calls |
| 5.4 | Dashboard PNG served as `<img>` or inline base64 |

### Phase 6: Deployment (Week 5)

| Step | Task |
|---|---|
| 6.1 | Create `deploy/si-vol.service` (systemd) |
| 6.2 | Create `deploy/si-vol-scheduler.timer` (daily 19:00 MSK) |
| 6.3 | Optional: Caddy reverse proxy for HTTPS |
| 6.4 | Backup script for SQLite + results |
| 6.5 | Monitoring: simple health check script |

**systemd unit (draft):**
```ini
[Unit]
Description=Si Volatility Analytics API
After=network.target

[Service]
Type=simple
User=si-vol
WorkingDirectory=/opt/si-vol
ExecStart=/opt/si-vol/venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
Environment=SI_VOL_DATA_DIR=/opt/si-vol/data
Environment=SI_VOL_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

**Timer (draft):**
```ini
[Unit]
Description=Daily Si Vol Pipeline Run

[Timer]
OnCalendar=Mon..Fri 19:00 Europe/Moscow
Persistent=true

[Install]
WantedBy=timers.target
```

---

## 5. TESTING STRATEGY

### 5.1 Unit tests (Phase 1)

| Module | Test focus | Mock strategy |
|---|---|---|
| `config.py` | Dataclass defaults, `makedirs()` | tmpdir fixture |
| `helpers.py` | `r_vol`, `null_val`, `vscalar`, `is_null` | Pure functions, no mocks |
| `moex/client.py` | Retry logic, error handling | `responses` library mock HTTP |
| `moex/contracts.py` | Ticker parsing, contract ordering | Mock `get_expiry` |
| `estimators/volatility.py` | YZ, CC, semivariance | Synthetic OHLCV DataFrames |
| `estimators/gaps.py` | Gap statistics, clustering | Synthetic price series |
| `estimators/forecast.py` | HAR coefficients, EWMA interpolation | Synthetic vol series |
| `analytics/classifiers.py` | All classify_* functions | Pure functions |
| `analytics/samuelson.py` | DTE bucketing, interpolation | Synthetic expired data |
| `analytics/gap_regime.py` | 3-level cascade logic | Synthetic gap_analysis dicts |
| `analytics/warnings.py` | Warning generation rules | Synthetic vol_data |
| `schemas/models.py` | Validate `example_report.json` | Load real JSON fixture |
| `pipeline/runner.py` | Full pipeline integration | Mock MoexClient |

### 5.2 Regression test (Phase 1, Step 1.9)

```python
def test_regression_vs_colab_output():
    """Compare server output to known-good Colab output."""
    expected = json.load(open("v2.6.0/core/example_report.json"))
    actual = run_pipeline(cfg)  # with mocked MOEX returning same data

    # Compare structure
    assert set(actual.keys()) == set(expected.keys())

    # Compare numeric values with tolerance
    for key in NUMERIC_FIELDS:
        assert abs(actual[key] - expected[key]) < 1e-6, f"{key} mismatch"
```

### 5.3 Schema test (Phase 4)

```python
def test_schema_validates_example_report():
    """Ensure expanded Pydantic schema accepts real pipeline output."""
    with open("v2.6.0/core/example_report.json") as f:
        data = json.load(f)
    report = SiVolatilityReport.model_validate(data)
    assert report.meta["schema_version"] == "2.6.0"
    assert "F1_current" in report.contracts
```

### 5.4 API tests (Phase 3)

```python
from httpx import AsyncClient

async def test_health(client: AsyncClient):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

async def test_run_pipeline(client: AsyncClient):
    r = await client.post("/api/run")
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    # Poll until complete
    ...
```

---

## 6. RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| MOEX API changes format | Low | High | Cache layer + validation; alert on parse errors |
| MOEX API downtime | Medium | Medium | Cache serves stale data with warning flag |
| Numeric drift (float precision) | Low | Medium | Regression test with tolerance |
| Home server power outage | Medium | Low | systemd `Persistent=true` runs on next boot |
| Python version mismatch | Low | Low | Pin Python 3.11+ in `pyproject.toml` |
| Large OHLCV data growth | Low | Low | SQLite VACUUM + 90-day cache TTL |

---

## 7. DEPENDENCIES

### Production
```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scipy>=1.10
requests>=2.28
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.0
python-dotenv>=1.0
```

### Development
```
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
responses>=0.25
mypy>=1.8
ruff>=0.3
```

---

## 8. MIGRATION CHECKLIST

### Pre-flight
- [ ] Verify `example_report.json` is current (run Colab pipeline, save output)
- [ ] Document MOEX API endpoints actually used (for mock fixtures)
- [ ] Collect sample MOEX API responses for test fixtures

### Phase 1 — Core Package
- [ ] `pyproject.toml` created, `pip install -e .` works
- [ ] `si_vol/schemas/models.py` — Pydantic models from `11_pydantic_schemas.py` (expanded)
- [ ] `si_vol/config.py` — all dataclasses, constants extracted
- [ ] `si_vol/exceptions.py` — AnalyticsError, DataValidationError
- [ ] `si_vol/helpers.py` — all precision/null helpers
- [ ] `si_vol/moex/client.py` — MoexClient with context manager
- [ ] `si_vol/moex/contracts.py` — identify_contracts, ticker logic
- [ ] `si_vol/data/loader.py` — moex_load_candles, load_all_candles
- [ ] `si_vol/data/validation.py` — validate_ohlcv
- [ ] `si_vol/estimators/volatility.py` — YZ, CC, semivar, percentile_rank
- [ ] `si_vol/estimators/gaps.py` — gap_statistics
- [ ] `si_vol/estimators/forecast.py` — HAR + EWMA
- [ ] `si_vol/analytics/classifiers.py` — all classify_* functions
- [ ] `si_vol/analytics/samuelson.py` — Samuelson table + status
- [ ] `si_vol/analytics/volume.py` — volume helpers
- [ ] `si_vol/analytics/gap_regime.py` — detect_emerging_gap_regime
- [ ] `si_vol/analytics/warnings.py` — build_data_quality_warnings
- [ ] `si_vol/pipeline/metrics.py` — compute_vol_metrics + helpers
- [ ] `si_vol/pipeline/forecasts.py` — compute_rv_forecasts
- [ ] `si_vol/pipeline/runner.py` — run_pipeline (main orchestrator)
- [ ] `si_vol/pipeline/self_tests.py` — run_self_tests
- [ ] `si_vol/pipeline/summary.py` — summarize_json_data
- [ ] `si_vol/viz/dashboard.py` — plot_analytics_dashboard
- [ ] All `print()` at module level removed
- [ ] All hardcoded `/content/` paths replaced
- [ ] **Regression test passes** vs `example_report.json`
- [ ] `pytest` — all unit tests green
- [ ] `mypy --strict` — no errors
- [ ] `ruff check` — no warnings

### Phase 2 — Cache + Config
- [ ] `si_vol/moex/cache.py` — SQLite OHLCV cache
- [ ] `.env` support for paths and log level
- [ ] Second pipeline run uses cache (no MOEX calls)
- [ ] Run history stored in SQLite

### Phase 3 — API
- [ ] FastAPI app with all endpoints
- [ ] Background pipeline execution
- [ ] API tests pass
- [ ] OpenAPI docs accessible at `/docs`

### Phase 4 — Pydantic Validation
- [ ] Schema expanded to cover all 17 JSON sections
- [ ] `validate_pipeline_output()` called at end of `run_pipeline()`
- [ ] `test_schema_validates_example_report` passes
- [ ] API returns validation errors as structured response

### Phase 5 — Web UI
- [ ] Static files served
- [ ] User's UI drafts integrated
- [ ] Dashboard images served via API

### Phase 6 — Deploy
- [ ] systemd service file
- [ ] systemd timer (daily 19:00 MSK)
- [ ] Backup script
- [ ] Health check script
- [ ] README with setup instructions

---

## 9. FUTURE PHASES (OUT OF SCOPE)

### Phase 7: HAR Backtest Migration
- Migrate `07_*` modules to `si_vol/backtest/`
- Add `/api/backtest/run` endpoint
- Separate systemd timer (weekly)

### Phase 8: Forecasting Engine (`predGL.json`)
- GAS-X + ARSVJ + EVT model
- Requires cloud GPU (or CPU-only simplified version)
- Separate microservice or cloud function

### Phase 9: Thursday Analysis
- Migrate `10_*` modules
- Integrate with weekly pipeline schedule

---

## 10. APPENDIX: KEY FUNCTION ROUTING

Quick reference for finding any function after migration:

| Function | Source | Target |
|---|---|---|
| `SiVolatilityReport` | 11 | `si_vol/schemas/models.py` |
| `validate_pipeline_output()` | 11 | `si_vol/schemas/models.py` |
| `NullableValue` | 11 | `si_vol/schemas/models.py` |
| `setup_logging()` | 01 | `si_vol/config.py` |
| `null_val()`, `is_null()`, `vscalar()` | 01 | `si_vol/helpers.py` |
| `r_vol()`, `r_ratio()`, `r_z()`, `r_pct()` | 01 | `si_vol/helpers.py` |
| `MoexClient` | 02 | `si_vol/moex/client.py` |
| `identify_contracts()` | 02 | `si_vol/moex/contracts.py` |
| `make_ticker()`, `parse_ticker()` | 02 | `si_vol/moex/contracts.py` |
| `moex_load_candles()` | 03 | `si_vol/data/loader.py` |
| `validate_ohlcv()` | 03 | `si_vol/data/validation.py` |
| `load_all_candles()` | 03 | `si_vol/data/loader.py` |
| `yang_zhang_series()` | 03 | `si_vol/estimators/volatility.py` |
| `cc_vol_series()` | 03 | `si_vol/estimators/volatility.py` |
| `semivariance_ann()` | 03 | `si_vol/estimators/volatility.py` |
| `percentile_rank_250()` | 03 | `si_vol/estimators/volatility.py` |
| `realized_skew()`, `realized_kurtosis()` | 03 | `si_vol/estimators/volatility.py` |
| `vol_of_vol()` | 03 | `si_vol/estimators/volatility.py` |
| `latest()`, `at_offset()` | 03 | `si_vol/estimators/volatility.py` |
| `gap_statistics()` | 03 | `si_vol/estimators/gaps.py` |
| `calc_har_rv_forecast()` | 03 | `si_vol/estimators/forecast.py` |
| `calc_session_aligned_rv_forecast()` | 03 | `si_vol/estimators/forecast.py` |
| `classify_skew()` | 04 | `si_vol/analytics/classifiers.py` |
| `classify_kurtosis()` | 04 | `si_vol/analytics/classifiers.py` |
| `classify_semivar_asymmetry()` | 04 | `si_vol/analytics/classifiers.py` |
| `classify_basis_vol()` | 04 | `si_vol/analytics/classifiers.py` |
| `build_samuelson_daily_1_10()` | 04 | `si_vol/analytics/samuelson.py` |
| `calc_samuelson_status()` | 04 | `si_vol/analytics/samuelson.py` |
| `avg_volume_nd()` | 04 | `si_vol/analytics/volume.py` |
| `activity_weight()` | 04 | `si_vol/analytics/volume.py` |
| `f2_reliability_label()` | 04 | `si_vol/analytics/volume.py` |
| `calc_historical_volume_ratio_by_dte()` | 04 | `si_vol/analytics/volume.py` |
| `detect_emerging_gap_regime()` | 04 | `si_vol/analytics/gap_regime.py` |
| `build_data_quality_warnings()` | 04 | `si_vol/analytics/warnings.py` |
| `compute_vol_metrics()` | 05_1 | `si_vol/pipeline/metrics.py` |
| `compute_semivar_asymmetry()` | 05_1 | `si_vol/pipeline/metrics.py` |
| `compute_roc()` | 05_1 | `si_vol/pipeline/metrics.py` |
| `classify_regime()` | 05_1 | `si_vol/pipeline/metrics.py` |
| `build_state_labels()` | 05_1 | `si_vol/pipeline/metrics.py` |
| `run_self_tests()` | 05_2 | `si_vol/pipeline/self_tests.py` |
| `plot_analytics_dashboard()` | 05_3 | `si_vol/viz/dashboard.py` |
| `compute_rv_forecasts()` | 05_4 | `si_vol/pipeline/forecasts.py` |
| `find_next_thursday()` | 05_4 | `si_vol/pipeline/forecasts.py` |
| `run_pipeline()` | 05_5 | `si_vol/pipeline/runner.py` |
| `summarize_json_data()` | 06 | `si_vol/pipeline/summary.py` |
| `print_summary_report()` | 06 | `si_vol/pipeline/summary.py` |
