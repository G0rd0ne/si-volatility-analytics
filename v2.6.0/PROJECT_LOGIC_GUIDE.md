# Si Volatility Analytics: Comprehensive Project Logic Guide

**Version:** 2.6.0  
**Last Updated:** 2025-04-08  
**Primary Execution Environment:** Google Colab  
**Target Audience:** AI Agents, ML Systems, Technical Analysts

---

## 📋 PROJECT SUMMARY (FOR CONTEXT)

**Название проекта:** Si Volatility Analytics & HAR Backtest System  
**Цель:** Анализ волатильности фьючерсов на российский рубль (Si) для опционной торговли

**Ключевые возможности:**
1. **Volatility State Analysis (v2.6.0 Pipeline)** — Полный анализ текущего состояния волатильности для F0/F1/F2 контрактов
2. **HAR-RV Backtest System** — Бэктест-система для калибровки β-коэффициентов HAR-модели прогнозирования волатильности
3. **Gap Regime Detection** — Детектирование emerging gap-режимов для risk management
4. **Samuelson Effect Analysis** — Анализ волатильности по DTE (days to expiry)
5. **Calendar Spread Analytics** — Cross-contract метрики для календарных спрэдов

**Execution Environment:**
- **Primary:** Google Colab (cell-based execution, shared globals, `/content/` filesystem)
- **Data Sources:** MOEX ISS API (free, primary and only source for all data)
- **Dependencies:** numpy, pandas, matplotlib, scipy, requests

---

## 🏗️ АРХИТЕКТУРА ПРОЕКТА

### DUAL-PATH ARCHITECTURE

Проект состоит из **двух независимых execution paths**:

**Path A: Production Pipeline (Cell 01-06)**
- Модульная архитектура (cell_05_1-5_5)
- Реал-тайм анализ волатильности F0/F1/F2
- JSON output для LLM-парсинга
- Dashboard visualization
- Self-tests для валидации

**Path B: HAR Backtest System (Cell 07_1, 07_2_01-09b, 07_3-4)**
- Microservice-style модули
- Калибровка β-коэффициентов HAR-модели
- Бэктест на исторических данных (MOEX ISS API)
- Rolling optimization (out-of-sample)
- Сравнение HAR vs EWMA baselines

**Shared Core (Cell 01-04)**
Оба пути используют общие модули:
- Config & utils (01)
- MOEX API client & contracts (02)
- Data loading & volatility estimators (03)
- Analytics & classification (04)

---

## 📦 МОДУЛЬНАЯ СТРУКТУРА

### CELL 01: Config & Utils (`cell_01_config_utils.py`)

**Назначение:** Единая конфигурация, константы, утилиты, логирование

**Ключевые компоненты:**

1. **Версионирование:**
   - `SCHEMA_VERSION = "2.6.0"` — версия схемы JSON output
   - `ENGINE_VERSION = "dual_vol_engine_0.6.0"` — версия движка

2. **Configuration Dataclasses:**
   ```python
   @dataclass(frozen=True)
   class MoexConfig:
       base_url: str = "https://iss.moex.com/iss"
       board: str = "RFUD"  # Futures board
       retry: int = 3
       timeout: int = 30
       page_size: int = 500
       session_end_minutes: int = 18 * 60 + 45  # 18:45 MSK
       session_duration_minutes: int = 8 * 60 + 45  # 8ч 45мин
   
   @dataclass(frozen=True)
   class VolConfig:
       tdy: int = 252  # Trading days per year
       history_days: int = 600  # Lookback для загрузки
       active_phase_days: int = 365  # Samuelson analysis window
       ewma_spans: tuple = (1, 3, 5, 10, 20)
       lookback_candles: int = 5
       dte_tolerance: int = 5
   
   @dataclass(frozen=True)
   class ThresholdConfig:
       semivar_absent: float = 1.1
       semivar_mild: float = 1.3
       semivar_moderate: float = 1.7
       vov_stable: float = 0.25
       vov_elevated: float = 0.40
       # ... 15+ thresholds для классификации
   ```

3. **Logging:**
   ```python
   def setup_logging(level: str = "INFO") -> logging.Logger:
       # Structured logging для pipeline
       return logging.getLogger("si_vol_pipeline")
   ```

4. **Domain Exceptions:**
   - `AnalyticsError` — base pipeline exception
   - `DataValidationError` — invalid market data

5. **Utility Functions:**
   - `null_val(reason)` — стандартный null-маркер
   - `is_null(v)` — проверка null-значений
   - `r_vol(v)`, `r_ratio(v)`, `r_z(v)`, `r_pct(v)` — precision helpers (6/4/2/1 decimals)
   - `vscalar(d, key)` — извлечение float из dict (поддерживает plain float и null_val)

6. **Constants:**
   ```python
   MONTH_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}  # Futures month codes
   Q_MONTHS = [3, 6, 9, 12]  # Quarterly expirations
   WINDOWS = (5, 20, 60, 250)  # Vol windows
   SEMIVAR_WINDOWS = (5, 20)
   GAP_WINDOWS = (5, 20, 60)
   ```

**Логика:**
Этот модуль — единый source of truth для всех параметров. Все числовые пороги вынесены в `ThresholdConfig`, что позволяет легко подстраивать чувствительность системы без изменения логики.

---

### CELL 02: MOEX Client & Contracts (`cell_02_moex_contracts.py`)

**Назначение:** HTTP client для MOEX ISS API, идентификация контрактов Si

**Ключевые компоненты:**

1. **MoexClient:**
   ```python
   class MoexClient:
       def __init__(self, cfg: MoexConfig):
           self._session = requests.Session()
           # Connection pooling, retry logic
       
       def get(self, url: str, params: Optional[dict] = None) -> requests.Response:
           # Exponential backoff retry (3 attempts)
           # Raises on failure after 3 retries
   ```

2. **Ticker Logic:**
   ```python
   def make_ticker(month: int, year: int) -> str:
       # SiH5, SiM5, SiU5, SiZ5 (quarter months only)
       return f"Si{CODE_BY_MONTH[month]}{year % 10}"
   
   def parse_ticker(secid: str, today: date) -> tuple[int, int]:
       # Extract month, year from ticker
       # Decade guessing для short tickers (SiH5 → 2025)
   ```

3. **Expiry Fetching:**
   ```python
   def get_expiry(client: MoexClient, cfg: MoexConfig, secid: str, today: date) -> date:
       # 1. Try LSTTRADE field (actual last trading date)
       # 2. Fallback: _third_thursday(year, month) estimation
       # Returns: expiry date
   ```

4. **Contract Identification:**
   ```python
   def identify_contracts(client: MoexClient, cfg: MoexConfig, today: date) -> dict[str, ContractMeta]:
       # Generates tickers for 5 years back + 3 years forward
       # Fetches all expiries via MOEX API
       # Sorts by expiry date → identifies F1 (nearest active)
       # Returns: {"F0": ContractMeta, "F1": ..., "F2": ..., "F-1": ..., ...}
       # O(N log N) complexity
   ```

**Логика:**
Система автоматически находит ближайший активный контракт (F1), следующий (F2), предыдущий истекший (F0), и все исторические контракты до F-10 (11 контрактов назад). Это критично для Samuelson analysis — нужна история expired контрактов для построения DTE-buckets.

**Идентификация контрактов:**
- F1 = ближайший активный (expiry >= today)
- F2 = следующий активный (expiry > F1.expiry)
- F0 = последний истекший (expiry < today, ближайший к today)
- F-1, F-2, ..., F-10 = исторические контракты (для Samuelson daily table)

---

### CELL 03: Data Estimators (`cell_03_data_estimators.py`)

**Назначение:** Загрузка OHLCV, volatility estimators, gap analysis, HAR/EWMA forecasts

**Ключевые компоненты:**

1. **OHLCV Loading:**
   ```python
   def moex_load_candles(client, cfg, secid, date_from, date_till) -> pd.DataFrame:
       # Pagination через MOEX ISS API (page_size=500)
       # Returns: DataFrame[date, open, high, low, close, volume]
       # Handles column name variations (BEGIN/TRADEDATE/DATE/END)
   
   def validate_ohlcv(df, secid) -> pd.DataFrame:
       # Validates: required columns, positive prices, high >= low
       # Removes duplicates, sorts by date
       # Converts dtypes (date → datetime, prices → float)
   
   def load_all_candles(contracts, today, client, cfg, vol_cfg):
       # F1, F2: history_days=600 до today
       # F0, F-1, ..., F-10: active_phase_days=365 до expiry
       # Returns: (ohlcv, samuelson_sources)
   ```

2. **Volatility Estimators:**
   ```python
   def yang_zhang_series(df, window, tdy=252) -> pd.Series:
       # Yang-Zhang estimator (high-low-open-close)
       # Формула: σ_YZ = sqrt(σ_o² + k·σ_c² + (1-k)·σ_rs²) * sqrt(tdy)
       # k = 0.34 / (1.34 + (n+1)/(n-1))
       # Returns: annualized vol series
   
   def cc_vol_series(df, window, tdy=252) -> pd.Series:
       # Close-to-close vol (простейший estimator)
       # log(close_t / close_{t-1}).rolling(window).std() * sqrt(tdy)
   
   def semivariance_ann(df, window, side: str, tdy=252) -> tuple[float, int]:
       # Полудисперсия (downside or upside)
       # side="down": только отрицательные returns
       # side="up": только положительные returns
       # Returns: (semivar, n_observations)
   ```

3. **Gap Analysis (ENHANCED in v2.6.0):**
   ```python
   def gap_statistics(df, window, cfg) -> dict:
       # Overnight gap анализ:
       # 1. Gap amplitude: log(open_t / close_{t-1})
       # 2. Normalization: gap / rolling_std → sigma units
       # 3. Distribution: binning по sigma (0-0.25σ, 0.25-0.5σ, ..., ≥1.0σ)
       # 4. Significant gaps: normalized >= 1.0σ
       # 5. Quantiles: p10, p25, p50, p75, p90, p95
       # 
       # NEW in v2.6.0:
       # 6. Gap clustering: autocorr lag-1 для binary gap occurrence
       # 7. Directional bias: up_gaps vs down_gaps frequency
       # 
       # Returns: {
       #   "frequency_pct": % дней с gap >= 1.0σ,
       #   "amplitude_avg": средняя амплитуда gap,
       #   "gap_quantiles": {...},
       #   "gap_sigma_distribution": {...},
       #   "gap_clustering": {autocorr, interpretation, llm_hint},
       #   "directional_bias": {up_freq, down_freq, bias_label, llm_hint}
       # }
   ```

4. **HAR-RV Forecast (Cell 03):**
   ```python
   def calc_har_rv_forecast(rv_1d, rv_5d, rv_20d, horizon_days, filter_gaps=False) -> dict:
       # HAR model: RV_t+h = β_d·RV_d + β_w·RV_w + β_m·RV_m
       # 
       # Dynamic β weights based on horizon:
       # - horizon <= 1d:  β_d=0.60, β_w=0.30, β_m=0.10 (short-term dominance)
       # - horizon <= 5d:  β_d=0.40, β_w=0.40, β_m=0.20
       # - horizon <= 10d: β_d=0.30, β_w=0.40, β_m=0.30
       # - horizon > 10d:  β_d=0.25, β_w=0.35, β_m=0.40 (long-term mean reversion)
       # 
       # Optional gap filtering (beta feature, not used in production)
       # 
       # Returns: {forecast_RV, method, coefficients, components, horizon_days, available}
   ```

5. **EWMA Forecast:**
   ```python
   def calc_session_aligned_rv_forecast(yz_series, current_time, target_datetime, cfg, ewma_spans):
       # Fractional EWMA interpolation
       # 
       # Horizon calculation:
       # - full_days = (target_date - current_date).days
       # - frac = (session_end - current_minute) / session_duration
       # - horizon = full_days + frac
       # 
       # Interpolation strategy:
       # - horizon <= 1: EWMA(1)
       # - horizon <= 3: 0.7·EWMA(1) + 0.3·EWMA(3)
       # - horizon <= 5: 0.6·EWMA(3) + 0.4·EWMA(5)
       # - horizon <= 10: 0.3·EWMA(5) + 0.7·EWMA(10)
       # - horizon <= 20: 0.5·EWMA(10) + 0.5·EWMA(20)
       # - horizon > 20: Mean reversion к долгосрочной vol
       # 
       # Returns: {target_datetime, fractional_trading_days, forecast_RV, method, llm_hint}
   ```

**Логика:**
Модуль реализует два конкурирующих метода прогнозирования волатильности:
- **HAR-RV (Cell 03):** Dynamic β weights, адаптируется к горизонту
- **EWMA (Cell 03):** Fractional interpolation, mean reversion для длинных горизонтов

В v2.6.0 gap_statistics дополнен clustering & directional bias метриками для улучшенного risk management.

---

### CELL 04: Analytics & Classification (`cell_04_analytics_classification.py`)

**Назначение:** Классификаторы, Samuelson effect, gap regime detection, warnings

**Ключевые компоненты:**

1. **Distribution Classifiers:**
   ```python
   def classify_skew(v, thr) -> str:
       # v < -0.5 → "left" (put skew)
       # v > 0.5 → "right" (call skew)
       # else → "symmetric"
   
   def classify_kurtosis(v, thr) -> str:
       # v < -1.0 → "platykurtic" (thin tails)
       # v > 1.0 → "leptokurtic" (fat tails)
       # else → "normal"
   ```

2. **Semivar Asymmetry:**
   ```python
   def classify_semivar_asymmetry(semivar_down, semivar_up, nd, nu, thr) -> dict:
       # CRITICAL FIX in v2.6.0: semivar_up=0.0 → "extreme_downside"
       # 
       # ratio = semivar_down / semivar_up
       # ratio >= 1.7 → "strong_downside_bias"
       # ratio >= 1.3 → "moderate_downside_bias"
       # ratio >= 1.1 → "mild_downside_bias"
       # ratio < 1.1 → "symmetric"
       # 
       # semivar_up=0.0 случай: all upside returns were zero → strongest realized bias
   ```

3. **Samuelson Daily Table (DTE 1-10):**
   ```python
   def build_samuelson_daily_1_10(expired_sources, contracts, tdy=252) -> dict:
       # Aggregates historical data for expired contracts (F0-F10)
       # For each contract:
       #   - Compute DTE for each bar: (expiry - bar_date).days
       #   - Extract log returns for DTE in [1, 10]
       # 
       # For each DTE bucket (1-10):
       #   - Aggregate returns from ALL expired contracts
       #   - Compute: mean_vol, std_bucket, p10, p90, n_obs
       #   - Shrinkage: if n < 10, blend with overall bucket mean
       #   - Interpolation: if missing DTE, interpolate from neighbors
       # 
       # Returns: {
       #   "DTE_1": {mean_vol, std_bucket, p10, p90, n_obs, reliability},
       #   "DTE_2": {...},
       #   ...
       #   "DTE_10": {...},
       #   "methodology": "..."
       # }
   ```

4. **Samuelson Status (Current F1 vs DTE bucket):**
   ```python
   def calc_samuelson_status(yz_20d_f1, dte_f1, samuelson_daily, thr) -> dict:
       # Compares current F1 YZ_20d to historical DTE bucket
       # 
       # z = (yz_20d_f1 - mean_vol) / std_bucket
       # 
       # z > 1.5 → "stressed" (vol выше нормы для данного DTE)
       # z < -1.5 → "complacent" (vol ниже нормы)
       # else → "normal_for_dte"
       # 
       # Returns: {status, z_score, deviation_pp, note}
   ```

5. **Emerging Gap Regime Detection (3-LEVEL):**
   ```python
   def detect_emerging_gap_regime(gap_analysis, thr) -> dict:
       # Трехуровневая проверка через gap_5d, gap_20d, gap_60d
       # 
       # Conditions для emerging regime:
       # 1. freq_5d >= 10.0% (high recent frequency)
       # 2. freq_60d < 6.0% (calm historical background)
       # 3. freq_5d / freq_60d >= 2.0 (acceleration)
       # 
       # Gradient analysis:
       # ratio_5d_to_20d = freq_5d / freq_20d
       # ratio_20d_to_60d = freq_20d / freq_60d
       # 
       # Regime types:
       # - "established": 5d ≈ 20d >> 60d (устойчивый, 2-4 weeks persistence)
       # - "recent_spike": 5d >> 20d > 60d (краткосрочный, 1-2 weeks)
       # - "gradual_buildup": плавный градиент (формируется, uncertain)
       # - "ambiguous": не попадает в категории
       # 
       # Enhanced metrics (v2.6.0):
       # - Percentile rank: freq_5d vs freq_60d distribution
       # - Amplitude ratio: amplitude_avg_5d / amplitude_avg_60d
       # - Gap clustering: autocorr from gap_5d
       # - Directional bias: up_gaps vs down_gaps
       # 
       # Returns: {
       #   emerging: bool,
       #   regime_type: str,
       #   confidence: "high" | "medium" | "low",
       #   severity: "high" | "moderate" | "low",
       #   persistence_estimate: "2-4 weeks" | ...,
       #   metrics: {...},
       #   interpretation: str,
       #   recommendation: str,
       #   enhanced_analysis: {clustering, amplitude, directional_bias},
       #   llm_hint: str
       # }
   ```

6. **Volume Ratio Historical Analysis:**
   ```python
   def calc_historical_volume_ratio_by_dte(ohlcv, contracts, samuelson_sources, today, tgt_f1, tgt_f2, vol_cfg):
       # Compares current F2/F1 volume ratio to historical analogs
       # 
       # Current ratio:
       # vol5_f1 = avg last 5 days volume F1
       # vol5_f2 = avg last 5 days volume F2
       # current_ratio = vol5_f2 / vol5_f1 * 100%
       # 
       # Historical analogs:
       # For each expired pair (F_i, F_{i+1}):
       #   - Find bars where DTE_F_i ≈ tgt_f1 ± 5 days AND DTE_F_{i+1} ≈ tgt_f2 ± 5 days
       #   - Compute avg volume ratio for those bars
       # 
       # Aggregate:
       # hist_mean = mean of all analog ratios
       # hist_std = std of all analog ratios
       # z = (current_ratio - hist_mean) / hist_std
       # 
       # Returns: {
       #   current_ratio_pct,
       #   historical_avg_pct,
       #   historical_std_pct,
       #   z_vs_history,
       #   interpretation: "аномально ликвиден" | "в пределах нормы" | "аномально неликвиден"
       # }
   ```

7. **Data Quality Warnings:**
   ```python
   def build_data_quality_warnings(vol_data, rv_forecast_session, basis_vol_20, f2_rel_label, gap_analysis, thr):
       # Automated warning generation:
       # 
       # 1. VoV check:
       #    vov_5d > 0.40 → VOMMA_RISK_HIGH (critical)
       # 
       # 2. Forecast identity:
       #    |forecast_RV_F1 - forecast_RV_F2| < 1e-7 → FORECAST_IDENTITY_F1_EQ_F2 (critical)
       #    (Calendar kill-signal на RV-уровне)
       # 
       # 3. Basis vol:
       #    basis_vol_20d > 0.025 → BASIS_VOL_HIGH (warning)
       # 
       # 4. Gap regime:
       #    detect_emerging_gap_regime() → GAP_REGIME_EMERGING (critical/warning/info)
       # 
       # 5. F2 gap artifact:
       #    F2 reliability=reduced AND F2_gap_5d > F1_gap_5d * 1.5 → F2_GAP_LIQUIDITY_ARTIFACT (warning)
       # 
       # Returns: [
       #   {code, severity, message, affected_strategies, regime_details?},
       #   ...
       # ]
   ```

**Логика:**
Этот модуль — аналитический движок. Классификаторы преобразуют raw metrics в human/LLM-readable labels. Samuelson analysis позволяет сравнивать текущую волатильность с исторической нормой для данного DTE. Gap regime detection — ключевая инновация v2.6.0 для раннего обнаружения emerging vol spikes.

---

### CELL 05.1-05.5: Modular Production Pipeline

**Cell 05.1: Metrics Computation** (`cell_05_1_metrics.py`)
```python
def compute_vol_metrics(ohlcv, days_since_f0, cfg):
    # Для F0, F1, F2:
    # - YZ vol для WINDOWS (5, 20, 60, 250)
    # - CC vol для WINDOWS
    # - Semivar для SEMIVAR_WINDOWS (5, 20)
    # - Skew, kurtosis для (5, 20, 60)
    # - Vol of vol для (5, 20, 60)
    # - F0 stale check: если days_since_f0 > 0, 5d metrics → null
    # 
    # Returns: (vol_data, vol_series, semi_nobs)

def compute_semivar_asymmetry(vol_data, semi_nobs, thr):
    # Для F1, F2 × windows (5, 20):
    #   - classify_semivar_asymmetry(semivar_down, semivar_up)
    # Returns: dict

def compute_roc(vol_series):
    # Rate of change для F1 YZ vol:
    # - YZ_5d: roc_5d, roc_10d, roc_20d
    # - YZ_20d: roc_5d, roc_10d, roc_20d
    # - Momentum classification: accelerating/decelerating cooling/warming
    # Returns: dict

def classify_regime(pct_ranks, vol_data):
    # Regime classification на основе percentile rank YZ_20d:
    # p < 20 → "low"
    # p < 65 → "normal"
    # p < 90 → "elevated"
    # p >= 90 → "crisis"
    # 
    # Dynamics: compression | stable | short_term_spike
    # Returns: {level, dynamics, composite}

def build_state_labels(cross, stress_obj, semivar_asym, f2_rel, regime, samuelson_status_obj, roc, cal_rv_fmb, basis_vol_20, thr):
    # Aggregates все labels:
    # - front_vs_next_vol
    # - front_specific_stress
    # - downside_asymmetry
    # - f2_reliability
    # - vol_regime
    # - samuelson_status
    # - vol_trend
    # - calendar_rv_direction
    # - basis_vol_label
```

**Cell 05.2: Self-Tests** (`cell_05_2_tests.py`)
```python
def run_self_tests(output, rv_forecast_session, samuelson_daily):
    # 7 validation tests:
    # T1: forecast_RV присутствует
    # T2: Samuelson DTE_1..DTE_10 найдены
    # T3: volume_ratio_historical присутствует
    # T4: data_quality.warnings is list
    # T5: semivar_up=0.0 → label="extreme_downside"
    # T6: gap_analysis keys valid
    # T7: VoV > 0.40 → VOMMA_RISK_HIGH warning present
    # 
    # Returns: list of error messages (empty if all pass)
```

**Cell 05.3: Dashboard** (`cell_05_3_dashboard.py`)
```python
def plot_analytics_dashboard(output, vol_data, roc, cfg, schema_version):
    # 6-panel matplotlib dashboard:
    # 1. Vol Term Structure (F1 vs F2 bars)
    # 2. Gap Quantiles F1 20d (line plot)
    # 3. Volume Ratio (placeholder)
    # 4. Samuelson Daily (placeholder)
    # 5. Momentum ROC (placeholder)
    # 6. Stress Components (placeholder)
    # 
    # Saves: /content/si_vol_analytics/charts/analytics_dashboard_2.6.0.png
```

**Cell 05.4: RV Forecasts** (`cell_05_4_forecasts.py`)
```python
def compute_rv_forecasts(vol_series, gap_analysis, today, cfg):
    # Вычисляет HAR и EWMA forecasts для:
    # - nearest_thu_1845: ближайший четверг после today 18:45 MSK
    # - next_thu_1845: nearest_thu + 7 дней 18:45 MSK
    # 
    # Для каждого target:
    # 1. EWMA forecast через calc_session_aligned_rv_forecast()
    # 2. HAR forecast через calc_har_rv_forecast()
    # 3. Spread = HAR_RV - EWMA_RV
    # 4. Consensus interpretation:
    #    |spread| < 0.5pp → "consensus"
    #    spread > 1.5pp → "HAR_significantly_higher"
    #    spread < -1.5pp → "EWMA_significantly_higher"
    #    else → "mild_disagreement"
    # 
    # Returns: {
    #   to_nearest_thu_1845: {EWMA_forecast, HAR_forecast, spread, consensus, recommended_forecast},
    #   to_next_thu_1845: {...},
    #   llm_hint: "..."
    # }
```

**Cell 05.5: Pipeline Orchestrator** (`cell_05_5_pipeline.py`)
```python
def run_pipeline(cfg=None) -> dict:
    # Main execution flow:
    # 
    # 1. Contract identification (Cell 02)
    # 2. OHLCV download (Cell 03)
    # 3. Metrics computation (Cell 05.1)
    # 4. RV forecasts (Cell 05.4)
    # 5. Samuelson daily table (Cell 04)
    # 6. Gap analysis (Cell 03)
    # 7. Volume ratio historical (Cell 04)
    # 8. Cross-contract metrics (basis vol, spread, stress)
    # 9. Data quality warnings (Cell 04)
    # 10. Self-tests (Cell 05.2)
    # 11. Dashboard (Cell 05.3)
    # 12. JSON export
    # 
    # Returns: output dict (11 sections)
```

**Логика Cell 05:**
Модульная архитектура позволяет легко тестировать и расширять pipeline. Каждый модуль (05.1-05.4) — standalone функция с четкими входами/выходами. Cell 05.5 — orchestrator, который вызывает все модули в правильном порядке.

---

### CELL 06: Data Summary (`cell_06_data_summary.py`)

**Назначение:** Post-processing утилита для валидации JSON output

**Ключевые функции:**
```python
def summarize_json_data(json_path) -> dict:
    # Читает JSON output от Cell 05.5
    # Проверяет 11 секций:
    # 1. Contracts: F0, F1, F2 present?
    # 2. Volume Analysis: avg_vol, ratio valid?
    # 3. Data Quality: warnings count, aligned_bars >= 21?
    # 4. Volatility: метрики для F0/F1/F2
    # 5. Samuelson: 10 DTE buckets filled?
    # 6. RV Forecast: nearest/next Thu forecasts valid?
    # 7. Gap Analysis: 3 contracts × 3 windows?
    # 8. Cross Contract: basis_vol_20d, stress valid?
    # 9. State Labels: unknown count?
    # 10. Volume Ratio Historical: z-score valid?
    # 
    # Returns: {
    #   meta: {...},
    #   sections: {
    #     contracts: {status: "✅ OK", summary: "..."},
    #     ...
    #   },
    #   overall_status: "✅ COMPLETE" | "⚠️ PARTIAL" | "❌ FAILED"
    # }

def print_summary_report(report):
    # Печатает ASCII-art report с иконками ✅/⚠️/❌
    # Детализирует RV Forecast (target dates, fractional days)
    # Детализирует Volume Ratio (z-score, interpretation)
```

**Логика:**
Независимый модуль для быстрой диагностики JSON output. Используется после запуска pipeline для проверки что все данные загружены корректно. Особенно полезен для LLM agents — может быстро определить какие секции имеют проблемы.

---

### CELL 07: HAR Backtest System (Modular Architecture)

**Cell 07.1: Config** (`cell_07_1_config.py`)
```python
@dataclass(frozen=True)
class BacktestConfig:
    forecast_horizons: tuple = (1, 2, 3, 5, 7)  # Trading days
    rolling_window: int = 60  # Out-of-sample window
    min_calibration_history: int = 40  # Min bars for beta fitting
    beta_grid_resolution: int = 10  # Grid search steps
    backtest_contracts: tuple = ("F0", "F-1", "F-2", ..., "F-10")
    output_dir: str = "./backtest_results"
    save_plots: bool = True
    save_json: bool = True
```

**Cell 07.2.01: Contract Identification** (`cell_07_2_01_contracts.py`)
```python
def identify_contracts_with_history(contracts, backtest_contracts):
    # Checks which contracts from backtest_contracts exist in contracts dict
    # Returns: {available, missing, total_available, total_missing}
```

**Cell 07.2.02: Data Loader** (`cell_07_2_02_data_loader.py`)
```python
class HistoricalDataLoader:
    # Loads OHLCV from MOEX ISS API
    def load_contract_data(contract, duration, bar_size):
        # Returns: DataFrame[date, open, high, low, close, volume]
```

**Cell 07.2.03: Realized Vol** (`cell_07_2_03_realized_vol.py`)
```python
def compute_realized_vol_forward(df, horizon_days, yang_zhang_series_fn, tdy):
    # Forward-looking realized vol для validation
    # For each bar t:
    #   realized_vol[t] = yang_zhang(df[t+1:t+horizon], horizon, tdy)
    # 
    # Returns: Series aligned with df.index (forward-shifted)
```

**Cell 07.2.04: HAR Forecast** (`cell_07_2_04_har_forecast.py`)
```python
def forecast_har_rolling(df, horizon_days, beta_d, beta_w, beta_m, yang_zhang_series_fn, tdy, min_calibration_history):
    # Rolling HAR forecast with FIXED betas (inputs)
    # For each bar t (after min_calibration_history):
    #   hist_df = df[:t]
    #   rv_1d = yang_zhang(hist_df, 1)
    #   rv_5d = yang_zhang(hist_df, 5)
    #   rv_20d = yang_zhang(hist_df, 20)
    #   forecast[t] = beta_d * rv_1d[-1] + beta_w * rv_5d[-1] + beta_m * rv_20d[-1]
    # 
    # Returns: Series[forecast_RV]
```

**CRITICAL DIFFERENCE: Cell 03 vs Cell 07.2.04 HAR:**
- **Cell 03 (`calc_har_rv_forecast`):**
  - **Dynamic β weights** based on horizon
  - Single-point forecast (latest values only)
  - Used in production pipeline (Cell 05)
  - No optimization — hardcoded adaptive weights

- **Cell 07.2.04 (`forecast_har_rolling`):**
  - **Fixed β weights** (input parameters)
  - Rolling forecast (entire history)
  - Used in backtest system (Cell 07)
  - Designed for optimization — betas are search variables

**Cell 07.2.05: EWMA Forecast** (`cell_07_2_05_ewma_forecast.py`)
```python
def forecast_ewma_rolling(df, horizon_days, yang_zhang_series_fn, span, tdy, min_calibration_history):
    # Rolling EWMA baseline для comparison
    # For each bar t:
    #   rv_series = yang_zhang(df[:t], 20)
    #   forecast[t] = rv_series.ewm(span=span).mean()[-1]
```

**Cell 07.2.06: Evaluation** (`cell_07_2_06_evaluation.py`)
```python
def evaluate_forecast(realized, forecast) -> dict:
    # Metrics:
    # - RMSE: sqrt(mean((y_true - y_pred)²))
    # - MAE: mean(|y_true - y_pred|)
    # - R²: 1 - SS_res / SS_tot
    # - Bias: mean(y_pred - y_true)
    # - Hit rate: % прогнозов в ±10% от realized
    # 
    # Returns: {RMSE, MAE, R2, Bias, Hit_Rate, n_obs}

def compare_forecasts(realized, har_forecast, ewma_forecast):
    # Compares HAR vs EWMA:
    # 1. Evaluate both forecasts
    # 2. Compute improvement metrics:
    #    - R2_gain_pp: HAR_R2 - EWMA_R2 (percentage points)
    #    - RMSE_reduction_pct: (EWMA_RMSE - HAR_RMSE) / EWMA_RMSE * 100
    # 
    # Returns: {HAR_metrics, EWMA_metrics, improvement}
```

**Cell 07.2.07: Optimization** (`cell_07_2_07_optimization.py`)
```python
def optimize_beta_coefficients(df, horizon_days, compute_realized_vol_fn, forecast_har_fn, evaluate_forecast_fn, yang_zhang_series_fn, tdy):
    # Оптимизация через scipy.optimize.minimize:
    # 
    # Objective function:
    #   def objective(betas):
    #       beta_d, beta_w, beta_m = betas
    #       forecast = forecast_har_rolling(df, horizon, beta_d, beta_w, beta_m, ...)
    #       realized = compute_realized_vol_forward(df, horizon, ...)
    #       return RMSE(realized, forecast)
    # 
    # Constraints:
    #   - beta_d + beta_w + beta_m = 1.0 (sum-to-one)
    #   - 0.0 <= beta_i <= 1.0 (non-negative)
    # 
    # Method: SLSQP (Sequential Least Squares Programming)
    # Initial guess: [0.35, 0.40, 0.25]
    # 
    # Returns: (beta_d, beta_w, beta_m, metrics)
```

**Cell 07.2.08: Engine** (`cell_07_2_08_engine.py`)
```python
# Aggregates all functions from Cell 07.2.01-07 into globals()
# Provides unified interface for Cell 07.2.09a/b
# 
# Functions loaded:
# - yang_zhang_series
# - compute_realized_vol_forward
# - forecast_har_rolling
# - forecast_ewma_rolling
# - evaluate_forecast
# - compare_forecasts
# - optimize_beta_coefficients
```

**Cell 07.2.09a: Data Pipeline** (`cell_07_2_09a_data_pipeline.py`)
```python
class BacktestDataPipeline:
    def __init__(self, moex_client, trading_days_per_year):
        # Wrapper for MOEX ISS API data loading
    
    def load_contract_history(self, contract, duration_days, bar_size):
        # Loads historical bars from MOEX ISS API
        # Returns: DataFrame[date, open, high, low, close, volume]
    
    def load_portfolio_history(self, contracts, duration_days, bar_size):
        # Loads data for multiple contracts from MOEX
        # Returns: {symbol: DataFrame}
    
    def validate_data_quality(self, df, symbol):
        # Quality checks:
        # - Min bars >= 60
        # - No NaN in OHLC
        # - high >= low
        # - Positive prices
```

**Cell 07.2.09b: Backtest Execution** (`cell_07_2_09b_backtest_execution.py`)
```python
class HARBacktestEngine:
    def __init__(self, horizon_days, trading_days_per_year):
        # HAR backtest engine
    
    def run_single_contract(self, df, symbol, optimize_betas):
        # 1. Compute forward realized vol
        # 2. If optimize_betas:
        #      beta_d, beta_w, beta_m = optimize_beta_coefficients(df, ...)
        #    Else:
        #      beta_d, beta_w, beta_m = (0.4, 0.4, 0.2)  # Default
        # 3. Forecast HAR rolling
        # 4. Forecast EWMA rolling (baseline)
        # 5. Compare forecasts
        # 
        # Returns: {
        #   symbol,
        #   betas: {daily, weekly, monthly},
        #   comparison: {HAR_metrics, EWMA_metrics, improvement},
        #   data_quality: {...}
        # }
    
    def run_portfolio(self, portfolio_data, optimize_betas):
        # Runs backtest for all contracts in portfolio
        # Returns: {symbol1: result1, symbol2: result2, ...}
```

**Cell 07.3: Visualization** (`cell_07_3_visualization.py`)
```python
def plot_backtest_results(results, output_dir, show_plots):
    # For each symbol:
    # 1. Plot forecast vs realized (3 panels: HAR, EWMA, comparison)
    # 2. Plot residuals histogram
    # 3. Plot cumulative error
    # Saves: {output_dir}/{symbol}_backtest_{timestamp}.png

def save_backtest_json(results, output_dir, timestamp):
    # Saves results to JSON with metadata
    # {output_dir}/har_backtest_results_{timestamp}.json

def validate_backtest_results(results):
    # Validation checks:
    # - All symbols have betas?
    # - R² in [-1, 1]?
    # - RMSE positive?
    # - Hit rate in [0, 100]?
    # 
    # Returns: {warnings, errors}
```

**Cell 07.4: Main Entrypoint** (`cell_07_4_main.py`)
```python
def run_har_backtest(moex_client, contracts, horizon_days=21, optimize_betas=True, ...):
    # Main backtest workflow:
    #
    # STEP 1: Load data from MOEX ISS API
    # pipeline = BacktestDataPipeline(moex_client, tdy)
    # portfolio_data = pipeline.load_portfolio_history(contracts, duration=365, bar_size='1 hour')
    # 
    # STEP 2: Run backtest
    # engine = HARBacktestEngine(horizon_days, tdy)
    # results = engine.run_portfolio(portfolio_data, optimize_betas=True)
    # 
    # STEP 3: Validate, plot, save
    # validate_backtest_results(results)
    # plot_backtest_results(results, output_dir)
    # save_backtest_json(results, output_dir, timestamp)
    # 
    # Returns: {symbol: backtest_result}

def run_backtest_self_tests() -> bool:
    # Self-tests для HAR system:
    # 1. Check globals loaded (BacktestDataPipeline, HARBacktestEngine, ...)
    # 2. Check visualization functions
    # 3. Check BacktestDataPipeline interface
    # 4. Check HARBacktestEngine interface
    # 5. Check compare_forecasts signature
    # 
    # Returns: True if all pass
```

**Логика Cell 07:**
Микросервисная архитектура для бэктеста. Каждый модуль (07.2.01-09b) — отдельный микросервис с четкой ответственностью:
- 01: идентификация контрактов
- 02: загрузка данных
- 03: realized vol (ground truth)
- 04: HAR forecast (fixed betas)
- 05: EWMA baseline
- 06: evaluation metrics
- 07: optimization (beta tuning)
- 08: aggregation engine
- 09a: data pipeline (MOEX ISS API integration)
- 09b: backtest execution

Cell 07.4 — единая точка входа для запуска полного backtest workflow.

---

## 🔄 ТЕКУЩИЙ ЭТАП: HAR BETA OPTIMIZATION

### Проблема:
HAR-модель в Cell 03 использует **hardcoded динамические веса** β-коэффициентов, которые подобраны эвристически:
```python
if horizon_days <= 1:
    beta_d, beta_w, beta_m = 0.60, 0.30, 0.10
elif horizon_days <= 5:
    beta_d, beta_w, beta_m = 0.40, 0.40, 0.20
# ...
```

Эти веса работают "в среднем", но не оптимальны для конкретного контракта Si на конкретном рынке MOEX.

### Цель:
Откалибровать оптимальные β-коэффициенты HAR-модели через **out-of-sample backtest** на исторических данных MOEX Si.

### Подход:
1. **Rolling optimization:**
   - Берем исторические данные (F0-F10 expired contracts) через MOEX ISS API
   - Для каждого контракта:
     - Split data: 60% train, 40% test (rolling window)
     - На train: optimize betas через SLSQP (minimize RMSE)
     - На test: validate forecast accuracy
   - Aggregate результаты по всем контрактам

2. **Сравнение HAR vs EWMA:**
   - HAR с оптимальными β — основная модель
   - EWMA с span=20 — baseline для сравнения
   - Метрики: R², RMSE, MAE, Hit Rate

3. **Horizon-specific optimization:**
   - Оптимизация для каждого horizon отдельно: 1d, 2d, 3d, 5d, 7d
   - Ожидаем: короткие горизонты → высокий β_d, длинные → высокий β_m

### Текущий статус (2025-04-08):
- ✅ Модульная архитектура Cell 07.2.01-09b завершена
- ✅ Self-tests для Cell 07.4 работают (`run_backtest_self_tests()` returns True)
- ✅ Data pipeline через MOEX ISS API готов
- ⏳ **В процессе:** Запуск первого полного backtest на реальных данных MOEX Si
- ⏳ **В процессе:** Анализ результатов и fine-tuning оптимизации

### Следующие шаги:
1. Инициализировать MOEX ISS API клиент
2. Загрузить исторические данные для F0-F10 (1 year history, 1h candles) через MOEX API
3. Запустить `run_har_backtest(moex_client, contracts, horizon_days=21, optimize_betas=True)`
4. Проанализировать:
   - Оптимальные β для каждого horizon
   - R² gain HAR vs EWMA
   - Improvement metrics (pp)
5. Интегрировать оптимальные β в Cell 03 production pipeline

---

## 📊 КЛЮЧЕВЫЕ КОНЦЕПЦИИ

### 1. DUAL-CONTRACT APPROACH
Система анализирует **три контракта одновременно:**
- **F0 (expired):** Недавно истекший контракт — используется для Samuelson analysis
- **F1 (current):** Ближайший активный — основной контракт для торговли
- **F2 (next):** Следующий активный — для calendar spreads и term structure

**Зачем нужны все три?**
- F1 vs F2: сравнение term structure (backwardation/contango)
- F0 vs F1: Samuelson effect (vol spike near expiry)
- F0-F10: исторические данные для DTE buckets (нормы vol по DTE)

### 2. SAMUELSON EFFECT
**Определение:** Волатильность фьючерса растет при приближении к экспирации (DTE → 0).

**Реализация:**
- Собираем исторические данные для всех expired контрактов (F0-F10)
- Для каждого бара вычисляем DTE = (expiry - bar_date).days
- Группируем returns по DTE buckets (1-10 дней)
- Вычисляем среднюю vol для каждого DTE bucket
- Сравниваем текущую F1 vol с нормой для текущего DTE

**Применение:**
- Если F1 vol > норма для DTE: "stressed" → short vol edges снижены
- Если F1 vol < норма для DTE: "complacent" → short vol edges повышены

### 3. GAP REGIME DETECTION
**Определение:** Периоды, когда overnight gaps становятся частыми и крупными.

**Метод (3-level cascade):**
1. **Short-term (5d):** Текущая частота gaps
2. **Mid-term (20d):** Среднесрочная частота
3. **Long-term (60d):** Фоновая частота

**Logic:**
- Если `freq_5d >= 10%` AND `freq_60d < 6%` AND `freq_5d / freq_60d >= 2.0` → emerging regime
- Gradient analysis: `freq_5d / freq_20d` и `freq_20d / freq_60d` определяют regime type

**Regime types:**
- **Established:** 5d ≈ 20d >> 60d — устойчивый gap-режим (2-4 weeks persistence)
- **Recent spike:** 5d >> 20d > 60d — краткосрочный всплеск (1-2 weeks)
- **Gradual buildup:** плавный градиент — формируется (uncertain)

**Enhanced metrics (v2.6.0):**
- **Gap clustering:** autocorr lag-1 для gap occurrence (gaps идут сериями?)
- **Directional bias:** up_gaps vs down_gaps frequency (asymmetric gaps?)
- **Amplitude ratio:** средняя амплитуда gap 5d vs 60d (gaps стали крупнее?)

**Применение:**
- Emerging gap regime → reduce wing width в Iron Condor
- Directional bias upside → prefer put calendars (avoid short calls)
- High clustering → gaps идут сериями, increase margin buffer

### 4. HAR-RV MODEL
**Heterogeneous AutoRegressive Realized Volatility** (Corsi, 2009)

**Формула:**
```
RV_{t+h} = β_d · RV_d + β_w · RV_w + β_m · RV_m
```
где:
- `RV_d` = daily realized vol (1d)
- `RV_w` = weekly realized vol (5d avg)
- `RV_m` = monthly realized vol (20d avg)
- `β_d + β_w + β_m = 1.0` (sum-to-one constraint)

**Интуиция:**
- Short-term component (β_d): реакция на текущие шоки
- Mid-term component (β_w): недельные тренды
- Long-term component (β_m): mean reversion к долгосрочной vol

**Two implementations:**
1. **Cell 03 (production):** Dynamic β weights by horizon (hardcoded heuristics)
2. **Cell 07 (backtest):** Fixed β as optimization variables (data-driven)

### 5. EWMA (EXPONENTIALLY WEIGHTED MOVING AVERAGE)
**Baseline метод для сравнения с HAR.**

**Формула:**
```
EWMA_t = α · RV_t + (1 - α) · EWMA_{t-1}
α = 2 / (span + 1)
```

**Реализация в Cell 03:**
- Fractional interpolation между EWMA(1), EWMA(3), EWMA(5), EWMA(10), EWMA(20)
- Short horizon → больше вес на EWMA(1)
- Long horizon → mean reversion к долгосрочной vol

**Проблема EWMA:**
- Не учитывает multi-horizon dependencies (только один компонент)
- Over-reacts к шокам на коротких горизонтах
- Under-reacts к трендам на длинных горизонтах

**HAR преимущество:**
- Балансирует short/mid/long components
- Better forecast accuracy (higher R²)

### 6. SEMIVARIANCE ASYMMETRY
**Определение:** Асимметрия downside vs upside волатильности.

**Формула:**
```
semivar_down = std(returns[returns < 0]) * sqrt(tdy)
semivar_up = std(returns[returns > 0]) * sqrt(tdy)
ratio = semivar_down / semivar_up
```

**Classification:**
- `ratio >= 1.7` → "strong_downside_bias"
- `ratio >= 1.3` → "moderate_downside_bias"
- `ratio >= 1.1` → "mild_downside_bias"
- `ratio < 1.1` → "symmetric"
- `semivar_up = 0.0` → "extreme_downside" (strongest realized bias)

**Применение:**
- Strong downside bias → higher put skew expected → prefer short puts (collect premium)
- Symmetric → balanced skew → Iron Condor friendly

### 7. BASIS VOL (CALENDAR SPREAD RISK)
**Определение:** Волатильность календарного спрэда F1-F2.

**Формула:**
```
basis_log[t] = log(F2_close[t] / F1_close[t])
basis_change[t] = basis_log[t] - basis_log[t-1]
basis_vol_20d = std(basis_change[-20:]) * sqrt(tdy)
```

**Classification:**
- `basis_vol < 1.5%` → "low" (calendar spreads stable)
- `basis_vol < 2.5%` → "elevated" (moderate calendar risk)
- `basis_vol >= 2.5%` → "high" (high calendar risk, avoid BWB/calendars)

**Применение:**
- High basis_vol → календарные спрэды рискованны (F1-F2 spread volatile)
- Low basis_vol → календари стабильны, можно строить complex calendars (BWB)

### 8. VOL OF VOL (VOMMA RISK)
**Определение:** Волатильность волатильности (second-order греческая буква).

**Формула:**
```
YZ_series = yang_zhang_series(df, 5, tdy)
vov_5d = std(YZ_series[-5:])  # NOT annualized again
```

**Classification:**
- `vov < 0.25` → "stable" (vol changes predictable)
- `vov < 0.40` → "elevated" (vol changes moderate)
- `vov >= 0.40` → "high" (vol changes erratic → vomma risk)

**Применение:**
- High VoV → vega hedging сложнее (vol changes непредсказуемы)
- Delay short-vega entry until VoV < 0.35 (entry delay threshold)
- Consider vomma hedging (calendar spreads, diagonal spreads)

### 9. FORWARD REALIZED VOL (GROUND TRUTH FOR BACKTEST)
**Определение:** Forward-looking realized vol для validation forecasts.

**Формула:**
```python
for t in range(len(df) - horizon):
    window_df = df[t+1 : t+1+horizon]
    realized_vol[t] = yang_zhang(window_df, horizon, tdy)
```

**Применение:**
- In-sample: calibration (fit betas)
- Out-of-sample: validation (test forecast accuracy)
- Метрики: R² = 1 - SS_res / SS_tot (ideal: R² > 0.5 для vol forecasts)

### 10. GOOGLE COLAB EXECUTION MODEL
**Критическая особенность проекта:** Весь код работает в Google Colab notebook environment.

**Implications:**
1. **Cell-based execution:** Код загружается через `exec()` в глобальное пространство имен
2. **No imports:** Функции доступны через globals(), не через import statements
3. **Sequential loading:** Ячейки должны выполняться последовательно (01 → 02 → 03 → ...)
4. **Shared globals:** Cell 05 и Cell 07 используют Cell 01-04 через globals()
5. **Restart runtime:** Между разными pipelines нужен "Restart Runtime" (чтобы очистить globals)

**Best practices:**
- Всегда запускай ячейки последовательно
- Используй `%run cell_XX.py` для программного выполнения
- Сохраняй outputs в `/content/` (временная файловая система Colab)
- Mount Google Drive для долгосрочного хранения: `from google.colab import drive; drive.mount('/content/drive')`

---

## 🔑 ИСТОРИЯ ПРОЕКТА: КЛЮЧЕВЫЕ ВЕХИ

### Phase 1: Monolithic Architecture (v1.0-v2.4.0)
- Единый `cell_05_pipeline_main.py` (500+ lines)
- HAR backtest в `cell_07_har_backtest.py` (400+ lines)
- **Проблемы:**
  - Невозможно тестировать отдельные компоненты
  - Дублирование кода (HAR forecast в Cell 03 и Cell 07)
  - Трудно расширять (добавление новых метрик требует редактирования monolith)

### Phase 2: Modular Refactoring (v2.5.0-v2.6.0)
- **Cell 05 разделен на 5 модулей:** 05.1 (metrics), 05.2 (tests), 05.3 (dashboard), 05.4 (forecasts), 05.5 (orchestrator)
- **Cell 07 разделен на 14 модулей:** 07.1 (config), 07.2.01-09b (microservices), 07.3 (viz), 07.4 (main), 07.5 (tests)
- **Архив:** `cell_05_pipeline_main.py` → `arxiv/cell_05_pipeline_main.py`
- **Архив:** `cell_07_har_backtest.py` → `arxiv/cell_07_har_backtest.py`
- **Результат:**
  - ✅ Каждый модуль < 200 lines
  - ✅ Clear separation of concerns
  - ✅ Testable components (Cell 05.2, Cell 07.5)
  - ✅ Easy to extend (новые метрики = новый модуль)

### Phase 3: Enhanced Gap Analysis (v2.6.0)
- **Gap clustering:** Autocorrelation lag-1 для gap occurrence
- **Directional bias:** Up-gaps vs down-gaps asymmetry
- **Emerging regime detection:** 3-level cascade (5d/20d/60d)
- **Amplitude analysis:** Gap size 5d vs 60d
- **Результат:**
  - ✅ Early warning для vol spikes
  - ✅ Directional strategy recommendations (prefer puts/calls)
  - ✅ Risk management для gap-prone periods

### Phase 4: HAR Optimization System (Current)
- **Goal:** Data-driven β-coefficients вместо hardcoded heuristics
- **Approach:** Out-of-sample rolling backtest на MOEX Si F0-F10
- **Expected outcome:**
  - Оптимальные β для каждого horizon (1d, 2d, 3d, 5d, 7d)
  - R² gain HAR vs EWMA (expect: +5-10pp improvement)
  - Production integration в Cell 03

### Future Phases (Roadmap)
- **Phase 5:** Multi-asset support (Gold, Brent, RTS index)
- **Phase 6:** Real-time execution integration (broker API layer, data still via MOEX)
- **Phase 7:** Machine learning features (LSTM, XGBoost для vol forecasting)
- **Phase 8:** Portfolio optimization (optimal position sizing based on vol forecasts)

---

## 📝 КОНЦЕПТУАЛЬНЫЕ НАХОДКИ ИЗ ДИАЛОГОВ

### 1. "Forecast Identity Kill-Signal"
**Проблема:** Если EWMA forecast для F1 идентичен F2 (разница < 1e-7), это означает что EWMA не дифференцирует term structure.

**Причина:** EWMA использует одну и ту же series для F1 и F2, поэтому forecasts совпадают.

**Решение:** Это критический warning (FORECAST_IDENTITY_F1_EQ_F2) → calendar spreads kill-signal.

**Концепция:** Календарные стратегии требуют дифференциации vol между F1 и F2. Если forecast method не видит разницы — strategy invalid.

### 2. "Semivar Up = 0.0 Edge Case"
**Проблема:** Если `semivar_up = 0.0` (все upside returns zero), стандартная классификация через ratio fails (division by zero).

**Решение:** Специальный case: `semivar_up = 0.0` → label="extreme_downside" (strongest realized bias).

**Концепция:** Edge cases в данных должны иметь explicit handling, не generic null/error.

### 3. "Emerging vs Established Gap Regime"
**Проблема:** Как отличить новый gap-режим от постоянного фонового шума?

**Решение:** 3-level cascade:
- High recent frequency (5d)
- Calm historical background (60d)
- Acceleration (5d / 60d >= 2.0)

**Концепция:** Emerging patterns требуют трехуровневого анализа: current state, historical norm, relative change.

### 4. "F0 Stale Check"
**Проблема:** После экспирации F0, 5-дневные метрики (YZ_5d, gap_5d) становятся stale (no new data).

**Решение:** Если `days_since_f0 > 0`, 5d metrics для F0 → null_val("F0_expired_5d_stale").

**Концепция:** Data staleness должна быть explicit в output, не implicit assumption.

### 5. "HAR Dynamic β vs Fixed β"
**Проблема:** Нужны ли разные β для разных горизонтов?

**Решение:**
- Cell 03: Dynamic β (heuristic) — быстро, работает "в среднем"
- Cell 07: Fixed β (optimization) — data-driven, optimal для конкретного asset

**Концепция:** Trade-off между speed (heuristics) и accuracy (optimization). Для production — hybrid: оптимизируй offline, используй online.

### 6. "Percentile Rank для Vol Regime"
**Проблема:** Как определить "elevated" vol в контексте исторической нормы?

**Решение:** Percentile rank в 250-дневном окне:
- < 20th percentile → "low"
- 20-65th → "normal"
- 65-90th → "elevated"
- > 90th → "crisis"

**Концепция:** Absolute thresholds не работают (vol changes over time). Relative percentile robust.

### 7. "Basis Vol для Calendar Risk"
**Проблема:** Когда календарные спрэды становятся рискованными?

**Решение:** Basis vol (volatility of F1-F2 spread):
- Low basis_vol → calendars stable
- High basis_vol → F1-F2 spread volatile → BWB/calendars high risk

**Концепция:** Calendar strategies require stable basis. Monitor basis_vol_20d as leading indicator.

### 8. "VoV Entry Delay Threshold"
**Проблема:** Когда безопасно входить в short-vega позиции?

**Решение:** VoV threshold strategy:
- VoV > 0.40 → критично высокий vomma risk (warning)
- VoV < 0.35 → entry delay cleared (safe to enter)

**Концепция:** Entry timing для vol strategies должен учитывать VoV (second-order risk).

### 9. "Gap Clustering для Position Sizing"
**Проблема:** Gaps идут сериями (autocorr > 0) → margin calls могут cluster.

**Решение:** Gap clustering metric (autocorr lag-1):
- High clustering (> 0.3) → gaps persist in series → increase margin buffer
- Low clustering (< 0.15) → gaps independent → standard margin

**Концепция:** Serial correlation в tail events требует adjusted risk management.

### 10. "Colab Globals() Architecture"
**Проблема:** Почему `run_tests.py` не работает standalone?

**Решение:** Проект designed для Colab cell execution model:
- Functions loaded через `exec()` в globals()
- No explicit imports
- Sequential cell loading required

**Концепция:** Execution environment — first-class design constraint. Code must match environment model.

---

## 🎯 SUMMARY: ЧТО НУЖНО ЗНАТЬ AI АГЕНТАМ

### Архитектура:
- **Dual-path:** Production pipeline (Cell 05) + HAR backtest (Cell 07)
- **Shared core:** Cell 01-04 используются обоими путями
- **Modular:** Каждый модуль < 200 lines, clear separation of concerns

### Execution:
- **Primary environment:** Google Colab (cell-based, shared globals)
- **Data sources:** MOEX ISS API (free, единственный источник данных для всего проекта)
- **Dependencies:** numpy, pandas, matplotlib, scipy, requests

### Ключевые концепции:
1. **Dual-contract approach:** F0/F1/F2 для term structure & Samuelson
2. **Samuelson effect:** Vol spike near expiry (DTE buckets)
3. **Gap regime detection:** 3-level cascade (5d/20d/60d) + clustering + directional bias
4. **HAR-RV model:** Multi-horizon vol forecasting (β_d, β_w, β_m)
5. **EWMA baseline:** Fractional interpolation для сравнения
6. **Semivariance asymmetry:** Downside vs upside vol → skew preference
7. **Basis vol:** Calendar spread risk metric
8. **VoV:** Vomma risk (vol of vol) → entry timing
9. **Forward realized vol:** Ground truth для backtest validation
10. **Colab globals:** Cell-based execution model

### Текущий этап:
- **Goal:** Оптимизация β-коэффициентов HAR через backtest на MOEX Si
- **Status:** Модульная архитектура завершена, self-tests pass, готов к запуску первого backtest
- **Next steps:** Инициализация MOEX клиента, загрузка данных F0-F10 через ISS API, запуск optimization

### Где искать код:
- **Production pipeline:** cell_05_5_pipeline.py (orchestrator), cell_05_1-4 (modules)
- **HAR backtest:** cell_07_4_main.py (entrypoint), cell_07_2_01-09b (microservices)
- **Core logic:** cell_01-04 (config, MOEX, estimators, analytics)
- **Archived monoliths:** arxiv/cell_05_pipeline_main.py, arxiv/cell_07_har_backtest.py

---

**END OF GUIDE**
