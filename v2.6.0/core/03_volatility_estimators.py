"""
03_volatility_estimators.py
Cell ID: lw7Cwpd3ZRFy
Exported: 2026-04-16T10:12:23.218584
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 3/5
Data Loading, Validation, Volatility Estimators
"""

# ════════════════════════════════════════════════════════════════
# OHLCV DOWNLOAD & VALIDATION
# ════════════════════════════════════════════════════════════════
def validate_ohlcv(df: pd.DataFrame, secid: str) -> pd.DataFrame:
    """Валидирует и нормализует OHLCV DataFrame."""
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"{secid}: missing columns {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = (
        out.dropna(subset=["date", "open", "high", "low", "close"])
        .query("open > 0 and high > 0 and low > 0 and close > 0 and high >= low")
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    return out


def moex_load_candles(
    client: MoexClient,
    cfg: MoexConfig,
    secid: str,
    date_from: str,
    date_till: str,
) -> pd.DataFrame:
    """Загружает дневные свечи с пагинацией."""
    empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frames: list[pd.DataFrame] = []
    start = 0

    while True:
        url = f"{cfg.base_url}/engines/futures/markets/forts/boards/{cfg.board}/securities/{secid}/candles.json"
        params = {"from": date_from, "till": date_till, "interval": 24, "start": start, "iss.meta": "off"}
        try:
            r = client.get(url, params)
        except requests.RequestException:
            log.error("Failed to load candles for %s", secid)
            return empty

        block = r.json().get("candles", {})
        data = block.get("data", [])
        cols = block.get("columns", [])
        if not data:
            break

        chunk = pd.DataFrame(data, columns=cols)
        frames.append(chunk)
        start += len(chunk)
        if len(chunk) < cfg.page_size:
            break

    if not frames:
        log.debug("No candles: %s (%s → %s)", secid, date_from, date_till)
        return empty

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.upper() for c in df.columns]
    dt_col = next((c for c in ("BEGIN", "TRADEDATE", "DATE", "END") if c in df.columns), None)
    if not dt_col:
        raise DataValidationError(f"No date column for {secid}: {list(df.columns)}")

    ts = pd.to_datetime(df[dt_col], errors="coerce")
    normalized = pd.DataFrame({
        "date": ts.dt.date,
        "open":   pd.to_numeric(df["OPEN"],   errors="coerce"),
        "high":   pd.to_numeric(df["HIGH"],   errors="coerce"),
        "low":    pd.to_numeric(df["LOW"],    errors="coerce"),
        "close":  pd.to_numeric(df["CLOSE"],  errors="coerce"),
        "volume": pd.to_numeric(df.get("VOLUME", df.get("VALUE", pd.Series(dtype="float64"))), errors="coerce"),
    })
    result = validate_ohlcv(normalized, secid)
    result["date"] = pd.to_datetime(result["date"])
    log.debug("Loaded %d bars for %s", len(result), secid)
    return result


def load_all_candles(
    contracts: dict[str, ContractMeta],
    today: date,
    client: MoexClient,
    cfg: MoexConfig,
    vol_cfg: VolConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Загружает OHLCV для всех контрактов."""
    today_str = today.isoformat()
    ohlcv: dict[str, pd.DataFrame] = {}
    samuelson_sources: dict[str, pd.DataFrame] = {}

    for role in ("F1", "F2"):
        t = contracts[role].ticker
        start = (today - timedelta(days=vol_cfg.history_days)).isoformat()
        log.info("Downloading %s: %s", role, t)
        ohlcv[role] = moex_load_candles(client, cfg, t, start, today_str)
        log.info("  %d bars", len(ohlcv[role]))

    hist_roles = sorted([r for r in contracts if r not in ("F1", "F2")], key=lambda x: contracts[x].expiry)
    for role in hist_roles:
        t = contracts[role].ticker
        exp = contracts[role].expiry
        start = (exp - timedelta(days=vol_cfg.active_phase_days)).isoformat()
        end = exp.isoformat()
        log.info("Downloading %s: %s (active phase)", role, t)
        df = moex_load_candles(client, cfg, t, start, end)
        samuelson_sources[role] = df
        log.info("  %d bars", len(df))

    if "F0" in contracts:
        ohlcv["F0"] = samuelson_sources.get("F0", pd.DataFrame())

    return ohlcv, samuelson_sources


# ════════════════════════════════════════════════════════════════
# VOLATILITY ESTIMATORS
# ════════════════════════════════════════════════════════════════
def yang_zhang_series(df: pd.DataFrame, window: int, tdy: int = 252) -> pd.Series:
    """Yang-Zhang estimator (аннуализированный)."""
    if len(df) < window:
        return pd.Series(dtype=float, index=df.index if hasattr(df, 'index') else None)

    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"]  / df["open"])
    log_co = np.log(df["close"]/ df["open"])

    # Overnight return: первая строка будет NaN, но это нормально для rolling
    log_oc = np.log(df["open"] / df["close"].shift(1))

    # Rogers-Satchell компонента
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34

    # CRITICAL FIX: Use min_periods=n instead of fillna(0) to properly handle first NaN
    # This avoids numerical instability from forcing zero variance in the first window
    var_o  = log_oc.rolling(n, min_periods=n).var(ddof=1)
    var_c  = log_co.rolling(n, min_periods=n).var(ddof=1)
    var_rs = rs.rolling(n, min_periods=n).mean()

    # Combine variance components
    variance_combined = var_o + k * var_c + (1 - k) * var_rs

    # CRITICAL: Convert to numpy array BEFORE clipping to avoid pandas type dispatch issues
    variance_np = variance_combined.to_numpy(dtype=np.float64, copy=True)

    # Replace inf/nan with 0 for safe sqrt calculation
    variance_np = np.where(np.isfinite(variance_np), variance_np, 0.0)

    # Clip negative variance (может возникнуть из-за округлений)
    variance_np = np.clip(variance_np, 0, None)

    # Calculate volatility and convert back to Series
    result = pd.Series(np.sqrt(variance_np * tdy), index=df.index)

    return result


def cc_vol_series(df: pd.DataFrame, window: int, tdy: int = 252) -> pd.Series:
    """Close-to-Close volatility (аннуализированная)."""
    return np.log(df["close"] / df["close"].shift(1)).rolling(window).std(ddof=1) * math.sqrt(tdy)


def latest(s: pd.Series) -> Optional[float]:
    if s is None or s.empty: return None
    v = s.iloc[-1]
    return float(v) if pd.notna(v) and np.isfinite(v) else None

def at_offset(s: pd.Series, offset: int) -> Optional[float]:
    if s is None or s.empty: return None
    idx = len(s) - 1 - offset
    if idx < 0: return None
    v = s.iloc[idx]
    return float(v) if pd.notna(v) and np.isfinite(v) else None


def semivariance_ann(df: pd.DataFrame, window: int, side: str, tdy: int = 252) -> tuple[Optional[float], int]:
    """Полудисперсия (аннуализированная)."""
    if len(df) < window + 1:
        return None, 0
    tail = np.log(df["close"] / df["close"].shift(1)).iloc[-window:]
    filt = tail[tail < 0] if side == "down" else tail[tail > 0]
    n = len(filt)
    if n < 2:
        return 0.0, n
    return math.sqrt(max(float(filt.var(ddof=1)) * tdy, 0.0)), n


def percentile_rank_250(s: pd.Series) -> Optional[float]:
    """Перцентильный ранг последнего значения в 250-дневном окне."""
    s = s.dropna()
    if len(s) < 30: return None
    w = s.iloc[-250:] if len(s) >= 250 else s
    return float((w < w.iloc[-1]).sum()) / len(w) * 100.0


def realized_skew(df: pd.DataFrame, window: int) -> Optional[float]:
    s = np.log(df["close"] / df["close"].shift(1)).rolling(window).skew()
    return latest(s)

def realized_kurtosis(df: pd.DataFrame, window: int) -> Optional[float]:
    k = np.log(df["close"] / df["close"].shift(1)).rolling(window).kurt()
    return latest(k)


def vol_of_vol(yz_series: pd.Series, window: int) -> Optional[float]:
    """VoV: std дневных YZ-значений. НЕ аннуализируется повторно."""
    valid = yz_series.dropna()
    if len(valid) < window + 1: return None
    vov = valid.rolling(window).std(ddof=1)
    return float(vov.iloc[-1]) if pd.notna(vov.iloc[-1]) else None


def gap_statistics(df: pd.DataFrame, window: int, cfg: PipelineConfig) -> dict[str, Any]:
    """Overnight gap анализ с sigma-распределением + percentile rank + clustering + directional bias."""
    if len(df) < window + 1:
        return null_val(f"insufficient_data_need_{window+1}_bars")

    lr = np.log(df["close"] / df["close"].shift(1))
    gaps_abs = np.abs(np.log(df["open"] / df["close"].shift(1)))
    gaps_signed = np.log(df["open"] / df["close"].shift(1))  # Signed gaps для directional bias
    rolling_std = lr.rolling(window).std(ddof=1)

    valid_mask = gaps_abs.notna() & rolling_std.notna() & (rolling_std > 0)
    valid_gaps = gaps_abs[valid_mask]
    valid_gaps_signed = gaps_signed[valid_mask]
    std_vals   = rolling_std[valid_mask]

    if len(valid_gaps) == 0:
        return null_val("no_valid_gap_data")

    normalized = valid_gaps / std_vals
    sigma_bins   = cfg.gap_sigma_bins
    sigma_labels = cfg.gap_sigma_labels

    distribution: dict[str, Any] = {}
    for i, (lo, hi) in enumerate(zip(sigma_bins[:-1], sigma_bins[1:])):
        mask  = (normalized >= lo) & (normalized < hi)
        count = int(mask.sum())
        item: dict[str, Any] = {"count": count, "frequency_pct": r_pct(count / len(normalized) * 100)}
        if count > 0:
            bg = valid_gaps[mask]
            item.update({
                "amplitude_avg": r_vol(float(bg.mean())),
                "amplitude_max": r_vol(float(bg.max())),
                "amplitude_p90": r_vol(float(np.percentile(bg, 90))),
            })
        distribution[sigma_labels[i]] = item

    sig_mask = normalized >= 1.0
    n_sig    = int(sig_mask.sum())

    base = {"gap_sigma_distribution": distribution, "rolling_std_mean": r_vol(float(std_vals.mean())), "available": True}

    # ═══ NEW METRICS ═══════════════════════════════════════════════

    # 1. GAP CLUSTERING (autocorrelation)
    gap_binary = (normalized >= 1.0).astype(int)  # 1 = gap occurred, 0 = no gap
    if len(gap_binary) >= 2:
        # Lag-1 autocorrelation для gap occurrences
        # Проверяем stddev чтобы избежать RuntimeWarning при константной серии
        gap_series = pd.Series(gap_binary)
        if gap_series.std(ddof=1) > 0:
            gap_autocorr = float(gap_series.autocorr(lag=1))
            gap_autocorr = gap_autocorr if not np.isnan(gap_autocorr) else 0.0
        else:
            # Константная серия (все 0 или все 1) → autocorr не определена
            gap_autocorr = 0.0
    else:
        gap_autocorr = 0.0

    clustering_interpretation = (
        "high_clustering" if gap_autocorr >= 0.3 else
        "moderate_clustering" if gap_autocorr >= 0.15 else
        "low_clustering" if gap_autocorr >= 0.0 else
        "negative_clustering"
    )

    # 2. DIRECTIONAL BIAS (up-gaps vs down-gaps)
    up_gaps = (valid_gaps_signed[sig_mask] > 0).sum() if n_sig > 0 else 0
    down_gaps = (valid_gaps_signed[sig_mask] < 0).sum() if n_sig > 0 else 0

    up_gap_freq = r_pct(up_gaps / len(valid_gaps) * 100) if len(valid_gaps) > 0 else 0.0
    down_gap_freq = r_pct(down_gaps / len(valid_gaps) * 100) if len(valid_gaps) > 0 else 0.0

    directional_bias = (
        "strong_upside_bias" if up_gaps > 0 and up_gaps / max(down_gaps, 1) >= 2.0 else
        "strong_downside_bias" if down_gaps > 0 and down_gaps / max(up_gaps, 1) >= 2.0 else
        "moderate_upside_bias" if up_gaps > down_gaps and up_gaps / max(down_gaps, 1) >= 1.5 else
        "moderate_downside_bias" if down_gaps > up_gaps and down_gaps / max(up_gaps, 1) >= 1.5 else
        "symmetric"
    )

    # ═══ ENHANCED METRICS OBJECT ═══════════════════════════════════

    enhanced_metrics = {
        "gap_clustering": {
            "autocorr_lag1": round(gap_autocorr, 3),
            "interpretation": clustering_interpretation,
            "llm_hint": (
                "Gaps cluster in series: after gap, expect another gap" if gap_autocorr >= 0.3 else
                "Moderate gap clustering: weak serial correlation" if gap_autocorr >= 0.15 else
                "Gaps are independent: no serial pattern"
            )
        },
        "directional_bias": {
            "up_gap_frequency_pct": up_gap_freq,
            "down_gap_frequency_pct": down_gap_freq,
            "up_gap_count": int(up_gaps),
            "down_gap_count": int(down_gaps),
            "bias_label": directional_bias,
            "llm_hint": (
                f"Strong upside gap bias: prefer put-side strategies" if "upside" in directional_bias and "strong" in directional_bias else
                f"Strong downside gap bias: prefer call-side strategies" if "downside" in directional_bias and "strong" in directional_bias else
                f"Symmetric gaps: no directional preference"
            )
        }
    }

    # ═══ FINAL OUTPUT ═══════════════════════════════════════════════

    if n_sig == 0:
        return {
            "frequency_pct": 0.0,
            "amplitude_avg": 0.0,
            "amplitude_max": 0.0,
            "gap_quantiles": {f"p{q}": 0.0 for q in (10, 25, 50, 75, 90, 95)},
            "n_gaps": 0,
            **base,
            **enhanced_metrics
        }

    qualifying = valid_gaps[sig_mask]
    quantiles = {f"p{q}": r_vol(float(np.percentile(qualifying, q))) for q in (10, 25, 50, 75, 90, 95)}

    return {
        "frequency_pct": r_pct(n_sig / len(valid_gaps) * 100),
        "amplitude_avg": r_vol(float(qualifying.mean())),
        "amplitude_max": r_vol(float(qualifying.max())),
        "gap_quantiles": quantiles,
        "n_gaps": n_sig,
        **base,
        **enhanced_metrics
    }


def calc_har_rv_forecast(
    rv_1d: pd.Series,
    rv_5d: pd.Series,
    rv_20d: pd.Series,
    horizon_days: float,
    filter_gaps: bool = False,
    gap_mask: Optional[pd.Series] = None,
) -> dict[str, Any]:
    """HAR-RV forecast (Heterogeneous AutoRegressive Realized Volatility).

    Corsi (2009) model: RV_t+h = β₀ + β_d×RV_d + β_w×RV_w + β_m×RV_m

    Args:
        rv_1d: Daily RV series (YZ or CC)
        rv_5d: 5-day rolling avg RV
        rv_20d: 20-day rolling avg RV
        horizon_days: Forecast horizon (fractional days allowed)
        filter_gaps: If True, exclude gap days from RV components
        gap_mask: Boolean mask (True = gap day, False = normal day)

    Returns:
        dict with HAR_forecast, coefficients, method metadata
    """
    # Validate inputs
    if rv_1d is None or rv_5d is None or rv_20d is None:
        return null_val("missing_rv_series")

    rv_1d_valid = rv_1d.dropna()
    rv_5d_valid = rv_5d.dropna()
    rv_20d_valid = rv_20d.dropna()

    if len(rv_1d_valid) < 20 or len(rv_5d_valid) < 20 or len(rv_20d_valid) < 20:
        return null_val("insufficient_har_history_need_20d")

    # Align series (use common index)
    common_idx = rv_1d_valid.index.intersection(rv_5d_valid.index).intersection(rv_20d_valid.index)
    if len(common_idx) < 20:
        return null_val("insufficient_aligned_data_for_har")

    rv_d = rv_1d_valid.loc[common_idx]
    rv_w = rv_5d_valid.loc[common_idx]
    rv_m = rv_20d_valid.loc[common_idx]

    # Gap filtering (optional)
    if filter_gaps and gap_mask is not None:
        gap_mask_aligned = gap_mask.reindex(common_idx, fill_value=False)
        clean_mask = ~gap_mask_aligned
        if clean_mask.sum() < 15:
            # Fallback: не хватает clean данных, используем все
            filter_gaps = False
        else:
            rv_d = rv_d[clean_mask]
            rv_w = rv_w[clean_mask]
            rv_m = rv_m[clean_mask]

    # HAR coefficients (calibrated for FX/currency futures, short-term horizons)
    # Standard HAR weights (Corsi 2009): β_d ≈ 0.3-0.4, β_w ≈ 0.3-0.4, β_m ≈ 0.2-0.3
    # For ≤7 DTE horizon: increase β_d weight, reduce β_m
    if horizon_days <= 1:
        beta_d, beta_w, beta_m = 0.60, 0.30, 0.10
    elif horizon_days <= 5:
        beta_d, beta_w, beta_m = 0.40, 0.40, 0.20
    elif horizon_days <= 10:
        beta_d, beta_w, beta_m = 0.30, 0.40, 0.30
    else:
        # Long horizon: standard HAR
        beta_d, beta_w, beta_m = 0.25, 0.35, 0.40

    # Extract latest values
    latest_d = float(rv_d.iloc[-1])
    latest_w = float(rv_w.iloc[-1])
    latest_m = float(rv_m.iloc[-1])

    # HAR forecast (no intercept for simplicity, assume mean-zero)
    har_forecast = beta_d * latest_d + beta_w * latest_w + beta_m * latest_m

    # Metadata
    method_note = "HAR-RV (Corsi 2009) with horizon-adaptive weights"
    if filter_gaps:
        method_note += " + gap-filtered components"

    return {
        "forecast_RV": r_vol(har_forecast),
        "method": "HAR-RV",
        "coefficients": {
            "beta_daily": round(beta_d, 3),
            "beta_weekly": round(beta_w, 3),
            "beta_monthly": round(beta_m, 3)
        },
        "components": {
            "RV_1d": r_vol(latest_d),
            "RV_5d_avg": r_vol(latest_w),
            "RV_20d_avg": r_vol(latest_m)
        },
        "horizon_days": round(horizon_days, 2),
        "gap_filtered": filter_gaps,
        "method_note": method_note,
        "available": True
    }


def calc_session_aligned_rv_forecast(
    yz_series: pd.Series, current_time: datetime, target_datetime: datetime,
    cfg: MoexConfig, ewma_spans: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, Any]:
    """Дробный EWMA-прогноз RV."""
    valid = yz_series.dropna()
    if len(valid) < 10:
        return null_val("insufficient_history_for_fractional_forecast")

    full_days   = max(0, (target_datetime.date() - current_time.date()).days)
    end_min     = cfg.session_end_minutes
    day_minutes = cfg.session_duration_minutes

    if full_days == 0:
        start_min = current_time.hour * 60 + current_time.minute
        frac = max(0.0, (end_min - start_min) / day_minutes)
    else:
        frac = end_min / day_minutes

    horizon = full_days + frac
    ewma_map = {w: float(valid.ewm(span=w, adjust=False).mean().iloc[-1]) for w in ewma_spans if len(valid) >= w}
    if not ewma_map:
        return null_val("insufficient_ewma_windows")

    # Адаптивная интерполация с учетом mean reversion для длинных горизонтов
    if horizon <= 1:
        forecast = ewma_map[1]
    elif horizon <= 3:
        forecast = ewma_map[1] * 0.7 + ewma_map.get(3, ewma_map[1]) * 0.3
    elif horizon <= 5:
        forecast = ewma_map.get(3, ewma_map.get(5, valid.iloc[-1])) * 0.6 + ewma_map.get(5, valid.iloc[-1]) * 0.4
    elif horizon <= 10:
        # Для горизонтов 5-10 дней: плавный переход к EWMA(10)
        forecast = ewma_map.get(5, valid.iloc[-1]) * 0.3 + ewma_map.get(10, valid.iloc[-1]) * 0.7
    elif horizon <= 20:
        # Для горизонтов 10-20 дней: микс EWMA(10) и EWMA(20)
        forecast = ewma_map.get(10, valid.iloc[-1]) * 0.5 + ewma_map.get(20, valid.iloc[-1]) * 0.5
    else:
        # Для длинных горизонтов (>20 дней): EWMA(20) с mean reversion к долгосрочной исторической vol
        long_term_vol = float(valid.mean())  # Долгосрочная средняя vol
        short_term_vol = ewma_map.get(20, float(valid.iloc[-1]))
        # Mean reversion weight зависит от горизонта: чем дальше, тем сильнее тянет к mean
        reversion_weight = min(0.5, (horizon - 20) / 100)  # До 50% веса на long-term для горизонтов >70 дней
        forecast = short_term_vol * (1 - reversion_weight) + long_term_vol * reversion_weight

    return {"target_datetime": target_datetime.isoformat(), "fractional_trading_days": round(horizon, 2),
            "forecast_RV": r_vol(forecast), "method": "fractional_EWMA_interpolated",
            "llm_hint": "Compare with ATM_IV of matching DTE. If IV − Forecast_RV < 1.5pp, premium insufficient for short-vega.",
            "available": True}

print("Cell 3/5: Data Loading & Estimators загружены")
