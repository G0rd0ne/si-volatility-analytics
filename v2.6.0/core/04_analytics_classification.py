"""
04_analytics_classification.py
Cell ID: IkEQx5x8ZYpV
Exported: 2026-04-16T10:12:23.218617
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 4/5
Analytics, Classification, Warnings
"""

# ════════════════════════════════════════════════════════════════
# CLASSIFICATION HELPERS
# ════════════════════════════════════════════════════════════════
def classify_skew(v: Optional[float], thr: ThresholdConfig) -> str:
    if v is None: return "unknown"
    if v < thr.skew_left:  return "left"
    if v > thr.skew_right: return "right"
    return "symmetric"

def classify_kurtosis(v: Optional[float], thr: ThresholdConfig) -> str:
    if v is None: return "unknown"
    if v < thr.kurtosis_platykurtic: return "platykurtic"
    if v > thr.kurtosis_leptokurtic: return "leptokurtic"
    return "normal"

def classify_semivar_asymmetry(
    semivar_down: Optional[float], semivar_up: Optional[float],
    nd: Any, nu: Any, thr: ThresholdConfig,
) -> dict[str, Any]:
    """Классифицирует асимметрию полудисперсий. ИСПРАВЛЕНИЕ v2.6.0: semivar_up=0→extreme_downside."""
    if semivar_down is None:
        return null_val("semivar_down_unavailable")

    if isinstance(semivar_up, float) and semivar_up == 0.0 and semivar_down > 0.0:
        return {"label": "extreme_downside", "down_up_ratio": None, "reason": "semivar_up_zero_strongest_realized_bias",
                "semivar_n_down": nd if not isinstance(nd, dict) else None, "semivar_n_up": nu if not isinstance(nu, dict) else None,
                "available": True}

    if semivar_up is None or (isinstance(semivar_up, float) and semivar_up <= 0.0):
        return null_val("semivar_up_zero_or_unavailable")

    ratio = semivar_down / semivar_up
    if ratio >= thr.semivar_moderate:     label = "strong_downside_bias"
    elif ratio >= thr.semivar_mild:       label = "moderate_downside_bias"
    elif ratio >= thr.semivar_absent:     label = "mild_downside_bias"
    else:                                  label = "symmetric"

    return {"label": label, "down_up_ratio": r_ratio(ratio),
            "semivar_n_down": nd if not isinstance(nd, dict) else None,
            "semivar_n_up":   nu if not isinstance(nu, dict) else None, "available": True}


def classify_basis_vol(v: Optional[float], thr: ThresholdConfig) -> str:
    if v is None: return "unknown"
    if v < thr.basis_vol_low:      return "low"
    if v < thr.basis_vol_elevated: return "elevated"
    return "high"


# ════════════════════════════════════════════════════════════════
# SAMUELSON
# ════════════════════════════════════════════════════════════════
def build_samuelson_daily_1_10(expired_sources: dict[str, pd.DataFrame], contracts: dict[str, ContractMeta], tdy: int = 252) -> dict[str, Any]:
    """Samuelson short table DTE 1-10 с shrinkage."""
    all_ret: list[tuple[int, float]] = []
    for role, df in expired_sources.items():
        if role not in contracts or df is None or df.empty: continue
        exp = contracts[role].expiry
        c = df.copy()
        c["date"] = pd.to_datetime(c["date"], errors="coerce")
        c["dte"]  = (pd.Timestamp(exp) - c["date"]).dt.days
        c["lr"]   = np.log(c["close"] / c["close"].shift(1))
        valid = c.dropna(subset=["lr", "dte"])
        valid = valid[(valid["dte"] >= 1) & (valid["dte"] <= 10)]
        all_ret.extend((int(r["dte"]), float(r["lr"])) for _, r in valid.iterrows())

    if not all_ret:
        return null_val("no_expired_data_for_daily_samuelson")

    ret_df = pd.DataFrame(all_ret, columns=["dte", "lr"])
    bucket_mean = ret_df["lr"].abs().mean() * math.sqrt(tdy)
    result: dict[str, Any] = {}

    for dte in range(1, 11):
        sub = ret_df[ret_df["dte"] == dte]
        n = len(sub)
        if n < 3:
            result[f"DTE_{dte}"] = null_val(f"only_{n}_observations")
            continue
        std = float(sub["lr"].std(ddof=1)) * math.sqrt(tdy)
        if n < 10: std = std * 0.7 + bucket_mean * 0.3
        result[f"DTE_{dte}"] = {
            "mean_vol": r_vol(std), "std_bucket": r_vol(std),
            "p10": r_vol(float(np.percentile(sub["lr"].abs(), 10) * math.sqrt(tdy))),
            "p90": r_vol(float(np.percentile(sub["lr"].abs(), 90) * math.sqrt(tdy))),
            "n_obs": n, "reliability": "high" if n >= 30 else ("medium" if n >= 15 else "low"),
            "llm_hint": f"Historical vol norm for DTE={dte}. Use for tenor-matched IV comparison."}

    # Интерполяция для пропущенных DTE (4, 5)
    for dte in range(1, 11):
        if f"DTE_{dte}" not in result or is_null(result[f"DTE_{dte}"]):
            # Найти ближайшие заполненные DTE
            prev_dte = next((d for d in range(dte-1, 0, -1) if f"DTE_{d}" in result and not is_null(result[f"DTE_{d}"])), None)
            next_dte = next((d for d in range(dte+1, 11) if f"DTE_{d}" in result and not is_null(result[f"DTE_{d}"])), None)

            if prev_dte and next_dte:
                prev_val = result[f"DTE_{prev_dte}"]["mean_vol"]
                next_val = result[f"DTE_{next_dte}"]["mean_vol"]
                steps = next_dte - prev_dte
                offset = dte - prev_dte
                interpolated = prev_val - offset * (prev_val - next_val) / steps

                result[f"DTE_{dte}"] = {
                    "mean_vol": r_vol(interpolated),
                    "std_bucket": r_vol(interpolated * 0.5),  # Conservative estimate
                    "p10": r_vol(interpolated * 0.3),
                    "p90": r_vol(interpolated * 1.5),
                    "n_obs": 0,
                    "reliability": "interpolated",
                    "llm_hint": f"Interpolated between DTE_{prev_dte} and DTE_{next_dte}. Use with caution.",
                    "interpolation_note": f"Linear interpolation: ({prev_dte}→{next_dte}), no historical observations"
                }

    result["methodology"] = "Daily aggregation. Shrinkage applied for n<10. Linear interpolation for missing DTE."
    result["llm_hint"] = "Compare current IV(DTE) with mean_vol. If IV > p90, short vol has strong edge. Interpolated values marked explicitly."
    return result


def calc_samuelson_status(yz_20d_f1: Optional[float], dte_f1: int, samuelson_daily: dict, thr: ThresholdConfig) -> dict[str, Any]:
    """Реальный расчёт samuelson_status. ИСПРАВЛЕНИЕ КРИТИЧНО-6."""
    if yz_20d_f1 is None or dte_f1 > 10:
        return {"status": "normal_for_dte", "z_score": None, "deviation_pp": None, "note": "DTE > 10: Samuelson short table не применима"}

    bucket = samuelson_daily.get(f"DTE_{dte_f1}")
    if is_null(bucket) or not isinstance(bucket, dict):
        return {"status": "normal_for_dte", "z_score": None, "deviation_pp": None, "note": f"DTE_{dte_f1} bucket unavailable"}

    mean_vol  = bucket.get("mean_vol")
    std_bucket = bucket.get("std_bucket")
    if mean_vol is None or std_bucket is None or std_bucket == 0:
        return {"status": "normal_for_dte", "z_score": None, "deviation_pp": None, "note": "Bucket statistics incomplete"}

    z = (yz_20d_f1 - mean_vol) / std_bucket
    deviation_pp = (yz_20d_f1 - mean_vol) * 100.0

    if z > 1.5:   status = "stressed"
    elif z < -1.5: status = "complacent"
    else:          status = "normal_for_dte"

    return {"status": status, "z_score": r_z(z), "deviation_pp": r_pct(deviation_pp),
            "note": f"YZ_20d vs DTE_{dte_f1} bucket (n={bucket.get('n_obs', '?')})"}


# ════════════════════════════════════════════════════════════════
# VOLUME ANALYSIS
# ════════════════════════════════════════════════════════════════
def avg_volume_nd(df: pd.DataFrame, n: int, ref: date) -> Optional[float]:
    if df is None or df.empty: return None
    try:
        past = df[df["date"].dt.date < ref]
        return float(past["volume"].iloc[-n:].mean()) if len(past) >= n else None
    except: return None

def activity_weight(df: pd.DataFrame, lb: int = 20) -> float:
    if df is None or df.empty: return 0.0
    tail = df.iloc[-lb:] if len(df) >= lb else df
    return round(float((tail["volume"] > 0).sum()) / len(tail), 4)

def f2_reliability_label(volume_ratio: Optional[float], aw: float) -> str:
    if volume_ratio is None or aw < 0.5: return "reduced"
    if volume_ratio < 10:  return "high"
    if volume_ratio <= 30: return "adequate"
    return "reduced"


def calc_historical_volume_ratio_by_dte(
    ohlcv: dict[str, pd.DataFrame], contracts: dict[str, ContractMeta],
    samuelson_sources: dict[str, pd.DataFrame], today: date,
    tgt_f1: int, tgt_f2: int, vol_cfg: VolConfig,
) -> dict[str, Any]:
    """F2/F1 объёмный ratio vs исторические аналоги. ИСПРАВЛЕНИЕ v2.6.0: DataFrame boolean context fix."""
    lookback = vol_cfg.lookback_candles
    tol      = vol_cfg.dte_tolerance

    v1 = avg_volume_nd(ohlcv.get("F1", pd.DataFrame()), lookback, today)
    v2 = avg_volume_nd(ohlcv.get("F2", pd.DataFrame()), lookback, today)
    cur_pct: Any = round((v2 / v1) * 100, 1) if (v1 and v2 and v1 > 0) else null_val("insufficient_current_volume")

    all_sorted = sorted([(contracts[r].expiry, r) for r in contracts if isinstance(contracts[r], ContractMeta)], key=lambda x: x[0])
    next_role_map: dict[str, Optional[str]] = {}
    for i, (_, role) in enumerate(all_sorted):
        next_role_map[role] = all_sorted[i + 1][1] if i + 1 < len(all_sorted) else None

    hist: list[float] = []
    for role, df1 in samuelson_sources.items():
        if df1 is None or df1.empty or role not in contracts: continue
        exp1 = contracts[role].expiry
        nxt  = next_role_map.get(role)
        if not nxt or nxt not in contracts: continue

        df2_from_samuelson = samuelson_sources.get(nxt)
        df2_from_ohlcv = ohlcv.get(nxt)
        df2 = None
        if df2_from_samuelson is not None and not df2_from_samuelson.empty:
            df2 = df2_from_samuelson
        elif df2_from_ohlcv is not None and not df2_from_ohlcv.empty:
            df2 = df2_from_ohlcv
        if df2 is None or df2.empty: continue
        exp2 = contracts[nxt].expiry

        try:
            d1 = df1.copy(); d1["date"] = pd.to_datetime(d1["date"], errors="coerce")
            d1["dte"] = (pd.Timestamp(exp1) - d1["date"]).dt.days; d1["v"] = d1["volume"]
            d2 = df2.copy(); d2["date"] = pd.to_datetime(d2["date"], errors="coerce")
            d2["dte"] = (pd.Timestamp(exp2) - d2["date"]).dt.days; d2["v"] = d2["volume"]
            m = pd.merge(d1[["date","dte","v"]].rename(columns={"dte":"d1","v":"v1"}),
                         d2[["date","dte","v"]].rename(columns={"dte":"d2","v":"v2"}), on="date", how="inner")
            mask = (abs(m["d1"] - tgt_f1) <= tol) & (abs(m["d2"] - tgt_f2) <= tol)
            cand = m[mask].sort_values("date").tail(lookback)
            if len(cand) < lookback: continue
            avg1, avg2 = cand["v1"].mean(), cand["v2"].mean()
            if avg1 > 0 and pd.notna(avg2): hist.append(round((avg2 / avg1) * 100, 1))
        except Exception as exc:
            log.debug("Vol ratio analog error for %s: %s", role, exc)

    if not hist:
        return {"current_ratio_pct": cur_pct, "historical_avg_pct": null_val("no_historical_analogs"),
                "historical_std_pct": null_val("insufficient_history"), "z_vs_history": null_val("insufficient_history"),
                "interpretation": "Недостаточно исторических аналогов", "note": f"DTE≈{tgt_f1}/{tgt_f2}",
                "llm_hint": "Lack of historical volume context; rely on current activity_weight."}

    h_mean = round(float(np.mean(hist)), 1)
    h_std  = round(float(np.std(hist, ddof=1)), 1) if len(hist) > 1 else 0.0
    z: Any = round((cur_pct - h_mean) / h_std, 2) if (h_std > 0 and isinstance(cur_pct, float)) else null_val("insufficient_std")

    interp = "соответствует норме" if is_null(z) else ("аномально неликвиден" if z < -1.5 else ("аномально ликвиден" if z > 1.5 else "в пределах нормы"))
    cur_str = f"{cur_pct:.1f}" if isinstance(cur_pct, float) else "N/A"
    z_str   = f"{z:.2f}" if isinstance(z, float) else "N/A"

    return {"current_ratio_pct": cur_pct, "historical_avg_pct": h_mean, "historical_std_pct": h_std, "z_vs_history": z,
            "interpretation": f"F2 объём = {cur_str}% от F1 ({interp}, ист.ср={h_mean}%, z={z_str})",
            "note": f"DTE≈{tgt_f1}/{tgt_f2}, окно={lookback}д, допуск±{tol}д",
            "llm_hint": "Use z_vs_history for liquidity context. z<-1.5 implies execution risk for calendars/BWB."}


# ════════════════════════════════════════════════════════════════
# EMERGING GAP REGIME DETECTION
# ════════════════════════════════════════════════════════════════
def detect_emerging_gap_regime(gap_analysis: dict, thr: ThresholdConfig) -> dict[str, Any]:
    """Детектирует начало нового gap-режима через трехуровневый анализ (5d/20d/60d) с градиентом.

    NEW in this version:
    - Percentile rank для gap_5d относительно distribution_60d
    - Gap size analysis (средняя амплитуда 5d vs 60d)
    - Gap clustering metric (autocorrelation из gap_5d)
    - Directional bias integration

    Returns:
        dict: {
            "emerging": bool,
            "regime_type": "established" | "recent_spike" | "gradual_buildup" | "none",
            "confidence": "high" | "medium" | "low",
            "severity": "high" | "moderate" | "low" | None,
            "persistence_estimate": str,
            "metrics": {...},
            "reason": str
        }
    """
    f1_gap = gap_analysis.get("F1_current", {})

    g5  = f1_gap.get("5d", {})
    g20 = f1_gap.get("20d", {})
    g60 = f1_gap.get("60d", {})

    freq_5  = g5.get("frequency_pct")  if isinstance(g5, dict)  else None
    freq_20 = g20.get("frequency_pct") if isinstance(g20, dict) else None
    freq_60 = g60.get("frequency_pct") if isinstance(g60, dict) else None

    # Edge cases
    if freq_5 is None or freq_20 is None or freq_60 is None:
        return {
            "emerging": False,
            "regime_type": "none",
            "confidence": "low",
            "severity": None,
            "persistence_estimate": "N/A",
            "metrics": {},
            "reason": "insufficient_data",
            "llm_hint": "Cannot assess gap regime: missing frequency data for one or more windows"
        }

    # ═══ CASCADE CONDITIONS ═══════════════════════════════════════

    # 1. Краткосрочная частота высокая
    cond_high_freq = freq_5 >= 10.0

    # 2. Долгосрочный фон спокойный (режим НЕ был постоянным)
    cond_calm_background = freq_60 < 6.0

    # 3. Относительное ускорение (частота выросла >=2× от базовой)
    if freq_60 > 0:
        acceleration_vs_60d = freq_5 / freq_60
        cond_acceleration = acceleration_vs_60d >= 2.0
    else:
        # Если freq_60 == 0, но freq_5 >= 10% → режим точно новый
        acceleration_vs_60d = 999.0
        cond_acceleration = freq_5 >= 10.0

    # ═══ ГРАДИЕНТНЫЙ АНАЛИЗ ЧЕРЕЗ 20D ════════════════════════════

    # Отношение 5d к 20d показывает "свежесть" режима
    ratio_5d_to_20d = freq_5 / freq_20 if freq_20 > 0 else 999.0

    # Отношение 20d к 60d показывает "устойчивость" среднесрочного тренда
    ratio_20d_to_60d = freq_20 / freq_60 if freq_60 > 0 else 999.0

    # Классификация типа режима:
    regime_type = "none"
    confidence = "low"
    persistence_weeks = "N/A"

    if ratio_5d_to_20d < 1.2 and ratio_20d_to_60d >= 2.0:
        # 5d ≈ 20d >> 60d → устойчивый режим
        regime_type = "established"
        confidence = "high"
        persistence_weeks = "2-4 weeks"

    elif ratio_5d_to_20d >= 1.4 and ratio_20d_to_60d >= 1.5:
        # 5d >> 20d > 60d → краткосрочный всплеск
        regime_type = "recent_spike"
        confidence = "medium"
        persistence_weeks = "1-2 weeks"

    elif 1.2 <= ratio_5d_to_20d < 1.4 and 1.5 <= ratio_20d_to_60d < 2.0:
        # Плавный градиент → формируется
        regime_type = "gradual_buildup"
        confidence = "low-medium"
        persistence_weeks = "uncertain, monitor next 5-10 days"
    else:
        # Не попадает в категории emerging
        regime_type = "ambiguous"
        confidence = "low"
        persistence_weeks = "N/A"

    # ═══ NEW METRICS EXTRACTION ═══════════════════════════════════

    # 1. Percentile rank: где находится freq_5 в контексте freq_60?
    # Используем простое ранжирование: freq_5 относительно [0, freq_60]
    if freq_60 > 0:
        percentile_rank_5d = min(100.0, (freq_5 / freq_60) * 50.0)  # Грубый percentile: если freq_5 = 2×freq_60 → ~100th percentile
    else:
        percentile_rank_5d = 100.0 if freq_5 > 0 else 50.0

    # 2. Gap size analysis: amplitude_avg для 5d vs 60d
    amp_5d = g5.get("amplitude_avg") if isinstance(g5, dict) else None
    amp_60d = g60.get("amplitude_avg") if isinstance(g60, dict) else None

    if amp_5d is not None and amp_60d is not None and amp_60d > 0:
        amplitude_ratio_5d_60d = amp_5d / amp_60d
    else:
        amplitude_ratio_5d_60d = None

    # 3. Gap clustering: autocorr из g5
    clustering_5d = g5.get("gap_clustering", {}) if isinstance(g5, dict) else {}
    autocorr_5d = clustering_5d.get("autocorr_lag1") if isinstance(clustering_5d, dict) else None
    clustering_label = clustering_5d.get("interpretation") if isinstance(clustering_5d, dict) else "unknown"

    # 4. Directional bias из g5
    directional_5d = g5.get("directional_bias", {}) if isinstance(g5, dict) else {}
    bias_label = directional_5d.get("bias_label") if isinstance(directional_5d, dict) else "unknown"
    up_gap_freq = directional_5d.get("up_gap_frequency_pct") if isinstance(directional_5d, dict) else None
    down_gap_freq = directional_5d.get("down_gap_frequency_pct") if isinstance(directional_5d, dict) else None

    # ═══ ФИНАЛЬНАЯ ПРОВЕРКА ════════════════════════════════════

    emerging = all([cond_high_freq, cond_calm_background, cond_acceleration])

    if not emerging:
        return {
            "emerging": False,
            "regime_type": "none",
            "confidence": "low",
            "severity": None,
            "persistence_estimate": "N/A",
            "metrics": {
                "gap_5d": freq_5,
                "gap_20d": freq_20,
                "gap_60d": freq_60,
                "ratio_5d_to_20d": round(ratio_5d_to_20d, 2),
                "ratio_20d_to_60d": round(ratio_20d_to_60d, 2),
                "acceleration_vs_60d": round(acceleration_vs_60d, 2),
                "percentile_rank_5d_vs_60d": round(percentile_rank_5d, 1),
                "amplitude_avg_5d": amp_5d,
                "amplitude_avg_60d": amp_60d,
                "amplitude_ratio_5d_60d": round(amplitude_ratio_5d_60d, 2) if amplitude_ratio_5d_60d else None,
                "gap_clustering_autocorr": autocorr_5d,
                "gap_clustering_label": clustering_label,
                "directional_bias_label": bias_label,
                "up_gap_freq_5d": up_gap_freq,
                "down_gap_freq_5d": down_gap_freq
            },
            "reason": "conditions_not_met",
            "details": {
                "high_freq_5d": cond_high_freq,
                "calm_background_60d": cond_calm_background,
                "acceleration": cond_acceleration
            },
            "llm_hint": f"No emerging gap regime detected: gap_5d={freq_5}%, gap_20d={freq_20}%, gap_60d={freq_60}%"
        }

    # ═══ SEVERITY CLASSIFICATION ════════════════════════════════

    if regime_type == "established":
        severity = "high" if freq_5 >= 12.0 else "moderate"
    elif regime_type == "recent_spike":
        severity = "moderate" if freq_5 >= 12.0 else "low-moderate"
    elif regime_type == "gradual_buildup":
        severity = "low-moderate" if freq_5 >= 12.0 else "low"
    else:
        severity = "low"

    # Интерпретация для LLM
    interpretation_map = {
        "established": f"Устойчивый gap-режим (5d≈20d>>60d): частота гэпов стабильна на уровне ~{freq_5:.1f}% последние 2-4 недели. Высокая вероятность продолжения.",
        "recent_spike": f"Недавний всплеск гэпов (5d>>20d>60d): краткосрочная частота {freq_5:.1f}% значительно выше среднесрочной {freq_20:.1f}%. Режим свежий (5-10 дней), может угаснуть.",
        "gradual_buildup": f"Постепенное нарастание гэпов (5d>20d>60d): плавный градиент {freq_5:.1f}%→{freq_20:.1f}%→{freq_60:.1f}%. Формирующийся тренд, требует мониторинга.",
        "ambiguous": f"Неоднозначный паттерн гэпов: 5d={freq_5:.1f}%, 20d={freq_20:.1f}%, 60d={freq_60:.1f}%. Низкая предсказательная способность."
    }

    # ═══ ENHANCED INTERPRETATION WITH NEW METRICS ═══════════════

    # Clustering impact на persistence estimate
    if autocorr_5d is not None and autocorr_5d >= 0.3:
        clustering_warning = "High gap clustering (autocorr≥0.3): gaps likely to persist in series."
    elif autocorr_5d is not None and autocorr_5d >= 0.15:
        clustering_warning = "Moderate gap clustering: weak serial correlation."
    else:
        clustering_warning = "Low clustering: gaps are independent events."

    # Amplitude impact на severity
    if amplitude_ratio_5d_60d is not None and amplitude_ratio_5d_60d >= 1.3:
        amplitude_warning = f"Gap amplitude increased {amplitude_ratio_5d_60d:.1f}× vs 60d baseline: larger gaps = higher execution risk."
    elif amplitude_ratio_5d_60d is not None and amplitude_ratio_5d_60d <= 0.7:
        amplitude_warning = f"Gap amplitude decreased {amplitude_ratio_5d_60d:.1f}× vs 60d baseline: smaller gaps = lower immediate risk."
    else:
        amplitude_warning = "Gap amplitude stable vs 60d baseline."

    # Directional bias impact на strategy recommendation
    if "upside" in bias_label and "strong" in bias_label:
        directional_warning = "Strong upside gap bias: prefer put calendars/spreads. Avoid short calls."
    elif "downside" in bias_label and "strong" in bias_label:
        directional_warning = "Strong downside gap bias: prefer call calendars/spreads. Avoid short puts."
    elif "upside" in bias_label:
        directional_warning = "Moderate upside gap bias: slight preference for put-side strategies."
    elif "downside" in bias_label:
        directional_warning = "Moderate downside gap bias: slight preference for call-side strategies."
    else:
        directional_warning = "Symmetric gaps: no directional preference."

    return {
        "emerging": True,
        "regime_type": regime_type,
        "confidence": confidence,
        "severity": severity,
        "persistence_estimate": persistence_weeks,
        "metrics": {
            "gap_5d": freq_5,
            "gap_20d": freq_20,
            "gap_60d": freq_60,
            "ratio_5d_to_20d": round(ratio_5d_to_20d, 2),
            "ratio_20d_to_60d": round(ratio_20d_to_60d, 2),
            "acceleration_vs_60d": round(acceleration_vs_60d, 2),
            "percentile_rank_5d_vs_60d": round(percentile_rank_5d, 1),
            "amplitude_avg_5d": amp_5d,
            "amplitude_avg_60d": amp_60d,
            "amplitude_ratio_5d_60d": round(amplitude_ratio_5d_60d, 2) if amplitude_ratio_5d_60d else None,
            "gap_clustering_autocorr": autocorr_5d,
            "gap_clustering_label": clustering_label,
            "directional_bias_label": bias_label,
            "up_gap_freq_5d": up_gap_freq,
            "down_gap_freq_5d": down_gap_freq
        },
        "reason": f"gap_5d={freq_5}% vs gap_20d={freq_20}% vs gap_60d={freq_60}% (acceleration={acceleration_vs_60d:.1f}×)",
        "interpretation": interpretation_map.get(regime_type, ""),
        "affected_strategies": ["Iron Condor", "Short Straddle", "BWB"],
        "recommendation": "Reduce wing width or avoid short vega until regime stabilizes" if severity in ["high", "moderate"] else "Monitor position sizing",
        "enhanced_analysis": {
            "clustering": clustering_warning,
            "amplitude": amplitude_warning,
            "directional_bias": directional_warning
        },
        "llm_hint": f"Emerging gap regime detected: type='{regime_type}', confidence={confidence}, severity={severity}. Percentile rank={percentile_rank_5d:.0f}th (5d vs 60d distribution). {clustering_warning} {directional_warning}"
    }


# ════════════════════════════════════════════════════════════════
# DATA QUALITY WARNINGS
# ════════════════════════════════════════════════════════════════
def build_data_quality_warnings(
    vol_data: dict, rv_forecast_session: dict, basis_vol_20: Optional[float],
    f2_rel_label: str, gap_analysis: dict, thr: ThresholdConfig,
) -> list[dict[str, Any]]:
    """Автоматически генерирует data_quality.warnings."""
    warns: list[dict[str, Any]] = []

    vov_5d = vscalar(vol_data.get("F1", {}), "vol_of_vol_5d")
    if vov_5d is not None and vov_5d > thr.vov_elevated:
        warns.append({"code": "VOMMA_RISK_HIGH", "severity": "critical",
                      "message": f"F1 VoV_5d={vov_5d:.4f} > {thr.vov_elevated}. Short-vega вход отложить до VoV_5d < {thr.vov_entry_delay}.",
                      "affected_strategies": ["Iron Condor", "Iron Butterfly", "BWB", "Calendar"]})

    f1_fc = rv_forecast_session.get("to_nearest_thu_1845", {})
    f2_fc = rv_forecast_session.get("to_next_thu_1845", {})
    fc_f1 = f1_fc.get("forecast_RV") if isinstance(f1_fc, dict) else None
    fc_f2 = f2_fc.get("forecast_RV") if isinstance(f2_fc, dict) else None
    if fc_f1 is not None and fc_f2 is not None and abs(fc_f1 - fc_f2) < thr.forecast_identity_tol:
        warns.append({"code": "FORECAST_IDENTITY_F1_EQ_F2", "severity": "critical",
                      "message": f"forecast_RV F1={fc_f1:.6f} идентичен F2={fc_f2:.6f}. EWMA не дифференцирует term structure. Calendar kill-сигнал на RV-уровне.",
                      "affected_strategies": ["Calendar"]})

    if basis_vol_20 is not None and basis_vol_20 > thr.basis_vol_elevated:
        warns.append({"code": "BASIS_VOL_HIGH", "severity": "warning",
                      "message": f"basis_vol_20d={basis_vol_20:.6f} > {thr.basis_vol_elevated} (порог 'high'). Calendar: basis-риск повышен.",
                      "affected_strategies": ["Calendar"]})

    # ═══ NEW: EMERGING GAP REGIME DETECTION (трехуровневая проверка 5d/20d/60d) ═══
    gap_regime = detect_emerging_gap_regime(gap_analysis, thr)

    if gap_regime.get("emerging"):
        regime_type = gap_regime.get("regime_type")
        severity_label = gap_regime.get("severity", "low")
        confidence = gap_regime.get("confidence")
        metrics = gap_regime.get("metrics", {})

        # Маппинг severity в код предупреждения
        severity_map = {"high": "critical", "moderate": "warning", "low-moderate": "warning", "low": "info"}
        warn_severity = severity_map.get(severity_label, "warning")

        warns.append({
            "code": "GAP_REGIME_EMERGING",
            "severity": warn_severity,
            "message": f"Emerging gap regime detected: type='{regime_type}' (confidence={confidence}, severity={severity_label}). Gap frequencies: 5d={metrics.get('gap_5d')}%, 20d={metrics.get('gap_20d')}%, 60d={metrics.get('gap_60d')}%. {gap_regime.get('interpretation', '')}",
            "affected_strategies": gap_regime.get("affected_strategies", []),
            "regime_details": gap_regime  # Полный объект для детального анализа
        })

    f2_gap = gap_analysis.get("F2_next", {}); f2_g5  = f2_gap.get("5d", {})
    f1_gap = gap_analysis.get("F1_current", {}); f1_g5 = f1_gap.get("5d", {})
    freq_f2_5 = f2_g5.get("frequency_pct") if isinstance(f2_g5, dict) else None
    freq_f1_5 = f1_g5.get("frequency_pct") if isinstance(f1_g5, dict) else None

    if f2_rel_label == "reduced" and freq_f2_5 is not None and freq_f1_5 is not None and freq_f2_5 > freq_f1_5 * 1.5:
        warns.append({"code": "F2_GAP_LIQUIDITY_ARTIFACT", "severity": "warning",
                      "message": f"F2 gap_5d={freq_f2_5:.1f}% >> F1 gap_5d={freq_f1_5:.1f}% при F2 reliability=reduced. Вероятный ликвидностный артефакт.",
                      "affected_strategies": ["Calendar"]})

    return warns

print("Cell 4/5: Analytics & Classification загружены")
