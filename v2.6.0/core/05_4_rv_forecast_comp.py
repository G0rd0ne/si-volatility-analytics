"""
05_4_rv_forecast_comp.py
Cell ID: ioODNCI2MSV8
Exported: 2026-04-16T10:12:23.218697
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 5.4/5.5
RV Forecast Computation

HAR-RV и EWMA forecasts к следующим экспирациям (nearest/next Thursday 18:45)
"""

from datetime import date, datetime, timedelta, time as dt_time
from typing import Optional

# ════════════════════════════════════════════════════════════════
# RV FORECAST HELPERS
# ════════════════════════════════════════════════════════════════
def find_next_thursday(start_date: date, today: date) -> date:
    """Возвращает ближайший четверг после start_date (или сам start_date, если он четверг)."""
    days_ahead = (3 - start_date.weekday()) % 7  # 3 = Thursday (0=Monday)
    if days_ahead == 0 and start_date > today:  # Если start_date сам четверг и в будущем
        return start_date
    elif days_ahead == 0:  # Если сегодня четверг, берем следующий
        days_ahead = 7
    return start_date + timedelta(days=days_ahead)


def compute_rv_forecasts(vol_series: dict, gap_analysis: dict, today: date, cfg: PipelineConfig) -> dict:
    """
    Вычисляет HAR-RV и EWMA forecasts для nearest_thu и next_thu 18:45.
    Возвращает rv_forecast_session dict.
    """
    now_dt = datetime.combine(today, dt_time(10, 0))

    # Ищем ближайший четверг после сегодня
    nearest_thu = find_next_thursday(today, today)
    # Следующий четверг = ближайший + 7 дней
    next_thu = nearest_thu + timedelta(days=7)

    f1_yz_20d_series = vol_series.get("F1", {}).get("YZ_20d", pd.Series(dtype=float))
    f1_yz_5d_series = vol_series.get("F1", {}).get("YZ_5d", pd.Series(dtype=float))
    f1_yz_1d_series = vol_series.get("F1", {}).get("YZ_1d", pd.Series(dtype=float)) if "YZ_1d" in vol_series.get("F1", {}) else pd.Series(dtype=float)

    # Compute horizons
    nearest_horizon = max(0, (datetime.combine(nearest_thu, dt_time(18, 45)) - now_dt).total_seconds() / 86400)
    next_horizon = max(0, (datetime.combine(next_thu, dt_time(18, 45)) - now_dt).total_seconds() / 86400)

    # EWMA forecasts (legacy method)
    ewma_nearest = calc_session_aligned_rv_forecast(f1_yz_20d_series, now_dt, datetime.combine(nearest_thu, dt_time(18, 45)), cfg.moex, cfg.vol.ewma_spans)
    ewma_next = calc_session_aligned_rv_forecast(f1_yz_20d_series, now_dt, datetime.combine(next_thu, dt_time(18, 45)), cfg.moex, cfg.vol.ewma_spans)

    # Build gap mask for HAR-RV filtering
    gap_mask_f1 = None
    if "F1_current" in gap_analysis and "5d" in gap_analysis["F1_current"]:
        gap_5d_data = gap_analysis["F1_current"]["5d"]
        if isinstance(gap_5d_data, dict) and gap_5d_data.get("available"):
            # Create binary mask: True if significant gap occurred (placeholder logic)
            # In production: extract actual gap dates from OHLCV analysis
            gap_mask_f1 = pd.Series(False, index=f1_yz_20d_series.index)

    # HAR forecasts
    har_nearest = calc_har_rv_forecast(
        rv_1d=f1_yz_1d_series if not f1_yz_1d_series.empty else f1_yz_5d_series,
        rv_5d=f1_yz_5d_series,
        rv_20d=f1_yz_20d_series,
        horizon_days=nearest_horizon,
        filter_gaps=False,
        gap_mask=gap_mask_f1
    )

    har_next = calc_har_rv_forecast(
        rv_1d=f1_yz_1d_series if not f1_yz_1d_series.empty else f1_yz_5d_series,
        rv_5d=f1_yz_5d_series,
        rv_20d=f1_yz_20d_series,
        horizon_days=next_horizon,
        filter_gaps=False,
        gap_mask=gap_mask_f1
    )

    # Compare HAR vs EWMA
    def calc_spread(har_obj, ewma_obj):
        if har_obj is None or ewma_obj is None:
            return None
        if not har_obj.get("available") or not ewma_obj.get("available"):
            return None
        har_rv = har_obj.get("forecast_RV")
        ewma_rv = ewma_obj.get("forecast_RV")
        if har_rv is None or ewma_rv is None:
            return None
        return r_vol(har_rv - ewma_rv)

    spread_nearest = calc_spread(har_nearest, ewma_nearest)
    spread_next = calc_spread(har_next, ewma_next)

    # Consensus interpretation
    def interpret_spread(spread_pp):
        if spread_pp is None:
            return "unknown"
        if abs(spread_pp) < 0.005:  # <0.5pp
            return "consensus"
        elif spread_pp > 0.015:  # >1.5pp
            return "HAR_significantly_higher"
        elif spread_pp < -0.015:
            return "EWMA_significantly_higher"
        else:
            return "mild_disagreement"

    # Extract top-level fields for LLM parsing convenience
    nearest_recommended = har_nearest.get("forecast_RV") if har_nearest and har_nearest.get("available") else (ewma_nearest.get("forecast_RV") if ewma_nearest and ewma_nearest.get("available") else None)
    next_recommended = har_next.get("forecast_RV") if har_next and har_next.get("available") else (ewma_next.get("forecast_RV") if ewma_next and ewma_next.get("available") else None)

    nearest_target = ewma_nearest.get("target_datetime") if ewma_nearest and ewma_nearest.get("available") else None
    next_target = ewma_next.get("target_datetime") if ewma_next and ewma_next.get("available") else None

    nearest_frac_days = ewma_nearest.get("fractional_trading_days") if ewma_nearest and ewma_nearest.get("available") else None
    next_frac_days = ewma_next.get("fractional_trading_days") if ewma_next and ewma_next.get("available") else None

    rv_forecast_session = {
        "to_nearest_thu_1845": {
            "EWMA_forecast": ewma_nearest,
            "HAR_forecast": har_nearest,
            "HAR_vs_EWMA_spread_pp": r_vol(spread_nearest * 100) if spread_nearest else None,
            "forecast_consensus": interpret_spread(spread_nearest),
            "recommended_forecast": nearest_recommended,
            "target_datetime": nearest_target,
            "fractional_trading_days": nearest_frac_days,
            "forecast_RV": nearest_recommended,
            "available": nearest_recommended is not None
        },
        "to_next_thu_1845": {
            "EWMA_forecast": ewma_next,
            "HAR_forecast": har_next,
            "HAR_vs_EWMA_spread_pp": r_vol(spread_next * 100) if spread_next else None,
            "forecast_consensus": interpret_spread(spread_next),
            "recommended_forecast": next_recommended,
            "target_datetime": next_target,
            "fractional_trading_days": next_frac_days,
            "forecast_RV": next_recommended,
            "available": next_recommended is not None
        },
        "llm_hint": "HAR-RV forecast captures multi-horizon vol persistence better than EWMA. If HAR > EWMA by >1.5pp, market participants expect vol expansion. Use 'recommended_forecast' for IV edge analysis."
    }

    return rv_forecast_session


if __name__ == "__main__":
    print("Cell 5.4/5.5: RV Forecast Module загружен")
