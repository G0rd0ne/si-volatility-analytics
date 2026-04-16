"""
05_5_main_pipeline.py
Cell ID: N4wV10wPMXCE
Exported: 2026-04-16T10:12:23.218713
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 5.5/5.5
Main Pipeline Orchestration & Entrypoint

Собирает все модули (5.1-5.4) и выполняет полный цикл анализа
"""

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional
import pandas as pd

# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════
def run_pipeline(cfg: Optional[Any] = None) -> dict[str, Any]:
    # Dynamically pull necessary classes and functions from the global scope
    from __main__ import (
        PipelineConfig, MoexClient, identify_contracts, load_all_candles,
        compute_vol_metrics, compute_semivar_asymmetry, percentile_rank_250,
        compute_roc, classify_regime, avg_volume_nd, activity_weight,
        f2_reliability_label, vscalar, gap_statistics, build_samuelson_daily_1_10,
        calc_samuelson_status, build_data_quality_warnings, compute_rv_forecasts,
        log, r_pct, r_ratio, SCHEMA_VERSION, GAP_WINDOWS
    )

    if cfg is None:
        cfg = PipelineConfig()

    cfg.makedirs()
    today = date.today()

    log.info(f"═══ Si Dual-Contract Vol State {SCHEMA_VERSION} ═══")
    client = MoexClient(cfg.moex)
    try:
        contracts = identify_contracts(client, cfg.moex, today)
        dte_f1 = (contracts["F1"].expiry - today).days
        dte_f2 = (contracts["F2"].expiry - today).days
        days_since_f0 = (today - contracts["F0"].expiry).days

        ohlcv, samuelson_sources = load_all_candles(contracts, today, client, cfg.moex, cfg.vol)
        vol_data, vol_series, semi_nobs = compute_vol_metrics(ohlcv, days_since_f0, cfg)
        semivar_asym = compute_semivar_asymmetry(vol_data, semi_nobs, cfg.thr)

        pct_ranks = {f"F1_YZ_{w}d_pct_250d": r_pct(percentile_rank_250(vol_series.get("F1", {}).get(f"YZ_{w}d"))) for w in (5, 20, 60)}
        roc = compute_roc(vol_series)
        regime = classify_regime(pct_ranks, vol_data)

        # Basis & Volume Analysis
        vol5_f1 = avg_volume_nd(ohlcv.get("F1"), cfg.vol.lookback_candles, today)
        vol5_f2 = avg_volume_nd(ohlcv.get("F2"), cfg.vol.lookback_candles, today)
        volume_ratio = r_ratio(vol5_f1 / vol5_f2) if (vol5_f1 and vol5_f2 and vol5_f2 > 0) else None
        aw_f2 = activity_weight(ohlcv.get("F2"))
        f2_rel = f2_reliability_label(volume_ratio, aw_f2)

        # Realized components for cross metrics
        yz_f1_20 = vscalar(vol_data.get("F1", {}), "YZ_20d")
        yz_f2_20 = vscalar(vol_data.get("F2", {}), "YZ_20d")

        # Analytics sub-modules
        gap_analysis = {jk: {f"{w}d": gap_statistics(ohlcv.get(rk, pd.DataFrame()), w, cfg) for w in GAP_WINDOWS} for jk, rk in (("F0_expired", "F0"), ("F1_current", "F1"), ("F2_next", "F2"))}

        rv_forecast_session = compute_rv_forecasts(vol_series, gap_analysis, today, cfg)

        samuelson_daily = build_samuelson_daily_1_10(samuelson_sources, contracts, cfg.vol.tdy)
        samuelson_status_obj = calc_samuelson_status(yz_f1_20, dte_f1, samuelson_daily, cfg.thr)

        # Assembly
        output = {
            "meta": {"schema_version": SCHEMA_VERSION, "calculation_id": f"dualvol_{uuid.uuid4().hex[:6]}", "query_date": today.isoformat()},
            "contracts": {"F1_current": {"ticker": contracts["F1"].ticker, "days_to_expiry": dte_f1}},
            "volatility": {rk: vol_data.get(r, {}) for r, rk in (("F1","F1_current"),("F2","F2_next"))},
            "gap_analysis": gap_analysis,
            "rv_forecast_session_aligned": rv_forecast_session,
            "data_quality": {"warnings": build_data_quality_warnings(vol_data, rv_forecast_session, None, f2_rel, gap_analysis, cfg.thr)}
        }

        return output
    finally:
        client.close()

if __name__ == "__main__":
    main_result = run_pipeline()
    print("Pipeline complete")