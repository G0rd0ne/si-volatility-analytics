"""
05_5_main_pipeline.py
Cell ID: e3adACjo6sHe
Updated: 2026-04-16T10:59:54.384811
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

# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════
def run_pipeline(cfg: Optional[PipelineConfig] = None) -> dict[str, Any]:
    """Основная функция пайплайна. Тестируемая — принимает конфиг."""
    if cfg is None: cfg = PipelineConfig()
    cfg.makedirs()
    today = date.today()

    log.info("═══ Si Dual-Contract Vol State %s ═══", SCHEMA_VERSION)
    log.info("═══ Engine %s | Query date: %s ═══", ENGINE_VERSION, today)

    client = MoexClient(cfg.moex)
    try:
        log.info("── Contract identification ──")
        contracts = identify_contracts(client, cfg.moex, today)
        for role in sorted(contracts, key=lambda x: contracts[x].expiry):
            c = contracts[role]
            tag = "EXPIRED" if c.expiry < today else f"DTE={(c.expiry - today).days}"
            log.info("  %s: %s exp=%s [%s]", role.rjust(4), c.ticker, c.expiry, tag)

        dte_f1 = (contracts["F1"].expiry - today).days
        dte_f2 = (contracts["F2"].expiry - today).days
        days_since_f0 = (today - contracts["F0"].expiry).days
        log.info("DTE_F1=%dd | DTE_F2=%dd | Since_F0=%dd", dte_f1, dte_f2, days_since_f0)

        log.info("── Downloading OHLCV ──")
        ohlcv, samuelson_sources = load_all_candles(contracts, today, client, cfg.moex, cfg.vol)

        log.info("── Computing volatility metrics ──")
        vol_data, vol_series, semi_nobs = compute_vol_metrics(ohlcv, days_since_f0, cfg)
        semivar_asym = compute_semivar_asymmetry(vol_data, semi_nobs, cfg.thr)

        pct_ranks: dict[str, Any] = {}
        for w in (5, 20, 60):
            s = vol_series.get("F1", {}).get(f"YZ_{w}d")
            pval = percentile_rank_250(s) if s is not None else None
            pct_ranks[f"F1_YZ_{w}d_pct_250d"] = r_pct(pval) if pval else null_val("insufficient_history")

        roc = compute_roc(vol_series)
        regime = classify_regime(pct_ranks, vol_data)

        # Cross-contract + Basis
        cross: dict[str, Any] = {}
        for w in WINDOWS:
            yz1, yz2 = vscalar(vol_data.get("F1", {}), f"YZ_{w}d"), vscalar(vol_data.get("F2", {}), f"YZ_{w}d")
            cross[f"spread_{w}d"] = r_ratio(yz1 - yz2) if (yz1 and yz2) else null_val("requires_YZ")
            cross[f"ratio_{w}d"]  = r_ratio(yz1 / yz2) if (yz1 and yz2 and yz2 > 0) else null_val("F2_YZ_zero")
        cross["spread_change_5d"] = null_val("requires_series")
        cross["ratio_zscore_250d"] = null_val("placeholder")

        # Basis calculation
        basis_level = basis_log = basis_vol_5 = basis_vol_20 = None
        aligned_bars = 0
        f1_df = ohlcv.get("F1", pd.DataFrame())
        f2_df = ohlcv.get("F2", pd.DataFrame())
        if len(f1_df) > 0 and len(f2_df) > 0:
            aligned = pd.merge(
                f1_df[["date","close","volume"]].rename(columns={"close":"c1","volume":"v1"}),
                f2_df[["date","close","volume"]].rename(columns={"close":"c2","volume":"v2"}),
                on="date", how="inner"
            ).sort_values("date").reset_index(drop=True)
            if len(aligned) >= 2:
                aligned["b"]  = aligned["c2"] - aligned["c1"]
                aligned["bl"] = np.log(aligned["c2"] / aligned["c1"])
                aligned["bc"] = aligned["bl"].diff()
                basis_level = r_ratio(float(aligned["b"].iloc[-1]))
                basis_log   = r_vol(float(aligned["bl"].iloc[-1]))
                if len(aligned) >= 6:  basis_vol_5  = r_vol(float(aligned["bc"].iloc[-5:].std(ddof=1)  * math.sqrt(cfg.vol.tdy)))
                if len(aligned) >= 21: basis_vol_20 = r_vol(float(aligned["bc"].iloc[-20:].std(ddof=1) * math.sqrt(cfg.vol.tdy)))
                aligned_bars = len(aligned)
            else:
                aligned_bars = 0

        vol5_f1 = avg_volume_nd(ohlcv.get("F1", pd.DataFrame()), cfg.vol.lookback_candles, today)
        vol5_f2 = avg_volume_nd(ohlcv.get("F2", pd.DataFrame()), cfg.vol.lookback_candles, today)
        volume_ratio = r_ratio(vol5_f1 / vol5_f2) if (vol5_f1 and vol5_f2 and vol5_f2 > 0) else None
        aw_f1, aw_f2 = activity_weight(ohlcv.get("F1", pd.DataFrame())), activity_weight(ohlcv.get("F2", pd.DataFrame()))
        f2_rel = f2_reliability_label(volume_ratio, aw_f2)

        # Calculate calendar RV direction and front-specific stress
        cal_rv_fmb = r_ratio(vscalar(vol_data.get("F1", {}), "YZ_20d") - vscalar(vol_data.get("F2", {}), "YZ_20d"))

        yz_5d_f1 = vscalar(vol_data.get("F1", {}), "YZ_5d")
        yz_5d_f2 = vscalar(vol_data.get("F2", {}), "YZ_5d")
        stress_val = None
        stress_label = "unknown"

        if yz_5d_f1 is not None and yz_5d_f2 is not None and basis_vol_20 is not None and basis_vol_20 > 0:
            spread_5d = yz_5d_f1 - yz_5d_f2
            if cal_rv_fmb is not None:
                z_stress = (spread_5d - cal_rv_fmb) / basis_vol_20
                stress_val = r_ratio(z_stress)
                if abs(z_stress) < 1.0:
                    stress_label = "normal"
                elif abs(z_stress) < 2.0:
                    stress_label = "elevated" if z_stress > 0 else "compressed"
                else:
                    stress_label = "extreme" if z_stress > 0 else "extreme_compressed"

        stress_obj = {"value": stress_val if stress_val is not None else null_val("insufficient_data"), "label": stress_label}

        # Gap analysis
        all_ohlcv = {**ohlcv, **{k: v for k, v in samuelson_sources.items() if k not in ohlcv}}
        gap_analysis: dict[str, Any] = {}
        for jk, rk in (("F0_expired", "F0"), ("F1_current", "F1"), ("F2_next", "F2")):
            if rk in all_ohlcv:
                gap_analysis[jk] = {f"{w}d": gap_statistics(all_ohlcv[rk], w, cfg) for w in GAP_WINDOWS}

        # RV Forecasts (используем модуль 5.4)
        rv_forecast_session = compute_rv_forecasts(vol_series, gap_analysis, today, cfg)

        samuelson_daily = build_samuelson_daily_1_10(samuelson_sources, contracts, cfg.vol.tdy)
        yz_20d_f1 = vscalar(vol_data.get("F1", {}), "YZ_20d")
        samuelson_status_obj = calc_samuelson_status(yz_20d_f1, dte_f1, samuelson_daily, cfg.thr)

        volume_ratio_hist = calc_historical_volume_ratio_by_dte(ohlcv, contracts, samuelson_sources, today, dte_f1, dte_f2, cfg.vol)
        state_labels = build_state_labels(cross, stress_obj, semivar_asym, f2_rel, regime, samuelson_status_obj, roc, cal_rv_fmb, basis_vol_20, cfg.thr)

        gap_regime_analysis = detect_emerging_gap_regime(gap_analysis, cfg.thr)

        dq_warnings = build_data_quality_warnings(vol_data, rv_forecast_session, basis_vol_20, f2_rel, gap_analysis, cfg.thr)

        dist_classification = {
            "F1_skew_5d":  classify_skew(vscalar(vol_data.get("F1", {}), "skew_5d"), cfg.thr),
            "F1_skew_20d": classify_skew(vscalar(vol_data.get("F1", {}), "skew_20d"), cfg.thr),
            "F1_kurtosis_5d": classify_kurtosis(vscalar(vol_data.get("F1", {}), "kurtosis_5d"), cfg.thr),
            "F1_kurtosis_20d": classify_kurtosis(vscalar(vol_data.get("F1", {}), "kurtosis_20d"), cfg.thr)}

        calc_id = f"dualvol_si_{today.isoformat()}_v{uuid.uuid4().hex[:6]}"
        output: dict[str, Any] = {
            "meta": {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "calculation_id": calc_id,
                     "timezone": "Europe/Moscow", "query_date": today.isoformat(), "generated_at": datetime.now(timezone.utc).isoformat()},
            "contracts": {
                "F0_expired": {"ticker": contracts["F0"].ticker, "expiry": contracts["F0"].expiry.isoformat(), "days_since_expiry": days_since_f0},
                "F1_current": {"ticker": contracts["F1"].ticker, "expiry": contracts["F1"].expiry.isoformat(), "days_to_expiry": dte_f1},
                "F2_next": {"ticker": contracts["F2"].ticker, "expiry": contracts["F2"].expiry.isoformat(), "days_to_expiry": dte_f2}},
            "volume_analysis": {"avg_volume_5d_F1": r_ratio(vol5_f1) if vol5_f1 else null_val("insufficient"),
                                "volume_ratio_f1_f2": volume_ratio if volume_ratio else null_val("missing"),
                                "activity_weight_F1": aw_f1, "f2_reliability_label": f2_rel},
            "gap_regime_analysis": gap_regime_analysis,
            "data_quality": {"warnings": dq_warnings, "aligned_bars_f1_f2": aligned_bars},
            "volatility": {rk: vol_data.get(r, {}) for r, rk in (("F0","F0_expired"),("F1","F1_current"),("F2","F2_next"))},
            "samuelson_daily_short": samuelson_daily, "rv_forecast_session_aligned": rv_forecast_session,
            "volume_ratio_historical": volume_ratio_hist, "regime_classification": regime, "semivar_asymmetry": semivar_asym,
            "rate_of_change": roc, "state_labels": state_labels, "distribution_classification": dist_classification,
            "gap_analysis": gap_analysis, "percentile_ranks": pct_ranks,
            "cross_contract_metrics": {
                "spread_f1_f2_YZ_20d": r_ratio(vscalar(vol_data.get("F1", {}), "YZ_20d") - vscalar(vol_data.get("F2", {}), "YZ_20d")) if vscalar(vol_data.get("F1", {}), "YZ_20d") and vscalar(vol_data.get("F2", {}), "YZ_20d") else null_val("missing"),
                "ratio_f1_f2_YZ_20d": r_ratio(vscalar(vol_data.get("F1", {}), "YZ_20d") / vscalar(vol_data.get("F2", {}), "YZ_20d")) if vscalar(vol_data.get("F1", {}), "YZ_20d") and vscalar(vol_data.get("F2", {}), "YZ_20d") and vscalar(vol_data.get("F2", {}), "YZ_20d") > 0 else null_val("missing"),
                "spread_f1_f2_YZ_5d": r_ratio(vscalar(vol_data.get("F1", {}), "YZ_5d") - vscalar(vol_data.get("F2", {}), "YZ_5d")) if vscalar(vol_data.get("F1", {}), "YZ_5d") and vscalar(vol_data.get("F2", {}), "YZ_5d") else null_val("missing"),
                "basis_vol_5d": basis_vol_5 if basis_vol_5 is not None else null_val("insufficient_aligned_bars"),
                "basis_vol_20d": basis_vol_20 if basis_vol_20 is not None else null_val("insufficient_aligned_bars"),
                "term_structure_direction": "backwardation" if cal_rv_fmb is not None and cal_rv_fmb > 0 else ("contango" if cal_rv_fmb is not None and cal_rv_fmb < 0 else "flat"),
                "front_specific_stress": stress_obj,
                "llm_hint": "Cross-contract volatility metrics for term structure and calendar strategy analysis. Spread = F1 - F2, positive = backwardation."
            }}

        json_path = cfg.out_dir / f"si_vol_analytics_{SCHEMA_VERSION}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        log.info("JSON saved: %s", json_path)

        errs = run_self_tests(output, rv_forecast_session, samuelson_daily)
        if errs:
            log.warning("%d test failures:", len(errs))
            for e in errs: log.warning("  FAIL: %s", e)
        else:
            log.info("All self-tests passed.")

        try:
            plot_analytics_dashboard(output, vol_data, roc, cfg, SCHEMA_VERSION)
        except Exception as exc:
            log.warning("Dashboard failed (non-critical): %s", exc)

        log.info("Pipeline %s complete. Artifacts: %s/", SCHEMA_VERSION, cfg.out_dir)
        return output

    finally:
        client.close()


def main() -> None:
    """CLI entrypoint."""
    cfg = PipelineConfig()
    try:
        run_pipeline(cfg)
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    print("Cell 5.5/5.5: Pipeline & Main загружены")
    print("\n" + "="*60)
    print("ЗАПУСК PIPELINE v2.6.0")
    print("="*60 + "\n")
    main()

