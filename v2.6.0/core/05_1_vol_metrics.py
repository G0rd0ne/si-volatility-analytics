"""
05_1_vol_metrics.py
Cell ID: 3_ubpWu6ZeZX
Exported: 2026-04-16T10:12:23.218669
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 5.1/5.5
Volatility Metrics Computation

Вычисление YZ/CC vol, semivariance, VoV, skew, kurtosis для F0/F1/F2
"""

# ════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ════════════════════════════════════════════════════════════════
def compute_vol_metrics(ohlcv: dict[str, pd.DataFrame], days_since_f0: int, cfg: PipelineConfig) -> tuple[dict, dict, dict]:
    """Вычисляет YZ/CC vol, semivariance, VoV, skew, kurtosis для F0/F1/F2."""
    tdy = cfg.vol.tdy
    vol_data, vol_series, semi_nobs = {}, {}, {}

    for role in ("F0", "F1", "F2"):
        df = ohlcv.get(role)
        if df is None or len(df) == 0: continue
        vd, vs, sn = {}, {}, {}
        is_stale = (role == "F0" and days_since_f0 > 0)

        for w in WINDOWS:
            if is_stale and w == 5:
                vd[f"YZ_{w}d"] = vd[f"CC_{w}d"] = null_val("F0_expired_5d_stale"); continue
            if len(df) >= w + 1:
                yz = yang_zhang_series(df, w, tdy); cc = cc_vol_series(df, w, tdy)
                vs[f"YZ_{w}d"] = yz; vs[f"CC_{w}d"] = cc
                vd[f"YZ_{w}d"] = r_vol(latest(yz)); vd[f"CC_{w}d"] = r_vol(latest(cc))
                if w in (5, 20, 60):
                    vd[f"skew_{w}d"] = r_ratio(realized_skew(df, w))
                    vd[f"kurtosis_{w}d"] = r_kurt(realized_kurtosis(df, w))
                    vd[f"vol_of_vol_{w}d"] = r_vol(vol_of_vol(yz, w))
            else:
                vd[f"YZ_{w}d"] = null_val(f"insufficient_data_need_{w+1}_bars")
                vd[f"CC_{w}d"] = null_val(f"insufficient_data_need_{w+1}_bars")

        for w in SEMIVAR_WINDOWS:
            if is_stale and w == 5:
                for key in (f"semivar_down_{w}d", f"semivar_up_{w}d"):
                    vd[key] = null_val("F0_expired_5d_stale")
                sn[f"n_down_{w}d"] = sn[f"n_up_{w}d"] = 0; continue
            sd, nd = semivariance_ann(df, w, "down", tdy)
            su, nu = semivariance_ann(df, w, "up", tdy)
            vd[f"semivar_down_{w}d"] = r_vol(sd); vd[f"semivar_up_{w}d"] = r_vol(su)
            sn[f"n_down_{w}d"] = nd; sn[f"n_up_{w}d"] = nu

        vol_data[role], vol_series[role], semi_nobs[role] = vd, vs, sn

    return vol_data, vol_series, semi_nobs


def compute_semivar_asymmetry(vol_data: dict, semi_nobs: dict, thr: ThresholdConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("F1", "F2"):
        vd, sn = vol_data.get(role, {}), semi_nobs.get(role, {})
        for w in (5, 20):
            sd_raw, su_raw = vd.get(f"semivar_down_{w}d"), vd.get(f"semivar_up_{w}d")
            sd = sd_raw if isinstance(sd_raw, float) else (sd_raw.get("value") if isinstance(sd_raw, dict) else None)
            su = su_raw if isinstance(su_raw, float) else (su_raw.get("value") if isinstance(su_raw, dict) else None)
            nd, nu = sn.get(f"n_down_{w}d"), sn.get(f"n_up_{w}d")
            result[f"{role}_{w}d"] = classify_semivar_asymmetry(sd, su, nd, nu, thr)
    result["note"] = "Ratio>=1.3=asymmetry. semivar_up=0→extreme_downside. Affects put-skew."
    return result


def compute_roc(vol_series: dict) -> dict[str, Any]:
    roc: dict[str, Any] = {}
    for w in (5, 20):
        s = vol_series.get("F1", {}).get(f"YZ_{w}d")
        for lb in (5, 10, 20):
            cur, prev = latest(s), at_offset(s, lb)
            roc[f"F1_YZ_{w}d_roc_{lb}d_pct"] = r_pct((cur - prev) / prev * 100) if (cur and prev and prev > 0) else null_val(f"insufficient_{lb}d")

    roc5, roc20 = roc.get("F1_YZ_5d_roc_5d_pct"), roc.get("F1_YZ_5d_roc_20d_pct")
    if is_null(roc5) or is_null(roc20):
        roc["momentum_5d_vs_20d"] = "unknown"
    else:
        a5, a20 = abs(roc5), abs(roc20)
        if a5 > a20 * 1.5:   roc["momentum_5d_vs_20d"] = "accelerating_cooling" if roc5 < 0 else "accelerating_warming"
        elif a5 < a20 * 0.5: roc["momentum_5d_vs_20d"] = "decelerating_cooling" if roc5 < 0 else "decelerating_warming"
        else:                roc["momentum_5d_vs_20d"] = "stable_momentum"
    return roc


def classify_regime(pct_ranks: dict, vol_data: dict) -> dict[str, Any]:
    p20 = pct_ranks.get("F1_YZ_20d_pct_250d")
    if is_null(p20):
        return {"level": "unknown", "dynamics": "unknown", "composite": "unknown", "note": "Percentile unavailable."}
    level = "low" if p20 < 20 else "normal" if p20 < 65 else "elevated" if p20 < 90 else "crisis"
    yz5, yz20 = vscalar(vol_data.get("F1", {}), "YZ_5d"), vscalar(vol_data.get("F1", {}), "YZ_20d")
    dyn = "unknown"
    if yz5 and yz20 and yz20 > 0:
        r = yz5 / yz20
        dyn = "compression" if r < 0.75 else ("stable" if r <= 1.1 else "short_term_spike")
    return {"level": level, "dynamics": dyn, "composite": f"{level}_{dyn}", "note": ""}


def build_state_labels(cross: dict, stress_obj: dict, semivar_asym: dict, f2_rel: str, regime: dict,
                       samuelson_status_obj: dict, roc: dict, cal_rv_fmb: Optional[float],
                       basis_vol_20: Optional[float], thr: ThresholdConfig) -> dict[str, Any]:
    r20, roc5 = cross.get("ratio_20d"), roc.get("F1_YZ_5d_roc_5d_pct")
    asym_20d = semivar_asym.get("F1_20d", {})
    asym_label = asym_20d.get("label", "") if isinstance(asym_20d, dict) else ""
    downside_asym = ("present" if asym_label in ("moderate_downside_bias", "strong_downside_bias", "extreme_downside")
                     else "mild" if asym_label == "mild_downside_bias" else "absent" if asym_label == "symmetric" else "unknown")
    basis_vol_label = classify_basis_vol(basis_vol_20, thr)
    if is_null(roc5):              vol_trend = "unknown"
    elif roc5 < thr.roc_fast_cooling: vol_trend = "fast_cooling"
    elif roc5 < 0:                 vol_trend = "gradual_cooling"
    else:                          vol_trend = "warming"
    return {
        "front_vs_next_vol": "unknown" if is_null(r20) else ("similar" if r20 < 1.03 else "front_richer"),
        "front_specific_stress": "unknown" if is_null(stress_obj.get("value")) else stress_obj.get("label", "unknown"),
        "downside_asymmetry": downside_asym, "f2_reliability": f2_rel, "vol_regime": regime.get("composite"),
        "samuelson_status": samuelson_status_obj.get("status", "normal_for_dte"),
        "samuelson_z_score": samuelson_status_obj.get("z_score"),
        "samuelson_deviation_pp": samuelson_status_obj.get("deviation_pp"),
        "vol_trend": vol_trend, "calendar_rv_direction": "backwardation" if cal_rv_fmb and cal_rv_fmb > 0 else "contango",
        "basis_vol_label": basis_vol_label}


if __name__ == "__main__":
    print("Cell 5.1/5.5: Metrics Computation Module загружен")
