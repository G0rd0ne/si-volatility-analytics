"""
05_2_self_tests.py
Cell ID: zvKoyi6IMICM
Exported: 2026-04-16T10:12:23.218681
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 5.2/5.5
Self-Tests Module

Integrity checks для выходных данных pipeline
"""

# ════════════════════════════════════════════════════════════════
# SELF-TESTS
# ════════════════════════════════════════════════════════════════
def run_self_tests(output: dict, rv_forecast_session: dict, samuelson_daily: dict) -> list[str]:
    errs = []
    if "forecast_RV" not in str(rv_forecast_session): errs.append("T1: forecast_RV отсутствует")
    if not any(f"DTE_{i}" in samuelson_daily for i in range(1, 11)): errs.append("T2: Ни один DTE_1..DTE_10 не найден")
    if "volume_ratio_historical" not in output: errs.append("T3: volume_ratio_historical отсутствует")
    dq = output.get("data_quality", {}).get("warnings", [])
    if not isinstance(dq, list): errs.append("T4: data_quality.warnings должен быть list")
    f1_vd = output.get("volatility", {}).get("F1_current", {})
    sup = f1_vd.get("semivar_up_5d")
    if sup == 0.0:
        asym = output.get("semivar_asymmetry", {}).get("F1_5d", {})
        if asym.get("label") != "extreme_downside": errs.append("T5: semivar_up_5d=0.0 но label != extreme_downside")
    ga = output.get("gap_analysis", {})
    for contract_key in ga:
        for w_key in ga[contract_key]:
            if not isinstance(w_key, str) or not w_key.endswith("d"):
                errs.append(f"T6: gap_analysis[{contract_key}] нестроковый ключ: {w_key!r}"); break
    vov_5d = vscalar(output.get("volatility", {}).get("F1_current", {}), "vol_of_vol_5d")
    if vov_5d is not None and vov_5d > 0.40:
        codes = [w.get("code") for w in dq]
        if "VOMMA_RISK_HIGH" not in codes: errs.append("T7: VoV > 0.40 но VOMMA_RISK_HIGH отсутствует")
    return errs


if __name__ == "__main__":
    print("Cell 5.2/5.5: Self-Tests Module загружен")
