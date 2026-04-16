"""
06_data_summary_report.py
Cell ID: GqDtzE_6k8cR
Exported: 2026-04-16T10:12:23.218723
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 6/6
Data Summary & Quality Report
Краткий анализ загрузки данных и вычислений для быстрой диагностики
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def is_null(val: Any) -> bool:
    """Проверка, является ли значение null-заглушкой."""
    if isinstance(val, dict):
        return val.get("available") is False or "reason" in val or "error" in val
    return val is None


def summarize_json_data(json_path: Path) -> dict[str, Any]:
    """
    Читает JSON output и генерирует краткую сводку по всем секциям:
    - Данные загружены корректно?
    - Все метрики вычислены?
    - Есть ли критические пропуски?
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = {
        "meta": {
            "calculation_id": data["meta"]["calculation_id"],
            "query_date": data["meta"]["query_date"],
            "schema_version": data["meta"]["schema_version"]
        },
        "sections": {}
    }

    # 1. Contracts
    contracts = data.get("contracts", {})
    report["sections"]["contracts"] = {
        "status": "✅ OK" if len(contracts) == 3 else "⚠️ INCOMPLETE",
        "summary": f"F0={contracts.get('F0_expired', {}).get('ticker', 'missing')}, F1={contracts.get('F1_current', {}).get('ticker', 'missing')}, F2={contracts.get('F2_next', {}).get('ticker', 'missing')}"
    }

    # 2. Volume Analysis
    vol_analysis = data.get("volume_analysis", {})
    avg_vol = vol_analysis.get("avg_volume_5d_F1")
    vol_ratio = vol_analysis.get("volume_ratio_f1_f2")
    vol_status = "✅ OK" if (avg_vol and not is_null(avg_vol) and vol_ratio and not is_null(vol_ratio)) else "⚠️ INCOMPLETE"
    report["sections"]["volume_analysis"] = {
        "status": vol_status,
        "summary": f"F1 avg_vol={avg_vol}, F1/F2 ratio={vol_ratio}, F2 reliability={vol_analysis.get('f2_reliability_label')}"
    }

    # 3. Data Quality
    dq = data.get("data_quality", {})
    warnings_count = len(dq.get("warnings", []))
    aligned_bars = dq.get("aligned_bars_f1_f2")
    critical_warnings = sum(1 for w in dq.get("warnings", []) if w.get("severity") == "critical")
    dq_summary = f"{warnings_count} warnings ({critical_warnings} critical), aligned_bars={aligned_bars}"
    dq_status = "✅ OK" if aligned_bars and aligned_bars >= 21 else "❌ INSUFFICIENT"
    report["sections"]["data_quality"] = {
        "status": dq_status,
        "summary": dq_summary,
        "warnings": [{"code": w.get("code"), "severity": w.get("severity")} for w in dq.get("warnings", [])]
    }

    # 4. Volatility
    volatility = data.get("volatility", {})
    f0_metrics = volatility.get("F0_expired", {})
    f1_metrics = volatility.get("F1_current", {})
    f2_metrics = volatility.get("F2_next", {})

    def count_valid_metrics(metrics: dict) -> int:
        return sum(1 for k, v in metrics.items() if not is_null(v) and not k.startswith("_"))

    f0_count = count_valid_metrics(f0_metrics)
    f1_count = count_valid_metrics(f1_metrics)
    f2_count = count_valid_metrics(f2_metrics)

    vol_status = "✅ OK" if f1_count >= 10 and f2_count >= 10 else "⚠️ INCOMPLETE"
    report["sections"]["volatility"] = {
        "status": vol_status,
        "summary": f"F0: {f0_count} metrics, F1: {f1_count} metrics, F2: {f2_count} metrics"
    }

    # 5. Samuelson Daily Short
    samuelson = data.get("samuelson_daily_short", {})
    dte_keys = [k for k in samuelson.keys() if k.startswith("DTE_")]
    interpolated_count = sum(1 for k in dte_keys if samuelson[k].get("reliability") == "interpolated")
    sam_status = "✅ OK" if len(dte_keys) == 10 else "⚠️ INCOMPLETE"
    report["sections"]["samuelson_daily_short"] = {
        "status": sam_status,
        "summary": f"{len(dte_keys)}/10 DTE buckets filled, {interpolated_count} interpolated"
    }

    # 6. RV Forecast (с явными значениями и таргет-датами)
    rv_forecast = data.get("rv_forecast_session_aligned", {})
    f1_forecast = rv_forecast.get("to_nearest_thu_1845", {})
    f2_forecast = rv_forecast.get("to_next_thu_1845", {})
    rv_status = "✅ OK" if (f1_forecast.get("available") and f2_forecast.get("available")) else "⚠️ INCOMPLETE"

    # Извлекаем явные значения прогноза
    f1_rv_value = f1_forecast.get('forecast_RV')
    f2_rv_value = f2_forecast.get('forecast_RV')
    f1_target_date = f1_forecast.get('target_datetime', 'N/A')
    f2_target_date = f2_forecast.get('target_datetime', 'N/A')

    report["sections"]["rv_forecast"] = {
        "status": rv_status,
        "summary": f"Nearest Thu 18:45 ({f1_target_date}): RV={f1_rv_value}, Next Thu 18:45 ({f2_target_date}): RV={f2_rv_value}",
        "nearest_thursday": {
            "target_datetime": f1_target_date,
            "forecast_RV": f1_rv_value,
            "fractional_days": f1_forecast.get('fractional_trading_days')
        },
        "next_thursday": {
            "target_datetime": f2_target_date,
            "forecast_RV": f2_rv_value,
            "fractional_days": f2_forecast.get('fractional_trading_days')
        }
    }

    # 7. Gap Analysis
    gap_analysis = data.get("gap_analysis", {})
    gap_contracts = list(gap_analysis.keys())
    gap_windows = [w for w in ["5d", "20d", "60d"] if all(w in gap_analysis.get(c, {}) for c in gap_contracts)]
    gap_status = "✅ OK" if len(gap_contracts) == 3 and len(gap_windows) == 3 else "⚠️ INCOMPLETE"
    report["sections"]["gap_analysis"] = {
        "status": gap_status,
        "summary": f"{len(gap_contracts)} contracts × {len(gap_windows)} windows"
    }

    # 8. Cross Contract Metrics
    cross_metrics = data.get("cross_contract_metrics", {})
    basis_vol_20d = cross_metrics.get("basis_vol_20d")
    stress = cross_metrics.get("front_specific_stress", {})
    cross_status = "✅ OK" if (basis_vol_20d and not is_null(basis_vol_20d) and stress.get("label") != "unknown") else "⚠️ INCOMPLETE"
    report["sections"]["cross_contract_metrics"] = {
        "status": cross_status,
        "summary": f"basis_vol_20d={basis_vol_20d}, spread_20d={cross_metrics.get('spread_f1_f2_YZ_20d')}, stress={stress.get('label')}"
    }

    # 9. State Labels
    state_labels = data.get("state_labels", {})
    unknown_count = sum(1 for v in state_labels.values() if v == "unknown")
    state_status = "✅ OK" if unknown_count == 0 else f"⚠️ {unknown_count} unknown"
    report["sections"]["state_labels"] = {
        "status": state_status,
        "summary": f"vol_regime={state_labels.get('vol_regime')}, basis_vol_label={state_labels.get('basis_vol_label')}, front_specific_stress={state_labels.get('front_specific_stress')}"
    }

    # 10. Volume Ratio Historical Analysis (с явным z-score и сравнением)
    vol_ratio_hist = data.get("volume_ratio_historical", {})
    current_ratio = vol_ratio_hist.get("current_ratio_pct")
    historical_avg = vol_ratio_hist.get("historical_avg_pct")
    historical_std = vol_ratio_hist.get("historical_std_pct")
    z_score = vol_ratio_hist.get("z_vs_history")
    interpretation = vol_ratio_hist.get("interpretation", "N/A")

    # Проверка на null-значения
    current_ratio_valid = current_ratio if not is_null(current_ratio) else None
    historical_avg_valid = historical_avg if not is_null(historical_avg) else None
    z_score_valid = z_score if not is_null(z_score) else None

    vol_ratio_status = "✅ OK" if (current_ratio_valid is not None and historical_avg_valid is not None and z_score_valid is not None) else "⚠️ INCOMPLETE"

    report["sections"]["volume_ratio_analysis"] = {
        "status": vol_ratio_status,
        "summary": f"Current F2/F1={current_ratio_valid}%, Historical avg={historical_avg_valid}%±{historical_std}, z-score={z_score_valid}",
        "current_ratio_pct": current_ratio_valid,
        "historical_avg_pct": historical_avg_valid,
        "historical_std_pct": historical_std,
        "z_score": z_score_valid,
        "interpretation": interpretation,
        "note": vol_ratio_hist.get("note", "")
    }

    # 11. Overall Status
    all_statuses = [s["status"] for s in report["sections"].values()]
    critical_fails = sum(1 for s in all_statuses if "❌" in s)
    warnings = sum(1 for s in all_statuses if "⚠️" in s)

    if critical_fails > 0:
        report["overall_status"] = "❌ FAILED"
    elif warnings > 0:
        report["overall_status"] = f"⚠️ PARTIAL ({warnings} warnings)"
    else:
        report["overall_status"] = "✅ COMPLETE"

    return report


def print_summary_report(report: dict[str, Any]) -> None:
    """Печатает краткий отчёт в stdout."""
    print("\n" + "="*80)
    print(f"DATA SUMMARY REPORT — {report['meta']['calculation_id']}")
    print(f"Query Date: {report['meta']['query_date']} | Schema: {report['meta']['schema_version']}")
    print("="*80)
    print(f"\nOVERALL STATUS: {report['overall_status']}\n")
    print("-"*80)

    for section_name, section_data in report["sections"].items():
        status_icon = section_data["status"]
        summary = section_data["summary"]
        print(f"{status_icon:12} | {section_name:30} | {summary}")

        # Печатать warnings если есть
        if "warnings" in section_data and section_data["warnings"]:
            for w in section_data["warnings"]:
                print(f"             └─ {w['severity']:8} {w['code']}")

        # Детализация для RV Forecast
        if section_name == "rv_forecast" and "nearest_thursday" in section_data:
            nearest = section_data["nearest_thursday"]
            next_thu = section_data["next_thursday"]
            print(f"             └─ Nearest Thu: {nearest['target_datetime']} → RV={nearest['forecast_RV']} ({nearest['fractional_days']} trading days)")
            print(f"             └─ Next Thu:    {next_thu['target_datetime']} → RV={next_thu['forecast_RV']} ({next_thu['fractional_days']} trading days)")

        # Детализация для Volume Ratio Analysis
        if section_name == "volume_ratio_analysis" and "z_score" in section_data:
            print(f"             └─ Interpretation: {section_data['interpretation']}")
            if section_data.get('note'):
                print(f"             └─ Note: {section_data['note']}")

    print("-"*80)
    print()


if __name__ == "__main__":
    # Ищем JSON файл
    import os

    # Проверяем известный путь из сообщения пользователя
    known_json_path = Path("/content/si_vol_analytics/si_vol_analytics_2.6.0.json")

    # Fallback поиск в текущей директории
    cwd = Path(os.getcwd())
    json_candidates = [
        known_json_path,
        cwd / "si_vol_analytics_2.6.0.json",
        cwd / "si_vol_analytics" / "si_vol_analytics_2.6.0.json",
    ]

    # Добавляем поиск через glob в текущей директории и поддиректориях
    json_candidates.extend(list(cwd.glob("**/*si_vol_analytics*.json")))

    # Находим первый существующий файл
    latest_json = None
    for candidate in json_candidates:
        if candidate.exists():
            latest_json = candidate
            break

    if latest_json is None:
        print(f"❌ JSON file not found")
        print(f"\nCurrent working directory: {os.getcwd()}")
        print("\nSearched locations:")
        for c in json_candidates[:3]:  # показываем только явные пути
            print(f"  - {c}")
        print("\nPlease ensure cell_05 has been executed and check the JSON output path.")
        exit(1)

    print(f"Analyzing: {latest_json}")

    report = summarize_json_data(latest_json)
    print_summary_report(report)

    # Сохраняем report рядом с исходным JSON
    report_path = latest_json.parent / f"data_summary_{report['meta']['query_date']}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary report saved: {report_path}")
