"""
07_2_6_forecast_evaluation.py
Cell ID: t3iEvDZ5Y5ti
Exported: 2026-04-16T10:12:23.218792
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.2.6
Forecast Evaluation Module

Модуль для оценки качества прогнозов волатильности
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("forecast_evaluation")


def evaluate_forecast(
    realized: pd.Series,
    forecast: pd.Series,
    verbose: bool = True
) -> dict[str, float]:
    """
    Метрики качества прогноза волатильности.

    Args:
        realized: фактическая realized volatility
        forecast: прогнозные значения
        verbose: детальное логирование

    Returns:
        dict с метриками: R2, MAE, RMSE, directional_accuracy, n_obs
    """
    if verbose:
        log.info(f"\n=== Forecast Evaluation ===")
        log.info(f"Realized: {len(realized)} obs")
        log.info(f"Forecast: {len(forecast)} obs")

    common_idx = realized.index.intersection(forecast.index)

    if verbose:
        log.info(f"Common indices: {len(common_idx)} obs")

    if len(common_idx) < 10:
        if verbose:
            log.warning(f"Insufficient overlapping data: {len(common_idx)} < 10")

        return {
            "R2": None,
            "MAE": None,
            "RMSE": None,
            "directional_accuracy": None,
            "n_obs": 0
        }

    y_true = realized.loc[common_idx].values
    y_pred = forecast.loc[common_idx].values

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else None

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    if len(y_true) > 1:
        diff_true = np.diff(y_true)
        diff_pred = y_pred[1:] - y_true[:-1]
        directional_correct = np.sum((diff_true > 0) == (diff_pred > 0))
        directional_accuracy = directional_correct / len(diff_true)
    else:
        directional_accuracy = None

    metrics = {
        "R2": round(r2, 4) if r2 is not None else None,
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "directional_accuracy": round(directional_accuracy, 3) if directional_accuracy else None,
        "n_obs": len(common_idx)
    }

    if verbose:
        log.info(f"\n=== Metrics ===")
        log.info(f"R²: {metrics['R2']}")
        log.info(f"MAE: {metrics['MAE']}")
        log.info(f"RMSE: {metrics['RMSE']}")
        log.info(f"Directional Accuracy: {metrics['directional_accuracy']}")
        log.info(f"N observations: {metrics['n_obs']}")

    return metrics


def compare_forecasts(
    realized: pd.Series,
    har_forecast: pd.Series,
    ewma_forecast: pd.Series,
    verbose: bool = True
) -> dict:
    """
    Сравнивает качество HAR и EWMA прогнозов.

    Args:
        realized: фактическая realized volatility
        har_forecast: HAR прогноз
        ewma_forecast: EWMA прогноз
        verbose: детальное логирование

    Returns:
        dict с метриками обоих прогнозов и улучшением HAR vs EWMA
    """
    if verbose:
        log.info(f"\n=== Comparing HAR vs EWMA ===")

    har_metrics = evaluate_forecast(realized, har_forecast, verbose=False)
    ewma_metrics = evaluate_forecast(realized, ewma_forecast, verbose=False)

    improvement_r2 = None
    improvement_rmse = None

    if har_metrics["R2"] and ewma_metrics["R2"]:
        improvement_r2 = round((har_metrics["R2"] - ewma_metrics["R2"]) * 100, 2)

    if har_metrics["RMSE"] and ewma_metrics["RMSE"]:
        improvement_rmse = round(
            (ewma_metrics["RMSE"] - har_metrics["RMSE"]) / ewma_metrics["RMSE"] * 100, 2
        )

    result = {
        "HAR_metrics": har_metrics,
        "EWMA_metrics": ewma_metrics,
        "improvement": {
            "R2_gain_pp": improvement_r2,
            "RMSE_reduction_pct": improvement_rmse
        }
    }

    if verbose:
        log.info(f"\n=== Comparison Results ===")
        log.info(f"HAR R²: {har_metrics['R2']}, EWMA R²: {ewma_metrics['R2']}")
        log.info(f"HAR RMSE: {har_metrics['RMSE']}, EWMA RMSE: {ewma_metrics['RMSE']}")
        log.info(f"R² improvement: {improvement_r2} pp")
        log.info(f"RMSE reduction: {improvement_rmse}%")

    return result


if __name__ == "__main__":
    print("✓ Cell 7.2.6: Forecast Evaluation Module загружен успешно")
    print(f"  - Функции: evaluate_forecast, compare_forecasts")

    try:
        import pandas
        import numpy
        print(f"  - pandas {pandas.__version__} ✓")
        print(f"  - numpy {numpy.__version__} ✓")
    except ImportError as e:
        print(f"  - Зависимости ✗ ({e})")
