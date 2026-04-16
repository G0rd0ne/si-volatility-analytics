"""
07_2_7_beta_optimization.py
Cell ID: 7b4i090EY_w0
Exported: 2026-04-16T10:12:23.218801
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.2.7
HAR Beta Optimization Module

Модуль для оптимизации β-коэффициентов HAR модели
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger("har_optimization")


def optimize_beta_coefficients(
    df: pd.DataFrame,
    horizon_days: int,
    compute_realized_vol_fn,
    forecast_har_fn,
    evaluate_forecast_fn,
    yang_zhang_series_fn,
    tdy: int = 252,
    verbose: bool = True
) -> tuple[float, float, float, dict]:
    """
    Оптимизация β-коэффициентов HAR модели через минимизацию RMSE.

    HAR модель: RV_t+h = β_d * RV_d + β_w * RV_w + β_m * RV_m
    Constraint: β_d + β_w + β_m = 1.0

    Args:
        df: DataFrame с OHLCV данными
        horizon_days: горизонт прогнозирования
        compute_realized_vol_fn: функция compute_realized_vol_forward
        forecast_har_fn: функция forecast_har_rolling
        evaluate_forecast_fn: функция evaluate_forecast
        yang_zhang_series_fn: функция yang_zhang_series
        tdy: торговых дней в году
        verbose: детальное логирование

    Returns:
        (beta_d, beta_w, beta_m, metrics)
    """
    if verbose:
        log.info(f"\n=== Optimizing HAR Beta Coefficients (horizon={horizon_days}d) ===")
        log.info(f"Input data: {len(df)} bars")

    # Compute forward realized volatility for full df
    realized_full = compute_realized_vol_fn(
        df, horizon_days, yang_zhang_series_fn, tdy, verbose=False
    )

    # CRITICAL FIX: Filter realized vol to match forecast range
    # forecast_har_rolling starts from min_calibration_history (default=60)
    # So realized vol must also start from that index to have valid intersection
    min_cal_history = 60  # Must match forecast_har_rolling default
    if len(df) <= min_cal_history:
        if verbose:
            log.warning(f"Insufficient data for optimization: {len(df)} <= {min_cal_history}")
        return (0.4, 0.4, 0.2, {})

    # Filter realized vol to start from min_calibration_history
    start_date = df['date'].iloc[min_cal_history]
    realized = realized_full[realized_full.index >= start_date]

    if len(realized) < 10:
        if verbose:
            log.warning(f"Insufficient realized vol after filtering: {len(realized)} < 10")

        return (0.4, 0.4, 0.2, {})

    if verbose:
        log.info(f"Forward RV computed: {len(realized_full)} total, {len(realized)} after min_cal filter")

    def objective(betas):
        """RMSE minimization objective function."""
        beta_d, beta_w, beta_m = betas

        if abs(beta_d + beta_w + beta_m - 1.0) > 0.05:
            return 1e6

        if any(b < 0 or b > 1 for b in betas):
            return 1e6

        forecast = forecast_har_fn(
            df, horizon_days, beta_d, beta_w, beta_m,
            yang_zhang_series_fn, tdy,
            min_calibration_history=60,
            verbose=False
        )

        common_idx = realized.index.intersection(forecast.index)
        if len(common_idx) < 10:
            return 1e6

        y_true = realized.loc[common_idx].values
        y_pred = forecast.loc[common_idx].values

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

    x0 = [0.35, 0.40, 0.25]

    if verbose:
        log.info(f"Starting optimization with initial guess: {x0}")

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints={'type': 'eq', 'fun': lambda b: sum(b) - 1.0},
        options={'maxiter': 100, 'ftol': 1e-6}
    )

    if not result.success:
        if verbose:
            log.warning(f"Optimization failed: {result.message}")

        return (0.4, 0.4, 0.2, {})

    beta_d, beta_w, beta_m = result.x

    if verbose:
        log.info(f"Optimization successful: {result.nit} iterations")
        log.info(f"Optimal betas: daily={beta_d:.3f}, weekly={beta_w:.3f}, monthly={beta_m:.3f}")

    optimal_forecast = forecast_har_fn(
        df, horizon_days, beta_d, beta_w, beta_m,
        yang_zhang_series_fn, tdy,
        min_calibration_history=60,
        verbose=False
    )

    metrics = evaluate_forecast_fn(realized, optimal_forecast, verbose=False)

    if verbose:
        log.info(f"Optimization metrics: {metrics}")

    # Add iteration count to metrics for tracking
    if 'n_iterations' not in metrics:
        metrics['n_iterations'] = result.nit if result.success else 0

    return (
        round(beta_d, 3),
        round(beta_w, 3),
        round(beta_m, 3),
        metrics
    )


if __name__ == "__main__":
    print("✓ Cell 7.2.7: HAR Beta Optimization Module загружен успешно")
    print(f"  - Функции: optimize_beta_coefficients")

    try:
        import pandas
        import numpy
        from scipy.optimize import minimize
        print(f"  - pandas {pandas.__version__} ✓")
        print(f"  - numpy {numpy.__version__} ✓")
        print(f"  - scipy.optimize ✓")
    except ImportError as e:
        print(f"  - Зависимости ✗ ({e})")
