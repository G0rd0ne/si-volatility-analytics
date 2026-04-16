"""
07_2_4_har_forecasting.py
Cell ID: t_U-l4bLYuPj
Exported: 2026-04-16T10:12:23.218773
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.2.4
HAR Forecasting Module

Модуль для HAR-RV прогнозирования волатильности
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("har_forecast")


def forecast_har_rolling(
    df: pd.DataFrame,
    horizon_days: int,
    beta_d: float,
    beta_w: float,
    beta_m: float,
    yang_zhang_series_fn,
    tdy: int = 252,
    min_calibration_history: int = 60,
    verbose: bool = True
) -> pd.Series:
    """
    Rolling HAR forecast с заданными коэффициентами.

    HAR модель: RV_t+h = β_d * RV_d + β_w * RV_w + β_m * RV_m
    где RV_d, RV_w, RV_m - дневная, недельная (5д), месячная (20д) волатильность

    Args:
        df: DataFrame с OHLCV данными
        horizon_days: горизонт прогнозирования
        beta_d, beta_w, beta_m: коэффициенты HAR модели
        yang_zhang_series_fn: функция yang_zhang_series
        tdy: торговых дней в году
        min_calibration_history: минимум истории для калибровки
        verbose: детальное логирование

    Returns:
        pd.Series с HAR прогнозами
    """
    if verbose:
        log.info(f"\n=== HAR Rolling Forecast (horizon={horizon_days}d) ===")
        log.info(f"Input: {df.shape[0]} rows, date range: {df['date'].min()} to {df['date'].max()}")
        log.info(f"Beta coefficients: daily={beta_d:.3f}, weekly={beta_w:.3f}, monthly={beta_m:.3f}")
        log.info(f"Min calibration history: {min_calibration_history}")

    har_forecasts = pd.Series(index=df["date"], dtype=float)

    stats = {
        "valid_count": 0,
        "skipped_count": 0,
        "reasons": {}
    }

    for i in range(min_calibration_history, len(df)):
        hist_df = df.iloc[:i]

        if len(hist_df) < 20:
            stats["skipped_count"] += 1
            reason = "hist_too_short"
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

            if verbose and i == min_calibration_history:
                log.warning(f"First iteration (i={i}): hist_df length={len(hist_df)} < 20")
            continue

        rv_1d = yang_zhang_series_fn(hist_df, 1, tdy)
        rv_5d = yang_zhang_series_fn(hist_df, 5, tdy)
        rv_20d = yang_zhang_series_fn(hist_df, 20, tdy)

        if rv_1d.empty or rv_5d.empty or rv_20d.empty:
            stats["skipped_count"] += 1
            reason = "empty_rv_components"
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

            if verbose and stats["valid_count"] == 0:
                log.warning(f"Iteration i={i}: RV series empty - 1d={len(rv_1d)}, 5d={len(rv_5d)}, 20d={len(rv_20d)}")
            continue

        latest_1d = float(rv_1d.iloc[-1])
        latest_5d = float(rv_5d.iloc[-1])
        latest_20d = float(rv_20d.iloc[-1])

        # Skip if any RV component is NaN (happens at rolling window boundaries)
        if pd.isna(latest_1d) or pd.isna(latest_5d) or pd.isna(latest_20d):
            stats["skipped_count"] += 1
            reason = "nan_rv_values"
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

            if verbose and stats["valid_count"] == 0:
                log.warning(f"Iteration i={i}: NaN RV values - 1d={latest_1d}, 5d={latest_5d}, 20d={latest_20d}")
            continue

        har_forecast = beta_d * latest_1d + beta_w * latest_5d + beta_m * latest_20d

        current_date = df["date"].iloc[i]
        har_forecasts.loc[current_date] = har_forecast
        stats["valid_count"] += 1

        if verbose and stats["valid_count"] == 1:
            log.info(f"First valid forecast at i={i}, date={current_date}")
            log.info(f"  RV_1d={latest_1d:.6f}, RV_5d={latest_5d:.6f}, RV_20d={latest_20d:.6f}")
            log.info(f"  Forecast={har_forecast:.6f}")

    result = har_forecasts.dropna()

    if verbose:
        log.info(f"\n=== HAR Forecast Complete ===")
        log.info(f"Valid forecasts: {stats['valid_count']}")
        log.info(f"Skipped: {stats['skipped_count']}")
        log.info(f"Skip reasons: {stats['reasons']}")
        log.info(f"After dropna: {len(result)} forecasts")

        if result.empty:
            log.error(f"CRITICAL: HAR forecast Series is EMPTY")
        else:
            log.info(f"Date range: {result.index.min()} to {result.index.max()}")
            log.info(f"Sample forecasts:\n{result.head(3)}")
            log.info(f"Stats: mean={result.mean():.4f}, std={result.std():.4f}")

    return result


if __name__ == "__main__":
    print("✓ Cell 7.2.4: HAR Forecasting Module загружен успешно")
    print(f"  - Функции: forecast_har_rolling")

    try:
        import pandas
        print(f"  - pandas {pandas.__version__} ✓")
    except ImportError as e:
        print(f"  - pandas ✗ ({e})")
