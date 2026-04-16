"""
07_2_5_ewma_forecasting.py
Cell ID: vjYxQ94tY0Da
Exported: 2026-04-16T10:12:23.218784
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.2.5
EWMA Forecasting Module

Модуль для EWMA (Exponentially Weighted Moving Average) прогнозирования волатильности
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("ewma_forecast")


def forecast_ewma_rolling(
    df: pd.DataFrame,
    horizon_days: int,
    yang_zhang_series_fn,
    tdy: int = 252,
    min_calibration_history: int = 60,
    ewma_span: int = 20,
    verbose: bool = True
) -> pd.Series:
    """
    Rolling EWMA forecast (baseline для сравнения с HAR).

    Использует EWMA на 20-дневной realized volatility (RiskMetrics λ=0.94 эквивалент).

    Args:
        df: DataFrame с OHLCV данными
        horizon_days: горизонт прогнозирования
        yang_zhang_series_fn: функция yang_zhang_series
        tdy: торговых дней в году
        min_calibration_history: минимум истории для калибровки
        ewma_span: span для EWMA (20 ≈ λ=0.94)
        verbose: детальное логирование

    Returns:
        pd.Series с EWMA прогнозами
    """
    if verbose:
        log.info(f"\n=== EWMA Rolling Forecast (horizon={horizon_days}d) ===")
        log.info(f"Input: {df.shape[0]} rows")
        log.info(f"EWMA span: {ewma_span} (approx λ=0.94)")

    ewma_forecasts = pd.Series(index=df["date"], dtype=float)

    stats = {
        "valid_count": 0,
        "skipped_count": 0
    }

    for i in range(min_calibration_history, len(df)):
        hist_df = df.iloc[:i]

        if len(hist_df) < 20:
            stats["skipped_count"] += 1
            continue

        rv_20d = yang_zhang_series_fn(hist_df, 20, tdy)

        if rv_20d.empty:
            stats["skipped_count"] += 1
            continue

        ewma_vol = rv_20d.ewm(span=ewma_span, adjust=False).mean().iloc[-1]

        current_date = df["date"].iloc[i]
        ewma_forecasts.loc[current_date] = float(ewma_vol)
        stats["valid_count"] += 1

    result = ewma_forecasts.dropna()

    if verbose:
        log.info(f"\n=== EWMA Forecast Complete ===")
        log.info(f"Valid forecasts: {stats['valid_count']}")
        log.info(f"Skipped: {stats['skipped_count']}")
        log.info(f"After dropna: {len(result)} forecasts")

        if not result.empty:
            log.info(f"Date range: {result.index.min()} to {result.index.max()}")
            log.info(f"Sample forecasts:\n{result.head(3)}")
            log.info(f"Stats: mean={result.mean():.4f}, std={result.std():.4f}")

    return result


if __name__ == "__main__":
    print("✓ Cell 7.2.5: EWMA Forecasting Module загружен успешно")
    print(f"  - Функции: forecast_ewma_rolling")

    try:
        import pandas
        print(f"  - pandas {pandas.__version__} ✓")
    except ImportError as e:
        print(f"  - pandas ✗ ({e})")
