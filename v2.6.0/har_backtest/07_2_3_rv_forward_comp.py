"""
07_2_3_rv_forward_comp.py
Cell ID: YeeeE3JfYkpw
Exported: 2026-04-16T10:12:23.218764
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.2.3
Realized Volatility Forward Computation

Модуль для вычисления forward realized volatility
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("har_realized_vol")


def compute_realized_vol_forward(
    df: pd.DataFrame,
    horizon_days: int,
    yang_zhang_series_fn,
    tdy: int = 252,
    verbose: bool = True
) -> pd.Series:
    """
    Вычисляет forward realized volatility для каждого дня.

    RV_realized[t] = фактическая YZ vol за следующие horizon_days дней от t.

    Args:
        df: DataFrame с OHLCV данными
        horizon_days: горизонт прогнозирования (дни)
        yang_zhang_series_fn: функция yang_zhang_series из cell_03
        tdy: торговых дней в году
        verbose: детальное логирование

    Returns:
        pd.Series с forward realized volatility, индекс = даты
    """
    if verbose:
        log.info(f"\n=== Computing Forward Realized Vol (horizon={horizon_days}d) ===")
        log.info(f"Input: {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")
        log.info(f"Expected iterations: {len(df) - horizon_days}")

    yz_forward = pd.Series(index=df["date"], dtype=float)

    stats = {
        "valid_windows": 0,
        "skipped_windows": 0,
        "empty_yz_count": 0
    }

    for i in range(len(df) - horizon_days):
        start_idx = i
        end_idx = i + horizon_days
        # CRITICAL FIX: Forward realized vol must use FUTURE data [i+1, i+horizon+1]
        # Previous: df.iloc[i:i+horizon+1] included current bar i → look-ahead bias
        # Correct: df.iloc[i+1:i+horizon+2] uses only future bars for realized vol
        window_df = df.iloc[start_idx + 1:end_idx + 2]

        if verbose and i < 3:
            log.info(f"Iteration {i}: window [{start_idx}:{end_idx+1}], length={len(window_df)}")
            log.info(f"  Date: {df['date'].iloc[i]}")

        if len(window_df) < horizon_days:
            stats["skipped_windows"] += 1
            if verbose and i < 3:
                log.warning(f"  Window too short ({len(window_df)} < {horizon_days})")
            continue

        yz_series = yang_zhang_series_fn(window_df, horizon_days, tdy)

        if verbose and i < 3:
            log.info(f"  yz_series: length={len(yz_series)}, empty={yz_series.empty}")
            if not yz_series.empty:
                log.info(f"  yz_series.iloc[-1]={yz_series.iloc[-1]}")

        if not yz_series.empty:
            scalar_value = float(yz_series.iloc[-1])
            # CRITICAL FIX: Assign realized vol to END of forward period, not start
            # Forward period is [i+1, i+horizon+1], so result belongs to date at i+horizon
            # This ensures realized vol aligns with forecast timestamps for backtesting
            result_idx = i + horizon_days
            if result_idx >= len(df):
                stats["skipped_windows"] += 1
                continue
            current_date = df["date"].iloc[result_idx]

            if verbose and i < 3:
                log.info(f"  Assigning {scalar_value:.6f} to date {current_date} (result_idx={result_idx})")

            yz_forward.loc[current_date] = scalar_value
            stats["valid_windows"] += 1
        else:
            stats["empty_yz_count"] += 1

    if verbose:
        log.info(f"\n=== Forward RV Computation Complete ===")
        log.info(f"Valid windows: {stats['valid_windows']}")
        log.info(f"Skipped windows: {stats['skipped_windows']}")
        log.info(f"Empty yz_series: {stats['empty_yz_count']}")
        log.info(f"Before dropna: {len(yz_forward)} total, {yz_forward.notna().sum()} non-null")

    result = yz_forward.dropna()

    if verbose:
        log.info(f"After dropna: {len(result)} observations")

        if result.empty:
            log.error(f"CRITICAL: Forward RV is EMPTY for horizon={horizon_days}d")
            log.error(f"Diagnosis: valid={stats['valid_windows']}, empty={stats['empty_yz_count']}, input={len(df)}")
        else:
            log.info(f"Date range: {result.index.min()} to {result.index.max()}")
            log.info(f"Sample values:\n{result.head(3)}")
            log.info(f"Stats: mean={result.mean():.4f}, std={result.std():.4f}, min={result.min():.4f}, max={result.max():.4f}")

    return result


if __name__ == "__main__":
    print("✓ Cell 7.2.3: Realized Volatility Forward Module загружен успешно")
    print(f"  - Функции: compute_realized_vol_forward")

    try:
        import pandas
        print(f"  - pandas {pandas.__version__} ✓")
    except ImportError as e:
        print(f"  - pandas ✗ ({e})")
