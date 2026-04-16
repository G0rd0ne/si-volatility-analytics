"""
10_1_thursday_helpers.py
Cell ID: u4HgpZTSWjRc
Exported: 2026-04-16T10:12:23.219075
"""

"""
Cell 10.1: Thursday-Aligned RV Computation Helpers
==================================================

Вспомогательные функции для календарной привязки RV к четвергам экспирации.

Data Source: MOEX ISS API (https://iss.moex.com/iss)

Usage в Colab:
    exec(open('cell_10_1_thursday_helpers.py').read())
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("thursday_helpers")
log.setLevel(logging.INFO)

# ============================================================================
# CALENDAR LOGIC
# ============================================================================

def find_next_thursday(start_date: date, today: date) -> date:
    """
    Находит ближайший четверг после start_date.

    Args:
        start_date: Дата, от которой ищем четверг
        today: Текущая дата (для проверки)

    Returns:
        date: Ближайший четверг (включая сам start_date, если это четверг в будущем)
    """
    days_ahead = (3 - start_date.weekday()) % 7  # 3 = Thursday (0=Monday)
    if days_ahead == 0 and start_date > today:
        return start_date
    elif days_ahead == 0:
        days_ahead = 7
    return start_date + timedelta(days=days_ahead)


def get_weekday_name(date_obj: date) -> str:
    """Возвращает название дня недели для даты."""
    weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    return weekdays[date_obj.weekday()]


def compute_trading_days_to_thursday(df: pd.DataFrame, current_date_idx: int) -> Optional[int]:
    """
    Вычисляет количество ТОРГОВЫХ дней от current_date до ближайшего четверга.

    Args:
        df: DataFrame с колонкой 'date'
        current_date_idx: Индекс текущей даты в df

    Returns:
        int: Количество торговых дней до ближайшего четверга (или None, если не найден)
    """
    if current_date_idx >= len(df):
        return None

    current_date = df['date'].iloc[current_date_idx]
    if isinstance(current_date, pd.Timestamp):
        current_date = current_date.date()

    # Найти календарный четверг
    next_thursday = find_next_thursday(current_date, current_date)

    # Посчитать торговые дни до четверга
    trading_days_count = 0
    for i in range(current_date_idx + 1, len(df)):
        check_date = df['date'].iloc[i]
        if isinstance(check_date, pd.Timestamp):
            check_date = check_date.date()

        trading_days_count += 1

        # Проверяем, достигли ли четверга (с точностью до одного дня)
        if check_date >= next_thursday:
            return trading_days_count

    return None


# ============================================================================
# THURSDAY-ALIGNED REALIZED VOL COMPUTATION
# ============================================================================

def yang_zhang_single_window(df: pd.DataFrame, tdy: int = 252) -> float:
    """
    Compute Yang-Zhang volatility for THE ENTIRE dataframe (no rolling).

    This is different from yang_zhang_series() which computes a rolling window.
    This function computes a SINGLE YZ vol value for the entire period.

    Args:
        df: OHLCV DataFrame (should contain entire forward period)
        tdy: Trading days per year (default 252)

    Returns:
        float: Annualized Yang-Zhang volatility for the period
    """
    if len(df) < 2:
        return np.nan

    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"]  / df["open"])
    log_co = np.log(df["close"]/ df["open"])
    log_oc = np.log(df["open"] / df["close"].shift(1))

    # Rogers-Satchell component
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    n = len(df)
    k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34

    # Compute variance over ENTIRE period (not rolling)
    # dropna() to handle first row's NaN in overnight returns
    var_o  = log_oc.dropna().var(ddof=1) if len(log_oc.dropna()) >= 2 else 0.0
    var_c  = log_co.var(ddof=1) if len(log_co) >= 2 else 0.0
    var_rs = rs.mean() if len(rs) >= 1 else 0.0

    # Handle NaN from variance calculation
    var_o = var_o if not np.isnan(var_o) else 0.0
    var_c = var_c if not np.isnan(var_c) else 0.0
    var_rs = var_rs if not np.isnan(var_rs) else 0.0

    variance = var_o + k * var_c + (1 - k) * var_rs

    # Clip negative variance (numerical stability)
    variance = max(variance, 0.0) if not np.isnan(variance) else 0.0

    return float(np.sqrt(variance * tdy))


def compute_realized_vol_to_thursday(
    df: pd.DataFrame,
    yang_zhang_series_fn=None,  # DEPRECATED: kept for backward compatibility
    tdy: int = 252,
    verbose: bool = False
) -> pd.Series:
    """
    Вычисляет forward realized volatility ДО БЛИЖАЙШЕГО ЧЕТВЕРГА для каждой даты.

    CRITICAL CHANGE (2026-04-15):
    - Теперь использует yang_zhang_single_window() вместо yang_zhang_series()
    - yang_zhang_single_window() вычисляет ОДНО YZ vol значение за весь forward период
    - Это даёт РАЗНЫЕ RV values для разных дат → non-zero variance в y_train

    Отличие от compute_realized_vol_forward:
    - Не фиксированный horizon_days, а переменный (зависит от weekday)
    - Целевая дата всегда четверг (день экспирации)

    Пример:
        - Если сегодня пятница → считаем RV до четверга (5-6 торговых дней)
        - Если сегодня среда → считаем RV до четверга (1 торговый день)

    Args:
        df: DataFrame с OHLCV данными и колонкой 'date'
        yang_zhang_series_fn: DEPRECATED (kept for backward compatibility)
        tdy: Торговых дней в году
        verbose: Детальное логирование

    Returns:
        pd.Series с forward RV до четверга, индекс = даты
    """
    if verbose:
        log.info(f"\n=== Computing Thursday-Aligned Forward RV ===")
        log.info(f"Input: {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")

    yz_forward = pd.Series(index=df["date"], dtype=float)

    stats = {
        "valid_windows": 0,
        "skipped_no_thursday": 0,
        "skipped_insufficient_data": 0,
        "empty_yz_count": 0,
        "weekday_distribution": {}
    }

    for i in range(len(df) - 7):  # Max horizon до четверга ~7 дней
        current_date = df['date'].iloc[i]
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()

        weekday = get_weekday_name(current_date)
        stats["weekday_distribution"][weekday] = stats["weekday_distribution"].get(weekday, 0) + 1

        # Найти количество торговых дней до четверга
        trading_days_to_thursday = compute_trading_days_to_thursday(df, i)

        if trading_days_to_thursday is None or trading_days_to_thursday < 1:
            stats["skipped_no_thursday"] += 1
            continue

        # Окно данных: [i+1, i+trading_days_to_thursday+1] (будущие дни)
        end_idx = i + trading_days_to_thursday + 1
        if end_idx >= len(df):
            stats["skipped_insufficient_data"] += 1
            continue

        window_df = df.iloc[i + 1:end_idx]

        if len(window_df) < 2:  # Need at least 2 bars for variance
            stats["skipped_insufficient_data"] += 1
            continue

        # CRITICAL FIX: Use yang_zhang_single_window() instead of yang_zhang_series()
        # yang_zhang_series() is a ROLLING window estimator (returns Series)
        # yang_zhang_single_window() computes ONE vol value for the ENTIRE forward period
        # This gives different RV values for different Fridays → non-zero variance in y_train
        scalar_value = yang_zhang_single_window(window_df, tdy)

        if not np.isnan(scalar_value) and np.isfinite(scalar_value):
            # Record forward RV at the STARTING date (i), not ending date
            # This way, when we look up RV for Friday, we get RV(Friday → next Thursday)
            result_date = df["date"].iloc[i]

            yz_forward.loc[result_date] = scalar_value
            stats["valid_windows"] += 1
        else:
            stats["empty_yz_count"] += 1

    if verbose:
        log.info(f"\n=== Thursday-Aligned RV Complete ===")
        log.info(f"Valid windows: {stats['valid_windows']}")
        log.info(f"Skipped (no thursday): {stats['skipped_no_thursday']}")
        log.info(f"Skipped (insufficient data): {stats['skipped_insufficient_data']}")
        log.info(f"Empty yz_series: {stats['empty_yz_count']}")
        log.info(f"Weekday distribution: {stats['weekday_distribution']}")

    return yz_forward.dropna()


# ============================================================================
# SELF-TEST
# ============================================================================

def test_thursday_helpers():
    """Тестирует календарную логику."""
    log.info("\n=== Testing Thursday Helpers ===")

    # Test 1: find_next_thursday
    test_dates = [
        (date(2024, 5, 13), "monday", date(2024, 5, 16)),
        (date(2024, 5, 14), "tuesday", date(2024, 5, 16)),
        (date(2024, 5, 15), "wednesday", date(2024, 5, 16)),
        (date(2024, 5, 16), "thursday", date(2024, 5, 23)),
        (date(2024, 5, 17), "friday", date(2024, 5, 23)),
    ]

    tests_passed = 0
    tests_failed = 0

    for test_date, weekday_expected, thursday_expected in test_dates:
        weekday = get_weekday_name(test_date)
        thursday = find_next_thursday(test_date, test_date)

        if weekday == weekday_expected and thursday == thursday_expected:
            log.info(f"  ✓ {test_date} ({weekday}) → {thursday}")
            tests_passed += 1
        else:
            log.error(f"  ✗ {test_date}: expected {thursday_expected}, got {thursday}")
            tests_failed += 1

    # Test 2: Check dependencies
    try:
        import pandas as pd
        log.info(f"  ✓ pandas {pd.__version__}")
        tests_passed += 1
    except ImportError:
        log.error(f"  ✗ pandas not found")
        tests_failed += 1

    log.info(f"\n{'='*60}")
    log.info(f"Tests: {tests_passed} passed, {tests_failed} failed")

    if tests_failed == 0:
        log.info("✓ ALL TESTS PASSED - Thursday helpers ready")
        return True
    else:
        log.error("✗ SOME TESTS FAILED")
        return False


# Run self-test on load
if __name__ == "__main__":
    test_thursday_helpers()
    print("\n✓ Cell 10.1: Thursday Helpers loaded into globals()")
