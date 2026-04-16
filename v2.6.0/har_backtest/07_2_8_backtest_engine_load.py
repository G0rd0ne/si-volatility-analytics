"""
07_2_8_backtest_engine_load.py
Cell ID: WAGiEd9iZgjY
Exported: 2026-04-16T10:12:23.218814
"""

from __future__ import annotations

import logging

log = logging.getLogger("har_optimization")

log.info("\n" + "=" * 80)
log.info("HAR Backtest Engine v2.6.0 - Colab Global Namespace Mode")
log.info("=" * 80)

# Получаем функции из глобального namespace (где они определены после выполнения ячеек)
try:
    # В Colab код выполняется в глобальном namespace IPython
    # Проверяем наличие функций напрямую

    g = globals()

    # Cell 07_2_01: contracts
    if 'identify_contracts_with_history' not in g:
        raise NameError("identify_contracts_with_history not found. Did you run cell 07_2_01?")
    identify_contracts_with_history = g['identify_contracts_with_history']
    log.info("✓ identify_contracts_with_history loaded from globals")

    # Cell 07_2_02: data loader
    if 'HistoricalDataLoader' not in g:
        raise NameError("HistoricalDataLoader not found. Did you run cell 07_2_02?")
    if 'aggregate_ohlcv_data' not in g:
        raise NameError("aggregate_ohlcv_data not found. Did you run cell 07_2_02?")
    HistoricalDataLoader = g['HistoricalDataLoader']
    aggregate_ohlcv_data = g['aggregate_ohlcv_data']
    log.info("✓ HistoricalDataLoader, aggregate_ohlcv_data loaded from globals")

    # Cell 07_2_03: realized vol
    if 'compute_realized_vol_forward' not in g:
        raise NameError("compute_realized_vol_forward not found. Did you run cell 07_2_03?")
    compute_realized_vol_forward = g['compute_realized_vol_forward']
    log.info("✓ compute_realized_vol_forward loaded from globals")

    # Cell 07_2_04: HAR forecast
    if 'forecast_har_rolling' not in g:
        raise NameError("forecast_har_rolling not found. Did you run cell 07_2_04?")
    forecast_har_rolling = g['forecast_har_rolling']
    log.info("✓ forecast_har_rolling loaded from globals")

    # Cell 07_2_05: EWMA forecast
    if 'forecast_ewma_rolling' not in g:
        raise NameError("forecast_ewma_rolling not found. Did you run cell 07_2_05?")
    forecast_ewma_rolling = g['forecast_ewma_rolling']
    log.info("✓ forecast_ewma_rolling loaded from globals")

    # Cell 07_2_06: evaluation
    if 'evaluate_forecast' not in g:
        raise NameError("evaluate_forecast not found. Did you run cell 07_2_06?")
    if 'compare_forecasts' not in g:
        raise NameError("compare_forecasts not found. Did you run cell 07_2_06?")
    evaluate_forecast = g['evaluate_forecast']
    compare_forecasts = g['compare_forecasts']
    log.info("✓ evaluate_forecast, compare_forecasts loaded from globals")

    # Cell 07_2_07: optimization
    if 'optimize_beta_coefficients' not in g:
        raise NameError("optimize_beta_coefficients not found. Did you run cell 07_2_07?")
    optimize_beta_coefficients = g['optimize_beta_coefficients']
    log.info("✓ optimize_beta_coefficients loaded from globals")

    log.info("\n" + "=" * 80)
    log.info("✓ HAR Backtest Engine: All functions loaded successfully from global namespace")
    log.info("=" * 80)

except NameError as e:
    log.error(f"\n" + "=" * 80)
    log.error("NAME ERROR: Function not found in global namespace")
    log.error("=" * 80)
    log.error(f"Details: {e}")
    log.error("\nInstructions:")
    log.error("1. Execute cells 07_2_01 through 07_2_07 in order BEFORE this cell")
    log.error("2. Each cell should print '✓ Cell X.X.X loaded successfully'")
    log.error("3. Do not skip any cells")
    log.error("4. If a cell failed, fix it and re-run before running this engine")
    log.error("=" * 80)
    raise
except Exception as e:
    log.error(f"Unexpected error during function loading: {e}")
    raise


# Экспортируем загруженные компоненты
__all__ = [
    'identify_contracts_with_history',
    'HistoricalDataLoader',
    'aggregate_ohlcv_data',
    'compute_realized_vol_forward',
    'forecast_har_rolling',
    'forecast_ewma_rolling',
    'evaluate_forecast',
    'compare_forecasts',
    'optimize_beta_coefficients'
]


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("✓ Cell 7.2.8: HAR Backtest Engine загружен успешно")
    print("=" * 80)
    print(f"\nДоступные функции и классы:")
    for item in __all__:
        print(f"  - {item}")
    print("=" * 80)
