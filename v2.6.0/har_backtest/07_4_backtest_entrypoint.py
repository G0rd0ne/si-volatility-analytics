"""
07_4_backtest_entrypoint.py
Cell ID: wtwa8dJkWHzX
Exported: 2026-04-16T10:12:23.218886
"""

"""
Cell 7.4: HAR Backtest Main Entrypoint
=======================================

Главный модуль для запуска бэктеста HAR-модели.

Функционал:
- Интеграция всех модулей Cell 7.2.01-09b
- Запуск полного pipeline бэктеста
- Визуализация и сохранение результатов
- Self-tests для валидации

Совместим с новой модульной архитектурой.

Author: Harvi (HAR Optimization System)
Version: 2.6.0
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from datetime import datetime

log = logging.getLogger("har_backtest_main")


def run_har_backtest(
    ib_client,
    contracts: list,
    horizon_days: int = 21,
    optimize_betas: bool = True,
    trading_days_per_year: int = 252,
    output_dir: str = "./backtest_results",
    save_plots: bool = True,
    save_json: bool = True,
    verbose: bool = True
) -> Dict[str, dict]:
    """
    Запуск HAR-RV бэктеста для портфеля контрактов.

    Args:
        ib_client: подключенный IB client (из ib_insync)
        contracts: список IB contract objects
        horizon_days: горизонт прогнозирования (21 = 1 месяц)
        optimize_betas: оптимизировать β-коэффициенты
        trading_days_per_year: торговых дней в году
        output_dir: путь для сохранения результатов
        save_plots: сохранять графики
        save_json: сохранять JSON
        verbose: детальное логирование

    Returns:
        Dict {symbol: backtest_result}

    Example:
        >>> from ib_insync import IB, Future
        >>> ib = IB()
        >>> ib.connect('127.0.0.1', 7497, clientId=1)
        >>> contracts = [Future('ES', '20250321', 'CME')]
        >>> results = run_har_backtest(ib, contracts, horizon_days=21)
        >>> print(results['ES']['comparison']['HAR_metrics']['R2'])
        0.6523
    """
    # Импортируем функции из globals (загружены через Cell 7.2.08)
    g = globals()
    BacktestDataPipeline = g['BacktestDataPipeline']
    HARBacktestEngine = g['HARBacktestEngine']
    plot_backtest_results = g.get('plot_backtest_results')
    save_backtest_json = g.get('save_backtest_json')
    validate_backtest_results = g.get('validate_backtest_results')

    if verbose:
        log.info(f"\n{'=' * 80}")
        log.info(f"HAR BACKTEST PIPELINE")
        log.info(f"Contracts: {len(contracts)}")
        log.info(f"Horizon: {horizon_days} days")
        log.info(f"Beta optimization: {'Enabled' if optimize_betas else 'Disabled'}")
        log.info(f"{'=' * 80}")

    # Шаг 1: Загрузка данных
    if verbose:
        log.info(f"\n[STEP 1/3] Loading historical data...")

    data_pipeline = BacktestDataPipeline(ib_client, trading_days_per_year)
    portfolio_data = data_pipeline.load_portfolio_history(
        contracts,
        duration_days=365,
        bar_size='1 hour',
        verbose=verbose
    )

    if not portfolio_data:
        log.error("✗ No data loaded, aborting backtest")
        return {}

    # Шаг 2: Запуск бэктеста
    if verbose:
        log.info(f"\n[STEP 2/3] Running backtest...")

    backtest_engine = HARBacktestEngine(horizon_days, trading_days_per_year)
    results = backtest_engine.run_portfolio(
        portfolio_data,
        optimize_betas=optimize_betas,
        verbose=verbose
    )

    if not results:
        log.error("✗ No backtest results, aborting")
        return {}

    # Шаг 3: Валидация, визуализация и сохранение
    if verbose:
        log.info(f"\n[STEP 3/3] Validation and persistence...")

    # Валидация результатов
    if validate_backtest_results:
        validation = validate_backtest_results(results)

        if validation['warnings']:
            log.warning(f"⚠ Validation warnings ({len(validation['warnings'])}):")
            for warning in validation['warnings']:
                log.warning(f"  - {warning}")

        if validation['errors']:
            log.error(f"✗ Validation errors ({len(validation['errors'])}):")
            for error in validation['errors']:
                log.error(f"  - {error}")

    # Визуализация
    if save_plots and plot_backtest_results:
        plot_backtest_results(results, output_dir, show_plots=False)

    # Сохранение JSON
    if save_json and save_backtest_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_backtest_json(results, output_dir, timestamp)

    if verbose:
        log.info(f"\n{'=' * 80}")
        log.info(f"HAR BACKTEST COMPLETE")
        log.info(f"✓ Processed {len(results)} contracts successfully")
        log.info(f"✓ Results saved to {output_dir}")
        log.info(f"{'=' * 80}")

    return results


def run_backtest_self_tests() -> bool:
    """
    Self-tests для валидации бэктест-системы.

    Returns:
        True если все тесты пройдены
    """
    log.info(f"\n{'=' * 80}")
    log.info(f"HAR BACKTEST SELF-TESTS")
    log.info(f"{'=' * 80}")

    tests_passed = 0
    tests_total = 0

    # Тест 1: Проверка импортов globals
    tests_total += 1
    log.info(f"\n[TEST 1/{5}] Checking global imports...")

    g = globals()
    required_globals = [
        'BacktestDataPipeline',
        'HARBacktestEngine',
        'yang_zhang_series',
        'compute_realized_vol_forward',
        'forecast_har_rolling',
        'forecast_ewma_rolling',
        'compare_forecasts'
    ]

    missing_globals = [name for name in required_globals if name not in g]

    if missing_globals:
        log.error(f"✗ Missing globals: {', '.join(missing_globals)}")
        log.error(f"  Hint: Run Cell 7.2.08 (engine) to load all dependencies")
    else:
        log.info(f"✓ All required globals loaded")
        tests_passed += 1

    # Тест 2: Проверка функций визуализации
    tests_total += 1
    log.info(f"\n[TEST 2/{5}] Checking visualization functions...")

    viz_functions = ['plot_backtest_results', 'save_backtest_json', 'validate_backtest_results']
    missing_viz = [name for name in viz_functions if name not in g]

    if missing_viz:
        log.error(f"✗ Missing visualization functions: {', '.join(missing_viz)}")
        log.error(f"  Hint: Run Cell 7.3 (visualization) to load viz functions")
    else:
        log.info(f"✓ All visualization functions loaded")
        tests_passed += 1

    # Тест 3: Проверка структуры BacktestDataPipeline
    tests_total += 1
    log.info(f"\n[TEST 3/{5}] Checking BacktestDataPipeline interface...")

    if 'BacktestDataPipeline' in g:
        cls = g['BacktestDataPipeline']
        required_methods = ['load_contract_history', 'load_portfolio_history', 'validate_data_quality']

        missing_methods = [m for m in required_methods if not hasattr(cls, m)]

        if missing_methods:
            log.error(f"✗ Missing methods in BacktestDataPipeline: {', '.join(missing_methods)}")
        else:
            log.info(f"✓ BacktestDataPipeline interface valid")
            tests_passed += 1
    else:
        log.error(f"✗ BacktestDataPipeline not loaded")

    # Тест 4: Проверка структуры HARBacktestEngine
    tests_total += 1
    log.info(f"\n[TEST 4/{5}] Checking HARBacktestEngine interface...")

    if 'HARBacktestEngine' in g:
        cls = g['HARBacktestEngine']
        required_methods = ['run_single_contract', 'run_portfolio']

        missing_methods = [m for m in required_methods if not hasattr(cls, m)]

        if missing_methods:
            log.error(f"✗ Missing methods in HARBacktestEngine: {', '.join(missing_methods)}")
        else:
            log.info(f"✓ HARBacktestEngine interface valid")
            tests_passed += 1
    else:
        log.error(f"✗ HARBacktestEngine not loaded")

    # Тест 5: Проверка совместимости comparison format
    tests_total += 1
    log.info(f"\n[TEST 5/{5}] Checking comparison format compatibility...")

    if 'compare_forecasts' in g:
        # Проверяем сигнатуру compare_forecasts
        import inspect
        sig = inspect.signature(g['compare_forecasts'])

        expected_params = ['realized', 'har_forecast', 'ewma_forecast']
        actual_params = list(sig.parameters.keys())

        if all(p in actual_params for p in expected_params):
            log.info(f"✓ compare_forecasts signature compatible")
            tests_passed += 1
        else:
            log.error(f"✗ compare_forecasts signature mismatch")
            log.error(f"  Expected: {expected_params}")
            log.error(f"  Actual: {actual_params}")
    else:
        log.error(f"✗ compare_forecasts not loaded")

    # Итоговая статистика
    log.info(f"\n{'=' * 80}")
    log.info(f"SELF-TESTS SUMMARY")
    log.info(f"Passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        log.info(f"✓ ALL TESTS PASSED")
    else:
        log.error(f"✗ {tests_total - tests_passed} TEST(S) FAILED")

    log.info(f"{'=' * 80}")

    return tests_passed == tests_total


if __name__ == "__main__":
    """
    Standalone execution example.

    Usage:
        # 1. Загрузите все модули Cell 7.2.01-09b
        # 2. Загрузите Cell 7.3 (visualization)
        # 3. Запустите этот файл

        python cell_07_4_main.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    print("✓ Cell 7.4: HAR Backtest Main Entrypoint загружен успешно")
    print(f"  - Функции: run_har_backtest, run_backtest_self_tests")
    print(f"\nДля запуска self-tests:")
    print(f"  >>> run_backtest_self_tests()")
    print(f"\nДля запуска бэктеста:")
    print(f"  >>> results = run_har_backtest(ib_client, contracts)")
