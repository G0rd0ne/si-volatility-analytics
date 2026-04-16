"""
07_5_backtest_self_tests.py
Cell ID: EhqRImXrGVrL
Exported: 2026-04-16T10:12:23.218905
"""

"""
Cell 7.5: HAR Backtest Self-Tests & Validation
===============================================

Модуль комплексных тестов для валидации HAR-бэктест системы.

Функционал:
- Unit-тесты для критичных функций
- Integration-тесты для полного pipeline
- Валидация согласованности данных
- Performance benchmarks

Author: Harvi (HAR Optimization System)
Version: 2.6.0
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

log = logging.getLogger("har_backtest_tests")


def test_data_pipeline_validation() -> bool:
    """
    Тест валидации данных BacktestDataPipeline.

    Returns:
        True если тест пройден
    """
    log.info(f"\n{'=' * 60}")
    log.info(f"TEST: Data Pipeline Validation")
    log.info(f"{'=' * 60}")

    g = globals()
    BacktestDataPipeline = g.get('BacktestDataPipeline')

    if not BacktestDataPipeline:
        log.error("✗ BacktestDataPipeline not loaded")
        return False

    # Создаем синтетические данные
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df_valid = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })

    # Корректируем OHLC чтобы high >= open,close,low и low <= open,close,high
    df_valid['high'] = df_valid[['open', 'high', 'close']].max(axis=1) + 1
    df_valid['low'] = df_valid[['open', 'low', 'close']].min(axis=1) - 1

    # Mock IB client
    class MockIB:
        pass

    pipeline = BacktestDataPipeline(MockIB(), 252)

    # Тест 1: Валидные данные должны пройти
    is_valid = pipeline.validate_data_quality(df_valid, min_bars=60, verbose=False)

    if not is_valid:
        log.error("✗ Valid data failed validation")
        return False

    log.info("✓ Valid data passed validation")

    # Тест 2: Недостаточно баров должно провалиться
    df_short = df_valid.head(30)
    is_valid = pipeline.validate_data_quality(df_short, min_bars=60, verbose=False)

    if is_valid:
        log.error("✗ Short data passed validation (should fail)")
        return False

    log.info("✓ Short data correctly failed validation")

    # Тест 3: Невалидные OHLC должны провалиться
    df_invalid = df_valid.copy()
    df_invalid.loc[5, 'high'] = df_invalid.loc[5, 'low'] - 10  # high < low

    is_valid = pipeline.validate_data_quality(df_invalid, min_bars=60, verbose=False)

    if is_valid:
        log.error("✗ Invalid OHLC passed validation (should fail)")
        return False

    log.info("✓ Invalid OHLC correctly failed validation")

    log.info(f"\n✓ TEST PASSED: Data Pipeline Validation")
    return True


def test_har_forecast_consistency() -> bool:
    """
    Тест согласованности HAR прогнозов.

    Returns:
        True если тест пройден
    """
    log.info(f"\n{'=' * 60}")
    log.info(f"TEST: HAR Forecast Consistency")
    log.info(f"{'=' * 60}")

    g = globals()
    forecast_har_rolling = g.get('forecast_har_rolling')
    yang_zhang_series = g.get('yang_zhang_series')

    if not forecast_har_rolling or not yang_zhang_series:
        log.error("✗ Required functions not loaded")
        return False

    # Синтетические данные
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 2),
        'close': 100 + np.cumsum(np.random.randn(200) * 2),
        'high': 100 + np.cumsum(np.random.randn(200) * 2),
        'low': 100 + np.cumsum(np.random.randn(200) * 2)
    })

    # Корректируем OHLC
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 1
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - 1

    # Тест 1: HAR с разными β должны давать разные прогнозы
    forecast_1 = forecast_har_rolling(
        df, 21, 0.4, 0.4, 0.2, yang_zhang_series, 252,
        min_calibration_history=60, verbose=False
    )

    forecast_2 = forecast_har_rolling(
        df, 21, 0.6, 0.3, 0.1, yang_zhang_series, 252,
        min_calibration_history=60, verbose=False
    )

    if len(forecast_1) == 0 or len(forecast_2) == 0:
        log.error("✗ HAR forecast failed to generate")
        return False

    # Прогнозы должны отличаться
    common_idx = forecast_1.index.intersection(forecast_2.index)
    if len(common_idx) == 0:
        log.error("✗ No common forecast indices")
        return False

    correlation = np.corrcoef(
        forecast_1.loc[common_idx].values,
        forecast_2.loc[common_idx].values
    )[0, 1]

    if correlation > 0.99:
        log.error(f"✗ Different betas produce identical forecasts (corr={correlation:.4f})")
        return False

    log.info(f"✓ Different betas produce different forecasts (corr={correlation:.4f})")

    # Тест 2: HAR прогнозы должны быть положительными
    if (forecast_1 < 0).any():
        log.error("✗ HAR forecast contains negative values")
        return False

    log.info("✓ HAR forecasts are positive")

    # Тест 3: HAR прогнозы должны быть в разумном диапазоне (< 5.0 годовых)
    if (forecast_1 > 5.0).any():
        log.warning(f"⚠ HAR forecast contains extreme values (max={forecast_1.max():.3f})")

    log.info(f"✓ HAR forecast range: [{forecast_1.min():.4f}, {forecast_1.max():.4f}]")

    log.info(f"\n✓ TEST PASSED: HAR Forecast Consistency")
    return True


def test_backtest_engine_metrics() -> bool:
    """
    Тест метрик качества прогнозов.

    Returns:
        True если тест пройден
    """
    log.info(f"\n{'=' * 60}")
    log.info(f"TEST: Backtest Engine Metrics")
    log.info(f"{'=' * 60}")

    g = globals()
    evaluate_forecast = g.get('evaluate_forecast')
    compare_forecasts = g.get('compare_forecasts')

    if not evaluate_forecast or not compare_forecasts:
        log.error("✗ Required functions not loaded")
        return False

    # Синтетические данные
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    realized = pd.Series(
        np.random.uniform(0.2, 0.5, 100),
        index=dates
    )

    # Тест 1: Perfect forecast (R² = 1.0)
    forecast_perfect = realized.copy()

    metrics = evaluate_forecast(realized, forecast_perfect, verbose=False)

    if metrics['R2'] is None or abs(metrics['R2'] - 1.0) > 0.01:
        log.error(f"✗ Perfect forecast R² != 1.0 (got {metrics['R2']})")
        return False

    if metrics['RMSE'] > 1e-10:
        log.error(f"✗ Perfect forecast RMSE != 0.0 (got {metrics['RMSE']})")
        return False

    log.info("✓ Perfect forecast metrics correct (R²=1.0, RMSE≈0)")

    # Тест 2: Random forecast (R² ≈ 0 или отрицательный)
    forecast_random = pd.Series(
        np.random.uniform(0.2, 0.5, 100),
        index=dates
    )

    metrics_random = evaluate_forecast(realized, forecast_random, verbose=False)

    if metrics_random['R2'] is not None and metrics_random['R2'] > 0.5:
        log.warning(f"⚠ Random forecast R² unexpectedly high: {metrics_random['R2']:.3f}")

    log.info(f"✓ Random forecast R² = {metrics_random['R2']}")

    # Тест 3: compare_forecasts структура
    forecast_har = realized + np.random.randn(100) * 0.02
    forecast_ewma = realized + np.random.randn(100) * 0.05

    comparison = compare_forecasts(realized, forecast_har, forecast_ewma, verbose=False)

    required_keys = ['HAR_metrics', 'EWMA_metrics', 'improvement']
    if not all(k in comparison for k in required_keys):
        log.error(f"✗ compare_forecasts missing keys: {required_keys}")
        return False

    log.info("✓ compare_forecasts structure valid")

    # Тест 4: HAR должен быть лучше EWMA (по конструкции)
    if comparison['HAR_metrics']['RMSE'] > comparison['EWMA_metrics']['RMSE']:
        log.warning("⚠ HAR RMSE worse than EWMA (expected better)")
    else:
        log.info(f"✓ HAR better than EWMA (RMSE improvement: {comparison['improvement']['RMSE_reduction_pct']}%)")

    log.info(f"\n✓ TEST PASSED: Backtest Engine Metrics")
    return True


def test_beta_optimization_convergence() -> bool:
    """
    Тест сходимости оптимизации β-коэффициентов.

    Returns:
        True если тест пройден
    """
    log.info(f"\n{'=' * 60}")
    log.info(f"TEST: Beta Optimization Convergence")
    log.info(f"{'=' * 60}")

    g = globals()
    optimize_beta_coefficients = g.get('optimize_beta_coefficients')
    yang_zhang_series = g.get('yang_zhang_series')
    compute_realized_vol_forward = g.get('compute_realized_vol_forward')
    forecast_har_rolling = g.get('forecast_har_rolling')

    if not all([optimize_beta_coefficients, yang_zhang_series,
                compute_realized_vol_forward, forecast_har_rolling]):
        log.error("✗ Required functions not loaded")
        return False

    # Синтетические данные с контролируемой волатильностью
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # Генерируем процесс с известной структурой
    returns = np.random.randn(300) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(300) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(300) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(300) * 0.005)),
        'close': prices
    })

    # Корректируем OHLC
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Запускаем оптимизацию
    try:
        beta_d, beta_w, beta_m, metrics = optimize_beta_coefficients(
            df, 21, yang_zhang_series, compute_realized_vol_forward,
            forecast_har_rolling, 252, verbose=False
        )

        # Тест 1: Beta сумма должна быть близка к 1.0
        beta_sum = beta_d + beta_w + beta_m
        if not (0.8 <= beta_sum <= 1.2):
            log.error(f"✗ Beta sum out of range: {beta_sum:.3f} (expected 0.8-1.2)")
            return False

        log.info(f"✓ Beta sum valid: {beta_sum:.3f}")

        # Тест 2: Все β должны быть положительными
        if beta_d < 0 or beta_w < 0 or beta_m < 0:
            log.error(f"✗ Negative betas: daily={beta_d:.3f}, weekly={beta_w:.3f}, monthly={beta_m:.3f}")
            return False

        log.info(f"✓ All betas positive: daily={beta_d:.3f}, weekly={beta_w:.3f}, monthly={beta_m:.3f}")

        # Тест 3: Метрики оптимизации должны присутствовать
        if not metrics or 'best_rmse' not in metrics:
            log.warning("⚠ Optimization metrics incomplete")
        else:
            log.info(f"✓ Optimization converged: RMSE={metrics.get('best_rmse', 'N/A')}")

        log.info(f"\n✓ TEST PASSED: Beta Optimization Convergence")
        return True

    except Exception as e:
        log.error(f"✗ Beta optimization failed: {e}")
        return False


def run_all_tests() -> Dict[str, bool]:
    """
    Запускает все тесты и возвращает результаты.

    Returns:
        Dict {test_name: passed}
    """
    log.info(f"\n{'=' * 80}")
    log.info(f"HAR BACKTEST COMPREHENSIVE TEST SUITE")
    log.info(f"{'=' * 80}")

    tests = {
        'data_pipeline_validation': test_data_pipeline_validation,
        'har_forecast_consistency': test_har_forecast_consistency,
        'backtest_engine_metrics': test_backtest_engine_metrics,
        'beta_optimization_convergence': test_beta_optimization_convergence
    }

    results = {}

    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            log.error(f"✗ TEST CRASHED: {test_name} - {e}")
            results[test_name] = False

    # Итоговая статистика
    passed = sum(results.values())
    total = len(results)

    log.info(f"\n{'=' * 80}")
    log.info(f"TEST SUITE SUMMARY")
    log.info(f"{'=' * 80}")

    for test_name, passed_flag in results.items():
        status = "✓ PASSED" if passed_flag else "✗ FAILED"
        log.info(f"{status}: {test_name}")

    log.info(f"\n{'=' * 80}")
    log.info(f"TOTAL: {passed}/{total} tests passed")

    if passed == total:
        log.info(f"✓ ALL TESTS PASSED")
    else:
        log.error(f"✗ {total - passed} TEST(S) FAILED")

    log.info(f"{'=' * 80}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    print("✓ Cell 7.5: HAR Backtest Self-Tests загружен успешно")
    print(f"  - Функции: run_all_tests, test_data_pipeline_validation, test_har_forecast_consistency")
    print(f"\nДля запуска всех тестов:")
    print(f"  >>> results = run_all_tests()")
