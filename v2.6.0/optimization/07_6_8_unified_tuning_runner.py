"""
07_6_8_unified_tuning_runner.py
Cell ID: H9jPKDsaf0ho
Exported: 2026-04-16T10:12:23.219054
"""

"""
Cell 7.6.8: HAR Unified Parameter Tuning Runner
================================================

Унифицированный runner для полного цикла подбора параметров HAR модели.

Workflow:
1. identify_contracts() → получить F0, F-1, F-2, ..., F-10 (11 контрактов)
2. load_portfolio_history() → скачать для каждого ≥252 котировок
3. run_grid_search() → подобрать β для каждого horizon
4. aggregate_results() → выбрать best β по медиане

Author: Harvi
Version: 2.6.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from datetime import date
import json
from pathlib import Path

log = logging.getLogger("har_unified_runner")

# Ensure logging is configured for Jupyter/Colab environments
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )


def run_full_har_parameter_tuning(
    moex_client,
    moex_config,
    today: date,
    tuning_config,
    output_dir: str = "./har_tuning_results",
    save_results: bool = True,
    verbose: bool = True
) -> Dict:
    """
    ПОЛНЫЙ унифицированный workflow для подбора HAR параметров.

    Реализует правильную логику:
    1. Идентификация F0, F-1, ..., F-10 (11 контрактов)
    2. Загрузка ≥252 дневных баров для каждого
    3. Grid search по β-коэффициентам
    4. Агрегация и выбор best parameters

    Args:
        moex_client: MoexClient instance
        moex_config: MoexConfig instance
        today: Текущая дата для идентификации контрактов
        tuning_config: HARTuningConfig instance
        output_dir: Путь для сохранения результатов
        save_results: Сохранять JSON файлы
        verbose: Детальное логирование

    Returns:
        Dict с best parameters для каждого horizon

    Example:
        >>> from cell_07_6_5_parameter_tuning_config import HARTuningConfig
        >>> from cell_02_moex_contracts import MoexClient, MoexConfig
        >>> from datetime import date
        >>>
        >>> client = MoexClient(MoexConfig())
        >>> config = HARTuningConfig()
        >>>
        >>> results = run_full_har_parameter_tuning(
        ...     client,
        ...     MoexConfig(),
        ...     date.today(),
        ...     config
        ... )
        >>>
        >>> # Best parameters для h=21 дней
        >>> best_21d = results[21]['best_params']
        >>> print(f"Optimal β for 21d: {best_21d}")
    """
    if verbose:
        log.info(f"\n{'=' * 80}")
        log.info(f"HAR UNIFIED PARAMETER TUNING PIPELINE")
        log.info(f"Date: {today}")
        log.info(f"Horizons: {tuning_config.horizon_days_grid}")
        log.info(f"Beta candidates: {len(tuning_config.beta_grid)}")
        log.info(f"{'=' * 80}")

    # Import from globals
    g = globals()
    identify_contracts = g.get('identify_contracts')
    BacktestDataPipeline = g.get('BacktestDataPipeline')
    HARParameterTuningEngine = g.get('HARParameterTuningEngine')

    if not all([identify_contracts, BacktestDataPipeline, HARParameterTuningEngine]):
        log.error("✗ Required modules not loaded. Please run cells 02, 07_2_09a, 07_6_6 first.")
        return {}

    # ========================================================================
    # ШАГ 1: ИДЕНТИФИКАЦИЯ КОНТРАКТОВ (F0, F-1, ..., F-10)
    # ========================================================================
    if verbose:
        log.info(f"\n[STEP 1/4] Identifying contracts...")

    try:
        contracts_dict = identify_contracts(moex_client, moex_config, today)
    except Exception as e:
        log.error(f"✗ Failed to identify contracts: {e}")
        return {}

    # Извлекаем нужные роли
    contracts = []
    missing_roles = []

    for role in tuning_config.contract_roles:
        if role in contracts_dict:
            contract = contracts_dict[role]
            contracts.append(contract)
            if verbose:
                log.info(f"✓ {role}: {contract.ticker} (expires {contract.expiry})")
        else:
            missing_roles.append(role)
            if verbose:
                log.warning(f"✗ {role}: not found")

    if missing_roles:
        log.warning(f"Missing roles: {missing_roles}")

    if not contracts:
        log.error("✗ No contracts available for tuning, aborting")
        return {}

    if verbose:
        log.info(f"\n✓ Identified {len(contracts)} contracts for parameter tuning")

    # ========================================================================
    # ШАГ 2: ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ (≥252 баров на контракт)
    # ========================================================================
    if verbose:
        log.info(f"\n[STEP 2/4] Loading historical data...")
        log.info(f"Target: ≥{tuning_config.min_bars_per_contract} bars per contract")

    # Создаем mapping symbol -> role для детального логирования
    symbol_to_role = {}
    for role in tuning_config.contract_roles:
        if role in contracts_dict:
            ticker = contracts_dict[role].ticker
            symbol_to_role[ticker] = role

    data_pipeline = BacktestDataPipeline(
        moex_client,
        tuning_config.tdy,
        cache_dir="./data/har_tuning_cache"
    )
    portfolio_data = data_pipeline.load_portfolio_history(
        contracts,
        duration_days=tuning_config.duration_days,
        bar_size=tuning_config.bar_size,
        verbose=verbose,
        contracts_roles=symbol_to_role
    )

    if not portfolio_data:
        log.error("✗ No data loaded, aborting tuning")
        return {}

    # Валидация минимального количества данных
    valid_data = {}
    for symbol, df in portfolio_data.items():
        if len(df) >= tuning_config.min_bars_per_contract:
            valid_data[symbol] = df
        else:
            if verbose:
                log.warning(
                    f"Skipping {symbol}: {len(df)} bars < {tuning_config.min_bars_per_contract} required"
                )

    if not valid_data:
        log.error(f"✗ No contracts with sufficient data (≥{tuning_config.min_bars_per_contract} bars)")
        return {}

    if verbose:
        log.info(f"\n✓ Loaded {len(valid_data)}/{len(contracts)} contracts with sufficient data")
        total_bars = sum(len(df) for df in valid_data.values())
        avg_bars = total_bars / len(valid_data)
        log.info(f"  Total bars: {total_bars}")
        log.info(f"  Average bars per contract: {avg_bars:.0f}")

    # ========================================================================
    # ШАГ 3: GRID SEARCH ПО β-КОЭФФИЦИЕНТАМ
    # ========================================================================
    if verbose:
        log.info(f"\n[STEP 3/4] Running grid search...")
        log.info(f"  Horizons: {tuning_config.horizon_days_grid}")
        log.info(f"  Beta candidates: {len(tuning_config.beta_grid)}")
        log.info(f"  Rolling windows: {tuning_config.rolling_windows}")

    tuning_engine = HARParameterTuningEngine(tuning_config, tuning_config.tdy)
    results_by_horizon = tuning_engine.run_grid_search(valid_data)

    if not results_by_horizon:
        log.error("✗ No tuning results, aborting")
        return {}

    # ========================================================================
    # ШАГ 4: АГРЕГАЦИЯ РЕЗУЛЬТАТОВ И ВЫБОР BEST β
    # ========================================================================
    if verbose:
        log.info(f"\n[STEP 4/4] Aggregating results...")

    summary = tuning_engine.aggregate_results(results_by_horizon)

    # Добавляем метаданные в summary
    summary_with_meta = {
        "metadata": {
            "date": str(today),
            "contracts_requested": len(tuning_config.contract_roles),
            "contracts_loaded": len(valid_data),
            "missing_roles": missing_roles,
            "horizons": tuning_config.horizon_days_grid,
            "beta_candidates": len(tuning_config.beta_grid),
            "rolling_windows": tuning_config.rolling_windows,
            "contract_details": {
                role: {
                    "ticker": contracts_dict[role].ticker,
                    "expiry": str(contracts_dict[role].expiry)
                }
                for role in tuning_config.contract_roles
                if role in contracts_dict
            }
        },
        "results": summary
    }

    # Сохранение результатов
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = today.strftime("%Y%m%d")
        json_file = output_path / f"har_tuning_results_{timestamp}.json"

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_with_meta, f, indent=2, ensure_ascii=False)

        if verbose:
            log.info(f"✓ Results saved to {json_file}")

    # Final summary
    if verbose:
        log.info(f"\n{'=' * 80}")
        log.info(f"HAR PARAMETER TUNING COMPLETE")
        log.info(f"✓ Optimized {len(summary)} horizons")

        for horizon, data in summary.items():
            best = data['best_params']
            metrics = data['best_metrics']
            stats = data['aggregate_stats']
            log.info(f"\nHorizon {horizon}d:")
            log.info(f"  Best β: [{best['beta_d']:.3f}, {best['beta_w']:.3f}, {best['beta_m']:.3f}]")
            log.info(f"  Test R²: {metrics['test_r2']:.4f}")
            log.info(f"  Test RMSE: {metrics['test_rmse']:.6f}")
            log.info(f"  Improvement: {metrics['improvement_r2_pp']:.2f}pp vs EWMA")
            log.info(f"  Contracts: {stats['n_contracts']}")

        log.info(f"{'=' * 80}")

    return summary


def run_full_har_parameter_tuning_auto():
    """
    Wrapper для run_full_har_parameter_tuning() с автоматическим созданием всех параметров.
    Использует стандартную конфигурацию для быстрого теста (3 контракта, 2 горизонта).

    Usage:
        >>> run_full_har_parameter_tuning_auto()
    """
    # Import from globals (Colab namespace)
    g = globals()
    MoexClient = g['MoexClient']
    MoexConfig = g['MoexConfig']
    HARTuningConfig = g['HARTuningConfig']

    log.info(f"\n{'=' * 80}")
    log.info(f"AUTO UNIFIED RUNNER (3 contracts: F0, F-1, F-2)")
    log.info(f"{'=' * 80}")

    # Создаем все необходимые объекты
    config = HARTuningConfig(
        contract_roles=["F0", "F-1", "F-2"],
        horizon_days_grid=[10, 21],
        beta_grid=[
            (0.35, 0.40, 0.25),
            (0.40, 0.35, 0.25),
            (0.33, 0.33, 0.34)
        ],
        rolling_windows=1,
        duration_days=180,
        min_bars_per_contract=100,
        verbose=True
    )

    client = MoexClient(MoexConfig())

    # Запускаем полный workflow
    results = run_full_har_parameter_tuning(
        client,
        MoexConfig(),
        date.today(),
        config,
        output_dir="./quick_unified_test",
        save_results=True,
        verbose=True
    )

    return results


def quick_unified_example():
    """
    Быстрый пример для тестирования unified runner на 3 контрактах (F0, F-1, F-2).
    Алиас для run_full_har_parameter_tuning_auto() для обратной совместимости.

    Usage:
        >>> quick_unified_example()
    """
    return run_full_har_parameter_tuning_auto()


def full_unified_example():
    """
    Полный пример для всех 11 контрактов (F0, F-1, ..., F-10).

    Usage:
        >>> full_unified_example()
    """
    # Import from globals (Colab namespace)
    g = globals()
    MoexClient = g['MoexClient']
    MoexConfig = g['MoexConfig']
    HARTuningConfig = g['HARTuningConfig']

    log.info(f"\n{'=' * 80}")
    log.info(f"FULL UNIFIED EXAMPLE (11 contracts: F0, F-1, ..., F-10)")
    log.info(f"{'=' * 80}")

    # Full config для production
    config = HARTuningConfig(
        contract_roles=["F0", "F-1", "F-2", "F-3", "F-4", "F-5", "F-6", "F-7", "F-8", "F-9", "F-10"],
        horizon_days_grid=[5, 10, 21, 60],  # 4 горизонта
        beta_grid=[
            (0.35, 0.40, 0.25),
            (0.50, 0.30, 0.20),
            (0.25, 0.50, 0.25),
            (0.20, 0.40, 0.40),
            (0.33, 0.33, 0.34),
            (0.40, 0.35, 0.25),
            (0.30, 0.30, 0.40)
        ],  # 7 кандидатов
        rolling_windows=5,  # 5 итераций
        duration_days=365,  # 1 год данных
        min_bars_per_contract=252,  # Требуем минимум 1 год торговых дней
        verbose=True
    )

    # Run unified workflow
    client = MoexClient(MoexConfig())
    results = run_full_har_parameter_tuning(
        client,
        MoexConfig(),
        date.today(),
        config,
        output_dir="./full_unified_results",
        save_results=True,
        verbose=True
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    print("✓ Cell 7.6.8: HAR Unified Parameter Tuning Runner загружен успешно")
    print(f"  - Functions:")
    print(f"    • run_full_har_parameter_tuning() - полный workflow (требует параметры)")
    print(f"    • run_full_har_parameter_tuning_auto() - автоматический запуск")
    print(f"    • quick_unified_example() - алиас для auto версии")
    print(f"    • full_unified_example() - полный запуск на 11 контрактах")
    print(f"\nДля запуска:")
    print(f"  >>> run_full_har_parameter_tuning_auto()  # Быстрый тест (рекомендуется)")
    print(f"  >>> quick_unified_example()              # То же самое")
    print(f"  >>> full_unified_example()               # Полный анализ")
