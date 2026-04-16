"""
10_3b_cache_comparison.py
Cell ID: W94CamopX0yn
Exported: 2026-04-16T10:12:23.219147
"""

"""
Cell 10.3b: 5d Thursday Comparison Runner (Cache-Only Mode)
============================================================

Запускает оптимизацию БЕЗ live MOEX API — работает только с кэшем.

Отличия от Cell 10.3:
- НЕ требует moex_client (работает только с кэшем)
- НЕ скачивает данные через MOEX ISS API
- Читает готовые parquet файлы из /content/data/har_tuning_cache

Usage в Colab:
    # 1. Загрузить prerequisite ячейки
    exec(open('cell_10_1_thursday_helpers.py').read())
    exec(open('cell_10_2_5d_thursday_optimization.py').read())
    exec(open('cell_10_3b_cache_only_comparison.py').read())

    # 2. Запустить сравнение БЕЗ moex_client
    results = run_5d_thursday_comparison_from_cache(
        cache_dir="/content/data/har_tuning_cache",
        duration_days=730,
        save_results=True
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm.auto import tqdm

log = logging.getLogger("5d_thursday_cache_comparison")
log.setLevel(logging.INFO)

# ============================================================================
# CONFIGURATION
# ============================================================================

COMPARISON_CONFIG = {
    "output_dir": "/content/thursday_optimization_results",
    "trading_days_per_year": 252,
    "train_test_split": 0.60,
    "save_json": True,
    "verbose": True,
    "cache_dir": "/content/data/har_tuning_cache"
}

# ============================================================================
# CACHE-ONLY DATA LOADER
# ============================================================================

class CachedDataLoader:
    """
    Загружает данные ТОЛЬКО из parquet кэша (без live MOEX API).
    """

    def __init__(self, cache_dir: str, trading_days_per_year: int = 252):
        """
        Args:
            cache_dir: путь к директории с parquet файлами
            trading_days_per_year: торговых дней в году для аннуализации
        """
        self.cache_dir = Path(cache_dir)
        self.tdy = trading_days_per_year
        self._cache: Dict[str, pd.DataFrame] = {}

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    def load_contract_from_cache(
        self,
        symbol: str,
        duration_days: int = 365,
        verbose: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Загружает данные для контракта из parquet кэша.

        Args:
            symbol: тикер контракта (e.g., "SiU4")
            duration_days: ожидаемая длительность данных (для поиска файла)
            verbose: логирование

        Returns:
            DataFrame с OHLCV или None если файл не найден
        """
        cache_key = f"{symbol}_{duration_days}"

        # Проверяем in-memory кэш
        if cache_key in self._cache:
            if verbose:
                log.info(f"✓ Using in-memory cache for {symbol}")
            return self._cache[cache_key]

        # Ищем parquet файл: {symbol}_{duration_days}d.parquet
        parquet_file = self.cache_dir / f"{symbol}_{duration_days}d.parquet"

        if not parquet_file.exists():
            # Пробуем альтернативные имена файлов
            alternatives = list(self.cache_dir.glob(f"{symbol}_*.parquet"))
            if alternatives:
                parquet_file = alternatives[0]
                if verbose:
                    log.info(f"✓ Found alternative cache file: {parquet_file.name}")
            else:
                if verbose:
                    log.warning(f"✗ Cache file not found for {symbol}: {parquet_file}")
                return None

        try:
            df = pd.read_parquet(parquet_file)

            # Валидация минимальных требований
            if len(df) < 60:
                if verbose:
                    log.warning(f"✗ Insufficient data in cache for {symbol}: {len(df)} bars (min 60)")
                return None

            if verbose:
                log.info(f"✓ Loaded {len(df)} bars for {symbol} from cache")
                log.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

            # Кэшируем в памяти
            self._cache[cache_key] = df

            return df

        except Exception as e:
            log.error(f"✗ Failed to read cache for {symbol}: {e}")
            return None

    def load_portfolio_from_cache(
        self,
        symbols: List[str],
        duration_days: int = 365,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Загружает данные для портфеля контрактов из кэша.

        Args:
            symbols: список тикеров (e.g., ["SiU4", "SiZ4", ...])
            duration_days: ожидаемая длительность данных
            verbose: логирование

        Returns:
            Dict {symbol: DataFrame} для успешно загруженных контрактов
        """
        if verbose:
            log.info(f"\n{'=' * 80}")
            log.info(f"LOADING PORTFOLIO FROM CACHE")
            log.info(f"Cache directory: {self.cache_dir}")
            log.info(f"Contracts: {len(symbols)}")
            log.info(f"Duration: {duration_days} days")
            log.info(f"{'=' * 80}")

        portfolio_data = {}
        failed_symbols = []

        for i, symbol in enumerate(symbols, 1):
            if verbose:
                log.info(f"\n[{i}/{len(symbols)}] Loading {symbol} from cache...")

            df = self.load_contract_from_cache(
                symbol=symbol,
                duration_days=duration_days,
                verbose=verbose
            )

            if df is not None:
                portfolio_data[symbol] = df
            else:
                failed_symbols.append(symbol)

        # Итоговая статистика
        if verbose:
            log.info(f"\n{'=' * 80}")
            log.info(f"CACHE LOADING SUMMARY")
            log.info(f"Successful: {len(portfolio_data)}/{len(symbols)} contracts")
            if failed_symbols:
                log.info(f"Failed: {', '.join(failed_symbols)}")
            log.info(f"{'=' * 80}\n")

        return portfolio_data

# ============================================================================
# MAIN COMPARISON RUNNER (CACHE-ONLY)
# ============================================================================

def run_5d_thursday_comparison_from_cache(
    cache_dir: str = "/content/data/har_tuning_cache",
    symbols: Optional[List[str]] = None,
    duration_days: int = 730,
    save_results: bool = True,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main entrypoint для сравнения 5d_for_thursday vs standard 5d (CACHE-ONLY).

    Workflow:
    1. Читает исторические данные ИЗ КЭША (НЕ скачивает через MOEX API)
    2. Запускает обе оптимизации для каждого контракта:
       a. Standard horizon=5d (baseline)
       b. Thursday-aligned 5d (new method)
    3. Сравнивает beta коэффициенты и метрики
    4. Генерирует отчёт

    Args:
        cache_dir: путь к директории с parquet файлами
        symbols: список тикеров (e.g., ["SiU4", "SiZ4"]). Если None — загружает все из кэша
        duration_days: ожидаемая длительность данных в кэше
        save_results: сохранить JSON на диск
        output_dir: директория для сохранения результатов
        verbose: детальное логирование

    Returns:
        {
            "meta": {...},
            "baseline_5d": {results for standard horizon=5},
            "thursday_5d": {results for 5d_for_thursday},
            "comparison": {...},
            "summary": {...}
        }
    """

    print("\n" + "="*80)
    print("5D-FOR-THURSDAY vs STANDARD 5D COMPARISON (CACHE-ONLY)")
    print("="*80)
    print(f"Cache directory: {cache_dir}")
    print(f"Duration: {duration_days} days")
    print(f"Data source: LOCAL PARQUET CACHE (no live API)")
    print("="*80 + "\n")

    # Setup
    if output_dir is None:
        output_dir = COMPARISON_CONFIG["output_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tdy = COMPARISON_CONFIG["trading_days_per_year"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================================================
    # STEP 1: Validate dependencies
    # ========================================================================

    print("\n[STEP 1/5] Validating dependencies...")

    required_functions = {
        "yang_zhang_series": "Cell 03",
        "optimize_beta_coefficients": "Cell 07.2.07",
        "compute_realized_vol_forward": "Cell 07.2.03",
        "forecast_har_rolling": "Cell 07.2.04",
        "evaluate_forecast": "Cell 07.2.06",
        "find_next_thursday": "Cell 10.1",
        "compute_realized_vol_to_thursday": "Cell 10.1",
        "optimize_5d_for_thursday_single_contract": "Cell 10.2"
    }

    validation_errors = []

    for fn_name, source in tqdm(required_functions.items(), desc="Checking dependencies"):
        if fn_name not in globals():
            validation_errors.append(f"  ✗ {fn_name} (from {source}) NOT FOUND")

    if validation_errors:
        print("\n❌ DEPENDENCY CHECK FAILED:")
        for err in validation_errors:
            print(err)
        print("\nHint: Make sure all prerequisite cells are loaded")
        return {"status": "error", "message": "Missing dependencies", "errors": validation_errors}

    print("✓ All dependencies validated")

    # ========================================================================
    # STEP 2: Load historical data from cache
    # ========================================================================

    print(f"\n[STEP 2/5] Loading historical data from cache...")

    loader = CachedDataLoader(cache_dir=cache_dir, trading_days_per_year=tdy)

    # Если symbols не указан — берём все файлы из кэша
    if symbols is None:
        cache_path = Path(cache_dir)
        parquet_files = list(cache_path.glob("*.parquet"))

        # Извлекаем symbol из имени файла: {symbol}_{duration}d.parquet
        symbols = []
        for pf in parquet_files:
            parts = pf.stem.split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                symbols.append(symbol)

        # Удаляем дубликаты
        symbols = sorted(set(symbols))

        print(f"✓ Auto-detected {len(symbols)} symbols from cache: {', '.join(symbols)}")

    portfolio_data = loader.load_portfolio_from_cache(
        symbols=symbols,
        duration_days=duration_days,
        verbose=verbose
    )

    if not portfolio_data:
        print("❌ No data loaded from cache")
        return {"status": "error", "message": "Cache loading failed"}

    print(f"✓ Loaded data for {len(portfolio_data)} contracts from cache")
    for symbol, df in portfolio_data.items():
        print(f"  {symbol}: {len(df)} bars ({df['date'].min()} to {df['date'].max()})")

    # ========================================================================
    # STEP 3: Run BASELINE optimization (standard horizon=5)
    # ========================================================================

    print("\n[STEP 3/5] Running BASELINE optimization (standard horizon=5)...")

    optimize_beta_coefficients = globals()["optimize_beta_coefficients"]
    compute_realized_vol_forward = globals()["compute_realized_vol_forward"]
    forecast_har_rolling = globals()["forecast_har_rolling"]
    evaluate_forecast = globals()["evaluate_forecast"]
    yang_zhang_series = globals()["yang_zhang_series"]

    baseline_results = []

    for symbol, df in tqdm(portfolio_data.items(), desc="Baseline 5d optimization"):
        try:
            # Split train/test
            split_idx = int(len(df) * COMPARISON_CONFIG["train_test_split"])
            df_train = df.iloc[:split_idx].copy()

            # Optimize on train set
            beta_d, beta_w, beta_m, metrics = optimize_beta_coefficients(
                df=df_train,
                horizon_days=5,
                compute_realized_vol_fn=compute_realized_vol_forward,
                forecast_har_fn=forecast_har_rolling,
                evaluate_forecast_fn=evaluate_forecast,
                yang_zhang_series_fn=yang_zhang_series,
                tdy=tdy,
                verbose=False
            )

            baseline_results.append({
                "symbol": symbol,
                "horizon_type": "standard_5d",
                "status": "success",
                "optimal_betas": {
                    "daily": round(beta_d, 4),
                    "weekly": round(beta_w, 4),
                    "monthly": round(beta_m, 4)
                },
                "in_sample_metrics": {
                    "R2": round(metrics.get("R2", 0), 4),
                    "RMSE": round(metrics.get("RMSE", 0), 6),
                    "MAE": round(metrics.get("MAE", 0), 6),
                    "n_obs": metrics.get("n_obs", 0)
                },
                "data_quality": {
                    "total_bars": len(df),
                    "train_bars": len(df_train)
                }
            })

        except Exception as e:
            log.error(f"  ✗ {symbol} baseline optimization failed: {e}")
            baseline_results.append({
                "symbol": symbol,
                "horizon_type": "standard_5d",
                "status": "failed",
                "error": str(e)
            })

    print(f"✓ Baseline optimization complete: {len([r for r in baseline_results if r['status'] == 'success'])}/{len(baseline_results)} successful")

    # ========================================================================
    # STEP 4: Run THURSDAY-ALIGNED optimization (5d_for_thursday)
    # ========================================================================

    print("\n[STEP 4/5] Running THURSDAY-ALIGNED optimization (5d_for_thursday)...")

    optimize_5d_for_thursday_single_contract = globals()["optimize_5d_for_thursday_single_contract"]
    compute_realized_vol_to_thursday = globals()["compute_realized_vol_to_thursday"]

    thursday_results = []

    for symbol, df in tqdm(portfolio_data.items(), desc="Thursday 5d optimization"):
        try:
            result = optimize_5d_for_thursday_single_contract(
                df=df,
                symbol=symbol,
                yang_zhang_series_fn=yang_zhang_series,
                compute_realized_vol_to_thursday_fn=compute_realized_vol_to_thursday,
                tdy=tdy,
                train_split=COMPARISON_CONFIG["train_test_split"],
                verbose=False
            )

            thursday_results.append(result)

        except Exception as e:
            log.error(f"  ✗ {symbol} thursday optimization failed: {e}")
            thursday_results.append({
                "symbol": symbol,
                "horizon_type": "5d_for_thursday",
                "status": "failed",
                "error": str(e)
            })

    print(f"✓ Thursday-aligned optimization complete: {len([r for r in thursday_results if r['status'] == 'success'])}/{len(thursday_results)} successful")

    # ========================================================================
    # STEP 5: Generate comparison report
    # ========================================================================

    print("\n[STEP 5/5] Generating comparison report...")

    comparison_table = []

    for baseline, thursday in zip(baseline_results, thursday_results):
        symbol = baseline["symbol"]

        if baseline["status"] == "success" and thursday["status"] == "success":
            beta_baseline = baseline["optimal_betas"]
            beta_thursday = thursday["optimal_betas"]

            comparison_table.append({
                "symbol": symbol,
                "baseline_beta_d": beta_baseline["daily"],
                "baseline_beta_w": beta_baseline["weekly"],
                "baseline_beta_m": beta_baseline["monthly"],
                "thursday_beta_d": beta_thursday["daily"],
                "thursday_beta_w": beta_thursday["weekly"],
                "thursday_beta_m": beta_thursday["monthly"],
                "delta_beta_d": round(beta_thursday["daily"] - beta_baseline["daily"], 4),
                "delta_beta_w": round(beta_thursday["weekly"] - beta_baseline["weekly"], 4),
                "delta_beta_m": round(beta_thursday["monthly"] - beta_baseline["monthly"], 4),
                "baseline_R2": baseline["in_sample_metrics"]["R2"],
                "thursday_R2": thursday["in_sample_metrics"]["R2"],
                "n_fridays": thursday["data_quality"].get("fridays_in_train", 0)
            })

    comparison_df = pd.DataFrame(comparison_table)

    # Aggregate statistics
    if not comparison_df.empty:
        aggregate_stats = {
            "median_baseline_betas": {
                "daily": float(comparison_df["baseline_beta_d"].median()),
                "weekly": float(comparison_df["baseline_beta_w"].median()),
                "monthly": float(comparison_df["baseline_beta_m"].median())
            },
            "median_thursday_betas": {
                "daily": float(comparison_df["thursday_beta_d"].median()),
                "weekly": float(comparison_df["thursday_beta_w"].median()),
                "monthly": float(comparison_df["thursday_beta_m"].median())
            },
            "median_delta": {
                "daily": float(comparison_df["delta_beta_d"].median()),
                "weekly": float(comparison_df["delta_beta_w"].median()),
                "monthly": float(comparison_df["delta_beta_m"].median())
            },
            "significant_differences": {
                "beta_d_changed": int((comparison_df["delta_beta_d"].abs() > 0.02).sum()),
                "beta_w_changed": int((comparison_df["delta_beta_w"].abs() > 0.02).sum()),
                "beta_m_changed": int((comparison_df["delta_beta_m"].abs() > 0.02).sum())
            }
        }
    else:
        aggregate_stats = {}

    # ========================================================================
    # Package final output
    # ========================================================================

    output = {
        "meta": {
            "timestamp": timestamp,
            "comparison_config": COMPARISON_CONFIG,
            "data_source": "LOCAL PARQUET CACHE (no live API)",
            "cache_dir": cache_dir,
            "symbols": symbols,
            "duration_days": duration_days
        },
        "baseline_5d": baseline_results,
        "thursday_5d": thursday_results,
        "comparison": {
            "table": comparison_table,
            "aggregate_stats": aggregate_stats
        },
        "summary": {
            "total_contracts": len(symbols),
            "baseline_successful": len([r for r in baseline_results if r["status"] == "success"]),
            "thursday_successful": len([r for r in thursday_results if r["status"] == "success"]),
            "both_successful": len(comparison_table)
        }
    }

    # ========================================================================
    # Save to JSON
    # ========================================================================

    if save_results:
        json_path = Path(output_dir) / f"5d_thursday_comparison_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {json_path}")

    # ========================================================================
    # Print summary report
    # ========================================================================

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Total contracts: {output['summary']['total_contracts']}")
    print(f"Baseline successful: {output['summary']['baseline_successful']}")
    print(f"Thursday successful: {output['summary']['thursday_successful']}")
    print(f"Both successful: {output['summary']['both_successful']}")

    if not comparison_df.empty:
        print("\n" + "-"*80)
        print("AGGREGATE BETA COMPARISON (median across contracts):")
        print("-"*80)
        print(f"Baseline 5d:        β_d={aggregate_stats['median_baseline_betas']['daily']:.3f}, "
              f"β_w={aggregate_stats['median_baseline_betas']['weekly']:.3f}, "
              f"β_m={aggregate_stats['median_baseline_betas']['monthly']:.3f}")
        print(f"Thursday 5d:        β_d={aggregate_stats['median_thursday_betas']['daily']:.3f}, "
              f"β_w={aggregate_stats['median_thursday_betas']['weekly']:.3f}, "
              f"β_m={aggregate_stats['median_thursday_betas']['monthly']:.3f}")
        print(f"Median delta:       Δβ_d={aggregate_stats['median_delta']['daily']:+.3f}, "
              f"Δβ_w={aggregate_stats['median_delta']['weekly']:+.3f}, "
              f"Δβ_m={aggregate_stats['median_delta']['monthly']:+.3f}")

        print("\n" + "-"*80)
        print("DETAILED COMPARISON BY CONTRACT:")
        print("-"*80)
        print(comparison_df.to_string(index=False))

        print("\n" + "-"*80)
        print("SIGNIFICANT DIFFERENCES (|Δβ| > 0.02):")
        print("-"*80)
        print(f"  β_d changed: {aggregate_stats['significant_differences']['beta_d_changed']}/{len(comparison_df)} contracts")
        print(f"  β_w changed: {aggregate_stats['significant_differences']['beta_w_changed']}/{len(comparison_df)} contracts")
        print(f"  β_m changed: {aggregate_stats['significant_differences']['beta_m_changed']}/{len(comparison_df)} contracts")

    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETE")
    print("="*80)

    # ========================================================================
    # SELF-CHECK: Validate results integrity
    # ========================================================================

    print("\n[SELF-CHECK] Validating results integrity...")

    checks_passed = 0
    checks_failed = 0

    # Check 1: All betas sum to 1.0
    for baseline in baseline_results:
        if baseline["status"] == "success":
            beta_sum = sum(baseline["optimal_betas"].values())
            if abs(beta_sum - 1.0) < 0.01:
                checks_passed += 1
            else:
                checks_failed += 1
                print(f"  ⚠ {baseline['symbol']} baseline: beta sum = {beta_sum:.3f} (expected 1.0)")

    for thursday in thursday_results:
        if thursday["status"] == "success":
            beta_sum = sum(thursday["optimal_betas"].values())
            if abs(beta_sum - 1.0) < 0.01:
                checks_passed += 1
            else:
                checks_failed += 1
                print(f"  ⚠ {thursday['symbol']} thursday: beta sum = {beta_sum:.3f} (expected 1.0)")

    # Check 2: All R² values in valid range [0, 1]
    for baseline in baseline_results:
        if baseline["status"] == "success":
            r2 = baseline["in_sample_metrics"]["R2"]
            if 0 <= r2 <= 1:
                checks_passed += 1
            else:
                checks_failed += 1
                print(f"  ⚠ {baseline['symbol']} baseline: R² = {r2:.3f} (out of range)")

    for thursday in thursday_results:
        if thursday["status"] == "success":
            r2 = thursday["in_sample_metrics"]["R2"]
            if 0 <= r2 <= 1:
                checks_passed += 1
            else:
                checks_failed += 1
                print(f"  ⚠ {thursday['symbol']} thursday: R² = {r2:.3f} (out of range)")

    # Check 3: Comparison table consistency
    if len(comparison_table) != output['summary']['both_successful']:
        checks_failed += 1
        print(f"  ⚠ Comparison table size mismatch: {len(comparison_table)} vs {output['summary']['both_successful']}")
    else:
        checks_passed += 1

    print(f"\n{'='*80}")
    print(f"Self-checks: {checks_passed} passed, {checks_failed} failed")

    if checks_failed == 0:
        print("✓ ALL SELF-CHECKS PASSED - Results are valid")
    else:
        print(f"⚠ {checks_failed} SELF-CHECKS FAILED - Review results carefully")

    print("="*80 + "\n")

    return output


if __name__ == "__main__":
    print("✓ Cell 10.3b: 5d Thursday Comparison Runner (Cache-Only) loaded into globals()")
