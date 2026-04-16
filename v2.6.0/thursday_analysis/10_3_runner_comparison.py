"""
10_3_runner_comparison.py
Cell ID: X4TsUjodW-Tr
Exported: 2026-04-16T10:12:23.219115
"""

"""
Cell 10.3: 5d-for-Thursday vs Standard 5d Comparison Runner
============================================================

Запускает оптимизацию для всех загруженных контрактов и сравнивает:
- Standard horizon=5d (абстрактный прогноз на 5 дней)
- Thursday-aligned 5d (прогноз с пятницы до четверга)

Data Source: MOEX ISS API (https://iss.moex.com/iss)

Usage в Colab:
    # 1. Загрузить все prerequisite ячейки
    exec(open('cell_10_1_thursday_helpers.py').read())
    exec(open('cell_10_2_5d_thursday_optimization.py').read())
    exec(open('cell_10_3_runner_and_comparison.py').read())

    # 2. Запустить сравнение
    results = run_5d_thursday_comparison(
        moex_client=client,
        contracts=si_contracts,
        duration_days=365,
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

log = logging.getLogger("5d_thursday_comparison")
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
    "cache_dir": "/content/data/har_tuning_cache"  # Используем существующий кэш
}

# ============================================================================
# MAIN COMPARISON RUNNER
# ============================================================================

def run_5d_thursday_comparison(
    moex_client,
    contracts: List,
    duration_days: int = 365,
    bar_size: str = "1 hour",
    save_results: bool = True,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main entrypoint для сравнения 5d_for_thursday vs standard 5d.

    Workflow:
    1. Load historical data for all contracts via MOEX ISS API
    2. Run BOTH optimizations for each contract:
       a. Standard horizon=5d (baseline)
       b. Thursday-aligned 5d (new method)
    3. Compare beta coefficients and metrics
    4. Generate comparative report

    Args:
        moex_client: MoexClient instance (from cell_02_moex_contracts.py)
        contracts: List of contract metadata dicts (from identify_contracts)
        duration_days: Historical data lookback
        bar_size: MOEX candle interval (e.g., "1 hour")
        save_results: Save JSON to disk
        output_dir: Output directory
        verbose: Detailed logging

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
    print("5D-FOR-THURSDAY vs STANDARD 5D COMPARISON")
    print("="*80)
    print(f"Contracts: {len(contracts)}")
    print(f"Duration: {duration_days} days")
    print(f"Bar size: {bar_size}")
    print(f"Data source: MOEX ISS API")
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
        "BacktestDataPipeline": "Cell 07.2.09a",
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
    # STEP 2: Check cache and load historical data
    # ========================================================================

    cache_dir = COMPARISON_CONFIG["cache_dir"]
    print(f"\n[STEP 2/5] Checking cache: {cache_dir}...")

    # Проверка существования кэша
    from pathlib import Path
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        print(f"\nHint: Run the data loading cell first to download historical data.")
        print(f"Expected location: {cache_dir}")
        return {"status": "error", "message": "Cache directory missing", "cache_dir": cache_dir}

    # Проверка наличия файлов в кэше
    cached_files = list(cache_path.glob("*.parquet"))

    if not cached_files:
        print(f"❌ No cached data found in: {cache_dir}")
        print(f"\nHint: Run the data loading cell first to download historical data.")
        return {"status": "error", "message": "Cache is empty", "cache_dir": cache_dir}

    print(f"✓ Cache directory found: {cache_dir}")
    print(f"✓ Cached files available: {len(cached_files)} contracts")

    # Загружаем данные через BacktestDataPipeline с правильным cache_dir
    print(f"\n[STEP 2/5] Loading historical data from cache...")

    BacktestDataPipeline = globals()["BacktestDataPipeline"]
    pipeline = BacktestDataPipeline(moex_client, tdy, cache_dir=cache_dir)

    portfolio_data = pipeline.load_portfolio_history(
        contracts=contracts,
        duration_days=duration_days,
        bar_size=bar_size
    )

    if not portfolio_data:
        print("❌ No data loaded (cache read failed or data validation failed)")
        return {"status": "error", "message": "Data loading failed"}

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
            "data_source": "MOEX ISS API",
            "moex_endpoint": moex_client.cfg.base_url if hasattr(moex_client, 'cfg') else "https://iss.moex.com/iss",
            "contracts": [c.get('symbol', str(c)) for c in contracts],
            "duration_days": duration_days,
            "bar_size": bar_size
        },
        "baseline_5d": baseline_results,
        "thursday_5d": thursday_results,
        "comparison": {
            "table": comparison_table,
            "aggregate_stats": aggregate_stats
        },
        "summary": {
            "total_contracts": len(contracts),
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


# ============================================================================
# USAGE EXAMPLE (commented out)
# ============================================================================

"""
# После загрузки всех ячеек, запускаем:

from cell_02_moex_contracts import MoexClient, MoexConfig, identify_contracts
from datetime import date

# 1. Инициализируем MOEX клиент
client = MoexClient(MoexConfig())
print(f"MOEX ISS API endpoint: {client.cfg.base_url}")

# 2. Определяем контракты
today = date.today()
contracts_dict = identify_contracts(client, client.cfg, today)

# Выбираем контракты для анализа (F0-F10)
si_contracts = [
    contracts_dict['F0'],   # Last expired
    contracts_dict['F-1'],  # Previous expired
    contracts_dict['F-2'],
    # ... добавь больше для статистики
]

# 3. Запускаем сравнение
results = run_5d_thursday_comparison(
    moex_client=client,
    contracts=si_contracts,
    duration_days=365,
    bar_size="1 hour",
    save_results=True,
    verbose=True
)

# 4. Анализируем результаты
print("\nMedian Beta Comparison:")
print(f"Baseline:  {results['comparison']['aggregate_stats']['median_baseline_betas']}")
print(f"Thursday:  {results['comparison']['aggregate_stats']['median_thursday_betas']}")
print(f"Delta:     {results['comparison']['aggregate_stats']['median_delta']}")
"""

if __name__ == "__main__":
    print("✓ Cell 10.3: 5d Thursday Comparison Runner loaded into globals()")
