"""
07_6_3_portfolio_runner.py
Cell ID: UaJZgbXFPp-C
Exported: 2026-04-16T10:12:23.218962
"""

#!/usr/bin/env python3
"""
Cell 07.6.3: HAR Parameter Optimization - Portfolio Runner
===========================================================

Multi-contract portfolio optimization orchestrator.

Execution Environment: Google Colab
Prerequisites: Cells 01-04, 07.2.01-09b, 07.6.1-2 должны быть загружены

Usage в Colab:
    exec(open('cell_07_6_3_portfolio_runner.py').read())
    test_cell_07_6_3()  # Should return True
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Настройка логирования
log = logging.getLogger("har_portfolio_runner")
log.setLevel(logging.INFO)

# ============================================================================
# PORTFOLIO OPTIMIZATION RUNNER
# ============================================================================

def run_har_parameter_optimization(
    moex_client,
    contracts: List,
    horizons: Optional[List[int]] = None,
    duration_days: int = 365,
    bar_size: str = "1 hour",
    save_results: bool = True,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main entrypoint для HAR parameter optimization.

    Workflow:
    1. Load historical data for all contracts (via MOEX ISS API)
    2. For each contract × horizon combination:
       - Run rolling optimization (Cell 07.6.2)
       - Validate out-of-sample
    3. Aggregate results across contracts (median betas)
    4. Identify optimal betas for each horizon
    5. Save results to JSON

    Args:
        moex_client: MoexClient instance (из cell_02_moex_contracts.py)
        contracts: List of MOEX contract metadata objects
        horizons: List of forecast horizons (default: [1, 2, 3, 5, 7])
        duration_days: Historical data lookback (default: 365)
        bar_size: MOEX candle interval (default: "1 hour")
        save_results: Save JSON to disk
        output_dir: Output directory (default: /content/har_optimization_results)
        verbose: Detailed logging

    Returns:
        {
            "meta": {...},
            "optimization_results": {
                "horizon_1d": {
                    "optimal_betas_median": {...},
                    "oos_performance": {...},
                    "by_contract": [...]
                },
                ...
            },
            "summary": {...}
        }
    """

    if verbose:
        log.info("\n" + "="*80)
        log.info("HAR PARAMETER OPTIMIZATION RUNNER")
        log.info("="*80)
        log.info(f"Contracts: {len(contracts)}")
        log.info(f"Horizons: {horizons or 'default'}")
        log.info(f"Duration: {duration_days} days")
        log.info(f"Bar size: {bar_size}")
        log.info("="*80)

    # ========================================================================
    # STEP 0: Load configuration
    # ========================================================================

    try:
        OPTIMIZATION_CONFIG = globals()["OPTIMIZATION_CONFIG"]
        optimize_single_horizon_rolling = globals()["optimize_single_horizon_rolling"]
        BacktestDataPipeline = globals()["BacktestDataPipeline"]
    except KeyError as e:
        log.error(f"✗ Missing required dependency: {e}")
        log.error("  Hint: Run Cell 07.6.1, 07.6.2, and 07.2.09a first")
        return {"status": "error", "message": f"Missing dependency: {e}"}

    # Default parameters
    if horizons is None:
        horizons = OPTIMIZATION_CONFIG["default_horizons"]

    if output_dir is None:
        output_dir = OPTIMIZATION_CONFIG["output_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tdy = OPTIMIZATION_CONFIG["trading_days_per_year"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================================================
    # STEP 1: Load historical data
    # ========================================================================

    log.info("\n[STEP 1/4] Loading historical data via MOEX ISS API...")

    pipeline = BacktestDataPipeline(moex_client, tdy)

    portfolio_data = pipeline.load_portfolio_history(
        contracts=contracts,
        duration_days=duration_days,
        bar_size=bar_size
    )

    if not portfolio_data:
        log.error("✗ No data loaded from MOEX ISS API")
        return {"status": "error", "message": "Data loading failed"}

    log.info(f"✓ Loaded data for {len(portfolio_data)} contracts")
    for symbol, df in portfolio_data.items():
        log.info(f"  {symbol}: {len(df)} bars")

    # ========================================================================
    # STEP 2: Run optimization for all contract × horizon combinations
    # ========================================================================

    log.info("\n[STEP 2/4] Running optimization for all horizons...")

    optimization_results = {f"horizon_{h}d": [] for h in horizons}

    for horizon in horizons:
        log.info(f"\n--- HORIZON: {horizon}d ---")

        for symbol, df in portfolio_data.items():
            result = optimize_single_horizon_rolling(
                df=df,
                symbol=symbol,
                horizon_days=horizon,
                tdy=tdy,
                train_split=OPTIMIZATION_CONFIG["train_test_split"],
                rolling_step=OPTIMIZATION_CONFIG["rolling_step"],
                verbose=verbose
            )

            optimization_results[f"horizon_{horizon}d"].append(result)

    # ========================================================================
    # STEP 3: Aggregate results (median betas across contracts)
    # ========================================================================

    log.info("\n[STEP 3/4] Aggregating results...")

    aggregated = {}

    for horizon_key, results in optimization_results.items():
        # Filter successful results
        successful = [r for r in results if r.get("status") == "success"]

        if not successful:
            log.warning(f"  ⚠ {horizon_key}: No successful optimizations")
            aggregated[horizon_key] = {
                "status": "no_successful_runs",
                "n_contracts": len(results),
                "n_successful": 0
            }
            continue

        # Extract betas
        beta_d_list = [r["optimal_betas"]["daily"] for r in successful]
        beta_w_list = [r["optimal_betas"]["weekly"] for r in successful]
        beta_m_list = [r["optimal_betas"]["monthly"] for r in successful]

        # Compute median (robust to outliers)
        beta_d_median = float(np.median(beta_d_list))
        beta_w_median = float(np.median(beta_w_list))
        beta_m_median = float(np.median(beta_m_list))

        # Renormalize to sum=1.0 (in case of numerical drift)
        beta_sum = beta_d_median + beta_w_median + beta_m_median
        beta_d_norm = beta_d_median / beta_sum
        beta_w_norm = beta_w_median / beta_sum
        beta_m_norm = beta_m_median / beta_sum

        # Aggregate OOS metrics
        oos_r2_har = [r["out_of_sample_metrics"]["HAR_R2"] for r in successful]
        oos_r2_ewma = [r["out_of_sample_metrics"]["EWMA_R2"] for r in successful]
        r2_gains = [r["improvement"]["R2_gain_pp"] for r in successful]

        aggregated[horizon_key] = {
            "status": "success",
            "n_contracts": len(results),
            "n_successful": len(successful),
            "optimal_betas_median": {
                "daily": round(beta_d_norm, 4),
                "weekly": round(beta_w_norm, 4),
                "monthly": round(beta_m_norm, 4)
            },
            "oos_performance": {
                "median_HAR_R2": round(float(np.median(oos_r2_har)), 4),
                "median_EWMA_R2": round(float(np.median(oos_r2_ewma)), 4),
                "median_R2_gain_pp": round(float(np.median(r2_gains)), 2)
            },
            "by_contract": successful
        }

        log.info(f"  ✓ {horizon_key}: β_d={beta_d_norm:.3f}, β_w={beta_w_norm:.3f}, β_m={beta_m_norm:.3f}")
        log.info(f"    OOS R² gain: {aggregated[horizon_key]['oos_performance']['median_R2_gain_pp']:.2f}pp")

    # ========================================================================
    # STEP 4: Package final output
    # ========================================================================

    log.info("\n[STEP 4/4] Packaging results...")

    output = {
        "meta": {
            "timestamp": timestamp,
            "optimization_config": OPTIMIZATION_CONFIG,
            "data_source": "MOEX ISS API",
            "contracts": [str(c) for c in contracts],
            "horizons": horizons,
            "duration_days": duration_days,
            "bar_size": bar_size
        },
        "optimization_results": aggregated,
        "summary": {
            "total_optimizations": sum(len(r) for r in optimization_results.values()),
            "successful_optimizations": sum(
                len([x for x in r if x.get("status") == "success"])
                for r in optimization_results.values()
            ),
            "horizons_with_results": [
                h for h, v in aggregated.items() if v.get("status") == "success"
            ]
        }
    }

    # ========================================================================
    # Save to JSON
    # ========================================================================

    if save_results:
        json_path = Path(output_dir) / f"har_optimization_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        log.info(f"\n✓ Results saved to: {json_path}")

    # ========================================================================
    # Print summary
    # ========================================================================

    log.info("\n" + "="*80)
    log.info("OPTIMIZATION COMPLETE")
    log.info("="*80)
    log.info(f"Total optimizations: {output['summary']['total_optimizations']}")
    log.info(f"Successful: {output['summary']['successful_optimizations']}")
    log.info("\nOptimal betas by horizon:")
    log.info("-" * 80)

    for horizon_key, agg in aggregated.items():
        if agg.get("status") == "success":
            betas = agg["optimal_betas_median"]
            perf = agg["oos_performance"]
            log.info(f"{horizon_key:12} | β_d={betas['daily']:.3f}, β_w={betas['weekly']:.3f}, "
                    f"β_m={betas['monthly']:.3f} | R² gain: {perf['median_R2_gain_pp']:+.2f}pp")

    log.info("="*80)

    return output


# ============================================================================
# SELF-TEST
# ============================================================================

def test_cell_07_6_3() -> bool:
    """
    Self-test для Cell 07.6.3.

    Проверяет:
    1. run_har_parameter_optimization загружена
    2. Все зависимости доступны
    3. Output directory может быть создана

    Returns:
        True if all tests pass
    """
    log.info("\n" + "="*80)
    log.info("CELL 07.6.3 SELF-TEST: Portfolio Runner")
    log.info("="*80)

    tests_passed = []
    tests_failed = []

    # Test 1: Function loaded
    try:
        assert "run_har_parameter_optimization" in globals()
        log.info("  ✓ run_har_parameter_optimization loaded")
        tests_passed.append("function_loaded")
    except AssertionError:
        log.error("  ✗ run_har_parameter_optimization not found")
        tests_failed.append("function_loaded")
        return False

    # Test 2: Dependencies available
    required_deps = [
        "OPTIMIZATION_CONFIG",
        "optimize_single_horizon_rolling",
        "BacktestDataPipeline"
    ]

    missing_deps = [fn for fn in required_deps if fn not in globals()]

    if not missing_deps:
        log.info(f"  ✓ All dependencies available")
        tests_passed.append("dependencies")
    else:
        log.error(f"  ✗ Missing dependencies: {', '.join(missing_deps)}")
        log.error("    Hint: Run Cell 07.6.1, 07.6.2, and 07.2.09a first")
        tests_failed.append("dependencies")

    # Test 3: Output directory
    try:
        OPTIMIZATION_CONFIG = globals()["OPTIMIZATION_CONFIG"]
        output_dir = Path(OPTIMIZATION_CONFIG["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir.exists()
        log.info(f"  ✓ Output directory ready: {output_dir}")
        tests_passed.append("output_dir")
    except Exception as e:
        log.error(f"  ✗ Output directory error: {e}")
        tests_failed.append("output_dir")

    # Summary
    log.info("\n" + "-"*80)
    log.info(f"Tests passed: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)}")

    if tests_failed:
        log.error(f"\n✗ FAILED tests: {', '.join(tests_failed)}")
        return False

    log.info("\n✓ CELL 07.6.3 READY - Portfolio Runner validated")
    log.info("  Note: Full test requires MOEX ISS API connection (run optimization manually)")
    return True


# ============================================================================
# AUTO-LOAD MESSAGE
# ============================================================================

log.info("✓ Cell 07.6.3: Portfolio Runner loaded into globals()")
log.info("  Data source: MOEX ISS API (https://iss.moex.com/iss)")
log.info("  Run test_cell_07_6_3() to validate runner")
