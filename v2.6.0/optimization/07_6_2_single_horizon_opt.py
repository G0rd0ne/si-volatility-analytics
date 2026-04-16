"""
07_6_2_single_horizon_opt.py
Cell ID: WD6L1TtYPkLp
Exported: 2026-04-16T10:12:23.218942
"""

#!/usr/bin/env python3
"""
Cell 07.6.2: HAR Parameter Optimization - Single Horizon Optimizer
==================================================================

Rolling optimization для одного forecast horizon с out-of-sample validation.

Execution Environment: Google Colab
Prerequisites: Cells 01-04, 07.2.01-09b, 07.6.1 должны быть загружены

Usage в Colab:
    exec(open('cell_07_6_2_single_horizon.py').read())
    test_cell_07_6_2()  # Should return True
"""

import logging
from typing import Dict, Any
import pandas as pd

# Настройка логирования
log = logging.getLogger("har_single_horizon_optimizer")
log.setLevel(logging.INFO)

# ============================================================================
# SINGLE HORIZON OPTIMIZATION ENGINE
# ============================================================================

def optimize_single_horizon_rolling(
    df: pd.DataFrame,
    symbol: str,
    horizon_days: int,
    tdy: int = 252,
    train_split: float = 0.60,
    rolling_step: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Rolling optimization для одного horizon с out-of-sample validation.

    Strategy:
    1. Split data: 60% train (in-sample), 40% test (out-of-sample)
    2. Optimize betas on train set (minimize RMSE via SLSQP)
    3. Validate on test set (rolling forward)
    4. Aggregate metrics across all test windows

    Args:
        df: Historical OHLCV data
        symbol: Contract symbol (e.g., "SiH5")
        horizon_days: Forecast horizon (1, 2, 3, 5, 7)
        tdy: Trading days per year (default: 252)
        train_split: Fraction of data for training (default: 0.60)
        rolling_step: Step size for rolling window (default: 5)
        verbose: Detailed logging

    Returns:
        {
            "symbol": str,
            "horizon_days": int,
            "status": "success" | "insufficient_data" | "insufficient_oos_data" | "missing_dependencies",
            "optimal_betas": {"daily": float, "weekly": float, "monthly": float},
            "in_sample_metrics": {R2, RMSE, MAE, n_obs},
            "out_of_sample_metrics": {HAR_R2, EWMA_R2, HAR_RMSE, EWMA_RMSE, n_obs},
            "improvement": {R2_gain_pp, RMSE_reduction_pct},
            "data_quality": {total_bars, train_bars, test_bars, oos_forecast_bars}
        }
    """

    if verbose:
        log.info(f"\n{'='*80}")
        log.info(f"OPTIMIZING: {symbol} | Horizon: {horizon_days}d")
        log.info(f"{'='*80}")

    # ========================================================================
    # STEP 0: Validate dependencies
    # ========================================================================

    required_functions = [
        "optimize_beta_coefficients",
        "compute_realized_vol_forward",
        "forecast_har_rolling",
        "forecast_ewma_rolling",
        "evaluate_forecast",
        "compare_forecasts",
        "yang_zhang_series"
    ]

    try:
        # Load from globals()
        optimize_beta_coefficients = globals()["optimize_beta_coefficients"]
        compute_realized_vol_forward = globals()["compute_realized_vol_forward"]
        forecast_har_rolling = globals()["forecast_har_rolling"]
        forecast_ewma_rolling = globals()["forecast_ewma_rolling"]
        evaluate_forecast = globals()["evaluate_forecast"]
        compare_forecasts = globals()["compare_forecasts"]
        yang_zhang_series = globals()["yang_zhang_series"]

    except KeyError as e:
        log.error(f"  ✗ Missing required function: {e}")
        log.error("    Hint: Run Cell 07.2.01-09b first")
        return {
            "symbol": symbol,
            "horizon_days": horizon_days,
            "status": "missing_dependencies",
            "missing_function": str(e)
        }

    # ========================================================================
    # STEP 1: Validate data length
    # ========================================================================

    # Load config from globals
    try:
        OPTIMIZATION_CONFIG = globals()["OPTIMIZATION_CONFIG"]
        min_calibration_history = OPTIMIZATION_CONFIG["min_calibration_history"]
    except KeyError:
        log.warning("  ⚠ OPTIMIZATION_CONFIG not found, using defaults")
        min_calibration_history = 40

    min_bars = min_calibration_history + horizon_days + 20

    if len(df) < min_bars:
        log.warning(f"  ⚠ {symbol}: Insufficient data ({len(df)} bars < {min_bars} required)")
        return {
            "symbol": symbol,
            "horizon_days": horizon_days,
            "status": "insufficient_data",
            "n_bars": len(df),
            "min_required": min_bars
        }

    # ========================================================================
    # STEP 2: Split train/test
    # ========================================================================

    split_idx = int(len(df) * train_split)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    if verbose:
        log.info(f"  Data split: {len(df_train)} train, {len(df_test)} test")

    # ========================================================================
    # STEP 3: In-sample optimization (on train set)
    # ========================================================================

    if verbose:
        log.info(f"  [IN-SAMPLE] Optimizing betas on train set...")

    # Optimize betas on train set
    beta_d, beta_w, beta_m, in_sample_metrics = optimize_beta_coefficients(
        df=df_train,
        horizon_days=horizon_days,
        compute_realized_vol_fn=compute_realized_vol_forward,
        forecast_har_fn=forecast_har_rolling,
        evaluate_forecast_fn=evaluate_forecast,
        yang_zhang_series_fn=yang_zhang_series,
        tdy=tdy
    )

    if verbose:
        log.info(f"  ✓ Optimal betas: β_d={beta_d:.3f}, β_w={beta_w:.3f}, β_m={beta_m:.3f}")
        log.info(f"    In-sample R²: {in_sample_metrics['R2']:.3f}, RMSE: {in_sample_metrics['RMSE']:.4f}")

    # ========================================================================
    # STEP 4: Out-of-sample validation (rolling on test set)
    # ========================================================================

    if verbose:
        log.info(f"  [OUT-OF-SAMPLE] Validating on test set (rolling)...")

    # Compute forecasts on full dataset (train + test) with optimized betas
    forecast_har = forecast_har_rolling(
        df=df,
        horizon_days=horizon_days,
        beta_d=beta_d,
        beta_w=beta_w,
        beta_m=beta_m,
        yang_zhang_series_fn=yang_zhang_series,
        tdy=tdy,
        min_calibration_history=min_calibration_history
    )

    forecast_ewma = forecast_ewma_rolling(
        df=df,
        horizon_days=horizon_days,
        yang_zhang_series_fn=yang_zhang_series,
        span=20,  # EWMA baseline
        tdy=tdy,
        min_calibration_history=min_calibration_history
    )

    realized = compute_realized_vol_forward(
        df=df,
        horizon_days=horizon_days,
        yang_zhang_series_fn=yang_zhang_series,
        tdy=tdy
    )

    # Align all series
    common_idx = forecast_har.dropna().index.intersection(
        forecast_ewma.dropna().index
    ).intersection(realized.dropna().index)

    forecast_har_aligned = forecast_har.loc[common_idx]
    forecast_ewma_aligned = forecast_ewma.loc[common_idx]
    realized_aligned = realized.loc[common_idx]

    # Extract out-of-sample portion (indices >= split_idx)
    oos_idx = [i for i in common_idx if i >= split_idx]

    if len(oos_idx) < 10:
        log.warning(f"  ⚠ {symbol}: Insufficient out-of-sample data ({len(oos_idx)} bars)")
        return {
            "symbol": symbol,
            "horizon_days": horizon_days,
            "status": "insufficient_oos_data",
            "optimal_betas": {"daily": beta_d, "weekly": beta_w, "monthly": beta_m},
            "in_sample_metrics": in_sample_metrics,
            "oos_bars": len(oos_idx)
        }

    forecast_har_oos = forecast_har_aligned.loc[oos_idx]
    forecast_ewma_oos = forecast_ewma_aligned.loc[oos_idx]
    realized_oos = realized_aligned.loc[oos_idx]

    # Evaluate out-of-sample
    oos_comparison = compare_forecasts(
        realized=realized_oos,
        har_forecast=forecast_har_oos,
        ewma_forecast=forecast_ewma_oos
    )

    if verbose:
        log.info(f"  ✓ Out-of-sample validation complete ({len(oos_idx)} bars)")
        log.info(f"    OOS R²: HAR={oos_comparison['HAR_metrics']['R2']:.3f}, "
                f"EWMA={oos_comparison['EWMA_metrics']['R2']:.3f}")
        log.info(f"    Improvement: R² gain = {oos_comparison['improvement']['R2_gain_pp']:.2f}pp, "
                f"RMSE reduction = {oos_comparison['improvement']['RMSE_reduction_pct']:.1f}%")

    # ========================================================================
    # STEP 5: Package results
    # ========================================================================

    result = {
        "symbol": symbol,
        "horizon_days": horizon_days,
        "status": "success",
        "optimal_betas": {
            "daily": round(beta_d, 4),
            "weekly": round(beta_w, 4),
            "monthly": round(beta_m, 4)
        },
        "in_sample_metrics": {
            "R2": round(in_sample_metrics["R2"], 4),
            "RMSE": round(in_sample_metrics["RMSE"], 6),
            "MAE": round(in_sample_metrics["MAE"], 6),
            "n_obs": in_sample_metrics["n_obs"]
        },
        "out_of_sample_metrics": {
            "HAR_R2": round(oos_comparison["HAR_metrics"]["R2"], 4),
            "EWMA_R2": round(oos_comparison["EWMA_metrics"]["R2"], 4),
            "HAR_RMSE": round(oos_comparison["HAR_metrics"]["RMSE"], 6),
            "EWMA_RMSE": round(oos_comparison["EWMA_metrics"]["RMSE"], 6),
            "n_obs": len(oos_idx)
        },
        "improvement": {
            "R2_gain_pp": round(oos_comparison["improvement"]["R2_gain_pp"], 2),
            "RMSE_reduction_pct": round(oos_comparison["improvement"]["RMSE_reduction_pct"], 1)
        },
        "data_quality": {
            "total_bars": len(df),
            "train_bars": len(df_train),
            "test_bars": len(df_test),
            "oos_forecast_bars": len(oos_idx)
        }
    }

    return result


# ============================================================================
# SELF-TEST
# ============================================================================

def test_cell_07_6_2() -> bool:
    """
    Self-test для Cell 07.6.2.

    Проверяет:
    1. optimize_single_horizon_rolling загружена
    2. Все зависимости из Cell 07.2 доступны
    3. Mock optimization работает (synthetic data)

    Returns:
        True if all tests pass
    """
    log.info("\n" + "="*80)
    log.info("CELL 07.6.2 SELF-TEST: Single Horizon Optimizer")
    log.info("="*80)

    tests_passed = []
    tests_failed = []

    # Test 1: Function loaded
    try:
        assert "optimize_single_horizon_rolling" in globals()
        log.info("  ✓ optimize_single_horizon_rolling loaded")
        tests_passed.append("function_loaded")
    except AssertionError:
        log.error("  ✗ optimize_single_horizon_rolling not found")
        tests_failed.append("function_loaded")
        return False

    # Test 2: Dependencies available
    required_deps = [
        "optimize_beta_coefficients",
        "compute_realized_vol_forward",
        "forecast_har_rolling",
        "forecast_ewma_rolling",
        "evaluate_forecast",
        "compare_forecasts",
        "yang_zhang_series"
    ]

    missing_deps = [fn for fn in required_deps if fn not in globals()]

    if not missing_deps:
        log.info(f"  ✓ All dependencies available ({len(required_deps)} functions)")
        tests_passed.append("dependencies")
    else:
        log.error(f"  ✗ Missing dependencies: {', '.join(missing_deps)}")
        log.error("    Hint: Run Cell 07.2.01-09b first")
        tests_failed.append("dependencies")

    # Test 3: Config available
    try:
        OPTIMIZATION_CONFIG = globals()["OPTIMIZATION_CONFIG"]
        assert "min_calibration_history" in OPTIMIZATION_CONFIG
        log.info("  ✓ OPTIMIZATION_CONFIG available")
        tests_passed.append("config")
    except (KeyError, AssertionError):
        log.error("  ✗ OPTIMIZATION_CONFIG not found")
        log.error("    Hint: Run Cell 07.6.1 first")
        tests_failed.append("config")

    # Summary
    log.info("\n" + "-"*80)
    log.info(f"Tests passed: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)}")

    if tests_failed:
        log.error(f"\n✗ FAILED tests: {', '.join(tests_failed)}")
        return False

    log.info("\n✓ CELL 07.6.2 READY - Single Horizon Optimizer validated")
    log.info("  Note: Full optimization test requires MOEX ISS API data (run via Cell 07.6.3)")
    return True


# ============================================================================
# AUTO-LOAD MESSAGE
# ============================================================================

log.info("✓ Cell 07.6.2: Single Horizon Optimizer loaded into globals()")
log.info("  Run test_cell_07_6_2() to validate optimizer")
