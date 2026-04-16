"""
07_6_4_opt_validation.py
Cell ID: j5fyjgzjPuvf
Exported: 2026-04-16T10:12:23.218989
"""

#!/usr/bin/env python3
"""
Cell 07.6.4: HAR Parameter Optimization - Validation & Self-Tests
==================================================================

Полная валидация HAR optimization system.

Execution Environment: Google Colab
Prerequisites: Cells 01-04, 07.2.01-09b, 07.6.1-3 должны быть загружены

Usage в Colab:
    exec(open('cell_07_6_4_validation.py').read())
    run_optimization_self_tests()  # Should return True
"""

import logging

# Настройка логирования
log = logging.getLogger("har_optimization_validation")
log.setLevel(logging.INFO)

# ============================================================================
# COMPREHENSIVE SELF-TESTS
# ============================================================================

def run_optimization_self_tests() -> bool:
    """
    Полная валидация HAR Parameter Optimization System.

    Проверяет:
    1. Cell 07.6.1: Configuration
    2. Cell 07.6.2: Single Horizon Optimizer
    3. Cell 07.6.3: Portfolio Runner
    4. Cell 07.2.01-09b: HAR Backtest dependencies
    5. Cell 07.3: Visualization

    Returns:
        True if all tests pass
    """

    log.info("\n" + "="*80)
    log.info("HAR PARAMETER OPTIMIZATION SELF-TESTS")
    log.info("="*80)

    all_tests_passed = True

    # ========================================================================
    # TEST 1: Cell 07.6.1 (Configuration)
    # ========================================================================

    log.info("\n[TEST 1/5] Cell 07.6.1: Configuration")

    try:
        test_cell_07_6_1 = globals()["test_cell_07_6_1"]
        if test_cell_07_6_1():
            log.info("  ✓ Cell 07.6.1 validated")
        else:
            log.error("  ✗ Cell 07.6.1 validation failed")
            all_tests_passed = False
    except KeyError:
        log.error("  ✗ test_cell_07_6_1() not found")
        log.error("    Hint: Run Cell 07.6.1 first")
        all_tests_passed = False

    # ========================================================================
    # TEST 2: Cell 07.6.2 (Single Horizon Optimizer)
    # ========================================================================

    log.info("\n[TEST 2/5] Cell 07.6.2: Single Horizon Optimizer")

    try:
        test_cell_07_6_2 = globals()["test_cell_07_6_2"]
        if test_cell_07_6_2():
            log.info("  ✓ Cell 07.6.2 validated")
        else:
            log.error("  ✗ Cell 07.6.2 validation failed")
            all_tests_passed = False
    except KeyError:
        log.error("  ✗ test_cell_07_6_2() not found")
        log.error("    Hint: Run Cell 07.6.2 first")
        all_tests_passed = False

    # ========================================================================
    # TEST 3: Cell 07.6.3 (Portfolio Runner)
    # ========================================================================

    log.info("\n[TEST 3/5] Cell 07.6.3: Portfolio Runner")

    try:
        test_cell_07_6_3 = globals()["test_cell_07_6_3"]
        if test_cell_07_6_3():
            log.info("  ✓ Cell 07.6.3 validated")
        else:
            log.error("  ✗ Cell 07.6.3 validation failed")
            all_tests_passed = False
    except KeyError:
        log.error("  ✗ test_cell_07_6_3() not found")
        log.error("    Hint: Run Cell 07.6.3 first")
        all_tests_passed = False

    # ========================================================================
    # TEST 4: Cell 07.2 Dependencies
    # ========================================================================

    log.info("\n[TEST 4/5] Cell 07.2: HAR Backtest Dependencies")

    required_functions = [
        "BacktestDataPipeline",
        "HARBacktestEngine",
        "optimize_beta_coefficients",
        "compute_realized_vol_forward",
        "forecast_har_rolling",
        "forecast_ewma_rolling",
        "evaluate_forecast",
        "compare_forecasts",
        "yang_zhang_series"
    ]

    missing = [fn for fn in required_functions if fn not in globals()]

    if not missing:
        log.info(f"  ✓ All Cell 07.2 dependencies available ({len(required_functions)} functions)")
    else:
        log.error(f"  ✗ Missing Cell 07.2 dependencies: {', '.join(missing)}")
        log.error("    Hint: Run Cell 07.2.01-09b first")
        all_tests_passed = False

    # ========================================================================
    # TEST 5: Cell 07.3 Visualization
    # ========================================================================

    log.info("\n[TEST 5/5] Cell 07.3: Visualization")

    viz_functions = ["plot_backtest_results", "save_backtest_json"]
    missing_viz = [fn for fn in viz_functions if fn not in globals()]

    if not missing_viz:
        log.info(f"  ✓ Visualization functions available")
    else:
        log.error(f"  ✗ Missing visualization functions: {', '.join(missing_viz)}")
        log.error("    Hint: Run Cell 07.3 first")
        all_tests_passed = False

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    log.info("\n" + "="*80)

    if all_tests_passed:
        log.info("✓ ALL TESTS PASSED - HAR Optimization System Ready")
        log.info("\nNext steps:")
        log.info("  1. Initialize MOEX client: from cell_02_moex_contracts import MoexClient, MoexConfig")
        log.info("  2. Load contracts: client = MoexClient(MoexConfig()); contracts = client.get_futures_chain('Si')")
        log.info("  3. Run optimization: results = run_har_parameter_optimization(client, contracts)")
        log.info("="*80)
        return True
    else:
        log.error("✗ SOME TESTS FAILED - Fix issues before running optimization")
        log.info("="*80)
        return False


# ============================================================================
# SELF-TEST FOR THIS MODULE
# ============================================================================

def test_cell_07_6_4() -> bool:
    """
    Self-test для Cell 07.6.4.

    Проверяет:
    1. run_optimization_self_tests загружена

    Returns:
        True if all tests pass
    """
    log.info("\n" + "="*80)
    log.info("CELL 07.6.4 SELF-TEST: Validation Module")
    log.info("="*80)

    try:
        assert "run_optimization_self_tests" in globals()
        log.info("  ✓ run_optimization_self_tests loaded")
        log.info("\n✓ CELL 07.6.4 READY - Validation module loaded")
        return True
    except AssertionError:
        log.error("  ✗ run_optimization_self_tests not found")
        return False


# ============================================================================
# AUTO-LOAD MESSAGE
# ============================================================================

log.info("✓ Cell 07.6.4: Validation & Self-Tests loaded into globals()")
log.info("  Run run_optimization_self_tests() to validate entire optimization system")
