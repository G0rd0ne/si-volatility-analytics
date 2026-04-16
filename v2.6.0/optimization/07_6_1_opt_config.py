"""
07_6_1_opt_config.py
Cell ID: Eqw_Dv4zPhvW
Exported: 2026-04-16T10:12:23.218927
"""

#!/usr/bin/env python3
"""
Cell 07.6.1: HAR Parameter Optimization - Configuration
========================================================

Конфигурация для HAR parameter optimization system.

Execution Environment: Google Colab
Prerequisites: Cells 01-04 должны быть загружены

Usage в Colab:
    exec(open('cell_07_6_1_config.py').read())
    test_cell_07_6_1()  # Should return True
"""

import logging
from pathlib import Path

# Настройка логирования
log = logging.getLogger("har_optimization_config")
log.setLevel(logging.INFO)

# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================

OPTIMIZATION_CONFIG = {
    # Trading calendar
    "trading_days_per_year": 252,

    # Rolling optimization parameters
    "min_calibration_history": 40,  # Min bars for beta fitting
    "train_test_split": 0.60,  # 60% train, 40% test (out-of-sample)
    "rolling_step": 5,  # Step size for rolling window

    # Output settings
    "output_dir": "/content/har_optimization_results",

    # Horizon settings
    "default_horizons": [1, 2, 3, 5, 7],  # Trading days

    # Beta optimization constraints
    "beta_bounds": (0.0, 1.0),  # Each beta must be in [0, 1]
    "beta_sum": 1.0,  # β_d + β_w + β_m = 1.0 (sum-to-one constraint)

    # I/O settings
    "save_json": True,
    "save_plots": True,
    "verbose": True,

    # MOEX ISS API data settings
    "default_duration_days": 365,  # 1 year historical data
    "default_bar_size": "1 hour",  # Recommended for vol analysis (MOEX candles)

    # Validation thresholds
    "min_success_rate": 0.70,  # At least 70% contracts должны успешно оптимизироваться
    "min_r2_improvement": 0.02,  # HAR должен улучшать R² минимум на 2pp
    "min_oos_observations": 10,  # Min bars in out-of-sample set
}


# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

VALIDATION_THRESHOLDS = {
    # R² thresholds (for vol forecasting)
    "excellent_r2": 0.50,  # R² > 0.50 — отличный результат
    "good_r2": 0.30,  # R² > 0.30 — хороший результат
    "poor_r2": 0.30,  # R² < 0.30 — плохой результат

    # Improvement thresholds (HAR vs EWMA)
    "significant_improvement": 5.0,  # R² gain > 5pp — значительное улучшение
    "moderate_improvement": 2.0,  # R² gain > 2pp — умеренное улучшение

    # Data quality thresholds
    "min_bars_per_contract": 60,  # Min bars required
    "max_missing_data_pct": 5.0,  # Max 5% missing bars allowed
}


# ============================================================================
# EXPECTED BETA PATTERNS (for validation)
# ============================================================================

EXPECTED_BETA_PATTERNS = {
    # Ожидаемые диапазоны β для каждого horizon
    # (используется для validation — если β выходят за эти пределы, это warning)

    "horizon_1d": {
        "daily": (0.50, 0.75),  # High daily weight expected
        "weekly": (0.15, 0.35),
        "monthly": (0.05, 0.20)
    },
    "horizon_2d": {
        "daily": (0.45, 0.65),
        "weekly": (0.20, 0.40),
        "monthly": (0.10, 0.25)
    },
    "horizon_3d": {
        "daily": (0.35, 0.55),
        "weekly": (0.25, 0.45),
        "monthly": (0.15, 0.30)
    },
    "horizon_5d": {
        "daily": (0.25, 0.45),
        "weekly": (0.30, 0.50),
        "monthly": (0.20, 0.35)
    },
    "horizon_7d": {
        "daily": (0.20, 0.40),
        "weekly": (0.30, 0.50),
        "monthly": (0.25, 0.45)
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_output_dir(output_dir=None):
    """
    Создает output directory если не существует.

    Args:
        output_dir: Custom output dir (default: OPTIMIZATION_CONFIG["output_dir"])

    Returns:
        Path object
    """
    if output_dir is None:
        output_dir = OPTIMIZATION_CONFIG["output_dir"]

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    return path


def validate_config():
    """
    Валидирует OPTIMIZATION_CONFIG на корректность.

    Returns:
        (is_valid, errors)
    """
    errors = []

    # Check critical parameters
    if OPTIMIZATION_CONFIG["min_calibration_history"] < 20:
        errors.append("min_calibration_history < 20 (too low for stable optimization)")

    if not (0.0 < OPTIMIZATION_CONFIG["train_test_split"] < 1.0):
        errors.append("train_test_split must be in (0, 1)")

    if OPTIMIZATION_CONFIG["beta_sum"] != 1.0:
        errors.append("beta_sum must be exactly 1.0 (sum-to-one constraint)")

    if OPTIMIZATION_CONFIG["trading_days_per_year"] != 252:
        errors.append("trading_days_per_year should be 252 for standard markets")

    # Check horizons
    horizons = OPTIMIZATION_CONFIG["default_horizons"]
    if not all(h > 0 for h in horizons):
        errors.append("All horizons must be positive")

    if len(set(horizons)) != len(horizons):
        errors.append("Duplicate horizons detected")

    return (len(errors) == 0, errors)


# ============================================================================
# SELF-TEST
# ============================================================================

def test_cell_07_6_1() -> bool:
    """
    Self-test для Cell 07.6.1.

    Проверяет:
    1. OPTIMIZATION_CONFIG загружен
    2. Все required keys присутствуют
    3. Config валиден
    4. Output directory может быть создана

    Returns:
        True if all tests pass
    """
    log.info("\n" + "="*80)
    log.info("CELL 07.6.1 SELF-TEST: Configuration")
    log.info("="*80)

    tests_passed = []
    tests_failed = []

    # Test 1: Config loaded
    try:
        assert "OPTIMIZATION_CONFIG" in globals()
        log.info("  ✓ OPTIMIZATION_CONFIG loaded")
        tests_passed.append("config_loaded")
    except AssertionError:
        log.error("  ✗ OPTIMIZATION_CONFIG not found in globals")
        tests_failed.append("config_loaded")

    # Test 2: Required keys present
    required_keys = [
        "trading_days_per_year",
        "min_calibration_history",
        "train_test_split",
        "default_horizons",
        "beta_bounds",
        "beta_sum",
        "output_dir"
    ]

    missing_keys = [k for k in required_keys if k not in OPTIMIZATION_CONFIG]

    if not missing_keys:
        log.info(f"  ✓ All required keys present ({len(required_keys)} keys)")
        tests_passed.append("required_keys")
    else:
        log.error(f"  ✗ Missing keys: {missing_keys}")
        tests_failed.append("required_keys")

    # Test 3: Config validation
    is_valid, errors = validate_config()

    if is_valid:
        log.info("  ✓ Config validation passed")
        tests_passed.append("config_valid")
    else:
        log.error(f"  ✗ Config validation failed:")
        for err in errors:
            log.error(f"    - {err}")
        tests_failed.append("config_valid")

    # Test 4: Output directory creation
    try:
        output_path = ensure_output_dir()
        assert output_path.exists()
        log.info(f"  ✓ Output directory created: {output_path}")
        tests_passed.append("output_dir")
    except Exception as e:
        log.error(f"  ✗ Failed to create output directory: {e}")
        tests_failed.append("output_dir")

    # Test 5: EXPECTED_BETA_PATTERNS loaded
    try:
        assert "EXPECTED_BETA_PATTERNS" in globals()
        assert len(EXPECTED_BETA_PATTERNS) == len(OPTIMIZATION_CONFIG["default_horizons"])
        log.info(f"  ✓ EXPECTED_BETA_PATTERNS loaded ({len(EXPECTED_BETA_PATTERNS)} horizons)")
        tests_passed.append("beta_patterns")
    except AssertionError:
        log.error("  ✗ EXPECTED_BETA_PATTERNS invalid")
        tests_failed.append("beta_patterns")

    # Summary
    log.info("\n" + "-"*80)
    log.info(f"Tests passed: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)}")

    if tests_failed:
        log.error(f"\n✗ FAILED tests: {', '.join(tests_failed)}")
        return False

    log.info("\n✓ CELL 07.6.1 READY - Configuration validated")
    return True


# ============================================================================
# AUTO-LOAD MESSAGE
# ============================================================================

log.info("✓ Cell 07.6.1: HAR Optimization Configuration loaded into globals()")
log.info("  Data source: MOEX ISS API (https://iss.moex.com/iss)")
log.info("  Run test_cell_07_6_1() to validate configuration")
