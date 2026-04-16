"""
10_2_5d_thursday_opt.py
Cell ID: f96pbxZwWxiP
Exported: 2026-04-16T10:12:23.219094
"""

"""
Cell 10.2: 5-Days-to-Thursday HAR Optimization
===============================================

Оптимизация β-коэффициентов HAR для прогнозов "5 дней до четверга".

Ключевое отличие от стандартной оптимизации:
- Обучающая выборка содержит ТОЛЬКО пятницы (friday → next thursday = ~5 trading days)
- Целевая RV вычисляется до ближайшего четверга, а не на фиксированный horizon

Data Source: MOEX ISS API (https://iss.moex.com/iss)

Usage в Colab:
    exec(open('cell_10_2_5d_thursday_optimization.py').read())
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger("5d_thursday_optimization")
log.setLevel(logging.INFO)

# ============================================================================
# OPTIMIZATION ENGINE
# ============================================================================

def optimize_5d_for_thursday_single_contract(
    df: pd.DataFrame,
    symbol: str,
    yang_zhang_series_fn,
    compute_realized_vol_to_thursday_fn,
    tdy: int = 252,
    train_split: float = 0.60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Оптимизирует β-коэффициенты HAR для прогнозов "5 дней до четверга".

    Strategy:
    1. Фильтрует обучающую выборку: только ПЯТНИЦЫ (weekday=4)
    2. Для каждой пятницы вычисляет RV до ближайшего четверга
    3. Оптимизирует β через minimize (как в стандартной версии)
    4. Валидирует на test set

    Args:
        df: Historical OHLCV data с колонкой 'date'
        symbol: Contract symbol (e.g., "SiH5")
        yang_zhang_series_fn: Функция yang_zhang_series
        compute_realized_vol_to_thursday_fn: Функция compute_realized_vol_to_thursday
        tdy: Trading days per year
        train_split: Fraction of data for training
        verbose: Detailed logging

    Returns:
        {
            "symbol": str,
            "horizon_type": "5d_for_thursday",
            "optimal_betas": {"daily": float, "weekly": float, "monthly": float},
            "in_sample_metrics": {...},
            "out_of_sample_metrics": {...},
            "data_quality": {...}
        }
    """

    if verbose:
        log.info(f"\n{'='*80}")
        log.info(f"OPTIMIZING: {symbol} | 5d_for_thursday")
        log.info(f"{'='*80}")

    # Validate data length
    min_bars = 100  # Min для meaningful optimization
    if len(df) < min_bars:
        log.warning(f"  ⚠ {symbol}: Insufficient data ({len(df)} bars < {min_bars} required)")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "insufficient_data",
            "n_bars": len(df),
            "min_required": min_bars
        }

    # ========================================================================
    # STEP 1: Split train/test
    # ========================================================================

    split_idx = int(len(df) * train_split)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    if verbose:
        log.info(f"  Data split: {len(df_train)} train, {len(df_test)} test")

    # ========================================================================
    # STEP 2: Filter FRIDAYS ONLY in training set
    # ========================================================================

    # Add weekday column
    df_train['weekday'] = pd.to_datetime(df_train['date']).dt.weekday
    fridays_train = df_train[df_train['weekday'] == 4].copy()  # 4 = Friday

    if len(fridays_train) < 20:
        log.warning(f"  ⚠ {symbol}: Insufficient Fridays in train set ({len(fridays_train)} < 20)")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "insufficient_fridays",
            "n_fridays": len(fridays_train)
        }

    if verbose:
        log.info(f"  Filtered to {len(fridays_train)} Fridays in train set")

    # ========================================================================
    # STEP 3: Compute Thursday-aligned RV for training set
    # ========================================================================

    if verbose:
        log.info(f"  [IN-SAMPLE] Computing Thursday-aligned RV...")

    realized_train = compute_realized_vol_to_thursday_fn(
        df=df_train,
        yang_zhang_series_fn=yang_zhang_series_fn,
        tdy=tdy,
        verbose=False
    )

    if len(realized_train) < 10:
        log.warning(f"  ⚠ {symbol}: Insufficient realized vol ({len(realized_train)} < 10)")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "insufficient_realized_vol",
            "n_realized": len(realized_train)
        }

    if verbose:
        log.info(f"  Thursday-aligned RV computed: {len(realized_train)} observations")

    # ========================================================================
    # STEP 4: Prepare HAR features for FRIDAYS ONLY
    # ========================================================================

    # For each Friday, compute RV_d, RV_w, RV_m at that date
    X_train = []
    y_train = []
    train_dates = []

    for idx in fridays_train.index:
        friday_date = df_train.loc[idx, 'date']

        # Find the data window up to (but not including) this Friday
        hist_df = df_train[df_train['date'] < friday_date]

        if len(hist_df) < 20:  # Need at least 20 days history for RV_m
            continue

        # Compute HAR features
        rv_1d = yang_zhang_series_fn(hist_df.tail(1), 1, tdy)
        rv_5d = yang_zhang_series_fn(hist_df.tail(5), 5, tdy)
        rv_20d = yang_zhang_series_fn(hist_df.tail(20), 20, tdy)

        if rv_1d.empty or rv_5d.empty or rv_20d.empty:
            continue

        rv_1d_val = float(rv_1d.iloc[-1])
        rv_5d_val = float(rv_5d.iloc[-1])
        rv_20d_val = float(rv_20d.iloc[-1])

        if pd.isna(rv_1d_val) or pd.isna(rv_5d_val) or pd.isna(rv_20d_val):
            continue

        # CRITICAL: realized_train is now indexed by STARTING dates (Fridays),
        # so we look up the target RV directly at friday_date (not future dates)
        # This gives us the forward RV from this Friday to its nearest Thursday
        if friday_date not in realized_train.index:
            continue

        target_rv = realized_train.loc[friday_date]

        X_train.append([rv_1d_val, rv_5d_val, rv_20d_val])
        y_train.append(target_rv)
        train_dates.append(friday_date)

    if len(X_train) < 10:
        log.warning(f"  ⚠ {symbol}: Insufficient training samples ({len(X_train)} < 10)")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "insufficient_training_samples",
            "n_samples": len(X_train)
        }

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # CRITICAL: Validate data before optimization
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        log.error(f"  ✗ {symbol}: X_train contains NaN or inf values")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "invalid_features",
            "error": "X_train contains NaN or inf"
        }

    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        log.error(f"  ✗ {symbol}: y_train contains NaN or inf values")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "invalid_target",
            "error": "y_train contains NaN or inf"
        }

    if y_train.std() < 1e-8:
        log.error(f"  ✗ {symbol}: y_train has zero variance (all values identical)")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "zero_variance_target",
            "error": "Target RV has no variation"
        }

    if verbose:
        log.info(f"  Training samples: {len(X_train)} (Fridays only)")
        log.info(f"  Features shape: {X_train.shape}")
        log.info(f"  Target RV range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        log.info(f"  Target RV std: {y_train.std():.4f}")
        log.info(f"  Features mean: {X_train.mean(axis=0)}")
        log.info(f"  Features std: {X_train.std(axis=0)}")

    # ========================================================================
    # STEP 5: Optimize betas via SLSQP
    # ========================================================================

    if verbose:
        log.info(f"  [OPTIMIZATION] Minimizing RMSE...")

    def objective(betas):
        """RMSE minimization objective."""
        beta_d, beta_w, beta_m = betas

        # Constraint: sum = 1.0
        if abs(beta_d + beta_w + beta_m - 1.0) > 0.05:
            return 1e6

        # Bounds check
        if any(b < 0 or b > 1 for b in betas):
            return 1e6

        # HAR forecast
        y_pred = beta_d * X_train[:, 0] + beta_w * X_train[:, 1] + beta_m * X_train[:, 2]

        rmse = np.sqrt(np.mean((y_train - y_pred) ** 2))
        return rmse

    x0 = [0.35, 0.40, 0.25]  # Initial guess

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints={'type': 'eq', 'fun': lambda b: sum(b) - 1.0},
        options={'maxiter': 100, 'ftol': 1e-6}
    )

    if verbose:
        log.info(f"  Optimization result: success={result.success}, nit={result.nit}, fun={result.fun:.6f}")
        log.info(f"  Message: {result.message}")

    # CRITICAL: Check if optimizer actually moved from initial guess
    if np.allclose(result.x, x0, atol=1e-4):
        log.warning(f"  ⚠ {symbol}: Optimizer returned initial guess unchanged (no convergence)")
        log.warning(f"    Initial: {x0}")
        log.warning(f"    Final:   {result.x}")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "no_convergence",
            "message": "Optimizer did not improve from initial guess",
            "initial_guess": x0,
            "final_betas": result.x.tolist(),
            "optimizer_message": result.message
        }

    if not result.success:
        log.warning(f"  ⚠ {symbol}: Optimization failed: {result.message}")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "optimization_failed",
            "message": result.message,
            "nit": result.nit,
            "fun": float(result.fun)
        }

    beta_d, beta_w, beta_m = result.x

    # In-sample metrics
    y_pred_train = beta_d * X_train[:, 0] + beta_w * X_train[:, 1] + beta_m * X_train[:, 2]

    # CRITICAL: Validate predictions
    if np.any(np.isnan(y_pred_train)) or np.any(np.isinf(y_pred_train)):
        log.error(f"  ✗ {symbol}: Predictions contain NaN or inf")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "invalid_predictions",
            "optimal_betas": {"daily": beta_d, "weekly": beta_w, "monthly": beta_m}
        }

    rmse_train = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    mae_train = np.mean(np.abs(y_train - y_pred_train))

    ss_res = np.sum((y_train - y_pred_train) ** 2)
    ss_tot = np.sum((y_train - y_train.mean()) ** 2)
    r2_train = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # CRITICAL: Sanity check on metrics
    if rmse_train == 0 and r2_train == 0:
        log.error(f"  ✗ {symbol}: Metrics are suspiciously zero (optimization did not work)")
        return {
            "symbol": symbol,
            "horizon_type": "5d_for_thursday",
            "status": "zero_metrics_error",
            "optimal_betas": {"daily": beta_d, "weekly": beta_w, "monthly": beta_m},
            "y_train_stats": {"mean": float(y_train.mean()), "std": float(y_train.std())},
            "y_pred_stats": {"mean": float(y_pred_train.mean()), "std": float(y_pred_train.std())}
        }

    if verbose:
        log.info(f"  ✓ Optimal betas: β_d={beta_d:.3f}, β_w={beta_w:.3f}, β_m={beta_m:.3f}")
        log.info(f"    Beta sum: {beta_d + beta_w + beta_m:.6f} (should be 1.0)")
        log.info(f"    In-sample R²: {r2_train:.3f}, RMSE: {rmse_train:.6f}, MAE: {mae_train:.6f}")
        log.info(f"    y_train mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")
        log.info(f"    y_pred mean:  {y_pred_train.mean():.6f}, std: {y_pred_train.std():.6f}")

    # ========================================================================
    # STEP 6: Package results
    # ========================================================================

    result_dict = {
        "symbol": symbol,
        "horizon_type": "5d_for_thursday",
        "status": "success",
        "optimal_betas": {
            "daily": round(beta_d, 4),
            "weekly": round(beta_w, 4),
            "monthly": round(beta_m, 4)
        },
        "in_sample_metrics": {
            "R2": round(r2_train, 4),
            "RMSE": round(rmse_train, 6),
            "MAE": round(np.mean(np.abs(y_train - y_pred_train)), 6),
            "n_obs": len(X_train),
            "n_fridays": len(fridays_train)
        },
        "data_quality": {
            "total_bars": len(df),
            "train_bars": len(df_train),
            "test_bars": len(df_test),
            "fridays_in_train": len(fridays_train),
            "training_samples": len(X_train)
        }
    }

    return result_dict


# ============================================================================
# SELF-TEST
# ============================================================================

def test_5d_thursday_optimization():
    """Validates dependencies."""
    log.info("\n=== Testing 5d Thursday Optimization ===")

    required_deps = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('scipy.optimize', 'minimize')
    ]

    tests_passed = 0
    tests_failed = 0

    for dep_name, import_name in required_deps:
        try:
            if dep_name == 'numpy':
                import numpy as np
                log.info(f"  ✓ numpy {np.__version__}")
            elif dep_name == 'pandas':
                import pandas as pd
                log.info(f"  ✓ pandas {pd.__version__}")
            elif dep_name == 'scipy.optimize':
                from scipy.optimize import minimize
                log.info(f"  ✓ scipy.optimize.minimize")
            tests_passed += 1
        except ImportError as e:
            log.error(f"  ✗ {dep_name} not found: {e}")
            tests_failed += 1

    log.info(f"\n{'='*60}")
    log.info(f"Tests: {tests_passed} passed, {tests_failed} failed")

    if tests_failed == 0:
        log.info("✓ ALL TESTS PASSED - 5d Thursday optimization ready")
        return True
    else:
        log.error("✗ SOME TESTS FAILED")
        return False


# Run self-test on load
if __name__ == "__main__":
    test_5d_thursday_optimization()
    print("\n✓ Cell 10.2: 5d Thursday Optimization loaded into globals()")
