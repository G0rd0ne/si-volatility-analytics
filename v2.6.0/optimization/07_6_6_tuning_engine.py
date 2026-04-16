"""
07_6_6_tuning_engine.py
Cell ID: qlxxGXLHfvbw
Exported: 2026-04-16T10:12:23.219027
"""

"""
Cell 7.6.6: HAR Parameter Tuning Engine
========================================

Grid search и rolling validation для подбора оптимальных HAR параметров.

Author: Harvi
Version: 2.6.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger("har_tuning_engine")


class HARParameterTuningEngine:
    """
    Engine для grid search оптимизации HAR параметров.

    Workflow:
    1. Load historical data для всех контрактов
    2. Для каждого horizon_days:
       - Split data на train/test rolling windows
       - Grid search по β-коэффициентам
       - Optimize на train, validate на test
       - Сравнение с EWMA baseline
    3. Aggregate results и выбор best parameters
    """

    def __init__(self, tuning_config, trading_days_per_year: int = 252):
        """
        Args:
            tuning_config: HARTuningConfig instance
            trading_days_per_year: торговых дней в году
        """
        self.config = tuning_config
        self.tdy = trading_days_per_year

        # Import required functions from globals
        g = globals()
        self.yang_zhang_series = g['yang_zhang_series']
        self.compute_realized_vol_forward = g['compute_realized_vol_forward']
        self.forecast_har_rolling = g['forecast_har_rolling']
        self.forecast_ewma_rolling = g['forecast_ewma_rolling']
        self.evaluate_forecast = g['evaluate_forecast']
        self.optimize_beta_coefficients = g['optimize_beta_coefficients']

        if self.config.verbose:
            log.info(f"HAR Parameter Tuning Engine initialized")
            log.info(f"  Trading days per year: {self.tdy}")
            log.info(f"  Config: {self.config}")

    def run_grid_search(
        self,
        portfolio_data: Dict[str, pd.DataFrame]
    ) -> Dict[int, List]:
        """
        Запуск grid search для всех контрактов и горизонтов.

        Args:
            portfolio_data: {symbol: DataFrame} с OHLCV данными

        Returns:
            Dict[horizon_days, List[OptimizationResult]]
        """
        results_by_horizon = {}

        for horizon in self.config.horizon_days_grid:
            if self.config.verbose:
                log.info(f"\n{'=' * 80}")
                log.info(f"GRID SEARCH: horizon={horizon} days")
                log.info(f"{'=' * 80}")

            horizon_results = []

            for symbol, df in portfolio_data.items():
                if len(df) < self.config.min_bars_per_contract:
                    log.warning(f"Skipping {symbol}: insufficient data ({len(df)} bars < {self.config.min_bars_per_contract})")
                    continue

                result = self._optimize_single_contract(symbol, df, horizon)

                if result and result.optimization_success:
                    horizon_results.append(result)

                    if self.config.verbose:
                        log.info(f"✓ {symbol}: β=[{result.beta_d:.3f}, {result.beta_w:.3f}, {result.beta_m:.3f}], "
                                f"test_R²={result.test_r2:.4f}, improvement={result.improvement_r2_pp:.2f}pp")

            results_by_horizon[horizon] = horizon_results

            if self.config.verbose:
                log.info(f"\nHorizon {horizon}d: {len(horizon_results)} successful optimizations")

        return results_by_horizon

    def _optimize_single_contract(
        self,
        symbol: str,
        df: pd.DataFrame,
        horizon_days: int
    ) -> Optional:
        """
        Оптимизация для одного контракта с rolling validation.

        Args:
            symbol: contract symbol
            df: OHLCV DataFrame
            horizon_days: прогнозный горизонт

        Returns:
            OptimizationResult или None
        """
        # Import from globals (Jupyter/Colab namespace)
        g = globals()
        OptimizationResult = g['OptimizationResult']

        # Data quality check
        missing_pct = df[['open', 'high', 'low', 'close']].isna().sum().sum() / (len(df) * 4) * 100

        if missing_pct > self.config.max_missing_bars_pct * 100:
            log.warning(f"{symbol}: too much missing data ({missing_pct:.2f}%)")
            return None

        # Split data: train/test
        split_idx = int(len(df) * self.config.train_size)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        if len(train_df) < self.config.min_calibration_history:
            log.warning(f"{symbol}: insufficient training data ({len(train_df)} bars)")
            return None

        # Optimize на training set
        beta_d, beta_w, beta_m, train_metrics = self.optimize_beta_coefficients(
            train_df,
            horizon_days,
            self.compute_realized_vol_forward,
            self.forecast_har_rolling,
            self.evaluate_forecast,
            self.yang_zhang_series,
            tdy=self.tdy,
            verbose=False
        )

        # Validate на test set (out-of-sample)
        test_realized = self.compute_realized_vol_forward(
            test_df, horizon_days, self.yang_zhang_series, self.tdy, verbose=False
        )

        # CRITICAL FIX: Use full df for forecast_har_rolling, then filter to test period
        # Rolling forecast needs ALL historical data (train + test) for calibration
        # Otherwise forecast_har_rolling gets test_df with < min_calibration_history rows
        full_har_forecast = self.forecast_har_rolling(
            df, horizon_days, beta_d, beta_w, beta_m,
            self.yang_zhang_series, self.tdy,
            min_calibration_history=self.config.min_calibration_history,
            verbose=False
        )

        # Extract only test period forecasts by date index
        test_start_date = test_df['date'].iloc[0]
        test_har_forecast = full_har_forecast[full_har_forecast.index >= test_start_date]

        # CRITICAL: Check forecast length BEFORE evaluation
        if len(test_har_forecast.dropna()) < 10:
            log.warning(f"{symbol}: insufficient test forecasts ({len(test_har_forecast.dropna())})")
            return None

        test_metrics = self.evaluate_forecast(test_realized, test_har_forecast, verbose=False)

        # EWMA baseline comparison
        # CRITICAL FIX: Same as HAR - use full df then filter to test period
        full_ewma_forecast = self.forecast_ewma_rolling(
            df, horizon_days,
            self.yang_zhang_series, self.tdy,
            min_calibration_history=self.config.min_calibration_history,
            ewma_span=20,
            verbose=False
        )

        test_ewma_forecast = full_ewma_forecast[full_ewma_forecast.index >= test_start_date]

        ewma_metrics = self.evaluate_forecast(test_realized, test_ewma_forecast, verbose=False)

        # CRITICAL: Handle None values from evaluate_forecast when common_idx < 10
        # .get('R2', 0) returns None if key exists with None value, not the default 0
        # Use 'or 0' to convert None to 0 for arithmetic operations
        test_r2 = test_metrics.get('R2') or 0
        ewma_r2 = ewma_metrics.get('R2') or 0
        test_rmse = test_metrics.get('RMSE') or 1e6
        ewma_rmse = ewma_metrics.get('RMSE') or 1e6

        improvement_r2_pp = (test_r2 - ewma_r2) * 100
        improvement_rmse_pct = ((ewma_rmse - test_rmse) / ewma_rmse) * 100

        # Forecast stability
        forecast_std = test_har_forecast.std()
        forecast_mean = test_har_forecast.mean()

        result = OptimizationResult(
            contract=symbol,
            horizon_days=horizon_days,
            beta_d=beta_d,
            beta_w=beta_w,
            beta_m=beta_m,
            train_rmse=train_metrics.get('RMSE', np.nan),
            train_r2=train_metrics.get('R2', np.nan),
            train_mae=train_metrics.get('MAE', np.nan),
            test_rmse=test_metrics.get('RMSE', np.nan),
            test_r2=test_metrics.get('R2', np.nan),
            test_mae=test_metrics.get('MAE', np.nan),
            improvement_r2_pp=improvement_r2_pp,
            improvement_rmse_pct=improvement_rmse_pct,
            forecast_std=forecast_std,
            forecast_mean=forecast_mean,
            optimization_iterations=train_metrics.get('n_iterations', 0),
            optimization_success=True,
            optimization_message="Success",
            n_train_samples=len(train_df),
            n_test_samples=len(test_df),
            missing_data_pct=missing_pct
        )

        return result

    def aggregate_results(
        self,
        results_by_horizon: Dict[int, List]
    ) -> Dict[int, Dict]:
        """
        Агрегация результатов по горизонтам и выбор best parameters.

        Args:
            results_by_horizon: результаты grid search

        Returns:
            Dict[horizon_days, best_params_summary]
        """
        summary = {}

        for horizon, results_list in results_by_horizon.items():
            if not results_list:
                log.warning(f"No successful optimizations for horizon={horizon}")
                continue

            # Выбор best result по weighted score
            best_result = max(results_list, key=lambda r: r.score(self.config.weights))

            # Aggregate statistics
            test_r2_values = [r.test_r2 for r in results_list if not np.isnan(r.test_r2)]
            test_rmse_values = [r.test_rmse for r in results_list if not np.isnan(r.test_rmse)]
            improvement_values = [r.improvement_r2_pp for r in results_list if not np.isnan(r.improvement_r2_pp)]

            summary[horizon] = {
                "best_params": {
                    "beta_d": best_result.beta_d,
                    "beta_w": best_result.beta_w,
                    "beta_m": best_result.beta_m
                },
                "best_contract": best_result.contract,
                "best_metrics": {
                    "test_r2": best_result.test_r2,
                    "test_rmse": best_result.test_rmse,
                    "improvement_r2_pp": best_result.improvement_r2_pp
                },
                "aggregate_stats": {
                    "n_contracts": len(results_list),
                    "mean_test_r2": np.mean(test_r2_values) if test_r2_values else None,
                    "std_test_r2": np.std(test_r2_values) if test_r2_values else None,
                    "mean_test_rmse": np.mean(test_rmse_values) if test_rmse_values else None,
                    "mean_improvement_pp": np.mean(improvement_values) if improvement_values else None
                },
                "all_results": [r.to_dict() for r in results_list]
            }

            if self.config.verbose:
                log.info(f"\n=== BEST PARAMETERS FOR HORIZON={horizon}d ===")
                log.info(f"Contract: {best_result.contract}")
                log.info(f"Beta: [{best_result.beta_d:.3f}, {best_result.beta_w:.3f}, {best_result.beta_m:.3f}]")
                log.info(f"Test R²: {best_result.test_r2:.4f}")
                log.info(f"Test RMSE: {best_result.test_rmse:.6f}")
                log.info(f"Improvement vs EWMA: {best_result.improvement_r2_pp:.2f}pp")
                log.info(f"Aggregate: {len(results_list)} contracts, mean R²={np.mean(test_r2_values):.4f}")

        return summary


def save_tuning_results(results_summary: Dict, output_dir: str = "./har_tuning_results"):
    """
    Сохранение результатов parameter tuning в JSON.

    Args:
        results_summary: результаты aggregate_results
        output_dir: директория для сохранения
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"har_tuning_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results_summary, f, indent=2)

    log.info(f"✓ Tuning results saved to {filename}")
    return str(filename)


if __name__ == "__main__":
    print("✓ Cell 7.6.5: HAR Parameter Tuning Engine загружен успешно")
    print(f"  - Classes: HARParameterTuningEngine")
    print(f"  - Functions: save_tuning_results")
