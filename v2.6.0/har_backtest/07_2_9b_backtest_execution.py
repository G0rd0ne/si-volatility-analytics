"""
07_2_9b_backtest_execution.py
Cell ID: EAkYYJ81CK0i
Exported: 2026-04-16T10:12:23.218849
"""

"""
Cell 7.2.9b: HAR Backtest Execution Engine
===========================================

Модуль выполнения бэктеста HAR-модели с оптимизацией и визуализацией.

Функционал:
- Оптимизация β-коэффициентов HAR
- Генерация HAR и EWMA прогнозов
- Сравнение качества прогнозов
- Визуализация результатов

Author: Harvi (HAR Optimization System)
Version: 2.6.0
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

log = logging.getLogger("har_optimization")


class HARBacktestEngine:
    """
    Движок выполнения бэктеста HAR-модели.
    """

    def __init__(
        self,
        horizon_days: int = 21,
        trading_days_per_year: int = 252
    ):
        """
        Args:
            horizon_days: горизонт прогнозирования (21 дней = 1 месяц)
            trading_days_per_year: торговых дней в году
        """
        self.horizon = horizon_days
        self.tdy = trading_days_per_year
        self.results: Dict[str, dict] = {}

    def run_single_contract(
        self,
        symbol: str,
        df: pd.DataFrame,
        optimize_betas: bool = True,
        verbose: bool = True
    ) -> Optional[dict]:
        """
        Запускает бэктест для одного контракта.

        Args:
            symbol: тикер контракта
            df: DataFrame с OHLCV
            optimize_betas: оптимизировать β-коэффициенты
            verbose: логирование

        Returns:
            Dict с результатами бэктеста или None при ошибке
        """
        # Получаем функции из globals
        g = globals()
        yang_zhang_series = g['yang_zhang_series']
        compute_realized_vol_forward = g['compute_realized_vol_forward']
        forecast_har_rolling = g['forecast_har_rolling']
        forecast_ewma_rolling = g['forecast_ewma_rolling']
        compare_forecasts = g['compare_forecasts']
        optimize_beta_coefficients = g.get('optimize_beta_coefficients')

        if verbose:
            log.info(f"\n{'=' * 80}")
            log.info(f"BACKTEST: {symbol}")
            log.info(f"Horizon: {self.horizon} days")
            log.info(f"Data: {len(df)} bars from {df['date'].min()} to {df['date'].max()}")
            log.info(f"{'=' * 80}")

        try:
            # Шаг 1: Оптимизация β-коэффициентов (если включена)
            if optimize_betas and optimize_beta_coefficients is not None:
                if verbose:
                    log.info("\n[1/4] Optimizing HAR beta coefficients...")

                beta_d, beta_w, beta_m, opt_metrics = optimize_beta_coefficients(
                    df,
                    self.horizon,
                    yang_zhang_series,
                    compute_realized_vol_forward,
                    forecast_har_rolling,
                    self.tdy,
                    verbose=verbose
                )

                if verbose:
                    log.info(f"✓ Optimal betas: daily={beta_d:.3f}, weekly={beta_w:.3f}, monthly={beta_m:.3f}")
            else:
                # Стандартные β для среднесрочного прогноза
                beta_d, beta_w, beta_m = 0.35, 0.40, 0.25
                opt_metrics = {}
                if verbose:
                    log.info(f"\n[1/4] Using default betas: daily={beta_d}, weekly={beta_w}, monthly={beta_m}")

            # Шаг 2: Вычисление realized volatility
            if verbose:
                log.info("\n[2/4] Computing realized volatility forward...")

            realized = compute_realized_vol_forward(
                df, self.horizon, yang_zhang_series, self.tdy, verbose=False
            )

            if len(realized) < 20:
                if verbose:
                    log.warning(f"✗ Insufficient realized vol observations: {len(realized)}")
                return None

            if verbose:
                log.info(f"✓ Computed {len(realized)} forward RV observations")

            # Шаг 3: HAR прогноз
            if verbose:
                log.info("\n[3/4] Generating HAR forecast...")

            har_forecast = forecast_har_rolling(
                df, self.horizon, beta_d, beta_w, beta_m,
                yang_zhang_series, self.tdy,
                min_calibration_history=60,
                verbose=False
            )

            if len(har_forecast) < 10:
                if verbose:
                    log.warning(f"✗ Insufficient HAR forecast: {len(har_forecast)}")
                return None

            if verbose:
                log.info(f"✓ Generated {len(har_forecast)} HAR forecasts")

            # Шаг 4: EWMA baseline прогноз
            if verbose:
                log.info("\n[4/4] Generating EWMA baseline forecast...")

            ewma_forecast = forecast_ewma_rolling(
                df, self.horizon, yang_zhang_series, self.tdy,
                ewma_span=20,
                min_calibration_history=60,
                verbose=False
            )

            if len(ewma_forecast) < 10:
                if verbose:
                    log.warning(f"✗ Insufficient EWMA forecast: {len(ewma_forecast)}")
                return None

            if verbose:
                log.info(f"✓ Generated {len(ewma_forecast)} EWMA forecasts")

            # Шаг 5: Сравнение качества прогнозов
            comparison = compare_forecasts(
                realized, har_forecast, ewma_forecast, verbose=False
            )

            # Результаты
            result = {
                'symbol': symbol,
                'horizon_days': self.horizon,
                'data_bars': len(df),
                'date_start': df['date'].min(),
                'date_end': df['date'].max(),
                'betas': {
                    'daily': beta_d,
                    'weekly': beta_w,
                    'monthly': beta_m
                },
                'optimization_metrics': opt_metrics,
                'realized': realized,
                'har_forecast': har_forecast,
                'ewma_forecast': ewma_forecast,
                'comparison': comparison
            }

            if verbose:
                self._print_backtest_summary(result)

            return result

        except Exception as e:
            log.error(f"✗ Backtest failed for {symbol}: {e}")
            return None

    def run_portfolio(
        self,
        portfolio_data: Dict[str, pd.DataFrame],
        optimize_betas: bool = True,
        verbose: bool = True
    ) -> Dict[str, dict]:
        """
        Запускает бэктест для портфеля контрактов.

        Args:
            portfolio_data: Dict {symbol: DataFrame}
            optimize_betas: оптимизировать β для каждого контракта
            verbose: логирование

        Returns:
            Dict {symbol: backtest_results}
        """
        if verbose:
            log.info(f"\n{'=' * 80}")
            log.info(f"PORTFOLIO BACKTEST")
            log.info(f"Contracts: {len(portfolio_data)}")
            log.info(f"Horizon: {self.horizon} days")
            log.info(f"Beta optimization: {'Enabled' if optimize_betas else 'Disabled'}")
            log.info(f"{'=' * 80}")

        results = {}
        failed_symbols = []

        for i, (symbol, df) in enumerate(portfolio_data.items(), 1):
            if verbose:
                log.info(f"\n[{i}/{len(portfolio_data)}] Running backtest for {symbol}...")

            result = self.run_single_contract(
                symbol, df,
                optimize_betas=optimize_betas,
                verbose=verbose
            )

            if result is not None:
                results[symbol] = result
                self.results[symbol] = result
            else:
                failed_symbols.append(symbol)

        # Итоговая статистика
        if verbose:
            log.info(f"\n{'=' * 80}")
            log.info(f"PORTFOLIO BACKTEST COMPLETE")
            log.info(f"✓ Successful: {len(results)} contracts")
            if failed_symbols:
                log.warning(f"✗ Failed: {len(failed_symbols)} contracts ({', '.join(failed_symbols)})")
            log.info(f"{'=' * 80}")

        return results

    def _print_backtest_summary(self, result: dict):
        """Выводит краткую сводку результатов бэктеста."""
        comp = result['comparison']

        log.info(f"\n{'=' * 80}")
        log.info(f"BACKTEST RESULTS: {result['symbol']}")
        log.info(f"{'=' * 80}")

        log.info(f"\nData period: {result['date_start']} to {result['date_end']}")
        log.info(f"Observations: {len(result['realized'])} forward RV points")

        log.info(f"\nOptimal HAR betas:")
        log.info(f"  β_daily   = {result['betas']['daily']:.3f}")
        log.info(f"  β_weekly  = {result['betas']['weekly']:.3f}")
        log.info(f"  β_monthly = {result['betas']['monthly']:.3f}")

        log.info(f"\nHAR Performance:")
        har_m = comp['har_metrics']
        log.info(f"  R²   = {har_m['R2']:.4f}" if har_m['R2'] else "  R²   = N/A")
        log.info(f"  RMSE = {har_m['RMSE']:.6f}")
        log.info(f"  MAE  = {har_m['MAE']:.6f}")

        log.info(f"\nEWMA Baseline:")
        ewma_m = comp['ewma_metrics']
        log.info(f"  R²   = {ewma_m['R2']:.4f}" if ewma_m['R2'] else "  R²   = N/A")
        log.info(f"  RMSE = {ewma_m['RMSE']:.6f}")
        log.info(f"  MAE  = {ewma_m['MAE']:.6f}")

        if comp.get('improvement_rmse'):
            log.info(f"\nHAR vs EWMA improvement:")
            log.info(f"  RMSE: {comp['improvement_rmse']:+.2%}")
            if comp.get('improvement_r2'):
                log.info(f"  R²:   {comp['improvement_r2']:+.2%}")

        log.info(f"{'=' * 80}")


if __name__ == "__main__":
    print("✓ Cell 7.2.9b: HAR Backtest Execution Engine загружен успешно")
    print(f"  - Класс: HARBacktestEngine")
    print(f"  - Методы: run_single_contract, run_portfolio")
