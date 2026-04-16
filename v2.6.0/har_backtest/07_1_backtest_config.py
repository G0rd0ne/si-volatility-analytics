"""
07_1_backtest_config.py
Cell ID: CMshlQIJOpAR
Exported: 2026-04-16T10:12:23.218743
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.1/7.4
HAR-RV Backtest Configuration

Параметры backtest для β-калибровки и out-of-sample тестирования
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    """Параметры backtest для HAR-RV калибровки."""
    # Горизонты прогнозирования (в торговых днях)
    forecast_horizons: tuple[int, ...] = (1, 2, 3, 5, 7)

    # Rolling window для out-of-sample backtest (дней)
    rolling_window: int = 60

    # Минимальная история для калибровки (дней)
    min_calibration_history: int = 40

    # Grid search для β-оптимизации
    beta_grid_resolution: int = 10  # 10 шагов по каждой оси

    # Контракты для backtest (F0-F10)
    backtest_contracts: tuple[str, ...] = ("F0", "F-1", "F-2", "F-3", "F-4",
                                            "F-5", "F-6", "F-7", "F-8", "F-9", "F-10")

    # Выходные файлы
    output_dir: str = "./backtest_results"
    save_plots: bool = True
    save_json: bool = True


if __name__ == "__main__":
    print("Cell 7.1/7.4: Backtest Config загружен")
