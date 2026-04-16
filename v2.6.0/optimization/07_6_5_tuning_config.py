"""
07_6_5_tuning_config.py
Cell ID: ZocF4QjFbBYu
Exported: 2026-04-16T10:12:23.219007
"""

"""
Cell 7.6.4: HAR Parameter Tuning Configuration
===============================================

Конфигурация для подбора оптимальных параметров HAR модели.
Включает grid search по β-коэффициентам и горизонтам прогнозирования.

Author: Harvi
Version: 2.6.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

log = logging.getLogger("har_parameter_tuning")


@dataclass
class HARTuningConfig:
    """Конфигурация для grid search оптимизации HAR параметров."""

    # Trading days per year
    tdy: int = 252

    # Горизонты прогнозирования для тестирования
    horizon_days_grid: List[int] = field(default_factory=lambda: [5, 10, 21, 60])

    # Grid search для β-коэффициентов (initial guesses)
    beta_grid: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.35, 0.40, 0.25),  # Default: weekly emphasis
        (0.50, 0.30, 0.20),  # Daily emphasis
        (0.25, 0.50, 0.25),  # Strong weekly emphasis
        (0.20, 0.40, 0.40),  # Monthly emphasis
        (0.33, 0.33, 0.34),  # Equal weights
        (0.40, 0.35, 0.25),  # Balanced daily-weekly
        (0.30, 0.30, 0.40),  # Monthly preference
    ])

    # Optimization settings
    optimization_method: str = "SLSQP"
    max_iterations: int = 100
    ftol: float = 1e-6

    # Rolling validation settings
    train_size: float = 0.6  # 60% для training
    test_size: float = 0.4   # 40% для validation
    rolling_windows: int = 5  # Количество rolling windows

    # Minimum data requirements
    min_calibration_history: int = 60
    min_bars_per_contract: int = 120

    # Data quality thresholds
    max_missing_bars_pct: float = 0.05  # 5% max missing data
    min_r2_threshold: float = 0.0  # Минимальный R² для валидного результата

    # Performance metrics weights (для multi-objective optimization)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "rmse": -1.0,      # Minimize RMSE (negative weight)
        "r2": 0.5,         # Maximize R²
        "mae": -0.3,       # Minimize MAE
        "stability": 0.2   # Maximize forecast stability (low variance)
    })

    # Контракты для backtest (F0-F10)
    contract_symbols: List[str] = field(default_factory=lambda: [
        f"F{i}" for i in range(11)
    ])

    # Data loading settings
    duration_days: int = 365  # 1 год исторических данных
    bar_size: str = "1 hour"

    # Output settings
    save_intermediate_results: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Валидация конфигурации."""
        assert 0 < self.train_size < 1, "train_size должен быть в (0, 1)"
        assert 0 < self.test_size < 1, "test_size должен быть в (0, 1)"
        assert abs(self.train_size + self.test_size - 1.0) < 0.01, "train_size + test_size должно = 1.0"
        assert self.rolling_windows >= 1, "rolling_windows должно быть >= 1"
        assert all(sum(b) - 1.0 < 0.05 for b in self.beta_grid), "Сумма β должна быть ≈ 1.0"

        if self.verbose:
            log.info(f"HAR Tuning Config initialized:")
            log.info(f"  Horizons: {self.horizon_days_grid}")
            log.info(f"  Beta grid: {len(self.beta_grid)} candidates")
            log.info(f"  Rolling windows: {self.rolling_windows}")
            log.info(f"  Train/test split: {self.train_size}/{self.test_size}")


@dataclass
class OptimizationResult:
    """Результат оптимизации для одного контракта и горизонта."""

    contract: str
    horizon_days: int

    # Оптимальные β-коэффициенты
    beta_d: float
    beta_w: float
    beta_m: float

    # Метрики на training set
    train_rmse: float
    train_r2: float
    train_mae: float

    # Метрики на test set (out-of-sample)
    test_rmse: float
    test_r2: float
    test_mae: float

    # Сравнение с EWMA baseline
    improvement_r2_pp: float  # Improvement in percentage points
    improvement_rmse_pct: float  # Improvement in percent

    # Stability metrics
    forecast_std: float
    forecast_mean: float

    # Optimization metadata
    optimization_iterations: int
    optimization_success: bool
    optimization_message: str

    # Data quality
    n_train_samples: int
    n_test_samples: int
    missing_data_pct: float

    def score(self, weights: Dict[str, float]) -> float:
        """Вычисляет взвешенный скор для multi-objective optimization."""
        score = 0.0

        # Используем test metrics для оценки (out-of-sample performance)
        score += weights.get("rmse", 0) * self.test_rmse
        score += weights.get("r2", 0) * self.test_r2
        score += weights.get("mae", 0) * self.test_mae

        # Stability penalty: prefer stable forecasts
        if self.forecast_mean > 0:
            cv = self.forecast_std / self.forecast_mean  # Coefficient of variation
            score += weights.get("stability", 0) * (1.0 - cv)

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в dict для сохранения."""
        return {
            "contract": self.contract,
            "horizon_days": self.horizon_days,
            "beta_d": round(self.beta_d, 4),
            "beta_w": round(self.beta_w, 4),
            "beta_m": round(self.beta_m, 4),
            "train_metrics": {
                "rmse": round(self.train_rmse, 6),
                "r2": round(self.train_r2, 4),
                "mae": round(self.train_mae, 6)
            },
            "test_metrics": {
                "rmse": round(self.test_rmse, 6),
                "r2": round(self.test_r2, 4),
                "mae": round(self.test_mae, 6)
            },
            "improvement_vs_ewma": {
                "r2_pp": round(self.improvement_r2_pp, 2),
                "rmse_pct": round(self.improvement_rmse_pct, 2)
            },
            "stability": {
                "forecast_std": round(self.forecast_std, 6),
                "forecast_mean": round(self.forecast_mean, 6),
                "cv": round(self.forecast_std / self.forecast_mean, 4) if self.forecast_mean > 0 else None
            },
            "optimization": {
                "iterations": self.optimization_iterations,
                "success": self.optimization_success,
                "message": self.optimization_message
            },
            "data_quality": {
                "n_train": self.n_train_samples,
                "n_test": self.n_test_samples,
                "missing_pct": round(self.missing_data_pct, 2)
            }
        }


if __name__ == "__main__":
    print("✓ Cell 7.6.4: HAR Parameter Tuning Configuration загружен успешно")
    print(f"  - Classes: HARTuningConfig, OptimizationResult")

    # Test config creation
    config = HARTuningConfig()
    print(f"\n  Default config:")
    print(f"    Horizons: {config.horizon_days_grid}")
    print(f"    Beta candidates: {len(config.beta_grid)}")
    print(f"    Rolling windows: {config.rolling_windows}")
