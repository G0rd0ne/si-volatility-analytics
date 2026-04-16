"""
07_3_backtest_viz.py
Cell ID: os-mMOfBO0eN
Exported: 2026-04-16T10:12:23.218868
"""

"""
Cell 7.3: HAR Backtest Visualization & Persistence
===================================================

Модуль визуализации результатов бэктеста HAR-модели.

Функционал:
- Визуализация метрик качества прогнозов
- Сравнение HAR vs EWMA
- Визуализация оптимальных β-коэффициентов
- Сохранение результатов в JSON

Совместим с новой архитектурой Cell 7.2.09a-09b (HARBacktestEngine)

Author: Harvi (HAR Optimization System)
Version: 2.6.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger("har_backtest_viz")


def plot_backtest_results(
    results: Dict[str, dict],
    output_dir: str = "./backtest_results",
    show_plots: bool = False
) -> None:
    """
    Визуализация результатов бэктеста для портфеля контрактов.

    Args:
        results: Dict {symbol: backtest_result} из HARBacktestEngine.run_portfolio()
        output_dir: путь для сохранения графиков
        show_plots: показывать графики (для notebook)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results:
        log.warning("No results to plot")
        return

    # Извлекаем метрики для всех контрактов
    symbols = list(results.keys())
    har_r2 = []
    ewma_r2 = []
    har_rmse = []
    ewma_rmse = []
    betas_daily = []
    betas_weekly = []
    betas_monthly = []

    for symbol in symbols:
        res = results[symbol]
        comp = res['comparison']

        har_r2.append(comp['HAR_metrics']['R2'] or 0)
        ewma_r2.append(comp['EWMA_metrics']['R2'] or 0)
        har_rmse.append(comp['HAR_metrics']['RMSE'])
        ewma_rmse.append(comp['EWMA_metrics']['RMSE'])

        betas_daily.append(res['betas']['daily'])
        betas_weekly.append(res['betas']['weekly'])
        betas_monthly.append(res['betas']['monthly'])

    # Plot 1: R² comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(symbols))
    width = 0.35

    ax.bar(x - width/2, har_r2, width, label='HAR-RV', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, ewma_r2, width, label='EWMA', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Contract', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² (Out-of-Sample)', fontsize=12, fontweight='bold')
    ax.set_title('HAR-RV vs EWMA: Forecast Accuracy (R²)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path / "backtest_r2_comparison.png", dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Plot 2: RMSE comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, har_rmse, width, label='HAR-RV', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, ewma_rmse, width, label='EWMA', color='#e67e22', alpha=0.8)

    ax.set_xlabel('Contract', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('HAR-RV vs EWMA: Forecast Error (RMSE, lower is better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path / "backtest_rmse_comparison.png", dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Plot 3: Beta coefficients по контрактам
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(symbols, betas_daily, 'o-', label='β_daily', linewidth=2.5, markersize=8, color='#e74c3c')
    ax.plot(symbols, betas_weekly, 's-', label='β_weekly', linewidth=2.5, markersize=8, color='#3498db')
    ax.plot(symbols, betas_monthly, '^-', label='β_monthly', linewidth=2.5, markersize=8, color='#2ecc71')

    ax.set_xlabel('Contract', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimal β Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('HAR-RV: Optimal Beta Weights by Contract', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticklabels(symbols, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path / "backtest_beta_weights.png", dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Plot 4: Forecast accuracy heatmap (R² improvement HAR vs EWMA)
    improvements = [r2_har - r2_ewma for r2_har, r2_ewma in zip(har_r2, ewma_r2)]

    fig, ax = plt.subplots(figsize=(12, 3))

    colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]
    ax.bar(symbols, improvements, color=colors, alpha=0.8)

    ax.set_xlabel('Contract', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Improvement (HAR - EWMA)', fontsize=12, fontweight='bold')
    ax.set_title('HAR Advantage over EWMA (Positive = HAR wins)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticklabels(symbols, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path / "backtest_improvement.png", dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    log.info(f"✓ Plots saved to {output_path}")


def save_backtest_json(
    results: Dict[str, dict],
    output_dir: str = "./backtest_results",
    timestamp: Optional[str] = None
) -> Path:
    """
    Сохранение результатов бэктеста в JSON.

    Args:
        results: Dict {symbol: backtest_result}
        output_dir: путь для сохранения JSON
        timestamp: метка времени (если None, создается автоматически)

    Returns:
        Path к созданному файлу
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = output_path / f"har_backtest_{timestamp}.json"

    # Преобразуем pandas Series в списки для JSON-сериализации
    serializable_results = {}

    for symbol, res in results.items():
        serializable_results[symbol] = {
            'symbol': res['symbol'],
            'horizon_days': res['horizon_days'],
            'data_bars': res['data_bars'],
            'date_start': str(res['date_start']),
            'date_end': str(res['date_end']),
            'betas': res['betas'],
            'optimization_metrics': res['optimization_metrics'],
            'comparison': {
                'HAR_metrics': res['comparison']['HAR_metrics'],
                'EWMA_metrics': res['comparison']['EWMA_metrics'],
                'improvement': res['comparison']['improvement']
            },
            # Сохраняем только summary realized/forecast данных
            'realized_summary': {
                'count': len(res['realized']),
                'mean': float(res['realized'].mean()),
                'std': float(res['realized'].std()),
                'min': float(res['realized'].min()),
                'max': float(res['realized'].max())
            },
            'har_forecast_summary': {
                'count': len(res['har_forecast']),
                'mean': float(res['har_forecast'].mean()),
                'std': float(res['har_forecast'].std()),
                'min': float(res['har_forecast'].min()),
                'max': float(res['har_forecast'].max())
            },
            'ewma_forecast_summary': {
                'count': len(res['ewma_forecast']),
                'mean': float(res['ewma_forecast'].mean()),
                'std': float(res['ewma_forecast'].std()),
                'min': float(res['ewma_forecast'].min()),
                'max': float(res['ewma_forecast'].max())
            }
        }

    # Добавляем общую метаинформацию
    output_data = {
        'metadata': {
            'backtest_timestamp': timestamp,
            'total_contracts': len(results),
            'contracts': list(results.keys())
        },
        'results': serializable_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log.info(f"✓ Results saved to {output_file}")

    return output_file


def validate_backtest_results(results: Dict[str, dict]) -> Dict[str, list]:
    """
    Валидирует результаты бэктеста на качество данных.

    Args:
        results: Dict {symbol: backtest_result}

    Returns:
        Dict с предупреждениями и ошибками
    """
    warnings = []
    errors = []

    for symbol, res in results.items():
        # Проверка 1: Достаточно наблюдений
        if res['comparison']['HAR_metrics']['n_obs'] < 20:
            warnings.append(
                f"{symbol}: Low sample size ({res['comparison']['HAR_metrics']['n_obs']} obs)"
            )

        # Проверка 2: HAR R² разумен
        har_r2 = res['comparison']['HAR_metrics']['R2']
        if har_r2 and (har_r2 < -0.5 or har_r2 > 1.0):
            errors.append(
                f"{symbol}: Suspicious HAR R² = {har_r2:.3f} (expected range: -0.5 to 1.0)"
            )

        # Проверка 3: EWMA R² разумен
        ewma_r2 = res['comparison']['EWMA_metrics']['R2']
        if ewma_r2 and (ewma_r2 < -0.5 or ewma_r2 > 1.0):
            errors.append(
                f"{symbol}: Suspicious EWMA R² = {ewma_r2:.3f} (expected range: -0.5 to 1.0)"
            )

        # Проверка 4: Beta сумма близка к 1
        beta_sum = res['betas']['daily'] + res['betas']['weekly'] + res['betas']['monthly']
        if not (0.8 <= beta_sum <= 1.2):
            warnings.append(
                f"{symbol}: Beta sum = {beta_sum:.3f} (expected ~1.0)"
            )

    return {
        'warnings': warnings,
        'errors': errors
    }


if __name__ == "__main__":
    print("✓ Cell 7.3: HAR Backtest Visualization загружен успешно")
    print(f"  - Функции: plot_backtest_results, save_backtest_json, validate_backtest_results")
