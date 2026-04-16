"""
10_4_execute_thursday_opt.py
Cell ID: 4zfJSpR6ZlRZ
Exported: 2026-04-16T10:12:23.219179
"""

"""
Cell 10.4: Execute 5d Thursday HAR Optimization
Запускает оптимизацию и выводит результаты
"""

import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# ВАЖНО: Cell 10.3b работает ТОЛЬКО с кэшем — НЕ требует MOEX клиент
# Функция ожидает:
#   cache_dir: путь к директории с parquet файлами
#   symbols: список тикеров (опционально, можно auto-detect)

# Параметры загрузки данных
DURATION_DAYS = 730  # 2 года истории для обучения
CACHE_DIR = "/content/data/har_tuning_cache"  # Директория с parquet кэшем

# Директория для сохранения результатов
OUTPUT_DIR = "/content/thursday_optimization_results"

# Контракты для оптимизации (опционально — можно auto-detect из кэша)
SYMBOLS = None  # None = auto-detect все контракты из кэша

# ============================================================================
# EXECUTION
# ============================================================================

print("="*80)
print("CELL 10.4: EXECUTE 5D THURSDAY HAR OPTIMIZATION (CACHE-ONLY)")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Cache directory: {CACHE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Duration: {DURATION_DAYS} days")
if SYMBOLS:
    print(f"Symbols: {', '.join(SYMBOLS)}")
else:
    print("Symbols: AUTO-DETECT from cache")

# Проверка наличия функции в globals()
if "run_5d_thursday_comparison_from_cache" not in globals():
    print("\n❌ ERROR: run_5d_thursday_comparison_from_cache() not found in globals()")
    print("Please run Cell 10.3b first to load the cache-only comparison runner")
    sys.exit(1)

print("\n✓ run_5d_thursday_comparison_from_cache() found in globals()")

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("PRE-FLIGHT CHECKS")
print("=" * 80)

# Test 1: Check cache directory exists
print("\n[1/5] Checking cache directory...")
cache_path = Path(CACHE_DIR)
if not cache_path.exists():
    print(f"❌ FATAL: Cache directory not found: {CACHE_DIR}")
    print("Hint: Run the data loading cell first to download historical data")
    sys.exit(1)
print(f"✓ Cache directory exists: {CACHE_DIR}")

# Test 2: Check cached data
print("\n[2/5] Checking cached data...")
cached_files = list(cache_path.glob("*.parquet"))
if not cached_files:
    print(f"❌ FATAL: No cached .parquet files found in {CACHE_DIR}")
    print("Hint: Run the data loading cell first to download historical data")
    sys.exit(1)
print(f"✓ Found {len(cached_files)} cached contract files")

# Test 3: Output directory writable
print("\n[3/5] Checking output directory...")
output_path = Path(OUTPUT_DIR)
try:
    output_path.mkdir(parents=True, exist_ok=True)
    test_file = output_path / ".write_test"
    test_file.write_text("test")
    test_file.unlink()
    print(f"✓ Output directory is writable: {OUTPUT_DIR}")
except Exception as e:
    print(f"❌ FATAL: Cannot write to output directory: {OUTPUT_DIR}")
    print(f"Error: {e}")
    sys.exit(1)

# Test 4: Required dependencies
print("\n[4/5] Checking required dependencies...")
try:
    import numpy
    import pandas
    import scipy.optimize
    from tqdm.auto import tqdm
    print("✓ numpy, pandas, scipy, tqdm available")
except ImportError as e:
    print(f"❌ FATAL: Missing required dependency: {e}")
    sys.exit(1)

# Test 5: Check prerequisite cells loaded
print("\n[5/5] Checking prerequisite cells...")
required_functions = [
    "BacktestDataPipeline",
    "yang_zhang_series",
    "optimize_beta_coefficients",
    "find_next_thursday",
    "optimize_5d_for_thursday_single_contract"
]
missing = [fn for fn in required_functions if fn not in globals()]
if missing:
    print(f"❌ FATAL: Missing required functions: {', '.join(missing)}")
    print("Hint: Load prerequisite cells in order:")
    print("  1. Cell 03 (data_estimators)")
    print("  2. Cell 07.2.* (backtest modules)")
    print("  3. Cell 10.1 (thursday_helpers)")
    print("  4. Cell 10.2 (5d_thursday_optimization)")
    sys.exit(1)
print(f"✓ All prerequisite functions available")

print("\n" + "=" * 80)
print("✓ ALL PRE-FLIGHT CHECKS PASSED")
print("=" * 80)

# Запуск оптимизации
print("\n" + "=" * 80)
print("STARTING OPTIMIZATION...")
print("=" * 80)

try:
    # Progress tracking
    print(f"\nStarting optimization from cache...")
    print("Progress: This will be displayed by Cell 10.3b internally\n")

    # Запуск оптимизации БЕЗ moex_client (cache-only mode)
    results = run_5d_thursday_comparison_from_cache(
        cache_dir=CACHE_DIR,          # ← Читаем из кэша
        symbols=SYMBOLS,              # ← None = auto-detect из кэша
        duration_days=DURATION_DAYS,
        save_results=True,
        output_dir=OUTPUT_DIR,
        verbose=True
    )

except KeyboardInterrupt:
    print("\n\n❌ Optimization interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\n❌ FATAL ERROR during optimization:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    print(traceback.format_exc())
    sys.exit(1)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETED")
print("=" * 80)

# Cell 10.3 возвращает детальный dict с результатами, но БЕЗ поля "status"
# Проверяем наличие ключевых полей для валидации успеха
if "summary" in results and results["summary"].get("both_successful", 0) > 0:
    print(f"\n✓ Optimization completed successfully")
    print(f"✓ Contracts processed: {results['summary']['total_contracts']}")
    print(f"✓ Baseline successful: {results['summary']['baseline_successful']}")
    print(f"✓ Thursday successful: {results['summary']['thursday_successful']}")
    print(f"✓ Both successful: {results['summary']['both_successful']}")

    # Показать путь к сохранённому JSON
    if "meta" in results and "timestamp" in results["meta"]:
        timestamp = results["meta"]["timestamp"]
        json_filename = f"5d_thursday_comparison_{timestamp}.json"
        json_path = Path(OUTPUT_DIR) / json_filename
        print(f"\n✓ Results saved to: {json_path}")

    print("\n" + "-" * 80)
    print("HOW TO ACCESS RESULTS:")
    print("-" * 80)
    print("""
import json
from pathlib import Path

# Загрузить последний результат
results_dir = Path('/content/thursday_optimization_results')
latest_result = sorted(results_dir.glob('5d_thursday_comparison_*.json'))[-1]

with open(latest_result, 'r') as f:
    data = json.load(f)

# Просмотреть результаты для конкретного контракта
for baseline, thursday in zip(data['baseline_5d'], data['thursday_5d']):
    if baseline['status'] == 'success' and thursday['status'] == 'success':
        symbol = baseline['symbol']
        print(f"\\n{symbol}:")
        print(f"  Baseline:  β_d={baseline['optimal_betas']['daily']:.3f}, "
              f"β_w={baseline['optimal_betas']['weekly']:.3f}, "
              f"β_m={baseline['optimal_betas']['monthly']:.3f}")
        print(f"  Thursday:  β_d={thursday['optimal_betas']['daily']:.3f}, "
              f"β_w={thursday['optimal_betas']['weekly']:.3f}, "
              f"β_m={thursday['optimal_betas']['monthly']:.3f}")
    """)

elif "status" in results and results["status"] == "error":
    print(f"\n❌ Status: {results['status']}")
    print(f"❌ Message: {results.get('message', 'Unknown error')}")
    if "errors" in results:
        print("\nErrors:")
        for err in results["errors"]:
            print(f"  {err}")
else:
    print("\n⚠ Unexpected results format. Raw output:")
    print(results)

print("\n" + "=" * 80)
print("Cell 10.4: Optimization execution completed")
print("=" * 80)
