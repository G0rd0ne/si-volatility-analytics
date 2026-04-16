"""
07_2_9a_data_pipeline.py
Cell ID: xZxqEXuxCI6U
Exported: 2026-04-16T10:12:23.218823
"""

"""
Cell 7.2.9a: HAR Backtest Data Pipeline
========================================

Модуль загрузки и подготовки исторических данных для бэктеста HAR-модели.

Функционал:
- Загрузка OHLCV данных для портфеля контрактов
- Агрегация на дневные бары
- Валидация данных
- Кэширование для повторных запусков

Author: Harvi (HAR Optimization System)
Version: 2.6.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

log = logging.getLogger("har_optimization")

# Ensure logging is configured for Jupyter/Colab environments
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )


class BacktestDataPipeline:
    """
    Пайплайн загрузки и подготовки данных для HAR-бэктеста.

    Data source: MOEX ISS API (https://iss.moex.com/iss)
    """

    def __init__(
        self,
        moex_client,
        trading_days_per_year: int = 252,
        cache_dir: str = "./data/cache"
    ):
        """
        Args:
            moex_client: MoexClient instance (из cell_02_moex_contracts.py)
            trading_days_per_year: торговых дней в году для аннуализации
            cache_dir: директория для сохранения parquet файлов
        """
        self.moex_client = moex_client
        self.tdy = trading_days_per_year
        self.cache_dir = cache_dir
        self._cache: Dict[str, pd.DataFrame] = {}

        # Создаем директорию для кэша
        from pathlib import Path
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def load_contract_history(
        self,
        contract,
        duration_days: int = 365,
        bar_size: str = '1 hour',
        verbose: bool = True,
        contract_role: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Загружает исторические данные для одного контракта через MOEX ISS API.

        Args:
            contract: MOEX contract metadata (ContractMeta object)
            duration_days: сколько дней истории загружать
            bar_size: размер бара (игнорируется, MOEX всегда дает дневные бары)
            verbose: логирование

        Returns:
            DataFrame с OHLCV или None при ошибке

        Data source: MOEX ISS API через moex_load_candles
        """
        # Получаем функции из globals (загружены через %run)
        g = globals()
        moex_load_candles = g.get('moex_load_candles')
        validate_ohlcv = g.get('validate_ohlcv')

        if not moex_load_candles or not validate_ohlcv:
            log.error("✗ Required functions not loaded. Please run cell_03_data_estimators.py first.")
            return None

        # Унифицированный доступ к ticker
        if isinstance(contract, str):
            ticker = contract
        else:
            ticker = getattr(contract, 'ticker', None) or getattr(contract, 'symbol', str(contract))

        cache_key = f"{ticker}_{duration_days}"

        if cache_key in self._cache:
            if verbose:
                log.info(f"✓ Using cached data for {ticker}")
            return self._cache[cache_key]

        if verbose:
            log.info(f"\n{'=' * 80}")
            if contract_role:
                log.info(f"Loading history for {contract_role} ({ticker})")
            else:
                log.info(f"Loading history for {ticker}")
            log.info(f"Duration: {duration_days} days")

        try:
            # Получаем expiry для определения диапазона дат
            if hasattr(contract, 'expiry'):
                # Expired contract: загружаем duration_days до expiry
                expiry = contract.expiry
                date_till = expiry.strftime("%Y-%m-%d")
                date_from = (expiry - timedelta(days=duration_days)).strftime("%Y-%m-%d")
            else:
                # Active contract: загружаем duration_days до сегодня
                from datetime import date
                today = date.today()
                date_till = today.strftime("%Y-%m-%d")
                date_from = (today - timedelta(days=duration_days)).strftime("%Y-%m-%d")

            if verbose:
                log.info(f"  Date range: {date_from} → {date_till}")

            # Загружаем дневные бары через MOEX ISS API
            df = moex_load_candles(
                self.moex_client,
                self.moex_client._cfg,  # MoexConfig (приватный атрибут)
                ticker,
                date_from,
                date_till
            )

            if df.empty:
                if verbose:
                    log.warning(f"✗ No data received for {ticker}")
                return None

            # Валидация OHLCV
            df = validate_ohlcv(df, ticker)

            if df.empty:
                if verbose:
                    log.warning(f"✗ Validation failed for {ticker}")
                return None

            # Валидация минимального количества данных
            if len(df) < 60:
                if verbose:
                    log.warning(
                        f"✗ Insufficient data for {ticker}: "
                        f"{len(df)} bars (minimum 60 required)"
                    )
                return None

            # Сохранение в parquet
            from pathlib import Path
            parquet_file = Path(self.cache_dir) / f"{ticker}_{duration_days}d.parquet"
            df.to_parquet(parquet_file, index=False)

            if verbose:
                role_label = f"{contract_role} ({ticker})" if contract_role else ticker
                log.info(f"✓ Loaded {len(df)} daily bars for {role_label}")
                log.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                log.info(f"  Saved to: {parquet_file}")
                log.info(f"  File size: {parquet_file.stat().st_size / 1024:.1f} KB")

            # Кэшируем результат в памяти
            self._cache[cache_key] = df

            return df

        except Exception as e:
            log.error(f"✗ Failed to load data for {ticker}: {e}")
            import traceback
            if verbose:
                log.debug(traceback.format_exc())
            return None

    def load_portfolio_history(
        self,
        contracts: List,
        duration_days: int = 365,
        bar_size: str = '1 hour',
        verbose: bool = True,
        contracts_roles: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Загружает исторические данные для портфеля контрактов через MOEX ISS API.

        Args:
            contracts: список MOEX contract metadata objects
            duration_days: сколько дней истории
            bar_size: размер бара
            verbose: логирование

        Returns:
            Dict {symbol: DataFrame} для успешно загруженных контрактов

        Data source: MOEX ISS API (https://iss.moex.com/iss)
        """
        if verbose:
            log.info(f"\n{'=' * 80}")
            log.info(f"PORTFOLIO DATA LOADING")
            log.info(f"Contracts: {len(contracts)}")
            log.info(f"Duration: {duration_days} days")
            log.info(f"{'=' * 80}")

        portfolio_data = {}
        failed_symbols = []

        for i, contract in enumerate(contracts, 1):
            # Унифицированный доступ: поддержка .symbol, .ticker или строки
            if isinstance(contract, str):
                symbol = contract
            else:
                symbol = getattr(contract, 'symbol', None) or getattr(contract, 'ticker', str(contract))

            # Получаем роль контракта (F0, F-1, ..., F-10)
            contract_role = None
            if contracts_roles:
                contract_role = contracts_roles.get(symbol)

            if verbose:
                if contract_role:
                    log.info(f"\n[{i}/{len(contracts)}] Processing {contract_role} ({symbol})...")
                else:
                    log.info(f"\n[{i}/{len(contracts)}] Processing {symbol}...")

            df = self.load_contract_history(
                contract,
                duration_days=duration_days,
                bar_size=bar_size,
                verbose=verbose,
                contract_role=contract_role
            )

            if df is not None:
                portfolio_data[symbol] = df
            else:
                failed_symbols.append(symbol)

        # Итоговая статистика
        if verbose:
            log.info(f"\n{'=' * 80}")
            log.info(f"PORTFOLIO DATA LOADING COMPLETE")
            log.info(f"✓ Successfully loaded: {len(portfolio_data)} contracts")
            if failed_symbols:
                log.warning(f"✗ Failed to load: {len(failed_symbols)} contracts")
                log.warning(f"  Failed symbols: {', '.join(failed_symbols)}")
            log.info(f"{'=' * 80}")

        return portfolio_data

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        min_bars: int = 60,
        max_missing_pct: float = 0.05,
        verbose: bool = True
    ) -> bool:
        """
        Проверяет качество данных для бэктеста.

        Args:
            df: DataFrame с OHLCV
            min_bars: минимальное количество баров
            max_missing_pct: максимальный % пропусков
            verbose: логирование

        Returns:
            True если данные подходят для бэктеста
        """
        issues = []

        # Проверка 1: Достаточно баров
        if len(df) < min_bars:
            issues.append(f"Insufficient bars: {len(df)} < {min_bars}")

        # Проверка 2: Пропуски в критичных колонках
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            missing = df[col].isna().sum()
            missing_pct = missing / len(df)
            if missing_pct > max_missing_pct:
                issues.append(
                    f"Too many missing values in '{col}': "
                    f"{missing_pct:.2%} > {max_missing_pct:.2%}"
                )

        # Проверка 3: Валидность OHLC соотношений
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()

        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC relationships: {invalid_ohlc} bars")

        # Результат
        if issues:
            if verbose:
                log.warning(f"✗ Data quality issues detected:")
                for issue in issues:
                    log.warning(f"  - {issue}")
            return False
        else:
            if verbose:
                log.info(f"✓ Data quality validation passed")
            return True


def preload_contracts_data(
    moex_client,
    moex_config,
    contract_roles: List[str] = None,
    duration_days: int = 365,
    cache_dir: str = "./data/har_tuning_cache",
    verbose: bool = True
):
    """
    Предзагрузка исторических данных для контрактов при запуске ячейки.

    Args:
        moex_client: MoexClient instance
        moex_config: MoexConfig instance
        contract_roles: Список ролей контрактов (F0, F-1, ..., F-10)
        duration_days: Количество дней истории
        cache_dir: Директория кэша
        verbose: Детальное логирование

    Returns:
        Dict с загруженными данными
    """
    from datetime import date

    if contract_roles is None:
        contract_roles = ["F0", "F-1", "F-2", "F-3", "F-4", "F-5", "F-6", "F-7", "F-8", "F-9", "F-10"]

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"ПРЕДЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
        print(f"Контракты: {', '.join(contract_roles)}")
        print(f"История: {duration_days} дней")
        print(f"Кэш: {cache_dir}")
        print(f"{'=' * 80}")

    # Import identify_contracts from globals
    g = globals()
    identify_contracts = g.get('identify_contracts')

    if not identify_contracts:
        print("✗ identify_contracts() не загружена. Запустите cell_02_moex_contracts.py сначала.")
        return {}

    # Идентификация контрактов
    try:
        contracts_dict = identify_contracts(moex_client, moex_config, date.today())
    except Exception as e:
        print(f"✗ Ошибка идентификации контрактов: {e}")
        return {}

    # Собираем контракты для загрузки
    contracts = []
    symbol_to_role = {}

    for role in contract_roles:
        if role in contracts_dict:
            contract = contracts_dict[role]
            contracts.append(contract)
            symbol_to_role[contract.ticker] = role
            if verbose:
                print(f"  ✓ {role}: {contract.ticker} (экспирация {contract.expiry})")
        else:
            if verbose:
                print(f"  ✗ {role}: не найден")

    if not contracts:
        print("✗ Нет доступных контрактов для загрузки")
        return {}

    # Загрузка данных
    pipeline = BacktestDataPipeline(moex_client, 252, cache_dir=cache_dir)
    portfolio_data = pipeline.load_portfolio_history(
        contracts,
        duration_days=duration_days,
        bar_size="1 hour",
        verbose=verbose,
        contracts_roles=symbol_to_role
    )

    # Итоговая сводка
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"ИТОГОВАЯ СВОДКА ЗАГРУЗКИ")
        print(f"{'=' * 80}")

        if portfolio_data:
            print(f"✓ Успешно загружено: {len(portfolio_data)} контрактов")
            print(f"\nДетали загрузки:")

            total_bars = 0
            for ticker, df in sorted(portfolio_data.items()):
                role = symbol_to_role.get(ticker, "?")
                bars = len(df)
                total_bars += bars
                date_range = f"{df['date'].min()} → {df['date'].max()}"
                print(f"  {role:4s} ({ticker:6s}): {bars:4d} баров, {date_range}")

            avg_bars = total_bars / len(portfolio_data)
            print(f"\n  Всего баров: {total_bars}")
            print(f"  Среднее на контракт: {avg_bars:.0f}")
            print(f"  Кэш директория: {cache_dir}")
        else:
            print("✗ Не удалось загрузить данные")

        print(f"{'=' * 80}\n")

    return portfolio_data


def auto_preload():
    """
    Автоматическая предзагрузка данных с явной индикацией.
    Вызывайте вручную после загрузки cell_02_moex_contracts.py:

    %run cell_02_moex_contracts.py
    %run cell_03_data_estimators.py
    %run cell_07_2_09a_data_pipeline.py
    auto_preload()
    """
    import sys
    from datetime import date

    # Получаем глобальное пространство имен вызывающего модуля (Jupyter notebook)
    caller_globals = sys._getframe(1).f_globals

    # Проверяем наличие зависимостей
    missing = []
    if 'MoexClient' not in caller_globals:
        missing.append('MoexClient')
    if 'MoexConfig' not in caller_globals:
        missing.append('MoexConfig')
    if 'identify_contracts' not in caller_globals:
        missing.append('identify_contracts')

    if missing:
        print(f"\n✗ Не хватает зависимостей: {', '.join(missing)}")
        print(f"\nСначала запустите:")
        print(f"  %run cell_02_moex_contracts.py")
        print(f"  %run cell_03_data_estimators.py")
        return None

    print(f"\n{'=' * 80}")
    print(f"→ ЗАПУСК АВТОМАТИЧЕСКОЙ ПРЕДЗАГРУЗКИ ДАННЫХ")
    print(f"{'=' * 80}\n")

    try:
        MoexClient = caller_globals['MoexClient']
        MoexConfig = caller_globals['MoexConfig']
        identify_contracts_fn = caller_globals['identify_contracts']

        moex_config = MoexConfig()
        moex_client = MoexClient(moex_config)

        # Идентификация контрактов
        print("[Шаг 1/3] Идентификация контрактов...")
        contracts_dict = identify_contracts_fn(moex_client, moex_config, date.today())
        print(f"✓ Найдено контрактов: {len(contracts_dict)}\n")

        # Предзагрузка данных
        print("[Шаг 2/3] Загрузка исторических данных...")
        portfolio_data = preload_contracts_data(
            moex_client,
            moex_config,
            contract_roles=["F0", "F-1", "F-2", "F-3", "F-4", "F-5", "F-6", "F-7", "F-8", "F-9", "F-10"],
            duration_days=365,
            cache_dir="./data/har_tuning_cache",
            verbose=True
        )

        print("\n[Шаг 3/3] Предзагрузка завершена")
        print(f"{'=' * 80}\n")

        return portfolio_data

    except Exception as e:
        import traceback
        print(f"\n✗ ОШИБКА ПРЕДЗАГРУЗКИ: {e}")
        print(f"\nПолный traceback:")
        traceback.print_exc()
        print(f"\n{'=' * 80}\n")
        return None


if __name__ == "__main__":
    print("✓ Cell 7.2.9a: HAR Backtest Data Pipeline загружен успешно")
    print(f"  - Класс: BacktestDataPipeline")
    print(f"  - Методы: load_contract_history, load_portfolio_history, validate_data_quality")
    print(f"  - Функция: preload_contracts_data() - предзагрузка данных")
    print(f"  - Функция: auto_preload() - автоматическая предзагрузка с индикацией")
    print(f"\n→ Для загрузки данных выполните: auto_preload()")
    print(f"  (после загрузки cell_02_moex_contracts.py и cell_03_data_estimators.py)")

