"""
07_2_2_hist_data_loader.py
Cell ID: Wk3l9M3XYc_M
Exported: 2026-04-16T10:12:23.218754
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.2.2
Historical Data Loader Module

Модуль для загрузки исторических OHLCV данных
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

log = logging.getLogger("har_data_loader")


class HistoricalDataLoader:
    """Загрузчик исторических OHLCV данных для бэктеста."""

    def __init__(self, cfg, bt_cfg):
        """
        Args:
            cfg: PipelineConfig из cell_01
            bt_cfg: BacktestConfig из cell_07_1
        """
        self.cfg = cfg
        self.bt_cfg = bt_cfg
        self.client = None

    def set_moex_client(self, client):
        """Устанавливает MOEX client."""
        self.client = client

    def load_historical_ohlcv(
        self,
        contracts: dict,
        today: date,
        moex_load_candles_fn,
        validate_ohlcv_fn
    ) -> dict[str, pd.DataFrame]:
        """
        Загружает OHLCV для всех expired contracts.

        Args:
            contracts: dict[str, ContractMeta]
            today: текущая дата
            moex_load_candles_fn: функция moex_load_candles из cell_03
            validate_ohlcv_fn: функция validate_ohlcv из cell_03

        Returns:
            dict[role, DataFrame] с OHLCV данными
        """
        log.info(f"\n=== Loading Historical OHLCV ===")
        log.info(f"Today: {today}")
        log.info(f"Contracts to load: {self.bt_cfg.backtest_contracts}")

        ohlcv_data = {}
        stats = {
            "loaded": [],
            "failed": [],
            "total_bars": 0
        }

        for role in self.bt_cfg.backtest_contracts:
            if role not in contracts:
                log.warning(f"Contract {role} not identified, skipping")
                stats["failed"].append((role, "not_identified"))
                continue

            meta = contracts[role]
            date_till = meta.expiry.strftime("%Y-%m-%d")
            date_from = (meta.expiry - timedelta(days=self.cfg.vol.active_phase_days)).strftime("%Y-%m-%d")

            log.info(f"Loading {role} ({meta.ticker}): {date_from} → {date_till}")

            try:
                df = moex_load_candles_fn(
                    self.client,
                    self.cfg.moex,
                    meta.ticker,
                    date_from,
                    date_till
                )

                if len(df) == 0:
                    log.warning(f"{role} ({meta.ticker}): no data available")
                    stats["failed"].append((role, "no_data"))
                    continue

                df = validate_ohlcv_fn(df, meta.ticker)
                ohlcv_data[role] = df
                stats["loaded"].append(role)
                stats["total_bars"] += len(df)

                log.info(f"✓ {role}: {len(df)} bars loaded")

            except Exception as e:
                log.error(f"✗ {role} ({meta.ticker}): loading failed - {e}")
                stats["failed"].append((role, str(e)))

        log.info(f"\n=== Load Summary ===")
        log.info(f"Successfully loaded: {len(stats['loaded'])} contracts")
        log.info(f"Failed: {len(stats['failed'])} contracts")
        log.info(f"Total bars: {stats['total_bars']}")

        if stats["failed"]:
            log.warning(f"Failed contracts: {stats['failed']}")

        return ohlcv_data


def aggregate_ohlcv_data(ohlcv_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Агрегирует OHLCV данные всех контрактов в единую временную серию.

    Args:
        ohlcv_data: dict[role, DataFrame]

    Returns:
        Агрегированный DataFrame, отсортированный по дате
    """
    log.info(f"\n=== Aggregating OHLCV Data ===")
    log.info(f"Contracts to aggregate: {list(ohlcv_data.keys())}")

    if len(ohlcv_data) == 0:
        log.error("No data to aggregate")
        return pd.DataFrame()

    all_bars = pd.concat(ohlcv_data.values(), ignore_index=True)
    all_bars = all_bars.sort_values("date").reset_index(drop=True)

    date_range = {
        "from": all_bars["date"].min(),
        "to": all_bars["date"].max()
    }

    log.info(f"Total bars aggregated: {len(all_bars)}")
    log.info(f"Date range: {date_range['from']} → {date_range['to']}")
    log.info(f"Sample data:\n{all_bars.head(3)}")

    return all_bars


if __name__ == "__main__":
    print("✓ Cell 7.2.2: Historical Data Loader Module загружен успешно")
    print(f"  - Классы: HistoricalDataLoader")
    print(f"  - Функции: aggregate_ohlcv_data")

    # Проверка зависимостей
    try:
        import pandas
        print(f"  - pandas {pandas.__version__} ✓")
    except ImportError as e:
        print(f"  - pandas ✗ ({e})")
