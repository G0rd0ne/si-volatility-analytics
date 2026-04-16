"""
07_2_1_contract_id.py
Cell ID: 8vPWTu_7YV9k
Exported: 2026-04-16T10:12:23.218749
"""

"""
Si Volatility Analytics v2.6.0 - Cell 7.2.1
Contract Identification Module

Модуль для идентификации исторических контрактов Si
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

log = logging.getLogger("har_contracts")


def identify_contracts_with_history(
    contracts: dict,
    backtest_contracts: list[str]
) -> dict[str, Any]:
    """
    Идентифицирует контракты для бэктеста и возвращает их метаданные.

    Args:
        contracts: dict[str, ContractMeta] из contract identifier
        backtest_contracts: список ролей для бэктеста, напр. ["F0", "F1", "F2"]

    Returns:
        dict с метаданными контрактов
    """
    log.info(f"\n=== Contract Identification ===")
    log.info(f"Requested contracts: {backtest_contracts}")

    available = {}
    missing = []

    for role in backtest_contracts:
        if role in contracts:
            meta = contracts[role]
            available[role] = {
                "ticker": meta.ticker,
                "expiry": meta.expiry,
                "role": role
            }
            log.info(f"✓ {role}: {meta.ticker} (expires {meta.expiry})")
        else:
            missing.append(role)
            log.warning(f"✗ {role}: not found in contract set")

    result = {
        "available": available,
        "missing": missing,
        "total_available": len(available),
        "total_missing": len(missing)
    }

    log.info(f"Summary: {len(available)} available, {len(missing)} missing")

    return result


if __name__ == "__main__":
    print("✓ Cell 7.2.1: Contract Identification Module загружен успешно")
    print(f"  - Функции: identify_contracts_with_history")
    print(f"  - Зависимости: logging, typing ✓")
