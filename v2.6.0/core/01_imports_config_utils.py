"""
01_imports_config_utils.py
Cell ID: EHJdFwWrZBWa
Exported: 2026-04-16T10:12:23.218529
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 1/5
Импорты, конфигурация, константы, утилиты
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, time as dt_time, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("seaborn-v0_8-whitegrid")

# ════════════════════════════════════════════════════════════════
# VERSIONING
# ════════════════════════════════════════════════════════════════
SCHEMA_VERSION  = "2.6.0"
ENGINE_VERSION  = "dual_vol_engine_0.6.0"

# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging. Returns root pipeline logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("si_vol_pipeline")

log = setup_logging()

# ════════════════════════════════════════════════════════════════
# DOMAIN EXCEPTIONS
# ════════════════════════════════════════════════════════════════
class AnalyticsError(Exception):
    """Base pipeline exception."""

class DataValidationError(AnalyticsError):
    """Raised on invalid market data."""

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════
MONTH_CODES: dict[str, int] = {"H": 3, "M": 6, "U": 9, "Z": 12}
CODE_BY_MONTH: dict[int, str] = {v: k for k, v in MONTH_CODES.items()}
Q_MONTHS: list[int] = sorted(MONTH_CODES.values())
WINDOWS: tuple[int, ...] = (5, 20, 60, 250)
SEMIVAR_WINDOWS: tuple[int, ...] = (5, 20)
GAP_WINDOWS: tuple[int, ...] = (5, 20, 60)


@dataclass(frozen=True)
class MoexConfig:
    """MOEX ISS API и параметры сессии."""
    base_url: str = "https://iss.moex.com/iss"
    board: str = "RFUD"
    retry: int = 3
    timeout: int = 30
    page_size: int = 500
    session_end_minutes: int = 18 * 60 + 45   # 18:45
    session_duration_minutes: int = 8 * 60 + 45  # 8ч 45мин


@dataclass(frozen=True)
class VolConfig:
    """Параметры расчёта волатильности."""
    tdy: int = 252
    history_days: int = 600
    active_phase_days: int = 365
    ewma_spans: tuple[int, ...] = (1, 3, 5, 10, 20)
    lookback_candles: int = 5
    dte_tolerance: int = 5


@dataclass(frozen=True)
class ThresholdConfig:
    """Единые числовые пороги классификации."""
    semivar_absent: float = 1.1
    semivar_mild: float = 1.3
    semivar_moderate: float = 1.7
    vov_stable: float = 0.25
    vov_elevated: float = 0.40
    vov_entry_delay: float = 0.35
    kurtosis_platykurtic: float = -1.0
    kurtosis_leptokurtic: float = 1.0
    skew_left: float = -0.5
    skew_right: float = 0.5
    basis_vol_low: float = 0.015
    basis_vol_elevated: float = 0.025
    gap_freq_low: float = 2.0
    gap_freq_elevated: float = 7.0
    premium_sufficient_pp: float = 0.015
    vol_ratio_z_risk: float = -1.5
    roc_fast_cooling: float = -20.0
    roc_accelerating: float = 5.0
    yz_ratio_compression: float = 0.75
    yz_ratio_spike: float = 1.1
    forecast_identity_tol: float = 1e-7


@dataclass
class PipelineConfig:
    """Общая конфигурация пайплайна."""
    moex: MoexConfig = field(default_factory=MoexConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    thr: ThresholdConfig = field(default_factory=ThresholdConfig)
    out_dir: Path = field(default_factory=lambda: Path("/content/si_vol_analytics"))
    gap_sigma_bins: tuple = (0.0, 0.25, 0.5, 0.75, 1.0, float("inf"))
    gap_sigma_labels: tuple = ("0.0-0.25σ", "0.25-0.5σ", "0.5-0.75σ", "0.75-1.0σ", "≥1.0σ")

    @property
    def chart_dir(self) -> Path:
        return self.out_dir / "charts"

    def makedirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ContractMeta:
    """Метаданные контракта."""
    ticker: str
    expiry: date

    @property
    def symbol(self) -> str:
        """Алиас для ticker (обратная совместимость с legacy кодом)."""
        return self.ticker


# ════════════════════════════════════════════════════════════════
# PRECISION HELPERS
# ════════════════════════════════════════════════════════════════
def _finite(v: Any) -> bool:
    return isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))

def _round(v: Any, d: int) -> Optional[float]:
    return round(float(v), d) if _finite(v) else None

def r_vol(v: Any)   -> Optional[float]: return _round(v, 6)
def r_ratio(v: Any) -> Optional[float]: return _round(v, 4)
def r_z(v: Any)     -> Optional[float]: return _round(v, 2)
def r_pct(v: Any)   -> Optional[float]: return _round(v, 1)
def r_kurt(v: Any)  -> Optional[float]: return _round(v, 3)

def null_val(reason: str) -> dict[str, Any]:
    """Стандартный null-маркер."""
    return {"value": None, "available": False, "reason": reason}

def is_null(v: Any) -> bool:
    """Проверяет является ли значение null_val или None."""
    if v is None:
        return True
    if isinstance(v, dict) and not v.get("available", True):
        return True
    return False

def vscalar(d: dict, key: str) -> Optional[float]:
    """Извлекает float-скаляр из dict (поддерживает plain float и null_val)."""
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        inner = v.get("value")
        return float(inner) if _finite(inner) else None
    return float(v) if _finite(v) else None

print(f"Cell 1/5: Config & Utils загружены | v{SCHEMA_VERSION}")
