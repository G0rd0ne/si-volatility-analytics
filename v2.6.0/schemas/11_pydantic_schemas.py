"""
11_pydantic_schemas.py
Cell ID: 45178ceb
Exported: 2026-04-16T10:12:23.219194
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, RootModel, ConfigDict

# ════════════════════════════════════════════════════════════════
# NULL-SAFE WRAPPERS
# ════════════════════════════════════════════════════════════════

class NullableValue(BaseModel):
    """Standard wrapper for missing/stale metrics."""
    value: Optional[float] = None
    available: bool = False
    reason: Optional[str] = None

MetricValue = Union[float, NullableValue, None]

# ════════════════════════════════════════════════════════════════
# SUB-MODELS
# ════════════════════════════════════════════════════════════════

class ContractMeta(BaseModel):
    ticker: str
    expiry: date
    days_to_expiry: Optional[int] = None
    days_since_expiry: Optional[int] = None

class VolatilityMetrics(BaseModel):
    # YZ and CC for various windows
    YZ_5d: MetricValue = None
    YZ_20d: MetricValue = None
    YZ_60d: MetricValue = None
    YZ_250d: MetricValue = None
    CC_5d: MetricValue = None
    CC_20d: MetricValue = None
    # Risk metrics
    skew_5d: MetricValue = None
    kurtosis_5d: MetricValue = None
    vol_of_vol_5d: MetricValue = None
    semivar_down_5d: MetricValue = None
    semivar_up_5d: MetricValue = None

class GapStats(BaseModel):
    frequency_pct: float
    amplitude_avg: float
    amplitude_max: float
    n_gaps: int
    gap_quantiles: Dict[str, float]
    available: bool = True

class HARForecast(BaseModel):
    forecast_RV: Optional[float]
    method: str = "HAR-RV"
    horizon_days: float
    available: bool

class DataWarning(BaseModel):
    code: str
    severity: str
    message: str
    affected_strategies: List[str]

# ════════════════════════════════════════════════════════════════
# MAIN REPORT SCHEMA v2.6.0
# ════════════════════════════════════════════════════════════════

class SiVolatilityReport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    meta: Dict[str, Any] = Field(..., description="Calculation metadata and calculation_id")

    contracts: Dict[str, ContractMeta] = Field(..., description="F0, F1, F2 details")

    volatility: Dict[str, VolatilityMetrics] = Field(..., description="Per-contract vol metrics")

    gap_analysis: Dict[str, Dict[str, Union[GapStats, NullableValue]]] = Field(
        ..., description="Nested contract -> window gaps"
    )

    cross_contract_metrics: Dict[str, Any] = Field(..., description="Spreads, basis, stress")

    rv_forecast_session_aligned: Dict[str, Any] = Field(..., description="HAR/EWMA predictions")

    state_labels: Dict[str, str] = Field(..., description="Regime classifications (e.g. vol_trend)")

    data_quality: Dict[str, Any] = Field(..., description="Warnings and aligned bar counts")

def validate_pipeline_output(json_data: Dict[str, Any]) -> SiVolatilityReport:
    """Entry point for strict validation of the generated report."""
    return SiVolatilityReport.model_validate(json_data)

print("✓ Pydantic validation schema v2.6.0 generated")