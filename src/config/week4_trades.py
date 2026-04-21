from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WeakWindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    start: date
    end: date

    @model_validator(mode="after")
    def _check_date_order(self) -> "WeakWindowConfig":
        if self.end < self.start:
            raise ValueError(f"weak window {self.name} end must be on or after start")
        return self


class PilotSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasons: list[str]
    earnings_window_days: int = Field(ge=0)
    gap_threshold_pct: float = Field(gt=0)
    weak_window_top_n: int = Field(gt=0)
    weak_windows: list[WeakWindowConfig]

    @field_validator("reasons")
    @classmethod
    def _validate_reasons(cls, value: list[str]) -> list[str]:
        allowed = {"earnings", "gap", "weak_window"}
        normalized = [str(item).strip().lower() for item in value]
        unknown = sorted(set(normalized) - allowed)
        if unknown:
            raise ValueError(f"unsupported pilot reasons: {unknown}")
        return normalized


class Stage2SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_n_liquidity: int = Field(gt=0)
    top_liquidity_lookback_days: int = Field(gt=0)


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pilot: PilotSamplingConfig
    stage2: Stage2SamplingConfig


class PolygonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entitlement_delay_minutes: int = Field(ge=0)
    rest_max_pages_per_request: int = Field(gt=0)
    rest_page_size: int = Field(gt=0)
    rest_min_interval_seconds: float = Field(gt=0)
    retry_max: int = Field(ge=0)


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_daily_api_calls: int = Field(gt=0)
    max_storage_gb: float = Field(gt=0)
    max_rows_per_ticker_day: int = Field(gt=0)
    expected_pilot_ticker_days: int = Field(gt=0)


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_threshold_dollars: float = Field(gt=0)
    size_threshold_min_cap_dollars: float = Field(ge=0)
    condition_allow_list: list[int]
    trf_exchange_codes: list[int]
    late_day_window_et: tuple[str, str]
    offhours_window_et_pre: tuple[str, str]
    offhours_window_et_post: tuple[str, str]


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage_min_pct: float = Field(gt=0, le=100)
    feature_missing_max_pct: float = Field(ge=0, le=100)
    feature_outlier_max_pct: float = Field(ge=0, le=100)
    min_passing_features: int = Field(gt=0)
    ic_threshold: float = Field(gt=0)
    abs_tstat_threshold: float = Field(gt=0)
    sign_consistent_windows_min: int = Field(gt=0)


class Week4TradesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    stage: str
    sampling: SamplingConfig
    polygon: PolygonConfig
    budgets: BudgetConfig
    features: FeatureConfig
    gate: GateConfig

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"pilot", "stage2"}:
            raise ValueError("stage must be 'pilot' or 'stage2'")
        return normalized
