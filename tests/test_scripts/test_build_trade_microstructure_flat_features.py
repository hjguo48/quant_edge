from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import scripts.build_trade_microstructure_features as task8_builder
import scripts.build_trade_microstructure_flat_features as builder


def _config_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "stage": "pilot",
        "sampling": {
            "pilot": {
                "reasons": ["earnings", "gap", "weak_window"],
                "earnings_window_days": 3,
                "gap_threshold_pct": 0.03,
                "weak_window_top_n": 100,
                "weak_windows": [{"name": "W5", "start": "2021-09-03", "end": "2022-02-25"}],
            },
            "stage2": {"top_n_liquidity": 200, "top_liquidity_lookback_days": 20},
        },
        "polygon": {
            "entitlement_delay_minutes": 15,
            "rest_max_pages_per_request": 50,
            "rest_page_size": 50000,
            "rest_min_interval_seconds": 0.05,
            "retry_max": 3,
        },
        "budgets": {
            "max_daily_api_calls": 50000,
            "max_storage_gb": 200,
            "max_rows_per_ticker_day": 2000000,
            "expected_pilot_ticker_days": 30000,
        },
        "features": {
            "size_threshold_dollars": 1000000,
            "size_threshold_min_cap_dollars": 250000,
            "condition_allow_list": [],
            "trf_exchange_codes": [4, 202],
            "late_day_window_et": ["15:00", "16:00"],
            "offhours_window_et_pre": ["04:00", "09:30"],
            "offhours_window_et_post": ["16:00", "20:00"],
        },
        "gate": {
            "coverage_min_pct": 95.0,
            "feature_missing_max_pct": 30.0,
            "feature_outlier_max_pct": 5.0,
            "min_passing_features": 2,
            "ic_threshold": 0.015,
            "abs_tstat_threshold": 2.0,
            "sign_consistent_windows_min": 7,
        },
    }


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "week4.yaml"
    path.write_text(yaml.safe_dump(_config_payload()))
    return path


def _load_config(tmp_path: Path):
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)
    return config, config_hash


def _et(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw, tz="America/New_York").tz_convert("UTC")


def _sample_trades(tickers: list[str], trading_date: date) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        rows.extend(
            [
                {
                    "ticker": ticker,
                    "trading_date": trading_date,
                    "sip_timestamp": _et(f"{trading_date} 10:00"),
                    "price": 100.0,
                    "size": 100.0,
                    "exchange": 1,
                    "trf_id": None,
                    "trf_timestamp": None,
                    "conditions": [0],
                },
                {
                    "ticker": ticker,
                    "trading_date": trading_date,
                    "sip_timestamp": _et(f"{trading_date} 10:01"),
                    "price": 101.0,
                    "size": 100.0,
                    "exchange": 4,
                    "trf_id": None,
                    "trf_timestamp": None,
                    "conditions": [0],
                },
                {
                    "ticker": ticker,
                    "trading_date": trading_date,
                    "sip_timestamp": _et(f"{trading_date} 19:00"),
                    "price": 99.0,
                    "size": 100.0,
                    "exchange": 1,
                    "trf_id": None,
                    "trf_timestamp": None,
                    "conditions": [0],
                },
            ],
        )
    return pd.DataFrame(rows)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[date, tuple[str, ...]]] = []

    def load_day_for_tickers(self, trading_date: date, tickers: list[str]) -> pd.DataFrame:
        self.calls.append((trading_date, tuple(tickers)))
        return _sample_trades(list(tickers), trading_date)


def test_build_flat_features_happy_path_and_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_hash = _load_config(tmp_path)
    dates = [date(2022, 1, 3), date(2022, 1, 4)]
    monkeypatch.setattr(builder, "session_dates", lambda start, end: dates)
    client = _FakeClient()
    output = tmp_path / "flat_features.parquet"

    summary = builder.build_flat_features(
        config=config,
        config_hash=config_hash,
        start_date=dates[0],
        end_date=dates[-1],
        top_n=2,
        output_path=output,
        client=client,
        top_liquidity_fn=lambda as_of_date, *, top_n, session_factory=None: ["AAPL", "MSFT"][:top_n],
    )

    assert summary["rows_computed"] == 4
    assert client.calls == [(dates[0], ("AAPL", "MSFT")), (dates[1], ("AAPL", "MSFT"))]
    frame = pd.read_parquet(output)
    assert list(frame.columns) == task8_builder.OUTPUT_COLUMNS == builder.OUTPUT_COLUMNS
    assert len(frame) == 4
    assert frame["trade_imbalance_proxy"].notna().all()
    assert frame["off_exchange_volume_ratio"].notna().all()


def test_build_flat_features_resume_skips_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_hash = _load_config(tmp_path)
    trading_date = date(2022, 1, 3)
    monkeypatch.setattr(builder, "session_dates", lambda start, end: [trading_date])
    output = tmp_path / "flat_features.parquet"
    existing = pd.DataFrame(
        [
            {
                "event_date": trading_date,
                "ticker": "AAPL",
                "knowledge_time_regular": pd.Timestamp("2022-01-03 21:15", tz="UTC"),
                "knowledge_time_offhours": pd.Timestamp("2022-01-04 01:15", tz="UTC"),
                "trade_imbalance_proxy": 0.1,
                "large_trade_ratio": 0.0,
                "late_day_aggressiveness": 1.0,
                "offhours_trade_ratio": 0.0,
                "off_exchange_volume_ratio": 0.0,
                "run_config_hash": config_hash,
            },
        ],
    )
    existing.to_parquet(output, index=False)
    client = _FakeClient()

    summary = builder.build_flat_features(
        config=config,
        config_hash=config_hash,
        start_date=trading_date,
        end_date=trading_date,
        top_n=1,
        output_path=output,
        client=client,
        top_liquidity_fn=lambda as_of_date, *, top_n, session_factory=None: ["AAPL"],
    )

    assert summary["rows_skipped_resume"] == 1
    assert summary["rows_computed"] == 0
    assert client.calls == []


def test_build_flat_features_respects_top_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_hash = _load_config(tmp_path)
    trading_date = date(2022, 1, 3)
    monkeypatch.setattr(builder, "session_dates", lambda start, end: [trading_date])
    client = _FakeClient()
    observed_top_n: list[int] = []

    def top_fn(as_of_date, *, top_n, session_factory=None):
        observed_top_n.append(top_n)
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    summary = builder.build_flat_features(
        config=config,
        config_hash=config_hash,
        start_date=trading_date,
        end_date=trading_date,
        top_n=2,
        output_path=tmp_path / "flat_features.parquet",
        client=client,
        top_liquidity_fn=top_fn,
    )

    assert observed_top_n == [2]
    assert client.calls == [(trading_date, ("AAPL", "MSFT"))]
    assert summary["rows_computed"] == 2
