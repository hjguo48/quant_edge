from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import yaml

import scripts.build_trades_sample_universe as builder
from src.data.db.models import TradesSamplingState
from src.data.event_calendar import SamplingEvent


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
                "weak_windows": [{"name": "W5", "start": "2022-10-01", "end": "2023-03-31"}],
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


def _session_factory():
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    TradesSamplingState.__table__.create(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _events() -> list[SamplingEvent]:
    return [
        SamplingEvent(ticker="AAPL", trading_date=date(2026, 1, 5), reason="earnings"),
        SamplingEvent(ticker="AAPL", trading_date=date(2026, 1, 5), reason="gap"),
        SamplingEvent(ticker="MSFT", trading_date=date(2026, 1, 6), reason="weak_window"),
    ]


def _patch_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    passed: bool,
) -> None:
    def fake_run_estimator(**kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")
        return {
            "pass": passed,
            "failure_reasons": [] if passed else ["storage_budget_exceeded"],
        }

    monkeypatch.setattr(builder.preflight_trades_estimator, "run_estimator", fake_run_estimator)


def test_build_sampling_universe_writes_parquet_and_pending_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight(monkeypatch, passed=True)
    monkeypatch.setattr(builder, "build_sampling_plan", lambda **kwargs: _events())
    session_factory = _session_factory()
    output = tmp_path / "plan.parquet"

    result = builder.build_trades_sample_universe(
        config_path=_write_config(tmp_path),
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 6),
        output_path=output,
        preflight_output_path=tmp_path / "preflight.json",
        session_factory=session_factory,
    )

    assert result is not None
    assert result["summary"]["total_rows"] == 3
    assert result["summary"]["reason_counts"] == {"earnings": 1, "gap": 1, "weak_window": 1}
    assert result["summary"]["state_rows_inserted"] == 3

    frame = pd.read_parquet(output)
    assert frame[["ticker", "reason"]].to_dict("records") == [
        {"ticker": "AAPL", "reason": "earnings"},
        {"ticker": "AAPL", "reason": "gap"},
        {"ticker": "MSFT", "reason": "weak_window"},
    ]

    with session_factory() as session:
        rows = session.query(TradesSamplingState).order_by(
            TradesSamplingState.ticker,
            TradesSamplingState.sampled_reason,
        ).all()
    assert [(row.ticker, row.sampled_reason, row.status, row.rows_ingested) for row in rows] == [
        ("AAPL", "earnings", "pending", None),
        ("AAPL", "gap", "pending", None),
        ("MSFT", "weak_window", "pending", None),
    ]


def test_preflight_fail_exits_one_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight(monkeypatch, passed=False)
    called = False

    def fake_build_sampling_plan(**kwargs):
        nonlocal called
        called = True
        return _events()

    monkeypatch.setattr(builder, "build_sampling_plan", fake_build_sampling_plan)

    exit_code = builder.main(
        [
            "--config",
            str(_write_config(tmp_path)),
            "--start-date",
            "2026-01-05",
            "--end-date",
            "2026-01-06",
            "--output",
            str(tmp_path / "plan.parquet"),
            "--preflight-output",
            str(tmp_path / "preflight.json"),
        ],
    )

    assert exit_code == 1
    assert called is False
    assert not (tmp_path / "plan.parquet").exists()


def test_preflight_fail_with_force_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight(monkeypatch, passed=False)
    monkeypatch.setattr(builder, "build_sampling_plan", lambda **kwargs: _events())
    session_factory = _session_factory()

    result = builder.build_trades_sample_universe(
        config_path=_write_config(tmp_path),
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 6),
        output_path=tmp_path / "plan.parquet",
        preflight_output_path=tmp_path / "preflight.json",
        force=True,
        session_factory=session_factory,
    )

    assert result is not None
    assert result["preflight"]["pass"] is False
    assert result["summary"]["state_rows_inserted"] == 3


def test_dry_run_writes_parquet_without_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight(monkeypatch, passed=True)
    monkeypatch.setattr(builder, "build_sampling_plan", lambda **kwargs: _events())
    session_factory = _session_factory()
    output = tmp_path / "plan.parquet"

    result = builder.build_trades_sample_universe(
        config_path=_write_config(tmp_path),
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 6),
        output_path=output,
        preflight_output_path=tmp_path / "preflight.json",
        dry_run=True,
        session_factory=session_factory,
    )

    assert result is not None
    assert output.exists()
    assert result["summary"]["state_rows_inserted"] == 0
    with session_factory() as session:
        assert session.query(TradesSamplingState).count() == 0


def test_existing_state_row_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight(monkeypatch, passed=True)
    monkeypatch.setattr(
        builder,
        "build_sampling_plan",
        lambda **kwargs: [SamplingEvent(ticker="AAPL", trading_date=date(2026, 1, 5), reason="earnings")],
    )
    session_factory = _session_factory()
    with session_factory() as session:
        session.add(
            TradesSamplingState(
                ticker="AAPL",
                trading_date=date(2026, 1, 5),
                sampled_reason="earnings",
                status="completed",
                rows_ingested=123,
            ),
        )
        session.commit()

    result = builder.build_trades_sample_universe(
        config_path=_write_config(tmp_path),
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 5),
        output_path=tmp_path / "plan.parquet",
        preflight_output_path=tmp_path / "preflight.json",
        session_factory=session_factory,
    )

    assert result is not None
    assert result["summary"]["state_rows_inserted"] == 0
    with session_factory() as session:
        row = session.get(TradesSamplingState, ("AAPL", date(2026, 1, 5), "earnings"))
    assert row is not None
    assert row.status == "completed"
    assert row.rows_ingested == 123
