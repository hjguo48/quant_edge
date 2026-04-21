from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import scripts.build_trade_microstructure_features as builder


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


def _et(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw, tz="America/New_York").tz_convert("UTC")


def _sample_trades() -> pd.DataFrame:
    """Minimal 4-trade day generating non-trivial features (tick imbalance > 0)."""
    return pd.DataFrame(
        [
            {
                "sip_timestamp": _et("2026-01-05 10:00"),
                "price": 100.0,
                "size": 100.0,
                "exchange": 1,
                "trf_id": None,
                "trf_timestamp": None,
                "conditions": [0],
            },
            {
                "sip_timestamp": _et("2026-01-05 10:01"),
                "price": 101.0,
                "size": 100.0,
                "exchange": 1,
                "trf_id": None,
                "trf_timestamp": None,
                "conditions": [0],
            },
            {
                "sip_timestamp": _et("2026-01-05 15:05"),
                "price": 102.0,
                "size": 100.0,
                "exchange": 4,
                "trf_id": None,
                "trf_timestamp": None,
                "conditions": [0],
            },
            {
                "sip_timestamp": _et("2026-01-05 08:00"),
                "price": 99.0,
                "size": 100.0,
                "exchange": 1,
                "trf_id": None,
                "trf_timestamp": None,
                "conditions": [0],
            },
        ],
    )


def _make_session_factory(
    groups: list[tuple[str, date]],
    trades_by_key: dict[tuple[str, date], pd.DataFrame],
):
    """Build a factory that returns a context-manager session whose helpers yield the fixtures."""

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def factory() -> _FakeSession:
        return _FakeSession()

    def fake_iter(session, start, end):
        for ticker, trading_date in groups:
            if start <= trading_date <= end:
                yield ticker, trading_date

    def fake_load(session, ticker, trading_date):
        key = (ticker, trading_date)
        return trades_by_key.get(key, pd.DataFrame())

    return factory, fake_iter, fake_load


@pytest.fixture
def monkeypatched_loaders(monkeypatch):
    """Helper fixture: returns a function to install iter + load mocks."""

    def install(groups, trades_by_key):
        factory, fake_iter, fake_load = _make_session_factory(groups, trades_by_key)
        monkeypatch.setattr(builder, "iter_ticker_dates", fake_iter)
        monkeypatch.setattr(builder, "load_trades_for_group", fake_load)
        return factory

    return install


def test_build_features_happy_path(tmp_path: Path, monkeypatched_loaders) -> None:
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)

    groups = [("AAPL", date(2026, 1, 5)), ("MSFT", date(2026, 1, 5))]
    factory = monkeypatched_loaders(groups, {k: _sample_trades() for k in groups})
    output = tmp_path / "features.parquet"

    summary = builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory,
        resume=True,
    )

    assert summary["groups_processed"] == 2
    assert summary["groups_skipped_resume"] == 0
    assert summary["rows_computed"] == 2
    assert summary["rows_total"] == 2

    frame = pd.read_parquet(output)
    assert list(frame.columns) == builder.OUTPUT_COLUMNS
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}
    assert all(frame["run_config_hash"] == config_hash)
    # knowledge_time_regular (T 16:15 ET = 21:15 UTC)
    assert frame["knowledge_time_regular"].iloc[0].hour == 21
    assert frame["knowledge_time_regular"].iloc[0].minute == 15
    # knowledge_time_offhours (T 20:15 ET = next-day 01:15 UTC)
    assert frame["knowledge_time_offhours"].iloc[0].hour == 1
    # trade_imbalance_proxy on up-tick day → positive
    assert frame["trade_imbalance_proxy"].iloc[0] > 0


def test_build_features_resume_skips_completed(tmp_path: Path, monkeypatched_loaders) -> None:
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)
    output = tmp_path / "features.parquet"

    # First pass: 1 group
    groups_v1 = [("AAPL", date(2026, 1, 5))]
    factory1 = monkeypatched_loaders(groups_v1, {k: _sample_trades() for k in groups_v1})
    builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory1,
        resume=True,
    )

    # Second pass: add 1 new group; existing AAPL must be skipped
    groups_v2 = [("AAPL", date(2026, 1, 5)), ("MSFT", date(2026, 1, 5))]
    factory2 = monkeypatched_loaders(groups_v2, {k: _sample_trades() for k in groups_v2})
    summary = builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory2,
        resume=True,
    )

    assert summary["groups_skipped_resume"] == 1  # AAPL skipped
    assert summary["rows_computed"] == 1  # only MSFT computed
    assert summary["rows_total"] == 2  # AAPL (from existing) + MSFT (new)

    frame = pd.read_parquet(output)
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}


def test_build_features_config_hash_mismatch_triggers_full_recompute(
    tmp_path: Path, monkeypatched_loaders
) -> None:
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)
    output = tmp_path / "features.parquet"

    # Seed parquet with OLD hash (different)
    pre_existing = pd.DataFrame(
        [
            {
                "event_date": date(2026, 1, 5),
                "ticker": "AAPL",
                "knowledge_time_regular": pd.Timestamp("2026-01-05 21:15", tz="UTC"),
                "knowledge_time_offhours": pd.Timestamp("2026-01-06 01:15", tz="UTC"),
                "trade_imbalance_proxy": 0.5,
                "large_trade_ratio": 0.1,
                "late_day_aggressiveness": 1.0,
                "offhours_trade_ratio": 0.2,
                "off_exchange_volume_ratio": 0.3,
                "run_config_hash": "old_hash_123",
            },
        ],
        columns=builder.OUTPUT_COLUMNS,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pre_existing.to_parquet(output, index=False)

    # New run with NEW hash should ignore existing (recompute)
    groups = [("AAPL", date(2026, 1, 5))]
    factory = monkeypatched_loaders(groups, {k: _sample_trades() for k in groups})
    summary = builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory,
        resume=True,
    )

    assert summary["groups_skipped_resume"] == 0  # hash mismatch → nothing skipped
    assert summary["rows_computed"] == 1
    assert summary["rows_total"] == 1

    frame = pd.read_parquet(output)
    # Old hash gone, new hash only
    assert set(frame["run_config_hash"].unique()) == {config_hash}


def test_build_features_no_resume_forces_recompute(tmp_path: Path, monkeypatched_loaders) -> None:
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)
    output = tmp_path / "features.parquet"

    # Pre-seed with SAME hash to verify resume=False ignores
    groups = [("AAPL", date(2026, 1, 5))]
    factory = monkeypatched_loaders(groups, {k: _sample_trades() for k in groups})
    builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory,
        resume=True,
    )

    # Re-run with resume=False — should recompute even though same hash
    summary = builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory,
        resume=False,
    )
    assert summary["groups_skipped_resume"] == 0
    assert summary["rows_computed"] == 1  # recomputed


def test_build_features_empty_range_writes_nothing(tmp_path: Path, monkeypatched_loaders) -> None:
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)
    output = tmp_path / "features.parquet"

    factory = monkeypatched_loaders([], {})
    summary = builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory,
        resume=True,
    )

    assert summary["groups_processed"] == 0
    assert summary["rows_computed"] == 0
    assert summary["rows_total"] == 0
    assert not output.exists()


def test_build_features_skips_group_with_empty_trades(tmp_path: Path, monkeypatched_loaders) -> None:
    """If iter_ticker_dates yields (ticker, date) but DB returns 0 trades, skip that row."""
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)
    output = tmp_path / "features.parquet"

    groups = [("AAPL", date(2026, 1, 5)), ("MSFT", date(2026, 1, 5))]
    trades_map = {("AAPL", date(2026, 1, 5)): _sample_trades()}  # MSFT returns empty
    factory = monkeypatched_loaders(groups, trades_map)

    summary = builder.build_features(
        config=config,
        config_hash=config_hash,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        output_path=output,
        session_factory=factory,
        resume=True,
    )
    assert summary["groups_processed"] == 2
    assert summary["rows_computed"] == 1
    assert summary["rows_total"] == 1
