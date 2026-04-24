from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import scripts.build_trade_microstructure_minute_proxy_features as builder


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


def _minute_bars(ticker: str, trading_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": ticker, "trading_date": trading_date, "minute_ts": _et(f"{trading_date} 08:00"), "close": 99.0, "volume": 100},
            {"ticker": ticker, "trading_date": trading_date, "minute_ts": _et(f"{trading_date} 10:00"), "close": 100.0, "volume": 100},
            {"ticker": ticker, "trading_date": trading_date, "minute_ts": _et(f"{trading_date} 10:01"), "close": 101.0, "volume": 100},
            {"ticker": ticker, "trading_date": trading_date, "minute_ts": _et(f"{trading_date} 19:00"), "close": 100.0, "volume": 100},
        ],
    )


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _session_factory():
    return _FakeSession()


def _load_config(tmp_path: Path):
    config_path = _write_config(tmp_path)
    config = builder.preflight_trades_estimator.load_config(config_path)
    config_hash = builder.preflight_trades_estimator.compute_config_hash(config)
    return config, config_hash


def test_build_minute_proxy_features_happy_path_two_tickers_three_dates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_hash = _load_config(tmp_path)
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    monkeypatch.setattr(builder, "session_dates", lambda start, end: dates)

    def top_fn(as_of_date, *, top_n, session_factory=None):
        assert top_n == 2
        return ["AAPL", "MSFT"]

    output = tmp_path / "proxy.parquet"
    summary = builder.build_minute_proxy_features(
        config=config,
        config_hash=config_hash,
        start_date=dates[0],
        end_date=dates[-1],
        top_n=2,
        output_path=output,
        session_factory=_session_factory,
        top_liquidity_fn=top_fn,
        load_minutes_fn=lambda session, ticker, trading_date: _minute_bars(ticker, trading_date),
    )

    assert summary["rows_computed"] == 6
    frame = pd.read_parquet(output)
    assert list(frame.columns) == builder.OUTPUT_COLUMNS
    assert len(frame) == 6
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}
    assert frame["trade_imbalance_proxy"].notna().all()


def test_build_minute_proxy_features_resume_skips_existing_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_hash = _load_config(tmp_path)
    trading_date = date(2026, 1, 5)
    monkeypatch.setattr(builder, "session_dates", lambda start, end: [trading_date])
    output = tmp_path / "proxy.parquet"
    existing = pd.DataFrame(
        [
            {
                "event_date": trading_date,
                "ticker": "AAPL",
                "knowledge_time_regular": pd.Timestamp("2026-01-05 21:15", tz="UTC"),
                "knowledge_time_offhours": pd.Timestamp("2026-01-06 01:15", tz="UTC"),
                "trade_imbalance_proxy": 0.1,
                "late_day_aggressiveness": 1.0,
                "offhours_trade_ratio": 0.2,
                "run_config_hash": config_hash,
            },
        ],
    )
    existing.to_parquet(output, index=False)

    summary = builder.build_minute_proxy_features(
        config=config,
        config_hash=config_hash,
        start_date=trading_date,
        end_date=trading_date,
        top_n=1,
        output_path=output,
        session_factory=_session_factory,
        top_liquidity_fn=lambda as_of_date, *, top_n, session_factory=None: ["AAPL"],
        load_minutes_fn=lambda session, ticker, trading_date: pytest.fail("resume should skip load"),
    )

    assert summary["rows_skipped_resume"] == 1
    assert summary["rows_computed"] == 0
    assert len(pd.read_parquet(output)) == 1


def test_top_n_cli_flag_caps_liquidity_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_hash = _load_config(tmp_path)
    trading_date = date(2026, 1, 5)
    monkeypatch.setattr(builder, "session_dates", lambda start, end: [trading_date])
    observed_top_n: list[int] = []

    def top_fn(as_of_date, *, top_n, session_factory=None):
        observed_top_n.append(top_n)
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    output = tmp_path / "proxy.parquet"
    summary = builder.build_minute_proxy_features(
        config=config,
        config_hash=config_hash,
        start_date=trading_date,
        end_date=trading_date,
        top_n=2,
        output_path=output,
        session_factory=_session_factory,
        top_liquidity_fn=top_fn,
        load_minutes_fn=lambda session, ticker, trading_date: _minute_bars(ticker, trading_date),
    )

    assert observed_top_n == [2]
    assert summary["ticker_days_seen"] == 2
    assert len(pd.read_parquet(output)) == 2
