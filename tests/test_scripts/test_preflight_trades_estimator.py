from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

import scripts.preflight_trades_estimator as estimator


def _config_payload(stage: str = "pilot", *, max_storage_gb: float = 200.0, max_daily_calls: int = 50000) -> dict:
    return {
        "version": 1,
        "stage": stage,
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
            "max_daily_api_calls": max_daily_calls,
            "max_storage_gb": max_storage_gb,
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


def _config(stage: str = "pilot", *, max_storage_gb: float = 200.0, max_daily_calls: int = 50000) -> estimator.Week4TradesConfig:
    return estimator.Week4TradesConfig.model_validate(
        _config_payload(stage=stage, max_storage_gb=max_storage_gb, max_daily_calls=max_daily_calls),
    )


def _event(ticker: str, day: date, reason: str = "earnings") -> estimator.CandidateEvent:
    return estimator.CandidateEvent(ticker=ticker, trading_date=day, reason=reason)


def _tx(ticker: str, day: date, transactions: int) -> estimator.TransactionEstimate:
    return estimator.TransactionEstimate(ticker=ticker, trading_date=day, transactions=transactions)


def test_preflight_estimate_passes_normal_case() -> None:
    cfg = _config()
    day = date(2026, 1, 5)
    report = estimator.estimate_from_candidates(
        cfg,
        [_event("AAPL", day, "earnings"), _event("AAPL", day, "gap"), _event("MSFT", day, "weak_window")],
        [_tx("AAPL", day, 100_000), _tx("MSFT", day, 25_000)],
        config_hash="abc",
        start_date=day,
        end_date=day,
        concurrency=2,
    )

    assert report["pass"] is True
    assert report["inputs"]["candidate_event_rows"] == 3
    assert report["inputs"]["unique_ticker_days"] == 2
    assert report["estimates"]["rows"] == 125_000
    assert report["estimates"]["api_calls"] == 3
    assert report["estimates"]["probe_calls"] == 0


def test_preflight_estimate_fails_storage_budget() -> None:
    cfg = _config(max_storage_gb=0.0001)
    day = date(2026, 1, 5)

    report = estimator.estimate_from_candidates(
        cfg,
        [_event("AAPL", day)],
        [_tx("AAPL", day, 10_000_000)],
        config_hash="abc",
        start_date=day,
        end_date=day,
    )

    assert report["pass"] is False
    assert "storage_budget_exceeded" in report["failure_reasons"]


def test_preflight_estimate_fails_when_transaction_data_missing() -> None:
    cfg = _config()
    day = date(2026, 1, 5)

    report = estimator.estimate_from_candidates(
        cfg,
        [_event("AAPL", day)],
        [_tx("AAPL", day, 0)],
        config_hash="abc",
        start_date=day,
        end_date=day,
    )

    assert report["pass"] is True

    report = estimator.estimate_from_candidates(
        cfg,
        [_event("AAPL", day)],
        [estimator.TransactionEstimate(ticker="AAPL", trading_date=day, transactions=0, has_minute_data=False)],
        config_hash="abc",
        start_date=day,
        end_date=day,
    )

    assert report["pass"] is False
    assert "missing_transaction_estimates" in report["failure_reasons"]
    assert report["inputs"]["missing_transaction_estimate_examples"] == [
        {"ticker": "AAPL", "trading_date": "2026-01-05"},
    ]


def test_preflight_estimate_accepts_stage2_top_liquidity_expansion() -> None:
    cfg = _config(stage="stage2")
    day = date(2026, 1, 5)

    report = estimator.estimate_from_candidates(
        cfg,
        [_event("AAPL", day, "top_liquidity"), _event("MSFT", day, "top_liquidity")],
        [_tx("AAPL", day, 50_000), _tx("MSFT", day, 50_000)],
        config_hash="abc",
        start_date=day,
        end_date=day,
    )

    assert report["pass"] is True
    assert report["metadata"]["stage"] == "stage2"
    assert report["inputs"]["reason_counts"] == {"top_liquidity": 2}


def test_preflight_estimate_fails_daily_call_budget_overflow() -> None:
    cfg = _config(max_daily_calls=2)
    day = date(2026, 1, 5)

    report = estimator.estimate_from_candidates(
        cfg,
        [_event("AAPL", day), _event("MSFT", day)],
        [_tx("AAPL", day, 100_000), _tx("MSFT", day, 100_000)],
        config_hash="abc",
        start_date=day,
        end_date=day,
    )

    assert report["pass"] is False
    assert "daily_api_call_budget_exceeded" in report["failure_reasons"]
    assert report["estimates"]["daily_api_call_overflow_examples"] == [
        {"trading_date": "2026-01-05", "api_calls": 4},
    ]


def test_week4_yaml_safe_load_roundtrip_has_no_placeholders() -> None:
    config_path = Path("configs/research/week4_trades_sampling.yaml")
    payload = yaml.safe_load(config_path.read_text())

    estimator.assert_no_placeholders(payload)
    cfg = estimator.Week4TradesConfig.model_validate(payload)
    config_hash = estimator.compute_config_hash(cfg)

    assert cfg.stage == "pilot"
    assert cfg.sampling.pilot.weak_window_top_n == 100
    assert cfg.features.condition_allow_list == []
    assert len(config_hash) == 64
    assert estimator.compute_config_hash(cfg) == config_hash


def test_placeholder_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        estimator.assert_no_placeholders({"bad": "TODO"})
