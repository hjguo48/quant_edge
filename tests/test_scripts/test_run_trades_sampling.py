from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import scripts.run_trades_sampling as runner
from src.data.polygon_trades import TradeRecord


def _config_payload(
    *,
    max_daily_api_calls: int = 50_000,
    max_storage_gb: float = 200.0,
    max_rows_per_ticker_day: int = 2_000_000,
    rest_page_size: int = 50_000,
    rest_max_pages_per_request: int = 50,
    condition_allow_list: list[int] | None = None,
) -> dict[str, Any]:
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
            "rest_max_pages_per_request": rest_max_pages_per_request,
            "rest_page_size": rest_page_size,
            "rest_min_interval_seconds": 0.05,
            "retry_max": 3,
        },
        "budgets": {
            "max_daily_api_calls": max_daily_api_calls,
            "max_storage_gb": max_storage_gb,
            "max_rows_per_ticker_day": max_rows_per_ticker_day,
            "expected_pilot_ticker_days": 30000,
        },
        "features": {
            "size_threshold_dollars": 1000000,
            "size_threshold_min_cap_dollars": 250000,
            "condition_allow_list": condition_allow_list or [],
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


def _write_config(tmp_path: Path, **overrides: Any) -> Path:
    path = tmp_path / "week4.yaml"
    path.write_text(yaml.safe_dump(_config_payload(**overrides)))
    return path


def _write_plan(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "plan.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _trade(
    *,
    ticker: str = "AAPL",
    day: date = date(2026, 1, 5),
    seq: int = 1,
    exchange: int = 4,
    trade_id: str = "T1",
    conditions: list[int] | None = None,
) -> TradeRecord:
    sip = datetime(2026, 1, 5, 14, 30, 0, tzinfo=timezone.utc) + pd.Timedelta(microseconds=seq)
    return TradeRecord(
        ticker=ticker,
        trading_date=day,
        sip_timestamp=sip,
        participant_timestamp=sip,
        trf_timestamp=None,
        price=Decimal("100.000000"),
        size=Decimal("100.0000"),
        decimal_size=None,
        exchange=exchange,
        tape=1,
        conditions=conditions if conditions is not None else [0],
        correction=0,
        sequence_number=seq,
        trade_id=trade_id,
        trf_id=None,
    )


class _FakePolygonClient:
    def __init__(
        self,
        responses: dict[tuple[str, date], list[TradeRecord] | tuple[list[TradeRecord], int] | Exception],
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, date, int | None, int | None]] = []

    def fetch_trades_for_day(self, ticker, trading_date, *, page_size=None, max_pages=None):
        self.calls.append((ticker, trading_date, page_size, max_pages))
        result = self.responses.get((ticker, trading_date), [])
        if isinstance(result, Exception):
            raise result
        if isinstance(result, tuple):
            return result
        return result, 1


class _FakeRepository:
    def __init__(
        self,
        *,
        states: dict[tuple[str, date, str], dict[str, Any]],
        relation_size_bytes: int = 0,
    ) -> None:
        self.states = states
        self.rows: list[dict[str, Any]] = []
        self.pk_seen: set[tuple[str, datetime, int, int]] = set()
        self._relation_size_bytes = relation_size_bytes
        self.budget_deferred: list[runner.SamplingJob] = []

    def load_jobs(self, plan: pd.DataFrame, *, resume: bool, limit: int | None = None):
        allowed = {"pending", "failed"} if resume else {"pending"}
        plan_keys = {
            (str(row.ticker).upper(), row.trading_date, str(row.reason).lower())
            for row in plan.itertuples(index=False)
        }
        grouped: dict[tuple[str, date], set[str]] = {}
        for key, state in self.states.items():
            if key in plan_keys and state["status"] in allowed:
                grouped.setdefault((key[0], key[1]), set()).add(key[2])
        jobs = [
            runner.SamplingJob(ticker=ticker, trading_date=trading_date, reasons=tuple(sorted(reasons)))
            for (ticker, trading_date), reasons in grouped.items()
        ]
        jobs.sort(key=lambda job: (job.trading_date, job.ticker, job.reasons))
        return jobs[:limit] if limit is not None else jobs

    def insert_trades(self, rows: list[dict[str, Any]], *, batch_size: int) -> int:
        inserted = 0
        for row in rows:
            key = (row["ticker"], row["sip_timestamp"], row["exchange"], row["sequence_number"])
            if key in self.pk_seen:
                continue
            self.pk_seen.add(key)
            self.rows.append(row)
            inserted += 1
        return inserted

    def persist_completed(self, result: runner.FetchResult, *, batch_size: int) -> int:
        inserted = self.insert_trades(result.rows, batch_size=batch_size)
        self.mark_completed(result, rows_ingested=inserted)
        return inserted

    def mark_completed(self, result: runner.FetchResult, *, rows_ingested: int) -> None:
        parts: list[str] = []
        if result.dropped_conditions:
            parts.append(f"dropped_conditions={result.dropped_conditions}")
        for reason in result.job.reasons:
            state = self.states[(result.job.ticker, result.job.trading_date, reason)]
            state.update(
                {
                    "status": "partial" if result.partial else "completed",
                    "rows_ingested": rows_ingested,
                    "pages_fetched": result.pages_fetched,
                    "api_calls_used": result.api_calls_used,
                    "error_message": "; ".join(parts) if parts else None,
                },
            )

    def mark_failed(self, job: runner.SamplingJob, error_message: str) -> None:
        for reason in job.reasons:
            self.states[(job.ticker, job.trading_date, reason)].update(
                {"status": "failed", "error_message": error_message},
            )

    def mark_budget_deferred(self, jobs, error_message: str) -> None:
        self.budget_deferred.extend(jobs)
        for job in jobs:
            for reason in job.reasons:
                self.states[(job.ticker, job.trading_date, reason)]["error_message"] = error_message

    def relation_size_bytes(self) -> int:
        return self._relation_size_bytes


def _state(rows: list[dict[str, Any]], *, default_status: str = "pending") -> dict[tuple[str, date, str], dict[str, Any]]:
    return {
        (row["ticker"], row["trading_date"], row["reason"]): {"status": row.get("status", default_status)}
        for row in rows
    }


def test_run_trades_sampling_basic_path_persists_rows_and_updates_state(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [{"ticker": "AAPL", "trading_date": day, "reason": "earnings"}]
    repo = _FakeRepository(states=_state(plan_rows))
    client = _FakePolygonClient({("AAPL", day): ([_trade(seq=i) for i in range(1, 6)], 3)})

    summary = runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path),
        max_workers=1,
        repository=repo,
        client_factory=lambda: client,
    )

    assert summary["completed_jobs"] == 1
    assert summary["rows_ingested"] == 5
    assert summary["api_calls_used"] == 3
    assert len(repo.rows) == 5
    assert repo.rows[0]["knowledge_time"] == repo.rows[0]["sip_timestamp"] + pd.Timedelta(minutes=15)
    assert repo.states[("AAPL", day, "earnings")]["status"] == "completed"
    assert repo.states[("AAPL", day, "earnings")]["rows_ingested"] == 5
    assert repo.states[("AAPL", day, "earnings")]["api_calls_used"] == 3


def test_max_pages_truncation_marks_completed_with_partial_warning(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [{"ticker": "TSLA", "trading_date": day, "reason": "gap"}]
    repo = _FakeRepository(states=_state(plan_rows))
    client = _FakePolygonClient({("TSLA", day): [_trade(ticker="TSLA", seq=1), _trade(ticker="TSLA", seq=2)]})

    summary = runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path, rest_page_size=2, rest_max_pages_per_request=1),
        max_workers=1,
        repository=repo,
        client_factory=lambda: client,
    )

    assert summary["completed_jobs"] == 1
    state = repo.states[("TSLA", day, "gap")]
    assert state["status"] == "partial"
    assert state["error_message"] is None
    assert state["rows_ingested"] == 2


def test_duplicate_trade_id_across_exchange_does_not_conflict(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [{"ticker": "AAPL", "trading_date": day, "reason": "earnings"}]
    repo = _FakeRepository(states=_state(plan_rows))
    trades = [
        _trade(seq=100, exchange=4, trade_id="same-id"),
        _trade(seq=100, exchange=5, trade_id="same-id"),
    ]
    client = _FakePolygonClient({("AAPL", day): trades})

    runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path),
        max_workers=1,
        repository=repo,
        client_factory=lambda: client,
    )

    assert len(repo.rows) == 2
    assert {(row["exchange"], row["sequence_number"]) for row in repo.rows} == {(4, 100), (5, 100)}


def test_daily_api_budget_exhaustion_stops_normally_and_leaves_remaining_pending(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [
        {"ticker": "AAPL", "trading_date": day, "reason": "earnings"},
        {"ticker": "MSFT", "trading_date": day, "reason": "earnings"},
    ]
    repo = _FakeRepository(states=_state(plan_rows))
    client = _FakePolygonClient(
        {
            ("AAPL", day): [_trade(ticker="AAPL", seq=1)],
            ("MSFT", day): [_trade(ticker="MSFT", seq=1)],
        },
    )

    summary = runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path, max_daily_api_calls=1),
        max_workers=1,
        repository=repo,
        client_factory=lambda: client,
    )

    assert summary["stopped_reason"] == "daily_api_budget_exhausted"
    assert summary["completed_jobs"] == 1
    assert repo.states[("AAPL", day, "earnings")]["status"] == "completed"
    assert repo.states[("MSFT", day, "earnings")]["status"] == "pending"
    assert repo.states[("MSFT", day, "earnings")]["error_message"] == "daily_api_budget_exhausted"
    assert [call[0] for call in client.calls] == ["AAPL"]


def test_storage_kill_switch_raises(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [{"ticker": "AAPL", "trading_date": day, "reason": "earnings"}]
    repo = _FakeRepository(states=_state(plan_rows), relation_size_bytes=10 * 1024**3)
    client = _FakePolygonClient({("AAPL", day): [_trade(seq=1)]})

    with pytest.raises(runner.StorageBudgetExceeded):
        runner.run_sampling(
            plan_path=_write_plan(tmp_path, plan_rows),
            config_path=_write_config(tmp_path, max_storage_gb=0.001),
            max_workers=1,
            repository=repo,
            client_factory=lambda: client,
        )


def test_resume_retries_pending_and_failed_but_skips_completed(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [
        {"ticker": "AAPL", "trading_date": day, "reason": "earnings", "status": "pending"},
        {"ticker": "MSFT", "trading_date": day, "reason": "gap", "status": "failed"},
        {"ticker": "NVDA", "trading_date": day, "reason": "weak_window", "status": "completed"},
    ]
    repo = _FakeRepository(states=_state(plan_rows))
    client = _FakePolygonClient(
        {
            ("AAPL", day): [_trade(ticker="AAPL", seq=1)],
            ("MSFT", day): [_trade(ticker="MSFT", seq=1)],
            ("NVDA", day): [_trade(ticker="NVDA", seq=1)],
        },
    )

    summary = runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path),
        max_workers=1,
        resume=True,
        repository=repo,
        client_factory=lambda: client,
    )

    assert summary["completed_jobs"] == 2
    assert [call[0] for call in client.calls] == ["AAPL", "MSFT"]
    assert repo.states[("AAPL", day, "earnings")]["status"] == "completed"
    assert repo.states[("MSFT", day, "gap")]["status"] == "completed"
    assert repo.states[("NVDA", day, "weak_window")]["status"] == "completed"


def test_same_ticker_date_multiple_reasons_single_fetch_updates_both_states(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [
        {"ticker": "AAPL", "trading_date": day, "reason": "earnings"},
        {"ticker": "AAPL", "trading_date": day, "reason": "gap"},
    ]
    repo = _FakeRepository(states=_state(plan_rows))
    client = _FakePolygonClient({("AAPL", day): [_trade(seq=1), _trade(seq=2)]})

    summary = runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path),
        max_workers=1,
        repository=repo,
        client_factory=lambda: client,
    )

    assert summary["completed_jobs"] == 1
    assert len(client.calls) == 1
    assert repo.states[("AAPL", day, "earnings")]["status"] == "completed"
    assert repo.states[("AAPL", day, "gap")]["status"] == "completed"
    assert repo.states[("AAPL", day, "earnings")]["rows_ingested"] == 2
    assert repo.states[("AAPL", day, "gap")]["rows_ingested"] == 2


def test_row_limit_exceeded_marks_state_failed_without_insert(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [{"ticker": "AAPL", "trading_date": day, "reason": "earnings"}]
    repo = _FakeRepository(states=_state(plan_rows))
    client = _FakePolygonClient({("AAPL", day): [_trade(seq=i) for i in range(1, 11)]})

    summary = runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path, max_rows_per_ticker_day=5),
        max_workers=1,
        repository=repo,
        client_factory=lambda: client,
    )

    assert summary["failed_jobs"] == 1
    assert repo.rows == []
    assert repo.states[("AAPL", day, "earnings")]["status"] == "failed"
    assert repo.states[("AAPL", day, "earnings")]["error_message"] == "row_limit_exceeded"


def test_dry_run_does_not_write_state_and_limit_runs_first_n_jobs(tmp_path: Path) -> None:
    day = date(2026, 1, 5)
    plan_rows = [
        {"ticker": "AAPL", "trading_date": day, "reason": "earnings"},
        {"ticker": "MSFT", "trading_date": day, "reason": "earnings"},
    ]
    repo = _FakeRepository(states=_state(plan_rows))
    client = _FakePolygonClient(
        {
            ("AAPL", day): [_trade(ticker="AAPL", seq=1)],
            ("MSFT", day): [_trade(ticker="MSFT", seq=1)],
        },
    )

    summary = runner.run_sampling(
        plan_path=_write_plan(tmp_path, plan_rows),
        config_path=_write_config(tmp_path),
        max_workers=1,
        dry_run=True,
        limit=1,
        repository=repo,
        client_factory=lambda: client,
    )

    assert summary["total_jobs"] == 1
    assert summary["completed_jobs"] == 1
    assert summary["rows_ingested"] == 1
    assert repo.rows == []
    assert [call[0] for call in client.calls] == ["AAPL"]
    assert repo.states[("AAPL", day, "earnings")]["status"] == "pending"
    assert repo.states[("MSFT", day, "earnings")]["status"] == "pending"
