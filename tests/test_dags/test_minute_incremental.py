from __future__ import annotations

from datetime import date, datetime, timezone
import importlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
try:
    from airflow.exceptions import AirflowException
    from airflow.utils.trigger_rule import TriggerRule
except ImportError:
    from dags._airflow_compat import AirflowException, TriggerRule

import dags.task_groups.minute_incremental as minute_module


def test_resolve_minute_dates_to_sync_finds_uncompleted_days() -> None:
    result = minute_module.resolve_minute_dates_to_sync(
        reference_date=date(2026, 4, 17),
        state_rows=[
            {"trading_date": date(2026, 4, 13), "status": "completed"},
            {"trading_date": date(2026, 4, 14), "status": "failed"},
            {"trading_date": date(2026, 4, 15), "status": "completed"},
            {"trading_date": date(2026, 4, 16), "status": "skipped_holiday"},
            {"trading_date": date(2026, 4, 17), "status": "in_progress"},
        ],
    )

    assert result["status"] == "ok"
    assert result["dates_to_sync"] == ["2026-04-14", "2026-04-17"]


def test_resolve_minute_dates_excludes_today_session_when_market_not_closed() -> None:
    result = minute_module.resolve_minute_dates_to_sync(
        current_time=datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc),
        state_rows=[
            {"trading_date": date(2026, 4, 9), "status": "completed"},
            {"trading_date": date(2026, 4, 10), "status": "completed"},
            {"trading_date": date(2026, 4, 13), "status": "completed"},
            {"trading_date": date(2026, 4, 14), "status": "completed"},
            {"trading_date": date(2026, 4, 15), "status": "completed"},
            {"trading_date": date(2026, 4, 16), "status": "failed"},
        ],
    )

    assert result["reference_date"] == "2026-04-16"
    assert "2026-04-17" not in result["candidate_session_dates"]
    assert result["dates_to_sync"] == ["2026-04-16"]


def test_resolve_minute_dates_uses_previous_when_today_not_session() -> None:
    result = minute_module.resolve_minute_dates_to_sync(
        current_time=datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc),
        state_rows=[
            {"trading_date": date(2026, 4, 10), "status": "completed"},
            {"trading_date": date(2026, 4, 13), "status": "completed"},
            {"trading_date": date(2026, 4, 14), "status": "completed"},
            {"trading_date": date(2026, 4, 15), "status": "completed"},
            {"trading_date": date(2026, 4, 16), "status": "completed"},
            {"trading_date": date(2026, 4, 17), "status": "failed"},
        ],
    )

    assert result["reference_date"] == "2026-04-17"
    assert "2026-04-18" not in result["candidate_session_dates"]
    assert result["dates_to_sync"] == ["2026-04-17"]


def test_sync_polygon_minute_incremental_calls_run_minute_backfill(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_runner(command, cwd, capture_output, text, check):  # noqa: ANN001
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "capture_output": capture_output,
                "text": text,
                "check": check,
            },
        )
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    result = minute_module.sync_polygon_minute_incremental(
        resolved_dates=[date(2026, 4, 15), date(2026, 4, 17)],
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert result["status"] == "ok"
    assert len(calls) == 1
    command = calls[0]["command"]
    assert command[0] == sys.executable
    assert command[1] == str(tmp_path / "scripts" / "run_minute_backfill.py")
    assert command[-4:] == ["--end-date", "2026-04-17", "--universe-from-membership", "--resume"]


def test_validate_minute_internal_quality_catches_gap() -> None:
    minute_frame = _build_minute_frame("AAPL", date(2026, 4, 16), 379)

    with pytest.raises(AirflowException, match="insufficient_bars"):
        minute_module.validate_minute_internal_quality(
            resolved_dates=[date(2026, 4, 16)],
            minute_frame=minute_frame,
        )


def test_validate_minute_internal_quality_allows_early_close_day() -> None:
    minute_frame = _build_minute_frame("AAPL", date(2025, 11, 28), 208)

    result = minute_module.validate_minute_internal_quality(
        resolved_dates=[date(2025, 11, 28)],
        minute_frame=minute_frame,
    )

    assert result["status"] == "ok"
    assert result["failure_count"] == 0


def test_validate_minute_internal_quality_blocks_insufficient_bars_normal_day() -> None:
    minute_frame = _build_minute_frame("AAPL", date(2026, 4, 16), 200)

    with pytest.raises(AirflowException, match="insufficient_bars"):
        minute_module.validate_minute_internal_quality(
            resolved_dates=[date(2026, 4, 16)],
            minute_frame=minute_frame,
        )


def test_validate_minute_day_reconciliation_aplus_blocks_on_ohl() -> None:
    minute_frame = _build_minute_frame("AAPL", date(2026, 4, 16), 391)
    daily_prices = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "trade_date": date(2026, 4, 16),
                "open": 110.0,
                "high": float(minute_frame["high"].max()),
                "low": float(minute_frame["low"].min()),
                "close": float(minute_frame["close"].iloc[-1]),
                "volume": int(minute_frame["volume"].sum()),
            },
        ],
    )

    with pytest.raises(AirflowException, match="open:"):
        minute_module.validate_minute_day_reconciliation_aplus(
            resolved_dates=[date(2026, 4, 16)],
            minute_frame=minute_frame,
            daily_prices=daily_prices,
            persist_fn=lambda *args, **kwargs: 0,
        )


def test_publish_minute_watermark_updates_state_table(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        def __init__(self, value):
            self._value = value

        def scalar(self):
            return self._value

    class FakeConn:
        def __init__(self):
            self.calls: list[tuple[str, dict[str, object] | None]] = []

        def execute(self, statement, params=None):
            sql = str(statement)
            self.calls.append((sql, params))
            if "select max(trading_date)" in sql:
                return FakeResult(date(2026, 4, 17))
            return FakeResult(None)

    class FakeBegin:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self.conn

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def __init__(self):
            self.conn = FakeConn()

        def begin(self):
            return FakeBegin(self.conn)

    fake_engine = FakeEngine()
    monkeypatch.setattr(minute_module, "get_engine", lambda: fake_engine)

    result = minute_module.publish_minute_watermark(
        resolved_dates=[date(2026, 4, 16)],
        now=datetime(2026, 4, 18, 1, 2, 3, tzinfo=timezone.utc),
        state_columns={"published_at"},
    )

    assert result["status"] == "ok"
    assert result["latest_covered_date"] == "2026-04-17"
    assert any("update minute_backfill_state" in sql for sql, _ in fake_engine.conn.calls)


def test_dag_daily_data_parses_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_daily_data_module(monkeypatch, enabled="false")

    assert module.dag.dag_id == "daily_data_pipeline"
    assert "fetch_prices" in module.dag.task_ids


def test_minute_incremental_group_instantiated_when_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_daily_data_module(monkeypatch, enabled="true")
    expected = {
        "minute_incremental.resolve_minute_dates_to_sync",
        "minute_incremental.sync_polygon_minute_incremental",
        "minute_incremental.validate_minute_internal_quality",
        "minute_incremental.validate_minute_day_reconciliation_aplus",
        "minute_incremental.publish_minute_watermark",
    }

    assert expected <= set(module.dag.task_ids)


def test_minute_incremental_group_not_instantiated_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_daily_data_module(monkeypatch, enabled="false")

    assert not any(task_id.startswith("minute_incremental.") for task_id in module.dag.task_ids)


def test_dag_import_ok_without_exchange_calendars(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "exchange_calendars":
            raise ImportError("simulated missing exchange_calendars")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("ENABLE_MINUTE_INCREMENTAL", "false")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("dags.dag_daily_data", None)
    importlib.invalidate_caches()

    module = importlib.import_module("dags.dag_daily_data")

    assert module.dag.dag_id == "daily_data_pipeline"
    assert not any(task_id.startswith("minute_incremental.") for task_id in module.dag.task_ids)


def test_minute_sync_failure_does_not_block_weekly_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_daily_data_module(monkeypatch, enabled="true")

    publish_task = module.dag.get_task("minute_incremental.publish_minute_watermark")
    assert publish_task.trigger_rule == TriggerRule.ALL_DONE


def _load_daily_data_module(monkeypatch: pytest.MonkeyPatch, *, enabled: str):
    monkeypatch.setenv("ENABLE_MINUTE_INCREMENTAL", enabled)
    for module_name in ("dags.dag_daily_data",):
        sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    return importlib.import_module("dags.dag_daily_data")


def _build_minute_frame(ticker: str, trade_day: date, count: int) -> pd.DataFrame:
    session_open = pd.Timestamp(minute_module.XNYS.session_open(pd.Timestamp(trade_day)))
    timestamps = pd.date_range(
        session_open,
        periods=count,
        freq="min",
    )
    rows = []
    for idx, minute_ts in enumerate(timestamps):
        open_px = 100.0 + idx * 0.01
        close_px = open_px + 0.005
        rows.append(
            {
                "ticker": ticker,
                "trade_date": trade_day,
                "minute_ts": minute_ts,
                "open": open_px,
                "high": close_px + 0.001,
                "low": open_px - 0.001,
                "close": close_px,
                "volume": 1_000 + idx,
                "vwap": (open_px + close_px) / 2.0,
                "transactions": 10 + idx,
            },
        )
    return pd.DataFrame(rows)
