from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd

from scripts import audit_benchmark, audit_feature_parity, audit_price_truth, audit_universe


class _FakeScalarResult:
    def __init__(self, values):
        self._values = list(values)

    def all(self):
        return list(self._values)


class _FakeQueryResult:
    def __init__(self, payload):
        self._payload = payload

    def mappings(self):
        return self

    def __iter__(self):
        if isinstance(self._payload, list):
            return iter(self._payload)
        return iter([self._payload])

    def first(self):
        if isinstance(self._payload, list):
            return self._payload[0] if self._payload else None
        return self._payload

    def all(self):
        if isinstance(self._payload, list):
            return list(self._payload)
        return [self._payload]

    def scalars(self):
        return _FakeScalarResult(self._payload)

    def scalar(self):
        return self._payload


class _FakeConnection:
    def __init__(self, payloads):
        self._payloads = payloads

    def execute(self, statement, *args, **kwargs):  # noqa: ANN002, ANN003
        sql = " ".join(str(statement).split()).lower()
        for needle, payload in self._payloads.items():
            if needle in sql:
                return _FakeQueryResult(payload() if callable(payload) else payload)
        raise AssertionError(f"Unexpected SQL in test: {sql}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201
        return False


class _FakeEngine:
    def __init__(self, payloads):
        self._payloads = payloads

    def connect(self):
        return _FakeConnection(self._payloads)


def _noop_write(payload, output_path):  # noqa: ANN001
    return None


def _fake_issue_summary(items):  # noqa: ANN001
    critical = sum(1 for item in items if item.get("severity") == "critical")
    warnings = sum(1 for item in items if item.get("severity") == "warning")
    return critical, warnings


def test_feature_parity_audit_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit_feature_parity, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        audit_feature_parity,
        "load_current_state",
        lambda: {"live_champion": {"bundle_path": "data/models/fusion_model_bundle_60d.json"}},
    )
    monkeypatch.setattr(
        audit_feature_parity,
        "load_family_registry",
        lambda: {
            "feature_to_family": {
                "vol_60d": "price_momentum",
                "curve_inverted_x_growth": "macro_regime",
                "is_missing_curve_inverted_x_growth": "macro_regime",
            }
        },
    )
    monkeypatch.setattr(
        audit_feature_parity,
        "extract_feature_sets",
        lambda: {
            "technical": {"vol_60d"},
            "composite": {"curve_inverted_x_growth"},
        },
    )
    monkeypatch.setattr(audit_feature_parity, "get_engine", lambda: object())
    monkeypatch.setattr(
        audit_feature_parity,
        "load_feature_store_stats",
        lambda engine: (
            {
                "vol_60d": {"row_count": 10, "null_rate": 0.1, "min_date": date(2026, 4, 1), "max_date": date(2026, 4, 16), "dtype": "numeric(20,8)", "any_filled": True},
                "curve_inverted_x_growth": {"row_count": 10, "null_rate": 0.0, "min_date": date(2026, 4, 1), "max_date": date(2026, 4, 16), "dtype": "numeric(20,8)", "any_filled": True},
                "is_missing_curve_inverted_x_growth": {"row_count": 10, "null_rate": 0.0, "min_date": date(2026, 4, 1), "max_date": date(2026, 4, 16), "dtype": "numeric(20,8)", "any_filled": True},
            },
            date(2026, 4, 1),
            date(2026, 4, 16),
        ),
    )
    monkeypatch.setattr(
        audit_feature_parity,
        "get_parquet_date_range",
        lambda path: (date(2026, 4, 1), date(2026, 4, 16)),
    )
    monkeypatch.setattr(
        audit_feature_parity,
        "scan_parquet_stats",
        lambda *args, **kwargs: {
            "vol_60d": {"row_count": 10, "null_rate": 0.1, "dtype": "double"},
            "curve_inverted_x_growth": {"row_count": 10, "null_rate": 0.0, "dtype": "double"},
            "is_missing_curve_inverted_x_growth": {"row_count": 10, "null_rate": 0.0, "dtype": "double"},
        },
    )
    monkeypatch.setattr(audit_feature_parity, "summarize_issues", _fake_issue_summary)
    monkeypatch.setattr(audit_feature_parity, "write_json_report", _noop_write)

    bundle_path = tmp_path / "data/models/fusion_model_bundle_60d.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(
            {
                "retained_features": [
                    "vol_60d",
                    "curve_inverted_x_growth",
                    "is_missing_curve_inverted_x_growth",
                ]
            }
        ),
        encoding="utf-8",
    )
    parquet_path = tmp_path / "data/features/all_features.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.touch()

    payload = audit_feature_parity.build_report(parquet_path, tmp_path / "out.json")

    assert set(payload) >= {
        "metadata",
        "summary",
        "issues",
        "warnings",
        "bundle_missing_features",
        "only_in_feature_store",
        "only_in_parquet",
        "mismatched",
        "gap_report",
        "features",
    }
    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["summary"], dict)
    assert isinstance(payload["issues"], list)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["features"], list)
    assert isinstance(payload["summary"]["expected_feature_count"], int)
    assert isinstance(payload["summary"]["feature_store_feature_count"], int)
    assert isinstance(payload["summary"]["parquet_feature_count"], int)
    assert isinstance(payload["summary"]["bundle_feature_count"], int)
    assert isinstance(payload["summary"]["bundle_missing_feature_count"], int)
    assert "parity_window" in payload["metadata"]
    assert payload["features"][0]["name"]
    assert "matches" in payload["features"][0]


def test_price_truth_audit_contract(tmp_path, monkeypatch) -> None:
    fake_engine = _FakeEngine(
        {
            "count(*) as total_rows": {
                "total_rows": 100,
                "total_tickers": 3,
                "min_date": date(2026, 1, 1),
                "max_date": date(2026, 4, 16),
            },
            "((knowledge_time at time zone 'utc')::date - trade_date) as lag_days": [
                {"lag_days": 1, "row_count": 95},
                {"lag_days": 2, "row_count": 5},
            ],
            "where ticker = 'spy'": {"rows": 50, "min_date": date(2026, 1, 2), "max_date": date(2026, 4, 16)},
            "min(sp.trade_date) as first_available_date": [
                {"ticker": "AAPL", "first_available_date": date(2026, 1, 2)},
                {"ticker": "MSFT", "first_available_date": date(2026, 1, 2)},
            ],
        }
    )
    monkeypatch.setattr(audit_price_truth, "get_engine", lambda: fake_engine)
    monkeypatch.setattr(audit_price_truth, "summarize_issues", _fake_issue_summary)
    monkeypatch.setattr(audit_price_truth, "write_json_report", _noop_write)

    def _fake_read_sql(statement, conn, params=None):  # noqa: ANN001, ARG001
        sql = " ".join(str(statement).split()).lower()
        if "from stock_prices where close is not null" in sql:
            return pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "trade_date": "2026-04-10",
                        "close": 100.0,
                        "adj_close": 25.0,
                        "volume": 1000,
                        "knowledge_time": datetime(2026, 4, 11, tzinfo=timezone.utc),
                        "source": "polygon",
                        "adj_ratio": 0.25,
                    }
                ]
            )
        if "from corporate_actions" in sql:
            return pd.DataFrame([{"ticker": "AAPL", "ex_date": "2026-04-10", "ratio": 0.25}])
        if "where coalesce(volume, 0) = 0" in sql:
            return pd.DataFrame(
                [
                    {
                        "ticker": "MSFT",
                        "trade_date": "2026-04-12",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "adj_close": 10.0,
                        "volume": 0,
                        "knowledge_time": datetime(2026, 4, 13, tzinfo=timezone.utc),
                    }
                ]
            )
        if "where ((knowledge_time at time zone 'utc')::date) <= trade_date" in sql:
            return pd.DataFrame(columns=["ticker", "trade_date", "knowledge_time", "source"])
        raise AssertionError(f"Unexpected read_sql: {sql}")

    monkeypatch.setattr(audit_price_truth.pd, "read_sql", _fake_read_sql)

    payload = audit_price_truth.build_report(tmp_path / "out.json")

    assert set(payload) >= {
        "metadata",
        "summary",
        "issues",
        "warnings",
        "split_anomalies",
        "zero_volume_rows",
        "pit_violations",
        "corporate_action_coverage",
        "spy_summary",
        "sp500_coverage",
    }
    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["summary"], dict)
    assert isinstance(payload["issues"], list)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["split_anomalies"], list)
    assert isinstance(payload["zero_volume_rows"], list)
    assert isinstance(payload["pit_violations"], list)
    assert isinstance(payload["summary"]["total_rows"], int)
    assert isinstance(payload["summary"]["total_tickers"], int)
    assert isinstance(payload["summary"]["lag_distribution"], dict)
    assert isinstance(payload["spy_summary"]["rows"], int)
    assert isinstance(payload["corporate_action_coverage"]["split_rows"], int)


def test_universe_audit_contract(tmp_path, monkeypatch) -> None:
    fake_engine = _FakeEngine(
        {
            "select max(trade_date) from stock_prices": date(2026, 4, 16),
            "select effective_date, count(distinct ticker) as ticker_count": [
                {"effective_date": date(2026, 4, 1), "ticker_count": 600},
                {"effective_date": date(2026, 5, 1), "ticker_count": 605},
            ],
            "count(*) as total_stocks": {"total_stocks": 710, "delisted_count": 25},
            "select ticker, delist_date, delist_reason": [
                {"ticker": "OLD", "delist_date": date(2025, 1, 1), "delist_reason": "Acquired"}
            ],
            "select distinct sp.ticker": ["AAPL", "MSFT", "NVDA"],
            "select min(effective_date) as min_date": {
                "min_date": date(2026, 4, 1),
                "max_date": date(2026, 5, 1),
                "rows": 1205,
                "tickers": 605,
            },
            "select ticker, effective_date, end_date": [
                {"ticker": "AAPL", "effective_date": date(2026, 4, 1), "end_date": None},
                {"ticker": "MSFT", "effective_date": date(2026, 4, 1), "end_date": None},
            ],
        }
    )
    monkeypatch.setattr(audit_universe, "get_engine", lambda: fake_engine)
    monkeypatch.setattr(
        audit_universe,
        "resolve_active_universe",
        lambda latest_trade_date, as_of: (["AAPL", "MSFT"], "universe_membership"),
    )
    monkeypatch.setattr(audit_universe, "get_universe_pit", lambda as_of, index_name="SP500": ["AAPL", "MSFT"])
    monkeypatch.setattr(audit_universe, "summarize_issues", _fake_issue_summary)
    monkeypatch.setattr(audit_universe, "write_json_report", _noop_write)

    payload = audit_universe.build_report(tmp_path / "out.json")

    assert set(payload) >= {
        "metadata",
        "summary",
        "issues",
        "warnings",
        "membership_coverage",
        "live_vs_research_diff",
        "delisted_tickers",
        "survivorship_bias_risk",
    }
    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["summary"], dict)
    assert isinstance(payload["issues"], list)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["membership_coverage"], dict)
    assert isinstance(payload["live_vs_research_diff"], dict)
    assert isinstance(payload["delisted_tickers"], list)
    assert isinstance(payload["survivorship_bias_risk"], dict)
    assert isinstance(payload["summary"]["membership_rows"], int)
    assert isinstance(payload["summary"]["membership_tickers"], int)
    assert isinstance(payload["summary"]["stocks_total"], int)
    assert isinstance(payload["summary"]["research_universe_count"], int)
    assert isinstance(payload["summary"]["live_universe_count"], int)


def test_benchmark_audit_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        audit_benchmark,
        "load_spy_quality",
        lambda engine: (
            {"rows": 100, "min_date": date(2016, 4, 18), "max_date": date(2026, 4, 16)},
            [{"trade_date": date(2026, 1, 1), "next_trade_date": date(2026, 1, 7), "gap_days": 6}],
            [date(2026, 4, 14), date(2026, 4, 15), date(2026, 4, 16)],
            pd.DataFrame(
                [
                    {
                        "ticker": "SPY",
                        "trade_date": date(2026, 4, 16),
                        "open": 1.0,
                        "high": 1.1,
                        "low": 0.9,
                        "close": 1.0,
                        "adj_close": 1.0,
                        "volume": 100,
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        audit_benchmark,
        "compare_with_polygon",
        lambda sample_prices, recent_dates: {
            "issues": [],
            "rows": [
                {
                    "ticker": "SPY",
                    "trade_date": date(2026, 4, 16),
                    "open_abs_diff": 0.0,
                    "high_abs_diff": 0.01,
                    "low_abs_diff": 0.01,
                    "close_abs_diff": 0.0,
                    "adj_close_abs_diff": 0.0,
                    "volume_abs_diff": 10,
                }
            ],
            "summary": {
                "sample_count": 1,
                "open_abs_mean_diff": 0.0,
                "high_abs_mean_diff": 0.01,
                "low_abs_mean_diff": 0.01,
                "close_abs_mean_diff": 0.0,
                "adj_close_abs_mean_diff": 0.0,
                "volume_abs_mean_diff": 10.0,
            },
        },
    )
    monkeypatch.setattr(
        audit_benchmark,
        "analyze_labels",
        lambda spy_start: {
            "1D": {
                "path": "data/labels/forward_returns_1d.parquet",
                "total_rows": 1000,
                "rows_before_spy_start": 100,
                "null_excess_return_rows": 0,
            },
            "10D": {
                "path": "data/labels/forward_returns_10d.parquet",
                "total_rows": 1000,
                "rows_before_spy_start": 10,
                "null_excess_return_rows": 5,
            },
            "60D": {
                "path": "data/labels/forward_returns_60d.parquet",
                "total_rows": 1000,
                "rows_before_spy_start": 10,
                "null_excess_return_rows": 0,
            },
        },
    )
    monkeypatch.setattr(audit_benchmark, "get_engine", lambda: object())
    monkeypatch.setattr(audit_benchmark, "summarize_issues", _fake_issue_summary)
    monkeypatch.setattr(audit_benchmark, "write_json_report", _noop_write)

    payload = audit_benchmark.build_report(tmp_path / "out.json")

    assert set(payload) >= {
        "metadata",
        "summary",
        "issues",
        "warnings",
        "spy_data_quality",
        "massive_vs_stock_prices_diff",
        "massive_vs_stock_prices_sample_rows",
        "label_impact_analysis",
    }
    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["summary"], dict)
    assert isinstance(payload["issues"], list)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["spy_data_quality"], dict)
    assert isinstance(payload["label_impact_analysis"], dict)
    assert isinstance(payload["summary"]["spy_rows"], int)
    assert isinstance(payload["spy_data_quality"]["large_gap_count"], int)
    assert isinstance(payload["massive_vs_stock_prices_diff"]["sample_count"], int)
    assert isinstance(payload["label_impact_analysis"]["1D"]["total_rows"], int)
