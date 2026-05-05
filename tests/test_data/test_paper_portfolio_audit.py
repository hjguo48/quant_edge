from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from scripts.compute_realized_returns import load_paper_portfolio, load_paper_portfolio_reports
from src.data.db.models import PaperPortfolioAudit
from src.data.paper_portfolio_audit import (
    load_db_paper_portfolio,
    save_paper_portfolio_snapshot,
)


def test_paper_portfolio_audit_round_trip(isolated_session_factory) -> None:
    with isolated_session_factory([PaperPortfolioAudit.__table__]) as session_factory:
        with session_factory() as session:
            conn = session.connection()
            rows = save_paper_portfolio_snapshot(
                bundle_version="w12_test",
                signal_date=date(2026, 5, 1),
                target_weights={"AAPL": 0.0125, "MSFT": 0.02},
                raw_scores={"AAPL": 1.5, "MSFT": -0.25},
                generated_at_utc=datetime(2026, 5, 2, 5, tzinfo=timezone.utc),
                run_id="week_01",
                conn=conn,
            )

            loaded = load_db_paper_portfolio(date(2026, 5, 1), conn=conn)

        assert rows == 2
        assert loaded == pytest.approx({"AAPL": 0.0125, "MSFT": 0.02})


def test_paper_portfolio_audit_same_signal_date_is_idempotent(isolated_session_factory) -> None:
    with isolated_session_factory([PaperPortfolioAudit.__table__]) as session_factory:
        with session_factory() as session:
            conn = session.connection()
            save_paper_portfolio_snapshot(
                bundle_version="w12_test",
                signal_date="2026-05-01",
                target_weights={"AAPL": 0.01, "MSFT": 0.02},
                generated_at_utc="2026-05-02T05:00:00+00:00",
                conn=conn,
            )
            rows = save_paper_portfolio_snapshot(
                bundle_version="w12_test",
                signal_date="2026-05-01",
                target_weights={"AAPL": 0.03},
                generated_at_utc="2026-05-02T06:00:00+00:00",
                conn=conn,
            )
            stored_count = session.execute(sa.select(sa.func.count()).select_from(PaperPortfolioAudit)).scalar_one()
            loaded = load_db_paper_portfolio("2026-05-01", conn=conn)

        assert rows == 1
        assert stored_count == 1
        assert loaded == pytest.approx({"AAPL": 0.03})


def test_load_paper_portfolio_falls_back_to_week_json(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "greyscale"
    report_dir.mkdir()
    (report_dir / "week_01.json").write_text(
        json.dumps(
            {
                "week_number": 1,
                "generated_at_utc": "2026-05-02T05:00:00+00:00",
                "live_outputs": {
                    "signal_date": "2026-05-01",
                    "target_weights_after_risk": {"AAPL": 0.01, "MSFT": 0.02},
                },
            },
        ),
    )

    with isolated_session_factory([PaperPortfolioAudit.__table__]) as session_factory:
        with session_factory() as session:
            loaded = load_paper_portfolio(
                date(2026, 5, 1),
                report_dir=report_dir,
                conn=session.connection(),
            )

    assert loaded == pytest.approx({"AAPL": 0.01, "MSFT": 0.02})


def test_load_paper_portfolio_reports_includes_db_only_signal_date(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    with isolated_session_factory([PaperPortfolioAudit.__table__]) as session_factory:
        with session_factory() as session:
            conn = session.connection()
            save_paper_portfolio_snapshot(
                bundle_version="w12_test",
                signal_date=date(2026, 5, 1),
                target_weights={"AAPL": 0.01},
                generated_at_utc=datetime(2026, 5, 2, 5, tzinfo=timezone.utc),
                conn=conn,
            )
            reports = load_paper_portfolio_reports(report_dir=tmp_path, conn=conn)

    assert len(reports) == 1
    assert reports[0]["paper_portfolio_source"] == "db"
    assert reports[0]["week_number"] == 1
    assert reports[0]["live_outputs"]["target_weights_after_risk"] == pytest.approx({"AAPL": 0.01})
