from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.services.greyscale_reader import GreyscaleReader


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    report = {
        "week_number": 3,
        "generated_at_utc": "2026-04-12T00:00:00+00:00",
        "live_outputs": {
            "signal_date": "2026-04-10",
            "top_10_fusion_scores": [
                {"ticker": "BKNG", "score": 13.21},
                {"ticker": "OGN", "score": 6.26},
            ],
            "target_weights_after_risk": {"BKNG": 0.015, "OGN": 0.012, "AA": 0.010},
            "target_weights_raw": {"BKNG": 0.020, "OGN": 0.015},
        },
        "score_vectors": {
            "fusion": {"BKNG": 13.21, "OGN": 6.26, "AA": 0.54, "AAPL": 0.08},
            "ridge": {"BKNG": 10.0, "OGN": 5.0, "AA": 0.3, "AAPL": 0.1},
            "xgboost": {"BKNG": 14.0, "OGN": 7.0, "AA": 0.6, "AAPL": 0.05},
            "lightgbm": {"BKNG": 12.0, "OGN": 6.0, "AA": 0.5, "AAPL": 0.09},
        },
        "fusion": {
            "live_weights": {"ridge": 0.25, "xgboost": 0.40, "lightgbm": 0.35},
            "weight_source": "seed_weights",
            "regime": {"vix": 21.04, "regime": "mid", "scalar": 0.8},
        },
        "risk_checks": {
            "layer3_portfolio": {
                "pass": True,
                "checks": {
                    "overall_pass": True,
                    "checks": {
                        "beta": {"pass": True, "value": 0.908},
                        "cvar": {"pass": True, "value": -0.0404},
                    },
                },
                "report": {
                    "portfolio_beta": 0.908,
                    "cvar_95": -0.0404,
                    "holding_count": 61,
                    "gross_exposure": 0.80,
                    "cash_weight": 0.20,
                },
            },
        },
        "portfolio_metrics": {
            "turnover_vs_previous": 0.17,
            "holding_count_after_risk": 61,
            "gross_exposure_after_risk": 0.80,
            "cash_weight_after_risk": 0.20,
        },
        "db_state": {
            "stock_universe_size": 503,
        },
    }
    (tmp_path / "week_03.json").write_text(json.dumps(report))

    report_w2 = {
        **report,
        "week_number": 2,
        "live_outputs": {
            **report["live_outputs"],
            "signal_date": "2026-04-03",
        },
    }
    (tmp_path / "week_02.json").write_text(json.dumps(report_w2))
    return tmp_path


def test_latest_report(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    report = reader.get_latest_report()
    assert report is not None
    assert report["week_number"] == 3


def test_all_scores_sorted(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    scores = reader.get_all_fusion_scores()
    assert len(scores) == 4
    assert scores[0]["ticker"] == "BKNG"
    assert scores[0]["score"] > scores[1]["score"]
    assert scores[0]["rank"] == 1
    assert scores[0]["percentile"] == 100.0


def test_ticker_detail(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    detail = reader.get_ticker_detail("BKNG")
    assert detail is not None
    assert detail["fusion_score"] == pytest.approx(13.21)
    assert "ridge" in detail["model_scores"]
    assert detail["weight"] == pytest.approx(0.015)


def test_ticker_not_found(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    detail = reader.get_ticker_detail("ZZZZ")
    assert detail is None


def test_portfolio_holdings(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    holdings = reader.get_portfolio_holdings()
    assert len(holdings) == 3
    assert holdings[0]["ticker"] == "BKNG"
    assert holdings[0]["score"] == pytest.approx(13.21)


def test_portfolio_summary(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    summary = reader.get_portfolio_summary()
    assert summary is not None
    assert summary["week_number"] == 3
    assert summary["risk_pass"] is True
    assert summary["portfolio_beta"] == pytest.approx(0.908)


def test_signal_history(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    history = reader.get_signal_history("BKNG")
    assert len(history) == 2
    assert history[0]["signal_date"] < history[1]["signal_date"]
    assert history[0]["rank"] == 1


def test_no_reports(tmp_path: Path) -> None:
    reader = GreyscaleReader(report_dir=tmp_path)
    assert reader.get_latest_report() is None
    assert reader.get_all_fusion_scores() == []


def test_invalidate_cache_reads_new_file(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    assert reader.get_latest_report()["week_number"] == 3

    week_04 = {
        "week_number": 4,
        "live_outputs": {"signal_date": "2026-04-17", "target_weights_after_risk": {}},
        "score_vectors": {"fusion": {"AAPL": 1.0}},
        "risk_checks": {"layer3_portfolio": {"report": {}}},
        "portfolio_metrics": {},
    }
    (report_dir / "week_04.json").write_text(json.dumps(week_04))

    reader.invalidate_cache()
    assert reader.get_latest_report()["week_number"] == 4
