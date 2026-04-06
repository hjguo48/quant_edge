from __future__ import annotations

from datetime import date
from pathlib import Path
import subprocess
import sys

import scripts.backfill_sp500_history as backfill_script


def test_backfill_sp500_history_parse_args_defaults() -> None:
    args = backfill_script._parse_args(["--phase", "prices"])

    assert args.phase == "prices"
    assert args.membership_start == date(2018, 1, 1)
    assert args.price_tail_days == 90
    assert args.strict_fmp is False
    assert args.dry_run is False
    assert args.limit is None
    assert args.tickers is None


def test_backfill_sp500_history_parse_args_supports_fundamentals_phase() -> None:
    args = backfill_script._parse_args(["--phase", "fundamentals", "--limit", "3"])

    assert args.phase == "fundamentals"
    assert args.limit == 3


def test_backfill_sp500_history_parse_tickers_normalizes_symbols() -> None:
    assert backfill_script._parse_tickers_arg("brk.b,bf-b, brk-b ") == ("BF-B", "BRK-B")


def test_backfill_sp500_history_build_price_intervals_merges_and_limits() -> None:
    membership_rows = [
        {
            "ticker": "AAA",
            "effective_date": date(2020, 1, 1),
            "end_date": date(2020, 3, 1),
        },
        {
            "ticker": "AAA",
            "effective_date": date(2020, 4, 15),
            "end_date": date(2020, 5, 1),
        },
        {
            "ticker": "BBB",
            "effective_date": date(2020, 2, 1),
            "end_date": date(2020, 2, 20),
        },
    ]

    intervals = backfill_script._build_price_intervals(
        membership_rows=membership_rows,
        price_tail_days=60,
        requested_tickers=("AAA", "BBB"),
        limit=1,
        max_end_date=date(2020, 6, 30),
    )

    assert intervals == [
        backfill_script.PriceInterval(
            ticker="AAA",
            effective_date=date(2020, 1, 1),
            end_date=date(2020, 6, 30),
        ),
    ]


def test_backfill_sp500_history_help_works_without_runtime_imports() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_sp500_history.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--phase" in result.stdout
    assert "--strict-fmp" in result.stdout
    assert "--failures-csv" in result.stdout


def test_backfill_sp500_history_required_tables_include_fundamentals() -> None:
    assert backfill_script._required_tables_for_phase("fundamentals") == (
        "universe_membership",
        "fundamentals_pit",
    )
    assert backfill_script._required_tables_for_phase("all") == (
        "universe_membership",
        "fundamentals_pit",
        "stock_prices",
    )
