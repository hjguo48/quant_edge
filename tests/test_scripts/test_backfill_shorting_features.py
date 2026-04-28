from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

import scripts.backfill_shorting_features as backfill_module


def _fake_shorting_frame(*, tickers: list[str], output_dates: list[date]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for output_date in output_dates:
        for ticker in tickers:
            for feature_name in backfill_module.SHORTING_FEATURE_NAMES:
                rows.append(
                    {
                        "ticker": ticker,
                        "trade_date": output_date,
                        "feature_name": feature_name,
                        "feature_value": 0.1,
                    },
                )
    return pd.DataFrame(rows)


def test_resolve_tickers_defaults_to_historical_active_universe_union(
    monkeypatch,
) -> None:
    seen_dates: list[date] = []

    def fake_resolve_active_universe(
        output_date: date,
        *,
        as_of: date | None = None,
        benchmark_ticker: str = "SPY",
        index_name: str = "SP500",
    ) -> tuple[list[str], str]:
        seen_dates.append(output_date)
        if output_date == date(2026, 4, 17):
            return (["AAPL", "MSFT", "MRSH"], "universe_membership")
        return (["AAPL", "NVDA", "PSKY"], "universe_membership")

    monkeypatch.setattr(backfill_module, "resolve_active_universe", fake_resolve_active_universe)
    monkeypatch.setattr(
        backfill_module,
        "resolve_price_continuity_counts",
        lambda **kwargs: {
            date(2026, 4, 17): {"AAPL": 250, "MSFT": 230, "MRSH": 64},
            date(2026, 4, 24): {"AAPL": 251, "NVDA": 240, "PSKY": 0},
        },
    )

    args = SimpleNamespace(
        tickers=None,
        use_dynamic_universe=False,
        use_frozen_universe=False,
        frozen_universe=backfill_module.DEFAULT_FROZEN_UNIVERSE,
        as_of=None,
    )
    tickers = backfill_module.resolve_tickers(
        args,
        output_dates=[date(2026, 4, 17), date(2026, 4, 24)],
    )

    assert seen_dates == [date(2026, 4, 17), date(2026, 4, 24)]
    assert tickers == ["AAPL", "MSFT", "NVDA"]


def test_main_use_dynamic_universe_dry_run_reports_universe_summary(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        backfill_module,
        "resolve_output_dates",
        lambda args: [date(2026, 4, 24)],
    )
    monkeypatch.setattr(
        backfill_module,
        "resolve_tickers",
        lambda args, output_dates: ["AAPL", "NVDA", "SPY"],
    )
    monkeypatch.setattr(
        backfill_module,
        "compute_shorting_features_batch",
        lambda *, tickers, output_dates: _fake_shorting_frame(tickers=tickers, output_dates=output_dates),
    )

    exit_code = backfill_module.main(["--use-dynamic-universe", "--recent-fridays", "1", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Universe size: 3" in captured.out
    assert "Contains MRSH: False | Contains PSKY: False" in captured.out


def test_resolve_tickers_preserves_explicit_frozen_universe_flag(
    tmp_path,
) -> None:
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text('{"tickers": ["MRSH", "PSKY", "AAPL"]}', encoding="utf-8")

    args = SimpleNamespace(
        tickers=None,
        use_dynamic_universe=False,
        use_frozen_universe=True,
        frozen_universe="frozen.json",
        as_of=None,
    )

    original_root = backfill_module.REPO_ROOT
    backfill_module.REPO_ROOT = tmp_path
    try:
        tickers = backfill_module.resolve_tickers(args, output_dates=[date(2026, 4, 24)])
    finally:
        backfill_module.REPO_ROOT = original_root
    assert tickers == ["MRSH", "PSKY", "AAPL"]
