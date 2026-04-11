from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from src.data.db.session import get_engine

EXPECTED_BRANCH = "feature/week10-bootstrap-fix"
TARGET_WINDOWS = ("W6", "W8")
COMPARISON_WINDOWS = ("W3", "W7")
TARGET_HORIZON = 60


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}.")

    extended_report = load_json(REPO_ROOT / args.extended_report_path)
    buffered_report = load_json(REPO_ROOT / args.extended_portfolio_report_path)
    predictions = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices = pd.read_parquet(REPO_ROOT / args.prices_path)
    labels = pd.read_parquet(REPO_ROOT / args.labels_path)
    stocks = load_stock_metadata()

    prepared = prepare_inputs(
        predictions=predictions,
        prices=prices,
        labels=labels,
        stocks=stocks,
        horizon=args.horizon,
    )
    portfolio_windows = build_window_lookup(buffered_report)

    diagnosis: dict[str, Any] = {}
    for window_id in TARGET_WINDOWS:
        window_report = portfolio_windows[window_id]
        diagnosis[window_id] = diagnose_window(
            window_id=window_id,
            window_report=window_report,
            prepared=prepared,
        )

    diagnosis["comparison_with_good_windows"] = {
        window_id: summarize_good_window(
            window_id=window_id,
            window_report=portfolio_windows[window_id],
            prepared=prepared,
        )
        for window_id in COMPARISON_WINDOWS
    }
    diagnosis["diagnosis_summary"] = build_summary(
        targets={window_id: diagnosis[window_id] for window_id in TARGET_WINDOWS},
        comparison=diagnosis["comparison_with_good_windows"],
    )
    diagnosis["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    diagnosis["git_branch"] = branch
    diagnosis["script"] = Path(__file__).name
    diagnosis["inputs"] = {
        "extended_report_path": str(REPO_ROOT / args.extended_report_path),
        "extended_portfolio_report_path": str(REPO_ROOT / args.extended_portfolio_report_path),
        "predictions_path": str(REPO_ROOT / args.predictions_path),
        "prices_path": str(REPO_ROOT / args.prices_path),
        "labels_path": str(REPO_ROOT / args.labels_path),
        "horizon": args.horizon,
    }

    output_path = REPO_ROOT / args.output_path
    write_json_atomic(output_path, json_safe(diagnosis))
    logger.info("saved W6/W8 diagnosis report to {}", output_path)
    log_summary(diagnosis)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why W6 and W8 failed to monetize despite strong IC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--extended-report-path", default="data/reports/extended_walkforward.json")
    parser.add_argument("--extended-portfolio-report-path", default="data/reports/extended_portfolio_buffered.json")
    parser.add_argument("--predictions-path", default="data/backtest/extended_walkforward_predictions.parquet")
    parser.add_argument("--prices-path", default="data/backtest/extended_walkforward_prices.parquet")
    parser.add_argument("--labels-path", default="data/labels/extended_walkforward_forward_returns_multi.parquet")
    parser.add_argument("--output-path", default="data/reports/w6_w8_diagnosis.json")
    parser.add_argument("--horizon", type=int, default=TARGET_HORIZON)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return __import__("json").loads(path.read_text())


def load_stock_metadata() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("select ticker, company_name, sector, industry from stocks"),
        )
        stocks = pd.DataFrame(result.fetchall(), columns=result.keys())
    stocks["ticker"] = stocks["ticker"].astype(str).str.upper()
    stocks["sector"] = stocks["sector"].fillna("Unknown").astype(str)
    stocks["industry"] = stocks["industry"].fillna("Unknown").astype(str)
    return stocks


def prepare_inputs(
    *,
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    labels: pd.DataFrame,
    stocks: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    predictions = predictions.copy()
    prices = prices.copy()
    labels = labels.loc[labels["horizon"].astype(int) == int(horizon)].copy()

    for frame in (predictions, prices, labels):
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])

    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    prices["dollar_volume"] = prices["adj_close"] * prices["volume"]

    merged = predictions.merge(
        labels[["ticker", "trade_date", "forward_return", "excess_return"]],
        on=["ticker", "trade_date"],
        how="inner",
        validate="many_to_one",
    ).merge(
        stocks[["ticker", "sector", "industry"]],
        on="ticker",
        how="left",
    )
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["industry"] = merged["industry"].fillna("Unknown")

    price_lookup = prices.set_index(["trade_date", "ticker"])[["adj_close", "close", "volume", "dollar_volume"]]
    return {
        "predictions": predictions,
        "prices": prices,
        "labels": labels,
        "stocks": stocks,
        "merged": merged,
        "price_lookup": price_lookup,
    }


def build_window_lookup(portfolio_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scheme = portfolio_report["schemes"]["equal_weight_buffered"]
    return {row["window_id"]: row for row in scheme["windows"]}


def diagnose_window(
    *,
    window_id: str,
    window_report: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    period_frame = periods_to_frame(window_report["portfolio"]["periods"])
    signal_diagnostics = build_signal_period_frame(
        window_id=window_id,
        period_frame=period_frame,
        prepared=prepared,
    )

    sector_analysis = compute_sector_analysis(signal_diagnostics)
    cost_analysis = compute_cost_analysis(period_frame)
    beta_analysis = compute_beta_analysis(period_frame, signal_diagnostics)
    signal_disconnect = compute_signal_disconnect(period_frame, signal_diagnostics)

    return {
        "test_period": str(window_report["test_period"]),
        "ic": float(window_report["test_ic"]),
        "rank_ic": float(window_report["test_rank_ic"]),
        "net_excess_annualized": float(window_report["annualized_net_excess"]),
        "gross_excess_annualized": float(window_report["annualized_gross_excess"]),
        "turnover": float(window_report["turnover"]),
        "sector_analysis": sector_analysis,
        "cost_analysis": cost_analysis,
        "beta_analysis": beta_analysis,
        "signal_disconnect": signal_disconnect,
    }


def summarize_good_window(
    *,
    window_id: str,
    window_report: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    period_frame = periods_to_frame(window_report["portfolio"]["periods"])
    signal_diagnostics = build_signal_period_frame(
        window_id=window_id,
        period_frame=period_frame,
        prepared=prepared,
    )
    cost_analysis = compute_cost_analysis(period_frame)
    beta_analysis = compute_beta_analysis(period_frame, signal_diagnostics)
    signal_disconnect = compute_signal_disconnect(period_frame, signal_diagnostics)
    return {
        "test_period": str(window_report["test_period"]),
        "ic": float(window_report["test_ic"]),
        "net_excess_annualized": float(window_report["annualized_net_excess"]),
        "gap_penalty_pct": float(cost_analysis["gap_penalty_pct"]),
        "gross_excess_total_return": float(beta_analysis["gross_excess_total_return"]),
        "avg_selected_rank_pct": float(signal_disconnect["avg_selected_rank_pct"]),
        "avg_top_decile_return": float(signal_disconnect["avg_top_decile_return"]),
        "avg_bottom_decile_return": float(signal_disconnect["avg_bottom_decile_return"]),
    }


def periods_to_frame(periods: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(periods)
    if frame.empty:
        raise RuntimeError("No portfolio periods found in report.")
    for column in ("signal_date", "execution_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame.sort_values("signal_date").reset_index(drop=True)


def build_signal_period_frame(
    *,
    window_id: str,
    period_frame: pd.DataFrame,
    prepared: dict[str, Any],
) -> pd.DataFrame:
    merged = prepared["merged"]
    prices = prepared["prices"]
    rows: list[dict[str, Any]] = []

    for period in period_frame.itertuples(index=False):
        signal_date = pd.Timestamp(period.signal_date)
        frame = merged.loc[
            (merged["window_id"] == window_id) & (merged["trade_date"] == signal_date)
        ].copy()
        if frame.empty:
            continue

        frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
        frame["rank_pct"] = (frame.index + 1) / len(frame)
        selected_tickers = {str(ticker).upper() for ticker in period.selected_tickers}
        selected = frame.loc[frame["ticker"].isin(selected_tickers)].copy()
        if selected.empty:
            continue

        bottom_decile_size = max(1, int(np.ceil(len(frame) * 0.10)))
        bottom = frame.tail(bottom_decile_size).copy()
        universe_median_forward_return = float(frame["forward_return"].median())
        universe_median_excess_return = float(frame["excess_return"].median())
        selected_avg_forward_return = float(selected["forward_return"].mean())
        selected_avg_excess_return = float(selected["excess_return"].mean())
        bottom_avg_forward_return = float(bottom["forward_return"].mean())
        bottom_avg_excess_return = float(bottom["excess_return"].mean())

        price_slice = prices.loc[prices["trade_date"] == signal_date, ["ticker", "dollar_volume"]]
        price_slice = price_slice.drop_duplicates("ticker")
        frame_with_dv = frame.merge(price_slice, on="ticker", how="left")
        selected_with_dv = frame_with_dv.loc[frame_with_dv["ticker"].isin(selected_tickers)]

        top_50_dollar_volume = set(frame_with_dv.sort_values("dollar_volume", ascending=False).head(50)["ticker"])

        selected_sector_weights = normalize_counts(selected["sector"])
        universe_sector_weights = normalize_counts(frame["sector"])

        rows.append(
            {
                "window_id": window_id,
                "signal_date": signal_date,
                "selected_count": int(len(selected)),
                "universe_size": int(len(frame)),
                "selected_avg_forward_return": selected_avg_forward_return,
                "selected_avg_excess_return": selected_avg_excess_return,
                "bottom_decile_avg_forward_return": bottom_avg_forward_return,
                "bottom_decile_avg_excess_return": bottom_avg_excess_return,
                "universe_median_forward_return": universe_median_forward_return,
                "universe_median_excess_return": universe_median_excess_return,
                "avg_selected_rank_pct": float(selected["rank_pct"].mean()),
                "avg_selected_score": float(selected["score"].mean()),
                "universe_median_score": float(frame["score"].median()),
                "selected_median_dollar_volume": float(selected_with_dv["dollar_volume"].median()),
                "universe_median_dollar_volume": float(frame_with_dv["dollar_volume"].median()),
                "selected_top_50_dollar_volume_share": float(
                    len(selected_tickers & top_50_dollar_volume) / max(len(selected_tickers), 1)
                ),
                "selected_sector_weights": selected_sector_weights,
                "universe_sector_weights": universe_sector_weights,
                "selected_tickers": sorted(selected_tickers),
            }
        )

    if not rows:
        raise RuntimeError(f"No signal diagnostics built for {window_id}.")
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def normalize_counts(values: pd.Series) -> dict[str, float]:
    normalized = values.fillna("Unknown").astype(str).value_counts(normalize=True).sort_index()
    return {sector: float(weight) for sector, weight in normalized.items()}


def compute_sector_analysis(signal_frame: pd.DataFrame) -> dict[str, Any]:
    selected_weights = pd.DataFrame(signal_frame["selected_sector_weights"].tolist()).fillna(0.0).mean().sort_values(ascending=False)
    universe_weights = pd.DataFrame(signal_frame["universe_sector_weights"].tolist()).fillna(0.0).mean().reindex(selected_weights.index.union(pd.DataFrame(signal_frame["universe_sector_weights"].tolist()).columns), fill_value=0.0)
    universe_weights = universe_weights.reindex(selected_weights.index.union(universe_weights.index), fill_value=0.0)
    selected_weights = selected_weights.reindex(universe_weights.index, fill_value=0.0)
    diff = (selected_weights - universe_weights).sort_values(ascending=False)

    return {
        "selected_sector_weights": {sector: float(value) for sector, value in selected_weights.sort_values(ascending=False).items()},
        "universe_sector_weights": {sector: float(value) for sector, value in universe_weights.sort_values(ascending=False).items()},
        "overweight_sectors": [
            {"sector": sector, "weight_diff": float(value)}
            for sector, value in diff.loc[diff > 0.01].items()
        ],
        "underweight_sectors": [
            {"sector": sector, "weight_diff": float(value)}
            for sector, value in diff.loc[diff < -0.01].sort_values().items()
        ],
        "max_overweight_sector": diff.index[0],
        "max_underweight_sector": diff.index[-1],
    }


def compute_cost_analysis(period_frame: pd.DataFrame) -> dict[str, Any]:
    totals = period_frame["cost_breakdown"].apply(lambda item: float(item["total"]))
    gap_penalties = period_frame["cost_breakdown"].apply(lambda item: float(item["gap_penalty"]))
    commissions = period_frame["cost_breakdown"].apply(lambda item: float(item["commission"]))
    spreads = period_frame["cost_breakdown"].apply(lambda item: float(item["spread"]))
    temporary = period_frame["cost_breakdown"].apply(lambda item: float(item["temporary_impact"]))
    permanent = period_frame["cost_breakdown"].apply(lambda item: float(item["permanent_impact"]))

    worst_idx = int(gap_penalties.idxmax())
    worst = period_frame.loc[worst_idx]
    worst_gap = float(gap_penalties.iloc[worst_idx])
    worst_total = float(totals.iloc[worst_idx])

    return {
        "total_cost": float(totals.sum()),
        "gap_penalty": float(gap_penalties.sum()),
        "gap_penalty_pct": float(gap_penalties.sum() / max(totals.sum(), 1e-12)),
        "commission": float(commissions.sum()),
        "spread": float(spreads.sum()),
        "temporary_impact": float(temporary.sum()),
        "permanent_impact": float(permanent.sum()),
        "worst_period": {
            "signal_date": worst["signal_date"].date().isoformat(),
            "execution_date": worst["execution_date"].date().isoformat(),
            "gap_penalty": worst_gap,
            "total_cost": worst_total,
            "gap_penalty_pct": float(worst_gap / max(worst_total, 1e-12)),
            "gross_excess_return": float(worst["gross_excess_return"]),
            "net_excess_return": float(worst["net_excess_return"]),
            "avg_gap": float(worst["avg_gap"]),
            "selected_count": int(worst["selected_count"]),
        },
    }


def compute_beta_analysis(period_frame: pd.DataFrame, signal_frame: pd.DataFrame) -> dict[str, Any]:
    benchmark_total_return = float(np.prod(1.0 + period_frame["benchmark_return"]) - 1.0)
    gross_total_return = float(np.prod(1.0 + period_frame["gross_return"]) - 1.0)
    net_total_return = float(np.prod(1.0 + period_frame["net_return"]) - 1.0)
    valid_benchmark = period_frame.loc[period_frame["benchmark_return"].abs() > 1e-4].copy()
    implied_beta_series = valid_benchmark["gross_return"] / valid_benchmark["benchmark_return"]

    benchmark_variance = float(period_frame["benchmark_return"].var(ddof=1))
    if benchmark_variance > 0.0:
        regression_beta = float(period_frame["gross_return"].cov(period_frame["benchmark_return"]) / benchmark_variance)
    else:
        regression_beta = float("nan")

    large_cap_proxy_ratio = float(
        signal_frame["selected_median_dollar_volume"].median()
        / max(signal_frame["universe_median_dollar_volume"].median(), 1e-12)
    )

    return {
        "spy_return": benchmark_total_return,
        "portfolio_gross_return": gross_total_return,
        "portfolio_net_return": net_total_return,
        "gross_excess_total_return": float(gross_total_return - benchmark_total_return),
        "net_excess_total_return": float(net_total_return - benchmark_total_return),
        "implied_beta_mean": float(implied_beta_series.mean()),
        "implied_beta_median": float(implied_beta_series.median()),
        "regression_beta": regression_beta,
        "selected_median_dollar_volume": float(signal_frame["selected_median_dollar_volume"].median()),
        "universe_median_dollar_volume": float(signal_frame["universe_median_dollar_volume"].median()),
        "selected_vs_universe_dollar_volume_ratio": large_cap_proxy_ratio,
        "selected_top_50_dollar_volume_share": float(signal_frame["selected_top_50_dollar_volume_share"].mean()),
    }


def compute_signal_disconnect(period_frame: pd.DataFrame, signal_frame: pd.DataFrame) -> dict[str, Any]:
    merged = signal_frame.merge(
        period_frame[
            [
                "signal_date",
                "gross_return",
                "gross_excess_return",
                "net_excess_return",
            ]
        ],
        on="signal_date",
        how="left",
    )
    return {
        "periods_where_top_decile_underperformed": int(
            (merged["selected_avg_forward_return"] < merged["universe_median_forward_return"]).sum()
        ),
        "periods_where_selected_underperformed_bottom_decile": int(
            (merged["selected_avg_forward_return"] < merged["bottom_decile_avg_forward_return"]).sum()
        ),
        "periods_with_positive_60d_signal_but_negative_realized_week": int(
            ((merged["selected_avg_forward_return"] > 0.0) & (merged["gross_return"] < 0.0)).sum()
        ),
        "total_periods": int(len(merged)),
        "avg_top_decile_return": float(merged["selected_avg_forward_return"].mean()),
        "avg_bottom_decile_return": float(merged["bottom_decile_avg_forward_return"].mean()),
        "avg_top_decile_excess_return": float(merged["selected_avg_excess_return"].mean()),
        "avg_universe_median_forward_return": float(merged["universe_median_forward_return"].mean()),
        "avg_selected_rank_pct": float(merged["avg_selected_rank_pct"].mean()),
        "avg_selected_score": float(merged["avg_selected_score"].mean()),
        "avg_realized_weekly_gross_return": float(merged["gross_return"].mean()),
        "avg_realized_weekly_gross_excess_return": float(merged["gross_excess_return"].mean()),
    }


def build_summary(
    *,
    targets: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    w6 = targets["W6"]
    w8 = targets["W8"]
    w3 = comparison["W3"]
    w7 = comparison["W7"]

    primary_cause = (
        "The main failure was monetization, not ranking. Both W6 and W8 kept positive 60D top-vs-bottom spreads, "
        "but the buffered long-only weekly book failed to convert that long-horizon signal into benchmark-relative returns."
    )

    secondary_causes = [
        (
            "Benchmark-relative exposure drift: both windows ran below-universe liquidity/size exposure "
            f"(dollar-volume ratios {w6['beta_analysis']['selected_vs_universe_dollar_volume_ratio']:.2f} and "
            f"{w8['beta_analysis']['selected_vs_universe_dollar_volume_ratio']:.2f}), so the book was tilted away from "
            "the largest benchmark-heavy names."
        ),
        (
            "Sector tilts were material. W6 leaned into Financials and Real Estate while underweighting Industrials and Healthcare; "
            "W8 leaned into Financials and Industrials while underweighting Healthcare and Consumer sectors."
        ),
        (
            "Gap costs stayed high in the bad windows, but they were not the unique root cause: "
            f"W6 gap penalty was {w6['cost_analysis']['gap_penalty_pct']:.1%} of cost and W8 was {w8['cost_analysis']['gap_penalty_pct']:.1%}, "
            f"yet good W3 was even higher at {w3['gap_penalty_pct']:.1%}. That makes cost a drag, not the primary differentiator."
        ),
        (
            "The weekly execution horizon is shorter than the trained 60D target horizon. "
            f"W8 averaged {w8['signal_disconnect']['avg_top_decile_return']:.2%} on the model's 60D forward label, "
            f"but only {w8['signal_disconnect']['avg_realized_weekly_gross_return']:.2%} in realized weekly gross return."
        ),
    ]

    phase2_recommendations = [
        "Align research and execution horizons. Test 20D-60D holding periods or explicitly train a shorter-horizon model for weekly turnover.",
        "Add benchmark-aware portfolio construction so the long book does not systematically underweight large-cap/liquid names when the market is mega-cap led.",
        "Revisit the 20%/25% buffered book. Monitor average selected rank percentile and cap the fraction of holdings that drift near the sell threshold.",
        "Separate signal quality from monetization quality in reporting: keep IC on the model horizon, but also track realized holding-period spread by window.",
        "Treat gap control as a secondary execution optimization, not the main explanation for W6/W8."
    ]

    return {
        "primary_cause": primary_cause,
        "secondary_causes": secondary_causes,
        "phase2_recommendations": phase2_recommendations,
    }


def log_summary(diagnosis: dict[str, Any]) -> None:
    for window_id in TARGET_WINDOWS:
        row = diagnosis[window_id]
        logger.info(
            "{} | ic={:.6f} net_excess={:.6f} gap_pct={:.1%} avg_rank_pct={:.3f}",
            window_id,
            row["ic"],
            row["net_excess_annualized"],
            row["cost_analysis"]["gap_penalty_pct"],
            row["signal_disconnect"]["avg_selected_rank_pct"],
        )
    logger.info("primary cause: {}", diagnosis["diagnosis_summary"]["primary_cause"])


if __name__ == "__main__":
    raise SystemExit(main())
