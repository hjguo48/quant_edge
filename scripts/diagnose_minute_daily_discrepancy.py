#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import sqlalchemy as sa

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_engine
from src.data.sources.polygon import PolygonDataSource, to_polygon_request_ticker
from src.features.intraday import aggregate_minute_to_daily

SMOKE_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "XOM",
    "UNH",
)
FIELDS = ("open", "high", "low", "close", "volume")
PROBLEMATIC_FIELDS = ("close", "volume")


@dataclass(frozen=True)
class SampleKey:
    ticker: str
    trade_date: date


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3.0.5 B-lite diagnosis for stock_prices vs Polygon daily vs minute aggregates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", default="2026-01-02")
    parser.add_argument("--end-date", default="2026-01-08")
    parser.add_argument("--tickers", nargs="*", default=list(SMOKE_TICKERS))
    parser.add_argument(
        "--report-output",
        default="data/reports/week3_blite_discrepancy_diagnosis_20260417.json",
    )
    parser.add_argument("--samples-per-category", type=int, default=6)
    return parser.parse_args(argv)


def parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    return value


def to_float_frame(frame: pd.DataFrame, columns: tuple[str, ...] = FIELDS) -> pd.DataFrame:
    converted = frame.copy()
    for column in columns:
        if column in converted.columns:
            converted[column] = pd.to_numeric(converted[column], errors="coerce").astype(float)
    return converted


def bp_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if pd.isna(a) or pd.isna(b):
        return None
    midpoint = (abs(a) + abs(b)) / 2.0
    if midpoint == 0:
        return 0.0
    return abs(float(a) - float(b)) / midpoint * 10_000.0


def load_minute_daily(*, tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    query = sa.text(
        """
        select
            ticker,
            trade_date,
            minute_ts,
            open,
            high,
            low,
            close,
            volume,
            vwap,
            transactions
        from stock_minute_aggs
        where ticker = any(:tickers)
          and trade_date >= :start_date
          and trade_date <= :end_date
        order by ticker, minute_ts
        """,
    )
    with get_engine().connect() as conn:
        frame = pd.read_sql_query(
            query,
            conn,
            params={
                "tickers": list(tickers),
                "start_date": start_date,
                "end_date": end_date,
            },
            parse_dates=["minute_ts"],
        )
    daily = aggregate_minute_to_daily(frame)
    daily = to_float_frame(daily)
    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    return daily


def load_stock_prices(*, tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    prices = get_prices_pit(
        tickers=tickers,
        start_date=start_date - pd.offsets.BDay(5),
        end_date=end_date,
        as_of=datetime.now(timezone.utc),
    )
    prices = prices.loc[:, ["ticker", "trade_date", *FIELDS]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    prices = to_float_frame(prices)
    return prices


def load_polygon_daily(sample_keys: list[SampleKey]) -> pd.DataFrame:
    provider = PolygonDataSource()
    rows: list[dict[str, Any]] = []
    for key in sample_keys:
        provider_ticker = to_polygon_request_ticker(key.ticker)
        bars = provider._list_aggs(provider_ticker, key.trade_date, key.trade_date, adjusted=False)
        bar = bars.get(key.trade_date)
        if bar is None:
            continue
        rows.append(
            {
                "ticker": key.ticker,
                "trade_date": key.trade_date,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
            },
        )
    return pd.DataFrame(rows, columns=["ticker", "trade_date", *FIELDS])


def build_ranking_frame(stock_prices: pd.DataFrame, minute_daily: pd.DataFrame) -> pd.DataFrame:
    merged = stock_prices.merge(
        minute_daily,
        on=["ticker", "trade_date"],
        how="inner",
        suffixes=("_s1", "_s3"),
    )
    for field in FIELDS:
        merged[f"{field}_bp_13"] = merged.apply(
            lambda row, f=field: bp_delta(row[f"{f}_s1"], row[f"{f}_s3"]),
            axis=1,
        )
    merged["problematic_mean_bp_13"] = merged[[f"{field}_bp_13" for field in PROBLEMATIC_FIELDS]].mean(axis=1)
    return merged


def query_special_candidates(
    *,
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> dict[SampleKey, list[str]]:
    tags: dict[SampleKey, list[str]] = defaultdict(list)
    with get_engine().connect() as conn:
        corp = pd.read_sql_query(
            sa.text(
                """
                select ticker, ex_date, action_type, ratio
                from corporate_actions
                where ticker = any(:tickers)
                  and ex_date between :start_date and :end_date
                order by ticker, ex_date
                """,
            ),
            conn,
            params={
                "tickers": list(tickers),
                "start_date": start_date - timedelta(days=20),
                "end_date": end_date,
            },
        )
        earnings = pd.read_sql_query(
            sa.text(
                """
                select ticker, fiscal_date
                from earnings_estimates
                where ticker = any(:tickers)
                  and fiscal_date between :start_date and :end_date
                order by ticker, fiscal_date
                """,
            ),
            conn,
            params={
                "tickers": list(tickers),
                "start_date": start_date - timedelta(days=2),
                "end_date": end_date + timedelta(days=10),
            },
        )
        vix = pd.read_sql_query(
            sa.text(
                """
                select observation_date, value
                from macro_series_pit
                where series_id = 'VIXCLS'
                  and observation_date between :start_date and :end_date
                order by observation_date
                """,
            ),
            conn,
            params={"start_date": start_date, "end_date": end_date},
        )

    if not corp.empty:
        corp["ticker"] = corp["ticker"].astype(str).str.upper()
        corp["ex_date"] = pd.to_datetime(corp["ex_date"]).dt.date
        for row in corp.itertuples(index=False):
            key = SampleKey(row.ticker, row.ex_date)
            tags[key].append(f"{str(row.action_type).lower()}_event")

    if not earnings.empty:
        earnings["ticker"] = earnings["ticker"].astype(str).str.upper()
        earnings["fiscal_date"] = pd.to_datetime(earnings["fiscal_date"]).dt.date
        smoke_days = pd.date_range(start_date, end_date, freq="D").date
        for ticker in tickers:
            ticker_dates = sorted(earnings.loc[earnings["ticker"] == ticker, "fiscal_date"].unique())
            for trade_day in smoke_days:
                future = [candidate for candidate in ticker_dates if trade_day <= candidate <= trade_day + timedelta(days=7)]
                if future:
                    tags[SampleKey(ticker, trade_day)].append("earnings_proximity")

    if not vix.empty:
        vix["observation_date"] = pd.to_datetime(vix["observation_date"]).dt.date
        vix["value"] = pd.to_numeric(vix["value"], errors="coerce")
        high_vix_days = (
            vix.sort_values("value", ascending=False)
            .head(2)["observation_date"]
            .tolist()
        )
        for trade_day in high_vix_days:
            for ticker in tickers:
                tags[SampleKey(ticker, trade_day)].append("high_vix_day")

    return tags


def choose_samples(
    ranking: pd.DataFrame,
    *,
    samples_per_category: int,
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    selected: set[SampleKey] = set()
    samples: list[dict[str, Any]] = []

    def add_rows(
        frame: pd.DataFrame,
        category: str,
        *,
        reason_field: str,
        tags_map: dict[SampleKey, list[str]] | None = None,
        allow_duplicates: bool = False,
    ) -> None:
        nonlocal samples
        for row in frame.itertuples(index=False):
            key = SampleKey(row.ticker, row.trade_date)
            if key in selected and not allow_duplicates:
                continue
            if not allow_duplicates:
                selected.add(key)
            samples.append(
                {
                    "ticker": row.ticker,
                    "trade_date": row.trade_date,
                    "category": category,
                    "selection_reason": reason_field,
                    "special_tags": sorted(set((tags_map or {}).get(key, []))),
                },
            )
            if len([item for item in samples if item["category"] == category]) >= samples_per_category:
                break

    tags_map = query_special_candidates(tickers=tickers, start_date=start_date, end_date=end_date)

    close_rank = ranking.sort_values("close_bp_13", ascending=False).head(samples_per_category * 3)
    add_rows(
        close_rank,
        "max_close_diff",
        reason_field="ranked_by_source1_vs_source3_close_bp",
        tags_map=tags_map,
    )

    volume_rank = ranking.sort_values("volume_bp_13", ascending=False).head(samples_per_category * 3)
    add_rows(
        volume_rank,
        "max_volume_diff",
        reason_field="ranked_by_source1_vs_source3_volume_bp",
        tags_map=tags_map,
    )

    baseline = ranking.loc[ranking["ticker"].isin(["AAPL", "MSFT", "NVDA"])].copy()
    baseline = baseline.sort_values(["problematic_mean_bp_13", "high_bp_13", "low_bp_13"])
    add_rows(
        baseline,
        "normal_baseline",
        reason_field="lowest_close_volume_bp_among_large_caps",
        tags_map=tags_map,
    )

    special_rows: list[dict[str, Any]] = []
    for key, tags in sorted(tags_map.items(), key=lambda item: (item[0].trade_date, item[0].ticker)):
        if not tags:
            continue
        special_rows.append(
            {
                "ticker": key.ticker,
                "trade_date": key.trade_date,
                "tag_count": len(set(tags)),
                "special_tags": sorted(set(tags)),
            },
        )
    special = pd.DataFrame(special_rows)
    if not special.empty:
        special = special.merge(ranking, on=["ticker", "trade_date"], how="inner")
        special["has_dividend"] = special["special_tags"].apply(lambda tags: int(any("dividend" in tag for tag in tags)))
        special["has_earnings"] = special["special_tags"].apply(lambda tags: int("earnings_proximity" in tags))
        special["has_high_vix"] = special["special_tags"].apply(lambda tags: int("high_vix_day" in tags))
        special.sort_values(
            ["has_dividend", "has_earnings", "has_high_vix", "tag_count", "problematic_mean_bp_13"],
            ascending=[False, False, False, False, False],
            inplace=True,
        )
        add_rows(
            special,
            "special_scenarios",
            reason_field="dividend_high_vix_or_earnings_proximity",
            tags_map=tags_map,
            allow_duplicates=True,
        )

    return samples


def frame_record(row: pd.Series, suffix: str) -> dict[str, Any]:
    return {field: None if pd.isna(row[f"{field}_{suffix}"]) else float(row[f"{field}_{suffix}"]) for field in FIELDS}


def delta_record(row: pd.Series, left_suffix: str, right_suffix: str) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field in FIELDS:
        left = row[f"{field}_{left_suffix}"]
        right = row[f"{field}_{right_suffix}"]
        record[field] = bp_delta(None if pd.isna(left) else float(left), None if pd.isna(right) else float(right))
    return record


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def pair_stats(rows: list[dict[str, Any]], pair_key: str) -> dict[str, Any]:
        result: dict[str, Any] = {"avg": {}, "median": {}}
        for field in FIELDS:
            values = [sample[pair_key][field] for sample in rows if sample[pair_key].get(field) is not None]
            result["avg"][field] = float(pd.Series(values).mean()) if values else None
            result["median"][field] = float(pd.Series(values).median()) if values else None
        return result

    by_category: dict[str, Any] = {}
    categories = sorted({sample["category"] for sample in samples})
    for category in categories:
        category_rows = [sample for sample in samples if sample["category"] == category]
        by_category[category] = {
            "sample_count": len(category_rows),
            "avg_delta_12_bp": pair_stats(category_rows, "delta_12_bp")["avg"],
            "avg_delta_13_bp": pair_stats(category_rows, "delta_13_bp")["avg"],
            "avg_delta_23_bp": pair_stats(category_rows, "delta_23_bp")["avg"],
            "median_delta_12_bp": pair_stats(category_rows, "delta_12_bp")["median"],
            "median_delta_13_bp": pair_stats(category_rows, "delta_13_bp")["median"],
            "median_delta_23_bp": pair_stats(category_rows, "delta_23_bp")["median"],
        }

    overall_rows = list(samples)
    by_category["overall"] = {
        "sample_count": len(overall_rows),
        "avg_delta_12_bp": pair_stats(overall_rows, "delta_12_bp")["avg"],
        "avg_delta_13_bp": pair_stats(overall_rows, "delta_13_bp")["avg"],
        "avg_delta_23_bp": pair_stats(overall_rows, "delta_23_bp")["avg"],
        "median_delta_12_bp": pair_stats(overall_rows, "delta_12_bp")["median"],
        "median_delta_13_bp": pair_stats(overall_rows, "delta_13_bp")["median"],
        "median_delta_23_bp": pair_stats(overall_rows, "delta_23_bp")["median"],
    }
    return by_category


def attribute_discrepancy(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    per_sample_12: list[float] = []
    per_sample_23: list[float] = []
    per_sample_13: list[float] = []
    for sample in samples:
        mean_12 = pd.Series([sample["delta_12_bp"][field] for field in PROBLEMATIC_FIELDS]).mean()
        mean_23 = pd.Series([sample["delta_23_bp"][field] for field in PROBLEMATIC_FIELDS]).mean()
        mean_13 = pd.Series([sample["delta_13_bp"][field] for field in PROBLEMATIC_FIELDS]).mean()
        per_sample_12.append(float(mean_12))
        per_sample_23.append(float(mean_23))
        per_sample_13.append(float(mean_13))

    avg_12 = float(pd.Series(per_sample_12).mean())
    avg_23 = float(pd.Series(per_sample_23).mean())
    avg_13 = float(pd.Series(per_sample_13).mean())
    share_23_gt_12 = float(pd.Series([int(d23 > d12) for d12, d23 in zip(per_sample_12, per_sample_23)]).mean())
    share_12_gt_23 = float(pd.Series([int(d12 > d23) for d12, d23 in zip(per_sample_12, per_sample_23)]).mean())

    if avg_23 > max(avg_12 * 2.0, 2.0) and share_23_gt_12 >= 0.70:
        primary = "polygon_daily_vs_minute"
        confidence = "high" if share_23_gt_12 >= 0.80 and avg_12 <= 2.0 else "medium"
        recommendation = "proceed_to_a_plus_gate"
        reasoning = (
            "Across the sampled close/volume mismatches, Source 2 vs Source 3 is materially larger "
            "than Source 1 vs Source 2, while Source 1 and Source 2 stay comparatively tight. "
            "This points to a Polygon daily-native vs minute-aggregate definition mismatch rather "
            "than a local stock_prices corruption."
        )
    elif avg_12 > max(avg_23 * 2.0, 2.0) and share_12_gt_23 >= 0.70:
        primary = "stock_prices_vs_polygon_daily"
        confidence = "high" if share_12_gt_23 >= 0.80 else "medium"
        recommendation = "fix_stock_prices_first"
        reasoning = (
            "Across the sampled close/volume mismatches, Source 1 vs Source 2 is larger than "
            "Source 2 vs Source 3. That suggests the local stock_prices layer diverges from "
            "Polygon native daily and should be repaired before advancing the gate."
        )
    else:
        primary = "mixed"
        confidence = "low"
        recommendation = "fix_stock_prices_first" if avg_12 > 5.0 else "proceed_to_a_plus_gate"
        reasoning = (
            "The sampled discrepancies do not isolate cleanly to one edge of the triangle. "
            "Both Source 1 vs Source 2 and Source 2 vs Source 3 contribute materially, so the "
            "root cause is mixed at this sample size."
        )

    attribution = {
        "primary_source_of_discrepancy": primary,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": {
            "problematic_fields": list(PROBLEMATIC_FIELDS),
            "avg_delta_12_bp_close_volume": avg_12,
            "avg_delta_23_bp_close_volume": avg_23,
            "avg_delta_13_bp_close_volume": avg_13,
            "share_samples_delta_23_gt_delta_12": share_23_gt_12,
            "share_samples_delta_12_gt_delta_23": share_12_gt_23,
        },
    }
    return attribution, recommendation


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    tickers = tuple(str(ticker).upper() for ticker in args.tickers)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    minute_daily = load_minute_daily(tickers=tickers, start_date=start_date, end_date=end_date)
    stock_prices = load_stock_prices(tickers=tickers, start_date=start_date, end_date=end_date)
    ranking = build_ranking_frame(stock_prices, minute_daily)
    selected = choose_samples(
        ranking,
        samples_per_category=args.samples_per_category,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    sample_keys = [SampleKey(item["ticker"], item["trade_date"]) for item in selected]
    polygon_daily = load_polygon_daily(sample_keys)
    source1 = stock_prices.rename(columns={field: f"{field}_s1" for field in FIELDS})
    source2 = polygon_daily.rename(columns={field: f"{field}_s2" for field in FIELDS})
    source3 = minute_daily.rename(columns={field: f"{field}_s3" for field in FIELDS})

    merged = (
        pd.DataFrame([{"ticker": key.ticker, "trade_date": key.trade_date} for key in sample_keys])
        .drop_duplicates()
        .merge(source1, on=["ticker", "trade_date"], how="left")
        .merge(source2, on=["ticker", "trade_date"], how="left")
        .merge(source3, on=["ticker", "trade_date"], how="left")
    )
    lookup = {(row.ticker, row.trade_date): row for row in merged.itertuples(index=False)}

    samples: list[dict[str, Any]] = []
    for item in selected:
        key = (item["ticker"], item["trade_date"])
        row = lookup[key]
        row_series = pd.Series(row._asdict())
        samples.append(
            {
                "ticker": item["ticker"],
                "date": item["trade_date"],
                "category": item["category"],
                "selection_reason": item["selection_reason"],
                "special_tags": item["special_tags"],
                "source1_stock_prices": frame_record(row_series, "s1"),
                "source2_polygon_daily": frame_record(row_series, "s2"),
                "source3_minute_agg": frame_record(row_series, "s3"),
                "delta_12_bp": delta_record(row_series, "s1", "s2"),
                "delta_13_bp": delta_record(row_series, "s1", "s3"),
                "delta_23_bp": delta_record(row_series, "s2", "s3"),
            },
        )

    summary_by_category = summarize_samples(samples)
    attribution, recommendation = attribute_discrepancy(samples)
    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_name": Path(__file__).name,
            "sample_window": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "tickers": list(tickers),
            "samples_per_category_target": args.samples_per_category,
            "sample_count": len(samples),
            "categories": sorted({sample["category"] for sample in samples}),
        },
        "samples": samples,
        "summary_by_category": summary_by_category,
        "attribution": attribution,
        "recommendation": recommendation,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    output_path = Path(args.report_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, json_safe(report))

    metadata = report["metadata"]
    print("Week 3 B-lite discrepancy diagnosis")
    print(f"Samples: {metadata['sample_count']}")
    for category, payload in report["summary_by_category"].items():
        print(
            f"- {category}: n={payload['sample_count']}, "
            f"avg close delta_12={payload['avg_delta_12_bp']['close']}, "
            f"avg close delta_23={payload['avg_delta_23_bp']['close']}, "
            f"avg volume delta_12={payload['avg_delta_12_bp']['volume']}, "
            f"avg volume delta_23={payload['avg_delta_23_bp']['volume']}",
        )
    print(f"Attribution: {report['attribution']['primary_source_of_discrepancy']} ({report['attribution']['confidence']})")
    print(f"Recommendation: {report['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
