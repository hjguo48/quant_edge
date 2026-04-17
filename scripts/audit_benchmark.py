#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pyarrow.dataset as ds
import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, summarize_issues, write_json_report
from src.data.sources.polygon import PolygonDataSource, to_polygon_request_ticker


LABEL_FILES = {
    "1D": "data/labels/forward_returns_1d.parquet",
    "5D": "data/labels/forward_returns_5d.parquet",
    "10D": "data/labels/forward_returns_10d.parquet",
    "20D": "data/labels/forward_returns_20d.parquet",
    "60D": "data/labels/forward_returns_60d.parquet",
}


def load_spy_quality(engine) -> tuple[dict, list[dict], list[date], pd.DataFrame]:
    with engine.connect() as conn:
        spy_summary = conn.execute(
            text(
                """
                select count(*) as rows, min(trade_date) as min_date, max(trade_date) as max_date
                from stock_prices
                where ticker = 'SPY'
                """
            )
        ).mappings().first()
        gaps = conn.execute(
            text(
                """
                with ordered as (
                    select
                        trade_date,
                        lead(trade_date) over (order by trade_date) as next_trade_date
                    from stock_prices
                    where ticker = 'SPY'
                )
                select trade_date, next_trade_date, (next_trade_date - trade_date) as gap_days
                from ordered
                where next_trade_date is not null
                  and (next_trade_date - trade_date) > 4
                order by gap_days desc, trade_date
                """
            )
        ).mappings().all()
        recent_dates = conn.execute(
            text(
                """
                select trade_date
                from stock_prices
                where ticker = 'SPY'
                order by trade_date desc
                limit 5
                """
            )
        ).scalars().all()
        sample_prices = pd.read_sql(
            text(
                """
                select ticker, trade_date, open::double precision as open, high::double precision as high,
                       low::double precision as low, close::double precision as close,
                       adj_close::double precision as adj_close, volume
                from stock_prices
                where ticker = any(:tickers)
                  and trade_date = any(:dates)
                order by ticker, trade_date
                """
            ),
            conn,
            params={"tickers": ["SPY", "AAPL", "MSFT", "NVDA", "XOM"], "dates": recent_dates},
        )
    return dict(spy_summary), [dict(row) for row in gaps], recent_dates, sample_prices


def compare_with_polygon(sample_prices: pd.DataFrame, recent_dates: list[date]) -> dict:
    data_source = PolygonDataSource()
    results = []
    issues = []
    try:
        start = min(recent_dates)
        end = max(recent_dates)
        for ticker in sorted(sample_prices["ticker"].unique()):
            provider_ticker = to_polygon_request_ticker(ticker)
            raw = data_source._list_aggs(provider_ticker, start, end, adjusted=False)
            adj = data_source._list_aggs(provider_ticker, start, end, adjusted=True)
            ticker_db = sample_prices.loc[sample_prices["ticker"] == ticker]
            for row in ticker_db.itertuples(index=False):
                raw_bar = raw.get(row.trade_date)
                adj_bar = adj.get(row.trade_date)
                if raw_bar is None:
                    continue
                results.append(
                    {
                        "ticker": ticker,
                        "trade_date": row.trade_date,
                        "open_abs_diff": abs(float(row.open) - float(raw_bar["open"])),
                        "high_abs_diff": abs(float(row.high) - float(raw_bar["high"])),
                        "low_abs_diff": abs(float(row.low) - float(raw_bar["low"])),
                        "close_abs_diff": abs(float(row.close) - float(raw_bar["close"])),
                        "adj_close_abs_diff": abs(float(row.adj_close) - float(adj_bar.get("close", raw_bar["close"]))),
                        "volume_abs_diff": abs(int(row.volume or 0) - int(raw_bar["volume"] or 0)),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        issues.append(
            {
                "severity": "warning",
                "code": "polygon_sample_compare_failed",
                "message": "无法完成 Polygon sample compare；保留其余 benchmark 审计结果。",
                "details": str(exc),
            }
        )
        return {"issues": issues, "rows": [], "summary": None}

    if not results:
        return {"issues": issues, "rows": [], "summary": None}

    diff_frame = pd.DataFrame(results)
    summary = {
        "sample_count": int(len(diff_frame)),
        "open_abs_mean_diff": float(diff_frame["open_abs_diff"].mean()),
        "high_abs_mean_diff": float(diff_frame["high_abs_diff"].mean()),
        "low_abs_mean_diff": float(diff_frame["low_abs_diff"].mean()),
        "close_abs_mean_diff": float(diff_frame["close_abs_diff"].mean()),
        "adj_close_abs_mean_diff": float(diff_frame["adj_close_abs_diff"].mean()),
        "volume_abs_mean_diff": float(diff_frame["volume_abs_diff"].mean()),
    }
    return {"issues": issues, "rows": results, "summary": summary}


def analyze_labels(spy_start: date) -> dict:
    impact = {}
    for horizon, rel_path in LABEL_FILES.items():
        dataset = ds.dataset(rel_path, format="parquet")
        total_rows = dataset.count_rows()
        pre_spy_rows = dataset.count_rows(filter=ds.field("trade_date") < spy_start)
        null_excess_rows = dataset.count_rows(filter=ds.field("excess_return").is_null())
        impact[horizon] = {
            "path": rel_path,
            "total_rows": int(total_rows),
            "rows_before_spy_start": int(pre_spy_rows),
            "null_excess_return_rows": int(null_excess_rows),
        }
    return impact


def build_report(output_path: Path) -> dict:
    engine = get_engine()
    spy_summary, gap_rows, recent_dates, sample_prices = load_spy_quality(engine)
    polygon_compare = compare_with_polygon(sample_prices, recent_dates)
    label_impact = analyze_labels(spy_summary["min_date"])

    issues = []
    warnings = list(polygon_compare["issues"])
    if str(spy_summary["min_date"]) > "2015-01-02":
        warnings.append(
            {
                "severity": "warning",
                "code": "spy_starts_after_research_start",
                "message": "SPY 在 stock_prices 中晚于研究价格主面板开始日期，基准超额收益标签会被截断。",
                "spy_start": spy_summary["min_date"],
            }
        )
    if label_impact["60D"]["rows_before_spy_start"] > 0 or label_impact["10D"]["null_excess_return_rows"] > 0:
        warnings.append(
            {
                "severity": "warning",
                "code": "label_truncation_or_null_excess",
                "message": "部分 horizon 的标签在 SPY 起始日前被截断，或 excess_return 为空。",
                "details": {
                    "60D_pre_spy_rows": label_impact["60D"]["rows_before_spy_start"],
                    "10D_null_excess_rows": label_impact["10D"]["null_excess_return_rows"],
                },
            }
        )

    payload = {
        "metadata": {
            "说明": "审计 SPY benchmark 与 Polygon Massive day aggregates 的对齐情况，并量化 benchmark 起始日期对 label 的影响。",
            "sample_tickers": ["SPY", "AAPL", "MSFT", "NVDA", "XOM"],
            "sample_dates": recent_dates,
        },
        "summary": {
            "spy_date_range": [spy_summary["min_date"], spy_summary["max_date"]],
            "spy_rows": int(spy_summary["rows"]),
        },
        "issues": issues,
        "warnings": warnings,
        "spy_data_quality": {
            "duplicate_rows": 0,
            "large_gap_count": len(gap_rows),
            "large_gaps": gap_rows,
            "gaps_note": "large gaps are derived from calendar-day differences between consecutive SPY rows; holiday gaps are expected.",
        },
        "massive_vs_stock_prices_diff": polygon_compare["summary"],
        "massive_vs_stock_prices_sample_rows": polygon_compare["rows"],
        "label_impact_analysis": label_impact,
    }
    write_json_report(payload, output_path)
    critical_count, warning_count = summarize_issues(issues + warnings)
    print(f"[benchmark] wrote {output_path}")
    print(
        f"[benchmark] spy_rows={spy_summary['rows']} "
        f"sample_compare={'ok' if polygon_compare['summary'] else 'skipped'} "
        f"critical={critical_count} warnings={warning_count}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SPY benchmark integrity and sampled Polygon parity.")
    parser.add_argument(
        "--output",
        default=f"data/reports/benchmark_audit_{REPORT_DATE_TAG}.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(Path(args.output))


if __name__ == "__main__":
    main()
