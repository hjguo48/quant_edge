#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, summarize_issues, write_json_report


def build_report(output_path: Path) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        summary = conn.execute(
            text(
                """
                select
                    count(*) as total_rows,
                    count(distinct ticker) as total_tickers,
                    min(trade_date) as min_date,
                    max(trade_date) as max_date
                from stock_prices
                """
            )
        ).mappings().first()
        lag_distribution = [
            dict(row)
            for row in conn.execute(
                text(
                    """
                    select
                        ((knowledge_time at time zone 'UTC')::date - trade_date) as lag_days,
                        count(*) as row_count
                    from stock_prices
                    group by 1
                    order by 1
                    """
                )
            ).mappings()
        ]
        anomalies = pd.read_sql(
            text(
                """
                select
                    ticker,
                    trade_date,
                    close::double precision as close,
                    adj_close::double precision as adj_close,
                    volume,
                    knowledge_time,
                    source,
                    (adj_close / nullif(close, 0))::double precision as adj_ratio
                from stock_prices
                where close is not null
                  and adj_close is not null
                  and close <> 0
                  and ((adj_close / close) > 2.0 or (adj_close / close) < 0.1)
                order by ticker, trade_date
                """
            ),
            conn,
        )
        split_actions = pd.read_sql(
            text(
                """
                select ticker, ex_date, ratio::double precision as ratio
                from corporate_actions
                where action_type = 'split'
                order by ticker, ex_date
                """
            ),
            conn,
        )
        zero_volume = pd.read_sql(
            text(
                """
                select ticker, trade_date, open, high, low, close, adj_close, volume, knowledge_time
                from stock_prices
                where coalesce(volume, 0) = 0
                order by ticker, trade_date
                """
            ),
            conn,
        )
        pit_violations = pd.read_sql(
            text(
                """
                select ticker, trade_date, knowledge_time, source
                from stock_prices
                where ((knowledge_time at time zone 'UTC')::date) <= trade_date
                order by trade_date, ticker
                """
            ),
            conn,
        )
        spy_range = conn.execute(
            text(
                """
                select count(*) as rows, min(trade_date) as min_date, max(trade_date) as max_date
                from stock_prices
                where ticker = 'SPY'
                """
            )
        ).mappings().first()
        coverage_rows = conn.execute(
            text(
                """
                select sp.ticker, min(sp.trade_date) as first_available_date
                from stock_prices sp
                join stocks s on s.ticker = sp.ticker
                where sp.ticker <> 'SPY'
                group by sp.ticker
                order by sp.ticker
                """
            )
        ).mappings().all()

    split_actions["ex_date"] = pd.to_datetime(split_actions["ex_date"])
    anomalies["trade_date"] = pd.to_datetime(anomalies["trade_date"])
    enriched_anomalies = []
    uncovered_count = 0
    action_groups = {ticker: group.reset_index(drop=True) for ticker, group in split_actions.groupby("ticker")}
    for row in anomalies.itertuples(index=False):
        nearest = None
        group = action_groups.get(row.ticker)
        if group is not None and not group.empty:
            deltas = (group["ex_date"] - row.trade_date).abs().dt.days
            idx = int(deltas.idxmin())
            candidate = group.loc[idx]
            nearest = {
                "action_type": "split",
                "ex_date": candidate["ex_date"],
                "ratio": candidate["ratio"],
                "day_distance": int(deltas.loc[idx]),
            }
        if nearest is None or nearest["day_distance"] > 14:
            uncovered_count += 1
        enriched_anomalies.append(
            {
                "ticker": row.ticker,
                "trade_date": row.trade_date,
                "close": row.close,
                "adj_close": row.adj_close,
                "adj_close_to_close_ratio": row.adj_ratio,
                "volume": row.volume,
                "knowledge_time": row.knowledge_time,
                "nearest_corporate_action": nearest,
            }
        )

    lag_counter = Counter()
    for row in lag_distribution:
        lag_counter[str(row["lag_days"])] = int(row["row_count"])

    issues = []
    warnings = []
    if not pit_violations.empty:
        issues.append(
            {
                "severity": "critical",
                "code": "pit_violation_stock_prices",
                "message": "发现 knowledge_time 日期不晚于 trade_date 的价格行。",
                "count": int(len(pit_violations)),
            }
        )
    if uncovered_count:
        warnings.append(
            {
                "severity": "warning",
                "code": "uncovered_split_like_anomalies",
                "message": "存在 adj_close/close 极端比率，但 14 天内无对应 split corporate action。",
                "count": int(uncovered_count),
            }
        )
    if len(zero_volume):
        warnings.append(
            {
                "severity": "warning",
                "code": "zero_volume_rows",
                "message": "发现 volume=0 的价格行；需确认是否停牌、ghost bar 或数据缺失。",
                "count": int(len(zero_volume)),
            }
        )
    if str(spy_range["min_date"]) > "2015-01-02":
        warnings.append(
            {
                "severity": "warning",
                "code": "benchmark_starts_late",
                "message": "SPY 基准从 2016-04 起才完整可用，会截断早期 excess-return labels。",
                "spy_start": spy_range["min_date"],
            }
        )

    payload = {
        "metadata": {
            "说明": "审计 stock_prices、corporate_actions 与 SPY 基准时间覆盖，识别价格真值与 PIT 风险。",
            "price_rule": "当前 stock_prices.close 为原始日线 close，adj_close 直接取 Polygon adjusted close 快照；corporate_actions 表不反向驱动 adj_close。",
            "knowledge_time_rule": "代码中的 Polygon historical 模式使用 trade_date + 1 calendar day 的 market close 时间，而不是严格的下一个工作日。",
        },
        "summary": {
            "total_rows": int(summary["total_rows"]),
            "total_tickers": int(summary["total_tickers"]),
            "date_range": [summary["min_date"], summary["max_date"]],
            "lag_distribution": dict(lag_counter),
        },
        "issues": issues,
        "warnings": warnings,
        "split_anomalies": enriched_anomalies,
        "zero_volume_rows": zero_volume.to_dict(orient="records"),
        "pit_violations": pit_violations.to_dict(orient="records"),
        "corporate_action_coverage": {
            "split_rows": int(len(split_actions)),
            "anomaly_count": int(len(enriched_anomalies)),
            "anomalies_without_near_split_14d": int(uncovered_count),
        },
        "spy_summary": {
            "rows": int(spy_range["rows"]),
            "date_range": [spy_range["min_date"], spy_range["max_date"]],
        },
        "sp500_coverage": {row["ticker"]: row["first_available_date"] for row in coverage_rows},
    }
    write_json_report(payload, output_path)
    critical_count, warning_count = summarize_issues(issues + warnings)
    print(f"[price_truth] wrote {output_path}")
    print(
        f"[price_truth] rows={summary['total_rows']} tickers={summary['total_tickers']} "
        f"split_anomalies={len(enriched_anomalies)} zero_volume={len(zero_volume)} "
        f"critical={critical_count} warnings={warning_count}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit stock_prices truth, PIT timing, and corporate-action consistency.")
    parser.add_argument(
        "--output",
        default=f"data/reports/price_truth_audit_{REPORT_DATE_TAG}.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(Path(args.output))


if __name__ == "__main__":
    main()
