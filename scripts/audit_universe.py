#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, summarize_issues, write_json_report
from src.universe.active import resolve_active_universe


def build_report(output_path: Path) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        latest_trade_date = conn.execute(text("select max(trade_date) from stock_prices")).scalar()
        membership_rows = conn.execute(
            text(
                """
                select effective_date, count(distinct ticker) as ticker_count
                from universe_membership
                group by effective_date
                order by effective_date
                """
            )
        ).mappings().all()
        stock_counts = conn.execute(
            text(
                """
                select
                    count(*) as total_stocks,
                    count(*) filter (where delist_date is not null) as delisted_count
                from stocks
                """
            )
        ).mappings().first()
        delisted_rows = conn.execute(
            text(
                """
                select ticker, delist_date, delist_reason
                from stocks
                where delist_date is not null
                order by delist_date nulls last, ticker
                """
            )
        ).mappings().all()
        research_rows = conn.execute(
            text(
                """
                select distinct sp.ticker
                from stock_prices sp
                join stocks s on s.ticker = sp.ticker
                where sp.ticker <> 'SPY'
                order by sp.ticker
                """
            )
        ).scalars().all()
        membership_bounds = conn.execute(
            text(
                """
                select min(effective_date) as min_date, max(effective_date) as max_date, count(*) as rows, count(distinct ticker) as tickers
                from universe_membership
                """
            )
        ).mappings().first()

    live_universe, live_source = resolve_active_universe(
        latest_trade_date,
        as_of=datetime.now(timezone.utc),
    )
    research_universe = [ticker for ticker in research_rows if ticker not in {"SPY"}]

    live_set = set(live_universe)
    research_set = set(research_universe)
    live_only = sorted(live_set - research_set)
    research_only = sorted(research_set - live_set)

    issues = []
    warnings = []
    if membership_bounds["min_date"] and str(membership_bounds["min_date"]) > "2025-01-01":
        warnings.append(
            {
                "severity": "warning",
                "code": "membership_history_missing",
                "message": "universe_membership 仅覆盖 2026 年，无法作为 2015-2025 的历史 PIT 成分表。",
                "coverage_start": membership_bounds["min_date"],
                "coverage_end": membership_bounds["max_date"],
            }
        )
    if research_only:
        warnings.append(
            {
                "severity": "warning",
                "code": "live_research_universe_gap",
                "message": "研究全量 universe 与当前 live active universe 存在明显差异。",
                "research_only_count": len(research_only),
                "live_only_count": len(live_only),
            }
        )

    survivorship_level = "high" if membership_bounds["min_date"] and str(membership_bounds["min_date"]) > "2020-01-01" else "medium"
    survivorship_note = (
        "如果直接用当前 universe_membership 回放历史，将出现显著 survivorship / constituent drift 风险；"
        "当前研究更多依赖 stock_prices+stocks 的宽 universe，而不是历史 PIT membership。"
    )

    payload = {
        "metadata": {
            "说明": "审计 live universe 解析路径与研究 universe 的差异，并评估历史 membership 缺失带来的偏差风险。",
            "live_universe_source": live_source,
        },
        "summary": {
            "latest_trade_date": latest_trade_date,
            "membership_rows": int(membership_bounds["rows"] or 0),
            "membership_tickers": int(membership_bounds["tickers"] or 0),
            "membership_date_range": [membership_bounds["min_date"], membership_bounds["max_date"]],
            "stocks_total": int(stock_counts["total_stocks"]),
            "stocks_delisted": int(stock_counts["delisted_count"]),
            "research_universe_count": len(research_set),
            "live_universe_count": len(live_set),
        },
        "issues": issues,
        "warnings": warnings,
        "membership_coverage": {str(row["effective_date"]): int(row["ticker_count"]) for row in membership_rows},
        "live_vs_research_diff": {
            "live_source": live_source,
            "live_only": live_only,
            "research_only": research_only,
        },
        "delisted_tickers": [
            {
                "ticker": row["ticker"],
                "delist_date": row["delist_date"],
                "delist_reason": row["delist_reason"],
            }
            for row in delisted_rows
        ],
        "survivorship_bias_risk": {
            "level": survivorship_level,
            "note": survivorship_note,
        },
    }
    write_json_report(payload, output_path)
    critical_count, warning_count = summarize_issues(issues + warnings)
    print(f"[universe] wrote {output_path}")
    print(
        f"[universe] membership_tickers={membership_bounds['tickers']} "
        f"research={len(research_set)} live={len(live_set)} "
        f"critical={critical_count} warnings={warning_count}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit research vs live universe coverage and survivorship risk.")
    parser.add_argument(
        "--output",
        default=f"data/reports/universe_audit_{REPORT_DATE_TAG}.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(Path(args.output))


if __name__ == "__main__":
    main()
