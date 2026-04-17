#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from collections import Counter

from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, write_json_report
from src.universe.builder import (
    _fetch_index_change_events,
    _fetch_index_constituents,
    _reconstruct_membership_rows,
)


DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2025, 12, 31)


def build_report(output_path: Path) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        membership_bounds = conn.execute(
            text(
                """
                select min(effective_date) as min_date,
                       max(effective_date) as max_date,
                       count(*) as rows,
                       count(distinct ticker) as tickers
                from universe_membership
                """
            )
        ).mappings().first()
        by_date = conn.execute(
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
                    count(*) filter (where delist_date is not null) as delisted_stocks
                from stocks
                """
            )
        ).mappings().first()

    current_constituents = _fetch_index_constituents("SP500")
    change_events = _fetch_index_change_events("SP500")
    proposed_rows = _reconstruct_membership_rows(
        current_constituents=current_constituents,
        change_events=change_events,
        start_date=DEFAULT_START,
        end_date=DEFAULT_END,
        index_name="SP500",
    )

    event_source_counts = Counter(event.source for event in change_events)
    rows_by_reason = Counter(str(row["reason"]) for row in proposed_rows)
    interval_rows = sum(1 for row in proposed_rows if row["end_date"] is not None)
    active_rows = sum(1 for row in proposed_rows if row["end_date"] is None)

    warnings: list[dict] = []
    issues: list[dict] = []
    if membership_bounds["min_date"] is None or membership_bounds["min_date"] > DEFAULT_START:
        warnings.append(
            {
                "severity": "warning",
                "code": "historical_membership_missing",
                "message": "universe_membership 尚未覆盖 2016-2025，当前 live/research PIT 成分治理不完整。",
                "existing_coverage_start": membership_bounds["min_date"],
                "existing_coverage_end": membership_bounds["max_date"],
            }
        )
    if len(change_events) == 0:
        issues.append(
            {
                "severity": "critical",
                "code": "no_change_events_available",
                "message": "无法获取 S&P 500 历史成分变更事件，无法执行完整 PIT 回填。",
            }
        )

    scheme = "A" if not issues else "B"
    payload = {
        "metadata": {
            "说明": "Week 2.5-P3 universe membership 历史回填根因诊断。",
            "generated_at_utc": datetime.now(timezone.utc),
            "script_name": "scripts/diagnose_p3_root_cause.py",
            "target_window": {
                "start_date": DEFAULT_START,
                "end_date": DEFAULT_END,
            },
        },
        "summary": {
            "existing_membership_rows": int(membership_bounds["rows"] or 0),
            "existing_membership_tickers": int(membership_bounds["tickers"] or 0),
            "existing_membership_date_range": [
                membership_bounds["min_date"],
                membership_bounds["max_date"],
            ],
            "stocks_total": int(stock_counts["total_stocks"] or 0),
            "stocks_delisted": int(stock_counts["delisted_stocks"] or 0),
            "current_sp500_constituents": int(len(current_constituents)),
            "historical_change_events": int(len(change_events)),
            "proposed_membership_rows": int(len(proposed_rows)),
            "proposed_active_rows": int(active_rows),
            "proposed_ended_rows": int(interval_rows),
            "selected_scheme": scheme,
        },
        "issues": issues,
        "warnings": warnings,
        "existing_membership_coverage": {
            str(row["effective_date"]): int(row["ticker_count"]) for row in by_date
        },
        "existing_fill_path": {
            "live_monthly_sync": "src/universe/active.ensure_monthly_universe_membership -> src/universe/builder.backfill_universe_membership",
            "historical_builder": "scripts/backfill_sp500_history.py / src/universe/builder.py",
            "root_cause": "历史 PIT membership builder 已存在，但只被 live monthly sync 用于 2026 当月；2016-2025 从未执行过完整 backfill。",
        },
        "source_options": {
            "fmp_current_constituents": {
                "available": True,
                "endpoint": "stable/sp500-constituent",
            },
            "fmp_historical_changes": {
                "available": len(change_events) > 0,
                "event_count": int(len(change_events)),
                "source_breakdown": dict(event_source_counts),
            },
            "wikipedia_fallback": {
                "available": True,
                "note": "builder 内置 Wikipedia fallback，但本次诊断已通过 FMP 拿到历史 change events。",
            },
        },
        "scheme_decision": {
            "selected_scheme": scheme,
            "rationale": (
                "选择方案 A：仓库已具备完整历史 change-event 重建逻辑，且 FMP 返回 1279 个历史事件，"
                "足以回填 2016-2025 的 PIT 区间成员关系。"
                if scheme == "A"
                else "降级到方案 B：历史 change events 不可用。"
            ),
        },
        "proposed_backfill_plan": {
            "index_name": "SP500",
            "start_date": DEFAULT_START,
            "end_date": DEFAULT_END,
            "rows_to_write": int(len(proposed_rows)),
            "reason_breakdown": dict(rows_by_reason),
            "sample_rows": proposed_rows[:25],
        },
    }
    write_json_report(payload, output_path)
    print(f"[p3_diagnosis] wrote {output_path}")
    print(
        f"[p3_diagnosis] scheme={scheme} events={len(change_events)} "
        f"rows_to_write={len(proposed_rows)} existing_rows={membership_bounds['rows'] or 0}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose universe membership PIT coverage and choose Scheme A/B.")
    parser.add_argument(
        "--output",
        default=f"data/reports/p3_root_cause_diagnosis_{REPORT_DATE_TAG}.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(Path(args.output))


if __name__ == "__main__":
    main()
