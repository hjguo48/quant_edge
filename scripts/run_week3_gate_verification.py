from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import sqlalchemy as sa
import yaml

from src.data.db.session import get_engine
from src.features.intraday import INTRADAY_FEATURE_NAMES

DEFAULT_START_DATE = date(2016, 4, 20)
DEFAULT_END_DATE = date(2026, 4, 17)
DEFAULT_REPORT_OUTPUT = Path(f"data/reports/week3_gate_verification_{date.today().strftime('%Y%m%d')}.json")
DEFAULT_LINEAGE_PATH = Path("configs/research/data_lineage.yaml")


@dataclass(frozen=True)
class FeatureCoverageMetric:
    feature_name: str
    expected: int
    actual: int
    coverage: float | None
    missing_examples: list[dict[str, str]]


@dataclass(frozen=True)
class FeatureQualityMetric:
    feature_name: str
    total_rows: int
    missing_rows: int
    missing_rate: float | None
    evaluated_rows: int
    outlier_rows: int
    outlier_rate: float | None
    lag_rule_documented: bool
    missing_examples: list[dict[str, Any]]
    outlier_examples: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 3 gate verification for intraday features.")
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--lineage-path", type=Path, default=DEFAULT_LINEAGE_PATH)
    return parser.parse_args()


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    return value


def load_lag_rule_documentation(lineage_path: Path) -> dict[str, Any]:
    if not lineage_path.exists():
        return {
            "lineage_path": str(lineage_path),
            "exists": False,
            "documented_features": [],
            "knowledge_time_rule": None,
            "rule_mentions_t_plus_one": False,
        }
    payload = yaml.safe_load(lineage_path.read_text()) or {}
    intraday_layer = payload.get("intraday_feature_layer") or {}
    knowledge_time_rule = intraday_layer.get("knowledge_time_rule")
    documented_features = [str(name) for name in intraday_layer.get("applies_to", [])]
    rule_text = str(knowledge_time_rule or "").lower()
    rule_mentions_t_plus_one = ("t+1" in rule_text) or ("next xnys session close" in rule_text) or (
        "business day close" in rule_text
    )
    return {
        "lineage_path": str(lineage_path),
        "exists": True,
        "documented_features": documented_features,
        "knowledge_time_rule": knowledge_time_rule,
        "rule_mentions_t_plus_one": rule_mentions_t_plus_one,
    }


def evaluate_gate1(feature_metrics: list[FeatureCoverageMetric], threshold: float = 0.95) -> dict[str, Any]:
    feature_payload: dict[str, Any] = {}
    failing_examples: list[dict[str, Any]] = []
    all_pass = True
    for metric in feature_metrics:
        coverage_pass = metric.coverage is not None and metric.coverage >= threshold
        all_pass = all_pass and coverage_pass
        feature_payload[metric.feature_name] = {
            "expected": metric.expected,
            "actual": metric.actual,
            "coverage": metric.coverage,
            "pass": coverage_pass,
            "missing_examples": metric.missing_examples,
        }
        if not coverage_pass:
            for example in metric.missing_examples:
                failing_examples.append(
                    {
                        "feature_name": metric.feature_name,
                        "issue_type": "missing_feature_row",
                        **example,
                    },
                )
    return {
        "pass": all_pass,
        "threshold": threshold,
        "features": feature_payload,
        "failing_examples": failing_examples,
    }


def evaluate_gate2(blocker_count: int, warning_count: int, sample_blockers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pass": blocker_count == 0,
        "blocker_count_last_30d": blocker_count,
        "warning_count_last_30d": warning_count,
        "sample_blockers": sample_blockers,
    }


def evaluate_gate3(
    feature_metrics: list[FeatureQualityMetric],
    *,
    missing_threshold: float = 0.05,
    outlier_threshold: float = 0.01,
) -> dict[str, Any]:
    feature_payload: dict[str, Any] = {}
    failing_examples: list[dict[str, Any]] = []
    all_pass = True
    for metric in feature_metrics:
        missing_pass = metric.missing_rate is not None and metric.missing_rate < missing_threshold
        outlier_pass = metric.outlier_rate is not None and metric.outlier_rate < outlier_threshold
        documented_pass = metric.lag_rule_documented
        feature_pass = missing_pass and outlier_pass and documented_pass
        all_pass = all_pass and feature_pass
        feature_payload[metric.feature_name] = {
            "total_rows": metric.total_rows,
            "missing_rows": metric.missing_rows,
            "missing_rate": metric.missing_rate,
            "missing_rate_pass": missing_pass,
            "evaluated_rows": metric.evaluated_rows,
            "outlier_rows": metric.outlier_rows,
            "outlier_rate": metric.outlier_rate,
            "outlier_rate_pass": outlier_pass,
            "lag_rule_documented": documented_pass,
            "pass": feature_pass,
            "missing_examples": metric.missing_examples,
            "outlier_examples": metric.outlier_examples,
        }
        if not missing_pass:
            for example in metric.missing_examples:
                failing_examples.append(
                    {
                        "feature_name": metric.feature_name,
                        "issue_type": "missing_or_filled",
                        **example,
                    },
                )
        if not outlier_pass:
            for example in metric.outlier_examples:
                failing_examples.append(
                    {
                        "feature_name": metric.feature_name,
                        "issue_type": "outlier",
                        **example,
                    },
                )
        if not documented_pass:
            failing_examples.append(
                {
                    "feature_name": metric.feature_name,
                    "issue_type": "lag_rule_undocumented",
                },
            )
    return {
        "pass": all_pass,
        "missing_threshold": missing_threshold,
        "outlier_threshold": outlier_threshold,
        "features": feature_payload,
        "failing_examples": failing_examples,
    }


def _expected_pairs_query() -> sa.TextClause:
    return sa.text(
        """
        with completed_days as (
            select trading_date
            from minute_backfill_state
            where lower(status) = 'completed'
              and trading_date between :start_date and :end_date
        ),
        expected_pairs as (
            select distinct um.ticker, cd.trading_date
            from completed_days cd
            join universe_membership um
              on um.index_name = :index_name
             and um.effective_date <= cd.trading_date
             and (um.end_date is null or um.end_date > cd.trading_date)
        )
        select count(*) as expected_pairs
        from expected_pairs
        """
    )


def fetch_expected_pair_count(conn: sa.Connection, *, start_date: date, end_date: date, index_name: str = "SP500") -> int:
    value = conn.execute(
        _expected_pairs_query(),
        {"start_date": start_date, "end_date": end_date, "index_name": index_name},
    ).scalar()
    return int(value or 0)


def fetch_actual_pair_counts(
    conn: sa.Connection,
    *,
    start_date: date,
    end_date: date,
    features: tuple[str, ...],
    index_name: str = "SP500",
) -> dict[str, int]:
    query = sa.text(
        """
        with completed_days as (
            select trading_date
            from minute_backfill_state
            where lower(status) = 'completed'
              and trading_date between :start_date and :end_date
        ),
        expected_pairs as (
            select distinct um.ticker, cd.trading_date
            from completed_days cd
            join universe_membership um
              on um.index_name = :index_name
             and um.effective_date <= cd.trading_date
             and (um.end_date is null or um.end_date > cd.trading_date)
        )
        select
            f.feature_name,
            count(distinct (f.ticker, f.calc_date)) as actual_pairs
        from feature_store f
        join expected_pairs e
          on e.ticker = f.ticker
         and e.trading_date = f.calc_date
        where f.feature_name = any(:features)
        group by f.feature_name
        """
    )
    rows = conn.execute(
        query,
        {"start_date": start_date, "end_date": end_date, "index_name": index_name, "features": list(features)},
    ).mappings()
    return {str(row["feature_name"]): int(row["actual_pairs"] or 0) for row in rows}


def fetch_missing_coverage_examples(
    conn: sa.Connection,
    *,
    feature_name: str,
    start_date: date,
    end_date: date,
    index_name: str = "SP500",
    limit: int = 25,
) -> list[dict[str, str]]:
    query = sa.text(
        """
        with completed_days as (
            select trading_date
            from minute_backfill_state
            where lower(status) = 'completed'
              and trading_date between :start_date and :end_date
        ),
        expected_pairs as (
            select distinct um.ticker, cd.trading_date
            from completed_days cd
            join universe_membership um
              on um.index_name = :index_name
             and um.effective_date <= cd.trading_date
             and (um.end_date is null or um.end_date > cd.trading_date)
        ),
        actual_pairs as (
            select distinct ticker, calc_date
            from feature_store
            where feature_name = :feature_name
              and calc_date between :start_date and :end_date
        )
        select e.ticker, e.trading_date
        from expected_pairs e
        left join actual_pairs a
          on a.ticker = e.ticker
         and a.calc_date = e.trading_date
        where a.ticker is null
        order by e.trading_date, e.ticker
        limit :limit_rows
        """
    )
    rows = conn.execute(
        query,
        {
            "feature_name": feature_name,
            "start_date": start_date,
            "end_date": end_date,
            "index_name": index_name,
            "limit_rows": limit,
        },
    ).mappings()
    return [{"ticker": str(row["ticker"]), "trade_date": row["trading_date"].isoformat()} for row in rows]


def collect_gate1_metrics(
    conn: sa.Connection,
    *,
    start_date: date,
    end_date: date,
    features: tuple[str, ...],
    index_name: str = "SP500",
) -> list[FeatureCoverageMetric]:
    expected_pairs = fetch_expected_pair_count(conn, start_date=start_date, end_date=end_date, index_name=index_name)
    actual_pairs = fetch_actual_pair_counts(
        conn,
        start_date=start_date,
        end_date=end_date,
        features=features,
        index_name=index_name,
    )
    metrics: list[FeatureCoverageMetric] = []
    for feature_name in features:
        actual = int(actual_pairs.get(feature_name, 0))
        coverage = (actual / expected_pairs) if expected_pairs else None
        missing_examples = []
        if coverage is None or coverage < 0.95:
            missing_examples = fetch_missing_coverage_examples(
                conn,
                feature_name=feature_name,
                start_date=start_date,
                end_date=end_date,
                index_name=index_name,
            )
        metrics.append(
            FeatureCoverageMetric(
                feature_name=feature_name,
                expected=expected_pairs,
                actual=actual,
                coverage=coverage,
                missing_examples=missing_examples,
            ),
        )
    return metrics


def fetch_recent_reconciliation_counts(
    conn: sa.Connection,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    date_query = sa.text(
        """
        select trading_date
        from minute_backfill_state
        where lower(status) = 'completed'
          and trading_date between :start_date and :end_date
        order by trading_date desc
        limit 30
        """
    )
    recent_dates = [row[0] for row in conn.execute(date_query, {"start_date": start_date, "end_date": end_date}).all()]
    if not recent_dates:
        return {"blocker_count": 0, "warning_count": 0, "sample_blockers": [], "recent_dates": []}
    event_query = sa.text(
        """
        select lower(severity) as severity, count(*) as rows
        from price_reconciliation_events
        where trade_date = any(:recent_dates)
        group by lower(severity)
        """
    )
    counts = {str(row["severity"]): int(row["rows"] or 0) for row in conn.execute(event_query, {"recent_dates": recent_dates}).mappings()}
    blocker_query = sa.text(
        """
        select ticker, trade_date, field, delta_bp, severity
        from price_reconciliation_events
        where trade_date = any(:recent_dates)
          and lower(severity) = 'blocker'
        order by trade_date desc, ticker asc
        limit 25
        """
    )
    sample_blockers = [
        {
            "ticker": str(row["ticker"]),
            "trade_date": row["trade_date"].isoformat(),
            "field": str(row["field"]),
            "delta_bp": float(row["delta_bp"]) if row["delta_bp"] is not None else None,
            "severity": str(row["severity"]),
        }
        for row in conn.execute(blocker_query, {"recent_dates": recent_dates}).mappings()
    ]
    return {
        "blocker_count": int(counts.get("blocker", 0)),
        "warning_count": int(counts.get("warning", 0)),
        "sample_blockers": sample_blockers,
        "recent_dates": [trade_date.isoformat() for trade_date in sorted(recent_dates)],
    }


def fetch_quality_metric(
    conn: sa.Connection,
    *,
    feature_name: str,
    start_date: date,
    end_date: date,
    lag_rule_documented: bool,
) -> FeatureQualityMetric:
    missing_query = sa.text(
        """
        select
            count(*) as total_rows,
            count(*) filter (where feature_value is null or coalesce(is_filled, false)) as missing_rows
        from feature_store
        where feature_name = :feature_name
          and calc_date between :start_date and :end_date
        """
    )
    missing_row = conn.execute(
        missing_query,
        {"feature_name": feature_name, "start_date": start_date, "end_date": end_date},
    ).mappings().one()
    total_rows = int(missing_row["total_rows"] or 0)
    missing_rows = int(missing_row["missing_rows"] or 0)
    missing_rate = (missing_rows / total_rows) if total_rows else None

    outlier_query = sa.text(
        """
        with base as (
            select
                ticker,
                calc_date,
                feature_value::double precision as feature_value
            from feature_store
            where feature_name = :feature_name
              and calc_date between :start_date and :end_date
              and feature_value is not null
              and not coalesce(is_filled, false)
        ),
        stats as (
            select
                calc_date,
                avg(feature_value) as mean_value,
                stddev_pop(feature_value) as std_value
            from base
            group by calc_date
        ),
        scored as (
            select
                b.ticker,
                b.calc_date,
                b.feature_value,
                case
                    when s.std_value is null or s.std_value = 0 then null
                    else abs((b.feature_value - s.mean_value) / s.std_value)
                end as zscore
            from base b
            join stats s using (calc_date)
        )
        select
            count(*) filter (where zscore is not null) as evaluated_rows,
            count(*) filter (where zscore > 5) as outlier_rows
        from scored
        """
    )
    outlier_row = conn.execute(
        outlier_query,
        {"feature_name": feature_name, "start_date": start_date, "end_date": end_date},
    ).mappings().one()
    evaluated_rows = int(outlier_row["evaluated_rows"] or 0)
    outlier_rows = int(outlier_row["outlier_rows"] or 0)
    outlier_rate = (outlier_rows / evaluated_rows) if evaluated_rows else None

    missing_examples_query = sa.text(
        """
        select ticker, calc_date, feature_value, is_filled
        from feature_store
        where feature_name = :feature_name
          and calc_date between :start_date and :end_date
          and (feature_value is null or coalesce(is_filled, false))
        order by calc_date, ticker
        limit 25
        """
    )
    missing_examples = [
        {
            "ticker": str(row["ticker"]),
            "trade_date": row["calc_date"].isoformat(),
            "feature_value": float(row["feature_value"]) if row["feature_value"] is not None else None,
            "is_filled": bool(row["is_filled"]),
        }
        for row in conn.execute(
            missing_examples_query,
            {"feature_name": feature_name, "start_date": start_date, "end_date": end_date},
        ).mappings()
    ]

    outlier_examples_query = sa.text(
        """
        with base as (
            select
                ticker,
                calc_date,
                feature_value::double precision as feature_value
            from feature_store
            where feature_name = :feature_name
              and calc_date between :start_date and :end_date
              and feature_value is not null
              and not coalesce(is_filled, false)
        ),
        stats as (
            select
                calc_date,
                avg(feature_value) as mean_value,
                stddev_pop(feature_value) as std_value
            from base
            group by calc_date
        ),
        scored as (
            select
                b.ticker,
                b.calc_date,
                b.feature_value,
                case
                    when s.std_value is null or s.std_value = 0 then null
                    else abs((b.feature_value - s.mean_value) / s.std_value)
                end as zscore
            from base b
            join stats s using (calc_date)
        )
        select ticker, calc_date, feature_value, zscore
        from scored
        where zscore > 5
        order by zscore desc nulls last, calc_date desc, ticker asc
        limit 25
        """
    )
    outlier_examples = [
        {
            "ticker": str(row["ticker"]),
            "trade_date": row["calc_date"].isoformat(),
            "feature_value": float(row["feature_value"]) if row["feature_value"] is not None else None,
            "zscore": float(row["zscore"]) if row["zscore"] is not None else None,
        }
        for row in conn.execute(
            outlier_examples_query,
            {"feature_name": feature_name, "start_date": start_date, "end_date": end_date},
        ).mappings()
    ]

    return FeatureQualityMetric(
        feature_name=feature_name,
        total_rows=total_rows,
        missing_rows=missing_rows,
        missing_rate=missing_rate,
        evaluated_rows=evaluated_rows,
        outlier_rows=outlier_rows,
        outlier_rate=outlier_rate,
        lag_rule_documented=lag_rule_documented,
        missing_examples=missing_examples,
        outlier_examples=outlier_examples,
    )


def collect_gate3_metrics(
    conn: sa.Connection,
    *,
    start_date: date,
    end_date: date,
    features: tuple[str, ...],
    lineage_documentation: dict[str, Any],
) -> list[FeatureQualityMetric]:
    documented_features = set(str(name) for name in lineage_documentation.get("documented_features", []))
    rule_mentions_t_plus_one = bool(lineage_documentation.get("rule_mentions_t_plus_one"))
    metrics: list[FeatureQualityMetric] = []
    for feature_name in features:
        metrics.append(
            fetch_quality_metric(
                conn,
                feature_name=feature_name,
                start_date=start_date,
                end_date=end_date,
                lag_rule_documented=rule_mentions_t_plus_one and feature_name in documented_features,
            ),
        )
    return metrics


def run_week3_gate_verification(
    *,
    start_date: date,
    end_date: date,
    report_output: Path,
    lineage_path: Path,
) -> dict[str, Any]:
    engine = get_engine()
    lineage_documentation = load_lag_rule_documentation(lineage_path)
    with engine.connect() as conn:
        gate1_metrics = collect_gate1_metrics(
            conn,
            start_date=start_date,
            end_date=end_date,
            features=INTRADAY_FEATURE_NAMES,
        )
        gate1 = evaluate_gate1(gate1_metrics)

        reconciliation_counts = fetch_recent_reconciliation_counts(conn, start_date=start_date, end_date=end_date)
        gate2 = evaluate_gate2(
            blocker_count=reconciliation_counts["blocker_count"],
            warning_count=reconciliation_counts["warning_count"],
            sample_blockers=reconciliation_counts["sample_blockers"],
        )
        gate2["recent_trading_dates"] = reconciliation_counts["recent_dates"]

        gate3_metrics = collect_gate3_metrics(
            conn,
            start_date=start_date,
            end_date=end_date,
            features=INTRADAY_FEATURE_NAMES,
            lineage_documentation=lineage_documentation,
        )
        gate3 = evaluate_gate3(gate3_metrics)

    report = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/run_week3_gate_verification.py",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lineage_path": str(lineage_path),
            "feature_names": list(INTRADAY_FEATURE_NAMES),
        },
        "gate1_coverage": gate1,
        "gate2_aplus": gate2,
        "gate3_quality": gate3,
        "overall_pass": bool(gate1["pass"] and gate2["pass"] and gate3["pass"]),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(_normalize_for_json(report), indent=2, sort_keys=True))
    return report


def main() -> int:
    args = parse_args()
    report = run_week3_gate_verification(
        start_date=args.start_date,
        end_date=args.end_date,
        report_output=args.report_output,
        lineage_path=args.lineage_path,
    )
    print(json.dumps(_normalize_for_json(report), indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
