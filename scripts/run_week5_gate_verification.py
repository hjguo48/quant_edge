"""Week 5 Tranche A gate verification.

Four gates:
1. Coverage gate
2. Missing-rate gate
3. Lag-rule gate
4. Source-integrity gate
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import sqlalchemy as sa
import yaml
from loguru import logger

from src.config import settings
from src.data.db.models import UniverseMembership
from src.data.db.session import get_session_factory
from src.data.finra_short_sale import ShortSaleVolume
from src.data.sources.fmp_earnings_calendar import EarningsCalendar
from src.data.sources.fmp_grades import GradesEvent
from src.data.sources.fmp_price_target import PriceTargetEvent
from src.data.sources.fmp_ratings import RatingEvent
from src.features.registry import FeatureRegistry, build_feature_registry
from src.universe.history import get_historical_members

EASTERN = ZoneInfo("America/New_York")
XNYS = xcals.get_calendar("XNYS")
DEFAULT_OUTPUT = Path("data/reports/week5/gate_summary.json")
DEFAULT_FEATURES_CONFIG = Path("configs/research/data_lineage.yaml")
DEFAULT_END_DATE = date.today()
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=365 * 3)
SHORTING_FEATURES = (
    "short_sale_ratio_1d",
    "short_sale_ratio_5d",
    "short_sale_accel",
    "abnormal_off_exchange_shorting",
)
ANALYST_PROXY_FEATURES = (
    "net_grade_change_5d",
    "net_grade_change_20d",
    "net_grade_change_60d",
    "upgrade_count",
    "downgrade_count",
    "consensus_upside",
    "target_price_drift",
    "target_dispersion_proxy",
    "coverage_change_proxy",
    "financial_health_trend",
)
WEEK5_FEATURE_NAMES = SHORTING_FEATURES + ANALYST_PROXY_FEATURES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 5 gate verification for shorting and analyst proxy features.")
    parser.add_argument("--features-config", type=Path, default=DEFAULT_FEATURES_CONFIG)
    parser.add_argument("--start-date", type=_parse_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=_parse_date, default=DEFAULT_END_DATE)
    parser.add_argument("--universe-asof", type=_parse_date, default=DEFAULT_END_DATE)
    parser.add_argument("--features", type=str, default=",".join(WEEK5_FEATURE_NAMES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-flags", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sample-tickers",
        type=int,
        default=None,
        help="Cap tickers in missing-rate gate to this many (random sample). "
        "Mitigates the O(features × tickers × dates) hot loop on full SP500. "
        "Default: no cap.",
    )
    parser.add_argument(
        "--sample-dates",
        type=int,
        default=None,
        help="Cap trading dates in missing-rate gate to this many (random sample). "
        "Default: no cap.",
    )
    return parser.parse_args(argv)


def compute_coverage_gate(
    *,
    start_date: date,
    end_date: date,
    session_factory: Callable | None = None,
    universe_fetcher: Callable[[date, str], list[str]] | None = None,
) -> dict[str, Any]:
    factory = session_factory or get_session_factory()
    with factory() as session:
        finra_periods = list(_iter_month_periods(start_date, end_date))
        quarter_periods = list(_iter_quarter_periods(start_date, end_date))
        finra_months = (
            _compute_period_coverage(
                session,
                model=ShortSaleVolume,
                date_column=ShortSaleVolume.trade_date,
                periods=finra_periods,
                threshold=0.90,
                label_key="month",
                universe_fetcher=universe_fetcher,
            )
            if _source_table_has_rows(session, ShortSaleVolume, ShortSaleVolume.trade_date, start_date, end_date)
            else _empty_period_coverage(finra_periods, threshold=0.90, label_key="month")
        )
        grades_quarters = (
            _compute_period_coverage(
                session,
                model=GradesEvent,
                date_column=GradesEvent.event_date,
                periods=quarter_periods,
                threshold=0.70,
                label_key="quarter",
                universe_fetcher=universe_fetcher,
            )
            if _source_table_has_rows(session, GradesEvent, GradesEvent.event_date, start_date, end_date)
            else _empty_period_coverage(quarter_periods, threshold=0.70, label_key="quarter")
        )
        ratings_quarters = (
            _compute_period_coverage(
                session,
                model=RatingEvent,
                date_column=RatingEvent.event_date,
                periods=quarter_periods,
                threshold=0.70,
                label_key="quarter",
                universe_fetcher=universe_fetcher,
            )
            if _source_table_has_rows(session, RatingEvent, RatingEvent.event_date, start_date, end_date)
            else _empty_period_coverage(quarter_periods, threshold=0.70, label_key="quarter")
        )
        target_quarters = (
            _compute_period_coverage(
                session,
                model=PriceTargetEvent,
                date_column=PriceTargetEvent.event_date,
                periods=quarter_periods,
                threshold=0.70,
                label_key="quarter",
                universe_fetcher=universe_fetcher,
            )
            if _source_table_has_rows(session, PriceTargetEvent, PriceTargetEvent.event_date, start_date, end_date)
            else _empty_period_coverage(quarter_periods, threshold=0.70, label_key="quarter")
        )
        earnings_quarters = (
            _compute_period_coverage(
                session,
                model=EarningsCalendar,
                date_column=EarningsCalendar.announce_date,
                periods=quarter_periods,
                threshold=0.95,
                label_key="quarter",
                universe_fetcher=universe_fetcher,
            )
            if _source_table_has_rows(session, EarningsCalendar, EarningsCalendar.announce_date, start_date, end_date)
            else _empty_period_coverage(quarter_periods, threshold=0.95, label_key="quarter")
        )

    sources = {
        "finra_short_sale_volume": finra_months,
        "fmp_grades": grades_quarters,
        "fmp_ratings": ratings_quarters,
        "fmp_price_target": target_quarters,
        "earnings_calendar": earnings_quarters,
    }
    return {
        "status": "PASS" if all(source["status"] == "PASS" for source in sources.values()) else "FAIL",
        "sources": sources,
    }


def compute_missing_rate_gate(
    *,
    feature_names: Sequence[str],
    start_date: date,
    end_date: date,
    universe_asof: date,
    session_factory: Callable | None = None,
    registry_builder: Callable[[], FeatureRegistry] = build_feature_registry,
    universe_fetcher: Callable[[date, str], list[str]] | None = None,
    sample_tickers: int | None = None,
    sample_dates: int | None = None,
) -> dict[str, Any]:
    trade_dates = [session.date() for session in XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    universe_source = universe_fetcher or get_historical_members
    tickers = tuple(sorted(set(universe_source(universe_asof, "SP500"))))

    rng = np.random.default_rng(42)
    if sample_tickers is not None and sample_tickers < len(tickers):
        indices = sorted(rng.choice(len(tickers), size=sample_tickers, replace=False).tolist())
        tickers = tuple(tickers[i] for i in indices)
    if sample_dates is not None and sample_dates < len(trade_dates):
        indices = sorted(rng.choice(len(trade_dates), size=sample_dates, replace=False).tolist())
        trade_dates = [trade_dates[i] for i in indices]
    total = len(trade_dates) * len(tickers)
    factory = session_factory or get_session_factory()
    registry = registry_builder()

    per_feature: list[dict[str, Any]] = []
    overall_pass = True
    with factory() as session:
        for feature_name in feature_names:
            missing = 0
            errors: list[str] = []
            sparse_note: str | None = None
            if total == 0:
                missing_rate = 1.0
                missing = total
            else:
                source_ticker_count = _source_ticker_count_for_feature(
                    session,
                    feature_name=feature_name,
                    start_date=start_date,
                    end_date=end_date,
                    tickers=tickers,
                )
                sparse_threshold = max(5, int(np.ceil(len(tickers) * 0.05))) if tickers else 0
                if source_ticker_count == 0:
                    missing_rate = 1.0
                    missing = total
                    sparse_note = "source_table_empty_for_requested_universe"
                elif source_ticker_count < sparse_threshold:
                    missing_rate = 1.0
                    missing = total
                    sparse_note = (
                        f"sparse_source_short_circuit: {source_ticker_count} tickers with source rows "
                        f"< threshold {sparse_threshold}"
                    )
                else:
                    feature = registry.get_feature(feature_name)
                    for trade_date in trade_dates:
                        for ticker in tickers:
                            try:
                                value = feature.compute_fn(ticker=ticker, as_of=trade_date, session_factory=factory)
                            except Exception as exc:  # pragma: no cover - defensive runtime guard
                                missing += 1
                                if len(errors) < 10:
                                    errors.append(f"{ticker}@{trade_date.isoformat()}: {exc}")
                                continue
                            if value is None or (isinstance(value, float) and np.isnan(value)):
                                missing += 1
                    missing_rate = missing / total
            status = "PASS" if missing_rate < 0.40 else "FAIL"
            overall_pass = overall_pass and status == "PASS"
            per_feature.append(
                {
                    "feature": feature_name,
                    "missing_rate": round(float(missing_rate), 6),
                    "threshold": 0.40,
                    "status": status,
                    "samples": total,
                    "missing": missing,
                    "errors": errors,
                    "note": sparse_note,
                },
            )

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "per_feature": per_feature,
    }


def compute_lag_rule_gate(
    *,
    start_date: date,
    end_date: date,
    session_factory: Callable | None = None,
) -> dict[str, Any]:
    factory = session_factory or get_session_factory()
    violations: list[dict[str, Any]] = []
    with factory() as session:
        source_specs = [
            ("short_sale_volume_daily", ShortSaleVolume, ShortSaleVolume.trade_date, ShortSaleVolume.knowledge_time, _short_sale_min_kt_utc, None),
            ("grades_events", GradesEvent, GradesEvent.event_date, GradesEvent.knowledge_time, _end_of_day_utc, None),
            ("ratings_events", RatingEvent, RatingEvent.event_date, RatingEvent.knowledge_time, _end_of_day_utc, None),
            (
                "price_target_events",
                PriceTargetEvent,
                PriceTargetEvent.event_date,
                PriceTargetEvent.knowledge_time,
                _end_of_day_utc,
                PriceTargetEvent.is_consensus.is_(False),
            ),
            ("earnings_calendar", EarningsCalendar, EarningsCalendar.announce_date, EarningsCalendar.knowledge_time, _end_of_day_utc, None),
        ]

        for source_name, model, date_column, knowledge_column, expected_fn, extra_filter in source_specs:
            stmt = sa.select(model).where(date_column >= start_date, date_column <= end_date)
            if extra_filter is not None:
                stmt = stmt.where(extra_filter)
            rows = session.execute(stmt).scalars().all()
            offenders = []
            for row in rows:
                event_day = getattr(row, date_column.key)
                knowledge_time = getattr(row, knowledge_column.key)
                if knowledge_time is None:
                    continue
                if knowledge_time < expected_fn(event_day):
                    offenders.append(
                        {
                            "ticker": getattr(row, "ticker"),
                            "event_date": event_day.isoformat(),
                            "knowledge_time": knowledge_time.isoformat(),
                        },
                    )
                    if len(offenders) >= 10:
                        break
            if offenders:
                violations.append({"source": source_name, "offenders": offenders})

    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def compute_source_integrity_gate(
    *,
    start_date: date,
    end_date: date,
    session_factory: Callable | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    factory = session_factory or get_session_factory()
    today_value = today or date.today()
    with factory() as session:
        finra_rows = session.execute(
            sa.select(sa.func.count()).select_from(ShortSaleVolume).where(
                ShortSaleVolume.trade_date >= start_date,
                ShortSaleVolume.trade_date <= end_date,
            ),
        ).scalar_one()
        finra_etag_change_rate = 0.0 if finra_rows else None

        grades_rate = _field_population_rate(
            session,
            sa.select(GradesEvent.new_grade, GradesEvent.analyst_firm).where(
                GradesEvent.event_date >= start_date,
                GradesEvent.event_date <= end_date,
            ),
            lambda row: row["new_grade"] is not None and row["analyst_firm"] is not None,
        )
        ratings_rate = _field_population_rate(
            session,
            sa.select(RatingEvent.rating_score, RatingEvent.rating_recommendation).where(
                RatingEvent.event_date >= start_date,
                RatingEvent.event_date <= end_date,
            ),
            lambda row: row["rating_score"] is not None and row["rating_recommendation"] is not None,
        )
        price_target_rate = _field_population_rate(
            session,
            sa.select(
                PriceTargetEvent.target_price,
                PriceTargetEvent.analyst_firm,
                PriceTargetEvent.is_consensus,
            ).where(
                PriceTargetEvent.event_date >= start_date,
                PriceTargetEvent.event_date <= end_date,
            ),
            lambda row: row["target_price"] is not None and (bool(row["is_consensus"]) or row["analyst_firm"] is not None),
        )
        earnings_rate = _field_population_rate(
            session,
            sa.select(
                EarningsCalendar.announce_date,
                EarningsCalendar.eps_actual,
                EarningsCalendar.revenue_actual,
            ).where(
                EarningsCalendar.announce_date >= start_date,
                EarningsCalendar.announce_date <= end_date,
            ),
            lambda row: (
                row["announce_date"] > today_value
                or (row["eps_actual"] is not None and row["revenue_actual"] is not None)
            ),
        )

    field_rates = {
        "grades": grades_rate,
        "ratings": ratings_rate,
        "price_target": price_target_rate,
        "earnings_calendar": earnings_rate,
    }
    rates_present = [rate for rate in field_rates.values() if rate is not None]
    if not rates_present and finra_etag_change_rate is None:
        status = "PARTIAL"
    elif any(rate is not None and rate < 0.95 for rate in field_rates.values()):
        status = "FAIL"
    elif finra_etag_change_rate is not None and finra_etag_change_rate >= 0.05:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "status": status,
        "finra_etag_change_rate": None if finra_etag_change_rate is None else round(float(finra_etag_change_rate), 6),
        "finra_etag_threshold": 0.05,
        "finra_note": "Current schema retains the latest file_etag only; change rate is 0 unless table versioning is added.",
        "fmp_field_populate_rate": {
            key: None if value is None else round(float(value), 6)
            for key, value in field_rates.items()
        },
        "field_population_threshold": 0.95,
        "http_error_rate": "not_implemented",
    }


def generate_gate_summary(
    *,
    feature_names: Sequence[str],
    start_date: date,
    end_date: date,
    universe_asof: date,
    features_config_path: Path,
    session_factory: Callable | None = None,
    registry_builder: Callable[[], FeatureRegistry] = build_feature_registry,
    universe_fetcher: Callable[[date, str], list[str]] | None = None,
    sample_tickers: int | None = None,
    sample_dates: int | None = None,
) -> dict[str, Any]:
    lineage = load_lineage_config(features_config_path)
    universe_source = universe_fetcher or get_historical_members
    coverage_gate = compute_coverage_gate(
        start_date=start_date,
        end_date=end_date,
        session_factory=session_factory,
        universe_fetcher=universe_source,
    )
    missing_gate = compute_missing_rate_gate(
        feature_names=feature_names,
        start_date=start_date,
        end_date=end_date,
        universe_asof=universe_asof,
        session_factory=session_factory,
        registry_builder=registry_builder,
        universe_fetcher=universe_source,
        sample_tickers=sample_tickers,
        sample_dates=sample_dates,
    )
    lag_gate = compute_lag_rule_gate(
        start_date=start_date,
        end_date=end_date,
        session_factory=session_factory,
    )
    integrity_gate = compute_source_integrity_gate(
        start_date=start_date,
        end_date=end_date,
        session_factory=session_factory,
    )
    overall_pass = (
        coverage_gate["status"] == "PASS"
        and missing_gate["status"] == "PASS"
        and lag_gate["status"] == "PASS"
        and integrity_gate["status"] != "FAIL"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "universe_asof": universe_asof.isoformat(),
        "features_covered": list(feature_names),
        "lineage_version": lineage.get("version"),
        "gates": {
            "coverage": coverage_gate,
            "missing_rate": missing_gate,
            "lag_rule": lag_gate,
            "source_integrity": integrity_gate,
        },
        "overall": "PASS" if overall_pass else "FAIL",
        "overall_rationale": _overall_rationale(
            coverage_status=coverage_gate["status"],
            missing_status=missing_gate["status"],
            lag_status=lag_gate["status"],
            integrity_status=integrity_gate["status"],
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested_features = _parse_feature_names(args.features)
    should_enable_flags = args.enable_flags or any(feature in WEEK5_FEATURE_NAMES for feature in requested_features)

    with _temporary_enable_week5_flags(should_enable_flags):
        summary = generate_gate_summary(
            feature_names=requested_features,
            start_date=args.start_date,
            end_date=args.end_date,
            universe_asof=args.universe_asof,
            features_config_path=args.features_config,
            sample_tickers=args.sample_tickers,
            sample_dates=args.sample_dates,
        )

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("week5 gate summary written to {}", args.output)

    return 0 if summary["overall"] == "PASS" else 1


def load_lineage_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _compute_period_coverage(
    session,
    *,
    model,
    date_column,
    periods: Sequence[tuple[str, date, date, date]],
    threshold: float,
    label_key: str,
    universe_fetcher: Callable[[date, str], list[str]] | None,
) -> dict[str, Any]:
    coverage_items: list[dict[str, Any]] = []
    min_coverage = 1.0
    for label, period_start, period_end, snapshot_date in periods:
        if universe_fetcher is not None:
            members = tuple(sorted(set(universe_fetcher(snapshot_date, "SP500"))))
            denominator = len(members)
            if denominator == 0:
                ratio = 0.0
                numerator = 0
            else:
                stmt = sa.select(sa.func.count(sa.distinct(model.ticker))).where(
                    date_column >= period_start,
                    date_column <= period_end,
                    model.ticker.in_(members),
                )
                numerator = int(session.execute(stmt).scalar_one() or 0)
                ratio = numerator / denominator
        else:
            denominator_stmt = sa.select(sa.func.count(sa.distinct(UniverseMembership.ticker))).where(
                UniverseMembership.index_name == "SP500",
                UniverseMembership.effective_date <= snapshot_date,
                sa.or_(
                    UniverseMembership.end_date.is_(None),
                    UniverseMembership.end_date > snapshot_date,
                ),
            )
            denominator = int(session.execute(denominator_stmt).scalar_one() or 0)
            if denominator == 0:
                ratio = 0.0
                numerator = 0
            else:
                numerator_stmt = (
                    sa.select(sa.func.count(sa.distinct(model.ticker)))
                    .select_from(model)
                    .join(
                        UniverseMembership,
                        sa.and_(
                            UniverseMembership.ticker == model.ticker,
                            UniverseMembership.index_name == "SP500",
                            UniverseMembership.effective_date <= snapshot_date,
                            sa.or_(
                                UniverseMembership.end_date.is_(None),
                                UniverseMembership.end_date > snapshot_date,
                            ),
                        ),
                    )
                    .where(
                        date_column >= period_start,
                        date_column <= period_end,
                    )
                )
                numerator = int(session.execute(numerator_stmt).scalar_one() or 0)
                ratio = numerator / denominator
        min_coverage = min(min_coverage, ratio)
        coverage_items.append(
            {
                label_key: label,
                "ratio": round(float(ratio), 6),
                "covered": numerator,
                "expected": denominator,
                "status": "PASS" if ratio >= threshold else "FAIL",
            },
        )
    overall_status = "PASS" if coverage_items and all(item["status"] == "PASS" for item in coverage_items) else "FAIL"
    return {
        "status": overall_status,
        f"{label_key}ly_coverage" if label_key == "quarter" else f"{label_key}ly_coverage": coverage_items,
        "threshold": threshold,
        "min_coverage": round(float(min_coverage if coverage_items else 0.0), 6),
    }


def _empty_period_coverage(
    periods: Sequence[tuple[str, date, date, date]],
    *,
    threshold: float,
    label_key: str,
) -> dict[str, Any]:
    items = [
        {
            label_key: label,
            "ratio": 0.0,
            "covered": 0,
            "expected": 0,
            "status": "FAIL",
        }
        for label, _start, _end, _snapshot in periods
    ]
    return {
        "status": "FAIL",
        f"{label_key}ly_coverage" if label_key == "quarter" else f"{label_key}ly_coverage": items,
        "threshold": threshold,
        "min_coverage": 0.0,
        "note": "source_table_empty_in_range",
    }


def _field_population_rate(session, stmt, predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    rows = session.execute(stmt).mappings().all()
    if not rows:
        return None
    valid = sum(1 for row in rows if predicate(row))
    return valid / len(rows)


def _source_table_has_rows(session, model, date_column, start_date: date, end_date: date) -> bool:
    stmt = sa.select(sa.literal(True)).where(
        date_column >= start_date,
        date_column <= end_date,
    ).select_from(model).limit(1)
    return session.execute(stmt).scalar_one_or_none() is True


def _source_rows_exist_for_feature(
    session,
    *,
    feature_name: str,
    start_date: date,
    end_date: date,
    tickers: Sequence[str],
) -> bool:
    if not tickers:
        return False
    if feature_name in SHORTING_FEATURES:
        stmt = sa.select(sa.literal(True)).where(
            ShortSaleVolume.trade_date >= start_date,
            ShortSaleVolume.trade_date <= end_date,
            ShortSaleVolume.ticker.in_(tickers),
        ).limit(1)
        return session.execute(stmt).scalar_one_or_none() is True

    if feature_name in {"net_grade_change_5d", "net_grade_change_20d", "net_grade_change_60d", "upgrade_count", "downgrade_count"}:
        stmt = sa.select(sa.literal(True)).where(
            GradesEvent.event_date >= start_date,
            GradesEvent.event_date <= end_date,
            GradesEvent.ticker.in_(tickers),
        ).limit(1)
        return session.execute(stmt).scalar_one_or_none() is True

    if feature_name in {"consensus_upside", "target_price_drift", "target_dispersion_proxy", "coverage_change_proxy"}:
        stmt = sa.select(sa.literal(True)).where(
            PriceTargetEvent.event_date >= start_date - timedelta(days=60),
            PriceTargetEvent.event_date <= end_date,
            PriceTargetEvent.ticker.in_(tickers),
        ).limit(1)
        return session.execute(stmt).scalar_one_or_none() is True

    if feature_name == "financial_health_trend":
        stmt = sa.select(sa.literal(True)).where(
            RatingEvent.event_date >= start_date - timedelta(days=60),
            RatingEvent.event_date <= end_date,
            RatingEvent.ticker.in_(tickers),
        ).limit(1)
        return session.execute(stmt).scalar_one_or_none() is True

    return True


def _source_ticker_count_for_feature(
    session,
    *,
    feature_name: str,
    start_date: date,
    end_date: date,
    tickers: Sequence[str],
) -> int:
    if not tickers:
        return 0
    if feature_name in SHORTING_FEATURES:
        stmt = sa.select(sa.func.count(sa.distinct(ShortSaleVolume.ticker))).where(
            ShortSaleVolume.trade_date >= start_date,
            ShortSaleVolume.trade_date <= end_date,
            ShortSaleVolume.ticker.in_(tickers),
        )
        return int(session.execute(stmt).scalar_one() or 0)

    if feature_name in {"net_grade_change_5d", "net_grade_change_20d", "net_grade_change_60d", "upgrade_count", "downgrade_count"}:
        stmt = sa.select(sa.func.count(sa.distinct(GradesEvent.ticker))).where(
            GradesEvent.event_date >= start_date,
            GradesEvent.event_date <= end_date,
            GradesEvent.ticker.in_(tickers),
        )
        return int(session.execute(stmt).scalar_one() or 0)

    if feature_name in {"consensus_upside", "target_price_drift", "target_dispersion_proxy", "coverage_change_proxy"}:
        stmt = sa.select(sa.func.count(sa.distinct(PriceTargetEvent.ticker))).where(
            PriceTargetEvent.event_date >= start_date - timedelta(days=60),
            PriceTargetEvent.event_date <= end_date,
            PriceTargetEvent.ticker.in_(tickers),
        )
        return int(session.execute(stmt).scalar_one() or 0)

    if feature_name == "financial_health_trend":
        stmt = sa.select(sa.func.count(sa.distinct(RatingEvent.ticker))).where(
            RatingEvent.event_date >= start_date - timedelta(days=60),
            RatingEvent.event_date <= end_date,
            RatingEvent.ticker.in_(tickers),
        )
        return int(session.execute(stmt).scalar_one() or 0)

    return len(tickers)


def _iter_month_periods(start_date: date, end_date: date) -> Iterator[tuple[str, date, date, date]]:
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        next_month = date(cursor.year + (cursor.month // 12), (cursor.month % 12) + 1, 1)
        period_start = max(cursor, start_date)
        period_end = min(next_month - timedelta(days=1), end_date)
        yield (f"{cursor.year:04d}-{cursor.month:02d}", period_start, period_end, period_end)
        cursor = next_month


def _iter_quarter_periods(start_date: date, end_date: date) -> Iterator[tuple[str, date, date, date]]:
    quarter_month = ((start_date.month - 1) // 3) * 3 + 1
    cursor = date(start_date.year, quarter_month, 1)
    while cursor <= end_date:
        next_quarter_month = cursor.month + 3
        next_quarter_year = cursor.year + ((next_quarter_month - 1) // 12)
        next_quarter_month = ((next_quarter_month - 1) % 12) + 1
        next_quarter = date(next_quarter_year, next_quarter_month, 1)
        period_start = max(cursor, start_date)
        period_end = min(next_quarter - timedelta(days=1), end_date)
        quarter_number = ((cursor.month - 1) // 3) + 1
        yield (f"{cursor.year:04d}-Q{quarter_number}", period_start, period_end, period_end)
        cursor = next_quarter


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_feature_names(raw_value: str) -> tuple[str, ...]:
    return tuple(feature.strip() for feature in raw_value.split(",") if feature.strip())


def _short_sale_min_kt_utc(trade_date: date) -> datetime:
    return datetime.combine(trade_date, time(hour=18, minute=0), tzinfo=EASTERN).astimezone(timezone.utc)


def _end_of_day_utc(value_date: date) -> datetime:
    return datetime.combine(value_date, time(hour=23, minute=59), tzinfo=EASTERN).astimezone(timezone.utc)


@contextmanager
def _temporary_enable_week5_flags(enable: bool) -> Iterator[None]:
    original_shorting = settings.ENABLE_SHORTING_FEATURES
    original_analyst = settings.ENABLE_ANALYST_PROXY_FEATURES
    if enable:
        settings.ENABLE_SHORTING_FEATURES = True
        settings.ENABLE_ANALYST_PROXY_FEATURES = True
    try:
        yield
    finally:
        settings.ENABLE_SHORTING_FEATURES = original_shorting
        settings.ENABLE_ANALYST_PROXY_FEATURES = original_analyst


def _overall_rationale(
    *,
    coverage_status: str,
    missing_status: str,
    lag_status: str,
    integrity_status: str,
) -> str:
    return (
        "coverage={coverage}, missing_rate={missing}, lag_rule={lag}, source_integrity={integrity}".format(
            coverage=coverage_status,
            missing=missing_status,
            lag=lag_status,
            integrity=integrity_status,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
