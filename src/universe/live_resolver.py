"""W13 LiveUniverseResolver — dynamic admission policy for live/greyscale.

Replaces the legacy `eligible_universe_path` (frozen JSON snapshot) approach.

Admission policy (positive criteria, no blacklist):
  1. Active S&P 500 member at `as_of` (universe_membership table)
  2. ≥ min_history_days of distinct calc_dates in feature_store
  3. All required features have at least one non-null value within
     recency_window_days of `as_of`
  4. Latest non-null feature_value within max_stale_days of `as_of`
  5. 20-day rolling ADV (close * volume) ≥ min_adv_dollars

Tickers like MRSH/PSKY fail naturally on (3) because they're missing
multiple required features outright.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import text

from src.universe.active import get_active_universe


@dataclass(frozen=True)
class LiveUniversePolicy:
    """Declarative policy for admitting tickers to the live universe."""
    min_history_days: int = 90
    max_stale_days: int = 7
    recency_window_days: int = 30
    min_adv_dollars: float = 1_000_000.0
    adv_window_days: int = 20
    # Price-continuity gate: number of distinct trading days with prices in
    # the last `price_continuity_window_days` calendar days. Catches discontinuity
    # caused by mergers / spin-offs / late additions whose old ticker prices
    # exist but recent coverage has gaps.
    min_continuous_price_days: int = 200
    price_continuity_window_days: int = 365
    # Minimum number of bundle's required features that must have at least one
    # non-null value within recency_window_days. Set < len(required_features)
    # to allow legitimate per-ticker NULLs (e.g. dividend_yield for
    # non-dividend-payers; the model handles these via is_missing_X flags).
    min_required_features_present: int = 26
    benchmark_ticker: str = "SPY"
    index_name: str = "SP500"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LiveUniversePolicy":
        if not data:
            return cls()
        return cls(
            min_history_days=int(data.get("min_history_days", 90)),
            max_stale_days=int(data.get("max_stale_days", 7)),
            recency_window_days=int(data.get("recency_window_days", 30)),
            min_adv_dollars=float(data.get("min_adv_dollars", 1_000_000.0)),
            adv_window_days=int(data.get("adv_window_days", 20)),
            min_continuous_price_days=int(data.get("min_continuous_price_days", 200)),
            price_continuity_window_days=int(data.get("price_continuity_window_days", 365)),
            min_required_features_present=int(data.get("min_required_features_present", 26)),
            benchmark_ticker=str(data.get("benchmark_ticker", "SPY")),
            index_name=str(data.get("index_name", "SP500")),
        )


@dataclass(frozen=True)
class TickerDiagnostic:
    ticker: str
    admitted: bool
    distinct_days: int
    required_features_present: int
    required_features_total: int
    latest_calc_date: date | None
    adv_dollars: float | None
    continuous_price_days: int = 0
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolveResult:
    admitted_tickers: list[str]
    diagnostics: list[TickerDiagnostic]
    policy: LiveUniversePolicy
    as_of: datetime
    trade_date: date
    source: str
    candidate_count: int

    def summary(self) -> dict[str, Any]:
        rejected = [d for d in self.diagnostics if not d.admitted]
        rejection_buckets: dict[str, int] = {}
        for d in rejected:
            for r in d.rejection_reasons:
                rejection_buckets[r] = rejection_buckets.get(r, 0) + 1
        return {
            "as_of": self.as_of.isoformat(),
            "trade_date": self.trade_date.isoformat(),
            "candidate_count": self.candidate_count,
            "admitted_count": len(self.admitted_tickers),
            "rejected_count": len(rejected),
            "rejection_buckets": rejection_buckets,
            "active_universe_source": self.source,
        }


def resolve_live_universe(
    *,
    as_of: datetime,
    trade_date: date,
    required_features: list[str],
    policy: LiveUniversePolicy,
    conn: Any,
) -> ResolveResult:
    """Compute the live admission result for a given as_of timestamp.

    Caller must provide a SQLAlchemy connection (or any object with .execute).
    """
    if not required_features:
        raise ValueError("required_features must be non-empty")

    candidates = get_active_universe(
        trade_date,
        as_of=as_of,
        benchmark_ticker=policy.benchmark_ticker,
        index_name=policy.index_name,
    )
    candidate_count = len(candidates)
    if not candidates:
        return ResolveResult(
            admitted_tickers=[],
            diagnostics=[],
            policy=policy,
            as_of=as_of,
            trade_date=trade_date,
            source="empty",
            candidate_count=0,
        )

    history_start = trade_date - timedelta(days=policy.min_history_days * 2)
    recency_start = trade_date - timedelta(days=policy.recency_window_days)
    adv_start = trade_date - timedelta(days=policy.adv_window_days * 2)
    continuity_start = trade_date - timedelta(days=policy.price_continuity_window_days)

    coverage_sql = text(
        """
        SELECT
            ticker,
            COUNT(DISTINCT calc_date) AS distinct_days,
            COUNT(DISTINCT CASE
                WHEN feature_value IS NOT NULL AND calc_date >= :recency_start
                THEN feature_name
            END) AS recent_features_present,
            MAX(CASE WHEN feature_value IS NOT NULL THEN calc_date END) AS latest_non_null
        FROM feature_store
        WHERE ticker = ANY(:tickers)
          AND feature_name = ANY(:features)
          AND calc_date >= :history_start
          AND calc_date <= :trade_date
        GROUP BY ticker
        """
    )

    adv_sql = text(
        """
        SELECT
            ticker,
            AVG(adj_close * volume) AS adv
        FROM (
            SELECT
                ticker,
                COALESCE(adj_close, close) AS adj_close,
                volume,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker
                    ORDER BY trade_date DESC
                ) AS rn
            FROM stock_prices
            WHERE ticker = ANY(:tickers)
              AND trade_date >= :adv_start
              AND trade_date <= :trade_date
              AND knowledge_time <= :as_of
        ) recent
        WHERE rn <= :adv_window_days
        GROUP BY ticker
        """
    )

    continuity_sql = text(
        """
        SELECT
            ticker,
            COUNT(DISTINCT trade_date) AS continuous_days
        FROM stock_prices
        WHERE ticker = ANY(:tickers)
          AND trade_date >= :continuity_start
          AND trade_date <= :trade_date
          AND knowledge_time <= :as_of
        GROUP BY ticker
        """
    )

    coverage_rows = conn.execute(
        coverage_sql,
        {
            "tickers": candidates,
            "features": required_features,
            "history_start": history_start,
            "recency_start": recency_start,
            "trade_date": trade_date,
        },
    ).mappings().all()
    coverage_by_ticker = {row["ticker"]: dict(row) for row in coverage_rows}

    adv_rows = conn.execute(
        adv_sql,
        {
            "tickers": candidates,
            "adv_start": adv_start,
            "adv_window_days": policy.adv_window_days,
            "trade_date": trade_date,
            "as_of": as_of,
        },
    ).mappings().all()
    adv_by_ticker = {row["ticker"]: float(row["adv"] or 0.0) for row in adv_rows}

    continuity_rows = conn.execute(
        continuity_sql,
        {
            "tickers": candidates,
            "continuity_start": continuity_start,
            "trade_date": trade_date,
            "as_of": as_of,
        },
    ).mappings().all()
    continuity_by_ticker = {row["ticker"]: int(row["continuous_days"] or 0) for row in continuity_rows}

    n_features = len(required_features)
    diagnostics: list[TickerDiagnostic] = []
    admitted: list[str] = []

    for ticker in candidates:
        cov = coverage_by_ticker.get(ticker, {})
        distinct_days = int(cov.get("distinct_days") or 0)
        recent_present = int(cov.get("recent_features_present") or 0)
        latest = cov.get("latest_non_null")
        adv = adv_by_ticker.get(ticker)
        continuous_days = continuity_by_ticker.get(ticker, 0)

        reasons: list[str] = []
        if distinct_days < policy.min_history_days:
            reasons.append(f"insufficient_history({distinct_days}<{policy.min_history_days})")
        if continuous_days < policy.min_continuous_price_days:
            reasons.append(
                f"discontinuous_prices({continuous_days}<{policy.min_continuous_price_days}"
                f" in {policy.price_continuity_window_days}d)"
            )
        if recent_present < policy.min_required_features_present:
            reasons.append(
                f"missing_required_features({recent_present}/{n_features},"
                f"<{policy.min_required_features_present})"
            )
        if latest is None:
            reasons.append("no_non_null_feature_values")
            latest_date = None
        else:
            if isinstance(latest, datetime):
                latest_date = latest.date()
            else:
                latest_date = latest
            if (trade_date - latest_date).days > policy.max_stale_days:
                reasons.append(f"stale({(trade_date - latest_date).days}d>{policy.max_stale_days}d)")
        if adv is None or adv < policy.min_adv_dollars:
            reasons.append(f"low_adv(${adv:,.0f}<${policy.min_adv_dollars:,.0f})" if adv else "no_adv_data")

        admitted_flag = not reasons
        diagnostics.append(
            TickerDiagnostic(
                ticker=ticker,
                admitted=admitted_flag,
                distinct_days=distinct_days,
                required_features_present=recent_present,
                required_features_total=n_features,
                latest_calc_date=latest_date,
                adv_dollars=adv,
                continuous_price_days=continuous_days,
                rejection_reasons=tuple(reasons),
            )
        )
        if admitted_flag:
            admitted.append(ticker)

    return ResolveResult(
        admitted_tickers=admitted,
        diagnostics=diagnostics,
        policy=policy,
        as_of=as_of,
        trade_date=trade_date,
        source="universe_membership+admission",
        candidate_count=candidate_count,
    )
