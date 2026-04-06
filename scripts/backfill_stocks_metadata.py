from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import sys
import time
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import requests
import sqlalchemy as sa
from loguru import logger

from _data_ops import configure_logging, ensure_tables_exist
from src.config import settings
from src.data.db.models import Stock, UniverseMembership
from src.data.db.session import get_session_factory

INDEX_NAME = "SP500"
PROFILE_ENDPOINT = "https://financialmodelingprep.com/stable/profile"
SHARES_FLOAT_ENDPOINT = "https://financialmodelingprep.com/stable/shares-float"


@dataclass(frozen=True)
class MembershipSummary:
    ticker: str
    latest_end_date: date | None
    latest_reason: str | None


@dataclass(frozen=True)
class ExistingStockSnapshot:
    ticker: str
    ipo_date: date | None
    delist_date: date | None
    delist_reason: str | None
    shares_outstanding: int | None


@dataclass(frozen=True)
class PlanItem:
    ticker: str
    action: str
    reasons: tuple[str, ...]
    membership_end_date: date | None
    membership_reason: str | None


@dataclass(frozen=True)
class FailureRecord:
    ticker: str
    action: str
    error_reason: str


@dataclass
class RunSummary:
    planned: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[FailureRecord] = field(default_factory=list)


class FMPMetadataClient:
    def __init__(self, api_key: str, *, min_request_interval: float = 0.25) -> None:
        if not api_key:
            raise RuntimeError("FMP_API_KEY is required for stocks metadata backfill.")
        self._api_key = api_key
        self._min_request_interval = min_request_interval
        self._last_request_started_at: float | None = None
        self._session = requests.Session()
        # FMP rejects some proxy-routed requests; match the existing datasource pattern.
        self._session.trust_env = False
        self._session.headers.update({"User-Agent": "QuantEdge/0.1.0"})

    def fetch_profile(self, ticker: str) -> dict[str, Any] | None:
        rows = self._get_endpoint_rows(PROFILE_ENDPOINT, ticker=ticker)
        if not rows:
            return None
        if not isinstance(rows[0], dict):
            raise RuntimeError(f"Unexpected FMP profile payload for {ticker}: {type(rows[0]).__name__}")
        return rows[0]

    def fetch_shares_outstanding(self, ticker: str) -> int | None:
        rows = self._get_endpoint_rows(SHARES_FLOAT_ENDPOINT, ticker=ticker)
        if not rows:
            return None
        if not isinstance(rows[0], dict):
            raise RuntimeError(
                f"Unexpected FMP shares-float payload for {ticker}: {type(rows[0]).__name__}",
            )
        raw_value = rows[0].get("outstandingShares")
        if raw_value in (None, "", 0, 0.0):
            return None
        try:
            value = int(float(raw_value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid outstandingShares value for {ticker}: {raw_value!r}") from exc
        return value if value > 0 else None

    def _get_endpoint_rows(self, endpoint: str, *, ticker: str) -> list[Any]:
        self._throttle(endpoint)
        response = self._session.get(
            endpoint,
            params={"apikey": self._api_key, "symbol": ticker},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"FMP request failed endpoint={endpoint} ticker={ticker} status={response.status_code} body={response.text[:200]!r}",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"FMP returned invalid JSON endpoint={endpoint} ticker={ticker}") from exc

        if isinstance(payload, dict) and payload.get("Error Message"):
            raise RuntimeError(f"FMP error endpoint={endpoint} ticker={ticker}: {payload['Error Message']}")
        if not isinstance(payload, list):
            raise RuntimeError(
                f"Unexpected FMP payload endpoint={endpoint} ticker={ticker}: {type(payload).__name__}",
            )
        return payload

    def _throttle(self, endpoint: str) -> None:
        now = time.monotonic()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            sleep_for = self._min_request_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._last_request_started_at = time.monotonic()
        logger.debug("requesting FMP endpoint={} ticker_backfill", endpoint)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    configure_logging("backfill_stocks_metadata")
    ensure_tables_exist(required_tables=("stocks", "universe_membership"))

    requested_tickers = _parse_tickers_arg(args.tickers)
    membership_by_ticker = _load_membership_summaries()
    stocks_by_ticker = _load_existing_stocks()

    plan = _build_plan(
        membership_by_ticker=membership_by_ticker,
        stocks_by_ticker=stocks_by_ticker,
        requested_tickers=requested_tickers,
        limit=args.limit,
    )
    _log_plan(plan=plan, dry_run=args.dry_run)

    if args.dry_run:
        logger.info(
            "stocks metadata dry-run summary planned={} inserts={} updates={} skips={}",
            len(plan),
            sum(1 for item in plan if item.action == "insert"),
            sum(1 for item in plan if item.action == "update"),
            sum(1 for item in plan if item.action == "skip"),
        )
        return 0

    summary = _execute_plan(
        plan=plan,
        membership_by_ticker=membership_by_ticker,
        client=FMPMetadataClient(settings.FMP_API_KEY),
    )
    if args.failures_csv:
        _write_failures_csv(Path(args.failures_csv), summary.failures)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill stocks table metadata for active and removed S&P 500 tickers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned inserts/updates without writing.")
    parser.add_argument("--limit", type=int, help="Process only the first N planned tickers.")
    parser.add_argument(
        "--tickers",
        help="Comma-separated ticker filter (for example AAPL,ABMD,BRK-B).",
    )
    parser.add_argument(
        "--failures-csv",
        help="Write failures to CSV with ticker,action,error_reason after execution.",
    )
    return parser.parse_args(argv)


def _parse_tickers_arg(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    tickers = sorted({_normalize_ticker(token) for token in value.split(",") if token.strip()})
    return tuple(ticker for ticker in tickers if ticker)


def _normalize_ticker(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper().replace(".", "-")


def _parse_iso_date(raw_value: object) -> date | None:
    if raw_value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(raw_value)[:10])
    except ValueError:
        return None


def _coerce_optional_str(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _coerce_optional_bool(raw_value: object) -> bool | None:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def _load_membership_summaries() -> dict[str, MembershipSummary]:
    statement = (
        sa.select(
            UniverseMembership.ticker.label("ticker"),
            UniverseMembership.end_date.label("end_date"),
            UniverseMembership.reason.label("reason"),
        )
        .where(UniverseMembership.index_name == INDEX_NAME)
        .order_by(UniverseMembership.ticker, UniverseMembership.end_date.desc().nullslast())
    )
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    summaries: dict[str, MembershipSummary] = {}
    for row in rows:
        ticker = _normalize_ticker(row["ticker"])
        if not ticker:
            continue
        current = summaries.get(ticker)
        end_date = row["end_date"]
        if current is None:
            summaries[ticker] = MembershipSummary(
                ticker=ticker,
                latest_end_date=end_date,
                latest_reason=_coerce_optional_str(row["reason"]),
            )
            continue
        if current.latest_end_date is None and end_date is not None:
            summaries[ticker] = MembershipSummary(
                ticker=ticker,
                latest_end_date=end_date,
                latest_reason=_coerce_optional_str(row["reason"]),
            )
    logger.info("loaded {} distinct membership tickers for {}", len(summaries), INDEX_NAME)
    return summaries


def _load_existing_stocks() -> dict[str, ExistingStockSnapshot]:
    statement = sa.select(
        Stock.ticker,
        Stock.ipo_date,
        Stock.delist_date,
        Stock.delist_reason,
        Stock.shares_outstanding,
    ).order_by(Stock.ticker)
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).all()

    snapshots = {
        _normalize_ticker(row.ticker): ExistingStockSnapshot(
            ticker=_normalize_ticker(row.ticker),
            ipo_date=row.ipo_date,
            delist_date=row.delist_date,
            delist_reason=row.delist_reason,
            shares_outstanding=row.shares_outstanding,
        )
        for row in rows
    }
    logger.info("loaded {} existing stocks rows", len(snapshots))
    return snapshots


def _build_plan(
    *,
    membership_by_ticker: dict[str, MembershipSummary],
    stocks_by_ticker: dict[str, ExistingStockSnapshot],
    requested_tickers: Sequence[str],
    limit: int | None,
) -> list[PlanItem]:
    if requested_tickers:
        candidate_tickers = list(requested_tickers)
    else:
        candidate_tickers = sorted(
            set(membership_by_ticker)
            | {
                ticker
                for ticker, stock in stocks_by_ticker.items()
                if _needs_existing_backfill(stock=stock, membership=membership_by_ticker.get(ticker))
            },
        )
    if limit is not None:
        candidate_tickers = candidate_tickers[:limit]

    plan: list[PlanItem] = []
    for ticker in candidate_tickers:
        existing = stocks_by_ticker.get(ticker)
        membership = membership_by_ticker.get(ticker)
        reasons = _plan_reasons(existing=existing, membership=membership)
        action = "insert" if existing is None else "update"
        if not reasons:
            action = "skip"
            reasons = ("already_complete",)
        plan.append(
            PlanItem(
                ticker=ticker,
                action=action,
                reasons=reasons,
                membership_end_date=membership.latest_end_date if membership is not None else None,
                membership_reason=membership.latest_reason if membership is not None else None,
            ),
        )
    return plan


def _needs_existing_backfill(
    *,
    stock: ExistingStockSnapshot,
    membership: MembershipSummary | None,
) -> bool:
    if stock.ipo_date is None or stock.shares_outstanding is None:
        return True
    if membership is None or membership.latest_end_date is None:
        return False
    return stock.delist_date is None or stock.delist_reason is None


def _plan_reasons(
    *,
    existing: ExistingStockSnapshot | None,
    membership: MembershipSummary | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if existing is None:
        reasons.append("missing_from_stocks")
        reasons.append("need_profile_insert")
        if membership is not None and membership.latest_end_date is not None:
            reasons.append("membership_delisted")
        return tuple(reasons)

    if existing.ipo_date is None:
        reasons.append("ipo_date_null")
    if existing.shares_outstanding is None:
        reasons.append("shares_outstanding_null")
    if membership is not None and membership.latest_end_date is not None:
        if existing.delist_date is None:
            reasons.append("delist_date_null")
        if existing.delist_reason is None:
            reasons.append("delist_reason_null")
    return tuple(reasons)


def _log_plan(*, plan: Sequence[PlanItem], dry_run: bool) -> None:
    mode = "stocks metadata dry-run plan" if dry_run else "stocks metadata plan"
    if not plan:
        logger.info("{} is empty", mode)
        return

    for item in plan:
        logger.info(
            "{} action={} ticker={} reasons={} membership_end_date={} membership_reason={}",
            mode,
            item.action,
            item.ticker,
            ",".join(item.reasons),
            item.membership_end_date,
            item.membership_reason,
        )


def _execute_plan(
    *,
    plan: Sequence[PlanItem],
    membership_by_ticker: dict[str, MembershipSummary],
    client: FMPMetadataClient,
) -> RunSummary:
    summary = RunSummary(planned=len(plan))
    session_factory = get_session_factory()

    with session_factory() as session:
        for item in plan:
            if item.action == "skip":
                summary.skipped += 1
                continue

            logger.info(
                "processing stocks metadata action={} ticker={} reasons={}",
                item.action,
                item.ticker,
                ",".join(item.reasons),
            )

            try:
                profile = client.fetch_profile(item.ticker)
            except Exception as exc:
                _record_failure(summary, ticker=item.ticker, action=item.action, error_reason=str(exc))
                logger.error("profile fetch failed ticker={} error={}", item.ticker, exc)
                continue

            if profile is None:
                _record_failure(summary, ticker=item.ticker, action=item.action, error_reason="empty_profile")
                logger.warning("profile fetch returned no rows for {}", item.ticker)
                continue

            shares_outstanding: int | None = None
            if item.action == "insert" or "shares_outstanding_null" in item.reasons:
                try:
                    shares_outstanding = client.fetch_shares_outstanding(item.ticker)
                except Exception as exc:
                    logger.warning(
                        "shares-float fetch failed ticker={} error={}; continuing without shares_outstanding",
                        item.ticker,
                        exc,
                    )

            try:
                if item.action == "insert":
                    created = _insert_stock(
                        session=session,
                        ticker=item.ticker,
                        profile=profile,
                        shares_outstanding=shares_outstanding,
                        membership=membership_by_ticker.get(item.ticker),
                    )
                    session.commit()
                    summary.inserted += 1
                    logger.info(
                        "inserted stocks row ticker={} ipo_date={} shares_outstanding={} delist_date={} delist_reason={}",
                        created.ticker,
                        created.ipo_date,
                        created.shares_outstanding,
                        created.delist_date,
                        created.delist_reason,
                    )
                    continue

                updated_fields = _update_stock(
                    session=session,
                    ticker=item.ticker,
                    profile=profile,
                    shares_outstanding=shares_outstanding,
                    membership=membership_by_ticker.get(item.ticker),
                )
                if not updated_fields:
                    session.rollback()
                    summary.skipped += 1
                    logger.info("no metadata changes available for {}", item.ticker)
                    continue

                session.commit()
                summary.updated += 1
                logger.info(
                    "updated stocks row ticker={} fields={}",
                    item.ticker,
                    ",".join(updated_fields),
                )
            except Exception as exc:
                session.rollback()
                _record_failure(summary, ticker=item.ticker, action=item.action, error_reason=str(exc))
                logger.error("stocks write failed ticker={} action={} error={}", item.ticker, item.action, exc)

    logger.info(
        "stocks metadata summary planned={} inserted={} updated={} skipped={} failed={}",
        summary.planned,
        summary.inserted,
        summary.updated,
        summary.skipped,
        summary.failed,
    )
    return summary


def _insert_stock(
    *,
    session: sa.orm.Session,
    ticker: str,
    profile: dict[str, Any],
    shares_outstanding: int | None,
    membership: MembershipSummary | None,
) -> Stock:
    company_name = _coerce_optional_str(profile.get("companyName")) or ticker
    sector = _coerce_optional_str(profile.get("sector"))
    industry = _coerce_optional_str(profile.get("industry"))
    ipo_date = _parse_iso_date(profile.get("ipoDate"))
    is_active = _coerce_optional_bool(profile.get("isActivelyTrading"))
    delist_date, delist_reason = _resolve_delist_fields(is_active=is_active, membership=membership)

    stock = Stock(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        industry=industry,
        ipo_date=ipo_date,
        delist_date=delist_date,
        delist_reason=delist_reason,
        shares_outstanding=shares_outstanding,
    )
    session.add(stock)
    return stock


def _update_stock(
    *,
    session: sa.orm.Session,
    ticker: str,
    profile: dict[str, Any],
    shares_outstanding: int | None,
    membership: MembershipSummary | None,
) -> list[str]:
    stock = session.get(Stock, ticker)
    if stock is None:
        raise RuntimeError(f"stocks row disappeared before update for {ticker}")

    updated_fields: list[str] = []
    ipo_date = _parse_iso_date(profile.get("ipoDate"))
    if stock.ipo_date is None and ipo_date is not None:
        stock.ipo_date = ipo_date
        updated_fields.append("ipo_date")

    if stock.shares_outstanding is None and shares_outstanding is not None:
        stock.shares_outstanding = shares_outstanding
        updated_fields.append("shares_outstanding")

    is_active = _coerce_optional_bool(profile.get("isActivelyTrading"))
    delist_date, delist_reason = _resolve_delist_fields(is_active=is_active, membership=membership)
    if stock.delist_date is None and delist_date is not None:
        stock.delist_date = delist_date
        updated_fields.append("delist_date")
    if stock.delist_reason is None and delist_reason is not None:
        stock.delist_reason = delist_reason
        updated_fields.append("delist_reason")

    return updated_fields


def _resolve_delist_fields(
    *,
    is_active: bool | None,
    membership: MembershipSummary | None,
) -> tuple[date | None, str | None]:
    if is_active is not False or membership is None:
        return None, None
    return membership.latest_end_date, membership.latest_reason


def _record_failure(
    summary: RunSummary,
    *,
    ticker: str,
    action: str,
    error_reason: str,
) -> None:
    summary.failed += 1
    summary.failures.append(
        FailureRecord(
            ticker=ticker,
            action=action,
            error_reason=error_reason,
        ),
    )


def _write_failures_csv(path: Path, failures: Sequence[FailureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "action", "error_reason"])
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "ticker": failure.ticker,
                    "action": failure.action,
                    "error_reason": failure.error_reason,
                },
            )
    logger.info("wrote {} failures to {}", len(failures), path)


if __name__ == "__main__":
    raise SystemExit(main())
