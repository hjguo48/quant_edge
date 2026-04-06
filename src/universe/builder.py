from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import UniverseMembership
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSourceError
from src.universe.history import get_historical_members


@dataclass(frozen=True)
class UniverseChangeEvent:
    effective_date: date
    added_ticker: str | None
    removed_ticker: str | None
    reason: str | None
    source: str


def build_universe(
    as_of: date | datetime,
    index_name: str = "SP500",
    min_adv_usd: float = 50_000_000,
) -> list[str]:
    as_of_ts = _coerce_as_of(as_of)
    as_of_date = as_of_ts.date()

    if as_of_date < date.today():
        constituents = get_historical_members(as_of_ts, index_name=index_name)
        if not constituents:
            raise ValueError(
                "Historical universe membership is missing. Refusing to backfill from current constituents.",
            )
    else:
        constituents = _fetch_index_constituents(index_name=index_name)

    filtered = _filter_by_adv(constituents, as_of_ts, min_adv_usd=min_adv_usd)
    if as_of_date >= date.today():
        _sync_membership(filtered, as_of_date=as_of_date, index_name=index_name, reason="rebalance")
    logger.info(
        "built {} universe with {} members after ADV filtering",
        index_name,
        len(filtered),
    )
    return filtered


def backfill_universe_membership(
    start_date: date | datetime,
    end_date: date | datetime,
    *,
    index_name: str = "SP500",
    strict_fmp: bool = False,
) -> int:
    start = start_date.date() if isinstance(start_date, datetime) else start_date
    end = end_date.date() if isinstance(end_date, datetime) else end_date
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    current_constituents = _fetch_index_constituents(index_name=index_name, strict_fmp=strict_fmp)
    change_events = _fetch_index_change_events(index_name=index_name, strict_fmp=strict_fmp)
    membership_rows = _reconstruct_membership_rows(
        current_constituents=current_constituents,
        change_events=change_events,
        start_date=start,
        end_date=end,
        index_name=index_name,
    )
    _replace_membership_rows(
        membership_rows,
        start_date=start,
        end_date=end,
        index_name=index_name,
    )
    logger.info(
        "backfilled {} historical universe membership rows for {} between {} and {}",
        len(membership_rows),
        index_name,
        start,
        end,
    )
    return len(membership_rows)


def _fetch_index_constituents(index_name: str, *, strict_fmp: bool = False) -> list[str]:
    if index_name.upper() != "SP500":
        raise ValueError(f"Unsupported index_name={index_name!r}. Only SP500 is implemented.")

    if settings.FMP_API_KEY:
        try:
            return _fetch_sp500_from_fmp()
        except Exception as exc:
            if strict_fmp:
                raise
            logger.warning("FMP constituent fetch failed, falling back to Wikipedia: {}", exc)

    return _fetch_sp500_from_wikipedia()


def _fetch_index_change_events(index_name: str, *, strict_fmp: bool = False) -> list[UniverseChangeEvent]:
    if index_name.upper() != "SP500":
        raise ValueError(f"Unsupported index_name={index_name!r}. Only SP500 is implemented.")

    if settings.FMP_API_KEY:
        try:
            events = _fetch_sp500_historical_changes_from_fmp()
            if strict_fmp or events:
                return events
        except Exception as exc:
            if strict_fmp:
                raise
            logger.warning(
                "FMP historical constituent fetch failed, falling back to Wikipedia changes: {}",
                exc,
            )

    return _fetch_sp500_historical_changes_from_wikipedia()


def _fetch_sp500_from_fmp() -> list[str]:
    try:
        import requests
    except ImportError as exc:
        raise DataSourceError(
            "requests is not installed. Add the phase1-week2 dependency group.",
        ) from exc

    session = requests.Session()
    # FMP may reject requests routed through the local proxy/VPN path.
    session.trust_env = False
    session.headers.update({"User-Agent": "QuantEdge/0.1.0"})
    response = session.get(
        "https://financialmodelingprep.com/stable/sp500-constituent",
        params={"apikey": settings.FMP_API_KEY},
        timeout=30,
    )
    if response.status_code != 200:
        raise DataSourceError(
            f"FMP constituent request failed with HTTP {response.status_code}: {response.text[:500]}",
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise DataSourceError(f"Unexpected FMP constituent payload: {type(payload).__name__}")

    tickers = [str(item.get("symbol", "")).replace(".", "-").upper() for item in payload]
    return sorted({ticker for ticker in tickers if ticker})


def _fetch_sp500_historical_changes_from_fmp() -> list[UniverseChangeEvent]:
    try:
        import requests
    except ImportError as exc:
        raise DataSourceError(
            "requests is not installed. Add the phase1-week2 dependency group.",
        ) from exc

    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "QuantEdge/0.1.0"})
    response = session.get(
        "https://financialmodelingprep.com/stable/historical-sp500-constituent",
        params={"apikey": settings.FMP_API_KEY},
        timeout=30,
    )
    if response.status_code != 200:
        raise DataSourceError(
            f"FMP historical constituent request failed with HTTP {response.status_code}: {response.text[:500]}",
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise DataSourceError(
            f"Unexpected FMP historical constituent payload: {type(payload).__name__}",
        )

    events: list[UniverseChangeEvent] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        effective_date = pd.to_datetime(
            item.get("date")
            or item.get("changeDate")
            or item.get("dateAdded")
            or item.get("addedDate"),
            errors="coerce",
        )
        if pd.isna(effective_date):
            continue

        added_ticker = _normalize_ticker(
            item.get("addedTicker")
            or item.get("symbolAdded")
            or item.get("addedSymbol")
            or item.get("addedSecurity"),
        )
        removed_ticker = _normalize_ticker(
            item.get("removedTicker")
            or item.get("symbolRemoved")
            or item.get("removedSymbol")
            or item.get("removedSecurity"),
        )
        if added_ticker is None and removed_ticker is None:
            continue

        events.append(
            UniverseChangeEvent(
                effective_date=effective_date.date(),
                added_ticker=added_ticker,
                removed_ticker=removed_ticker,
                reason=str(item.get("reason")).strip() if item.get("reason") else None,
                source="fmp",
            ),
        )

    return sorted(events, key=lambda event: (event.effective_date, event.added_ticker or "", event.removed_ticker or ""))


def _fetch_sp500_from_wikipedia() -> list[str]:
    try:
        import requests
    except ImportError as exc:
        raise DataSourceError(
            "requests is not installed. Add the phase1-week2 dependency group.",
        ) from exc

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "QuantEdge/0.1.0"},
        timeout=30,
    )
    response.raise_for_status()
    tables = _load_wikipedia_tables(response.text)
    table = _find_wikipedia_table(
        tables,
        required_columns=("symbol", "security", "gicssector", "gicssubindustry"),
    )
    if table is None:
        raise DataSourceError("Wikipedia did not return an S&P 500 constituent table.")

    symbol_column = "Symbol" if "Symbol" in table.columns else table.columns[0]
    tickers = table[symbol_column].astype(str).str.replace(".", "-", regex=False).str.upper()
    return sorted(set(tickers))


def _fetch_sp500_historical_changes_from_wikipedia() -> list[UniverseChangeEvent]:
    try:
        import requests
    except ImportError as exc:
        raise DataSourceError(
            "requests is not installed. Add the phase1-week2 dependency group.",
        ) from exc

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "QuantEdge/0.1.0"},
        timeout=30,
    )
    response.raise_for_status()
    tables = _load_wikipedia_tables(response.text)
    if not tables:
        raise DataSourceError("Wikipedia did not return any S&P 500 tables.")

    for table in tables:
        flattened = _flatten_wikipedia_columns(table)
        normalized_columns = {column: _normalize_column_name(column) for column in flattened.columns}
        date_column = _extract_change_column(normalized_columns, required_tokens=("date",))
        added_column = _extract_change_column(
            normalized_columns,
            required_tokens=("added",),
            preferred_tokens=("ticker",),
        ) or _extract_change_column(
            normalized_columns,
            required_tokens=("added",),
            preferred_tokens=("symbol",),
        ) or _extract_change_column(normalized_columns, required_tokens=("added",))
        removed_column = _extract_change_column(
            normalized_columns,
            required_tokens=("removed",),
            preferred_tokens=("ticker",),
        ) or _extract_change_column(
            normalized_columns,
            required_tokens=("removed",),
            preferred_tokens=("symbol",),
        ) or _extract_change_column(normalized_columns, required_tokens=("removed",))
        if date_column is None or added_column is None or removed_column is None:
            continue

        reason_column = _extract_change_column(normalized_columns, required_tokens=("reason",))
        events: list[UniverseChangeEvent] = []
        for row in flattened.to_dict("records"):
            effective_date = pd.to_datetime(row.get(date_column), errors="coerce")
            if pd.isna(effective_date):
                continue

            added_ticker = _extract_ticker(row.get(added_column))
            removed_ticker = _extract_ticker(row.get(removed_column))
            if added_ticker is None and removed_ticker is None:
                continue

            reason_value = row.get(reason_column) if reason_column else None
            reason = str(reason_value).strip() if reason_value is not None and not pd.isna(reason_value) else None
            events.append(
                UniverseChangeEvent(
                    effective_date=effective_date.date(),
                    added_ticker=added_ticker,
                    removed_ticker=removed_ticker,
                    reason=reason,
                    source="wikipedia",
                ),
            )

        if events:
            return sorted(events, key=lambda event: (event.effective_date, event.added_ticker or "", event.removed_ticker or ""))

    raise DataSourceError("Wikipedia historical S&P 500 change table could not be parsed.")


def _filter_by_adv(tickers: list[str], as_of: datetime, *, min_adv_usd: float) -> list[str]:
    if not tickers:
        return []

    lookback_start = as_of.date() - timedelta(days=40)
    prices = get_prices_pit(
        tickers=tickers,
        start_date=lookback_start,
        end_date=as_of.date(),
        as_of=as_of,
    )
    if prices.empty:
        logger.warning("ADV filter found no PIT prices; returning an empty universe")
        return []

    price_frame = prices.copy()
    close_series = pd.to_numeric(price_frame["close"], errors="coerce")
    adj_close_series = pd.to_numeric(price_frame["adj_close"], errors="coerce")
    volume_series = pd.to_numeric(price_frame["volume"], errors="coerce")
    price_frame["dollar_volume"] = close_series.fillna(adj_close_series) * volume_series
    price_frame.sort_values(["ticker", "trade_date"], inplace=True)
    latest_20 = price_frame.groupby("ticker", group_keys=False).tail(20)
    adv = latest_20.groupby("ticker")["dollar_volume"].mean()
    observation_count = latest_20.groupby("ticker").size()
    liquid = adv[(adv >= min_adv_usd) & (observation_count >= 20)].index.tolist()
    return sorted(liquid)


def _reconstruct_membership_rows(
    *,
    current_constituents: list[str],
    change_events: list[UniverseChangeEvent],
    start_date: date,
    end_date: date,
    index_name: str,
) -> list[dict[str, object]]:
    active_tickers = {_normalize_ticker(ticker) for ticker in current_constituents}
    active_tickers.discard(None)
    interval_end_by_ticker: dict[str, date | None] = {
        str(ticker): None for ticker in active_tickers if ticker is not None
    }
    rows: list[dict[str, object]] = []

    for event in sorted(change_events, key=lambda item: item.effective_date, reverse=True):
        if event.effective_date > end_date:
            _apply_reverse_change(active_tickers, interval_end_by_ticker, event)
            continue
        if event.effective_date < start_date:
            break

        if event.added_ticker:
            ticker = event.added_ticker
            if ticker in active_tickers:
                rows.append(
                    {
                        "ticker": ticker,
                        "index_name": index_name,
                        "effective_date": event.effective_date,
                        "end_date": interval_end_by_ticker.get(ticker),
                        "reason": event.reason or f"{event.source}_change",
                    },
                )
                active_tickers.remove(ticker)
                interval_end_by_ticker.pop(ticker, None)

        if event.removed_ticker and event.removed_ticker not in active_tickers:
            active_tickers.add(event.removed_ticker)
            interval_end_by_ticker[event.removed_ticker] = event.effective_date

    for ticker in sorted(active_tickers):
        rows.append(
            {
                "ticker": ticker,
                "index_name": index_name,
                "effective_date": start_date,
                "end_date": interval_end_by_ticker.get(ticker),
                "reason": "historical_backfill_anchor",
            },
        )

    return sorted(rows, key=lambda row: (str(row["ticker"]), row["effective_date"]))


def _apply_reverse_change(
    active_tickers: set[str | None],
    interval_end_by_ticker: dict[str, date | None],
    event: UniverseChangeEvent,
) -> None:
    if event.added_ticker and event.added_ticker in active_tickers:
        active_tickers.remove(event.added_ticker)
        interval_end_by_ticker.pop(event.added_ticker, None)

    if event.removed_ticker and event.removed_ticker not in active_tickers:
        active_tickers.add(event.removed_ticker)
        interval_end_by_ticker[event.removed_ticker] = event.effective_date


def _replace_membership_rows(
    rows: list[dict[str, object]],
    *,
    start_date: date,
    end_date: date,
    index_name: str,
) -> None:
    session_factory = get_session_factory()

    with session_factory() as session:
        try:
            session.execute(
                sa.delete(UniverseMembership).where(
                    UniverseMembership.index_name == index_name,
                    UniverseMembership.effective_date <= end_date,
                    sa.or_(
                        UniverseMembership.end_date.is_(None),
                        UniverseMembership.end_date > start_date,
                    ),
                ),
            )

            if rows:
                statement = insert(UniverseMembership).values(rows)
                upsert = statement.on_conflict_do_update(
                    constraint="uq_universe_membership_entry",
                    set_={
                        "end_date": statement.excluded.end_date,
                        "reason": statement.excluded.reason,
                    },
                )
                session.execute(upsert)

            session.commit()
        except Exception as exc:
            session.rollback()
            logger.opt(exception=exc).error("failed to backfill historical universe membership")
            raise DataSourceError("Failed to persist historical universe membership.") from exc


def _flatten_wikipedia_columns(table: pd.DataFrame) -> pd.DataFrame:
    flattened = table.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            " ".join(str(part) for part in column if part and str(part) != "nan").strip()
            for column in flattened.columns
        ]
    else:
        flattened.columns = [str(column).strip() for column in flattened.columns]
    return flattened


def _extract_change_column(
    normalized_columns: dict[str, str],
    *,
    required_tokens: tuple[str, ...],
    preferred_tokens: tuple[str, ...] = (),
) -> str | None:
    for original, normalized in normalized_columns.items():
        if all(token in normalized for token in required_tokens) and all(
            token in normalized for token in preferred_tokens
        ):
            return original
    for original, normalized in normalized_columns.items():
        if all(token in normalized for token in required_tokens):
            return original
    return None


def _extract_ticker(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    candidates = [raw]
    for delimiter in ("(", ",", ";"):
        if delimiter in raw:
            candidates.append(raw.split(delimiter, maxsplit=1)[0].strip())
            if delimiter == "(" and ")" in raw:
                candidates.append(raw.split("(", maxsplit=1)[1].split(")", maxsplit=1)[0].strip())

    for candidate in candidates:
        normalized = _normalize_ticker(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_ticker(value: object) -> str | None:
    if value is None:
        return None
    ticker = str(value).strip().upper().replace(".", "-")
    if not ticker or ticker == "NAN":
        return None
    if not all(character.isalnum() or character == "-" for character in ticker):
        return None
    if len(ticker) > 10:
        return None
    return ticker


def _normalize_column_name(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _load_wikipedia_tables(html: str) -> list[pd.DataFrame]:
    parser = _WikipediaTableParser()
    parser.feed(html)
    return parser.to_frames()


def _find_wikipedia_table(
    tables: list[pd.DataFrame],
    *,
    required_columns: tuple[str, ...],
) -> pd.DataFrame | None:
    for table in tables:
        flattened = _flatten_wikipedia_columns(table)
        normalized = {_normalize_column_name(column) for column in flattened.columns}
        if all(column in normalized for column in required_columns):
            return flattened
    return None


class _WikipediaTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_target_table = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._current_table: list[list[str]] = []
        self._tables: list[list[list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}

        if tag == "table" and not self._in_target_table:
            classes = attr_map.get("class", "")
            if "wikitable" in classes:
                self._in_target_table = True
                self._current_table = []
            return

        if not self._in_target_table:
            return

        if tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_target_table and self._in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_target_table:
            return

        if tag in {"td", "th"} and self._in_cell:
            value = " ".join(part.strip() for part in self._current_cell if part.strip()).strip()
            self._current_row.append(value)
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if any(cell for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = []
            self._in_row = False
        elif tag == "table":
            if self._current_table:
                self._tables.append(self._current_table)
            self._current_table = []
            self._in_target_table = False

    def to_frames(self) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for rows in self._tables:
            if len(rows) < 2:
                continue
            if rows[0][:4] == ["Effective Date", "Added", "Removed", "Reason"] and len(rows) >= 3:
                header = [
                    "Effective Date",
                    "Added Ticker",
                    "Added Security",
                    "Removed Ticker",
                    "Removed Security",
                    "Reason",
                ]
                body = [row[:6] for row in rows[2:] if len(row) >= 6]
            else:
                header = rows[0]
                body = rows[1:]
            normalized_header = [column or f"column_{index}" for index, column in enumerate(header)]
            frames.append(pd.DataFrame(body, columns=normalized_header))
        return frames


def _sync_membership(
    tickers: list[str],
    *,
    as_of_date: date,
    index_name: str,
    reason: str,
) -> None:
    desired = set(tickers)
    session_factory = get_session_factory()

    with session_factory() as session:
        try:
            active_memberships = (
                session.execute(
                    sa.select(UniverseMembership).where(
                        UniverseMembership.index_name == index_name,
                        UniverseMembership.effective_date <= as_of_date,
                        sa.or_(
                            UniverseMembership.end_date.is_(None),
                            UniverseMembership.end_date > as_of_date,
                        ),
                    ),
                )
                .scalars()
                .all()
            )
            active_by_ticker = {row.ticker: row for row in active_memberships}

            for ticker, membership in active_by_ticker.items():
                if ticker not in desired:
                    membership.end_date = as_of_date
                    membership.reason = f"{reason}_removed"

            new_rows = [
                {
                    "ticker": ticker,
                    "index_name": index_name,
                    "effective_date": as_of_date,
                    "end_date": None,
                    "reason": reason,
                }
                for ticker in sorted(desired - set(active_by_ticker))
            ]
            if new_rows:
                statement = insert(UniverseMembership).values(new_rows)
                upsert = statement.on_conflict_do_update(
                    constraint="uq_universe_membership_entry",
                    set_={
                        "end_date": statement.excluded.end_date,
                        "reason": statement.excluded.reason,
                    },
                )
                session.execute(upsert)

            session.commit()
        except Exception as exc:
            session.rollback()
            logger.opt(exception=exc).error("failed to synchronize universe membership")
            raise DataSourceError("Failed to write universe membership.") from exc


def _coerce_as_of(as_of: date | datetime) -> datetime:
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=timezone.utc)
        return as_of
    return datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)
