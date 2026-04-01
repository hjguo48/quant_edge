from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
import random
import time as time_module
from typing import Any

import pandas as pd
import sqlalchemy as sa
from loguru import logger

from src.config import settings
from src.data.db.models import CorporateAction, Stock
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSourceAuthError, DataSourceError, DataSourceTransientError, RetryConfig


def adjust_for_splits(prices_df: pd.DataFrame, splits_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = prices_df.copy()
    if adjusted.empty or splits_df.empty:
        return adjusted

    _require_base_price_columns(adjusted)
    adjusted["trade_date"] = pd.to_datetime(adjusted["trade_date"]).dt.date
    split_events = splits_df.copy()
    split_events["ex_date"] = pd.to_datetime(split_events["ex_date"]).dt.date
    split_events = _ensure_action_tickers(split_events, adjusted)
    if "ratio" not in split_events.columns and {"split_from", "split_to"} <= set(split_events.columns):
        split_events["ratio"] = split_events["split_to"] / split_events["split_from"]

    adjustment_factor = pd.Series(1.0, index=adjusted.index, dtype=float)
    for split in split_events.sort_values("ex_date").itertuples(index=False):
        ratio = float(getattr(split, "ratio", 0) or 0)
        if ratio <= 0:
            continue
        mask = (adjusted["ticker"] == split.ticker) & (adjusted["trade_date"] < split.ex_date)
        adjustment_factor.loc[mask] = adjustment_factor.loc[mask] * ratio

    for column in ["open", "high", "low", "close", "adj_close"]:
        if column in adjusted.columns:
            adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") / adjustment_factor
    if "volume" in adjusted.columns:
        adjusted["volume"] = (
            pd.to_numeric(adjusted["volume"], errors="coerce") * adjustment_factor
        ).round()

    adjusted["split_adjustment_factor"] = adjustment_factor
    return adjusted


def adjust_for_dividends(prices_df: pd.DataFrame, dividends_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = prices_df.copy()
    if adjusted.empty or dividends_df.empty:
        return adjusted

    _require_base_price_columns(adjusted)
    if "close" not in adjusted.columns:
        raise ValueError("prices_df must contain close for dividend adjustments.")
    adjusted["trade_date"] = pd.to_datetime(adjusted["trade_date"]).dt.date
    adjusted.sort_values(["ticker", "trade_date"], inplace=True)

    dividend_events = dividends_df.copy()
    dividend_events["ex_date"] = pd.to_datetime(dividend_events["ex_date"]).dt.date
    dividend_events = _ensure_action_tickers(dividend_events, adjusted)
    if "cash_amount" in dividend_events.columns:
        amount_column = "cash_amount"
    elif "amount" in dividend_events.columns:
        amount_column = "amount"
    else:
        amount_column = "ratio"
    if amount_column not in dividend_events.columns:
        raise ValueError("dividends_df must contain cash_amount, amount, or ratio.")

    adjustment_factor = pd.Series(1.0, index=adjusted.index, dtype=float)
    for dividend in dividend_events.sort_values("ex_date").itertuples(index=False):
        cash_amount = float(getattr(dividend, amount_column, 0) or 0)
        if cash_amount <= 0:
            continue

        ticker_mask = adjusted["ticker"] == dividend.ticker
        prior_rows = adjusted.loc[ticker_mask & (adjusted["trade_date"] < dividend.ex_date)].sort_values(
            "trade_date",
        )
        if prior_rows.empty:
            continue

        prior_close = float(pd.to_numeric(prior_rows["close"], errors="coerce").iloc[-1])
        if prior_close <= 0:
            continue

        factor = max((prior_close - cash_amount) / prior_close, 0.0)
        mask = ticker_mask & (adjusted["trade_date"] < dividend.ex_date)
        adjustment_factor.loc[mask] = adjustment_factor.loc[mask] * factor

    for column in ["open", "high", "low", "close", "adj_close"]:
        if column in adjusted.columns:
            adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") * adjustment_factor

    adjusted["dividend_adjustment_factor"] = adjustment_factor
    return adjusted


def track_ticker_changes(old_ticker: str, new_ticker: str, effective_date: date | datetime) -> None:
    change_date = effective_date.date() if isinstance(effective_date, datetime) else effective_date
    old_symbol = old_ticker.upper()
    new_symbol = new_ticker.upper()
    session_factory = get_session_factory()

    with session_factory() as session:
        try:
            old_stock = session.get(Stock, old_symbol)
            new_stock = session.get(Stock, new_symbol)

            if old_stock is None and new_stock is None:
                raise DataSourceError(
                    f"Cannot record ticker change {old_symbol}->{new_symbol}: {old_symbol} is absent from stocks.",
                )

            if old_stock is not None:
                old_stock.delist_date = change_date
                old_stock.delist_reason = "ticker_change"

            if new_stock is None and old_stock is not None:
                new_stock = Stock(
                    ticker=new_symbol,
                    company_name=old_stock.company_name,
                    sector=old_stock.sector,
                    industry=old_stock.industry,
                    ipo_date=old_stock.ipo_date,
                    shares_outstanding=old_stock.shares_outstanding,
                )
                session.add(new_stock)

            session.add(
                CorporateAction(
                    ticker=new_symbol,
                    action_type="ticker_change",
                    ex_date=change_date,
                    old_ticker=old_symbol,
                    new_ticker=new_symbol,
                    details_json={"reason": "symbol_change"},
                ),
            )
            session.commit()
            logger.info("recorded ticker change {} -> {} effective {}", old_symbol, new_symbol, change_date)
        except Exception:
            session.rollback()
            raise


def fetch_corporate_actions(
    tickers: Sequence[str],
    start_date: date | datetime,
    end_date: date | datetime,
) -> pd.DataFrame:
    normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")

    start = start_date.date() if isinstance(start_date, datetime) else start_date
    end = end_date.date() if isinstance(end_date, datetime) else end_date
    session = _get_http_session()

    rows: list[dict[str, Any]] = []
    for ticker in normalized_tickers:
        rows.extend(_fetch_splits(session, ticker, start, end))
        rows.extend(_fetch_dividends(session, ticker, start, end))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "action_type",
                "ex_date",
                "ratio",
                "old_ticker",
                "new_ticker",
                "details_json",
            ],
        )

    frame.sort_values(["ticker", "ex_date", "action_type"], inplace=True)
    _persist_corporate_actions(frame, normalized_tickers, start, end)
    logger.info(
        "persisted {} corporate actions for {} tickers between {} and {}",
        len(frame),
        len(normalized_tickers),
        start,
        end,
    )
    return frame.reset_index(drop=True)


def _get_http_session() -> Any:
    if not settings.POLYGON_API_KEY:
        raise DataSourceError("POLYGON_API_KEY is required for corporate action downloads.")

    try:
        import requests
    except ImportError as exc:
        raise DataSourceError(
            "requests is not installed. Add the phase1-week2 dependency group.",
        ) from exc

    session = requests.Session()
    session.headers.update({"User-Agent": "QuantEdge/0.1.0"})
    return session


def _fetch_splits(session: Any, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _iterate_polygon_results(
        session,
        "https://api.polygon.io/v3/reference/splits",
        {"ticker": ticker, "limit": 1_000, "sort": "execution_date"},
    ):
        ex_date = pd.to_datetime(item.get("execution_date"), errors="coerce")
        if pd.isna(ex_date):
            continue
        ex_date_value = ex_date.date()
        if ex_date_value < start or ex_date_value > end:
            continue

        split_from = item.get("split_from")
        split_to = item.get("split_to")
        ratio = None
        if split_from and split_to:
            ratio = Decimal(str(split_to / split_from))

        rows.append(
            {
                "ticker": ticker,
                "action_type": "split",
                "ex_date": ex_date_value,
                "ratio": ratio,
                "old_ticker": None,
                "new_ticker": None,
                "details_json": item,
            },
        )
    return rows


def _fetch_dividends(session: Any, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _iterate_polygon_results(
        session,
        "https://api.polygon.io/v3/reference/dividends",
        {"ticker": ticker, "limit": 1_000, "sort": "ex_dividend_date"},
    ):
        ex_date = pd.to_datetime(item.get("ex_dividend_date"), errors="coerce")
        if pd.isna(ex_date):
            continue
        ex_date_value = ex_date.date()
        if ex_date_value < start or ex_date_value > end:
            continue

        cash_amount = item.get("cash_amount")
        rows.append(
            {
                "ticker": ticker,
                "action_type": "dividend",
                "ex_date": ex_date_value,
                "ratio": Decimal(str(cash_amount)) if cash_amount is not None else None,
                "old_ticker": None,
                "new_ticker": None,
                "details_json": item,
            },
        )
    return rows


def _iterate_polygon_results(session: Any, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_url: str | None = url
    next_params = params | {"apiKey": settings.POLYGON_API_KEY}

    while next_url:
        payload = _request_polygon_payload(session, next_url, next_params)
        results.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        next_params = {"apiKey": settings.POLYGON_API_KEY} if next_url else {}

    return results


def _require_base_price_columns(df: pd.DataFrame) -> None:
    required_columns = {"ticker", "trade_date"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"prices_df is missing required columns for corporate-action adjustments: {missing_columns}",
        )


def _ensure_action_tickers(action_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    if "ticker" in action_df.columns:
        normalized = action_df.copy()
        normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
        return normalized

    unique_tickers = prices_df["ticker"].dropna().astype(str).str.upper().unique()
    if len(unique_tickers) == 1:
        normalized = action_df.copy()
        normalized["ticker"] = unique_tickers[0]
        return normalized

    raise ValueError("Action data must include ticker when adjusting a multi-ticker price dataframe.")


def _request_polygon_payload(
    session: Any,
    url: str,
    params: dict[str, Any],
    *,
    retry_config: RetryConfig | None = None,
) -> dict[str, Any]:
    active_retry_config = retry_config or RetryConfig()
    delay = active_retry_config.initial_delay

    for attempt in range(1, active_retry_config.max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=30)
        except Exception as exc:
            transient_error = DataSourceTransientError(
                f"Polygon corporate action request transport failure: {exc}",
            )
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    transient_error = DataSourceTransientError(
                        f"Polygon corporate action response was not valid JSON: {exc}",
                    )
                else:
                    if not isinstance(payload, dict):
                        raise DataSourceError(
                            f"Unexpected Polygon corporate action payload type: {type(payload).__name__}",
                        )
                    return payload
            elif response.status_code in {401, 403}:
                raise DataSourceAuthError(
                    f"Polygon corporate action request failed with HTTP {response.status_code}: {response.text[:500]}",
                )
            elif response.status_code == 429 or response.status_code >= 500:
                transient_error = DataSourceTransientError(
                    f"Polygon corporate action request failed with HTTP {response.status_code}: {response.text[:500]}",
                )
            else:
                raise DataSourceError(
                    f"Polygon corporate action request failed with HTTP {response.status_code}: {response.text[:500]}",
                )

        if attempt >= active_retry_config.max_attempts:
            logger.error(
                "polygon corporate action request exhausted retries after {} attempts for {}",
                active_retry_config.max_attempts,
                url,
            )
            raise transient_error

        sleep_for = min(delay, active_retry_config.max_delay)
        jitter_multiplier = 1 + random.uniform(
            -active_retry_config.jitter,
            active_retry_config.jitter,
        )
        sleep_for = max(sleep_for * jitter_multiplier, 0.0)
        logger.warning(
            "polygon corporate action request failed on attempt {}/{} for {}: {}. Retrying in {:.2f}s",
            attempt,
            active_retry_config.max_attempts,
            url,
            transient_error,
            sleep_for,
        )
        time_module.sleep(sleep_for)
        delay = min(delay * active_retry_config.backoff_factor, active_retry_config.max_delay)

    raise DataSourceError("Unreachable Polygon retry state.")


def _persist_corporate_actions(
    frame: pd.DataFrame,
    tickers: Sequence[str],
    start: date,
    end: date,
) -> None:
    session_factory = get_session_factory()

    with session_factory() as session:
        try:
            session.execute(
                sa.delete(CorporateAction).where(
                    CorporateAction.ticker.in_(tickers),
                    CorporateAction.action_type.in_(["split", "dividend"]),
                    CorporateAction.ex_date >= start,
                    CorporateAction.ex_date <= end,
                ),
            )
            session.add_all(
                [
                    CorporateAction(
                        ticker=row.ticker,
                        action_type=row.action_type,
                        ex_date=row.ex_date,
                        ratio=row.ratio,
                        old_ticker=row.old_ticker,
                        new_ticker=row.new_ticker,
                        details_json=row.details_json,
                    )
                    for row in frame.itertuples(index=False)
                ],
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.opt(exception=exc).error("failed to persist corporate actions")
            raise DataSourceError("Failed to persist corporate actions.") from exc
