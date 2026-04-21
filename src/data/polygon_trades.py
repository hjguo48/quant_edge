"""Polygon/Massive trades helpers.

Task 1 only provides the stable sequence fallback required by the trades schema.
The full REST client is intentionally left for Week 4 Task 4.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date, datetime
from decimal import Decimal
import hashlib
from typing import Any

import pandas as pd

from src.data.sources.base import DataSource


def stable_sequence_fallback(
    trade_id: str | None,
    price: Decimal,
    size: Decimal,
    participant_timestamp_ns: int | None,
    conditions: tuple[int, ...],
) -> int:
    key = "|".join(
        [
            trade_id or "",
            f"{price:.6f}",
            f"{size:.4f}",
            str(participant_timestamp_ns or 0),
            ",".join(str(condition) for condition in conditions),
        ],
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    int64_from_first_8 = int.from_bytes(digest[:8], "big", signed=False) & ((1 << 62) - 1)
    return -(int64_from_first_8 + 1)


class PolygonTradesClient(DataSource):
    """Placeholder client; Task 4 will implement REST pagination and retries."""

    source_name = "polygon_trades"

    def fetch_trades_for_day(
        self,
        ticker: str,
        trading_date: date,
        *,
        page_size: int | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        raise NotImplementedError("PolygonTradesClient is implemented in Week 4 Task 4.")

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("PolygonTradesClient is implemented in Week 4 Task 4.")

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("PolygonTradesClient is implemented in Week 4 Task 4.")

    def health_check(self) -> bool:
        raise NotImplementedError("PolygonTradesClient is implemented in Week 4 Task 4.")
