"""Polygon News + Sentiment data source.

Fetches per-ticker news articles from Polygon /v2/reference/news endpoint.
Each article includes pre-computed sentiment insights (positive/neutral/negative)
from Polygon's NLP pipeline — no local LLM needed.

PIT discipline: knowledge_time = published_utc (when article was published).
Note: Sentiment insights only available from ~2020 onwards. Pre-2020 articles lack sentiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from loguru import logger
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert, JSONB

from src.config import settings
from src.data.db.models import Base, TimestampMixin
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource

try:
    from sqlalchemy.orm import Mapped, mapped_column
except ImportError:
    class Mapped:
        @classmethod
        def __class_getitem__(cls, _item: Any) -> Any:
            return Any

    def mapped_column(*args: Any, **kwargs: Any) -> sa.Column[Any]:
        return sa.Column(*args, **kwargs)


class NewsSentiment(TimestampMixin, Base):
    """Stores per-ticker news sentiment derived from Polygon article insights."""

    __tablename__ = "news_sentiment"
    __table_args__ = (
        sa.UniqueConstraint(
            "ticker",
            "article_id",
            name="uq_news_sentiment_article",
        ),
        sa.Index(
            "idx_news_sentiment_lookup",
            "ticker",
            "published_utc",
        ),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    article_id: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    published_utc: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    title: Mapped[str | None] = mapped_column(sa.String(500))
    publisher: Mapped[str | None] = mapped_column(sa.String(100))
    sentiment: Mapped[str | None] = mapped_column(sa.String(20))
    sentiment_reasoning: Mapped[str | None] = mapped_column(sa.Text)
    knowledge_time: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    source: Mapped[str | None] = mapped_column(sa.String(20))


NEWS_SENTIMENT_COLUMNS = [
    "ticker",
    "article_id",
    "published_utc",
    "title",
    "publisher",
    "sentiment",
    "sentiment_reasoning",
    "knowledge_time",
    "source",
]


class PolygonNewsSource(DataSource):
    """Fetches news articles + sentiment from Polygon /v2/reference/news."""

    source_name = "polygon_news"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.25,
    ) -> None:
        super().__init__(
            api_key or settings.POLYGON_API_KEY,
            min_request_interval=min_request_interval,
        )
        self._http_session: Any | None = None

    def _get_session(self) -> Any:
        if self._http_session is None:
            import requests

            self._http_session = requests.Session()
        return self._http_session

    @DataSource.retryable()
    def _fetch_ticker_news(
        self,
        ticker: str,
        *,
        published_utc_gte: str | None = None,
        published_utc_lte: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch news articles for a single ticker with pagination."""
        self._before_request(f"news/{ticker}")
        session = self._get_session()

        all_articles: list[dict[str, Any]] = []
        next_url: str | None = None
        page = 0

        while True:
            if next_url:
                response = session.get(next_url, timeout=30)
            else:
                url = "https://api.polygon.io/v2/reference/news"
                params: dict[str, Any] = {
                    "ticker": ticker,
                    "limit": min(limit, 100),
                    "order": "desc",
                    "apiKey": self.api_key,
                }
                if published_utc_gte:
                    params["published_utc.gte"] = published_utc_gte
                if published_utc_lte:
                    params["published_utc.lte"] = published_utc_lte
                response = session.get(url, params=params, timeout=30)

            if not response.ok:
                self.classify_http_error(
                    response.status_code, response.text, context=f"news/{ticker}"
                )

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            all_articles.extend(results)
            page += 1

            if len(all_articles) >= limit:
                break

            next_url = data.get("next_url")
            if next_url:
                next_url = f"{next_url}&apiKey={self.api_key}"
            else:
                break

            self._throttle()

        return all_articles[:limit]

    def _parse_articles(
        self, ticker: str, articles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract per-ticker sentiment rows from articles with insights."""
        rows: list[dict[str, Any]] = []

        for article in articles:
            article_id = article.get("id", "")
            published_str = article.get("published_utc", "")
            title = article.get("title", "")
            publisher_info = article.get("publisher", {})
            publisher_name = (
                publisher_info.get("name", "")
                if isinstance(publisher_info, dict)
                else str(publisher_info)
            )

            try:
                published_utc = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                continue

            insights = article.get("insights", [])
            if not insights:
                continue

            # Find the insight for this specific ticker
            ticker_upper = ticker.upper()
            for insight in insights:
                if insight.get("ticker", "").upper() == ticker_upper:
                    sentiment = insight.get("sentiment", "")
                    reasoning = insight.get("sentiment_reasoning", "")

                    rows.append(
                        {
                            "ticker": ticker_upper,
                            "article_id": article_id,
                            "published_utc": published_utc,
                            "title": title[:500] if title else None,
                            "publisher": publisher_name[:100] if publisher_name else None,
                            "sentiment": sentiment,
                            "sentiment_reasoning": reasoning[:2000] if reasoning else None,
                            "knowledge_time": published_utc,
                            "source": "polygon",
                        }
                    )
                    break

        return rows

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        all_rows: list[dict[str, Any]] = []

        for ticker in self.normalize_tickers(tickers):
            articles = self._fetch_ticker_news(
                ticker,
                published_utc_gte=f"{start}T00:00:00Z",
                published_utc_lte=f"{end}T23:59:59Z",
            )
            rows = self._parse_articles(ticker, articles)
            all_rows.extend(rows)

        frame = self.dataframe_or_empty(all_rows, NEWS_SENTIMENT_COLUMNS)
        if not frame.empty:
            self._persist_sentiment(frame)

        logger.info(
            "polygon_news fetched {} sentiment rows across {} tickers between {} and {}",
            len(frame),
            len(set(frame["ticker"])) if not frame.empty else 0,
            start,
            end,
        )
        return frame

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        cutoff = self.coerce_datetime(since_date)
        lookback_start = self.coerce_date(since_date) - timedelta(days=7)
        frame = self.fetch_historical(tickers, lookback_start, date.today())
        if frame.empty:
            return frame
        incremental = frame.loc[
            pd.to_datetime(frame["knowledge_time"], utc=True) >= cutoff
        ]
        return incremental.reset_index(drop=True)

    def health_check(self) -> bool:
        try:
            articles = self._fetch_ticker_news("AAPL", limit=1)
            return len(articles) > 0
        except Exception:
            return False

    def _persist_sentiment(self, frame: pd.DataFrame) -> None:
        """Upsert news sentiment records into the database."""
        session_factory = get_session_factory()
        with session_factory() as session:
            for _, row in frame.iterrows():
                stmt = insert(NewsSentiment).values(
                    ticker=row["ticker"],
                    article_id=row["article_id"],
                    published_utc=row["published_utc"],
                    title=row["title"],
                    publisher=row["publisher"],
                    sentiment=row["sentiment"],
                    sentiment_reasoning=row["sentiment_reasoning"],
                    knowledge_time=row["knowledge_time"],
                    source=row["source"],
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_news_sentiment_article",
                    set_={
                        "sentiment": stmt.excluded.sentiment,
                        "sentiment_reasoning": stmt.excluded.sentiment_reasoning,
                        "title": stmt.excluded.title,
                        "publisher": stmt.excluded.publisher,
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info("polygon_news persisted {} sentiment rows", len(frame))
