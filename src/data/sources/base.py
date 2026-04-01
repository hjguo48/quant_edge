from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from functools import wraps
import random
import time as time_module
from threading import Lock
from typing import Any, ParamSpec, TypeVar

import pandas as pd
from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


class DataSourceError(RuntimeError):
    """Base exception raised for all datasource failures."""


class DataSourceAuthError(DataSourceError):
    """Raised when an API key is missing or rejected."""


class DataSourceTransientError(DataSourceError):
    """Raised for retryable upstream failures."""


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 5
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.2
    retry_on: tuple[type[BaseException], ...] = (
        DataSourceTransientError,
        ConnectionError,
        TimeoutError,
    )


class DataSource(ABC):
    """Shared contract and resiliency helpers for external data providers."""

    source_name = "base"

    def __init__(
        self,
        api_key: str,
        *,
        min_request_interval: float = 0.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self.api_key = api_key
        self.min_request_interval = min_request_interval
        self.retry_config = retry_config or RetryConfig()
        self._rate_limit_lock = Lock()
        self._last_request_at = 0.0

    @staticmethod
    def retryable(
        *,
        retry_config: RetryConfig | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(self: DataSource, *args: P.args, **kwargs: P.kwargs) -> R:
                active_config = retry_config or self.retry_config
                delay = active_config.initial_delay

                for attempt in range(1, active_config.max_attempts + 1):
                    try:
                        return func(self, *args, **kwargs)
                    except active_config.retry_on as exc:
                        if attempt >= active_config.max_attempts:
                            logger.opt(exception=exc).error(
                                "{} exhausted retries for {} after {} attempts",
                                self.source_name,
                                func.__name__,
                                active_config.max_attempts,
                            )
                            raise

                        sleep_for = min(delay, active_config.max_delay)
                        jitter_multiplier = 1 + random.uniform(
                            -active_config.jitter,
                            active_config.jitter,
                        )
                        sleep_for = max(sleep_for * jitter_multiplier, 0.0)

                        logger.warning(
                            "{} transient error in {} on attempt {}/{}: {}. "
                            "Retrying in {:.2f}s",
                            self.source_name,
                            func.__name__,
                            attempt,
                            active_config.max_attempts,
                            exc,
                            sleep_for,
                        )
                        time_module.sleep(sleep_for)
                        delay = min(delay * active_config.backoff_factor, active_config.max_delay)

                raise DataSourceError(f"Unreachable retry state in {func.__name__}")

            return wrapper

        return decorator

    @abstractmethod
    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        """Fetch and persist a bounded historical slice."""

    @abstractmethod
    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        """Fetch and persist new records since the supplied cutoff."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the datasource can serve authenticated requests."""

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise DataSourceAuthError(
                f"{self.source_name} API key is not configured in src.config settings.",
            )

    def _throttle(self) -> None:
        if self.min_request_interval <= 0:
            return

        with self._rate_limit_lock:
            elapsed = time_module.monotonic() - self._last_request_at
            remaining = self.min_request_interval - elapsed
            if remaining > 0:
                logger.debug(
                    "{} sleeping {:.2f}s to respect upstream rate limits",
                    self.source_name,
                    remaining,
                )
                time_module.sleep(remaining)
            self._last_request_at = time_module.monotonic()

    def _before_request(self, context: str) -> None:
        self._require_api_key()
        self._throttle()
        logger.debug("{} request: {}", self.source_name, context)

    @staticmethod
    def normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker))
        if not normalized:
            raise ValueError("At least one ticker/series identifier is required.")
        return normalized

    @staticmethod
    def coerce_date(value: date | datetime) -> date:
        if isinstance(value, datetime):
            return value.date()
        return value

    @staticmethod
    def coerce_datetime(value: date | datetime, *, end_of_day: bool = False) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        default_time = time.max if end_of_day else time.min
        return datetime.combine(value, default_time, tzinfo=timezone.utc)

    @staticmethod
    def dataframe_or_empty(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(columns))
        return pd.DataFrame(rows, columns=list(columns))

    @staticmethod
    def classify_http_error(status_code: int, response_text: str, *, context: str) -> None:
        message = f"{context} failed with HTTP {status_code}: {response_text[:500]}"
        if status_code in {401, 403}:
            raise DataSourceAuthError(message)
        if status_code == 429 or status_code >= 500:
            raise DataSourceTransientError(message)
        raise DataSourceError(message)
