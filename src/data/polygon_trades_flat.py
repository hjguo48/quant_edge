from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, datetime
from http.client import IncompleteRead
from pathlib import Path
import gzip
import tempfile
from typing import Any

import pandas as pd

from src.data import polygon_flat_files as flat_files
from src.data.polygon_flat_files import (
    DEFAULT_TRADES_PREFIX,
    FlatFileHeader,
    PolygonFlatFilesClient,
    parse_trades_day_bytes,
)
from src.data.sources.base import DataSource, DataSourceError, DataSourceTransientError


class PolygonTradesFlatFilesClient(PolygonFlatFilesClient):
    source_name = "polygon_trades_flat"

    def __init__(
        self,
        *args,
        chunksize: int = 1_000_000,
        temp_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, prefix=DEFAULT_TRADES_PREFIX, **kwargs)
        self.chunksize = chunksize
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None

    def _parse_trades(
        self,
        data: bytes,
        *,
        trading_date: date | datetime,
        tickers: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return parse_trades_day_bytes(
            data,
            trading_date=trading_date,
            universe_tickers=tickers,
            chunksize=self.chunksize,
        )

    def load_day_for_tickers(
        self,
        trading_date: date | datetime,
        tickers: Iterable[str],
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield one ticker DataFrame at a time.

        The original day-level DataFrame return path was intentionally removed
        because top-50 mega-cap days can require 7GB+ RSS after concatenation.
        """
        return self.yield_per_ticker_trades(trading_date, tickers)

    def yield_per_ticker_trades(
        self,
        trading_date: date | datetime,
        tickers: Iterable[str],
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        trade_day = self.coerce_date(trading_date)
        normalized = tuple(dict.fromkeys(self.normalize_tickers(tickers)))
        if not normalized:
            return
        temp_root = str(self.temp_dir) if self.temp_dir is not None else None
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            work_dir = Path(directory)
            local_path = work_dir / f"{trade_day:%Y-%m-%d}.csv.gz"
            self.download_day_to_path(trade_day, local_path)
            parts_by_ticker = self._partition_trades_file_by_ticker(
                local_path,
                trading_date=trade_day,
                tickers=normalized,
                output_dir=work_dir / "ticker_parts",
            )
            for ticker in normalized:
                part_paths = parts_by_ticker.get(ticker, [])
                if not part_paths:
                    yield ticker, pd.DataFrame(columns=flat_files.TRADES_FLAT_COLUMNS)
                    continue
                frame = pd.concat((pd.read_parquet(path) for path in part_paths), ignore_index=True)
                frame.sort_values(["sip_timestamp", "exchange", "sequence_number"], inplace=True)
                frame.reset_index(drop=True, inplace=True)
                yield ticker, frame[flat_files.TRADES_FLAT_COLUMNS]

    def _partition_trades_file_by_ticker(
        self,
        path: Path,
        *,
        trading_date: date,
        tickers: Iterable[str],
        output_dir: Path,
    ) -> dict[str, list[Path]]:
        universe = set(tickers)
        parts_by_ticker: dict[str, list[Path]] = {ticker: [] for ticker in universe}
        part_counter = 0
        try:
            with gzip.open(path, mode="rt", newline="") as handle:
                reader = pd.read_csv(handle, chunksize=self.chunksize)
                for chunk in reader:
                    parsed = flat_files._parse_trades_chunk(
                        chunk,
                        trading_date=trading_date,
                        universe_tickers=universe,
                    )
                    if parsed.empty:
                        continue
                    for ticker, group in parsed.groupby("ticker", sort=False):
                        ticker_key = str(ticker).upper()
                        ticker_dir = output_dir / _safe_ticker_path(ticker_key)
                        ticker_dir.mkdir(parents=True, exist_ok=True)
                        part_path = ticker_dir / f"part_{part_counter:06d}.parquet"
                        group.to_parquet(part_path, index=False)
                        parts_by_ticker.setdefault(ticker_key, []).append(part_path)
                    part_counter += 1
        except pd.errors.EmptyDataError:
            return parts_by_ticker
        return parts_by_ticker

    @DataSource.retryable()
    def download_day_to_path(self, trading_date: date | datetime, destination: Path) -> FlatFileHeader:
        trade_day = self.coerce_date(trading_date)
        key = self.build_s3_key(trade_day)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._before_request(f"download trades flat file {key}")
        try:
            response = self._get_s3_client().get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            self._classify_s3_error(exc, context=f"Polygon trades flat-file download {key}")

        body = response.get("Body")
        if body is None:
            raise DataSourceError(f"Polygon trades flat-file download {key} returned no body")

        expected_length = int(response.get("ContentLength") or 0)
        written = 0
        try:
            # Opening with "wb" intentionally truncates partial bytes from a prior retry attempt.
            with destination.open("wb") as output:
                for chunk in _iter_body_chunks(body):
                    output.write(chunk)
                    written += len(chunk)
        except _streaming_exception_types() as exc:
            raise DataSourceTransientError(f"Polygon trades flat-file stream {key} failed after {written} bytes: {exc}") from exc

        if expected_length > 0 and written != expected_length:
            raise DataSourceTransientError(
                f"Polygon trades flat-file stream {key} incomplete: wrote {written} bytes, expected {expected_length}",
            )

        return FlatFileHeader(
            source_file=f"s3://{self.bucket}/{key}",
            content_length=expected_length or destination.stat().st_size,
            etag=str(response.get("ETag") or "").strip('"') or None,
        )


def _iter_body_chunks(body: Any, *, chunk_size: int = 8 * 1024 * 1024) -> Iterator[bytes]:
    if hasattr(body, "iter_chunks"):
        yield from body.iter_chunks(chunk_size=chunk_size)
        return
    while True:
        try:
            chunk = body.read(chunk_size)
        except TypeError:
            chunk = body.read()
        if not chunk:
            break
        yield chunk


def _safe_ticker_path(ticker: str) -> str:
    return ticker.replace("/", "_").replace("\\", "_")


def _streaming_exception_types() -> tuple[type[BaseException], ...]:
    exceptions: list[type[BaseException]] = [
        ConnectionError,
        TimeoutError,
        OSError,
        IncompleteRead,
    ]
    try:  # botocore is present in runtime, but keep imports optional for lightweight test envs.
        from botocore.exceptions import ReadTimeoutError, ResponseStreamingError
    except ImportError:  # pragma: no cover
        pass
    else:
        exceptions.extend([ReadTimeoutError, ResponseStreamingError])
    return tuple(dict.fromkeys(exceptions))
