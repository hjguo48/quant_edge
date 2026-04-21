from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, datetime
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
from src.data.sources.base import DataSource, DataSourceError


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

    def load_day_for_tickers(self, trading_date: date | datetime, tickers: Iterable[str]) -> pd.DataFrame:
        trade_day = self.coerce_date(trading_date)
        normalized = set(self.normalize_tickers(tickers))
        if not normalized:
            return pd.DataFrame(columns=flat_files.TRADES_FLAT_COLUMNS)

        temp_root = str(self.temp_dir) if self.temp_dir is not None else None
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            local_path = Path(directory) / f"{trade_day:%Y-%m-%d}.csv.gz"
            self.download_day_to_path(trade_day, local_path)
            return self._parse_trades_file(local_path, trading_date=trade_day, tickers=normalized)

    def _parse_trades_file(
        self,
        path: Path,
        *,
        trading_date: date,
        tickers: Iterable[str],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        try:
            with gzip.open(path, mode="rt", newline="") as handle:
                reader = pd.read_csv(handle, chunksize=self.chunksize)
                for chunk in reader:
                    parsed = flat_files._parse_trades_chunk(
                        chunk,
                        trading_date=trading_date,
                        universe_tickers=set(tickers),
                    )
                    if not parsed.empty:
                        frames.append(parsed)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=flat_files.TRADES_FLAT_COLUMNS)
        if not frames:
            return pd.DataFrame(columns=flat_files.TRADES_FLAT_COLUMNS)
        frame = pd.concat(frames, ignore_index=True)
        frame.sort_values(["ticker", "sip_timestamp", "exchange", "sequence_number"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame[flat_files.TRADES_FLAT_COLUMNS]

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

        with destination.open("wb") as output:
            for chunk in _iter_body_chunks(body):
                output.write(chunk)

        return FlatFileHeader(
            source_file=f"s3://{self.bucket}/{key}",
            content_length=int(response.get("ContentLength") or destination.stat().st_size),
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
