from __future__ import annotations

from datetime import date
from http.client import IncompleteRead
import gzip
import io
from pathlib import Path

import pandas as pd

from src.data.polygon_flat_files import TRADES_FLAT_COLUMNS, parse_trades_day_bytes
from src.data.polygon_trades_flat import PolygonTradesFlatFilesClient
from src.data.sources.base import RetryConfig


class _FakeBody:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self, size: int | None = None) -> bytes:
        return self._buffer.read() if size is None else self._buffer.read(size)


class _FakeS3Client:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.get_calls: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.get_calls.append((Bucket, Key))
        return {"Body": _FakeBody(self.payload), "ContentLength": len(self.payload), "ETag": '"abc123"'}


class _FlakyStreamingBody:
    def __init__(self, chunks: list[bytes], fail_after_chunks: int | None = None) -> None:
        self._chunks = chunks
        self._fail_after_chunks = fail_after_chunks

    def iter_chunks(self, chunk_size: int):
        for idx, chunk in enumerate(self._chunks):
            if self._fail_after_chunks is not None and idx >= self._fail_after_chunks:
                raise IncompleteRead(partial=b"partial", expected=10)
            yield chunk


class _RetryS3Client:
    def __init__(self, first_payload: bytes, second_payload: bytes) -> None:
        self._first_payload = first_payload
        self._second_payload = second_payload
        self.get_calls = 0

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.get_calls += 1
        if self.get_calls == 1:
            return {
                "Body": _FlakyStreamingBody([self._first_payload[:5], self._first_payload[5:]], fail_after_chunks=1),
                "ContentLength": len(self._first_payload),
                "ETag": '"first"',
            }
        return {
            "Body": _FlakyStreamingBody([self._second_payload]),
            "ContentLength": len(self._second_payload),
            "ETag": '"second"',
        }


def _gzip_csv(text_payload: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
        gz.write(text_payload.encode("utf-8"))
    return buffer.getvalue()


def _ns(raw: str) -> int:
    return pd.Timestamp(raw, tz="America/New_York").tz_convert("UTC").value


def _sample_payload() -> bytes:
    csv_text = "\n".join(
        [
            "ticker,conditions,correction,exchange,id,participant_timestamp,price,sequence_number,sip_timestamp,size,tape,trf_id,trf_timestamp,decimal_size",
            f"AAPL,\"[12,37]\",0,4,T1,{_ns('2022-01-03 10:00')},100.25,101,{_ns('2022-01-03 10:00')},200,1,TRF1,{_ns('2022-01-03 10:00')},200.0",
            f"MSFT,,0,1,T2,{_ns('2022-01-03 10:01')},250.50,102,{_ns('2022-01-03 10:01')},100,1,,0,100.0",
            f"TSLA,,0,1,T3,{_ns('2022-01-03 10:02')},300.00,103,{_ns('2022-01-03 10:02')},50,1,,,50.0",
        ],
    )
    return _gzip_csv(csv_text)


def test_parse_trades_day_bytes_schema_and_timestamp_contract() -> None:
    frame = parse_trades_day_bytes(_sample_payload(), trading_date=date(2022, 1, 3), universe_tickers=["AAPL", "MSFT"])

    assert list(frame.columns) == TRADES_FLAT_COLUMNS
    assert frame["ticker"].tolist() == ["AAPL", "MSFT"]
    assert str(frame["sip_timestamp"].dt.tz) == "UTC"
    assert frame.loc[frame["ticker"] == "AAPL", "conditions"].iloc[0] == [12, 37]
    assert frame.loc[frame["ticker"] == "AAPL", "trf_id"].iloc[0] == "TRF1"
    assert pd.isna(frame.loc[frame["ticker"] == "MSFT", "trf_timestamp"].iloc[0])
    assert frame["trading_date"].unique().tolist() == [date(2022, 1, 3)]


def test_load_day_for_tickers_filters_subset_and_uses_trades_prefix() -> None:
    fake_s3 = _FakeS3Client(_sample_payload())
    client = PolygonTradesFlatFilesClient(
        access_key="access",
        secret_key="secret",
        s3_client=fake_s3,
        chunksize=2,
    )

    yielded = list(client.load_day_for_tickers(date(2022, 1, 3), ["AAPL"]))

    assert [ticker for ticker, _ in yielded] == ["AAPL"]
    frame = yielded[0][1]
    assert frame["ticker"].tolist() == ["AAPL"]
    assert fake_s3.get_calls[0][1] == "us_stocks_sip/trades_v1/2022/01/2022-01-03.csv.gz"


def test_yield_per_ticker_trades_isolates_each_ticker() -> None:
    fake_s3 = _FakeS3Client(_sample_payload())
    client = PolygonTradesFlatFilesClient(
        access_key="access",
        secret_key="secret",
        s3_client=fake_s3,
        chunksize=2,
    )

    yielded = list(client.yield_per_ticker_trades(date(2022, 1, 3), ["AAPL", "MSFT"]))

    assert [ticker for ticker, _ in yielded] == ["AAPL", "MSFT"]
    for ticker, frame in yielded:
        assert frame["ticker"].unique().tolist() == [ticker]
    assert yielded[0][1]["trade_id"].tolist() == ["T1"]
    assert yielded[1][1]["trade_id"].tolist() == ["T2"]


def test_download_day_to_path_retries_streaming_incomplete_read(tmp_path: Path) -> None:
    payload = _sample_payload()
    s3 = _RetryS3Client(payload, payload)
    client = PolygonTradesFlatFilesClient(
        access_key="access",
        secret_key="secret",
        s3_client=s3,
    )
    client.retry_config = RetryConfig(max_attempts=2, initial_delay=0.0, jitter=0.0)
    destination = tmp_path / "trades.csv.gz"

    header = client.download_day_to_path(date(2022, 1, 3), destination)

    assert s3.get_calls == 2
    assert destination.read_bytes() == payload
    assert header.content_length == len(payload)
    assert header.etag == "second"


def test_parse_empty_trades_response_returns_empty_contract() -> None:
    frame = parse_trades_day_bytes(_gzip_csv(""), trading_date=date(2022, 1, 3), universe_tickers=["AAPL"])

    assert frame.empty
    assert list(frame.columns) == TRADES_FLAT_COLUMNS
