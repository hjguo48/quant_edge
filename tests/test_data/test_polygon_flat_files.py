from __future__ import annotations

from datetime import date
import gzip
import io

import pandas as pd

from src.data.polygon_flat_files import PolygonFlatFilesClient


class _FakeBody:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeS3Client:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.head_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, str]] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.head_calls.append((Bucket, Key))
        return {"ContentLength": len(self.payload), "ETag": '"abc123"'}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.get_calls.append((Bucket, Key))
        return {
            "Body": _FakeBody(self.payload),
            "ContentLength": len(self.payload),
            "ETag": '"abc123"',
        }


def _gzip_csv(text_payload: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
        gz.write(text_payload.encode("utf-8"))
    return buffer.getvalue()


def test_flat_file_client_downloads_and_parses_regular_session_rows() -> None:
    # Polygon bar `t` = START of 1-min window [t, t+1min).
    # Valid regular bars: t in [session_open, session_close).
    # 09:30 ET = first regular bar; 15:59 ET = last regular bar (covers 15:59-16:00).
    # 16:00 ET bar covers 16:00-16:01 (post-close) and must be excluded.
    ts_0930 = pd.Timestamp("2026-01-05 09:30", tz="America/New_York").tz_convert("UTC").value
    ts_1559 = pd.Timestamp("2026-01-05 15:59", tz="America/New_York").tz_convert("UTC").value
    ts_1600 = pd.Timestamp("2026-01-05 16:00", tz="America/New_York").tz_convert("UTC").value
    ts_1601 = pd.Timestamp("2026-01-05 16:01", tz="America/New_York").tz_convert("UTC").value
    csv_text = "\n".join(
        [
            "ticker,window_start,open,high,low,close,volume,transactions",
            f"AAPL,{ts_0930},100,101,99,100.5,1000,10",
            f"AAPL,{ts_1559},101,102,100,101.5,2000,20",
            f"AAPL,{ts_1600},101.5,102,101,101.8,1500,15",
            f"AAPL,{ts_1601},101.8,102,101,101.9,1200,12",
            f"MSFT,{ts_1601},200,201,199,200.5,500,5",
        ],
    )
    payload = _gzip_csv(csv_text)
    client = PolygonFlatFilesClient(
        access_key="access",
        secret_key="secret",
        s3_client=_FakeS3Client(payload),
    )

    header = client.head_day(date(2026, 1, 5))
    result = client.load_day(date(2026, 1, 5), universe_tickers=["AAPL"])

    assert header.source_file.endswith("2026/01/2026-01-05.csv.gz")
    assert result.rows_raw == 5
    assert result.rows_kept == 2
    assert result.tickers_loaded == 1
    assert result.frame["ticker"].tolist() == ["AAPL", "AAPL"]
    assert pd.to_datetime(result.frame["minute_ts"]).dt.tz_convert("America/New_York").dt.strftime("%H:%M").tolist() == ["09:30", "15:59"]
    assert set(pd.to_datetime(result.frame["knowledge_time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S%z")) == {"2026-01-06 21:00:00+0000"}


def test_flat_file_client_builds_expected_s3_key() -> None:
    client = PolygonFlatFilesClient(access_key="access", secret_key="secret", s3_client=_FakeS3Client(b""))

    assert client.build_s3_key(date(2026, 4, 17)) == "us_stocks_sip/minute_aggs_v1/2026/04/2026-04-17.csv.gz"
