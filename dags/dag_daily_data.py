from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    from airflow import DAG
    from airflow.exceptions import AirflowException
    from airflow.operators.python import PythonOperator
except ImportError:
    from dags._airflow_compat import DAG, AirflowException, PythonOperator
try:
    import pendulum
except ImportError:
    pendulum = None

from dags.task_groups.minute_incremental import (
    build_minute_incremental_task_group,
    minute_incremental_enabled,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = "data/features/all_features.parquet"
LEGACY_FEATURES_PATH = "/opt/airflow/daily_features/all_features.parquet"
FAST_VIX_SERIES = ("VIXCLS",)
SLOW_MACRO_SERIES = ("DGS10", "DGS2", "BAA10Y", "AAA10Y", "FEDFUNDS")
DEFAULT_PRICE_BOOTSTRAP_DAYS = 30
DEFAULT_FEATURE_BOOTSTRAP_DAYS = 30


def _dag_start_datetime(year: int, month: int, day: int) -> datetime:
    if pendulum is not None:
        return pendulum.datetime(year, month, day, tz="America/New_York")
    return datetime(year, month, day, tzinfo=timezone.utc)


def _project_root() -> Path:
    candidates = [
        os.environ.get("QUANTEDGE_REPO_ROOT"),
        str(Path(__file__).resolve().parents[1]),
        "/opt/quantedge",
        "/home/jiahao/quant_edge",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if (path / "src").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise RuntimeError("Unable to locate QuantEdge project root for DAG execution.")


def _result(step: str, status: str, **payload: Any) -> dict[str, Any]:
    return {
        "step": step,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


def _run_task(step: str, handler: Any, **context: Any) -> dict[str, Any]:
    try:
        repo_root = _project_root()
        payload = handler(repo_root=repo_root, context=context) or {}
        if not isinstance(payload, dict):
            payload = {"payload": payload}
        payload.setdefault("status", "ok")
        payload.setdefault("step", step)
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        if str(payload.get("status", "")).lower() == "error":
            raise AirflowException(payload.get("error") or f"{step} returned status=error")
        return payload
    except Exception as exc:
        LOGGER.exception("daily_data_pipeline task %s failed", step)
        raise AirflowException(str(exc)) from exc


def _artifact_path(repo_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return repo_root / relative_path


def _feature_path_candidates(repo_root: Path) -> list[Path]:
    return [
        _artifact_path(repo_root, DEFAULT_FEATURES_PATH),
        _artifact_path(repo_root, LEGACY_FEATURES_PATH),
    ]


def _fallback_feature_path(repo_root: Path) -> Path:
    fallback = _artifact_path(repo_root, LEGACY_FEATURES_PATH)
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def _runtime_feature_path(repo_root: Path, *, prefer_fallback: bool = False) -> Path:
    if prefer_fallback:
        return _fallback_feature_path(repo_root)
    primary = _artifact_path(repo_root, DEFAULT_FEATURES_PATH)
    if primary.parent.exists() and os.access(primary.parent, os.W_OK):
        return primary
    return _fallback_feature_path(repo_root)


def _latest_feature_date_for_path(feature_path: Path) -> str | None:
    import pandas as pd

    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    if pq is not None:
        parquet_file = pq.ParquetFile(feature_path)
        if parquet_file.metadata.num_rows <= 0:
            return None
        # Feature cache files are written sorted by trade_date, so the trailing
        # row group contains the latest snapshot and avoids scanning the whole file.
        last_group = parquet_file.read_row_group(
            parquet_file.metadata.num_row_groups - 1,
            columns=["trade_date"],
        )
        if last_group.num_rows <= 0:
            return None
        latest_feature_date = pd.to_datetime(
            last_group.to_pandas()["trade_date"],
            errors="coerce",
        ).max()
        if pd.isna(latest_feature_date):
            return None
        return latest_feature_date.date().isoformat()

    frame = pd.read_parquet(feature_path, columns=["trade_date"])
    if frame.empty:
        return None
    latest_feature_date = pd.to_datetime(frame["trade_date"], errors="coerce").max()
    if pd.isna(latest_feature_date):
        return None
    return latest_feature_date.date().isoformat()


def _latest_feature_dates(repo_root: Path) -> tuple[str | None, str | None]:
    candidates: list[tuple[tuple[int, float], str, str | None]] = []
    for feature_path in _feature_path_candidates(repo_root):
        if not feature_path.exists():
            continue
        latest_feature_iso = _latest_feature_date_for_path(feature_path)
        if latest_feature_iso is None:
            candidates.append(((0, feature_path.stat().st_mtime), str(feature_path), None))
            continue
        candidates.append(
            (
                (date.fromisoformat(latest_feature_iso).toordinal(), feature_path.stat().st_mtime),
                str(feature_path),
                latest_feature_iso,
            ),
        )
    if not candidates:
        return None, None
    _, feature_path, latest_feature_iso = max(candidates, key=lambda item: item[0])
    return feature_path, latest_feature_iso


def _live_fetch_enabled() -> bool:
    return os.environ.get("QUANTEDGE_ENABLE_LIVE_FETCH", "").lower() in {"1", "true", "yes"}


def _feature_refresh_enabled() -> bool:
    return os.environ.get("QUANTEDGE_ENABLE_FEATURE_REFRESH", "").lower() in {"1", "true", "yes"}


def _fundamentals_fetch_enabled() -> bool:
    return os.environ.get("QUANTEDGE_ENABLE_FUNDAMENTALS_FETCH", "").lower() in {"1", "true", "yes"}


def _alternative_fetch_enabled() -> bool:
    return os.environ.get("QUANTEDGE_ENABLE_ALTERNATIVE_FETCH", "1").lower() in {"1", "true", "yes"}


def _serialize_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _load_price_state(conn: Any) -> dict[str, Any]:
    from sqlalchemy import text

    row = conn.execute(
        text(
            """
            select
                max(trade_date) as latest_trade_date,
                count(*) as row_count,
                count(distinct ticker) filter (
                    where trade_date = (select max(trade_date) from stock_prices)
                ) as latest_ticker_count
                ,
                count(distinct ticker) filter (
                    where trade_date = (
                        select max(trade_date)
                        from stock_prices
                        where trade_date < (select max(trade_date) from stock_prices)
                    )
                ) as previous_ticker_count
            from stock_prices
            """,
        ),
    ).mappings().one()
    return {
        "latest_trade_date": row["latest_trade_date"],
        "row_count": int(row["row_count"] or 0),
        "latest_ticker_count": int(row["latest_ticker_count"] or 0),
        "previous_ticker_count": int(row["previous_ticker_count"] or 0),
    }


def _load_macro_state(conn: Any) -> dict[str, Any]:
    from sqlalchemy import text

    row = conn.execute(
        text(
            """
            select
                count(*) as row_count,
                max(observation_date) as latest_observation_date,
                max(knowledge_time) as latest_knowledge_time
            from macro_series_pit
            """,
        ),
    ).mappings().one()
    return {
        "row_count": int(row["row_count"] or 0),
        "latest_observation_date": row["latest_observation_date"],
        "latest_knowledge_time": row["latest_knowledge_time"],
    }


def _load_macro_series_state(conn: Any, *, series_id: str) -> dict[str, Any]:
    from sqlalchemy import text

    row = conn.execute(
        text(
            """
            select
                count(*) as row_count,
                max(observation_date) as latest_observation_date,
                max(knowledge_time) as latest_knowledge_time
            from macro_series_pit
            where series_id = :series_id
            """,
        ),
        {"series_id": series_id},
    ).mappings().one()
    return {
        "row_count": int(row["row_count"] or 0),
        "latest_observation_date": row["latest_observation_date"],
        "latest_knowledge_time": row["latest_knowledge_time"],
    }


def _load_fundamentals_state(conn: Any) -> dict[str, Any]:
    from sqlalchemy import text

    row = conn.execute(
        text(
            """
            select
                count(*) as row_count,
                count(distinct ticker) as ticker_count,
                max(knowledge_time) filter (
                    where knowledge_time <= now()
                ) as latest_knowledge_time
            from fundamentals_pit
            """,
        ),
    ).mappings().one()
    return {
        "row_count": int(row["row_count"] or 0),
        "ticker_count": int(row["ticker_count"] or 0),
        "latest_knowledge_time": row["latest_knowledge_time"],
    }


def _load_active_universe_tickers(conn: Any, *, trade_date: date) -> list[str]:
    from src.universe.active import get_active_universe

    return get_active_universe(trade_date)


def _sync_universe_membership_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text

    from src.data.db.session import get_engine
    from src.universe.active import ensure_monthly_universe_membership

    with get_engine().connect() as conn:
        latest_trade_date = conn.execute(text("select max(trade_date) from stock_prices")).scalar() or date.today()

    refresh = ensure_monthly_universe_membership(latest_trade_date)
    return _result(
        "sync_universe_membership",
        "ok",
        trade_date=latest_trade_date.isoformat(),
        **refresh,
    )


def sync_universe_membership(**context: Any) -> dict[str, Any]:
    return _run_task("sync_universe_membership", _sync_universe_membership_impl, **context)


def _fetch_prices_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts._data_ops import get_tracked_tickers
    from src.config import settings
    from src.data.db.session import get_engine
    from src.data.sources.polygon import PolygonDataSource
    from src.features.sector_rotation import SECTOR_ROTATION_ETF_TICKERS

    live_fetch_enabled = _live_fetch_enabled()
    as_of = datetime.now(timezone.utc)
    tracked_tickers = sorted(set(get_tracked_tickers()) | set(SECTOR_ROTATION_ETF_TICKERS))
    if not tracked_tickers:
        raise RuntimeError("No active tracked tickers are available in stocks.")

    with get_engine().connect() as conn:
        price_state_before = _load_price_state(conn)

    if not settings.POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is required when QUANTEDGE_ENABLE_LIVE_FETCH is enabled.")

    if not live_fetch_enabled:
        return _result(
            "fetch_prices",
            "skipped",
            reason="live_fetch_disabled",
            latest_trade_date=_serialize_date(price_state_before["latest_trade_date"]),
            ticker_count=price_state_before["latest_ticker_count"],
            tracked_ticker_count=len(tracked_tickers),
            market_data_end=None,
        )

    polygon_source = PolygonDataSource(min_request_interval=0.15)
    if not polygon_source.health_check():
        raise RuntimeError("Polygon health check failed before incremental price ingestion.")

    market_data_end = polygon_source.resolve_latest_available_trade_date(reference_time=as_of)
    latest_trade_date_before = price_state_before["latest_trade_date"]
    bootstrap_start = market_data_end - timedelta(days=DEFAULT_PRICE_BOOTSTRAP_DAYS)
    latest_trade_ticker_count_before = int(price_state_before["latest_ticker_count"])
    previous_ticker_count_before = int(price_state_before["previous_ticker_count"])
    expected_complete_ticker_count = max(latest_trade_ticker_count_before, previous_ticker_count_before)
    if latest_trade_date_before is None:
        request_start = bootstrap_start
    elif expected_complete_ticker_count > 0 and latest_trade_ticker_count_before < expected_complete_ticker_count:
        request_start = latest_trade_date_before
    else:
        request_start = latest_trade_date_before + timedelta(days=1)

    if request_start > market_data_end:
        return _result(
            "fetch_prices",
            "ok",
            market_data_end=market_data_end.isoformat(),
            datasource=polygon_source.source_name,
            tracked_ticker_count=len(tracked_tickers),
            updated_ticker_count=0,
            skipped_ticker_count=len(tracked_tickers),
            price_rows_written=0,
            fetched_trade_dates=[],
            request_start=request_start.isoformat(),
            latest_trade_date_before=_serialize_date(latest_trade_date_before),
            latest_trade_date_after=_serialize_date(latest_trade_date_before),
            latest_trade_ticker_count_before=latest_trade_ticker_count_before,
            previous_trade_ticker_count_before=previous_ticker_count_before,
            latest_trade_ticker_count_after=price_state_before["latest_ticker_count"],
            stock_price_rows_before=price_state_before["row_count"],
            stock_price_rows_after=price_state_before["row_count"],
        )

    fetched_trade_dates: set[date] = set()
    updated_tickers: list[str] = []
    skipped_tickers: list[str] = []
    failed_tickers: list[str] = []
    price_rows_written = 0
    fetch_mode = "grouped_daily"
    fallback_reason = None
    tracked_ticker_set = {ticker.upper() for ticker in tracked_tickers}

    try:
        frame = polygon_source.fetch_grouped_daily_range(
            request_start,
            market_data_end,
            tickers=tracked_tickers,
            knowledge_time_mode="observed_at",
            observed_at=as_of,
        )
        if frame.empty:
            skipped_tickers = sorted(tracked_ticker_set)
        else:
            updated_ticker_set = {str(ticker).upper() for ticker in frame["ticker"].tolist()}
            updated_tickers = sorted(updated_ticker_set)
            skipped_tickers = sorted(tracked_ticker_set - updated_ticker_set)
            price_rows_written = int(len(frame))
            fetched_trade_dates.update(
                trade_date.date() if hasattr(trade_date, "date") else trade_date
                for trade_date in frame["trade_date"].tolist()
            )
            if skipped_tickers:
                for ticker in list(skipped_tickers):
                    try:
                        fallback_frame = polygon_source.fetch_historical(
                            [ticker],
                            request_start,
                            market_data_end,
                            knowledge_time_mode="observed_at",
                            observed_at=as_of,
                        )
                    except Exception:
                        LOGGER.exception(
                            "daily_data_pipeline failed to backfill missing grouped-daily ticker %s",
                            ticker,
                        )
                        failed_tickers.append(ticker)
                        continue

                    if fallback_frame.empty:
                        continue

                    updated_tickers.append(ticker)
                    price_rows_written += int(len(fallback_frame))
                    fetched_trade_dates.update(
                        trade_date.date() if hasattr(trade_date, "date") else trade_date
                        for trade_date in fallback_frame["trade_date"].tolist()
                    )
                updated_ticker_set = {ticker.upper() for ticker in updated_tickers}
                updated_tickers = sorted(updated_ticker_set)
                skipped_tickers = sorted(tracked_ticker_set - updated_ticker_set)
    except Exception as exc:
        LOGGER.exception(
            "daily_data_pipeline grouped Polygon price ingestion failed; falling back to per-ticker fetch",
        )
        fetch_mode = "per_ticker_fallback"
        fallback_reason = str(exc)
        for ticker in tracked_tickers:
            try:
                frame = polygon_source.fetch_historical(
                    [ticker],
                    request_start,
                    market_data_end,
                    knowledge_time_mode="observed_at",
                    observed_at=as_of,
                )
            except Exception:
                LOGGER.exception("daily_data_pipeline failed to fetch incremental Polygon bars for %s", ticker)
                failed_tickers.append(ticker)
                continue

            if frame.empty:
                skipped_tickers.append(ticker)
                continue

            updated_tickers.append(ticker)
            price_rows_written += int(len(frame))
            fetched_trade_dates.update(
                trade_date.date() if hasattr(trade_date, "date") else trade_date
                for trade_date in frame["trade_date"].tolist()
            )

    if failed_tickers:
        preview = ", ".join(failed_tickers[:10])
        raise RuntimeError(
            f"Polygon incremental price ingestion failed for {len(failed_tickers)} tickers: {preview}",
        )

    with get_engine().connect() as conn:
        price_state_after = _load_price_state(conn)

    return _result(
        "fetch_prices",
        "ok",
        market_data_end=market_data_end.isoformat(),
        datasource=polygon_source.source_name,
        fetch_mode=fetch_mode,
        fallback_reason=fallback_reason,
        tracked_ticker_count=len(tracked_tickers),
        updated_ticker_count=len(updated_tickers),
        skipped_ticker_count=len(skipped_tickers),
        price_rows_written=price_rows_written,
        fetched_trade_dates=[trade_date.isoformat() for trade_date in sorted(fetched_trade_dates)],
        request_start=request_start.isoformat(),
        latest_trade_date_before=_serialize_date(price_state_before["latest_trade_date"]),
        latest_trade_date_after=_serialize_date(price_state_after["latest_trade_date"]),
        latest_trade_ticker_count_before=latest_trade_ticker_count_before,
        previous_trade_ticker_count_before=previous_ticker_count_before,
        latest_trade_ticker_count_after=price_state_after["latest_ticker_count"],
        stock_price_rows_before=price_state_before["row_count"],
        stock_price_rows_after=price_state_after["row_count"],
    )


def fetch_prices(**context: Any) -> dict[str, Any]:
    return _run_task("fetch_prices", _fetch_prices_impl, **context)


def _fetch_vix_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from src.config import settings
    from src.data.db.session import get_engine
    from src.data.sources.fmp_market import FMPMarketDataSource
    from src.data.sources.polygon import PolygonDataSource

    live_fetch_enabled = _live_fetch_enabled()
    with get_engine().connect() as conn:
        vix_state_before = _load_macro_series_state(conn, series_id="VIXCLS")

    if not live_fetch_enabled:
        return _result(
            "fetch_vix",
            "skipped",
            reason="live_fetch_disabled",
            latest_observation_date=_serialize_date(vix_state_before["latest_observation_date"]),
            latest_knowledge_time=_serialize_datetime(vix_state_before["latest_knowledge_time"]),
        )

    if not settings.FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is required when QUANTEDGE_ENABLE_LIVE_FETCH is enabled.")

    polygon_source = PolygonDataSource(min_request_interval=0.15)
    if not polygon_source.health_check():
        raise RuntimeError("Polygon health check failed before fast VIX ingestion.")
    market_data_end = polygon_source.resolve_latest_available_trade_date(reference_time=datetime.now(timezone.utc))

    source = FMPMarketDataSource(min_request_interval=0.10)
    if not source.health_check():
        raise RuntimeError("FMP market health check failed before fast VIX ingestion.")

    latest_observation_date_before = vix_state_before["latest_observation_date"]
    request_start = market_data_end if latest_observation_date_before is None else latest_observation_date_before + timedelta(days=1)
    if request_start > market_data_end:
        return _result(
            "fetch_vix",
            "ok",
            datasource=source.source_name,
            request_start=request_start.isoformat(),
            latest_observation_date_before=_serialize_date(latest_observation_date_before),
            latest_observation_date_after=_serialize_date(latest_observation_date_before),
            latest_knowledge_time_before=_serialize_datetime(vix_state_before["latest_knowledge_time"]),
            latest_knowledge_time_after=_serialize_datetime(vix_state_before["latest_knowledge_time"]),
            rows_written=0,
        )

    frame = source.fetch_incremental(
        list(FAST_VIX_SERIES),
        request_start,
        end_date=market_data_end,
    )

    with get_engine().connect() as conn:
        vix_state_after = _load_macro_series_state(conn, series_id="VIXCLS")

    return _result(
        "fetch_vix",
        "ok",
        datasource=source.source_name,
        request_start=request_start.isoformat(),
        latest_observation_date_before=_serialize_date(latest_observation_date_before),
        latest_observation_date_after=_serialize_date(vix_state_after["latest_observation_date"]),
        latest_knowledge_time_before=_serialize_datetime(vix_state_before["latest_knowledge_time"]),
        latest_knowledge_time_after=_serialize_datetime(vix_state_after["latest_knowledge_time"]),
        rows_written=int(len(frame)),
    )


def fetch_vix(**context: Any) -> dict[str, Any]:
    return _run_task("fetch_vix", _fetch_vix_impl, **context)


def _fetch_macro_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from src.config import settings
    from src.data.db.session import get_engine
    from src.data.sources.fred import FredDataSource

    live_fetch_enabled = _live_fetch_enabled()
    with get_engine().connect() as conn:
        macro_state_before = _load_macro_state(conn)

    if not live_fetch_enabled:
        return _result(
            "fetch_macro",
            "skipped",
            reason="live_fetch_disabled",
            latest_observation_date=_serialize_date(macro_state_before["latest_observation_date"]),
            latest_knowledge_time=_serialize_datetime(macro_state_before["latest_knowledge_time"]),
        )

    if not settings.FRED_API_KEY:
        return _result(
            "fetch_macro",
            "warning",
            reason="missing_fred_api_key",
            latest_observation_date_before=_serialize_date(macro_state_before["latest_observation_date"]),
            latest_knowledge_time_before=_serialize_datetime(macro_state_before["latest_knowledge_time"]),
        )

    source = FredDataSource(min_request_interval=0.10)
    if not source.health_check():
        return _result(
            "fetch_macro",
            "warning",
            reason="fred_health_check_failed",
            datasource=source.source_name,
            latest_observation_date_before=_serialize_date(macro_state_before["latest_observation_date"]),
            latest_knowledge_time_before=_serialize_datetime(macro_state_before["latest_knowledge_time"]),
        )

    latest_knowledge_time_before = macro_state_before["latest_knowledge_time"]
    since_date = latest_knowledge_time_before or (datetime.now(timezone.utc) - timedelta(days=30))

    try:
        frame = source.fetch_incremental(list(SLOW_MACRO_SERIES), since_date)
        status = "ok"
        error_reason = None
        rows_written = int(len(frame))
    except Exception:
        LOGGER.exception("daily_data_pipeline slow FRED macro ingestion failed")
        status = "warning"
        error_reason = "fred_incremental_fetch_failed"
        rows_written = 0

    with get_engine().connect() as conn:
        macro_state_after = _load_macro_state(conn)

    return _result(
        "fetch_macro",
        status,
        reason=error_reason,
        datasource=source.source_name,
        requested_series=list(SLOW_MACRO_SERIES),
        rows_written=rows_written,
        latest_observation_date_before=_serialize_date(macro_state_before["latest_observation_date"]),
        latest_observation_date_after=_serialize_date(macro_state_after["latest_observation_date"]),
        latest_knowledge_time_before=_serialize_datetime(latest_knowledge_time_before),
        latest_knowledge_time_after=_serialize_datetime(macro_state_after["latest_knowledge_time"]),
    )


def fetch_macro(**context: Any) -> dict[str, Any]:
    return _run_task("fetch_macro", _fetch_macro_impl, **context)


def _fetch_fundamentals_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text

    from scripts._data_ops import current_market_data_end_date
    from src.config import settings
    from src.data.db.session import get_engine
    from src.data.sources.fmp import FMPDataSource

    market_data_end = current_market_data_end_date()
    fundamentals_enabled = _fundamentals_fetch_enabled()

    with get_engine().connect() as conn:
        latest_trade_date = conn.execute(text("select max(trade_date) from stock_prices")).scalar()
        active_universe_date = latest_trade_date or market_data_end
        active_tickers = _load_active_universe_tickers(conn, trade_date=active_universe_date)
        fundamentals_state_before = _load_fundamentals_state(conn)

    if not active_tickers:
        raise RuntimeError(
            "No active universe_membership tickers are available for fundamentals ingestion.",
        )

    if not fundamentals_enabled:
        return _result(
            "fetch_fundamentals",
            "skipped",
            reason="fundamentals_fetch_disabled",
            ticker_count=len(active_tickers),
            active_universe_date=active_universe_date.isoformat(),
            since_date=_serialize_datetime(fundamentals_state_before["latest_knowledge_time"]),
            latest_knowledge_time_before=_serialize_datetime(
                fundamentals_state_before["latest_knowledge_time"],
            ),
        )

    if not settings.FMP_API_KEY:
        raise RuntimeError(
            "FMP_API_KEY is required when QUANTEDGE_ENABLE_FUNDAMENTALS_FETCH is enabled.",
        )

    source = FMPDataSource()
    if not source.health_check():
        raise RuntimeError("FMP health check failed before incremental fundamentals ingestion.")

    latest_knowledge_time_before = fundamentals_state_before["latest_knowledge_time"]
    since_date = latest_knowledge_time_before or datetime.combine(
        market_data_end - timedelta(days=180),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )

    LOGGER.info(
        "daily_data_pipeline fundamentals ingestion starting for %s active tickers since %s",
        len(active_tickers),
        since_date.isoformat(),
    )

    successful_tickers: list[str] = []
    skipped_tickers: list[str] = []
    failed_tickers: list[str] = []
    fetched_rows = 0

    for index, ticker in enumerate(active_tickers, start=1):
        LOGGER.info(
            "daily_data_pipeline fundamentals [%s/%s] ingesting %s since %s",
            index,
            len(active_tickers),
            ticker,
            since_date.isoformat(),
        )
        try:
            frame = source.fetch_incremental([ticker], since_date)
        except Exception:
            LOGGER.exception(
                "daily_data_pipeline failed to fetch incremental FMP fundamentals for %s",
                ticker,
            )
            failed_tickers.append(ticker)
            continue

        if frame.empty:
            skipped_tickers.append(ticker)
            continue

        successful_tickers.append(ticker)
        fetched_rows += int(len(frame))

    if failed_tickers:
        LOGGER.warning(
            "daily_data_pipeline fundamentals failed for %s tickers: %s",
            len(failed_tickers),
            ",".join(failed_tickers),
        )

    if len(failed_tickers) > (len(active_tickers) / 2):
        raise RuntimeError(
            "FMP incremental fundamentals ingestion failed for more than 50% of active tickers: "
            f"{len(failed_tickers)}/{len(active_tickers)}",
        )

    with get_engine().connect() as conn:
        fundamentals_state_after = _load_fundamentals_state(conn)

    LOGGER.info(
        "daily_data_pipeline fundamentals ingestion finished: new_rows=%s success=%s skipped=%s failed=%s",
        max(fundamentals_state_after["row_count"] - fundamentals_state_before["row_count"], 0),
        len(successful_tickers),
        len(skipped_tickers),
        len(failed_tickers),
    )

    return _result(
        "fetch_fundamentals",
        "ok",
        datasource=source.source_name,
        ticker_count=len(active_tickers),
        active_universe_date=active_universe_date.isoformat(),
        since_date=since_date.isoformat(),
        fetched_rows=fetched_rows,
        new_rows_written=max(fundamentals_state_after["row_count"] - fundamentals_state_before["row_count"], 0),
        successful_ticker_count=len(successful_tickers),
        skipped_ticker_count=len(skipped_tickers),
        failed_ticker_count=len(failed_tickers),
        failed_ticker_preview=failed_tickers[:25],
        latest_knowledge_time_before=_serialize_datetime(latest_knowledge_time_before),
        latest_knowledge_time_after=_serialize_datetime(
            fundamentals_state_after["latest_knowledge_time"],
        ),
        fundamentals_rows_before=fundamentals_state_before["row_count"],
        fundamentals_rows_after=fundamentals_state_after["row_count"],
        fundamentals_ticker_count_after=fundamentals_state_after["ticker_count"],
    )


def fetch_fundamentals(**context: Any) -> dict[str, Any]:
    return _run_task("fetch_fundamentals", _fetch_fundamentals_impl, **context)


def _fetch_alternative_data_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from src.config import settings

    enabled = _alternative_fetch_enabled()
    if not enabled:
        return _result("fetch_alternative_data", "skipped", reason="alternative_fetch_disabled")

    end_date = datetime.now(timezone.utc).date().isoformat()
    available_sources: list[str] = []
    missing_providers: list[str] = []
    if settings.FMP_API_KEY:
        available_sources.extend(["earnings", "insider", "analyst"])
    else:
        missing_providers.append("FMP_API_KEY")
    if settings.POLYGON_API_KEY:
        available_sources.append("short-interest")
    else:
        missing_providers.append("POLYGON_API_KEY")

    if not available_sources:
        return _result(
            "fetch_alternative_data",
            "skipped",
            reason="missing_api_keys",
            missing_providers=missing_providers,
        )

    script_path = repo_root / "scripts" / "backfill_earnings_and_news.py"
    source_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for source in available_sources:
        command = [
            sys.executable,
            str(script_path),
            "--source",
            source,
            "--incremental",
            "--current-universe",
            "--end",
            end_date,
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        stderr_lines = [line for line in completed.stderr.strip().splitlines() if line.strip()]
        stdout_lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
        preview = stderr_lines[-5:] if stderr_lines else stdout_lines[-5:]
        source_payload = {
            "source": source,
            "returncode": int(completed.returncode),
            "log_preview": preview,
        }
        source_results.append(source_payload)
        if completed.returncode != 0:
            failures.append(source)

    if failures:
        raise RuntimeError(
            "Alternative data incremental refresh failed for sources: " + ", ".join(failures),
        )

    return _result(
        "fetch_alternative_data",
        "ok",
        source_count=len(available_sources),
        sources=available_sources,
        end_date=end_date,
        source_results=source_results,
        missing_providers=missing_providers,
    )


def fetch_alternative_data(**context: Any) -> dict[str, Any]:
    return _run_task("fetch_alternative_data", _fetch_alternative_data_impl, **context)


def _check_quality_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from sqlalchemy import text

    from scripts._data_ops import get_tracked_tickers
    from src.data.db.session import get_engine
    from src.risk.data_risk import DataRiskMonitor

    monitor = DataRiskMonitor()
    as_of = datetime.now(timezone.utc)
    feature_path_str, _ = _latest_feature_dates(repo_root)
    feature_path = None if feature_path_str is None else Path(feature_path_str)

    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                select
                    max(trade_date) as latest_trade_date,
                    max(trade_date) filter (where knowledge_time <= :as_of) as latest_pit_trade_date
                from stock_prices
                """,
            ),
            {"as_of": as_of},
        ).mappings().one()
        latest_trade_date = row["latest_pit_trade_date"] or row["latest_trade_date"]
        latest_prices = pd.DataFrame(columns=["ticker", "trade_date"])
        if latest_trade_date is not None:
            result = conn.execute(
                text(
                    """
                    select ticker, trade_date
                    from stock_prices
                    where trade_date = :trade_date
                      and knowledge_time <= :as_of
                    """,
                ),
                {"trade_date": latest_trade_date, "as_of": as_of},
            )
            latest_prices = pd.DataFrame(result.fetchall(), columns=result.keys())
    universe_size = len(get_tracked_tickers())

    if latest_trade_date is None or latest_prices.empty or feature_path is None or not feature_path.exists():
        return _result(
            "check_quality",
            "skipped",
            reason="missing_prices_or_feature_cache",
            latest_trade_date=latest_trade_date.isoformat() if latest_trade_date else None,
            feature_path=feature_path_str,
        )

    feature_frame = pd.read_parquet(
        feature_path,
        columns=["trade_date", "ticker", "feature_name", "feature_value"],
    )
    feature_frame["trade_date"] = pd.to_datetime(feature_frame["trade_date"])
    latest_feature_date = feature_frame["trade_date"].max()
    if pd.notna(latest_feature_date) and latest_feature_date.date() < latest_trade_date:
        return _result(
            "check_quality",
            "skipped",
            reason="stale_feature_cache",
            latest_trade_date=latest_trade_date.isoformat(),
            latest_feature_date=latest_feature_date.date().isoformat(),
            feature_path=feature_path_str,
        )
    current_long = feature_frame.loc[feature_frame["trade_date"] == latest_feature_date].copy()
    historical_long = feature_frame.loc[feature_frame["trade_date"] < latest_feature_date].copy()
    if current_long.empty or historical_long.empty:
        return _result(
            "check_quality",
            "skipped",
            reason="insufficient_feature_history",
            latest_feature_date=latest_feature_date.date().isoformat() if pd.notna(latest_feature_date) else None,
        )

    current_features = current_long.pivot_table(
        index="ticker",
        columns="feature_name",
        values="feature_value",
        aggfunc="first",
    )
    historical_features = (
        historical_long
        .pivot_table(
            index=["trade_date", "ticker"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        )
        .sort_index()
    )

    report = monitor.run_all_checks(
        data=latest_prices,
        universe_size=universe_size,
        current_features=current_features,
        historical_features=historical_features,
        response_times=[0.18, 0.22, 0.20],
        error_count=0,
        consecutive_failures=0,
    )
    return _result(
        "check_quality",
        "ok" if not report.halt_pipeline else "warning",
        latest_trade_date=latest_trade_date.isoformat(),
        latest_feature_date=latest_feature_date.date().isoformat(),
        report=report.to_dict(),
    )


def check_quality(**context: Any) -> dict[str, Any]:
    return _run_task("check_quality", _check_quality_impl, **context)


def _store_to_db_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from src.data.db.session import get_engine

    ti = context.get("ti")
    upstream = ti.xcom_pull(task_ids="fetch_prices") if ti is not None else {}
    upstream = upstream or {}
    if upstream.get("status") == "error":
        raise RuntimeError("fetch_prices reported an error; refusing to validate DB state.")
    if upstream.get("status") == "skipped":
        return _result(
            "store_to_db",
            "skipped",
            reason=upstream.get("reason", "upstream_fetch_skipped"),
        )

    with get_engine().connect() as conn:
        price_state = _load_price_state(conn)

    latest_trade_date = price_state["latest_trade_date"]
    if latest_trade_date is None:
        raise RuntimeError("stock_prices is empty after fetch_prices completed.")

    expected_latest_trade_date = upstream.get("latest_trade_date_after")
    if expected_latest_trade_date and _serialize_date(latest_trade_date) != expected_latest_trade_date:
        raise RuntimeError(
            "stock_prices latest_trade_date does not match fetch_prices post-ingestion state: "
            f"expected {expected_latest_trade_date}, got {_serialize_date(latest_trade_date)}",
        )

    validated_trade_dates: dict[str, int] = {}
    fetched_trade_dates = [date.fromisoformat(value) for value in list(upstream.get("fetched_trade_dates") or [])]
    if fetched_trade_dates:
        from sqlalchemy import text

        start_date = min(fetched_trade_dates)
        end_date = max(fetched_trade_dates)
        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    select trade_date, count(*) as row_count
                    from stock_prices
                    where trade_date between :start_date and :end_date
                    group by trade_date
                    order by trade_date
                    """,
                ),
                {"start_date": start_date, "end_date": end_date},
            ).mappings().all()
        counts_by_date = {
            row["trade_date"]: int(row["row_count"] or 0)
            for row in rows
        }
        missing_dates = [
            trade_date.isoformat()
            for trade_date in fetched_trade_dates
            if counts_by_date.get(trade_date, 0) <= 0
        ]
        if missing_dates:
            raise RuntimeError(
                f"Validated stock_prices coverage is missing expected trade dates: {', '.join(missing_dates)}",
            )
        validated_trade_dates = {
            trade_date.isoformat(): counts_by_date.get(trade_date, 0)
            for trade_date in fetched_trade_dates
        }

    return _result(
        "store_to_db",
        "ok",
        validation_passed=True,
        latest_trade_date=latest_trade_date.isoformat(),
        latest_trade_ticker_count=price_state["latest_ticker_count"],
        stock_price_rows=price_state["row_count"],
        validated_trade_dates=validated_trade_dates,
    )


def store_to_db(**context: Any) -> dict[str, Any]:
    return _run_task("store_to_db", _store_to_db_impl, **context)


def _update_features_cache_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts._data_ops import get_tracked_tickers
    from scripts.build_feature_matrix import export_feature_panel
    from src.data.db.session import get_engine
    from src.features.pipeline import FeaturePipeline

    from sqlalchemy import text

    as_of = datetime.now(timezone.utc)
    pipeline = FeaturePipeline()
    feature_path, latest_feature_date = _latest_feature_dates(repo_root)
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                select
                    min(trade_date) as min_trade_date,
                    max(trade_date) as latest_trade_date,
                    max(trade_date) filter (where knowledge_time <= :as_of) as latest_pit_trade_date
                from stock_prices
                """,
            ),
            {"as_of": as_of},
        ).mappings().one()

    latest_trade_date = row["latest_trade_date"]
    latest_pit_trade_date = row["latest_pit_trade_date"]
    min_trade_date = row["min_trade_date"]

    latest_trade_iso = latest_trade_date.isoformat() if latest_trade_date else None
    latest_pit_trade_iso = latest_pit_trade_date.isoformat() if latest_pit_trade_date else None
    if latest_trade_iso is None or latest_pit_trade_iso is None:
        return _result("update_features_cache", "skipped", reason="no_pit_visible_price_data_available")
    if latest_feature_date and latest_feature_date >= latest_pit_trade_iso:
        return _result(
            "update_features_cache",
            "skipped",
            reason="feature_cache_current",
            latest_trade_date=latest_trade_iso,
            latest_pit_trade_date=latest_pit_trade_iso,
            latest_feature_date=latest_feature_date,
            feature_path=feature_path,
            pipeline_class=pipeline.__class__.__name__,
        )

    if not _feature_refresh_enabled():
        return _result(
            "update_features_cache",
            "skipped",
            reason="feature_refresh_disabled",
            latest_trade_date=latest_trade_iso,
            latest_pit_trade_date=latest_pit_trade_iso,
            latest_feature_date=latest_feature_date,
            feature_path=feature_path,
            pipeline_class=pipeline.__class__.__name__,
        )

    tracked_tickers = get_tracked_tickers()
    if not tracked_tickers:
        raise RuntimeError("No active tracked tickers are available for feature refresh.")
    if min_trade_date is None:
        raise RuntimeError("stock_prices is missing min_trade_date for feature refresh bootstrap.")

    latest_feature_dt = date.fromisoformat(latest_feature_date) if latest_feature_date else None
    stale_cache_gap_days = None
    if latest_feature_date is not None:
        refresh_start = latest_feature_dt + timedelta(days=1)
        stale_cache_gap_days = (latest_pit_trade_date - latest_feature_dt).days
    else:
        refresh_start = max(min_trade_date, latest_pit_trade_date - timedelta(days=DEFAULT_FEATURE_BOOTSTRAP_DAYS))
    if refresh_start > latest_pit_trade_date:
        return _result(
            "update_features_cache",
            "skipped",
            reason="feature_cache_current",
            latest_trade_date=latest_trade_iso,
            latest_pit_trade_date=latest_pit_trade_iso,
            latest_feature_date=latest_feature_date,
            feature_path=feature_path,
            pipeline_class=pipeline.__class__.__name__,
        )

    target_path = _runtime_feature_path(repo_root, prefer_fallback=False)
    export_summary = export_feature_panel(
        start_date=refresh_start,
        end_date=latest_pit_trade_date,
        as_of=as_of,
        output_path=target_path,
        batch_size=25,
        max_workers=8,
        progress_interval=100,
        sync_feature_store=True,
        clear_store_range_flag=True,
    )
    latest_feature_date_after = export_summary["end_date"]
    return _result(
        "update_features_cache",
        "ok",
        latest_trade_date=latest_trade_iso,
        latest_pit_trade_date=latest_pit_trade_iso,
        refresh_start_date=refresh_start.isoformat(),
        latest_feature_date=latest_feature_date,
        latest_feature_date_after=latest_feature_date_after,
        feature_path=str(target_path),
        pipeline_class=pipeline.__class__.__name__,
        batch_id=export_summary["slice_summaries"][-1]["batch_id"] if export_summary["slice_summaries"] else None,
        stale_cache_gap_days=stale_cache_gap_days,
        reseeded_recent_window=False,
        feature_rows_generated=int(export_summary["feature_rows_total"]),
        feature_store_rows_saved=int(export_summary["feature_store_rows_saved"]),
        cache_rows_after=int(export_summary["parquet_rows_total"]),
    )


def update_features_cache(**context: Any) -> dict[str, Any]:
    return _run_task("update_features_cache", _update_features_cache_impl, **context)


with DAG(
    dag_id="daily_data_pipeline",
    description="Overnight slow market, macro, and fundamental refresh.",
    schedule="0 2 * * 1-5",
    start_date=_dag_start_datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quantedge", "data", "daily"],
    default_args={"owner": "quantedge"},
) as dag:
    fetch_prices_task = PythonOperator(task_id="fetch_prices", python_callable=fetch_prices)
    fetch_vix_task = PythonOperator(task_id="fetch_vix", python_callable=fetch_vix)
    fetch_macro_task = PythonOperator(task_id="fetch_macro", python_callable=fetch_macro)
    sync_universe_membership_task = PythonOperator(
        task_id="sync_universe_membership",
        python_callable=sync_universe_membership,
    )
    fetch_fundamentals_task = PythonOperator(
        task_id="fetch_fundamentals",
        python_callable=fetch_fundamentals,
    )
    fetch_alternative_data_task = PythonOperator(
        task_id="fetch_alternative_data",
        python_callable=fetch_alternative_data,
    )
    check_quality_task = PythonOperator(task_id="check_quality", python_callable=check_quality)
    store_to_db_task = PythonOperator(task_id="store_to_db", python_callable=store_to_db)
    update_features_cache_task = PythonOperator(
        task_id="update_features_cache",
        python_callable=update_features_cache,
    )
    minute_incremental_group = build_minute_incremental_task_group(dag=dag)

    fetch_prices_task >> store_to_db_task
    store_to_db_task >> sync_universe_membership_task
    sync_universe_membership_task >> [fetch_fundamentals_task, fetch_alternative_data_task]
    if minute_incremental_enabled():
        sync_universe_membership_task >> minute_incremental_group
    [
        store_to_db_task,
        fetch_vix_task,
        fetch_macro_task,
        fetch_fundamentals_task,
        fetch_alternative_data_task,
    ] >> update_features_cache_task
    update_features_cache_task >> check_quality_task


with DAG(
    dag_id="market_close_fast_pipeline",
    description="Repeated post-close price and VIX refresh until same-day market data is visible.",
    schedule="10,20,30,40,50 16-20 * * 1-5",
    start_date=_dag_start_datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["quantedge", "data", "market-close", "fast"],
    default_args={"owner": "quantedge"},
) as market_close_fast_dag:
    fast_fetch_prices_task = PythonOperator(task_id="fetch_prices", python_callable=fetch_prices)
    fast_fetch_vix_task = PythonOperator(task_id="fetch_vix", python_callable=fetch_vix)
    fast_store_to_db_task = PythonOperator(task_id="store_to_db", python_callable=store_to_db)
    fast_update_features_cache_task = PythonOperator(
        task_id="update_features_cache",
        python_callable=update_features_cache,
    )
    fast_check_quality_task = PythonOperator(
        task_id="check_quality",
        python_callable=check_quality,
    )

    fast_fetch_prices_task >> fast_store_to_db_task
    [fast_store_to_db_task, fast_fetch_vix_task] >> fast_update_features_cache_task
    fast_update_features_cache_task >> fast_check_quality_task
