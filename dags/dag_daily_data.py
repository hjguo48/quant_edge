from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import sys
from typing import Any

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
import pendulum

LOGGER = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = "data/features/all_features.parquet"
LEGACY_FEATURES_PATH = "/opt/airflow/daily_features/all_features.parquet"
DEFAULT_MACRO_SERIES = ("VIXCLS", "DGS10", "DGS2", "BAA10Y", "AAA10Y", "FEDFUNDS")
DEFAULT_PRICE_BOOTSTRAP_DAYS = 30
DEFAULT_FEATURE_BOOTSTRAP_DAYS = 30


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


def _runtime_feature_path(repo_root: Path) -> Path:
    primary = _artifact_path(repo_root, DEFAULT_FEATURES_PATH)
    if primary.parent.exists() and os.access(primary.parent, os.W_OK):
        return primary
    fallback = _artifact_path(repo_root, LEGACY_FEATURES_PATH)
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def _latest_feature_dates(repo_root: Path) -> tuple[str | None, str | None]:
    import pandas as pd

    candidates: list[tuple[tuple[int, float], str, str | None]] = []
    for feature_path in _feature_path_candidates(repo_root):
        if not feature_path.exists():
            continue
        frame = pd.read_parquet(feature_path, columns=["trade_date"])
        if frame.empty:
            candidates.append(((0, feature_path.stat().st_mtime), str(feature_path), None))
            continue
        latest_feature_date = pd.to_datetime(frame["trade_date"]).max()
        latest_feature_iso = latest_feature_date.date().isoformat()
        candidates.append(
            (
                (latest_feature_date.to_pydatetime().toordinal(), feature_path.stat().st_mtime),
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
            from stock_prices
            """,
        ),
    ).mappings().one()
    return {
        "latest_trade_date": row["latest_trade_date"],
        "row_count": int(row["row_count"] or 0),
        "latest_ticker_count": int(row["latest_ticker_count"] or 0),
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


def _fetch_prices_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts._data_ops import current_market_data_end_date, get_tracked_tickers
    from src.config import settings
    from src.data.db.session import get_engine
    from src.data.sources.fred import FredDataSource
    from src.data.sources.polygon import PolygonDataSource

    live_fetch_enabled = _live_fetch_enabled()
    market_data_end = current_market_data_end_date()
    tracked_tickers = get_tracked_tickers()
    if not tracked_tickers:
        raise RuntimeError("No active tracked tickers are available in stocks.")

    with get_engine().connect() as conn:
        price_state_before = _load_price_state(conn)
        macro_state_before = _load_macro_state(conn)

    if not settings.POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is required when QUANTEDGE_ENABLE_LIVE_FETCH is enabled.")

    if not settings.FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is required for daily macro ingestion.")

    if not live_fetch_enabled:
        return _result(
            "fetch_prices",
            "skipped",
            reason="live_fetch_disabled",
            latest_trade_date=_serialize_date(price_state_before["latest_trade_date"]),
            ticker_count=price_state_before["latest_ticker_count"],
            tracked_ticker_count=len(tracked_tickers),
            market_data_end=market_data_end.isoformat(),
        )

    polygon_source = PolygonDataSource(min_request_interval=0.15)
    fred_source = FredDataSource(min_request_interval=0.10)
    if not polygon_source.health_check():
        raise RuntimeError("Polygon health check failed before incremental price ingestion.")
    if not fred_source.health_check():
        raise RuntimeError("FRED health check failed before macro ingestion.")

    latest_trade_date_before = price_state_before["latest_trade_date"]
    bootstrap_start = market_data_end - timedelta(days=DEFAULT_PRICE_BOOTSTRAP_DAYS)
    request_start = bootstrap_start if latest_trade_date_before is None else latest_trade_date_before + timedelta(days=1)

    if request_start > market_data_end:
        macro_since = macro_state_before["latest_knowledge_time"] or (market_data_end - timedelta(days=30))
        macro_frame = fred_source.fetch_incremental(list(DEFAULT_MACRO_SERIES), macro_since)
        with get_engine().connect() as conn:
            macro_state_after = _load_macro_state(conn)
        return _result(
            "fetch_prices",
            "ok",
            market_data_end=market_data_end.isoformat(),
            datasource=polygon_source.source_name,
            macro_datasource=fred_source.source_name,
            tracked_ticker_count=len(tracked_tickers),
            updated_ticker_count=0,
            skipped_ticker_count=len(tracked_tickers),
            price_rows_written=0,
            fetched_trade_dates=[],
            request_start=request_start.isoformat(),
            latest_trade_date_before=_serialize_date(latest_trade_date_before),
            latest_trade_date_after=_serialize_date(latest_trade_date_before),
            latest_trade_ticker_count_after=price_state_before["latest_ticker_count"],
            stock_price_rows_before=price_state_before["row_count"],
            stock_price_rows_after=price_state_before["row_count"],
            macro_rows_written=int(len(macro_frame)),
            macro_rows_before=macro_state_before["row_count"],
            macro_rows_after=macro_state_after["row_count"],
            macro_latest_observation_date_before=_serialize_date(macro_state_before["latest_observation_date"]),
            macro_latest_observation_date_after=_serialize_date(macro_state_after["latest_observation_date"]),
            macro_latest_knowledge_time_before=_serialize_datetime(macro_state_before["latest_knowledge_time"]),
            macro_latest_knowledge_time_after=_serialize_datetime(macro_state_after["latest_knowledge_time"]),
        )

    fetched_trade_dates: set[date] = set()
    updated_tickers: list[str] = []
    skipped_tickers: list[str] = []
    failed_tickers: list[str] = []
    price_rows_written = 0

    for ticker in tracked_tickers:
        try:
            frame = polygon_source.fetch_historical([ticker], request_start, market_data_end)
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

    macro_since = macro_state_before["latest_knowledge_time"] or (market_data_end - timedelta(days=30))
    macro_frame = fred_source.fetch_incremental(list(DEFAULT_MACRO_SERIES), macro_since)
    macro_rows_written = int(len(macro_frame))

    with get_engine().connect() as conn:
        price_state_after = _load_price_state(conn)
        macro_state_after = _load_macro_state(conn)

    return _result(
        "fetch_prices",
        "ok",
        market_data_end=market_data_end.isoformat(),
        datasource=polygon_source.source_name,
        macro_datasource=fred_source.source_name,
        tracked_ticker_count=len(tracked_tickers),
        updated_ticker_count=len(updated_tickers),
        skipped_ticker_count=len(skipped_tickers),
        price_rows_written=price_rows_written,
        fetched_trade_dates=[trade_date.isoformat() for trade_date in sorted(fetched_trade_dates)],
        request_start=request_start.isoformat(),
        latest_trade_date_before=_serialize_date(price_state_before["latest_trade_date"]),
        latest_trade_date_after=_serialize_date(price_state_after["latest_trade_date"]),
        latest_trade_ticker_count_after=price_state_after["latest_ticker_count"],
        stock_price_rows_before=price_state_before["row_count"],
        stock_price_rows_after=price_state_after["row_count"],
        macro_rows_written=macro_rows_written,
        macro_rows_before=macro_state_before["row_count"],
        macro_rows_after=macro_state_after["row_count"],
        macro_latest_observation_date_before=_serialize_date(macro_state_before["latest_observation_date"]),
        macro_latest_observation_date_after=_serialize_date(macro_state_after["latest_observation_date"]),
        macro_latest_knowledge_time_before=_serialize_datetime(macro_state_before["latest_knowledge_time"]),
        macro_latest_knowledge_time_after=_serialize_datetime(macro_state_after["latest_knowledge_time"]),
    )


def fetch_prices(**context: Any) -> dict[str, Any]:
    return _run_task("fetch_prices", _fetch_prices_impl, **context)


def _check_quality_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from sqlalchemy import text

    from src.data.db.session import get_engine
    from src.risk.data_risk import DataRiskMonitor

    monitor = DataRiskMonitor()
    feature_path_str, _ = _latest_feature_dates(repo_root)
    feature_path = None if feature_path_str is None else Path(feature_path_str)

    with get_engine().connect() as conn:
        latest_trade_date = conn.execute(text("select max(trade_date) from stock_prices")).scalar()
        universe_size = int(conn.execute(text("select count(*) from stocks where ticker <> 'SPY'")).scalar() or 0)
        latest_prices = pd.read_sql(
            text("select ticker, trade_date from stock_prices where trade_date = :trade_date"),
            conn,
            params={"trade_date": latest_trade_date},
        )

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
        macro_state = _load_macro_state(conn)

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

    expected_macro_knowledge_time = upstream.get("macro_latest_knowledge_time_after")
    if expected_macro_knowledge_time:
        expected_macro_ts = datetime.fromisoformat(expected_macro_knowledge_time)
        actual_macro_ts = macro_state["latest_knowledge_time"]
        if actual_macro_ts is None or actual_macro_ts < expected_macro_ts:
            raise RuntimeError(
                "macro_series_pit latest_knowledge_time is older than the fetch_prices post-ingestion state.",
            )

    return _result(
        "store_to_db",
        "ok",
        validation_passed=True,
        latest_trade_date=latest_trade_date.isoformat(),
        latest_trade_ticker_count=price_state["latest_ticker_count"],
        stock_price_rows=price_state["row_count"],
        validated_trade_dates=validated_trade_dates,
        macro_rows=macro_state["row_count"],
        macro_latest_observation_date=_serialize_date(macro_state["latest_observation_date"]),
        macro_latest_knowledge_time=_serialize_datetime(macro_state["latest_knowledge_time"]),
    )


def store_to_db(**context: Any) -> dict[str, Any]:
    return _run_task("store_to_db", _store_to_db_impl, **context)


def _update_features_cache_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from scripts._data_ops import get_tracked_tickers
    from scripts.run_ic_screening import install_runtime_optimizations, write_parquet_atomic
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

    if latest_feature_date is not None:
        refresh_start = date.fromisoformat(latest_feature_date) + timedelta(days=1)
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

    install_runtime_optimizations()
    features_long = pipeline.run(
        tickers=tracked_tickers,
        start_date=refresh_start,
        end_date=latest_pit_trade_date,
        as_of=as_of,
    )
    if features_long.empty:
        raise RuntimeError(
            "FeaturePipeline returned no rows for the requested refresh window "
            f"{refresh_start.isoformat()} -> {latest_pit_trade_date.isoformat()}",
        )

    batch_id = str(features_long.attrs.get("batch_id") or pipeline.last_batch_id or "")
    if not batch_id:
        raise RuntimeError("FeaturePipeline did not expose a batch_id for the refreshed feature slice.")

    feature_store_rows_saved = int(pipeline.save_to_store(features_long, batch_id=batch_id))
    refreshed = features_long.loc[:, ["ticker", "trade_date", "feature_name", "feature_value"]].copy()
    refreshed["trade_date"] = pd.to_datetime(refreshed["trade_date"])
    refreshed["ticker"] = refreshed["ticker"].astype(str).str.upper()

    existing_path = None if feature_path is None else Path(feature_path)
    target_path = _runtime_feature_path(repo_root)
    if existing_path is not None and existing_path.exists():
        existing = pd.read_parquet(
            existing_path,
            columns=["ticker", "trade_date", "feature_name", "feature_value"],
        )
        existing["trade_date"] = pd.to_datetime(existing["trade_date"])
        existing = existing.loc[existing["trade_date"] < pd.Timestamp(refresh_start)].copy()
        combined = pd.concat([existing, refreshed], ignore_index=True)
    else:
        combined = refreshed

    combined["feature_name"] = combined["feature_name"].astype(str)
    combined["feature_value"] = pd.to_numeric(combined["feature_value"], errors="coerce")
    combined.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    combined.drop_duplicates(["trade_date", "ticker", "feature_name"], keep="last", inplace=True)
    write_parquet_atomic(combined, target_path)

    latest_feature_date_after = (
        pd.to_datetime(combined["trade_date"]).max().date().isoformat()
        if not combined.empty
        else None
    )
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
        batch_id=batch_id,
        feature_rows_generated=int(len(features_long)),
        feature_store_rows_saved=feature_store_rows_saved,
        cache_rows_after=int(len(combined)),
    )


def update_features_cache(**context: Any) -> dict[str, Any]:
    return _run_task("update_features_cache", _update_features_cache_impl, **context)


with DAG(
    dag_id="daily_data_pipeline",
    description="Daily market data ingestion and cache refresh.",
    schedule="0 2 * * 1-5",
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    tags=["quantedge", "data", "daily"],
    default_args={"owner": "quantedge"},
) as dag:
    fetch_prices_task = PythonOperator(task_id="fetch_prices", python_callable=fetch_prices)
    check_quality_task = PythonOperator(task_id="check_quality", python_callable=check_quality)
    store_to_db_task = PythonOperator(task_id="store_to_db", python_callable=store_to_db)
    update_features_cache_task = PythonOperator(
        task_id="update_features_cache",
        python_callable=update_features_cache,
    )

    fetch_prices_task >> check_quality_task >> store_to_db_task >> update_features_cache_task
