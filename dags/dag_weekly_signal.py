from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import math
import logging
import os
from pathlib import Path
import sys
from typing import Any

try:
    import pendulum
except ModuleNotFoundError:  # pragma: no cover - local import fallback outside Airflow runtime.
    from zoneinfo import ZoneInfo

    class _PendulumFallback:
        @staticmethod
        def datetime(year: int, month: int, day: int, *, tz: str) -> datetime:
            return datetime(year, month, day, tzinfo=ZoneInfo(tz))

    pendulum = _PendulumFallback()

try:
    from airflow import DAG
    from airflow.exceptions import AirflowException
    from airflow.operators.python import PythonOperator
except ModuleNotFoundError:  # pragma: no cover - local import fallback outside Airflow runtime.
    class AirflowException(Exception):
        pass

    class DAG:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self) -> "DAG":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    class PythonOperator:  # type: ignore[override]
        def __init__(self, *, task_id: str, python_callable: Any, **kwargs: Any) -> None:
            self.task_id = task_id
            self.python_callable = python_callable
            self.kwargs = kwargs

        def __rshift__(self, other: Any) -> Any:
            return other

LOGGER = logging.getLogger(__name__)
BENCHMARK_TICKER = "SPY"
DEFAULT_BUNDLE_PATH = "data/models/fusion_model_bundle_60d.json"
DEFAULT_GREYSCALE_REPORT_DIR = "data/reports/greyscale"
DEFAULT_G4_SUMMARY_PATH = "data/reports/greyscale/g4_gate_summary.json"
LEGACY_GREYSCALE_REPORT_DIR = "/opt/airflow/live_weekly/greyscale_reports"
LEGACY_G4_SUMMARY_PATH = "/opt/airflow/live_weekly/g4_gate_summary.json"
DEFAULT_HISTORY_LOOKBACK_DAYS = 400
DEFAULT_MIN_SIGNAL_CROSS_SECTION = 50
DEFAULT_SIGNAL_LOOKBACK_POINTS = 12
DEFAULT_FUSION_TEMPERATURE = 5.0
LIVE_FEATURE_MATRIX_PATH = "data/reports/greyscale/weekly_signal_feature_matrix.parquet"
LIVE_PRICES_PATH = "data/reports/greyscale/weekly_signal_prices.parquet"
LIVE_PREDICTIONS_PATH = "data/reports/greyscale/weekly_signal_predictions.parquet"
LIVE_MANIFEST_PATH = "data/reports/greyscale/weekly_signal_state.json"
LEGACY_LIVE_FEATURE_MATRIX_PATH = "/opt/airflow/live_weekly/feature_matrix.parquet"
LEGACY_LIVE_PRICES_PATH = "/opt/airflow/live_weekly/prices.parquet"
LEGACY_LIVE_PREDICTIONS_PATH = "/opt/airflow/live_weekly/predictions.parquet"
LEGACY_LIVE_MANIFEST_PATH = "/opt/airflow/live_weekly/state.json"
REPO_PATH_ANCHORS = ("data", "src", "scripts", "dags", "configs", "mlruns")


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


def _serialize_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


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
        LOGGER.exception("weekly_signal_pipeline task %s failed", step)
        raise AirflowException(str(exc)) from exc


def _artifact_path(repo_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return repo_root / relative_path


def _signal_state_rank(path: Path, payload: dict[str, Any]) -> tuple[int, float, float]:
    feature_pipeline = dict(payload.get("feature_pipeline") or {})
    artifacts = dict(payload.get("artifacts") or {})
    richness = sum(
        int(flag)
        for flag in (
            bool(payload.get("strategy")),
            bool(payload.get("model")),
            bool(payload.get("mlflow")),
            bool(payload.get("portfolio_state")),
            bool(feature_pipeline.get("batch_id")),
            bool(artifacts.get("feature_matrix_path")),
            bool(artifacts.get("prediction_snapshot_path")),
        )
    )
    generated_raw = str(payload.get("generated_at_utc") or payload.get("timestamp_utc") or "")
    try:
        generated_ts = datetime.fromisoformat(generated_raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        generated_ts = 0.0
    return richness, generated_ts, path.stat().st_mtime


def _load_signal_state(repo_root: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, float, float], dict[str, Any]]] = []
    for path in (
        _artifact_path(repo_root, LIVE_MANIFEST_PATH),
        _artifact_path(repo_root, LEGACY_LIVE_MANIFEST_PATH),
    ):
        if path.exists():
            payload = json.loads(path.read_text())
            candidates.append((_signal_state_rank(path, payload), payload))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _runtime_artifact_path(repo_root: Path, primary_path: str, fallback_path: str) -> Path:
    primary = _artifact_path(repo_root, primary_path)
    if primary.parent.exists() and os.access(primary.parent, os.W_OK):
        return primary
    fallback = _artifact_path(repo_root, fallback_path)
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def _runtime_report_dir(repo_root: Path) -> Path:
    primary = _artifact_path(repo_root, DEFAULT_GREYSCALE_REPORT_DIR)
    if primary.exists() and os.access(primary, os.W_OK):
        return primary
    fallback = _artifact_path(repo_root, LEGACY_GREYSCALE_REPORT_DIR)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _write_signal_state(repo_root: Path, payload: dict[str, Any]) -> None:
    from scripts.run_ic_screening import write_json_atomic
    from scripts.run_single_window_validation import json_safe

    path = _runtime_artifact_path(repo_root, LIVE_MANIFEST_PATH, LEGACY_LIVE_MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, json_safe(payload))


def _load_legacy_signal_state() -> dict[str, Any] | None:
    path = Path(LEGACY_LIVE_MANIFEST_PATH)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _seed_signal_state(repo_root: Path) -> dict[str, Any]:
    return (_load_signal_state(repo_root) or _load_legacy_signal_state() or {}).copy()


def _build_runtime_strategy_metadata(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts.live_strategy import load_live_strategy_config
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()
    champion = registry.get_champion("ridge_60d")
    if champion is None or champion.metadata is None:
        raise RuntimeError("Champion model ridge_60d is unavailable; weekly signal strategy config cannot be built.")

    strategy_config = load_live_strategy_config(
        repo_root=repo_root,
        champion=champion,
        registry_tracking_uri=registry.tracking_uri,
    )
    model_state = {
        "model_name": "ic_weighted_fusion_60d",
        "base_model_name": champion.name,
        "version": int(champion.version),
        "stage": champion.stage.value,
        "run_id": champion.run_id,
        "horizon": str(champion.metadata.horizon),
        "n_features": int(champion.metadata.n_features),
        "signal_label": strategy_config.get("signal_label"),
        "load_audit": {
            "tracking_uri": registry.tracking_uri,
            "source": "weekly_signal_pipeline_runtime_strategy",
        },
    }
    mlflow_state = {
        "tracking_uri": registry.tracking_uri,
        "run_id": champion.run_id,
        "model_name": champion.name,
        "version": int(champion.version),
    }
    return strategy_config, model_state, mlflow_state


def _resolve_repo_bound_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if candidate.exists():
        return candidate

    repo_name = repo_root.name
    if repo_name in candidate.parts:
        suffix = candidate.parts[candidate.parts.index(repo_name) + 1 :]
        rebound = repo_root.joinpath(*suffix)
        if rebound.exists():
            return rebound

    for anchor in REPO_PATH_ANCHORS:
        if anchor in candidate.parts:
            suffix = candidate.parts[candidate.parts.index(anchor) :]
            rebound = repo_root.joinpath(*suffix)
            if rebound.exists():
                return rebound

    rebound = repo_root / candidate.name
    if rebound.exists():
        return rebound
    return candidate


def _load_bundle_manifest(repo_root: Path) -> dict[str, Any]:
    path = _artifact_path(repo_root, DEFAULT_BUNDLE_PATH)
    if not path.exists():
        raise RuntimeError(f"Fusion bundle manifest is missing: {path}")
    payload = json.loads(path.read_text())
    retained_features = payload.get("retained_features") or []
    if not retained_features:
        raise RuntimeError("Fusion bundle manifest does not contain retained features.")
    for model_name, model_payload in dict(payload.get("models") or {}).items():
        resolved = _resolve_repo_bound_path(repo_root, str(model_payload.get("artifact_path", "")))
        if not resolved.exists():
            raise RuntimeError(f"Fusion model artifact for {model_name} is missing: {resolved}")
        model_payload["artifact_path"] = str(resolved)
    return payload


def _load_latest_greyscale_report(repo_root: Path) -> dict[str, Any] | None:
    from scripts.run_greyscale_live import load_greyscale_reports

    reports: list[dict[str, Any]] = []
    for report_dir in (
        _artifact_path(repo_root, DEFAULT_GREYSCALE_REPORT_DIR),
        _artifact_path(repo_root, LEGACY_GREYSCALE_REPORT_DIR),
    ):
        if report_dir.exists():
            reports.extend(load_greyscale_reports(report_dir))
    if not reports:
        return None
    reports.sort(
        key=lambda report: (
            str(report.get("live_outputs", {}).get("signal_date", "")),
            str(report.get("generated_at_utc", "")),
        ),
    )
    return reports[-1]


def _load_g4_summary(repo_root: Path) -> dict[str, Any] | None:
    for path in (
        _artifact_path(repo_root, DEFAULT_G4_SUMMARY_PATH),
        _artifact_path(repo_root, LEGACY_G4_SUMMARY_PATH),
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None


def _load_live_universe(*, trade_date: Any) -> tuple[list[str], str]:
    from src.universe.active import resolve_active_universe

    return resolve_active_universe(
        trade_date,
        benchmark_ticker=BENCHMARK_TICKER,
    )


def _load_feature_matrix(repo_root: Path):
    import pandas as pd

    for path in (
        _artifact_path(repo_root, LIVE_FEATURE_MATRIX_PATH),
        _artifact_path(repo_root, LEGACY_LIVE_FEATURE_MATRIX_PATH),
    ):
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        return frame.set_index(["trade_date", "ticker"]).sort_index()
    return None


def _load_price_snapshot(repo_root: Path):
    import pandas as pd

    for path in (
        _artifact_path(repo_root, LIVE_PRICES_PATH),
        _artifact_path(repo_root, LEGACY_LIVE_PRICES_PATH),
    ):
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        return frame
    return None


def _latest_prediction_snapshot(repo_root: Path) -> dict[str, Any] | None:
    import pandas as pd

    for prediction_path in (
        _artifact_path(repo_root, LIVE_PREDICTIONS_PATH),
        _artifact_path(repo_root, LEGACY_LIVE_PREDICTIONS_PATH),
    ):
        if not prediction_path.exists():
            continue

        predictions = pd.read_parquet(prediction_path)
        if predictions.empty:
            continue

        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        latest_trade_date = predictions["trade_date"].max()
        latest = predictions.loc[predictions["trade_date"] == latest_trade_date].copy()
        latest["ticker"] = latest["ticker"].astype(str).str.upper()
        latest.sort_values("score", ascending=False, inplace=True)

        state = _load_signal_state(repo_root) or {}
        return {
            "trade_date": latest_trade_date.date().isoformat(),
            "window_ids": sorted(latest["window_id"].astype(str).unique().tolist()),
            "score_series": latest.set_index("ticker")["score"].astype(float),
            "ticker_count": int(latest["ticker"].nunique()),
            "top_tickers": latest["ticker"].head(10).tolist(),
            "frame": latest,
            "state": state,
        }
    return None


def _expected_latest_market_trade_date(as_of: datetime) -> date | None:
    from src.config import settings
    from src.data.sources.polygon import PolygonDataSource

    if not settings.POLYGON_API_KEY:
        return None

    source = PolygonDataSource(min_request_interval=0.05)
    try:
        return source.resolve_latest_available_trade_date(reference_time=as_of)
    except Exception:
        LOGGER.exception("weekly_signal_pipeline failed to resolve latest available Polygon trade date")
        return None


def _latest_visible_macro_observation_date(*, as_of: datetime, series_id: str) -> date | None:
    from sqlalchemy import text

    from src.data.db.session import get_engine

    with get_engine().connect() as conn:
        return conn.execute(
            text(
                """
                select max(observation_date)
                from macro_series_pit
                where series_id = :series_id
                  and knowledge_time <= :as_of
                """,
            ),
            {"series_id": series_id, "as_of": as_of},
        ).scalar()


def _check_data_freshness_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_live_pipeline import load_db_state

    as_of = datetime.now(timezone.utc)
    db_state = load_db_state(as_of=as_of)
    latest_pit_trade_date = db_state["latest_pit_trade_date"]
    if latest_pit_trade_date is None:
        return _result("check_data_freshness", "skipped", reason="no_pit_visible_price_data_available")

    expected_latest_trade_date = _expected_latest_market_trade_date(as_of)
    latest_vix_observation_date = _latest_visible_macro_observation_date(as_of=as_of, series_id="VIXCLS")
    if expected_latest_trade_date is not None and latest_pit_trade_date < expected_latest_trade_date:
        raise RuntimeError(
            "weekly_signal_pipeline is stale: latest_pit_trade_date="
            f"{latest_pit_trade_date.isoformat()} but Polygon exposes {expected_latest_trade_date.isoformat()}",
        )
    if expected_latest_trade_date is not None and (
        latest_vix_observation_date is None or latest_vix_observation_date < expected_latest_trade_date
    ):
        raise RuntimeError(
            "weekly_signal_pipeline is stale: latest visible VIX observation date="
            f"{latest_vix_observation_date.isoformat() if latest_vix_observation_date else 'none'} "
            f"but expected {expected_latest_trade_date.isoformat()}",
        )
    latest_pit_trade_ticker_count = int(db_state.get("latest_pit_trade_ticker_count", 0) or 0)
    expected_universe_size = int(
        db_state.get("expected_live_universe_count")
        or db_state.get("universe_membership_live_count")
        or 0,
    )
    coverage_tolerance = max(5, math.ceil(expected_universe_size * 0.01)) if expected_universe_size else 0
    minimum_required_coverage = max(expected_universe_size - coverage_tolerance, 0)
    if latest_pit_trade_ticker_count < minimum_required_coverage:
        raise RuntimeError(
            "weekly_signal_pipeline price coverage is incomplete for the latest PIT trade date: "
            f"visible_tickers={latest_pit_trade_ticker_count} expected_universe={expected_universe_size} "
            f"minimum_required={minimum_required_coverage}",
        )

    latest_report = _load_latest_greyscale_report(repo_root)
    latest_snapshot_date = None
    if latest_report is not None:
        latest_snapshot_date = latest_report.get("live_outputs", {}).get("signal_date")
    if latest_snapshot_date is None:
        state = _load_signal_state(repo_root) or {}
        latest_snapshot_date = state.get("signal_date")
    stale = bool(latest_snapshot_date and latest_snapshot_date < latest_pit_trade_date.isoformat())
    return _result(
        "check_data_freshness",
        "warning" if stale else "ok",
        as_of_utc=as_of.isoformat(),
        latest_stored_trade_date=db_state["latest_stored_trade_date"].isoformat()
        if db_state["latest_stored_trade_date"]
        else None,
        latest_pit_trade_date=latest_pit_trade_date.isoformat(),
        previous_pit_trade_date=_serialize_date(db_state.get("previous_pit_trade_date")),
        expected_latest_trade_date=_serialize_date(expected_latest_trade_date),
        latest_visible_vix_observation_date=_serialize_date(latest_vix_observation_date),
        latest_pit_trade_ticker_count=latest_pit_trade_ticker_count,
        previous_signal_snapshot_date=latest_snapshot_date,
        previous_snapshot_current=not stale,
        expected_live_universe_count=expected_universe_size,
        minimum_required_live_universe_count=minimum_required_coverage,
        universe_membership_live_count=int(db_state.get("universe_membership_live_count", 0) or 0),
    )


def check_data_freshness(**context: Any) -> dict[str, Any]:
    return _run_task("check_data_freshness", _check_data_freshness_impl, **context)


def _compute_features_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_greyscale_live import build_feature_matrix as build_greyscale_feature_matrix
    from scripts.run_greyscale_live import load_live_universe
    from scripts.run_ic_screening import install_runtime_optimizations, write_parquet_atomic
    from scripts.run_live_pipeline import load_db_state
    from scripts.run_single_window_validation import feature_matrix_to_frame
    from src.data.db.pit import get_prices_pit
    from src.features.pipeline import FeaturePipeline

    as_of = datetime.now(timezone.utc)
    bundle = _load_bundle_manifest(repo_root)
    db_state = load_db_state(as_of=as_of)
    live_trade_date = db_state["latest_pit_trade_date"]
    if live_trade_date is None:
        return _result("compute_features", "skipped", reason="no_pit_visible_price_data_available")

    live_universe = load_live_universe(
        trade_date=live_trade_date,
        as_of=as_of,
        benchmark_ticker=BENCHMARK_TICKER,
    )
    if not live_universe:
        raise RuntimeError("No live universe was available for weekly signal inference.")

    feature_start = live_trade_date - timedelta(days=DEFAULT_HISTORY_LOOKBACK_DAYS)
    latest_report = _load_latest_greyscale_report(repo_root)
    report_signal_date = None if latest_report is None else latest_report.get("live_outputs", {}).get("signal_date")
    seed_state = _seed_signal_state(repo_root)
    strategy_config, model_state, mlflow_state = _build_runtime_strategy_metadata(repo_root)

    retained_features = list(bundle["retained_features"])
    model_artifacts = {
        model_name: str(bundle["models"][model_name]["artifact_path"])
        for model_name in sorted(dict(bundle.get("models") or {}))
    }
    price_tickers = [*live_universe, BENCHMARK_TICKER]
    prices = get_prices_pit(
        tickers=price_tickers,
        start_date=feature_start,
        end_date=live_trade_date,
        as_of=as_of,
    )
    if prices.empty:
        raise RuntimeError("No PIT price snapshot was available for weekly signal feature computation.")

    install_runtime_optimizations()
    pipeline = FeaturePipeline()
    features_long = pipeline.run(
        tickers=live_universe,
        start_date=live_trade_date,
        end_date=live_trade_date,
        as_of=as_of,
    )
    if features_long.empty:
        raise RuntimeError("FeaturePipeline returned no live rows for weekly signal computation.")

    batch_id = str(features_long.attrs.get("batch_id") or pipeline.last_batch_id or "")
    if not batch_id:
        raise RuntimeError("Weekly signal feature computation did not produce a batch_id.")

    current_feature_matrix = build_greyscale_feature_matrix(
        features_long=features_long,
        retained_features=retained_features,
    )
    if current_feature_matrix.empty:
        raise RuntimeError("Weekly signal feature matrix is empty after retained-feature alignment.")

    feature_matrix_path = _runtime_artifact_path(
        repo_root,
        LIVE_FEATURE_MATRIX_PATH,
        LEGACY_LIVE_FEATURE_MATRIX_PATH,
    )
    price_snapshot_path = _runtime_artifact_path(
        repo_root,
        LIVE_PRICES_PATH,
        LEGACY_LIVE_PRICES_PATH,
    )
    write_parquet_atomic(feature_matrix_to_frame(current_feature_matrix), feature_matrix_path)
    write_parquet_atomic(prices.reset_index(drop=True), price_snapshot_path)

    state = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_utc": as_of.isoformat(),
        "signal_date": live_trade_date.isoformat(),
        "db_state": {
            "latest_stored_trade_date": db_state["latest_stored_trade_date"].isoformat()
            if db_state["latest_stored_trade_date"]
            else None,
            "latest_pit_trade_date": live_trade_date.isoformat(),
            "universe_membership_live_count": int(db_state["universe_membership_live_count"]),
        },
        "universe": {"size": int(len(live_universe)), "source": "stock_prices_pit"},
        "model_bundle": {
            "path": str(_artifact_path(repo_root, DEFAULT_BUNDLE_PATH)),
            "window_id": str(bundle["window_id"]),
            "horizon_days": int(bundle["horizon_days"]),
            "retained_features": retained_features,
            "retained_feature_count": int(len(retained_features)),
            "models": model_artifacts,
            "seed_weights": bundle.get("seed_weights", {}),
            "regime_weights": bundle.get("regime_weights", {}),
            "temperature": float(bundle.get("fusion_temperature", DEFAULT_FUSION_TEMPERATURE)),
        },
        "strategy": dict(seed_state.get("strategy") or strategy_config),
        "portfolio_state": dict(seed_state.get("portfolio_state") or {}),
        "model": dict(seed_state.get("model") or model_state),
        "mlflow": dict(seed_state.get("mlflow") or mlflow_state),
        "feature_pipeline": {
            "start_date": feature_start.isoformat(),
            "end_date": live_trade_date.isoformat(),
            "mode": "materialized_for_weekly_signal_pipeline",
            "batch_id": batch_id,
            "feature_rows_total": int(len(features_long)),
            "feature_rows_live_date": int(len(features_long)),
            "feature_matrix_shape": {
                "n_stocks": int(current_feature_matrix.shape[0]),
                "n_features": int(current_feature_matrix.shape[1]),
            },
            "expected_feature_count": int(len(retained_features)),
            "reference_feature_matrix_path": "data/features/walkforward_feature_matrix_60d.parquet",
        },
        "artifacts": {
            "report_dir": DEFAULT_GREYSCALE_REPORT_DIR,
            "monitor_summary_path": DEFAULT_G4_SUMMARY_PATH,
            "feature_matrix_path": str(feature_matrix_path),
            "price_snapshot_path": str(price_snapshot_path),
            "prediction_snapshot_path": str(
                _runtime_artifact_path(
                    repo_root,
                    LIVE_PREDICTIONS_PATH,
                    LEGACY_LIVE_PREDICTIONS_PATH,
                ),
            ),
        },
        "latest_greyscale_report": {
            "signal_date": report_signal_date,
            "report_path": None if latest_report is None else latest_report.get("_report_path"),
        },
    }
    _write_signal_state(repo_root, state)

    return _result(
        "compute_features",
        "ok",
        signal_date=live_trade_date.isoformat(),
        universe_size=int(len(live_universe)),
        universe_source="stock_prices_pit",
        retained_feature_count=int(len(retained_features)),
        retained_features=retained_features,
        bundle_window_id=str(bundle["window_id"]),
        horizon_days=int(bundle["horizon_days"]),
        existing_report_signal_date=report_signal_date,
        batch_id=batch_id,
        feature_rows_live_date=int(len(features_long)),
        feature_matrix_path=str(feature_matrix_path),
        price_snapshot_path=str(price_snapshot_path),
    )


def compute_features(**context: Any) -> dict[str, Any]:
    return _run_task("compute_features", _compute_features_impl, **context)


def _model_inference_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from scripts.run_greyscale_live import main as run_greyscale_live_main
    from scripts.run_ic_screening import write_parquet_atomic
    from scripts.run_live_pipeline import load_db_state

    state = _load_signal_state(repo_root)
    if not state:
        raise RuntimeError("Live signal manifest is missing; compute_features must run first.")
    as_of = datetime.now(timezone.utc)
    db_state = load_db_state(as_of=as_of)
    live_trade_date = db_state["latest_pit_trade_date"]
    if live_trade_date is None:
        return _result("model_inference", "skipped", reason="no_pit_visible_price_data_available")

    latest_report = _load_latest_greyscale_report(repo_root)
    execution_mode = "generated_new_report"
    if latest_report is not None and latest_report.get("live_outputs", {}).get("signal_date") == live_trade_date.isoformat():
        report = latest_report
        execution_mode = "reused_existing_report"
    else:
        status_code = run_greyscale_live_main(
            [
                "--bundle-path",
                DEFAULT_BUNDLE_PATH,
                "--report-dir",
                DEFAULT_GREYSCALE_REPORT_DIR,
                "--as-of",
                as_of.isoformat(),
                "--dry-run",
            ],
        )
        if status_code != 0:
            raise RuntimeError(f"run_greyscale_live.py exited with status {status_code}")
        report = _load_latest_greyscale_report(repo_root)
        if report is None:
            raise RuntimeError("Greyscale live runner completed without producing a weekly report.")

    fusion_scores = dict(report.get("score_vectors", {}).get("fusion") or {})
    top_fusion_scores = list(report.get("live_outputs", {}).get("top_10_fusion_scores") or [])
    signal_date = str(report.get("live_outputs", {}).get("signal_date"))
    if not fusion_scores or not signal_date:
        raise RuntimeError("Greyscale report is missing fusion scores or signal_date.")

    prediction_rows: list[dict[str, Any]] = []
    for ticker, score in sorted(
        fusion_scores.items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    ):
        row = {
            "window_id": f"GREYSCALE_{signal_date.replace('-', '')}",
            "trade_date": signal_date,
            "ticker": str(ticker).upper(),
            "score": float(score),
        }
        for model_name in ("ridge", "xgboost", "lightgbm"):
            model_score = (report.get("score_vectors", {}).get(model_name) or {}).get(ticker)
            if model_score is not None:
                row[f"score_{model_name}"] = float(model_score)
        prediction_rows.append(row)

    prediction_frame = pd.DataFrame(prediction_rows).sort_values("score", ascending=False).reset_index(drop=True)
    prediction_path = _runtime_artifact_path(
        repo_root,
        LIVE_PREDICTIONS_PATH,
        LEGACY_LIVE_PREDICTIONS_PATH,
    )
    write_parquet_atomic(prediction_frame, prediction_path)

    if not state.get("strategy") or not state.get("model") or not state.get("mlflow"):
        strategy_config, model_state, mlflow_state = _build_runtime_strategy_metadata(repo_root)
        state["strategy"] = dict(state.get("strategy") or strategy_config)
        state["model"] = dict(state.get("model") or model_state)
        state["mlflow"] = dict(state.get("mlflow") or mlflow_state)
    state.setdefault("portfolio_state", {})
    state.setdefault("artifacts", {})
    state["artifacts"]["prediction_snapshot_path"] = str(prediction_path)

    state.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "signal_date": signal_date,
            "live_outputs": {
                "signal_date": signal_date,
                "top_10_scores": top_fusion_scores,
                "ticker_count": int(len(fusion_scores)),
            },
            "latest_greyscale_report": {
                "signal_date": signal_date,
                "report_path": report.get("_report_path"),
                "weight_source": report.get("fusion", {}).get("weight_source"),
                "regime": report.get("fusion", {}).get("regime", {}),
            },
        },
    )
    _write_signal_state(repo_root, state)

    return _result(
        "model_inference",
        "ok",
        model_name="ic_weighted_fusion_60d",
        horizon=int(report.get("model_bundle", {}).get("horizon_days", 60)),
        latest_prediction_date=signal_date,
        ticker_count=int(len(fusion_scores)),
        top_tickers=[str(row["ticker"]) for row in top_fusion_scores[:10]],
        top_scores=[float(row["score"]) for row in top_fusion_scores[:10]],
        retained_feature_count=int(report.get("model_bundle", {}).get("retained_feature_count", 0)),
        weekly_report_path=report.get("_report_path"),
        weight_source=report.get("fusion", {}).get("weight_source"),
        live_weights=report.get("fusion", {}).get("live_weights", {}),
        execution_mode=execution_mode,
        prediction_snapshot_path=str(prediction_path),
    )


def model_inference(**context: Any) -> dict[str, Any]:
    return _run_task("model_inference", _model_inference_impl, **context)


def _signal_risk_check_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    state = _load_signal_state(repo_root) or {}
    report = _load_latest_greyscale_report(repo_root)
    if report is None:
        raise RuntimeError("No greyscale weekly report is available; model_inference must run first.")

    signal_risk = dict(report.get("risk_checks", {}).get("layer2_signal") or {})
    state["signal_risk"] = signal_risk
    state["latest_greyscale_report"] = {
        "signal_date": report.get("live_outputs", {}).get("signal_date"),
        "report_path": report.get("_report_path"),
        "weight_source": report.get("fusion", {}).get("weight_source"),
        "regime": report.get("fusion", {}).get("regime", {}),
    }
    _write_signal_state(repo_root, state)

    return _result(
        "signal_risk_check",
        "warning" if signal_risk.get("recommend_switch") else "ok",
        latest_prediction_date=report.get("live_outputs", {}).get("signal_date"),
        history_dates=int(signal_risk.get("history_dates", 0) or 0),
        history_rows=int(signal_risk.get("history_rows", 0) or 0),
        live_ic=signal_risk.get("fusion_ic"),
        severity=signal_risk.get("severity"),
        layer1_pass=bool(report.get("risk_checks", {}).get("layer1_data", {}).get("pass", False)),
        layer2_pass=bool(signal_risk.get("pass", False)),
        layer3_pass=bool(report.get("risk_checks", {}).get("layer3_portfolio", {}).get("pass", False)),
        layer4_pass=bool(report.get("risk_checks", {}).get("layer4_operational", {}).get("pass", False)),
        report=signal_risk,
        weekly_report_path=report.get("_report_path"),
    )


def signal_risk_check(**context: Any) -> dict[str, Any]:
    return _run_task("signal_risk_check", _signal_risk_check_impl, **context)


def _greyscale_monitor_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_greyscale_monitor import main as run_greyscale_monitor_main

    as_of = datetime.now(timezone.utc)
    report_dir_candidates = (
        _artifact_path(repo_root, DEFAULT_GREYSCALE_REPORT_DIR),
        _artifact_path(repo_root, LEGACY_GREYSCALE_REPORT_DIR),
    )
    report_dir = next(
        (
            path
            for path in report_dir_candidates
            if path.exists() and any(path.glob("week_*.json"))
        ),
        report_dir_candidates[0],
    )
    summary_path = _runtime_artifact_path(repo_root, DEFAULT_G4_SUMMARY_PATH, LEGACY_G4_SUMMARY_PATH)
    status_code = run_greyscale_monitor_main(
        [
            "--report-dir",
            str(report_dir),
            "--output-path",
            str(summary_path),
            "--as-of",
            as_of.isoformat(),
        ],
    )
    if status_code != 0:
        raise RuntimeError(f"run_greyscale_monitor.py exited with status {status_code}")
    summary_payload = _load_g4_summary(repo_root)
    if summary_payload is None:
        raise RuntimeError("Greyscale monitor completed without writing a G4 summary.")

    summary = dict(summary_payload.get("summary") or {})
    state = _load_signal_state(repo_root) or {}
    state["greyscale_monitor"] = {
        "generated_at_utc": summary_payload.get("generated_at_utc"),
        "summary_path": str(summary_path),
        "summary": summary,
    }
    _write_signal_state(repo_root, state)

    gate_status = str(summary.get("gate_status", "PENDING"))
    result_status = "warning" if gate_status == "FAIL" else "ok"
    return _result(
        "greyscale_monitor",
        result_status,
        gate_status=gate_status,
        reports_seen=int(summary.get("reports_seen", 0) or 0),
        matured_weeks=int(summary.get("matured_weeks", 0) or 0),
        mean_live_ic=summary.get("mean_live_ic"),
        mean_turnover=summary.get("mean_turnover"),
        mean_pairwise_rank_correlation=summary.get("mean_pairwise_rank_correlation"),
        summary_path=str(summary_path),
    )


def greyscale_monitor(**context: Any) -> dict[str, Any]:
    return _run_task("greyscale_monitor", _greyscale_monitor_impl, **context)


with DAG(
    dag_id="weekly_signal_pipeline",
    description="Friday greyscale fusion signal generation and G4 monitoring.",
    schedule="30 20 * * 5",
    start_date=pendulum.datetime(2026, 1, 2, tz="America/New_York"),
    catchup=False,
    max_active_runs=1,
    tags=["quantedge", "signals", "weekly"],
    default_args={"owner": "quantedge"},
) as dag:
    check_data_freshness_task = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness,
        retries=12,
        retry_delay=timedelta(minutes=10),
    )
    compute_features_task = PythonOperator(
        task_id="compute_features",
        python_callable=compute_features,
    )
    model_inference_task = PythonOperator(
        task_id="model_inference",
        python_callable=model_inference,
    )
    signal_risk_check_task = PythonOperator(
        task_id="signal_risk_check",
        python_callable=signal_risk_check,
    )
    greyscale_monitor_task = PythonOperator(
        task_id="greyscale_monitor",
        python_callable=greyscale_monitor,
    )

    check_data_freshness_task >> compute_features_task >> model_inference_task >> signal_risk_check_task >> greyscale_monitor_task
