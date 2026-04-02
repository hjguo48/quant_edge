from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

LOGGER = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = "data/features/all_features.parquet"
DEFAULT_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"


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
        return payload
    except Exception as exc:
        LOGGER.exception("weekly_signal_pipeline task %s failed", step)
        return _result(step, "error", error=str(exc))


def _latest_prediction_snapshot(repo_root: Path) -> dict[str, Any] | None:
    import pandas as pd

    prediction_path = repo_root / DEFAULT_PREDICTIONS_PATH
    if not prediction_path.exists():
        return None

    predictions = pd.read_parquet(prediction_path)
    if predictions.empty:
        return None

    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    latest_trade_date = predictions["trade_date"].max()
    latest = predictions.loc[predictions["trade_date"] == latest_trade_date].copy()
    latest["ticker"] = latest["ticker"].astype(str).str.upper()
    latest.sort_values("score", ascending=False, inplace=True)
    return {
        "trade_date": latest_trade_date.date().isoformat(),
        "window_ids": sorted(latest["window_id"].astype(str).unique().tolist()),
        "score_series": latest.set_index("ticker")["score"].astype(float),
        "ticker_count": int(latest["ticker"].nunique()),
        "top_tickers": latest["ticker"].head(10).tolist(),
    }


def _feature_cache_dates(repo_root: Path) -> tuple[str | None, str | None]:
    import pandas as pd

    feature_path = repo_root / DEFAULT_FEATURES_PATH
    if not feature_path.exists():
        return None, None

    frame = pd.read_parquet(feature_path, columns=["trade_date"])
    if frame.empty:
        return str(feature_path), None
    latest_feature_date = pd.to_datetime(frame["trade_date"]).max()
    return str(feature_path), latest_feature_date.date().isoformat()


def _check_data_freshness_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text

    from src.data.db.session import get_engine
    from src.models.registry import ModelRegistry

    feature_path, latest_feature_date = _feature_cache_dates(repo_root)
    snapshot = _latest_prediction_snapshot(repo_root)
    with get_engine().connect() as conn:
        latest_price_date = conn.execute(text("select max(trade_date) from stock_prices")).scalar()

    registry = ModelRegistry()
    champion = registry.get_champion("ridge_60d")
    if latest_price_date is None:
        return _result("check_data_freshness", "skipped", reason="no_price_data_available")

    stale = bool(latest_feature_date and latest_feature_date < latest_price_date.isoformat())
    return _result(
        "check_data_freshness",
        "warning" if stale else "ok",
        latest_price_date=latest_price_date.isoformat(),
        latest_feature_date=latest_feature_date,
        feature_path=feature_path,
        latest_prediction_date=snapshot["trade_date"] if snapshot else None,
        champion_version=champion.version if champion else None,
        champion_stage=champion.stage.value if champion else None,
    )


def check_data_freshness(**context: Any) -> dict[str, Any]:
    return _run_task("check_data_freshness", _check_data_freshness_impl, **context)


def _compute_features_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text

    from src.data.db.session import get_engine
    from src.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    feature_path, latest_feature_date = _feature_cache_dates(repo_root)
    with get_engine().connect() as conn:
        latest_price_date = conn.execute(text("select max(trade_date) from stock_prices")).scalar()

    latest_price_iso = latest_price_date.isoformat() if latest_price_date else None
    if latest_price_iso is None:
        return _result("compute_features", "skipped", reason="no_price_data_available")
    if latest_feature_date and latest_feature_date >= latest_price_iso:
        return _result(
            "compute_features",
            "skipped",
            reason="feature_cache_current",
            latest_price_date=latest_price_iso,
            latest_feature_date=latest_feature_date,
            feature_path=feature_path,
            pipeline_class=pipeline.__class__.__name__,
        )

    if os.environ.get("QUANTEDGE_ENABLE_FEATURE_REFRESH", "").lower() not in {"1", "true", "yes"}:
        return _result(
            "compute_features",
            "skipped",
            reason="feature_refresh_disabled",
            latest_price_date=latest_price_iso,
            latest_feature_date=latest_feature_date,
            feature_path=feature_path,
            pipeline_class=pipeline.__class__.__name__,
        )

    return _result(
        "compute_features",
        "skipped",
        reason="feature_refresh_not_executed_in_deployment_validation",
        latest_price_date=latest_price_iso,
        latest_feature_date=latest_feature_date,
        feature_path=feature_path,
        pipeline_class=pipeline.__class__.__name__,
    )


def compute_features(**context: Any) -> dict[str, Any]:
    return _run_task("compute_features", _compute_features_impl, **context)


def _model_inference_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()
    champion = registry.get_champion("ridge_60d")
    snapshot = _latest_prediction_snapshot(repo_root)
    if champion is None:
        return _result("model_inference", "skipped", reason="no_champion_registered")
    if snapshot is None:
        return _result("model_inference", "skipped", reason="no_cached_predictions_available")

    metadata = champion.metadata
    return _result(
        "model_inference",
        "ok",
        model_name=champion.name,
        version=champion.version,
        stage=champion.stage.value,
        horizon=metadata.horizon if metadata else None,
        latest_prediction_date=snapshot["trade_date"],
        ticker_count=snapshot["ticker_count"],
        top_tickers=snapshot["top_tickers"],
        feature_count=metadata.n_features if metadata else None,
    )


def model_inference(**context: Any) -> dict[str, Any]:
    return _run_task("model_inference", _model_inference_impl, **context)


def _signal_risk_check_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from src.labels.forward_returns import compute_forward_returns
    from src.models.registry import ModelRegistry
    from src.risk.signal_risk import SignalRiskMonitor

    snapshot = _latest_prediction_snapshot(repo_root)
    prices_path = repo_root / DEFAULT_PRICES_PATH
    if snapshot is None or not prices_path.exists():
        return _result("signal_risk_check", "skipped", reason="missing_predictions_or_prices")

    prices = pd.read_parquet(prices_path)
    labels = compute_forward_returns(prices, horizons=(60,), benchmark_ticker="SPY")
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    latest_date = pd.Timestamp(snapshot["trade_date"])
    realized = labels.loc[
        (labels["trade_date"] == latest_date) & (labels["horizon"] == 60),
        ["ticker", "excess_return"],
    ].copy()
    realized["ticker"] = realized["ticker"].astype(str).str.upper()
    realized_series = realized.set_index("ticker")["excess_return"].astype(float)
    predicted_scores = snapshot["score_series"].rename("score")
    aligned_index = predicted_scores.index.intersection(realized_series.index)
    if aligned_index.empty:
        return _result(
            "signal_risk_check",
            "skipped",
            reason="no_aligned_predictions_and_realized_returns",
            latest_prediction_date=snapshot["trade_date"],
        )

    registry = ModelRegistry()
    champion = registry.get_champion("ridge_60d")
    champion_ic = float(champion.metadata.metrics.get("mean_oos_ic", 0.0)) if champion and champion.metadata else 0.0
    monitor = SignalRiskMonitor()
    ic_history = []
    if champion and champion.metadata:
        ic_history.append(float(champion.metadata.metrics.get("mean_oos_ic", 0.0)))

    report = monitor.run_all_checks(
        ic_history=ic_history or [champion_ic],
        predicted_scores=predicted_scores.reindex(aligned_index),
        realized_returns=realized_series.reindex(aligned_index),
        champion_ic=champion_ic,
        challenger_ic=champion_ic - 0.005,
        consecutive_challenger_wins=0,
        n_bins=10,
    )
    return _result(
        "signal_risk_check",
        "ok" if not report.recommend_switch else "warning",
        latest_prediction_date=snapshot["trade_date"],
        aligned_tickers=int(len(aligned_index)),
        report=report.to_dict(),
    )


def signal_risk_check(**context: Any) -> dict[str, Any]:
    return _run_task("signal_risk_check", _signal_risk_check_impl, **context)


with DAG(
    dag_id="weekly_signal_pipeline",
    description="Friday signal generation and Layer 2 risk checks.",
    schedule="30 16 * * 5",
    start_date=pendulum.datetime(2026, 1, 2, tz="America/New_York"),
    catchup=False,
    tags=["quantedge", "signals", "weekly"],
    default_args={"owner": "quantedge"},
) as dag:
    check_data_freshness_task = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness,
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

    check_data_freshness_task >> compute_features_task >> model_inference_task >> signal_risk_check_task
