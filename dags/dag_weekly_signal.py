from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
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
DEFAULT_HISTORY_LOOKBACK_DAYS = 400
DEFAULT_MIN_SIGNAL_CROSS_SECTION = 50
DEFAULT_SIGNAL_LOOKBACK_POINTS = 12
DEFAULT_FUSION_TEMPERATURE = 5.0
LIVE_FEATURE_MATRIX_PATH = "data/reports/greyscale/weekly_signal_feature_matrix.parquet"
LIVE_PRICES_PATH = "data/reports/greyscale/weekly_signal_prices.parquet"
LIVE_PREDICTIONS_PATH = "data/reports/greyscale/weekly_signal_predictions.parquet"
LIVE_MANIFEST_PATH = "data/reports/greyscale/weekly_signal_state.json"


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
        LOGGER.exception("weekly_signal_pipeline task %s failed", step)
        raise AirflowException(str(exc)) from exc


def _artifact_path(repo_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return repo_root / relative_path


def _load_signal_state(repo_root: Path) -> dict[str, Any] | None:
    path = _artifact_path(repo_root, LIVE_MANIFEST_PATH)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_signal_state(repo_root: Path, payload: dict[str, Any]) -> None:
    from scripts.run_ic_screening import write_json_atomic
    from scripts.run_single_window_validation import json_safe

    path = _artifact_path(repo_root, LIVE_MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, json_safe(payload))


def _load_bundle_manifest(repo_root: Path) -> dict[str, Any]:
    path = _artifact_path(repo_root, DEFAULT_BUNDLE_PATH)
    if not path.exists():
        raise RuntimeError(f"Fusion bundle manifest is missing: {path}")
    payload = json.loads(path.read_text())
    retained_features = payload.get("retained_features") or []
    if not retained_features:
        raise RuntimeError("Fusion bundle manifest does not contain retained features.")
    for model_name, model_payload in dict(payload.get("models") or {}).items():
        artifact_path = Path(str(model_payload.get("artifact_path", "")))
        resolved = artifact_path if artifact_path.is_absolute() else repo_root / artifact_path
        if not resolved.exists():
            raise RuntimeError(f"Fusion model artifact for {model_name} is missing: {resolved}")
    return payload


def _load_latest_greyscale_report(repo_root: Path) -> dict[str, Any] | None:
    from scripts.run_greyscale_live import load_greyscale_reports

    report_dir = _artifact_path(repo_root, DEFAULT_GREYSCALE_REPORT_DIR)
    reports = load_greyscale_reports(report_dir)
    if not reports:
        return None
    return reports[-1]


def _load_g4_summary(repo_root: Path) -> dict[str, Any] | None:
    path = _artifact_path(repo_root, DEFAULT_G4_SUMMARY_PATH)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_live_universe(*, trade_date: Any) -> tuple[list[str], str]:
    from sqlalchemy import text

    from src.data.db.session import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        membership_rows = conn.execute(
            text(
                """
                select distinct ticker
                from universe_membership
                where index_name = 'SP500'
                  and effective_date <= :trade_date
                  and (end_date is null or end_date > :trade_date)
                  and upper(ticker) <> :exclude_ticker
                order by ticker
                """,
            ),
            {"trade_date": trade_date, "exclude_ticker": BENCHMARK_TICKER},
        ).scalars().all()
        if membership_rows:
            return [str(ticker).upper() for ticker in membership_rows], "universe_membership"

        stock_rows = conn.execute(
            text(
                """
                select ticker
                from stocks
                where ticker is not null
                  and upper(ticker) <> :exclude_ticker
                order by ticker
                """,
            ),
            {"exclude_ticker": BENCHMARK_TICKER},
        ).scalars().all()
    return [str(ticker).upper() for ticker in stock_rows], "stocks_fallback"


def _load_feature_matrix(repo_root: Path):
    import pandas as pd

    path = _artifact_path(repo_root, LIVE_FEATURE_MATRIX_PATH)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame.set_index(["trade_date", "ticker"]).sort_index()


def _load_price_snapshot(repo_root: Path):
    import pandas as pd

    path = _artifact_path(repo_root, LIVE_PRICES_PATH)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame


def _latest_prediction_snapshot(repo_root: Path) -> dict[str, Any] | None:
    import pandas as pd

    prediction_path = _artifact_path(repo_root, LIVE_PREDICTIONS_PATH)
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


def _check_data_freshness_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_live_pipeline import load_db_state

    as_of = datetime.now(timezone.utc)
    db_state = load_db_state(as_of=as_of)
    latest_pit_trade_date = db_state["latest_pit_trade_date"]
    if latest_pit_trade_date is None:
        return _result("check_data_freshness", "skipped", reason="no_pit_visible_price_data_available")

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
        previous_signal_snapshot_date=latest_snapshot_date,
        previous_snapshot_current=not stale,
        universe_membership_live_count=int(db_state["universe_membership_live_count"]),
    )


def check_data_freshness(**context: Any) -> dict[str, Any]:
    return _run_task("check_data_freshness", _check_data_freshness_impl, **context)


def _compute_features_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_greyscale_live import load_live_universe
    from scripts.run_live_pipeline import load_db_state

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

    retained_features = list(bundle["retained_features"])
    model_artifacts = {
        model_name: str(bundle["models"][model_name]["artifact_path"])
        for model_name in sorted(dict(bundle.get("models") or {}))
    }

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
        "feature_pipeline": {
            "start_date": feature_start.isoformat(),
            "end_date": live_trade_date.isoformat(),
            "mode": "delegated_to_run_greyscale_live",
            "expected_feature_count": int(len(retained_features)),
            "reference_feature_matrix_path": "data/features/walkforward_feature_matrix_60d.parquet",
        },
        "artifacts": {
            "report_dir": DEFAULT_GREYSCALE_REPORT_DIR,
            "monitor_summary_path": DEFAULT_G4_SUMMARY_PATH,
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
    )


def compute_features(**context: Any) -> dict[str, Any]:
    return _run_task("compute_features", _compute_features_impl, **context)


def _model_inference_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_greyscale_live import main as run_greyscale_live_main
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
    status_code = run_greyscale_monitor_main(
        [
            "--report-dir",
            DEFAULT_GREYSCALE_REPORT_DIR,
            "--output-path",
            DEFAULT_G4_SUMMARY_PATH,
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
        "summary_path": str(_artifact_path(repo_root, DEFAULT_G4_SUMMARY_PATH)),
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
        summary_path=str(_artifact_path(repo_root, DEFAULT_G4_SUMMARY_PATH)),
    )


def greyscale_monitor(**context: Any) -> dict[str, Any]:
    return _run_task("greyscale_monitor", _greyscale_monitor_impl, **context)


with DAG(
    dag_id="weekly_signal_pipeline",
    description="Friday greyscale fusion signal generation and G4 monitoring.",
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
    greyscale_monitor_task = PythonOperator(
        task_id="greyscale_monitor",
        python_callable=greyscale_monitor,
    )

    check_data_freshness_task >> compute_features_task >> model_inference_task >> signal_risk_check_task >> greyscale_monitor_task
