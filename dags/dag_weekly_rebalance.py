from __future__ import annotations

from datetime import datetime, timezone
import json
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
LIVE_PREDICTIONS_PATH = "data/reports/greyscale/weekly_signal_predictions.parquet"
LIVE_PRICES_PATH = "data/reports/greyscale/weekly_signal_prices.parquet"
LIVE_MANIFEST_PATH = "data/reports/greyscale/weekly_signal_state.json"
LEGACY_LIVE_PREDICTIONS_PATH = "/opt/airflow/live_weekly/predictions.parquet"
LEGACY_LIVE_PRICES_PATH = "/opt/airflow/live_weekly/prices.parquet"
LEGACY_LIVE_MANIFEST_PATH = "/opt/airflow/live_weekly/state.json"


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
        LOGGER.exception("weekly_rebalance_pipeline task %s failed", step)
        raise AirflowException(str(exc)) from exc


def _artifact_path(repo_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return repo_root / relative_path


def _artifact_candidates(repo_root: Path, *relative_paths: str) -> list[Path]:
    return [_artifact_path(repo_root, relative_path) for relative_path in relative_paths]


def _runtime_artifact_path(repo_root: Path, primary_path: str, fallback_path: str) -> Path:
    primary = _artifact_path(repo_root, primary_path)
    if primary.parent.exists() and os.access(primary.parent, os.W_OK):
        return primary
    fallback = _artifact_path(repo_root, fallback_path)
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def _signal_state_rank(path: Path, payload: dict[str, Any]) -> tuple[int, float, float]:
    feature_pipeline = dict(payload.get("feature_pipeline") or {})
    artifacts = dict(payload.get("artifacts") or {})
    rebalance_pipeline = dict(payload.get("rebalance_pipeline") or {})
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
            bool(rebalance_pipeline.get("load_signals")),
            bool(rebalance_pipeline.get("portfolio_optimize")),
            bool(rebalance_pipeline.get("portfolio_risk_check")),
            bool(rebalance_pipeline.get("generate_orders")),
            bool(rebalance_pipeline.get("audit_log")),
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
    for path in _artifact_candidates(repo_root, LIVE_MANIFEST_PATH, LEGACY_LIVE_MANIFEST_PATH):
        if path.exists():
            payload = json.loads(path.read_text())
            candidates.append((_signal_state_rank(path, payload), payload))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _write_signal_state(repo_root: Path, payload: dict[str, Any]) -> None:
    from scripts.run_ic_screening import write_json_atomic
    from scripts.run_single_window_validation import json_safe

    write_json_atomic(
        _runtime_artifact_path(repo_root, LIVE_MANIFEST_PATH, LEGACY_LIVE_MANIFEST_PATH),
        json_safe(payload),
    )


def _stored_rebalance_step(state: dict[str, Any], step: str) -> dict[str, Any]:
    return dict((state.get("rebalance_pipeline") or {}).get(step) or {})


def _persist_rebalance_step(repo_root: Path, state: dict[str, Any], step: str, payload: dict[str, Any]) -> None:
    persisted_state = dict(state or {})
    pipeline_state = dict(persisted_state.get("rebalance_pipeline") or {})
    pipeline_state[step] = dict(payload)
    pipeline_state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    persisted_state["rebalance_pipeline"] = pipeline_state
    _write_signal_state(repo_root, persisted_state)


def _load_price_snapshot(repo_root: Path):
    import pandas as pd

    for path in _artifact_candidates(repo_root, LIVE_PRICES_PATH, LEGACY_LIVE_PRICES_PATH):
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        return frame
    return None


def _latest_signal_snapshot(repo_root: Path) -> dict[str, Any] | None:
    import pandas as pd

    for prediction_path in _artifact_candidates(repo_root, LIVE_PREDICTIONS_PATH, LEGACY_LIVE_PREDICTIONS_PATH):
        if not prediction_path.exists():
            continue
        frame = pd.read_parquet(prediction_path)
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        latest_trade_date = frame["trade_date"].max()
        latest = frame.loc[frame["trade_date"] == latest_trade_date].copy()
        latest["ticker"] = latest["ticker"].astype(str).str.upper()
        latest.sort_values("score", ascending=False, inplace=True)
        return {
            "trade_date": latest_trade_date.date().isoformat(),
            "frame": latest,
            "ticker_count": int(latest["ticker"].nunique()),
            "window_ids": sorted(latest["window_id"].astype(str).unique().tolist()),
            "state": _load_signal_state(repo_root) or {},
        }
    return None


def _resolve_execution_date(execution, signal_date):
    import pandas as pd

    execution_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    signal_ts = pd.Timestamp(signal_date)
    eligible_trade_dates = execution_dates[execution_dates > signal_ts]
    if len(eligible_trade_dates):
        return eligible_trade_dates[0], "next_trade_date"
    if signal_ts in set(execution_dates):
        return signal_ts, "same_day_indicative"
    return None, "missing"


def _load_signals_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    snapshot = _latest_signal_snapshot(repo_root)
    if snapshot is None:
        return _result("load_signals", "skipped", reason="no_fresh_live_signal_snapshot")

    top_scores = snapshot["frame"][["ticker", "score"]].head(10).to_dict(orient="records")
    state = dict(snapshot["state"] or {})
    strategy = dict(state.get("strategy") or {})
    portfolio_state = dict(state.get("portfolio_state") or {})
    payload = _result(
        "load_signals",
        "ok",
        signal_source="fresh_live_snapshot",
        model_name=((state.get("model") or {}).get("model_name")),
        signal_method=strategy.get("method"),
        holding_period=strategy.get("holding_period"),
        latest_signal_date=snapshot["trade_date"],
        ticker_count=snapshot["ticker_count"],
        top_scores=top_scores,
        mlflow_run_id=(state.get("mlflow") or {}).get("run_id"),
        feature_batch_id=((state.get("feature_pipeline") or {}).get("batch_id")),
        current_portfolio_holdings=int(len(portfolio_state.get("current_weights") or {})),
        last_rebalance_signal_date=portfolio_state.get("last_rebalance_signal_date"),
    )
    _persist_rebalance_step(repo_root, state, "load_signals", payload)
    return payload


def load_signals(**context: Any) -> dict[str, Any]:
    return _run_task("load_signals", _load_signals_impl, **context)


def _portfolio_optimize_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from scripts.live_strategy import build_buffered_target_weights, should_rebalance

    snapshot = _latest_signal_snapshot(repo_root)
    if snapshot is None:
        return _result("portfolio_optimize", "skipped", reason="no_fresh_live_signal_snapshot")

    state = dict(snapshot["state"] or {})
    strategy = dict(state.get("strategy") or {})
    if not strategy:
        payload = _result("portfolio_optimize", "skipped", reason="missing_live_strategy_config")
        _persist_rebalance_step(repo_root, state, "portfolio_optimize", payload)
        return payload

    portfolio_state = dict(state.get("portfolio_state") or {})
    current_weights = {
        str(ticker): float(weight)
        for ticker, weight in dict(portfolio_state.get("current_weights") or {}).items()
    }
    rebalance_due = should_rebalance(
        signal_date=snapshot["trade_date"],
        holding_period=str(strategy.get("holding_period") or "4W"),
        last_rebalance_signal_date=portfolio_state.get("last_rebalance_signal_date"),
    )
    if not rebalance_due and current_weights:
        top_weights = pd.Series(current_weights, dtype=float).sort_values(ascending=False).head(10).to_dict()
        payload = _result(
            "portfolio_optimize",
            "ok",
            latest_signal_date=snapshot["trade_date"],
            holding_count=len(current_weights),
            gross_exposure=float(sum(current_weights.values())),
            optimizer_name="hold_previous_portfolio",
            fallback_used=False,
            rebalance_due=False,
            target_weights=current_weights,
            top_weights=top_weights,
            last_rebalance_signal_date=portfolio_state.get("last_rebalance_signal_date"),
            holding_period=strategy.get("holding_period"),
        )
        _persist_rebalance_step(repo_root, state, "portfolio_optimize", payload)
        return payload

    scores = snapshot["frame"].set_index("ticker")["score"].astype(float)
    weights, selection_details = build_buffered_target_weights(
        scores=scores,
        current_weights=current_weights,
        strategy_config=strategy,
    )
    if not weights:
        payload = _result("portfolio_optimize", "skipped", reason="optimizer_returned_empty_weights")
        _persist_rebalance_step(repo_root, state, "portfolio_optimize", payload)
        return payload

    top_weights = pd.Series(weights, dtype=float).sort_values(ascending=False).head(10).to_dict()
    payload = _result(
        "portfolio_optimize",
        "ok",
        latest_signal_date=snapshot["trade_date"],
        holding_count=len(weights),
        gross_exposure=float(sum(weights.values())),
        optimizer_name="equal_weight_portfolio",
        fallback_used=False,
        rebalance_due=True,
        holding_period=strategy.get("holding_period"),
        selection_pct=float(strategy.get("selection_pct", 0.20)),
        sell_buffer_pct=strategy.get("sell_buffer_pct"),
        candidate_count=int(selection_details["candidate_count"]),
        retained_from_buffer=selection_details["retained_from_buffer"],
        retained_count=int(selection_details["retained_count"]),
        previous_holding_count=len(current_weights),
        last_rebalance_signal_date=portfolio_state.get("last_rebalance_signal_date"),
        target_weights=weights,
        top_weights=top_weights,
    )
    _persist_rebalance_step(repo_root, state, "portfolio_optimize", payload)
    return payload


def portfolio_optimize(**context: Any) -> dict[str, Any]:
    return _run_task("portfolio_optimize", _portfolio_optimize_impl, **context)


def _portfolio_risk_check_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from sqlalchemy import text

    from src.backtest.execution import prepare_execution_price_frame
    from src.data.db.session import get_engine
    from src.risk.portfolio_risk import PortfolioRiskEngine

    snapshot = _latest_signal_snapshot(repo_root)
    if snapshot is None:
        return _result("portfolio_risk_check", "skipped", reason="no_fresh_live_signal_snapshot")

    state = dict(snapshot.get("state") or {})
    optimize_payload = (
        context["ti"].xcom_pull(task_ids="portfolio_optimize")
        or _stored_rebalance_step(state, "portfolio_optimize")
        or {}
    )
    if optimize_payload.get("status") != "ok":
        payload = _result("portfolio_risk_check", "skipped", reason="no_optimized_portfolio_available")
        _persist_rebalance_step(repo_root, state, "portfolio_risk_check", payload)
        return payload

    strategy = dict(state.get("strategy") or {})
    portfolio_state = dict(state.get("portfolio_state") or {})
    current_weights = {
        str(ticker): float(weight)
        for ticker, weight in dict(portfolio_state.get("current_weights") or {}).items()
    }
    if not bool(optimize_payload.get("rebalance_due", True)) and current_weights:
        payload = _result(
            "portfolio_risk_check",
            "ok",
            latest_signal_date=snapshot["trade_date"] if snapshot else None,
            execution_date=None,
            execution_mode="holding_period_gate",
            holding_count=len(current_weights),
            turnover=0.0,
            portfolio_beta=None,
            approved_weights=current_weights,
            sector_weights={},
            triggered_rules=[],
            warnings=["holding_period_active"],
            audit_entries=0,
            rebalance_due=False,
        )
        _persist_rebalance_step(repo_root, state, "portfolio_risk_check", payload)
        return payload

    prices = _load_price_snapshot(repo_root)
    if prices is None:
        payload = _result("portfolio_risk_check", "skipped", reason="missing_live_signal_or_price_snapshot")
        _persist_rebalance_step(repo_root, state, "portfolio_risk_check", payload)
        return payload

    execution = prepare_execution_price_frame(prices)
    latest_signal_date = pd.Timestamp(snapshot["trade_date"])
    execution_date, execution_mode = _resolve_execution_date(execution, latest_signal_date)
    if execution_date is None:
        payload = _result("portfolio_risk_check", "skipped", reason="no_execution_date_available")
        _persist_rebalance_step(repo_root, state, "portfolio_risk_check", payload)
        return payload

    score_frame = snapshot["frame"].copy()
    scores = score_frame.set_index("ticker")["score"].astype(float).sort_values(ascending=False)
    raw_weights = {
        str(ticker): float(weight)
        for ticker, weight in pd.Series(optimize_payload.get("target_weights") or {}, dtype=float).items()
    }
    if not raw_weights:
        payload = _result("portfolio_risk_check", "skipped", reason="empty_raw_weights")
        _persist_rebalance_step(repo_root, state, "portfolio_risk_check", payload)
        return payload

    with get_engine().connect() as conn:
        stocks = pd.read_sql(text("select ticker, sector from stocks"), conn)
    sector_map = (
        stocks.assign(
            ticker=stocks["ticker"].astype(str).str.upper(),
            sector=stocks["sector"].fillna("Unknown").astype(str),
        )
        .set_index("ticker")["sector"]
        .to_dict()
    )

    score_tickers = scores.index.astype(str).tolist()
    benchmark_weight = 1.0 / len(score_tickers) if score_tickers else 0.0
    benchmark_weights = {ticker: benchmark_weight for ticker in score_tickers}
    trailing_history = (
        execution["daily_return"]
        .unstack("ticker")
        .sort_index()
        .loc[:execution_date]
        .iloc[:-1]
    )
    spy_returns = trailing_history["SPY"] if "SPY" in trailing_history.columns else None

    engine = PortfolioRiskEngine()
    constrained = engine.apply_all_constraints(
        weights=raw_weights,
        benchmark_weights=benchmark_weights,
        sector_map=sector_map,
        return_history=trailing_history.reindex(columns=list(raw_weights)),
        spy_returns=spy_returns,
        current_weights=current_weights,
        candidate_ranking=score_tickers,
        max_single_stock_weight=float(strategy.get("max_weight", 0.05)),
        max_sector_deviation=0.15,
        min_holdings=int(strategy.get("min_holdings", 20)),
    )
    triggered_rules = [entry.rule_name for entry in constrained.audit_trail if entry.triggered]
    payload = _result(
        "portfolio_risk_check",
        "warning" if triggered_rules or constrained.warnings else "ok",
        latest_signal_date=snapshot["trade_date"],
        execution_date=pd.Timestamp(execution_date).date().isoformat(),
        execution_mode=execution_mode,
        holding_count=constrained.holding_count,
        turnover=constrained.turnover,
        portfolio_beta=constrained.portfolio_beta,
        approved_weights={str(ticker): float(weight) for ticker, weight in constrained.weights.items()},
        sector_weights=constrained.sector_weights,
        triggered_rules=triggered_rules,
        warnings=constrained.warnings,
        audit_entries=len(constrained.audit_trail),
        rebalance_due=True,
    )
    _persist_rebalance_step(repo_root, state, "portfolio_risk_check", payload)
    return payload


def portfolio_risk_check(**context: Any) -> dict[str, Any]:
    return _run_task("portfolio_risk_check", _portfolio_risk_check_impl, **context)


def _generate_orders_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from scripts.live_strategy import normalize_weight_dict
    from src.backtest.cost_model import AlmgrenChrissCostModel
    from src.backtest.execution import prepare_execution_price_frame

    snapshot = _latest_signal_snapshot(repo_root)
    if snapshot is None:
        return _result("generate_orders", "skipped", reason="no_fresh_live_signal_snapshot")

    state = dict(snapshot.get("state") or {})
    risk_payload = (
        context["ti"].xcom_pull(task_ids="portfolio_risk_check")
        or _stored_rebalance_step(state, "portfolio_risk_check")
        or {}
    )
    if risk_payload.get("status") == "skipped":
        payload = _result("generate_orders", "skipped", reason="no_risk_approved_portfolio")
        _persist_rebalance_step(repo_root, state, "generate_orders", payload)
        return payload

    strategy = dict(state.get("strategy") or {})
    portfolio_state = dict(state.get("portfolio_state") or {})
    current_weights = {
        str(ticker): float(weight)
        for ticker, weight in dict(portfolio_state.get("current_weights") or {}).items()
    }
    prices = _load_price_snapshot(repo_root)
    if prices is None:
        payload = _result("generate_orders", "skipped", reason="missing_live_signal_or_price_snapshot")
        _persist_rebalance_step(repo_root, state, "generate_orders", payload)
        return payload

    target_weights = pd.Series(risk_payload.get("approved_weights") or {}, dtype=float)
    if target_weights.empty:
        payload = _result("generate_orders", "skipped", reason="empty_approved_weights")
        _persist_rebalance_step(repo_root, state, "generate_orders", payload)
        return payload
    if not bool(risk_payload.get("rebalance_due", True)):
        state["portfolio_state"] = {
            **portfolio_state,
            "current_weights": normalize_weight_dict(current_weights),
            "holding_count": int(len(current_weights)),
            "holding_period": strategy.get("holding_period"),
            "selection_pct": strategy.get("selection_pct"),
            "sell_buffer_pct": strategy.get("sell_buffer_pct"),
            "last_reviewed_signal_date": snapshot["trade_date"],
            "rebalance_due_last_run": False,
        }
        payload = _result(
            "generate_orders",
            "skipped",
            reason="holding_period_active",
            latest_signal_date=snapshot["trade_date"],
            execution_date=None,
            execution_mode="holding_period_gate",
            order_count=0,
            estimated_total_cost=0.0,
            orders=[],
            rebalance_due=False,
        )
        _persist_rebalance_step(repo_root, state, "generate_orders", payload)
        return payload

    execution = prepare_execution_price_frame(prices)
    latest_signal_date = pd.Timestamp(snapshot["trade_date"])
    execution_date, execution_mode = _resolve_execution_date(execution, latest_signal_date)
    if execution_date is None:
        payload = _result("generate_orders", "skipped", reason="no_execution_date_available")
        _persist_rebalance_step(repo_root, state, "generate_orders", payload)
        return payload
    entry_slice = execution.xs(pd.Timestamp(execution_date), level="trade_date")

    portfolio_value = 1_000_000.0
    cost_model = AlmgrenChrissCostModel()
    orders: list[dict[str, Any]] = []
    for ticker in sorted(set(target_weights.index.astype(str)) | set(current_weights)):
        target_weight = float(target_weights.get(ticker, 0.0))
        current_weight = float(current_weights.get(ticker, 0.0))
        delta_weight = target_weight - current_weight
        if abs(delta_weight) <= 1e-12:
            continue
        if ticker not in entry_slice.index:
            continue
        bar = entry_slice.loc[ticker]
        notional = float(delta_weight) * portfolio_value
        execution_price = max(float(bar["execution_price"]), 1e-12)
        shares = notional / execution_price
        estimate = cost_model.estimate_trade(
            order_shares=abs(shares),
            execution_price=execution_price,
            sigma_20d=float(bar["sigma_20d"]),
            adv_20d_shares=float(bar["adv_20d_shares"]),
            open_gap=float(bar["open_gap"]),
            execution_volume_ratio=float(bar["volume_ratio"]),
        )
        orders.append(
            {
                "ticker": str(ticker),
                "current_weight": float(current_weight),
                "target_weight": float(target_weight),
                "delta_weight": float(delta_weight),
                "side": "buy" if delta_weight > 0.0 else "sell",
                "estimated_shares": float(shares),
                "estimated_cost": float(estimate.total_cost),
            },
        )

    state["portfolio_state"] = {
        **portfolio_state,
        "current_weights": normalize_weight_dict({str(ticker): float(weight) for ticker, weight in target_weights.items()}),
        "previous_weights": normalize_weight_dict(current_weights),
        "holding_count": int(len(target_weights)),
        "holding_period": strategy.get("holding_period"),
        "selection_pct": strategy.get("selection_pct"),
        "sell_buffer_pct": strategy.get("sell_buffer_pct"),
        "last_rebalance_signal_date": snapshot["trade_date"],
        "last_reviewed_signal_date": snapshot["trade_date"],
        "last_execution_date": pd.Timestamp(execution_date).date().isoformat(),
        "execution_mode": execution_mode,
        "rebalance_due_last_run": True,
    }
    payload = _result(
        "generate_orders",
        "ok" if orders else "skipped",
        latest_signal_date=snapshot["trade_date"],
        execution_date=pd.Timestamp(execution_date).date().isoformat(),
        execution_mode=execution_mode,
        order_count=len(orders),
        estimated_total_cost=float(sum(order["estimated_cost"] for order in orders)),
        orders=orders[:20],
        rebalance_due=True,
    )
    _persist_rebalance_step(repo_root, state, "generate_orders", payload)
    return payload


def generate_orders(**context: Any) -> dict[str, Any]:
    return _run_task("generate_orders", _generate_orders_impl, **context)


def _audit_log_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_live_pipeline import persist_audit_log
    from src.risk.operational_risk import OperationalRiskMonitor

    snapshot = _latest_signal_snapshot(repo_root)
    if snapshot is None:
        return _result("audit_log", "skipped", reason="no_live_signal_snapshot")

    state = snapshot["state"]
    feature_batch_id = ((state.get("feature_pipeline") or {}).get("batch_id"))
    model_version_id = ((state.get("model") or {}).get("run_id"))

    monitor = OperationalRiskMonitor()
    events = []
    critical_alerts: list[str] = []
    for task_id in ("load_signals", "portfolio_optimize", "portfolio_risk_check", "generate_orders"):
        payload = context["ti"].xcom_pull(task_ids=task_id) or _stored_rebalance_step(state, task_id) or {}
        if payload.get("status") in {"warning", "error"}:
            critical_alerts.append(f"{task_id}_{payload.get('status')}")
        events.append(
            monitor.audit_decision(
                action=f"{task_id}_completed",
                actor="weekly_rebalance_pipeline",
                details={
                    "status": payload.get("status"),
                    "step": payload.get("step"),
                    "latest_signal_date": payload.get("latest_signal_date"),
                },
            ),
        )

    report = monitor.run_all_checks(
        runtime_seconds=0.0,
        critical_alerts=[],
        audit_events=events,
        max_runtime_seconds=3600.0,
    )
    audit_insert = persist_audit_log(
        audit_records=report.audit_log,
        model_version_id=model_version_id,
        feature_batch_id=feature_batch_id,
    )
    payload = _result(
        "audit_log",
        "ok",
        latest_signal_date=snapshot["trade_date"],
        audit_entries=len(report.audit_log),
        inserted_count=int(audit_insert["inserted_count"]),
        inserted_ids=audit_insert["inserted_ids"],
        overall_severity=report.overall_severity.value,
        fail_safe_mode=report.fail_safe_mode,
        critical_alerts=critical_alerts,
        report=report.to_dict(),
    )
    _persist_rebalance_step(repo_root, state, "audit_log", payload)
    return payload


def audit_log(**context: Any) -> dict[str, Any]:
    return _run_task("audit_log", _audit_log_impl, **context)


with DAG(
    dag_id="weekly_rebalance_pipeline",
    description="Friday rebalance orchestration with portfolio risk controls.",
    schedule="0 21 * * 5",
    start_date=pendulum.datetime(2026, 1, 2, tz="America/New_York"),
    catchup=False,
    tags=["quantedge", "rebalance", "weekly"],
    default_args={"owner": "quantedge"},
) as dag:
    load_signals_task = PythonOperator(task_id="load_signals", python_callable=load_signals)
    portfolio_optimize_task = PythonOperator(
        task_id="portfolio_optimize",
        python_callable=portfolio_optimize,
    )
    portfolio_risk_check_task = PythonOperator(
        task_id="portfolio_risk_check",
        python_callable=portfolio_risk_check,
    )
    generate_orders_task = PythonOperator(
        task_id="generate_orders",
        python_callable=generate_orders,
    )
    audit_log_task = PythonOperator(task_id="audit_log", python_callable=audit_log)

    load_signals_task >> portfolio_optimize_task >> portfolio_risk_check_task >> generate_orders_task >> audit_log_task
