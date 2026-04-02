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
        LOGGER.exception("weekly_rebalance_pipeline task %s failed", step)
        return _result(step, "error", error=str(exc))


def _latest_signal_snapshot(repo_root: Path) -> dict[str, Any] | None:
    import pandas as pd

    prediction_path = repo_root / DEFAULT_PREDICTIONS_PATH
    if not prediction_path.exists():
        return None
    frame = pd.read_parquet(prediction_path)
    if frame.empty:
        return None
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    latest_trade_date = frame["trade_date"].max()
    latest = frame.loc[frame["trade_date"] == latest_trade_date].copy()
    latest["ticker"] = latest["ticker"].astype(str).str.upper()
    latest.sort_values("score", ascending=False, inplace=True)
    return {
        "trade_date": latest_trade_date.date().isoformat(),
        "frame": latest,
        "ticker_count": int(latest["ticker"].nunique()),
    }


def _load_signals_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()
    champion = registry.get_champion("ridge_60d")
    snapshot = _latest_signal_snapshot(repo_root)
    if champion is None:
        return _result("load_signals", "skipped", reason="no_champion_registered")
    if snapshot is None:
        return _result("load_signals", "skipped", reason="no_cached_signal_snapshot")

    top_scores = (
        snapshot["frame"][["ticker", "score"]]
        .head(10)
        .to_dict(orient="records")
    )
    return _result(
        "load_signals",
        "ok",
        model_name=champion.name,
        version=champion.version,
        latest_signal_date=snapshot["trade_date"],
        ticker_count=snapshot["ticker_count"],
        top_scores=top_scores,
    )


def load_signals(**context: Any) -> dict[str, Any]:
    return _run_task("load_signals", _load_signals_impl, **context)


def _portfolio_optimize_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from src.portfolio.constraints import PortfolioConstraints
    from src.portfolio.equal_weight import equal_weight_portfolio

    snapshot = _latest_signal_snapshot(repo_root)
    if snapshot is None:
        return _result("portfolio_optimize", "skipped", reason="no_cached_signal_snapshot")

    scores = (
        snapshot["frame"]
        .set_index("ticker")["score"]
        .astype(float)
    )
    constraints = PortfolioConstraints(max_weight=0.05, min_holdings=20)
    weights = equal_weight_portfolio(
        scores,
        selection_pct=0.20,
        constraints=constraints,
    )
    if not weights:
        return _result("portfolio_optimize", "skipped", reason="optimizer_returned_empty_weights")

    top_weights = (
        pd.Series(weights, dtype=float)
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )
    return _result(
        "portfolio_optimize",
        "ok",
        latest_signal_date=snapshot["trade_date"],
        holding_count=len(weights),
        gross_exposure=float(sum(weights.values())),
        target_weights=weights,
        top_weights=top_weights,
    )


def portfolio_optimize(**context: Any) -> dict[str, Any]:
    return _run_task("portfolio_optimize", _portfolio_optimize_impl, **context)


def _portfolio_risk_check_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from sqlalchemy import text

    from src.backtest.execution import prepare_execution_price_frame
    from src.data.db.session import get_engine
    from src.risk.portfolio_risk import PortfolioRiskEngine, compute_sector_weights

    optimize_payload = context["ti"].xcom_pull(task_ids="portfolio_optimize") or {}
    if optimize_payload.get("status") != "ok":
        return _result("portfolio_risk_check", "skipped", reason="no_optimized_portfolio_available")

    snapshot = _latest_signal_snapshot(repo_root)
    prices_path = repo_root / DEFAULT_PRICES_PATH
    if snapshot is None or not prices_path.exists():
        return _result("portfolio_risk_check", "skipped", reason="missing_signal_or_price_cache")

    prices = pd.read_parquet(prices_path)
    execution = prepare_execution_price_frame(prices)
    latest_signal_date = pd.Timestamp(snapshot["trade_date"])
    execution_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    eligible_trade_dates = execution_dates[execution_dates > latest_signal_date]
    if eligible_trade_dates.empty:
        return _result("portfolio_risk_check", "skipped", reason="no_execution_date_after_signal")
    execution_date = pd.Timestamp(eligible_trade_dates[0])

    score_frame = snapshot["frame"].copy()
    scores = score_frame.set_index("ticker")["score"].astype(float).sort_values(ascending=False)
    raw_weights = {
        str(ticker): float(weight)
        for ticker, weight in (
            pd.Series(optimize_payload["target_weights"], dtype=float)
            if optimize_payload.get("target_weights")
            else pd.Series(dtype=float)
        ).items()
    }
    if not raw_weights:
        from src.portfolio.constraints import PortfolioConstraints
        from src.portfolio.equal_weight import equal_weight_portfolio

        raw_weights = equal_weight_portfolio(
            scores,
            selection_pct=0.20,
            constraints=PortfolioConstraints(max_weight=0.05, min_holdings=20),
        )
    if not raw_weights:
        return _result("portfolio_risk_check", "skipped", reason="empty_raw_weights")

    with get_engine().connect() as conn:
        stocks = pd.read_sql(text("select ticker, sector from stocks"), conn)
    sector_map = (
        stocks.assign(ticker=stocks["ticker"].astype(str).str.upper(), sector=stocks["sector"].fillna("Unknown").astype(str))
        .set_index("ticker")["sector"]
        .to_dict()
    )

    score_tickers = scores.index.astype(str).tolist()
    benchmark_weight = 1.0 / len(score_tickers) if score_tickers else 0.0
    benchmark_weights = {ticker: benchmark_weight for ticker in score_tickers}
    entry_slice = execution.xs(execution_date, level="trade_date")
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
        current_weights={},
        candidate_ranking=score_tickers,
        max_single_stock_weight=0.05,
        max_sector_deviation=0.15,
        min_holdings=20,
    )
    triggered_rules = [entry.rule_name for entry in constrained.audit_trail if entry.triggered]
    return _result(
        "portfolio_risk_check",
        "warning" if triggered_rules or constrained.warnings else "ok",
        latest_signal_date=snapshot["trade_date"],
        holding_count=constrained.holding_count,
        turnover=constrained.turnover,
        portfolio_beta=constrained.portfolio_beta,
        sector_weights=constrained.sector_weights,
        triggered_rules=triggered_rules,
        warnings=constrained.warnings,
        audit_entries=len(constrained.audit_trail),
    )


def portfolio_risk_check(**context: Any) -> dict[str, Any]:
    return _run_task("portfolio_risk_check", _portfolio_risk_check_impl, **context)


def _generate_orders_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from src.backtest.cost_model import AlmgrenChrissCostModel
    from src.backtest.execution import prepare_execution_price_frame

    risk_payload = context["ti"].xcom_pull(task_ids="portfolio_risk_check") or {}
    if risk_payload.get("status") == "skipped":
        return _result("generate_orders", "skipped", reason="no_risk_approved_portfolio")

    optimize_payload = context["ti"].xcom_pull(task_ids="portfolio_optimize") or {}
    snapshot = _latest_signal_snapshot(repo_root)
    prices_path = repo_root / DEFAULT_PRICES_PATH
    if optimize_payload.get("status") != "ok" or snapshot is None or not prices_path.exists():
        return _result("generate_orders", "skipped", reason="missing_optimizer_output_or_prices")

    target_weights = pd.Series(optimize_payload["target_weights"], dtype=float)
    if target_weights.empty:
        return _result("generate_orders", "skipped", reason="empty_target_weights")

    prices = pd.read_parquet(prices_path)
    execution = prepare_execution_price_frame(prices)
    latest_signal_date = pd.Timestamp(snapshot["trade_date"])
    execution_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    eligible_trade_dates = execution_dates[execution_dates > latest_signal_date]
    if eligible_trade_dates.empty:
        return _result("generate_orders", "skipped", reason="no_execution_date_after_signal")
    execution_date = pd.Timestamp(eligible_trade_dates[0])
    entry_slice = execution.xs(execution_date, level="trade_date")

    portfolio_value = 1_000_000.0
    cost_model = AlmgrenChrissCostModel()
    orders: list[dict[str, Any]] = []
    for ticker, weight in target_weights.items():
        if ticker not in entry_slice.index:
            continue
        bar = entry_slice.loc[ticker]
        notional = float(weight) * portfolio_value
        execution_price = max(float(bar["execution_price"]), 1e-12)
        shares = notional / execution_price
        estimate = cost_model.estimate_trade(
            order_shares=shares,
            execution_price=execution_price,
            sigma_20d=float(bar["sigma_20d"]),
            adv_20d_shares=float(bar["adv_20d_shares"]),
            open_gap=float(bar["open_gap"]),
            execution_volume_ratio=float(bar["volume_ratio"]),
        )
        orders.append(
            {
                "ticker": str(ticker),
                "target_weight": float(weight),
                "estimated_shares": float(shares),
                "estimated_cost": float(estimate.total_cost),
            },
        )

    return _result(
        "generate_orders",
        "ok" if orders else "skipped",
        latest_signal_date=snapshot["trade_date"],
        execution_date=execution_date.date().isoformat(),
        order_count=len(orders),
        estimated_total_cost=float(sum(order["estimated_cost"] for order in orders)),
        orders=orders[:20],
    )


def generate_orders(**context: Any) -> dict[str, Any]:
    return _run_task("generate_orders", _generate_orders_impl, **context)


def _audit_log_impl(*, repo_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    from src.risk.operational_risk import OperationalRiskMonitor

    monitor = OperationalRiskMonitor()
    events = []
    for task_id in ("load_signals", "portfolio_optimize", "portfolio_risk_check", "generate_orders"):
        payload = context["ti"].xcom_pull(task_ids=task_id) or {}
        events.append(
            monitor.audit_decision(
                action=f"{task_id}_completed",
                actor="airflow",
                details={
                    "status": payload.get("status"),
                    "step": payload.get("step"),
                },
            ),
        )
    report = monitor.run_all_checks(
        runtime_seconds=0.0,
        critical_alerts=[],
        audit_events=events,
        max_runtime_seconds=3600.0,
    )
    return _result(
        "audit_log",
        "ok",
        audit_entries=len(report.audit_log),
        overall_severity=report.overall_severity.value,
        fail_safe_mode=report.fail_safe_mode,
        report=report.to_dict(),
    )


def audit_log(**context: Any) -> dict[str, Any]:
    return _run_task("audit_log", _audit_log_impl, **context)


with DAG(
    dag_id="weekly_rebalance_pipeline",
    description="Friday rebalance orchestration with portfolio risk controls.",
    schedule="0 17 * * 5",
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
