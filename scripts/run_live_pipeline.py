from __future__ import annotations
# ruff: noqa: E402

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
import time
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import install_runtime_optimizations, write_json_atomic
from scripts.run_live_ic_validation import filter_minimum_cross_sections, load_champion_model
from scripts.run_single_window_validation import align_panel, configure_logging, fill_feature_matrix, json_safe, long_to_feature_matrix
from src.backtest.execution import prepare_execution_price_frame
from src.data.db.models import AuditLog
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_engine, get_session_factory
from src.features.pipeline import FeaturePipeline
from src.labels.forward_returns import compute_forward_returns
from src.models.champion_challenger import ChampionChallengerRunner
from src.models.bundle_validator import BundleSchemaError, BundleValidator
from src.models.evaluation import information_coefficient_series
from src.models.registry import ModelRegistry
from src.portfolio.equal_weight import equal_weight_portfolio
from src.risk.data_risk import DataRiskMonitor
from src.risk.operational_risk import OperationalRiskMonitor
from src.risk.portfolio_risk import PortfolioRiskEngine, compute_sector_weights, sector_weight_deviation
from src.risk.signal_risk import SignalRiskMonitor
from src.universe.active import get_active_universe

DEFAULT_OUTPUT_PATH = REPO_ROOT / "data/reports/day0_live_pipeline.json"
DEFAULT_BUNDLE_PATH = REPO_ROOT / "data/models/fusion_model_bundle_60d.json"
BENCHMARK_TICKER = "SPY"
DEFAULT_MODEL_NAME = "ridge_60d"
DEFAULT_SELECTION_PCT = 0.10
DEFAULT_HISTORY_LOOKBACK_DAYS = 120
DEFAULT_FEATURE_DRIFT_LOOKBACK_DAYS = 60
DEFAULT_MIN_SIGNAL_CROSS_SECTION = 50
PORTFOLIO_MAX_SINGLE_STOCK_WEIGHT = 0.10
PORTFOLIO_MAX_SECTOR_DEVIATION = 0.15
PORTFOLIO_CVAR_FLOOR = -0.05
PORTFOLIO_BETA_BOUNDS = (0.7, 1.3)
PORTFOLIO_TURNOVER_CAP = 0.40
PORTFOLIO_MIN_HOLDINGS = 20
PORTFOLIO_STRESS_WARNING = -0.12


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    install_runtime_optimizations()
    started = time.perf_counter()
    as_of = datetime.now(timezone.utc)

    bundle_validator = BundleValidator(DEFAULT_BUNDLE_PATH)
    session_factory = get_session_factory()
    with session_factory() as session:
        try:
            bundle_validation = bundle_validator.assert_valid(session)
        except BundleSchemaError as exc:
            logger.error(
                "Bundle schema validation failed for {}: {}",
                DEFAULT_BUNDLE_PATH,
                exc,
            )
            raise

    db_state = load_db_state(as_of=as_of)
    if db_state["latest_pit_trade_date"] is None:
        raise RuntimeError("No PIT-visible prices are available in stock_prices.")

    live_trade_date = db_state["latest_pit_trade_date"]
    live_universe = load_live_universe(
        trade_date=live_trade_date,
        as_of=as_of,
        exclude_ticker=BENCHMARK_TICKER,
    )
    if not live_universe:
        raise RuntimeError("No live tickers were available in stocks after excluding SPY.")

    registry = ModelRegistry()
    champion = registry.get_champion(args.model_name)
    if champion is None:
        raise RuntimeError(f"No champion model is registered for {args.model_name!r}.")
    if champion.metadata is None:
        raise RuntimeError(f"Champion model {args.model_name!r} is missing metadata.")

    model, model_load_audit = load_champion_model(registry=registry, champion=champion)
    model_features = list(champion.metadata.features)
    if not model_features:
        raise RuntimeError("Champion metadata.features is empty.")
    logger.info(
        "Validated live bundle {} (version={}, required_features={}, fingerprint={})",
        DEFAULT_BUNDLE_PATH,
        bundle_validation.metadata.get("version"),
        bundle_validation.metadata.get("required_feature_count"),
        bundle_validation.metadata.get("computed_fingerprint"),
    )

    horizon_days = parse_horizon_days(champion.metadata.horizon)
    feature_start = live_trade_date - timedelta(days=args.history_lookback_days)
    price_tickers = [*live_universe, BENCHMARK_TICKER]
    prices = get_prices_pit(
        tickers=price_tickers,
        start_date=feature_start,
        end_date=live_trade_date,
        as_of=as_of,
    )
    if prices.empty:
        raise RuntimeError("No PIT prices were returned for live pipeline execution.")

    pipeline = FeaturePipeline()
    features_long = pipeline.run(
        tickers=live_universe,
        start_date=feature_start,
        end_date=live_trade_date,
        as_of=as_of,
    )
    if features_long.empty:
        raise RuntimeError("Feature pipeline returned no rows for the live execution window.")

    batch_id = str(features_long.attrs.get("batch_id") or pipeline.last_batch_id or "")
    current_features_long = features_long.loc[
        pd.to_datetime(features_long["trade_date"]).dt.date == live_trade_date,
    ].copy()
    if current_features_long.empty:
        raise RuntimeError(f"No live features were produced for PIT-visible date {live_trade_date.isoformat()}.")

    feature_rows_saved = pipeline.save_to_store(current_features_long, batch_id=batch_id) if batch_id else 0
    historical_feature_matrix = build_feature_matrix(features_long, model_features)
    current_feature_matrix = historical_feature_matrix.loc[
        historical_feature_matrix.index.get_level_values("trade_date") == pd.Timestamp(live_trade_date),
    ].copy()
    if current_feature_matrix.empty:
        raise RuntimeError("Live feature matrix is empty after model-column alignment.")

    live_scores = ChampionChallengerRunner._predict_series(model, current_feature_matrix).sort_values(ascending=False)
    if live_scores.empty:
        raise RuntimeError("Champion model returned no live scores.")
    live_scores_by_ticker = flatten_score_index(live_scores)

    raw_weights = equal_weight_portfolio(live_scores_by_ticker, selection_pct=args.selection_pct)
    if not raw_weights:
        raise RuntimeError("Equal-weight portfolio construction returned no target weights.")

    return_history, spy_returns = build_return_history(prices)
    sector_map = load_sector_map()
    benchmark_weights = {ticker: 1.0 / len(live_scores_by_ticker) for ticker in live_scores_by_ticker.index.astype(str)}

    data_monitor = DataRiskMonitor()
    signal_monitor = SignalRiskMonitor()
    portfolio_engine = PortfolioRiskEngine()
    operational_monitor = OperationalRiskMonitor()

    historical_only_matrix = historical_feature_matrix.loc[
        historical_feature_matrix.index.get_level_values("trade_date") < pd.Timestamp(live_trade_date),
    ].copy()
    data_report = data_monitor.run_all_checks(
        data=pd.DataFrame({"ticker": live_scores_by_ticker.index.astype(str)}),
        universe_size=len(live_universe),
        current_features=current_feature_matrix.droplevel("trade_date"),
        historical_features=historical_only_matrix,
        response_times=[],
        error_count=0,
        consecutive_failures=0,
        feature_lookback_days=args.feature_drift_lookback_days,
    )

    signal_inputs = build_signal_risk_inputs(
        model=model,
        feature_matrix=historical_feature_matrix,
        prices=prices,
        horizon_days=horizon_days,
        min_cross_section=args.min_signal_cross_section,
    )
    signal_report = signal_monitor.run_all_checks(
        ic_history=signal_inputs["ic_history"],
        predicted_scores=signal_inputs["predicted_scores"],
        realized_returns=signal_inputs["realized_returns"],
        champion_ic=signal_inputs["champion_ic"],
        challenger_ic=signal_inputs["challenger_ic"],
        consecutive_challenger_wins=0,
    )

    constrained = portfolio_engine.apply_all_constraints(
        weights=raw_weights,
        benchmark_weights=benchmark_weights,
        sector_map=sector_map,
        return_history=return_history.reindex(columns=list(live_scores_by_ticker.index.astype(str))),
        spy_returns=spy_returns,
        current_weights={},
        candidate_ranking=list(live_scores_by_ticker.index.astype(str)),
        max_single_stock_weight=PORTFOLIO_MAX_SINGLE_STOCK_WEIGHT,
        max_sector_deviation=PORTFOLIO_MAX_SECTOR_DEVIATION,
        cvar_floor=PORTFOLIO_CVAR_FLOOR,
        beta_hard_bounds=PORTFOLIO_BETA_BOUNDS,
        turnover_cap=PORTFOLIO_TURNOVER_CAP,
        min_holdings=PORTFOLIO_MIN_HOLDINGS,
        stress_warning_threshold=PORTFOLIO_STRESS_WARNING,
    )
    portfolio_checks = summarize_portfolio_checks(
        constrained=constrained,
        benchmark_weights=benchmark_weights,
        sector_map=sector_map,
    )

    critical_alerts: list[str] = []
    if data_report.halt_pipeline:
        critical_alerts.append("layer1_data_halt")
    if signal_report.recommend_switch:
        critical_alerts.append("layer2_signal_switch")
    if not portfolio_checks["overall_pass"]:
        critical_alerts.append("layer3_portfolio_fail")

    runtime_seconds = time.perf_counter() - started
    operational_report = operational_monitor.run_all_checks(
        runtime_seconds=runtime_seconds,
        critical_alerts=critical_alerts,
        audit_events=[
            operational_monitor.audit_decision(
                action="live_features_generated",
                actor="run_live_pipeline",
                details={
                    "trade_date": live_trade_date.isoformat(),
                    "feature_batch_id": batch_id,
                    "feature_rows": int(len(current_features_long)),
                },
            ),
            operational_monitor.audit_decision(
                action="live_scores_generated",
                actor="run_live_pipeline",
                details={
                    "model_name": champion.name,
                    "model_version": int(champion.version),
                    "top_score_ticker": str(live_scores_by_ticker.index[0]),
                },
            ),
            operational_monitor.audit_decision(
                action="portfolio_risk_evaluated",
                actor="run_live_pipeline",
                details={
                    "overall_pass": bool(portfolio_checks["overall_pass"]),
                    "triggered_rules": portfolio_checks["triggered_rules"],
                },
            ),
        ],
    )

    audit_insert = persist_audit_log(
        audit_records=operational_report.audit_log,
        model_version_id=champion.run_id,
        feature_batch_id=batch_id or None,
    )

    issues = []
    if db_state["latest_stored_trade_date"] and db_state["latest_stored_trade_date"] > live_trade_date:
        issues.append(
            {
                "code": "pit_lag_due_to_knowledge_time",
                "message": (
                    "Latest stored price rows are newer than the PIT-visible cutoff; "
                    "live scoring used the latest knowledge-time-safe date."
                ),
            },
        )
    if db_state["universe_membership_live_count"] == 0:
        issues.append(
            {
                "code": "stale_universe_membership",
                "message": (
                    "universe_membership has no active 2026 constituents, so the live run used "
                    "the tracked stocks table as the current universe source."
                ),
            },
        )
    if signal_inputs["challenger_available"] is False:
        issues.append(
            {
                "code": "no_challenger_registered",
                "message": "No challenger model is registered; model-switch checks ran in neutral mode.",
            },
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_success": True,
        "db_state": {
            "latest_stored_trade_date": none_or_iso(db_state["latest_stored_trade_date"]),
            "latest_pit_trade_date": none_or_iso(live_trade_date),
            "as_of_utc": as_of.isoformat(),
            "stock_universe_size": int(len(live_universe)),
            "universe_membership_live_count": int(db_state["universe_membership_live_count"]),
        },
        "feature_pipeline": {
            "start_date": feature_start.isoformat(),
            "end_date": live_trade_date.isoformat(),
            "batch_id": batch_id,
            "feature_rows_total": int(len(features_long)),
            "feature_rows_live_date": int(len(current_features_long)),
            "feature_store_rows_saved": int(feature_rows_saved),
            "feature_matrix_shape": {
                "n_stocks": int(current_feature_matrix.shape[0]),
                "n_features": int(current_feature_matrix.shape[1]),
            },
        },
        "model": {
            "model_name": champion.name,
            "version": int(champion.version),
            "stage": champion.stage.value,
            "run_id": champion.run_id,
            "horizon": champion.metadata.horizon,
            "n_features": int(champion.metadata.n_features),
            "load_audit": model_load_audit,
        },
        "live_outputs": {
            "signal_date": live_trade_date.isoformat(),
            "top_10_scores": [
                {"ticker": str(ticker), "score": float(score)}
                for ticker, score in live_scores_by_ticker.head(10).items()
            ],
            "target_weights_raw": normalize_weight_dict(raw_weights),
            "target_weights_after_risk": normalize_weight_dict(constrained.weights),
        },
        "risk_checks": {
            "layer1_data": {
                "pass": bool(not data_report.halt_pipeline),
                "severity": data_report.overall_severity.value,
                "report": data_report.to_dict(),
            },
            "layer2_signal": {
                "pass": bool(not signal_report.recommend_switch),
                "severity": signal_report.overall_severity.value,
                "report": signal_report.to_dict(),
                "history_dates": int(signal_inputs["history_dates"]),
                "history_rows": int(signal_inputs["history_rows"]),
            },
            "layer3_portfolio": {
                "pass": bool(portfolio_checks["overall_pass"]),
                "checks": portfolio_checks,
                "report": constrained.to_dict(),
            },
            "layer4_operational": {
                "pass": bool(not operational_report.halt_pipeline),
                "severity": operational_report.overall_severity.value,
                "report": operational_report.to_dict(),
            },
        },
        "audit_log": audit_insert,
        "issues": issues,
    }

    write_json_atomic(args.output_path, json_safe(report))
    logger.info("saved live pipeline report to {}", args.output_path)
    logger.info(
        "live pipeline success trade_date={} feature_shape={} top_ticker={} audit_rows={}",
        live_trade_date,
        tuple(current_feature_matrix.shape),
        live_scores_by_ticker.index[0],
        audit_insert["inserted_count"],
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PIT-safe live pipeline end to end and persist a Day 0 execution report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--selection-pct", type=float, default=DEFAULT_SELECTION_PCT)
    parser.add_argument("--history-lookback-days", type=int, default=DEFAULT_HISTORY_LOOKBACK_DAYS)
    parser.add_argument("--feature-drift-lookback-days", type=int, default=DEFAULT_FEATURE_DRIFT_LOOKBACK_DAYS)
    parser.add_argument("--min-signal-cross-section", type=int, default=DEFAULT_MIN_SIGNAL_CROSS_SECTION)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def load_db_state(*, as_of: datetime) -> dict[str, Any]:
    engine = get_engine()
    with engine.connect() as conn:
        latest_stored_trade_date = conn.execute(text("select max(trade_date) from stock_prices")).scalar()
        latest_pit_trade_date = conn.execute(
            text("select max(trade_date) from stock_prices where knowledge_time <= :as_of"),
            {"as_of": as_of},
        ).scalar()
        if latest_pit_trade_date is None:
            universe_membership_live_count = 0
            latest_pit_trade_ticker_count = 0
            previous_pit_trade_date = None
            expected_live_universe_count = 0
        else:
            previous_pit_trade_date = conn.execute(
                text(
                    """
                    select max(trade_date)
                    from stock_prices
                    where knowledge_time <= :as_of
                      and trade_date < :trade_date
                    """,
                ),
                {"as_of": as_of, "trade_date": latest_pit_trade_date},
            ).scalar()
            universe_membership_live_count = int(
                conn.execute(
                    text(
                        """
                        select count(distinct ticker)
                        from universe_membership
                        where index_name = 'SP500'
                          and effective_date <= :trade_date
                          and (end_date is null or end_date > :trade_date)
                        """,
                    ),
                    {"trade_date": latest_pit_trade_date},
                ).scalar()
                or 0,
            )
            latest_pit_trade_ticker_count = int(
                conn.execute(
                    text(
                        """
                        with live_universe as (
                            select distinct ticker
                            from universe_membership
                            where index_name = 'SP500'
                              and effective_date <= :trade_date
                              and (end_date is null or end_date > :trade_date)
                        )
                        select count(distinct sp.ticker)
                        from stock_prices sp
                        join live_universe u on u.ticker = sp.ticker
                        where sp.trade_date = :trade_date
                          and sp.knowledge_time <= :as_of
                        """,
                    ),
                    {"trade_date": latest_pit_trade_date, "as_of": as_of},
                ).scalar()
                or 0,
            )
            baseline_trade_date = previous_pit_trade_date or latest_pit_trade_date
            baseline_coverage = int(
                conn.execute(
                    text(
                        """
                        with live_universe as (
                            select distinct ticker
                            from universe_membership
                            where index_name = 'SP500'
                              and effective_date <= :latest_trade_date
                              and (end_date is null or end_date > :latest_trade_date)
                        )
                        select count(distinct sp.ticker)
                        from stock_prices sp
                        join live_universe u on u.ticker = sp.ticker
                        where sp.trade_date = :baseline_trade_date
                          and sp.knowledge_time <= :as_of
                        """,
                    ),
                    {
                        "latest_trade_date": latest_pit_trade_date,
                        "baseline_trade_date": baseline_trade_date,
                        "as_of": as_of,
                    },
                ).scalar()
                or 0,
            )
            membership_floor = int(universe_membership_live_count * 0.95)
            expected_live_universe_count = max(baseline_coverage, membership_floor)
    return {
        "latest_stored_trade_date": latest_stored_trade_date,
        "latest_pit_trade_date": latest_pit_trade_date,
        "previous_pit_trade_date": previous_pit_trade_date,
        "universe_membership_live_count": universe_membership_live_count,
        "expected_live_universe_count": expected_live_universe_count,
        "latest_pit_trade_ticker_count": latest_pit_trade_ticker_count,
    }


def load_live_universe(*, trade_date: date, as_of: datetime, exclude_ticker: str) -> list[str]:
    return get_active_universe(
        trade_date,
        as_of=as_of,
        benchmark_ticker=exclude_ticker,
    )


def load_sector_map() -> dict[str, str]:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("select ticker, sector from stocks"))
        frame = pd.DataFrame(result.fetchall(), columns=result.keys())
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["sector"] = frame["sector"].fillna("Unknown").astype(str)
    return frame.set_index("ticker")["sector"].to_dict()


def build_feature_matrix(features_long: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    matrix = long_to_feature_matrix(
        features_long.loc[:, ["ticker", "trade_date", "feature_name", "feature_value"]].copy(),
        model_features,
    ).reindex(columns=model_features)
    return fill_feature_matrix(matrix)


def build_return_history(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    execution = prepare_execution_price_frame(prices)
    return_history = (
        execution["daily_return"]
        .unstack("ticker")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    spy_returns = return_history[BENCHMARK_TICKER].dropna() if BENCHMARK_TICKER in return_history.columns else pd.Series(dtype=float)
    return return_history, spy_returns


def build_signal_risk_inputs(
    *,
    model: Any,
    feature_matrix: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int,
    min_cross_section: int,
) -> dict[str, Any]:
    price_frame = prices.copy()
    price_frame["trade_date"] = pd.to_datetime(price_frame["trade_date"])
    labels = compute_forward_returns(
        prices_df=price_frame,
        horizons=(horizon_days,),
        benchmark_ticker=BENCHMARK_TICKER,
    )
    labels = labels.loc[
        (labels["horizon"] == horizon_days)
        & (labels["ticker"].astype(str).str.upper() != BENCHMARK_TICKER)
    ].copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    label_series = labels.set_index(["trade_date", "ticker"])["excess_return"].sort_index().dropna()

    prediction_series = ChampionChallengerRunner._predict_series(model, feature_matrix)
    aligned_X, aligned_y = align_panel(feature_matrix, label_series)
    aligned_predictions = prediction_series.reindex(aligned_X.index).dropna()
    aligned_y = aligned_y.reindex(aligned_predictions.index).dropna()
    aligned_predictions = aligned_predictions.reindex(aligned_y.index)

    filtered_y, filtered_pred, sizes = filter_minimum_cross_sections(
        y=aligned_y,
        y_pred=aligned_predictions,
        min_size=min_cross_section,
    )
    ic_series = information_coefficient_series(y_true=filtered_y, y_pred=filtered_pred)

    return {
        "predicted_scores": filtered_pred,
        "realized_returns": filtered_y,
        "ic_history": [float(value) for value in ic_series.tolist()],
        "champion_ic": float(np.nanmean(ic_series.to_numpy(dtype=float))) if len(ic_series) else 0.0,
        "challenger_ic": float(np.nanmean(ic_series.to_numpy(dtype=float))) if len(ic_series) else 0.0,
        "challenger_available": False,
        "history_dates": int(len(sizes)),
        "history_rows": int(len(filtered_y)),
    }


def flatten_score_index(scores: pd.Series) -> pd.Series:
    if isinstance(scores.index, pd.MultiIndex) and "ticker" in scores.index.names:
        return pd.Series(
            scores.to_numpy(dtype=float),
            index=pd.Index(scores.index.get_level_values("ticker").astype(str), name="ticker"),
            name=str(scores.name or "score"),
            dtype=float,
        ).sort_values(ascending=False)
    return pd.Series(scores, dtype=float).sort_values(ascending=False)


def summarize_portfolio_checks(
    *,
    constrained: Any,
    benchmark_weights: dict[str, float],
    sector_map: dict[str, str],
) -> dict[str, Any]:
    max_weight = max(constrained.weights.values()) if constrained.weights else 0.0
    sector_deviation = sector_weight_deviation(
        constrained.sector_weights,
        compute_sector_weights(benchmark_weights, sector_map),
    )
    max_abs_sector_deviation = max((abs(value) for value in sector_deviation.values()), default=0.0)
    stress_entry = next((entry for entry in constrained.audit_trail if entry.rule_name == "stress_test"), None)
    stress_return = (
        float(stress_entry.after.get("stressed_return"))
        if stress_entry is not None and stress_entry.after.get("stressed_return") is not None
        else None
    )
    checks = {
        "max_single_stock_weight": {
            "pass": bool(max_weight <= PORTFOLIO_MAX_SINGLE_STOCK_WEIGHT + 1e-12),
            "value": float(max_weight),
            "limit": float(PORTFOLIO_MAX_SINGLE_STOCK_WEIGHT),
        },
        "sector_deviation_cap": {
            "pass": bool(max_abs_sector_deviation <= PORTFOLIO_MAX_SECTOR_DEVIATION + 1e-12),
            "value": float(max_abs_sector_deviation),
            "limit": float(PORTFOLIO_MAX_SECTOR_DEVIATION),
        },
        "cvar_floor": {
            "pass": bool(constrained.cvar_99 is None or constrained.cvar_99 >= PORTFOLIO_CVAR_FLOOR - 1e-12),
            "value": None if constrained.cvar_99 is None else float(constrained.cvar_99),
            "limit": float(PORTFOLIO_CVAR_FLOOR),
        },
        "beta_bounds": {
            "pass": bool(
                constrained.portfolio_beta is None
                or (PORTFOLIO_BETA_BOUNDS[0] <= constrained.portfolio_beta <= PORTFOLIO_BETA_BOUNDS[1])
            ),
            "value": None if constrained.portfolio_beta is None else float(constrained.portfolio_beta),
            "bounds": [float(PORTFOLIO_BETA_BOUNDS[0]), float(PORTFOLIO_BETA_BOUNDS[1])],
        },
        "turnover_cap": {
            "pass": bool(constrained.turnover <= PORTFOLIO_TURNOVER_CAP + 1e-12),
            "value": float(constrained.turnover),
            "limit": float(PORTFOLIO_TURNOVER_CAP),
        },
        "min_holdings": {
            "pass": bool(constrained.holding_count >= PORTFOLIO_MIN_HOLDINGS),
            "value": int(constrained.holding_count),
            "limit": int(PORTFOLIO_MIN_HOLDINGS),
        },
        "stress_test": {
            "pass": bool(stress_return is None or stress_return >= PORTFOLIO_STRESS_WARNING - 1e-12),
            "value": stress_return,
            "limit": float(PORTFOLIO_STRESS_WARNING),
        },
        "audit_trail_complete": {
            "pass": bool(len(constrained.audit_trail) == 8),
            "value": int(len(constrained.audit_trail)),
            "limit": 8,
        },
        "cash_non_negative": {
            "pass": bool(constrained.cash_weight >= -1e-12),
            "value": float(constrained.cash_weight),
            "limit": 0.0,
        },
        "gross_exposure_non_negative": {
            "pass": bool(constrained.gross_exposure >= -1e-12),
            "value": float(constrained.gross_exposure),
            "limit": 0.0,
        },
    }
    return {
        "overall_pass": bool(all(item["pass"] for item in checks.values())),
        "checks": checks,
        "triggered_rules": [entry.rule_name for entry in constrained.audit_trail if entry.triggered],
        "warnings": list(constrained.warnings),
        "sector_deviation_vs_equal_weight_universe": sector_deviation,
        "max_abs_sector_deviation_vs_equal_weight_universe": float(max_abs_sector_deviation),
    }


def persist_audit_log(
    *,
    audit_records: list[Any],
    model_version_id: str | None,
    feature_batch_id: str | None,
) -> dict[str, Any]:
    session_factory = get_session_factory()
    inserted_ids: list[int] = []
    with session_factory() as session:
        for record in audit_records:
            row = AuditLog(
                action=str(record.action),
                actor=str(record.actor),
                model_version_id=model_version_id,
                feature_batch_id=feature_batch_id,
                details_json=dict(record.details),
            )
            session.add(row)
            session.flush()
            inserted_ids.append(int(row.id))
        session.commit()
    return {
        "inserted_count": int(len(inserted_ids)),
        "inserted_ids": inserted_ids,
        "feature_batch_id": feature_batch_id,
    }


def parse_horizon_days(value: str) -> int:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        raise ValueError(f"Unable to parse horizon days from {value!r}.")
    return int(digits)


def normalize_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    return {str(ticker): float(weight) for ticker, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0]))}


def none_or_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
