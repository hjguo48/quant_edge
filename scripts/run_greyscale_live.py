from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import install_runtime_optimizations, write_json_atomic
from scripts.run_ic_weighted_fusion import cross_sectional_zscore, softmax
from scripts.run_live_ic_validation import filter_minimum_cross_sections
from scripts.run_live_pipeline import (
    BENCHMARK_TICKER,
    PORTFOLIO_BETA_BOUNDS,
    PORTFOLIO_CVAR_FLOOR,
    PORTFOLIO_MAX_SECTOR_DEVIATION,
    PORTFOLIO_MAX_SINGLE_STOCK_WEIGHT,
    PORTFOLIO_MIN_HOLDINGS,
    PORTFOLIO_STRESS_WARNING,
    PORTFOLIO_TURNOVER_CAP,
    build_return_history,
    flatten_score_index,
    load_db_state,
    load_sector_map,
    none_or_iso,
    normalize_weight_dict,
    summarize_portfolio_checks,
)
from scripts.run_regime_analysis import classify_vix, load_macro_series
from scripts.run_single_window_validation import fill_feature_matrix, long_to_feature_matrix, restore_feature_matrix_index
from scripts.run_walkforward_comparison import json_safe
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_engine
from src.features.pipeline import FeaturePipeline
from src.labels.forward_returns import compute_forward_returns
from src.models.evaluation import information_coefficient, information_coefficient_series
from src.portfolio.equal_weight import equal_weight_portfolio
from src.portfolio.event_overlay import NullOverlay
from src.portfolio.constraints import (
    PortfolioConstraints,
    apply_turnover_buffer,
    apply_weight_constraints,
)
from src.backtest.execution import select_candidate_tickers
from src.stats.psi import compute_feature_psi_report
from src.risk.data_risk import DataRiskMonitor
from src.risk.operational_risk import OperationalRiskMonitor
from src.risk.portfolio_risk import PortfolioRiskEngine, compute_turnover
from src.risk.signal_risk import SignalRiskMonitor

DEFAULT_BUNDLE_PATH = "data/models/fusion_model_bundle_60d.json"
DEFAULT_REPORT_DIR = "data/reports/greyscale"
DEFAULT_REFERENCE_FEATURE_MATRIX_PATH = "data/features/walkforward_feature_matrix_60d.parquet"
DEFAULT_SELECTION_PCT = 0.25
DEFAULT_HISTORY_LOOKBACK_DAYS = 400
DEFAULT_FEATURE_DRIFT_LOOKBACK_DAYS = 60
DEFAULT_MIN_SIGNAL_CROSS_SECTION = 50
DEFAULT_SIGNAL_LOOKBACK_POINTS = 12
DEFAULT_LOW_VIX_THRESHOLD = 20.0
DEFAULT_HIGH_VIX_THRESHOLD = 30.0
MODEL_NAMES = ("ridge", "xgboost", "lightgbm")
FUSION_NAME = "fusion"
WEEK_REPORT_PATTERN = re.compile(r"week_(\d+)\.json$")
REPO_PATH_ANCHORS = ("data", "src", "scripts", "dags", "configs", "mlruns")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    install_runtime_optimizations()

    as_of = parse_as_of(args.as_of)
    started = time.perf_counter()
    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle = json.loads((REPO_ROOT / args.bundle_path).read_text())
    models = load_models(bundle)
    retained_features = list(bundle["retained_features"])
    seed_weights = normalize_weight_dict_local(bundle["seed_weights"])
    regime_weights = normalize_weight_dict_local(bundle.get("regime_weights", {}), fill_unknown=True)
    output_path = resolve_output_path(report_dir=report_dir, explicit=args.output_path)

    db_state = load_db_state(as_of=as_of)
    live_trade_date = db_state["latest_pit_trade_date"]
    if live_trade_date is None:
        raise RuntimeError("No PIT-visible trade date available for greyscale live run.")

    live_universe = load_live_universe(trade_date=live_trade_date, as_of=as_of, benchmark_ticker=args.benchmark_ticker)
    live_universe = apply_universe_filters(
        tickers=live_universe,
        limit=args.limit_tickers,
        requested_tickers=args.tickers,
    )
    if not live_universe:
        raise RuntimeError("No live universe tickers were available for the PIT-visible trade date.")

    existing_reports = load_greyscale_reports(report_dir)
    realized_labels = build_realized_label_series_from_reports(
        reports=existing_reports,
        latest_pit_trade_date=live_trade_date,
        as_of=as_of,
        horizon=args.horizon,
        benchmark_ticker=args.benchmark_ticker,
    )
    realized_ic_frame = build_realized_ic_frame(
        reports=existing_reports,
        realized_labels=realized_labels,
        model_names=list(MODEL_NAMES),
    )
    live_weights, weight_source, historical_live_ic_frame = resolve_live_weights(
        realized_ic_frame=realized_ic_frame,
        seed_weights=seed_weights,
        model_names=list(MODEL_NAMES),
        rolling_window=args.signal_lookback_points,
        temperature=float(bundle.get("fusion_temperature", args.temperature)),
    )

    feature_start = live_trade_date - timedelta(days=args.history_lookback_days)
    price_tickers = [*live_universe, args.benchmark_ticker]
    prices = get_prices_pit(
        tickers=price_tickers,
        start_date=feature_start,
        end_date=live_trade_date,
        as_of=as_of,
    )
    if prices.empty:
        raise RuntimeError("No PIT prices returned for greyscale live execution.")

    pipeline = FeaturePipeline()
    features_long = pipeline.run(
        tickers=live_universe,
        start_date=live_trade_date,
        end_date=live_trade_date,
        as_of=as_of,
    )
    if features_long.empty:
        raise RuntimeError("FeaturePipeline returned no rows for greyscale live execution.")

    batch_id = str(features_long.attrs.get("batch_id") or pipeline.last_batch_id or "")
    current_features_long = features_long.copy()
    if current_features_long.empty:
        raise RuntimeError(f"No live features were produced for {live_trade_date.isoformat()}.")

    current_feature_matrix = build_feature_matrix(
        features_long=current_features_long,
        retained_features=retained_features,
    )
    if current_feature_matrix.empty:
        raise RuntimeError("Current greyscale feature matrix is empty after alignment.")
    historical_feature_matrix = load_reference_feature_matrix(
        path=REPO_ROOT / args.reference_feature_matrix_path,
        retained_features=retained_features,
        tickers=live_universe,
        max_dates=max(args.feature_drift_lookback_days, args.signal_lookback_points),
    )

    psi_cfg = bundle.get("psi_monitoring", {})
    psi_report_data: list[dict] = []
    if psi_cfg.get("enabled") and not historical_feature_matrix.empty:
        try:
            ref_df = historical_feature_matrix.reset_index(level="trade_date", drop=True)
            cur_df = current_feature_matrix.reset_index(level="trade_date", drop=True)
            psi_report_data = compute_feature_psi_report(
                reference_df=ref_df,
                current_df=cur_df,
                feature_columns=retained_features,
                n_bins=int(psi_cfg.get("n_bins", 10)),
                psi_alert_threshold=float(psi_cfg.get("psi_alert_threshold", 0.25)),
                fill_rate_change_threshold=float(psi_cfg.get("fill_rate_change_threshold", 0.05)),
            )
            psi_alerts = [r for r in psi_report_data if r.get("psi_alert")]
            if psi_alerts:
                logger.warning("PSI alerts on {} features: {}", len(psi_alerts),
                               [r["feature"] for r in psi_alerts])
            else:
                logger.info("PSI monitoring: all {} features stable", len(psi_report_data))
        except Exception as exc:
            logger.warning("PSI monitoring failed: {}", exc)

    raw_predictions, normalized_predictions = score_live_cross_section(
        models=models,
        current_feature_matrix=current_feature_matrix,
    )
    current_vix, regime_name, regime_scalar = resolve_current_regime(
        live_trade_date=live_trade_date,
        as_of=as_of,
        regime_weights=regime_weights,
        low_threshold=args.low_vix_threshold,
        high_threshold=args.high_vix_threshold,
    )
    regime_adjusted_weights = apply_regime_to_model_weights(live_weights, regime_scalar)
    fusion_scores = combine_current_predictions(normalized_predictions, regime_adjusted_weights).rename(FUSION_NAME)

    overlay = NullOverlay()
    fused_scores_by_ticker = overlay.apply(
        flatten_score_index(fusion_scores),
        pd.DataFrame(),
    ).sort_values(ascending=False)
    model_scores_by_ticker = {
        model_name: flatten_score_index(series).sort_values(ascending=False)
        for model_name, series in raw_predictions.items()
    }
    model_top_10_scores = {
        model_name: series.head(10)
        for model_name, series in model_scores_by_ticker.items()
    }
    pairwise_rank_correlation = compute_pairwise_rank_correlations(model_scores_by_ticker)

    previous_target_weights = extract_previous_target_weights(existing_reports)
    turnover_cfg = bundle.get("turnover_controls", {})
    if turnover_cfg.get("enabled") and turnover_cfg.get("weighting_scheme") == "score_weighted":
        raw_weights = build_score_weighted_portfolio(
            scores=fused_scores_by_ticker,
            previous_weights=previous_target_weights,
            turnover_cfg=turnover_cfg,
            selection_pct=args.selection_pct,
        )
        portfolio_scheme = "score_weighted"
    else:
        raw_weights = equal_weight_portfolio(fused_scores_by_ticker, selection_pct=args.selection_pct)
        portfolio_scheme = "equal_weight"
    if not raw_weights:
        raise RuntimeError("Portfolio construction returned no candidate weights.")

    return_history, spy_returns = build_return_history(prices)
    sector_map = load_sector_map()
    benchmark_weights = {
        ticker: 1.0 / len(fused_scores_by_ticker)
        for ticker in fused_scores_by_ticker.index.astype(str)
    }

    data_monitor = DataRiskMonitor()
    signal_monitor = SignalRiskMonitor()
    portfolio_engine = PortfolioRiskEngine()
    operational_monitor = OperationalRiskMonitor()

    data_report = data_monitor.run_all_checks(
        data=pd.DataFrame({"ticker": fused_scores_by_ticker.index.astype(str)}),
        universe_size=len(live_universe),
        current_features=current_feature_matrix.droplevel("trade_date"),
        historical_features=historical_feature_matrix,
        response_times=[],
        error_count=0,
        consecutive_failures=0,
        feature_lookback_days=args.feature_drift_lookback_days,
    )

    signal_state = build_signal_risk_state(
        signal_monitor=signal_monitor,
        realized_ic_frame=realized_ic_frame,
        required_points=args.signal_lookback_points,
        model_names=list(MODEL_NAMES),
    )

    constrained = portfolio_engine.apply_all_constraints(
        weights=raw_weights,
        benchmark_weights=benchmark_weights,
        sector_map=sector_map,
        return_history=return_history.reindex(columns=list(fused_scores_by_ticker.index.astype(str))),
        spy_returns=spy_returns,
        current_weights=previous_target_weights,
        candidate_ranking=list(fused_scores_by_ticker.index.astype(str)),
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

    top_tickers_for_shap = list(fused_scores_by_ticker.head(100).index.astype(str))
    try:
        shap_data = compute_shap_for_top_tickers(
            models=models,
            feature_matrix=current_feature_matrix,
            top_tickers=top_tickers_for_shap,
        )
    except Exception as exc:
        logger.warning("failed to compute SHAP values for greyscale live report: {}", exc)
        shap_data = {}

    critical_alerts: list[str] = []
    if data_report.halt_pipeline:
        critical_alerts.append("layer1_data_halt")
    if signal_state["recommend_switch"]:
        critical_alerts.append("layer2_signal_switch")
    if not portfolio_checks["overall_pass"]:
        critical_alerts.append("layer3_portfolio_fail")

    runtime_seconds = time.perf_counter() - started
    operational_report = operational_monitor.run_all_checks(
        runtime_seconds=runtime_seconds,
        critical_alerts=critical_alerts,
        audit_events=[
            operational_monitor.audit_decision(
                action="greyscale_features_generated",
                actor="run_greyscale_live",
                details={
                    "trade_date": live_trade_date.isoformat(),
                    "feature_batch_id": batch_id,
                    "feature_rows_live_date": int(len(current_features_long)),
                },
            ),
            operational_monitor.audit_decision(
                action="greyscale_fusion_scored",
                actor="run_greyscale_live",
                details={
                    "top_ticker": str(fused_scores_by_ticker.index[0]),
                    "weight_source": weight_source,
                    "regime": regime_name,
                    "regime_scalar": float(regime_scalar),
                    "regime_adjusted_weights": regime_adjusted_weights,
                },
            ),
            operational_monitor.audit_decision(
                action="greyscale_portfolio_risk_evaluated",
                actor="run_greyscale_live",
                details={
                    "overall_pass": bool(portfolio_checks["overall_pass"]),
                    "triggered_rules": portfolio_checks["triggered_rules"],
                },
            ),
        ],
    )

    week_number = extract_week_number(output_path)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "week_number": week_number,
        "db_state": {
            "latest_stored_trade_date": none_or_iso(db_state["latest_stored_trade_date"]),
            "latest_pit_trade_date": none_or_iso(live_trade_date),
            "as_of_utc": as_of.isoformat(),
            "stock_universe_size": int(len(live_universe)),
            "universe_membership_live_count": int(db_state["universe_membership_live_count"]),
        },
        "model_bundle": {
            "path": str((REPO_ROOT / args.bundle_path).resolve()),
            "window_id": str(bundle["window_id"]),
            "horizon_days": int(bundle["horizon_days"]),
            "retained_feature_count": int(len(retained_features)),
            "models": {name: bundle["models"][name]["artifact_path"] for name in MODEL_NAMES},
        },
        "feature_pipeline": {
            "start_date": feature_start.isoformat(),
            "end_date": live_trade_date.isoformat(),
            "batch_id": batch_id,
            "feature_rows_total": int(len(features_long)),
            "feature_rows_live_date": int(len(current_features_long)),
            "feature_matrix_shape": {
                "n_stocks": int(current_feature_matrix.shape[0]),
                "n_features": int(current_feature_matrix.shape[1]),
            },
        },
        "fusion": {
            "temperature": float(bundle.get("fusion_temperature", args.temperature)),
            "rolling_window_rebalance_points": int(args.signal_lookback_points),
            "weight_source": weight_source,
            "seed_weights": seed_weights,
            "live_weights": live_weights,
            "historical_live_ic_points": int(len(historical_live_ic_frame)),
            "historical_live_ic_frame": json_safe(
                historical_live_ic_frame.reset_index().to_dict(orient="records")
                if not historical_live_ic_frame.empty
                else []
            ),
            "regime": {
                "vix": None if current_vix is None or not np.isfinite(current_vix) else float(current_vix),
                "regime": regime_name,
                "scalar": float(regime_scalar),
                "weights": regime_weights,
                "regime_adjusted_model_weights": regime_adjusted_weights,
            },
        },
        "live_outputs": {
            "signal_date": live_trade_date.isoformat(),
            "top_10_fusion_scores": series_to_ranked_records(fused_scores_by_ticker.head(10)),
            "top_10_model_scores": {
                name: series_to_ranked_records(series)
                for name, series in model_top_10_scores.items()
            },
            "top_10_holdings_after_risk": weight_dict_to_records(constrained.weights, limit=10),
            "target_weights_raw": normalize_weight_dict(raw_weights),
            "target_weights_after_risk": normalize_weight_dict(constrained.weights),
            "pairwise_rank_correlation": pairwise_rank_correlation,
        },
        "score_vectors": {
            **{name: series_to_float_dict(series) for name, series in model_scores_by_ticker.items()},
            "fusion": series_to_float_dict(fused_scores_by_ticker),
            "fusion_pre_regime": series_to_float_dict(
                flatten_score_index(combine_current_predictions(normalized_predictions, live_weights))
            ),
        },
        "shap_values": json_safe(shap_data),
        "risk_checks": {
            "layer1_data": {
                "pass": bool(not data_report.halt_pipeline),
                "severity": data_report.overall_severity.value,
                "report": data_report.to_dict(),
            },
            "layer2_signal": signal_state,
            "layer3_portfolio": {
                "pass": bool(portfolio_checks["overall_pass"]),
                "checks": portfolio_checks,
                "beta_contributions": constrained.beta_contributions,
                "top_beta_contributors": constrained.top_beta_contributors,
                "report": constrained.to_dict(),
            },
            "layer4_operational": {
                "pass": bool(not operational_report.halt_pipeline),
                "severity": operational_report.overall_severity.value,
                "report": operational_report.to_dict(),
            },
        },
        "portfolio_metrics": {
            "portfolio_scheme": portfolio_scheme,
            "turnover_controls": json_safe(turnover_cfg) if portfolio_scheme == "score_weighted" else None,
            "turnover_vs_previous": float(compute_turnover(constrained.weights, previous_target_weights)),
            "holding_count_after_risk": int(constrained.holding_count),
            "gross_exposure_after_risk": float(constrained.gross_exposure),
            "cash_weight_after_risk": float(constrained.cash_weight),
        },
        "feature_drift_psi": json_safe(psi_report_data) if psi_report_data else None,
        "notes": [
            "Fusion ranking uses cross-sectional z-scored model outputs with rolling-IC weights.",
            "Regime adjusts per-model fusion weights (blend toward equal-weight) to change cross-sectional rankings.",
            f"Portfolio scheme: {portfolio_scheme} (Phase B optimal turnover controls)." if portfolio_scheme == "score_weighted" else "Portfolio scheme: equal_weight.",
            "PSI monitoring checks feature distribution drift against historical reference.",
            "Dry-run mode writes a report but does not place trades or persist audit rows.",
        ],
    }
    write_json_atomic(output_path, json_safe(report))
    logger.info(
        "saved greyscale live report to {} trade_date={} top_ticker={} dry_run={}",
        output_path,
        live_trade_date,
        fused_scores_by_ticker.index[0],
        args.dry_run,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 60D three-model IC-weighted greyscale live pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle-path", default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--reference-feature-matrix-path", default=DEFAULT_REFERENCE_FEATURE_MATRIX_PATH)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--as-of")
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--selection-pct", type=float, default=DEFAULT_SELECTION_PCT)
    parser.add_argument("--history-lookback-days", type=int, default=DEFAULT_HISTORY_LOOKBACK_DAYS)
    parser.add_argument("--feature-drift-lookback-days", type=int, default=DEFAULT_FEATURE_DRIFT_LOOKBACK_DAYS)
    parser.add_argument("--min-signal-cross-section", type=int, default=DEFAULT_MIN_SIGNAL_CROSS_SECTION)
    parser.add_argument("--signal-lookback-points", type=int, default=DEFAULT_SIGNAL_LOOKBACK_POINTS)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--low-vix-threshold", type=float, default=DEFAULT_LOW_VIX_THRESHOLD)
    parser.add_argument("--high-vix-threshold", type=float, default=DEFAULT_HIGH_VIX_THRESHOLD)
    parser.add_argument("--limit-tickers", type=int)
    parser.add_argument("--tickers")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if "T" in value:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return datetime.combine(date.fromisoformat(value), datetime.max.time(), tzinfo=timezone.utc)


def load_models(bundle: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        artifact_path = _resolve_bundle_artifact_path(Path(bundle["models"][model_name]["artifact_path"]))
        with artifact_path.open("rb") as handle:
            models[model_name] = pickle.load(handle)
    return models


def _resolve_bundle_artifact_path(artifact_path: Path) -> Path:
    if artifact_path.exists():
        return artifact_path
    if not artifact_path.is_absolute():
        rebound = REPO_ROOT / artifact_path
        if rebound.exists():
            return rebound
    if REPO_ROOT.name in artifact_path.parts:
        suffix = artifact_path.parts[artifact_path.parts.index(REPO_ROOT.name) + 1 :]
        rebound = REPO_ROOT.joinpath(*suffix)
        if rebound.exists():
            return rebound
    for anchor in REPO_PATH_ANCHORS:
        if anchor in artifact_path.parts:
            suffix = artifact_path.parts[artifact_path.parts.index(anchor) :]
            rebound = REPO_ROOT.joinpath(*suffix)
            if rebound.exists():
                return rebound
    rebound = REPO_ROOT / artifact_path.name
    if rebound.exists():
        return rebound
    raise FileNotFoundError(f"Bundle artifact path is not available in this runtime: {artifact_path}")


def resolve_output_path(*, report_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit if explicit.is_absolute() else REPO_ROOT / explicit
    existing = sorted(report_dir.glob("week_*.json"))
    next_number = 1
    if existing:
        next_number = max(extract_week_number(path) for path in existing) + 1
    return report_dir / f"week_{next_number:02d}.json"


def extract_week_number(path: Path) -> int:
    match = WEEK_REPORT_PATTERN.search(path.name)
    return int(match.group(1)) if match else 0


def load_live_universe(*, trade_date: date, as_of: datetime, benchmark_ticker: str) -> list[str]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                select distinct ticker
                from stock_prices
                where trade_date = :trade_date
                  and knowledge_time <= :as_of
                  and upper(ticker) <> :benchmark
                order by ticker
                """,
            ),
            {
                "trade_date": trade_date,
                "as_of": as_of,
                "benchmark": benchmark_ticker.upper(),
            },
        ).scalars().all()
    return [str(ticker).upper() for ticker in rows]


def apply_universe_filters(
    *,
    tickers: list[str],
    limit: int | None,
    requested_tickers: str | None,
) -> list[str]:
    selected = list(tickers)
    if requested_tickers:
        requested = {ticker.strip().upper() for ticker in requested_tickers.split(",") if ticker.strip()}
        selected = [ticker for ticker in selected if ticker in requested]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit_tickers must be positive.")
        selected = selected[:limit]
    return selected


def build_feature_matrix(*, features_long: pd.DataFrame, retained_features: list[str]) -> pd.DataFrame:
    filtered = features_long.loc[
        features_long["feature_name"].astype(str).isin(retained_features),
        ["ticker", "trade_date", "feature_name", "feature_value"],
    ].copy()
    matrix = long_to_feature_matrix(filtered, retained_features).reindex(columns=retained_features)
    return fill_feature_matrix(matrix)


def load_reference_feature_matrix(
    *,
    path: Path,
    retained_features: list[str],
    tickers: list[str],
    max_dates: int,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=retained_features, dtype=float)
    frame = pd.read_parquet(path)
    matrix = restore_feature_matrix_index(frame).reindex(columns=retained_features)
    matrix = matrix.loc[matrix.index.get_level_values("ticker").isin([ticker.upper() for ticker in tickers])].copy()
    if matrix.empty:
        return matrix
    unique_dates = sorted(matrix.index.get_level_values("trade_date").unique())
    selected_dates = unique_dates[-max_dates:]
    matrix = matrix.loc[matrix.index.get_level_values("trade_date").isin(selected_dates)].copy()
    return matrix.sort_index()


def score_live_cross_section(
    *,
    models: dict[str, Any],
    current_feature_matrix: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    raw_predictions: dict[str, pd.Series] = {}
    normalized_predictions: dict[str, pd.Series] = {}
    for model_name, model in models.items():
        current_X = current_feature_matrix.reindex(columns=list(model.feature_names_))
        predictions = model.predict(current_X).rename(model_name)
        raw_predictions[model_name] = predictions
        normalized_predictions[model_name] = cross_sectional_zscore(predictions).rename(model_name)
    return raw_predictions, normalized_predictions


def resolve_current_regime(
    *,
    live_trade_date: date,
    as_of: datetime,
    regime_weights: dict[str, float],
    low_threshold: float,
    high_threshold: float,
) -> tuple[float | None, str, float]:
    vix_history = load_macro_series(
        series_id="VIXCLS",
        start_date=live_trade_date - timedelta(days=120),
        end_date=live_trade_date,
        as_of=as_of.date(),
    )
    current_vix = None
    if not vix_history.empty:
        current_vix = float(vix_history.loc[vix_history.index <= live_trade_date].iloc[-1])
    regime_name = classify_vix(
        float(current_vix) if current_vix is not None else float("nan"),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    regime_scalar = float(regime_weights.get(regime_name, regime_weights.get("unknown", 1.0)))
    return current_vix, regime_name, regime_scalar


def apply_regime_to_model_weights(
    model_weights: dict[str, float],
    regime_scalar: float,
) -> dict[str, float]:
    """Adjust per-model fusion weights based on regime.

    Instead of multiplying the final fusion score by a scalar (which does NOT
    change cross-sectional rankings), blend model weights toward equal weights.
    Higher regime stress (lower scalar) means less trust in IC-based model
    selection — blend more toward 1/N equal model weighting.

    blend_factor = 1.0 - regime_scalar:
      - regime_scalar=1.0 (low VIX): blend=0.0, use IC weights as-is
      - regime_scalar=0.8 (mid/high VIX): blend=0.2, 80% IC + 20% equal weights

    This changes cross-sectional rankings because different models produce
    different stock orderings.
    """
    n_models = len(model_weights)
    if n_models == 0:
        return model_weights
    blend_factor = max(0.0, min(1.0, 1.0 - float(regime_scalar)))
    equal_weight = 1.0 / n_models
    adjusted = {
        name: (1.0 - blend_factor) * w + blend_factor * equal_weight
        for name, w in model_weights.items()
    }
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {name: w / total for name, w in adjusted.items()}
    return adjusted


def build_score_weighted_portfolio(
    *,
    scores: pd.Series,
    previous_weights: dict[str, float],
    turnover_cfg: dict[str, Any],
    selection_pct: float,
) -> dict[str, float]:
    """Build score-weighted portfolio with Phase B turnover controls.

    Applies in order: candidate selection with hysteresis → score weighting →
    weight shrinkage → no-trade zone → turnover buffer.
    """
    sel_pct = float(turnover_cfg.get("selection_pct", selection_pct))
    sell_buffer_pct = float(turnover_cfg.get("sell_buffer_pct", 0.40))
    weight_shrinkage = float(turnover_cfg.get("weight_shrinkage", 0.0))
    no_trade_zone = float(turnover_cfg.get("no_trade_zone", 0.0))
    min_trade_weight = float(turnover_cfg.get("min_trade_weight", 0.005))
    max_weight = float(turnover_cfg.get("max_weight", 0.05))
    min_holdings = int(turnover_cfg.get("min_holdings", 20))

    ranked = scores.dropna().astype(float).sort_values(ascending=False)
    if ranked.empty:
        return {}

    ranking = ranked.index.astype(str).tolist()
    constraints = PortfolioConstraints(
        max_weight=max_weight,
        min_holdings=min_holdings,
        turnover_buffer=min_trade_weight,
    )

    # Step 1: Candidate selection with hysteresis
    candidate_tickers = select_candidate_tickers(
        ranking=ranking,
        current_weights=previous_weights,
        selection_pct=sel_pct,
        sell_buffer_pct=sell_buffer_pct,
        min_holdings=min_holdings,
        max_weight=max_weight,
    )
    candidate_scores = ranked.reindex(candidate_tickers).dropna()
    if candidate_scores.empty:
        return {}

    # Step 2: Score-weighted targets
    pos_scores = candidate_scores[candidate_scores > 0.0]
    if pos_scores.empty:
        return {}
    raw = pos_scores / pos_scores.sum()
    raw = raw.clip(upper=max_weight)
    total = float(raw.sum())
    if total <= 0.0:
        return {}
    raw = raw / total
    target_weights = {str(t): float(w) for t, w in raw.items() if w > 0.0}

    # Step 3: Weight shrinkage — blend toward previous
    if weight_shrinkage > 0.0 and previous_weights:
        all_tickers = set(target_weights) | set(previous_weights)
        blended: dict[str, float] = {}
        for ticker in all_tickers:
            t = target_weights.get(ticker, 0.0)
            p = previous_weights.get(ticker, 0.0)
            w = (1.0 - weight_shrinkage) * t + weight_shrinkage * p
            if w > 1e-8:
                blended[ticker] = w
        bl_total = sum(blended.values())
        if bl_total > 0:
            target_weights = {t: w / bl_total for t, w in blended.items()}

    # Step 4: No-trade zone — keep previous weight if change is tiny
    if no_trade_zone > 0.0 and previous_weights:
        result: dict[str, float] = {}
        all_tickers_ntz = set(target_weights) | set(previous_weights)
        for ticker in all_tickers_ntz:
            t = target_weights.get(ticker, 0.0)
            p = previous_weights.get(ticker, 0.0)
            if abs(t - p) < no_trade_zone and p > 0:
                result[ticker] = p
            elif t > 0:
                result[ticker] = t
        ntz_total = sum(result.values())
        if ntz_total > 0:
            target_weights = {t: w / ntz_total for t, w in result.items() if w > 0}

    # Step 5: Turnover buffer (min_trade_weight)
    if min_trade_weight > 0.0:
        buffer_ref = {t: w for t, w in previous_weights.items() if t in set(ranking)}
        target_weights = apply_turnover_buffer(
            target_weights,
            current_weights=buffer_ref,
            min_trade_weight=min_trade_weight,
            ranking=ranking,
            constraints=constraints,
        )
    else:
        target_weights = apply_weight_constraints(
            target_weights,
            ranking=ranking,
            constraints=constraints,
        )

    return target_weights


def combine_current_predictions(
    normalized_predictions: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    weighted = None
    for model_name in MODEL_NAMES:
        component = normalized_predictions[model_name] * float(weights[model_name])
        weighted = component if weighted is None else weighted.add(component, fill_value=0.0)
    if weighted is None:
        raise RuntimeError("No normalized model predictions were available for fusion.")
    return weighted.rename(FUSION_NAME)


def load_greyscale_reports(report_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("week_*.json")):
        payload = json.loads(path.read_text())
        payload["_report_path"] = str(path.resolve())
        reports.append(payload)
    deduped: dict[str, dict[str, Any]] = {}
    for report in reports:
        signal_date = str(report.get("live_outputs", {}).get("signal_date"))
        generated_at = str(report.get("generated_at_utc", ""))
        existing = deduped.get(signal_date)
        if existing is None or generated_at > str(existing.get("generated_at_utc", "")):
            deduped[signal_date] = report
    return sorted(
        [report for key, report in deduped.items() if key and key != "None"],
        key=lambda report: str(report.get("live_outputs", {}).get("signal_date")),
    )


def build_realized_label_series_from_reports(
    *,
    reports: list[dict[str, Any]],
    latest_pit_trade_date: date,
    as_of: datetime,
    horizon: int,
    benchmark_ticker: str,
) -> pd.Series:
    if not reports:
        return pd.Series(dtype=float)
    signal_dates = sorted(
        {
            date.fromisoformat(str(report.get("live_outputs", {}).get("signal_date")))
            for report in reports
            if report.get("live_outputs", {}).get("signal_date")
        },
    )
    if not signal_dates:
        return pd.Series(dtype=float)
    tickers = sorted(
        {
            str(ticker).upper()
            for report in reports
            for ticker in report.get("score_vectors", {}).get(FUSION_NAME, {}).keys()
        },
    )
    if not tickers:
        return pd.Series(dtype=float)

    prices = get_prices_pit(
        tickers=[*tickers, benchmark_ticker.upper()],
        start_date=min(signal_dates),
        end_date=latest_pit_trade_date,
        as_of=as_of,
    )
    if prices.empty:
        return pd.Series(dtype=float)

    labels = compute_forward_returns(
        prices_df=prices.copy(),
        horizons=(horizon,),
        benchmark_ticker=benchmark_ticker,
    )
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    filtered = labels.loc[
        (labels["horizon"] == horizon)
        & (labels["ticker"] != benchmark_ticker.upper())
        & (labels["trade_date"].dt.date.isin(signal_dates))
    ].copy()
    return filtered.set_index(["trade_date", "ticker"])["excess_return"].sort_index().dropna()


def build_realized_ic_frame(
    *,
    reports: list[dict[str, Any]],
    realized_labels: pd.Series,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in reports:
        realized_ics = compute_report_realized_ics(
            report=report,
            realized_labels=realized_labels,
            model_names=model_names,
        )
        signal_date = report.get("live_outputs", {}).get("signal_date")
        if not signal_date or not realized_ics:
            continue
        rows.append(
            {
                "trade_date": pd.Timestamp(signal_date),
                **{name: float(realized_ics[name]) for name in realized_ics},
            },
        )
    if not rows:
        frame = pd.DataFrame(columns=[*model_names, FUSION_NAME], dtype=float)
        frame.index.name = "trade_date"
        return frame
    return pd.DataFrame(rows).set_index("trade_date").sort_index()


def compute_report_realized_ics(
    *,
    report: dict[str, Any],
    realized_labels: pd.Series,
    model_names: list[str],
) -> dict[str, float]:
    signal_date_raw = report.get("live_outputs", {}).get("signal_date")
    if not signal_date_raw:
        return {}
    signal_ts = pd.Timestamp(signal_date_raw)
    if realized_labels.empty or signal_ts not in realized_labels.index.get_level_values("trade_date"):
        return {}

    date_labels = realized_labels.xs(signal_ts, level="trade_date")
    results: dict[str, float] = {}
    score_vectors = report.get("score_vectors", {})
    for name in [*model_names, FUSION_NAME]:
        score_map = score_vectors.get(name, {})
        if not score_map:
            continue
        scores = pd.Series(score_map, dtype=float)
        scores.index = scores.index.astype(str).str.upper()
        aligned = pd.concat(
            [date_labels.rename("y_true"), scores.rename("y_pred")],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            continue
        results[name] = float(aligned["y_true"].corr(aligned["y_pred"], method="pearson"))
    return results


def resolve_live_weights(
    *,
    realized_ic_frame: pd.DataFrame,
    seed_weights: dict[str, float],
    model_names: list[str],
    rolling_window: int,
    temperature: float,
) -> tuple[dict[str, float], str, pd.DataFrame]:
    history = realized_ic_frame.reindex(columns=model_names).dropna(how="any").sort_index()

    if len(history) >= rolling_window:
        scores = history.tail(rolling_window).mean().to_numpy(dtype=float)
        weights = softmax(scores * float(temperature))
        return (
            {name: float(weight) for name, weight in zip(model_names, weights, strict=True)},
            "rolling_live_ic",
            history,
        )
    return seed_weights, "seed_w11_avg_weights", history


def extract_previous_target_weights(reports: list[dict[str, Any]]) -> dict[str, float]:
    if not reports:
        return {}
    latest = reports[-1]
    weights = latest.get("live_outputs", {}).get("target_weights_after_risk", {})
    return {str(ticker): float(weight) for ticker, weight in weights.items()}


def build_signal_risk_state(
    *,
    signal_monitor: SignalRiskMonitor,
    realized_ic_frame: pd.DataFrame,
    required_points: int,
    model_names: list[str],
) -> dict[str, Any]:
    fusion_history = (
        realized_ic_frame[FUSION_NAME].dropna().astype(float).tolist()
        if FUSION_NAME in realized_ic_frame.columns
        else []
    )
    challenger_ic = 0.0
    if not realized_ic_frame.empty:
        challenger_candidates = [
            float(realized_ic_frame[name].dropna().mean())
            for name in model_names
            if name in realized_ic_frame.columns and not realized_ic_frame[name].dropna().empty
        ]
        if challenger_candidates:
            challenger_ic = float(max(challenger_candidates))

    if len(fusion_history) < required_points:
        return {
            "pass": True,
            "severity": "green",
            "report": {
                "mode": "cold_start_seed_weights",
                "message": (
                    "Insufficient matured live history for rolling signal-risk checks; "
                    "the greyscale run stays on seeded fusion weights."
                ),
                "matured_history_points": int(len(fusion_history)),
                "required_points": int(required_points),
            },
            "history_dates": int(len(fusion_history)),
            "history_rows": 0,
            "fusion_ic": None if not fusion_history else float(np.mean(fusion_history)),
            "challenger_ic": challenger_ic,
            "recommend_switch": False,
        }

    rolling_alert = signal_monitor.check_rolling_ic(
        ic_history=fusion_history,
        lookback=required_points,
        threshold_ratio=0.5,
        consecutive_limit=4,
    )
    return {
        "pass": bool(not rolling_alert.switch_model),
        "severity": rolling_alert.severity.value,
        "report": {
            "mode": "matured_live_history",
            "rolling_ic_alert": rolling_alert.to_dict(),
        },
        "history_dates": int(len(fusion_history)),
        "history_rows": 0,
        "fusion_ic": float(np.mean(fusion_history)),
        "challenger_ic": challenger_ic,
        "recommend_switch": bool(rolling_alert.switch_model),
    }


def compute_pairwise_rank_correlations(model_scores_by_ticker: dict[str, pd.Series]) -> dict[str, float]:
    correlations: dict[str, float] = {}
    for left_idx, left_name in enumerate(MODEL_NAMES):
        for right_name in MODEL_NAMES[left_idx + 1 :]:
            aligned = pd.concat(
                [
                    model_scores_by_ticker[left_name].rename("left"),
                    model_scores_by_ticker[right_name].rename("right"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            if len(aligned) < 2:
                correlations[f"{left_name}__{right_name}"] = float("nan")
                continue
            correlations[f"{left_name}__{right_name}"] = float(
                spearmanr(aligned["left"], aligned["right"], nan_policy="omit").statistic,
            )
    return correlations


def _patch_shap_xgb_loader() -> None:
    """Monkey-patch SHAP's XGBTreeModelLoader to handle XGBoost 3.x base_score format.

    XGBoost >= 3.0 serialises ``base_score`` as ``"[-1.23E-3]"`` (bracket-
    wrapped string).  SHAP <= 0.49 does ``float(base_score)`` which raises
    ``ValueError``.  This one-time patch strips the brackets before conversion.
    """
    try:
        from shap.explainers._tree import XGBTreeModelLoader
    except ImportError:
        return

    if getattr(XGBTreeModelLoader, "_patched_for_xgb3", False):
        return

    _orig_init = XGBTreeModelLoader.__init__

    def _patched_init(self, xgb_model):  # type: ignore[no-untyped-def]
        try:
            _orig_init(self, xgb_model)
        except ValueError as exc:
            if "could not convert string to float" not in str(exc):
                raise
            # Re-run with patched float() that strips brackets
            import builtins
            _orig_float = builtins.float

            def _tolerant_float(v):  # type: ignore[no-untyped-def]
                if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                    return _orig_float(v.strip("[]"))
                return _orig_float(v)

            builtins.float = _tolerant_float  # type: ignore[assignment]
            try:
                _orig_init(self, xgb_model)
            finally:
                builtins.float = _orig_float  # type: ignore[assignment]

    XGBTreeModelLoader.__init__ = _patched_init
    XGBTreeModelLoader._patched_for_xgb3 = True  # type: ignore[attr-defined]


_patch_shap_xgb_loader()


def compute_shap_for_top_tickers(
    models: dict[str, Any],
    feature_matrix: pd.DataFrame,
    top_tickers: list[str],
    max_tickers: int = 100,
) -> dict[str, dict[str, Any]]:
    """Compute SHAP values for tree models on the top-ranked live tickers."""

    if feature_matrix.empty:
        return {}

    available_tickers = set(feature_matrix.index.get_level_values("ticker").astype(str))
    tickers_to_explain = list(dict.fromkeys(str(ticker).upper() for ticker in top_tickers if str(ticker)))[:max_tickers]
    shap_payload: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        if model_name == "ridge":
            continue

        estimator = getattr(model, "estimator_", None) or model
        feature_names = list(getattr(model, "feature_names_", []) or list(feature_matrix.columns))
        try:
            explainer = shap.TreeExplainer(estimator)
        except Exception as exc:
            logger.warning("failed to initialize TreeExplainer for {}: {}", model_name, exc)
            continue

        ticker_payload: dict[str, Any] = {}
        for ticker in tickers_to_explain:
            if ticker not in available_tickers:
                continue

            row = feature_matrix.xs(ticker, level="ticker").tail(1)
            if row.empty:
                continue

            ordered_row = row.reindex(columns=feature_names)
            try:
                shap_values, base_value = _compute_row_shap(explainer, ordered_row)
            except Exception as exc:
                logger.warning("failed to compute SHAP for {} {}: {}", model_name, ticker, exc)
                continue

            if len(shap_values) != len(feature_names):
                logger.warning(
                    "skipping SHAP payload for {} {} because feature lengths mismatch ({} != {})",
                    model_name,
                    ticker,
                    len(shap_values),
                    len(feature_names),
                )
                continue

            ticker_payload[ticker] = {
                "base_value": round(base_value, 6),
                "features": {
                    feature: round(float(value), 6)
                    for feature, value in zip(feature_names, shap_values, strict=True)
                },
            }

        if ticker_payload:
            shap_payload[model_name] = ticker_payload

    return shap_payload


def _compute_row_shap(explainer: shap.TreeExplainer, row: pd.DataFrame) -> tuple[np.ndarray, float]:
    try:
        explanation = explainer(row)
        values = getattr(explanation, "values", explanation)
        base_values = getattr(explanation, "base_values", getattr(explainer, "expected_value", 0.0))
    except Exception:
        values = explainer.shap_values(row)
        base_values = getattr(explainer, "expected_value", 0.0)

    return _flatten_shap_values(values), _flatten_base_value(base_values)


def _flatten_shap_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    while array.ndim > 1:
        array = array[0]
    return array.astype(float)


def _flatten_base_value(base_values: Any) -> float:
    array = np.asarray(base_values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


def series_to_ranked_records(series: pd.Series) -> list[dict[str, float]]:
    return [
        {"ticker": str(ticker), "score": float(score)}
        for ticker, score in series.items()
    ]


def series_to_float_dict(series: pd.Series) -> dict[str, float]:
    return {str(index): float(value) for index, value in series.items()}


def weight_dict_to_records(weights: dict[str, float], *, limit: int) -> list[dict[str, float]]:
    ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    return [{"ticker": str(ticker), "weight": float(weight)} for ticker, weight in ordered[:limit]]


def normalize_weight_dict_local(weights: dict[str, Any], *, fill_unknown: bool = False) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in weights.items()}
    if fill_unknown and "unknown" not in normalized:
        normalized["unknown"] = 1.0
    total = sum(value for key, value in normalized.items() if key in MODEL_NAMES)
    if total > 0.0:
        return {
            key: (value / total if key in MODEL_NAMES else value)
            for key, value in normalized.items()
        }
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
