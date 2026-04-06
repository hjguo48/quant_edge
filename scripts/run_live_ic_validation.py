from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import os
from pathlib import Path
import json
import pickle
import sys
from typing import Any
from urllib.parse import urlparse

from loguru import logger
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.run_ic_screening import install_runtime_optimizations, write_json_atomic
from scripts.run_single_window_validation import (
    align_panel,
    configure_logging,
    current_git_branch,
    fill_feature_matrix,
    load_retained_features,
    long_to_feature_matrix,
)
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_engine
from src.features.pipeline import FeaturePipeline
from src.labels.forward_returns import compute_forward_returns
from src.models.champion_challenger import ChampionChallengerRunner
from src.models.evaluation import (
    icir,
    information_coefficient,
    information_coefficient_series,
    rank_information_coefficient,
    rank_information_coefficient_series,
)
from src.models.registry import ModelRegistry, RegisteredModelVersion

MODEL_NAME = "ridge_60d"
EVAL_START = date(2024, 1, 1)
EVAL_END = date(2025, 12, 31)
AS_OF = date(2026, 4, 1)
REBALANCE_WEEKDAY = 4
HORIZON_DAYS = 60
MIN_CROSS_SECTION_SIZE = 10
DEFAULT_IC_REPORT_PATH = REPO_ROOT / "data/features/ic_screening_report_v2.csv"
DEFAULT_WALKFORWARD_REPORT_PATH = REPO_ROOT / "data/reports/walkforward_backtest.json"
DEFAULT_EXTENDED_WALKFORWARD_REPORT_PATH = REPO_ROOT / "data/reports/extended_walkforward.json"
DEFAULT_SHADOW_MODE_REPORT_PATH = REPO_ROOT / "data/reports/shadow_mode_dry_run.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data/reports/live_ic_consistency.json"
BACKTEST_AGGREGATE_IC = 0.06479914614243343
BENCHMARK_TICKER = "SPY"


@dataclass(frozen=True)
class ReferenceBaseline:
    name: str
    ic: float
    source: str
    report_path: str | None
    report_exists: bool
    generated_at_utc: str | None
    window_count: int
    coverage_start: str | None
    coverage_end: str | None
    overlap_days: int
    overlaps_live_window: bool
    fully_covers_live_window: bool


def _resolve_local_tracking_uri(tracking_uri: str) -> str:
    parsed = urlparse(tracking_uri)
    if parsed.scheme != "file":
        return tracking_uri

    original_path = Path(parsed.path)
    if original_path.exists():
        return tracking_uri

    repo_parts = REPO_ROOT.parts
    original_parts = original_path.parts
    candidates: list[Path] = []
    if REPO_ROOT.name in original_parts:
        repo_index = original_parts.index(REPO_ROOT.name)
        suffix = original_parts[repo_index + 1 :]
        candidates.append(REPO_ROOT.joinpath(*suffix))
    candidates.append(REPO_ROOT / original_path.name)

    for candidate in candidates:
        if candidate.exists():
            logger.info("rewrote tracking URI {} -> {}", tracking_uri, candidate.as_uri())
            return candidate.as_uri()
    return tracking_uri


def _resolve_file_store_artifact_path(
    *,
    tracking_uri: str,
    run_id: str,
    artifact_path: str,
) -> Path | None:
    parsed = urlparse(tracking_uri)
    if parsed.scheme != "file":
        return None

    store_root = Path(parsed.path)
    if not store_root.exists():
        return None

    pattern = f"*/{run_id}/artifacts/{artifact_path}"
    for candidate in store_root.glob(pattern):
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    configure_logging()

    try:
        report, exit_code = run_validation()
    except Exception as exc:
        logger.opt(exception=exc).error("live IC validation failed")
        error_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_branch": safe_git_branch(),
            "script": Path(__file__).name,
            "status": "error",
            "message": str(exc),
        }
        write_json_atomic(DEFAULT_OUTPUT_PATH, error_report)
        return 2

    write_json_atomic(DEFAULT_OUTPUT_PATH, report)
    log_summary(report)
    return exit_code


def run_validation() -> tuple[dict[str, Any], int]:
    registry = ModelRegistry()
    champion = registry.get_champion(MODEL_NAME)
    if champion is None:
        logger.error("no champion model registered for {}", MODEL_NAME)
        error_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_branch": safe_git_branch(),
            "script": Path(__file__).name,
            "status": "error",
            "message": f"no champion model registered for {MODEL_NAME}",
        }
        write_json_atomic(DEFAULT_OUTPUT_PATH, error_report)
        return error_report, 2

    if champion.metadata is None:
        raise RuntimeError(f"Champion model {MODEL_NAME!r} is missing metadata.")

    retained_features = load_retained_features(DEFAULT_IC_REPORT_PATH)
    model_features = list(champion.metadata.features)
    if not model_features:
        raise RuntimeError("Champion metadata.features is empty.")

    model_object, model_load_audit = load_champion_model(registry=registry, champion=champion)
    live_observation = load_live_ic_observation(
        model_object=model_object,
        retained_features=retained_features,
        model_features=model_features,
    )
    live_ic = float(live_observation["live_ic"])
    live_rank_ic = float(live_observation["live_rank_ic"])
    live_icir = float(live_observation["live_icir"])
    ic_series = live_observation["ic_series"]
    rank_ic_series = live_observation["rank_ic_series"]
    cross_section_sizes = live_observation["cross_section_sizes"]
    feature_diagnostics = live_observation["feature_diagnostics"]
    label_diagnostics = live_observation["label_diagnostics"]
    sample_size = live_observation["sample_size"]

    walkforward_reference = load_reference_baseline(
        name="walkforward_backtest",
        path=DEFAULT_WALKFORWARD_REPORT_PATH,
        fallback_ic=BACKTEST_AGGREGATE_IC,
    )
    extended_reference = load_reference_baseline(
        name="extended_walkforward",
        path=DEFAULT_EXTENDED_WALKFORWARD_REPORT_PATH,
        fallback_ic=None,
    )
    champion_recorded_ic = float(champion.metadata.metrics.get("mean_oos_ic", np.nan))
    champion_reference = build_champion_metadata_reference(
        champion_recorded_ic=champion_recorded_ic,
        extended_reference=extended_reference,
    )
    primary_reference = select_primary_reference(
        walkforward_reference,
        extended_reference,
        champion_reference,
    )
    primary_reference_reason = describe_primary_reference_choice(
        primary_reference=primary_reference,
        walkforward_reference=walkforward_reference,
        extended_reference=extended_reference,
    )
    walkforward_error_pct = compute_error_pct(live_ic, walkforward_reference.ic)
    extended_error_pct = compute_error_pct(live_ic, extended_reference.ic)
    champion_error_pct = compute_error_pct(live_ic, champion_reference.ic)
    primary_error_pct = compute_error_pct(live_ic, primary_reference.ic)
    passed = bool(pd.notna(primary_error_pct) and primary_error_pct < 0.20)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": safe_git_branch(),
        "script": Path(__file__).name,
        "status": "pass" if passed else "fail",
        "comparison_basis": primary_reference.name,
        "pit_mode": "batch_scoring_as_of",
        "live_metrics_source": live_observation["source"],
        "model_source": {
            "registry_model_name": MODEL_NAME,
            "registry_tracking_uri": registry.tracking_uri,
            "stage": champion.stage.value,
            "version": int(champion.version),
            "run_id": champion.run_id,
            "model_uri": registry.model_uri(model_name=MODEL_NAME, alias="champion"),
            "load_path": model_load_audit,
        },
        "evaluation_window": {
            "start": EVAL_START.isoformat(),
            "end": EVAL_END.isoformat(),
            "as_of": AS_OF.isoformat(),
            "horizon_days": HORIZON_DAYS,
            "rebalance_weekday": REBALANCE_WEEKDAY,
            "benchmark_ticker": BENCHMARK_TICKER,
        },
        "preprocessing": {
            "feature_fill_method": "cross_sectional_median_then_0.5",
            "friday_only": True,
            "min_cross_section_size": MIN_CROSS_SECTION_SIZE,
            "feature_source": "FeaturePipeline.run",
        },
        "references": {
            "backtest_aggregate_ic": float(walkforward_reference.ic),
            "champion_recorded_mean_oos_ic": champion_recorded_ic,
            "extended_walkforward_aggregate_ic": (
                float(extended_reference.ic) if pd.notna(extended_reference.ic) else None
            ),
            "reference_details": {
                "walkforward_backtest": asdict(walkforward_reference),
                "extended_walkforward": asdict(extended_reference),
                "champion_metadata": asdict(champion_reference),
            },
        },
        "metrics": {
            "live_ic": live_ic,
            "live_rank_ic": optional_float(live_rank_ic),
            "live_icir": optional_float(live_icir),
        },
        "comparison": {
            "primary_reference": primary_reference.name,
            "primary_reference_ic": float(primary_reference.ic),
            "primary_error_pct": primary_error_pct,
            "primary_reference_reason": primary_reference_reason,
            "walkforward_backtest_reference_ic": float(walkforward_reference.ic),
            "walkforward_backtest_error_pct": walkforward_error_pct,
            "extended_walkforward_reference_ic": (
                float(extended_reference.ic) if pd.notna(extended_reference.ic) else None
            ),
            "extended_walkforward_error_pct": extended_error_pct,
            "champion_metadata_reference_ic": champion_recorded_ic,
            "champion_metadata_error_pct": champion_error_pct,
            "threshold_pct": 0.20,
            "passed": passed,
            "reference_window_mismatch": {
                "walkforward_backtest": not walkforward_reference.overlaps_live_window,
                "extended_walkforward_partial_coverage": (
                    extended_reference.overlaps_live_window and not extended_reference.fully_covers_live_window
                ),
            },
        },
        "analysis": {
            "live_window": {
                "start": EVAL_START.isoformat(),
                "end": EVAL_END.isoformat(),
                "days": int((EVAL_END - EVAL_START).days + 1),
            },
            "reference_alignment": {
                "walkforward_backtest": {
                    "coverage_start": walkforward_reference.coverage_start,
                    "coverage_end": walkforward_reference.coverage_end,
                    "overlap_days": walkforward_reference.overlap_days,
                    "fully_covers_live_window": walkforward_reference.fully_covers_live_window,
                },
                "extended_walkforward": {
                    "coverage_start": extended_reference.coverage_start,
                    "coverage_end": extended_reference.coverage_end,
                    "overlap_days": extended_reference.overlap_days,
                    "fully_covers_live_window": extended_reference.fully_covers_live_window,
                },
            },
            "observed_live_metrics_source": live_observation["source_details"],
            "summary": primary_reference_reason,
        },
        "feature_audit": {
            "retained_feature_count": int(len(retained_features)),
            "metadata_feature_count": int(len(model_features)),
            "metadata_not_in_retained": sorted(set(model_features) - set(retained_features)),
            "retained_not_in_metadata": sorted(set(retained_features) - set(model_features)),
            "feature_diagnostics": feature_diagnostics,
        },
        "label_audit": label_diagnostics,
        "sample_size": sample_size,
        "champion_metadata": asdict(champion.metadata),
        "ic_series": series_with_sizes_to_records(ic_series, cross_section_sizes),
        "rank_ic_series": series_to_records(rank_ic_series),
    }
    return report, (0 if passed else 1)


def load_champion_model(
    *,
    registry: ModelRegistry,
    champion: RegisteredModelVersion,
) -> tuple[Any, dict[str, Any]]:
    model_uri = registry.model_uri(model_name=MODEL_NAME, alias="champion")
    load_exception: Exception | None = None
    try:
        return mlflow.pyfunc.load_model(model_uri), {
            "method": "registry_pyfunc",
            "model_uri": model_uri,
        }
    except Exception as exc:
        load_exception = exc
        logger.warning("registry pyfunc load failed for {}: {}", model_uri, exc)

    registry_client = MlflowClient(tracking_uri=registry.tracking_uri)
    packaging_run = registry_client.get_run(champion.run_id)
    source_run_id = packaging_run.data.tags.get("source_run_id")
    source_tracking_uri = packaging_run.data.tags.get("source_tracking_uri")
    if not source_run_id or not source_tracking_uri:
        raise RuntimeError(
            "Champion registry artifact is not loadable and source_run_id/source_tracking_uri tags are missing.",
        ) from load_exception

    resolved_source_tracking_uri = _resolve_local_tracking_uri(source_tracking_uri)
    source_client = MlflowClient(tracking_uri=resolved_source_tracking_uri)
    local_model_path, artifact_resolution = resolve_source_model_pickle_path(
        source_client=source_client,
        source_run_id=source_run_id,
        source_tracking_uri=resolved_source_tracking_uri,
    )
    with open(local_model_path, "rb") as handle:
        model = pickle.load(handle)
    return model, {
        "method": "source_run_pickle_fallback",
        "registry_pyfunc_error": str(load_exception) if load_exception is not None else None,
        "source_run_id": source_run_id,
        "source_tracking_uri": source_tracking_uri,
        "resolved_source_tracking_uri": resolved_source_tracking_uri,
        "artifact_resolution": artifact_resolution,
        "artifact_path": "model/model.pkl",
        "local_model_path": local_model_path,
    }


def build_feature_matrix(
    *,
    tickers: list[str],
    retained_features: list[str],
    model_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    install_runtime_optimizations()
    pipeline = FeaturePipeline()
    features_long = pipeline.run(
        tickers=tickers,
        start_date=EVAL_START - timedelta(days=400),
        end_date=EVAL_END,
        as_of=AS_OF,
    )
    if features_long.empty:
        raise RuntimeError("Feature pipeline returned no rows.")

    filtered_long = features_long.loc[
        features_long["feature_name"].astype(str).isin(retained_features),
        ["ticker", "trade_date", "feature_name", "feature_value"],
    ].copy()
    matrix = long_to_feature_matrix(filtered_long, retained_features)
    matrix = matrix.reindex(columns=model_features)
    filled = fill_feature_matrix(matrix)

    trade_dates = pd.to_datetime(pd.Index(filled.index.get_level_values("trade_date")))
    friday_mask = (
        (trade_dates >= pd.Timestamp(EVAL_START))
        & (trade_dates <= pd.Timestamp(EVAL_END))
        & (trade_dates.weekday == REBALANCE_WEEKDAY)
    )
    friday_matrix = filled.loc[friday_mask].sort_index()
    if friday_matrix.empty:
        raise RuntimeError("Feature matrix is empty after Friday filtering.")

    feature_counts = friday_matrix.groupby(level="trade_date").size()
    diagnostics = {
        "rows": int(len(friday_matrix)),
        "dates": int(friday_matrix.index.get_level_values("trade_date").nunique()),
        "tickers": int(friday_matrix.index.get_level_values("ticker").nunique()),
        "min_date": pd.Timestamp(friday_matrix.index.get_level_values("trade_date").min()).date().isoformat(),
        "max_date": pd.Timestamp(friday_matrix.index.get_level_values("trade_date").max()).date().isoformat(),
        "avg_cross_section_size": float(feature_counts.mean()),
        "min_cross_section_size": int(feature_counts.min()),
        "max_cross_section_size": int(feature_counts.max()),
    }
    return friday_matrix, diagnostics


def build_label_series(*, tickers: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    prices = get_prices_pit(
        tickers=[*tickers, BENCHMARK_TICKER],
        start_date=EVAL_START,
        end_date=AS_OF,
        as_of=_as_of_datetime(AS_OF),
    )
    if prices.empty:
        raise RuntimeError("No PIT prices available for label computation.")

    labels = compute_forward_returns(
        prices_df=prices,
        horizons=(HORIZON_DAYS,),
        benchmark_ticker=BENCHMARK_TICKER,
    )
    labels = labels.loc[
        (labels["horizon"] == HORIZON_DAYS)
        & (labels["ticker"].astype(str).str.upper() != BENCHMARK_TICKER)
        & (pd.to_datetime(labels["trade_date"]) >= pd.Timestamp(EVAL_START))
        & (pd.to_datetime(labels["trade_date"]) <= pd.Timestamp(EVAL_END))
    ].copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")

    friday_mask = pd.to_datetime(labels["trade_date"]).dt.weekday == REBALANCE_WEEKDAY
    labels = labels.loc[friday_mask].copy()
    series = (
        labels.set_index(["trade_date", "ticker"])["excess_return"]
        .sort_index()
        .dropna()
    )
    if series.empty:
        raise RuntimeError("Label series is empty after Friday filtering.")

    diagnostics = {
        "rows": int(len(series)),
        "dates": int(series.index.get_level_values("trade_date").nunique()),
        "tickers": int(series.index.get_level_values("ticker").nunique()),
        "min_date": pd.Timestamp(series.index.get_level_values("trade_date").min()).date().isoformat(),
        "max_date": pd.Timestamp(series.index.get_level_values("trade_date").max()).date().isoformat(),
        "horizon_days": HORIZON_DAYS,
        "benchmark_ticker": BENCHMARK_TICKER,
    }
    return series, diagnostics


def load_live_ic_observation(
    *,
    model_object: Any,
    retained_features: list[str],
    model_features: list[str],
) -> dict[str, Any]:
    shadow_snapshot = load_shadow_mode_live_snapshot(DEFAULT_SHADOW_MODE_REPORT_PATH)
    if shadow_snapshot is not None:
        logger.info(
            "using shadow-mode live IC snapshot from {} instead of rebuilding the full historical PIT panel",
            DEFAULT_SHADOW_MODE_REPORT_PATH,
        )
        history_rows = int(shadow_snapshot["history_rows"])
        history_dates = int(shadow_snapshot["history_dates"])
        cross_section_size = int(round(history_rows / history_dates)) if history_dates else 0
        sample_size = {
            "prediction_rows": history_rows,
            "label_rows": history_rows,
            "aligned_rows_before_filter": history_rows,
            "aligned_rows_after_filter": history_rows,
            "friday_dates_before_filter": history_dates,
            "dates_with_ic": history_dates,
            "avg_cross_section_size": float(cross_section_size) if history_dates else float("nan"),
            "min_cross_section_size": cross_section_size,
            "max_cross_section_size": cross_section_size,
        }
        return {
            "source": "shadow_mode_report",
            "source_details": shadow_snapshot,
            "live_ic": shadow_snapshot["live_ic"],
            "live_rank_ic": float("nan"),
            "live_icir": float("nan"),
            "ic_series": pd.Series(dtype=float),
            "rank_ic_series": pd.Series(dtype=float),
            "cross_section_sizes": pd.Series(dtype=int),
            "feature_diagnostics": {
                "source": "shadow_mode_dry_run",
                "report_path": str(DEFAULT_SHADOW_MODE_REPORT_PATH),
                "feature_rows_total": int(shadow_snapshot["feature_rows_total"]),
                "feature_rows_live_date": int(shadow_snapshot["feature_rows_live_date"]),
                "feature_matrix_shape": dict(shadow_snapshot["feature_matrix_shape"]),
                "pit_signal_date": shadow_snapshot["pit_signal_date"],
            },
            "label_diagnostics": {
                "source": "shadow_mode_dry_run",
                "report_path": str(DEFAULT_SHADOW_MODE_REPORT_PATH),
                "history_dates": history_dates,
                "history_rows": history_rows,
                "aligned_prediction_rows": int(shadow_snapshot["aligned_prediction_rows"]),
            },
            "sample_size": sample_size,
        }

    tickers = load_tracked_tickers(exclude_ticker=BENCHMARK_TICKER)
    feature_matrix, feature_diagnostics = build_feature_matrix(
        tickers=tickers,
        retained_features=retained_features,
        model_features=model_features,
    )
    label_series, label_diagnostics = build_label_series(tickers=tickers)

    aligned_X, aligned_y = align_panel(feature_matrix, label_series)
    predictions = ChampionChallengerRunner._predict_series(model_object, aligned_X)

    filtered_y, filtered_pred, cross_section_sizes = filter_minimum_cross_sections(
        y=aligned_y,
        y_pred=predictions,
        min_size=MIN_CROSS_SECTION_SIZE,
    )
    if filtered_y.empty or filtered_pred.empty:
        raise RuntimeError("No aligned Friday observations remain after cross-section filtering.")

    ic_series = information_coefficient_series(y_true=filtered_y, y_pred=filtered_pred)
    rank_ic_series = rank_information_coefficient_series(y_true=filtered_y, y_pred=filtered_pred)
    if ic_series.empty:
        raise RuntimeError("IC series is empty after filtering.")

    sample_size = {
        "prediction_rows": int(len(feature_matrix)),
        "label_rows": int(len(label_series)),
        "aligned_rows_before_filter": int(len(aligned_X)),
        "aligned_rows_after_filter": int(len(filtered_y)),
        "friday_dates_before_filter": int(aligned_X.index.get_level_values("trade_date").nunique()),
        "dates_with_ic": int(len(ic_series)),
        "avg_cross_section_size": float(cross_section_sizes.mean()),
        "min_cross_section_size": int(cross_section_sizes.min()),
        "max_cross_section_size": int(cross_section_sizes.max()),
    }
    return {
        "source": "recomputed_historical_pit_panel",
        "source_details": {
            "method": "FeaturePipeline.run + PIT price labels",
            "evaluation_window_start": EVAL_START.isoformat(),
            "evaluation_window_end": EVAL_END.isoformat(),
            "as_of": AS_OF.isoformat(),
        },
        "live_ic": float(information_coefficient(y_true=filtered_y, y_pred=filtered_pred)),
        "live_rank_ic": float(rank_information_coefficient(y_true=filtered_y, y_pred=filtered_pred)),
        "live_icir": float(icir(y_true=filtered_y, y_pred=filtered_pred)),
        "ic_series": ic_series,
        "rank_ic_series": rank_ic_series,
        "cross_section_sizes": cross_section_sizes,
        "feature_diagnostics": feature_diagnostics,
        "label_diagnostics": label_diagnostics,
        "sample_size": sample_size,
    }


def filter_minimum_cross_sections(
    *,
    y: pd.Series,
    y_pred: pd.Series,
    min_size: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    aligned = pd.concat(
        [y.rename("y_true"), y_pred.rename("y_pred")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise RuntimeError("No aligned observations between predictions and labels.")

    sizes = aligned.groupby(level="trade_date").size()
    valid_dates = sizes.loc[sizes >= min_size].index
    if len(valid_dates) == 0:
        raise RuntimeError(f"No dates meet the minimum cross-section size of {min_size}.")

    filtered = aligned.loc[
        aligned.index.get_level_values("trade_date").isin(valid_dates),
    ].sort_index()
    filtered_sizes = filtered.groupby(level="trade_date").size().astype(int)
    return filtered["y_true"], filtered["y_pred"], filtered_sizes


def load_tracked_tickers(*, exclude_ticker: str) -> list[str]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT ticker
                FROM stocks
                WHERE ticker IS NOT NULL
                  AND upper(ticker) <> :exclude_ticker
                ORDER BY ticker
                """,
            ),
            {"exclude_ticker": exclude_ticker.upper()},
        ).scalars().all()
    tickers = [str(ticker).upper() for ticker in rows]
    if not tickers:
        raise RuntimeError("No tracked tickers found in stocks table.")
    return tickers


def load_shadow_mode_live_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text())
    model_name = payload.get("weekly_signal_pipeline", {}).get("model", {}).get("model_name")
    if model_name and str(model_name) != MODEL_NAME:
        logger.warning(
            "shadow-mode live IC snapshot model_name={} does not match validation model_name={}",
            model_name,
            MODEL_NAME,
        )
        return None
    signal_risk = payload.get("weekly_signal_pipeline", {}).get("signal_risk", {})
    if "live_ic" not in signal_risk:
        return None

    return {
        "report_path": str(path),
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "pit_signal_date": payload.get("weekly_signal_pipeline", {}).get("pit_signal_date"),
        "feature_rows_total": int(payload.get("weekly_signal_pipeline", {}).get("feature_rows_total", 0)),
        "feature_rows_live_date": int(payload.get("weekly_signal_pipeline", {}).get("feature_rows_live_date", 0)),
        "feature_matrix_shape": dict(payload.get("weekly_signal_pipeline", {}).get("feature_matrix_shape", {})),
        "history_dates": int(signal_risk.get("history_dates", 0)),
        "history_rows": int(signal_risk.get("history_rows", 0)),
        "aligned_prediction_rows": int(signal_risk.get("aligned_prediction_rows", 0)),
        "live_ic": float(signal_risk["live_ic"]),
        "calibration_spearman": float(signal_risk.get("calibration_spearman", np.nan)),
        "calibration_monotonic": bool(signal_risk.get("calibration_monotonic", False)),
        "model_name": model_name,
        "model_version": payload.get("weekly_signal_pipeline", {}).get("model", {}).get("version"),
        "model_run_id": payload.get("weekly_signal_pipeline", {}).get("model", {}).get("run_id"),
    }


def load_reference_baseline(
    *,
    name: str,
    path: Path,
    fallback_ic: float | None,
) -> ReferenceBaseline:
    if not path.exists():
        return ReferenceBaseline(
            name=name,
            ic=float(fallback_ic) if fallback_ic is not None else float("nan"),
            source="fallback_constant" if fallback_ic is not None else "missing_report",
            report_path=str(path),
            report_exists=False,
            generated_at_utc=None,
            window_count=0,
            coverage_start=None,
            coverage_end=None,
            overlap_days=0,
            overlaps_live_window=False,
            fully_covers_live_window=False,
        )

    payload = json.loads(path.read_text())
    aggregate = payload.get("walkforward", {}).get("aggregate", {})
    windows = payload.get("walkforward", {}).get("windows", [])
    coverage_start, coverage_end = extract_test_coverage(windows)
    overlap_days = compute_overlap_days(EVAL_START, EVAL_END, coverage_start, coverage_end)

    default_ic = fallback_ic if fallback_ic is not None else float("nan")
    return ReferenceBaseline(
        name=name,
        ic=float(aggregate.get("mean_test_ic", default_ic)),
        source="walkforward_report",
        report_path=str(path),
        report_exists=True,
        generated_at_utc=payload.get("generated_at_utc"),
        window_count=int(len(windows)),
        coverage_start=coverage_start.isoformat() if coverage_start is not None else None,
        coverage_end=coverage_end.isoformat() if coverage_end is not None else None,
        overlap_days=overlap_days,
        overlaps_live_window=overlap_days > 0,
        fully_covers_live_window=(
            coverage_start is not None
            and coverage_end is not None
            and coverage_start <= EVAL_START
            and coverage_end >= EVAL_END
        ),
    )


def load_backtest_aggregate_ic(path: Path) -> float:
    return load_reference_baseline(
        name="walkforward_backtest",
        path=path,
        fallback_ic=BACKTEST_AGGREGATE_IC,
    ).ic


def build_champion_metadata_reference(
    *,
    champion_recorded_ic: float,
    extended_reference: ReferenceBaseline,
) -> ReferenceBaseline:
    coverage_start = extended_reference.coverage_start
    coverage_end = extended_reference.coverage_end
    overlap_days = extended_reference.overlap_days if extended_reference.report_exists else 0
    overlaps_live_window = extended_reference.overlaps_live_window if extended_reference.report_exists else False
    fully_covers_live_window = (
        extended_reference.fully_covers_live_window if extended_reference.report_exists else False
    )
    source = "registry_metadata"
    report_path: str | None = None
    if (
        extended_reference.report_exists
        and pd.notna(champion_recorded_ic)
        and pd.notna(extended_reference.ic)
        and np.isclose(champion_recorded_ic, extended_reference.ic, atol=1e-12, rtol=1e-9)
    ):
        source = "registry_metadata_matched_extended_walkforward"
        report_path = extended_reference.report_path

    return ReferenceBaseline(
        name="champion_metadata",
        ic=champion_recorded_ic,
        source=source,
        report_path=report_path,
        report_exists=report_path is not None,
        generated_at_utc=extended_reference.generated_at_utc if report_path is not None else None,
        window_count=extended_reference.window_count if report_path is not None else 0,
        coverage_start=coverage_start if report_path is not None else None,
        coverage_end=coverage_end if report_path is not None else None,
        overlap_days=overlap_days,
        overlaps_live_window=overlaps_live_window,
        fully_covers_live_window=fully_covers_live_window,
    )


def select_primary_reference(*candidates: ReferenceBaseline) -> ReferenceBaseline:
    available = [candidate for candidate in candidates if pd.notna(candidate.ic)]
    if not available:
        raise RuntimeError("No valid reference IC was available for comparison.")

    return max(
        available,
        key=lambda candidate: (
            int(candidate.fully_covers_live_window),
            int(candidate.overlaps_live_window),
            candidate.overlap_days,
            int(candidate.source == "walkforward_report"),
            int(candidate.report_exists),
            int(candidate.source != "fallback_constant"),
        ),
    )


def describe_primary_reference_choice(
    *,
    primary_reference: ReferenceBaseline,
    walkforward_reference: ReferenceBaseline,
    extended_reference: ReferenceBaseline,
) -> str:
    live_window_text = f"{EVAL_START.isoformat()} to {EVAL_END.isoformat()}"
    if primary_reference.name == "walkforward_backtest":
        return (
            f"Using walkforward_backtest because its reported test coverage best matches the live evaluation window "
            f"({live_window_text})."
        )
    if primary_reference.name == "extended_walkforward":
        if not walkforward_reference.overlaps_live_window:
            return (
                f"walkforward_backtest coverage ends on {walkforward_reference.coverage_end}, before the live "
                f"evaluation window starts on {EVAL_START.isoformat()}; extended_walkforward overlaps the live "
                f"window through {extended_reference.coverage_end} and is the closest report-backed reference."
            )
        return (
            f"extended_walkforward provides better date overlap with the live evaluation window ({live_window_text}) "
            f"than walkforward_backtest."
        )
    if not walkforward_reference.overlaps_live_window and not extended_reference.overlaps_live_window:
        return (
            f"No report-backed walkforward baseline overlaps the live evaluation window ({live_window_text}); "
            f"falling back to champion metadata."
        )
    return (
        f"Using champion metadata because it is the best available model-specific reference for the live window "
        f"({live_window_text})."
    )


def resolve_source_model_pickle_path(
    *,
    source_client: MlflowClient,
    source_run_id: str,
    source_tracking_uri: str,
) -> tuple[str, str]:
    artifact_path = "model/model.pkl"
    download_exception: Exception | None = None
    try:
        downloaded_path = Path(source_client.download_artifacts(source_run_id, artifact_path))
        if downloaded_path.exists() and downloaded_path.is_file():
            return str(downloaded_path), "mlflow_download_artifacts"
        logger.warning(
            "download_artifacts returned non-file path {} for source run {} artifact {}",
            downloaded_path,
            source_run_id,
            artifact_path,
        )
    except Exception as exc:
        download_exception = exc
        logger.warning(
            "download_artifacts failed for source run {} artifact {}: {}",
            source_run_id,
            artifact_path,
            exc,
        )

    direct_model_path = _resolve_file_store_artifact_path(
        tracking_uri=source_tracking_uri,
        run_id=source_run_id,
        artifact_path=artifact_path,
    )
    if direct_model_path is not None:
        return str(direct_model_path), "direct_file_store_lookup"

    if download_exception is not None:
        raise RuntimeError(
            f"Unable to resolve source-run pickle artifact {artifact_path!r} for run {source_run_id}.",
        ) from download_exception
    raise FileNotFoundError(f"Resolved source-run artifact path is missing: run={source_run_id} path={artifact_path}")


def extract_test_coverage(windows: list[dict[str, Any]]) -> tuple[date | None, date | None]:
    starts: list[date] = []
    ends: list[date] = []
    for window in windows:
        start, end = parse_period_range(window.get("test_period"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    return (min(starts), max(ends)) if starts and ends else (None, None)


def parse_period_range(period: Any) -> tuple[date | None, date | None]:
    if not period or "->" not in str(period):
        return None, None
    start_raw, end_raw = [item.strip() for item in str(period).split("->", maxsplit=1)]
    try:
        return date.fromisoformat(start_raw), date.fromisoformat(end_raw)
    except ValueError:
        return None, None


def compute_overlap_days(
    live_start: date,
    live_end: date,
    reference_start: date | None,
    reference_end: date | None,
) -> int:
    if reference_start is None or reference_end is None:
        return 0
    overlap_start = max(live_start, reference_start)
    overlap_end = min(live_end, reference_end)
    if overlap_end < overlap_start:
        return 0
    return int((overlap_end - overlap_start).days + 1)


def compute_error_pct(value: float, reference: float) -> float:
    if pd.isna(value) or pd.isna(reference) or np.isclose(reference, 0.0):
        return float("nan")
    return float(abs(value - reference) / abs(reference))


def optional_float(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def series_to_records(series: pd.Series) -> list[dict[str, Any]]:
    return [
        {
            "trade_date": pd.Timestamp(index).date().isoformat(),
            "value": float(value),
        }
        for index, value in series.items()
        if pd.notna(value)
    ]


def series_with_sizes_to_records(series: pd.Series, sizes: pd.Series) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, value in series.items():
        if pd.isna(value):
            continue
        records.append(
            {
                "trade_date": pd.Timestamp(index).date().isoformat(),
                "ic": float(value),
                "cross_section_size": int(sizes.get(index, 0)),
            },
        )
    return records


def safe_git_branch() -> str | None:
    try:
        return current_git_branch()
    except Exception:
        return None


def log_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    comparison = report["comparison"]
    sample_size = report["sample_size"]
    logger.info(
        "live IC validation {} | live_ic={:.6f} rank_ic={:.6f} icir={:.6f} error_pct={:.4f} dates={} rows={}",
        report["status"],
        metrics["live_ic"],
        float(metrics["live_rank_ic"]) if metrics["live_rank_ic"] is not None else float("nan"),
        float(metrics["live_icir"]) if metrics["live_icir"] is not None else float("nan"),
        comparison["primary_error_pct"],
        sample_size["dates_with_ic"],
        sample_size["aligned_rows_after_filter"],
    )
    logger.info("saved live IC validation report to {}", DEFAULT_OUTPUT_PATH)


def _as_of_datetime(as_of: date) -> datetime:
    return datetime.combine(as_of, time.max, tzinfo=timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
