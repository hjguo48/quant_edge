from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import gc
import os
from pathlib import Path
import json
import pickle
import resource
import sys
import tempfile
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
import src.features.pipeline as feature_pipeline_module
from src.features.pipeline import FeaturePipeline
from src.features.pipeline import compute_composite_features
import src.features.preprocessing as preprocessing_module
from src.features.technical import compute_technical_features
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

DEFAULT_MODEL_NAME = "ridge_60d"
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
DEFAULT_FUSION_BUNDLE_PATH = REPO_ROOT / "data/models/fusion_model_bundle_60d.json"
DEFAULT_FUSION_REPORT_PATH = REPO_ROOT / "data/reports/fusion_analysis_60d.json"
DEFAULT_FUSION_OUTPUT_PATH = REPO_ROOT / "data/reports/live_ic_validation_fusion.json"
DEFAULT_FEATURE_AUDIT_OUTPUT_PATH = REPO_ROOT / "data/reports/s1_14_feature_audit.json"
DEFAULT_GREYSCALE_REPORT_DIR = REPO_ROOT / "data/reports/greyscale"
BACKTEST_AGGREGATE_IC = 0.06479914614243343
FUSION_BACKTEST_AGGREGATE_IC = 0.09137099618774651
BENCHMARK_TICKER = "SPY"
FUSION_MODEL_REGISTRY_NAMES = {
    "ridge": "ridge_60d",
    "xgboost": "xgboost_60d",
    "lightgbm": "lightgbm_60d",
}
MAX_FUNDAMENTAL_WORKERS = min(8, max(1, (os.cpu_count() or 4) // 2))
FUSION_TICKER_BATCH_SIZE = 100
MEMORY_LIMIT_GB = 6.0
CROSS_SECTIONAL_TECHNICAL_FEATURE_NAMES = (
    "momentum_rank_20d",
    "momentum_rank_60d",
    "vol_rank",
)


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


@dataclass(frozen=True)
class ValidationConfig:
    model_name: str
    output_path: Path
    ic_report_path: Path
    walkforward_report_path: Path
    extended_walkforward_report_path: Path
    shadow_mode_report_path: Path
    fusion: bool = False
    fusion_bundle_path: Path = DEFAULT_FUSION_BUNDLE_PATH
    fusion_report_path: Path = DEFAULT_FUSION_REPORT_PATH
    feature_audit_output_path: Path = DEFAULT_FEATURE_AUDIT_OUTPUT_PATH
    greyscale_report_dir: Path = DEFAULT_GREYSCALE_REPORT_DIR
    fusion_batch_size: int = FUSION_TICKER_BATCH_SIZE


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate live IC against backtest baselines.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Registry model name to validate in single-model mode.",
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Evaluate the deployed fusion bundle used by greyscale live.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional report output path. Defaults to the legacy report in single-model mode or the fusion report in --fusion mode.",
    )
    parser.add_argument("--ic-report-path", type=Path, default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--walkforward-report-path", type=Path, default=DEFAULT_WALKFORWARD_REPORT_PATH)
    parser.add_argument(
        "--extended-walkforward-report-path",
        type=Path,
        default=DEFAULT_EXTENDED_WALKFORWARD_REPORT_PATH,
    )
    parser.add_argument("--shadow-mode-report-path", type=Path, default=DEFAULT_SHADOW_MODE_REPORT_PATH)
    parser.add_argument("--fusion-bundle-path", type=Path, default=DEFAULT_FUSION_BUNDLE_PATH)
    parser.add_argument("--fusion-report-path", type=Path, default=DEFAULT_FUSION_REPORT_PATH)
    parser.add_argument("--greyscale-report-dir", type=Path, default=DEFAULT_GREYSCALE_REPORT_DIR)
    parser.add_argument(
        "--fusion-batch-size",
        type=int,
        default=FUSION_TICKER_BATCH_SIZE,
        help="Number of tickers to process per batch in --fusion mode.",
    )
    parser.add_argument(
        "--feature-audit-output-path",
        type=Path,
        default=DEFAULT_FEATURE_AUDIT_OUTPUT_PATH,
    )
    return parser.parse_args(argv)


def build_validation_config(args: argparse.Namespace) -> ValidationConfig:
    output_path = args.output_path or (DEFAULT_FUSION_OUTPUT_PATH if args.fusion else DEFAULT_OUTPUT_PATH)
    return ValidationConfig(
        model_name=str(args.model_name),
        output_path=output_path,
        ic_report_path=args.ic_report_path,
        walkforward_report_path=args.walkforward_report_path,
        extended_walkforward_report_path=args.extended_walkforward_report_path,
        shadow_mode_report_path=args.shadow_mode_report_path,
        fusion=bool(args.fusion),
        fusion_bundle_path=args.fusion_bundle_path,
        fusion_report_path=args.fusion_report_path,
        feature_audit_output_path=args.feature_audit_output_path,
        greyscale_report_dir=args.greyscale_report_dir,
        fusion_batch_size=int(args.fusion_batch_size),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_validation_config(args)
    configure_logging()

    try:
        report, exit_code = run_validation(config)
    except Exception as exc:
        logger.opt(exception=exc).error("live IC validation failed")
        error_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_branch": safe_git_branch(),
            "script": Path(__file__).name,
            "mode": "fusion" if config.fusion else "single_model",
            "model_name": config.model_name,
            "status": "error",
            "message": str(exc),
        }
        write_json_atomic(config.output_path, error_report)
        return 2

    write_json_atomic(config.output_path, report)
    log_summary(report, output_path=config.output_path)
    return exit_code


def run_validation(config: ValidationConfig) -> tuple[dict[str, Any], int]:
    if config.fusion:
        return run_fusion_validation(config)
    return run_single_model_validation(config)


def run_single_model_validation(config: ValidationConfig) -> tuple[dict[str, Any], int]:
    registry = ModelRegistry()
    champion = registry.get_champion(config.model_name)
    if champion is None:
        logger.error("no champion model registered for {}", config.model_name)
        error_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_branch": safe_git_branch(),
            "script": Path(__file__).name,
            "status": "error",
            "message": f"no champion model registered for {config.model_name}",
        }
        write_json_atomic(config.output_path, error_report)
        return error_report, 2

    if champion.metadata is None:
        raise RuntimeError(f"Champion model {config.model_name!r} is missing metadata.")

    retained_features = load_retained_features(config.ic_report_path)
    model_features = list(champion.metadata.features)
    if not model_features:
        raise RuntimeError("Champion metadata.features is empty.")

    model_object, model_load_audit = load_champion_model(
        registry=registry,
        champion=champion,
        model_name=config.model_name,
    )
    live_observation = load_live_ic_observation(
        model_object=model_object,
        retained_features=retained_features,
        model_features=model_features,
        model_name=config.model_name,
        shadow_mode_report_path=config.shadow_mode_report_path,
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
        path=config.walkforward_report_path,
        fallback_ic=BACKTEST_AGGREGATE_IC,
    )
    extended_reference = load_reference_baseline(
        name="extended_walkforward",
        path=config.extended_walkforward_report_path,
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
            "registry_model_name": config.model_name,
            "registry_tracking_uri": registry.tracking_uri,
            "stage": champion.stage.value,
            "version": int(champion.version),
            "run_id": champion.run_id,
            "model_uri": registry.model_uri(model_name=config.model_name, alias="champion"),
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


def run_fusion_validation(config: ValidationConfig) -> tuple[dict[str, Any], int]:
    bundle = json.loads(config.fusion_bundle_path.read_text())
    retained_features = [str(feature) for feature in bundle.get("retained_features", [])]
    if not retained_features:
        raise RuntimeError(f"Fusion bundle is missing retained_features: {config.fusion_bundle_path}")

    bundle_models = dict(bundle.get("models", {}))
    if not bundle_models:
        raise RuntimeError(f"Fusion bundle is missing model definitions: {config.fusion_bundle_path}")

    legacy_retained_features = load_retained_features(config.ic_report_path)
    feature_audit_report = build_feature_audit_report(
        config=config,
        legacy_retained_features=legacy_retained_features,
        fusion_retained_features=retained_features,
        bundle=bundle,
    )
    write_json_atomic(config.feature_audit_output_path, feature_audit_report)

    universe_tickers, greyscale_report_path = load_latest_greyscale_universe(config.greyscale_report_dir)
    label_series, label_diagnostics = build_label_series(tickers=universe_tickers)
    registry = ModelRegistry()

    component_reports: dict[str, Any] = {}
    component_predictions: dict[str, pd.Series] = {}
    component_observations: dict[str, dict[str, Any]] = {}
    registry_audit: dict[str, Any] = {}
    unavailable_models: list[str] = []
    available_model_specs: dict[str, dict[str, Any]] = {}

    for model_key in sorted(bundle_models):
        registry_model_name = FUSION_MODEL_REGISTRY_NAMES.get(model_key, f"{model_key}_{HORIZON_DAYS}d")
        bundle_model_payload = dict(bundle_models[model_key])
        registry_details, champion = inspect_registry_champion(
            registry=registry,
            registry_model_name=registry_model_name,
            bundle_model_payload=bundle_model_payload,
            bundle_features=retained_features,
        )
        registry_audit[model_key] = registry_details

        model_object, load_audit = load_deployed_bundle_model(
            model_key=model_key,
            bundle_model_payload=bundle_model_payload,
            registry=registry,
            registry_model_name=registry_model_name,
            champion=champion,
        )
        if model_object is None:
            unavailable_models.append(model_key)
            component_reports[model_key] = {
                "status": "unavailable",
                "registry_model_name": registry_model_name,
                "registry_audit": registry_details,
                "load_audit": load_audit,
            }
            continue

        model_features = [
            str(feature)
            for feature in bundle_model_payload.get("feature_names", retained_features)
        ] or list(retained_features)
        available_model_specs[model_key] = {
            "model_object": model_object,
            "feature_names": model_features,
            "registry_model_name": registry_model_name,
            "registry_audit": registry_details,
            "load_audit": load_audit,
            "bundle_model_payload": bundle_model_payload,
        }

    with tempfile.TemporaryDirectory(prefix="live_ic_validation_fusion_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        trade_dates = materialize_fusion_date_partitions(
            tickers=universe_tickers,
            batch_size=config.fusion_batch_size,
            temp_dir=temp_dir,
        )
        component_predictions, feature_diagnostics, sample_size_base, peak_rss_gb = score_fusion_models_batched(
            temp_dir=temp_dir,
            trade_dates=trade_dates,
            tickers=universe_tickers,
            retained_features=retained_features,
            model_specs=available_model_specs,
            label_series=label_series,
            batch_size=config.fusion_batch_size,
        )

    for model_key in sorted(bundle_models):
        if model_key not in available_model_specs:
            continue

        spec = available_model_specs[model_key]
        predictions = component_predictions.get(model_key)
        if predictions is None or predictions.empty:
            unavailable_models.append(model_key)
            component_reports[model_key] = {
                "status": "unavailable",
                "registry_model_name": spec["registry_model_name"],
                "registry_audit": spec["registry_audit"],
                "load_audit": {
                    **spec["load_audit"],
                    "reason": "no batched predictions were generated",
                },
            }
            continue

        try:
            observation = evaluate_prediction_series(
                aligned_y=label_series,
                predictions=predictions,
                sample_size_base=sample_size_base,
            )
        except Exception as exc:
            unavailable_models.append(model_key)
            component_reports[model_key] = {
                "status": "unavailable",
                "registry_model_name": spec["registry_model_name"],
                "registry_audit": spec["registry_audit"],
                "load_audit": {
                    **spec["load_audit"],
                    "prediction_error": str(exc),
                },
            }
            logger.warning("fusion component {} could not be scored: {}", model_key, exc)
            continue

        component_observations[model_key] = observation
        component_reports[model_key] = {
            "status": "ok",
            "registry_model_name": spec["registry_model_name"],
            "bundle_reference_test_ic": optional_float(
                float(spec["bundle_model_payload"].get("reference_test_metrics", {}).get("ic", np.nan)),
            ),
            "registry_audit": spec["registry_audit"],
            "load_audit": spec["load_audit"],
            "metrics": {
                "live_ic": float(observation["live_ic"]),
                "live_rank_ic": optional_float(float(observation["live_rank_ic"])),
                "live_icir": optional_float(float(observation["live_icir"])),
            },
        }

    if not component_observations:
        raise RuntimeError(
            "Unable to score any fusion component model. See registry_audit/load_audit in the report for details.",
        )

    requested_weights = {
        model_key: float(bundle.get("seed_weights", {}).get(model_key, 0.0))
        for model_key in component_observations
    }
    normalized_weights = normalize_weight_dict_local(requested_weights)
    fusion_predictions = combine_prediction_series(
        {model_key: component_predictions[model_key] for model_key in component_observations},
        normalized_weights,
    )
    fusion_observation = evaluate_prediction_series(
        aligned_y=label_series,
        predictions=fusion_predictions,
        sample_size_base=sample_size_base,
    )

    backtest_reference = load_fusion_reference(config.fusion_report_path)
    live_ic = float(fusion_observation["live_ic"])
    live_rank_ic = float(fusion_observation["live_rank_ic"])
    live_icir = float(fusion_observation["live_icir"])
    ic_series = fusion_observation["ic_series"]
    rank_ic_series = fusion_observation["rank_ic_series"]
    cross_section_sizes = fusion_observation["cross_section_sizes"]
    sample_size = fusion_observation["sample_size"]
    live_error_pct = compute_error_pct(live_ic, backtest_reference["backtest_ic"])
    live_decay_pct = compute_decay_pct(live_ic, backtest_reference["backtest_ic"])
    memory_limit_passed = bool(peak_rss_gb < MEMORY_LIMIT_GB)
    passed = bool(pd.notna(live_error_pct) and live_error_pct < 0.20 and memory_limit_passed)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": safe_git_branch(),
        "script": Path(__file__).name,
        "mode": "fusion",
        "status": "pass" if passed else "fail",
        "comparison_basis": backtest_reference["name"],
        "pit_mode": "batch_scoring_as_of",
        "live_metrics_source": "recomputed_historical_pit_panel",
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
            "feature_source": "FeaturePipeline.run (ticker-batched, date-recombined)",
            "retained_feature_source": "fusion_model_bundle_60d.json",
        },
        "greyscale_bundle": {
            "bundle_path": str(config.fusion_bundle_path),
            "generated_at_utc": bundle.get("generated_at_utc"),
            "window_id": bundle.get("window_id"),
            "tracking_uri": bundle.get("tracking_uri"),
            "fallback_tracking_uri": bundle.get("fallback_tracking_uri"),
            "model_names": sorted(bundle_models),
            "universe_report_path": str(greyscale_report_path),
            "universe_ticker_count": int(len(universe_tickers)),
            "retained_feature_count": int(len(retained_features)),
            "retained_features": retained_features,
            "source_artifacts": dict(bundle.get("source_artifacts", {})),
        },
        "feature_audit_report_path": str(config.feature_audit_output_path),
        "registry_audit": registry_audit,
        "component_models": component_reports,
        "runtime_diagnostics": {
            "fusion_batch_size": int(config.fusion_batch_size),
            "peak_rss_gb": float(peak_rss_gb),
            "memory_limit_gb": MEMORY_LIMIT_GB,
            "memory_limit_passed": memory_limit_passed,
        },
        "fusion": {
            "weights": normalized_weights,
            "unavailable_models": unavailable_models,
            "live_ic": live_ic,
            "live_rank_ic": optional_float(live_rank_ic),
            "live_icir": optional_float(live_icir),
            "backtest_reference_ic": optional_float(backtest_reference["backtest_ic"]),
            "backtest_reference_path": backtest_reference["report_path"],
            "backtest_reference_source": backtest_reference["source"],
            "live_vs_backtest_error_pct": live_error_pct,
            "live_decay_pct": live_decay_pct,
            "threshold_pct": 0.20,
            "memory_limit_gb": MEMORY_LIMIT_GB,
            "memory_limit_passed": memory_limit_passed,
            "passed": passed,
        },
        "feature_audit": feature_audit_report["comparison"],
        "feature_diagnostics": feature_diagnostics,
        "label_diagnostics": label_diagnostics,
        "sample_size": sample_size,
        "ic_series": series_with_sizes_to_records(ic_series, cross_section_sizes),
        "rank_ic_series": series_to_records(rank_ic_series),
    }
    return report, (0 if passed else 1)


def build_feature_audit_report(
    *,
    config: ValidationConfig,
    legacy_retained_features: list[str],
    fusion_retained_features: list[str],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    legacy_only = sorted(set(legacy_retained_features) - set(fusion_retained_features))
    fusion_only = sorted(set(fusion_retained_features) - set(legacy_retained_features))
    overlap = sorted(set(legacy_retained_features) & set(fusion_retained_features))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": safe_git_branch(),
        "script": Path(__file__).name,
        "status": "ok",
        "legacy_live_validator": {
            "model_name": DEFAULT_MODEL_NAME,
            "ic_report_path": str(config.ic_report_path),
            "retained_feature_count": int(len(legacy_retained_features)),
            "retained_features": legacy_retained_features,
        },
        "greyscale_fusion": {
            "bundle_path": str(config.fusion_bundle_path),
            "generated_at_utc": bundle.get("generated_at_utc"),
            "model_names": sorted(dict(bundle.get("models", {}))),
            "retained_feature_count": int(len(fusion_retained_features)),
            "retained_features": fusion_retained_features,
            "source_artifacts": dict(bundle.get("source_artifacts", {})),
        },
        "comparison": {
            "legacy_feature_count": int(len(legacy_retained_features)),
            "fusion_feature_count": int(len(fusion_retained_features)),
            "overlap_feature_count": int(len(overlap)),
            "overlap_features": overlap,
            "legacy_only_feature_count": int(len(legacy_only)),
            "legacy_only_features": legacy_only,
            "fusion_only_feature_count": int(len(fusion_only)),
            "fusion_only_features": fusion_only,
        },
    }


def inspect_registry_champion(
    *,
    registry: ModelRegistry,
    registry_model_name: str,
    bundle_model_payload: dict[str, Any],
    bundle_features: list[str],
) -> tuple[dict[str, Any], RegisteredModelVersion | None]:
    bundle_run = dict(bundle_model_payload.get("mlflow_run", {}))
    bundle_run_id = bundle_run.get("run_id")
    try:
        champion = registry.get_champion(registry_model_name)
    except Exception as exc:
        return {
            "registry_model_name": registry_model_name,
            "champion_exists": False,
            "error": str(exc),
            "bundle_run_id": bundle_run_id,
            "bundle_feature_count": int(len(bundle_features)),
        }, None

    if champion is None:
        return {
            "registry_model_name": registry_model_name,
            "champion_exists": False,
            "bundle_run_id": bundle_run_id,
            "bundle_feature_count": int(len(bundle_features)),
        }, None

    metadata_features = list(champion.metadata.features) if champion.metadata is not None else []
    return {
        "registry_model_name": registry_model_name,
        "champion_exists": True,
        "stage": champion.stage.value,
        "version": int(champion.version),
        "run_id": champion.run_id,
        "bundle_run_id": bundle_run_id,
        "bundle_run_id_match": bool(bundle_run_id and champion.run_id == bundle_run_id),
        "metadata_feature_count": int(len(metadata_features)),
        "metadata_matches_bundle_features": (
            metadata_features == bundle_features if metadata_features else None
        ),
        "champion_metadata_missing": champion.metadata is None,
        "mean_oos_ic": (
            optional_float(float(champion.metadata.metrics.get("mean_oos_ic", np.nan)))
            if champion.metadata is not None
            else None
        ),
    }, champion


def load_deployed_bundle_model(
    *,
    model_key: str,
    bundle_model_payload: dict[str, Any],
    registry: ModelRegistry,
    registry_model_name: str,
    champion: RegisteredModelVersion | None,
) -> tuple[Any | None, dict[str, Any]]:
    artifact_path_raw = bundle_model_payload.get("artifact_path")
    artifact_path = Path(str(artifact_path_raw)) if artifact_path_raw else None
    load_exception: Exception | None = None
    if artifact_path is not None and artifact_path.exists():
        try:
            with open(artifact_path, "rb") as handle:
                return pickle.load(handle), {
                    "method": "bundle_pickle",
                    "model_key": model_key,
                    "artifact_path": str(artifact_path),
                    "bundle_run_id": bundle_model_payload.get("mlflow_run", {}).get("run_id"),
                }
        except Exception as exc:
            load_exception = exc

    if champion is None:
        return None, {
            "method": "unavailable",
            "model_key": model_key,
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifact_exists": bool(artifact_path is not None and artifact_path.exists()),
            "bundle_pickle_error": str(load_exception) if load_exception is not None else None,
            "reason": "bundle artifact missing and no registry champion available",
            "registry_model_name": registry_model_name,
        }

    model_object, load_audit = load_champion_model(
        registry=registry,
        champion=champion,
        model_name=registry_model_name,
    )
    return model_object, {
        "method": "registry_champion_fallback",
        "model_key": model_key,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "artifact_exists": bool(artifact_path is not None and artifact_path.exists()),
        "bundle_pickle_error": str(load_exception) if load_exception is not None else None,
        "registry_model_name": registry_model_name,
        "registry_load": load_audit,
    }


def build_live_evaluation_panel(
    *,
    tickers: list[str] | None = None,
    retained_features: list[str],
    model_features: list[str],
) -> dict[str, Any]:
    if tickers is None:
        tickers = load_tracked_tickers(exclude_ticker=BENCHMARK_TICKER)
    feature_matrix, feature_diagnostics = build_feature_matrix(
        tickers=tickers,
        retained_features=retained_features,
        model_features=model_features,
    )
    label_series, label_diagnostics = build_label_series(tickers=tickers)
    aligned_X, aligned_y = align_panel(feature_matrix, label_series)
    if aligned_X.empty or aligned_y.empty:
        raise RuntimeError("Aligned live evaluation panel is empty.")

    return {
        "feature_matrix": feature_matrix,
        "label_series": label_series,
        "aligned_X": aligned_X,
        "aligned_y": aligned_y,
        "tickers": tickers,
        "feature_diagnostics": feature_diagnostics,
        "label_diagnostics": label_diagnostics,
        "sample_size_base": {
            "prediction_rows": int(len(feature_matrix)),
            "label_rows": int(len(label_series)),
            "aligned_rows_before_filter": int(len(aligned_X)),
            "friday_dates_before_filter": int(aligned_X.index.get_level_values("trade_date").nunique()),
        },
    }


def evaluate_prediction_series(
    *,
    aligned_y: pd.Series,
    predictions: pd.Series,
    sample_size_base: dict[str, Any],
) -> dict[str, Any]:
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
        **sample_size_base,
        "aligned_rows_after_filter": int(len(filtered_y)),
        "dates_with_ic": int(len(ic_series)),
        "avg_cross_section_size": float(cross_section_sizes.mean()),
        "min_cross_section_size": int(cross_section_sizes.min()),
        "max_cross_section_size": int(cross_section_sizes.max()),
    }
    return {
        "live_ic": float(information_coefficient(y_true=filtered_y, y_pred=filtered_pred)),
        "live_rank_ic": float(rank_information_coefficient(y_true=filtered_y, y_pred=filtered_pred)),
        "live_icir": float(icir(y_true=filtered_y, y_pred=filtered_pred)),
        "ic_series": ic_series,
        "rank_ic_series": rank_ic_series,
        "cross_section_sizes": cross_section_sizes,
        "sample_size": sample_size,
    }


def combine_prediction_series(
    prediction_map: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    combined: pd.Series | None = None
    for model_key, predictions in prediction_map.items():
        weight = float(weights.get(model_key, 0.0))
        if np.isclose(weight, 0.0):
            continue
        weighted = predictions.mul(weight)
        combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)
    if combined is None:
        raise RuntimeError("No fusion component predictions were available for combination.")
    return combined.rename("score")


def normalize_weight_dict_local(weights: dict[str, float]) -> dict[str, float]:
    positive = {key: float(value) for key, value in weights.items() if float(value) > 0.0}
    if not positive:
        count = max(len(weights), 1)
        return {key: 1.0 / count for key in weights}
    total = sum(positive.values())
    return {key: value / total for key, value in positive.items()}


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def current_peak_rss_gb() -> float:
    max_rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss_kb / (1024.0 * 1024.0)


def is_missing_indicator_feature(feature_name: str) -> bool:
    return str(feature_name).startswith("is_missing_")


def base_feature_name(feature_name: str) -> str:
    name = str(feature_name)
    if is_missing_indicator_feature(name):
        return name[len("is_missing_") :]
    return name


def expand_requested_feature_names(retained_features: list[str]) -> list[str]:
    expanded = set(retained_features)
    expanded.update(base_feature_name(feature_name) for feature_name in retained_features)
    return sorted(expanded)


def load_fusion_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": "fusion_analysis_summary",
            "backtest_ic": FUSION_BACKTEST_AGGREGATE_IC,
            "source": "fallback_constant",
            "report_path": str(path),
        }

    payload = json.loads(path.read_text())
    mean_ic = payload.get("summary", {}).get("ic_weighted_fusion", {}).get("mean_ic")
    return {
        "name": "fusion_analysis_summary",
        "backtest_ic": float(mean_ic if mean_ic is not None else FUSION_BACKTEST_AGGREGATE_IC),
        "source": "fusion_analysis_report",
        "report_path": str(path),
    }


def load_latest_greyscale_universe(report_dir: Path) -> tuple[list[str], Path]:
    report_paths = sorted(report_dir.glob("week_*.json"))
    if not report_paths:
        raise FileNotFoundError(f"No greyscale weekly reports found in {report_dir}")

    def _week_number(path: Path) -> int:
        stem = path.stem
        suffix = stem.split("_")[-1]
        try:
            return int(suffix)
        except ValueError:
            return -1

    latest_report = max(report_paths, key=_week_number)
    payload = json.loads(latest_report.read_text())
    fusion_scores = dict(payload.get("score_vectors", {}).get("fusion", {}))
    tickers = sorted(str(ticker).upper() for ticker in fusion_scores if str(ticker).upper() != BENCHMARK_TICKER)
    if not tickers:
        raise RuntimeError(f"Greyscale report {latest_report} does not contain any fusion score tickers.")
    return tickers, latest_report


def load_champion_model(
    *,
    registry: ModelRegistry,
    champion: RegisteredModelVersion,
    model_name: str,
) -> tuple[Any, dict[str, Any]]:
    model_uri = registry.model_uri(model_name=model_name, alias="champion")
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


def materialize_fusion_date_partitions(
    *,
    tickers: list[str],
    batch_size: int,
    temp_dir: Path,
) -> list[pd.Timestamp]:
    install_runtime_optimizations()
    trade_dates: set[pd.Timestamp] = set()
    ticker_batches = chunked(tickers, batch_size)
    for batch_number, ticker_batch in enumerate(ticker_batches, start=1):
        logger.info(
            "materializing fusion feature batch {}/{} for {} tickers",
            batch_number,
            len(ticker_batches),
            len(ticker_batch),
        )
        batch_features = build_forward_filled_base_features_long_fast(
            tickers=ticker_batch,
            start_date=EVAL_START - timedelta(days=400),
            end_date=EVAL_END,
            as_of=AS_OF,
        )
        if batch_features.empty:
            logger.warning("fusion feature batch {} returned no rows", batch_number)
            continue

        batch_features["trade_date"] = pd.to_datetime(batch_features["trade_date"])
        friday_mask = (
            (batch_features["trade_date"] >= pd.Timestamp(EVAL_START))
            & (batch_features["trade_date"] <= pd.Timestamp(EVAL_END))
            & (batch_features["trade_date"].dt.weekday == REBALANCE_WEEKDAY)
        )
        friday_features = batch_features.loc[friday_mask].copy()
        if friday_features.empty:
            logger.warning("fusion feature batch {} has no Friday rows inside the evaluation window", batch_number)
            continue

        for trade_date, group in friday_features.groupby("trade_date", sort=True):
            date_key = pd.Timestamp(trade_date)
            trade_dates.add(date_key)
            date_dir = temp_dir / date_key.date().isoformat()
            date_dir.mkdir(parents=True, exist_ok=True)
            group.to_parquet(date_dir / f"batch_{batch_number:03d}.parquet", index=False)

        del batch_features
        del friday_features
        gc.collect()

    if not trade_dates:
        raise RuntimeError("No fusion feature partitions were materialized for the evaluation window.")
    return sorted(trade_dates)


def build_forward_filled_base_features_long_fast(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    as_of: date,
) -> pd.DataFrame:
    base_features = build_raw_base_features_long_fast(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
    )
    if base_features.empty:
        return pd.DataFrame(
            columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled", "_raw_missing"],
        )

    base_features["_raw_missing"] = base_features["feature_value"].isna()
    forward_filled = preprocessing_module.forward_fill_features(base_features)
    return forward_filled.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def build_raw_base_features_long_fast(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    as_of: date,
) -> pd.DataFrame:
    normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")

    as_of_ts = _as_of_datetime(as_of)
    pipeline = FeaturePipeline()
    history_start = start_date - timedelta(days=400)
    prices = get_prices_pit(
        tickers=normalized_tickers,
        start_date=history_start,
        end_date=end_date,
        as_of=as_of_ts,
    )
    if prices.empty:
        logger.warning("feature pipeline found no PIT prices for requested tickers")
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"])

    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    prices.sort_values(["ticker", "trade_date"], inplace=True)
    output_prices = prices.loc[
        (prices["trade_date"] >= start_date)
        & (prices["trade_date"] <= end_date)
    ].copy()
    if output_prices.empty:
        logger.warning("feature pipeline has no prices inside the requested output window")
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"])

    technical = compute_technical_features(prices)
    technical = technical.loc[
        (technical["trade_date"] >= start_date)
        & (technical["trade_date"] <= end_date)
        & (technical["ticker"].isin(normalized_tickers))
        & (~technical["feature_name"].isin(CROSS_SECTIONAL_TECHNICAL_FEATURE_NAMES))
    ].copy()
    technical["is_filled"] = False

    fundamental_frames: list[pd.DataFrame] = []
    grouped_prices = {
        str(ticker).upper(): group.copy()
        for ticker, group in output_prices.groupby("ticker", sort=False)
        if not group.empty
    }
    compute_fundamental = feature_pipeline_module.compute_fundamental_features
    worker_count = min(MAX_FUNDAMENTAL_WORKERS, max(len(grouped_prices), 1))
    if worker_count <= 1:
        for ticker, ticker_prices in grouped_prices.items():
            fundamental_frames.append(
                compute_fundamental(
                    ticker=ticker,
                    as_of=as_of_ts,
                    prices_df=ticker_prices,
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    compute_fundamental,
                    ticker=ticker,
                    as_of=as_of_ts,
                    prices_df=ticker_prices,
                ): ticker
                for ticker, ticker_prices in grouped_prices.items()
            }
            for future in as_completed(future_map):
                fundamental_frames.append(future.result())

    fundamentals = (
        pd.concat(fundamental_frames, ignore_index=True)
        if fundamental_frames
        else pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    )
    fundamentals["is_filled"] = False

    macro = pipeline._compute_broadcast_macro_features(output_prices, as_of_ts)
    macro["is_filled"] = False

    base_features = pd.concat([technical, fundamentals, macro], ignore_index=True)
    base_features["ticker"] = base_features["ticker"].astype(str).str.upper()
    base_features["trade_date"] = pd.to_datetime(base_features["trade_date"]).dt.date
    base_features["feature_name"] = base_features["feature_name"].astype(str)
    base_features["feature_value"] = pd.to_numeric(base_features["feature_value"], errors="coerce")
    base_features["is_filled"] = base_features["is_filled"].fillna(False).astype(bool)
    return base_features.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def load_partitioned_base_features_for_date(date_dir: Path) -> pd.DataFrame:
    partition_paths = sorted(date_dir.glob("batch_*.parquet"))
    if not partition_paths:
        raise RuntimeError(f"No fusion feature partitions found in {date_dir}")

    frames = [pd.read_parquet(path) for path in partition_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["ticker"] = combined["ticker"].astype(str).str.upper()
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])
    combined["feature_name"] = combined["feature_name"].astype(str)
    combined["feature_value"] = pd.to_numeric(combined["feature_value"], errors="coerce")
    combined["is_filled"] = combined["is_filled"].fillna(False).astype(bool)
    combined["_raw_missing"] = combined["_raw_missing"].fillna(True).astype(bool)
    return combined.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def build_cross_sectional_technical_features_for_date(base_features: pd.DataFrame) -> pd.DataFrame:
    wide = (
        base_features.pivot_table(
            index=["ticker", "trade_date"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        )
        .sort_index()
    )
    if wide.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled", "_raw_missing"])

    derived: list[pd.DataFrame] = []
    for source_name, target_name in (
        ("ret_20d", "momentum_rank_20d"),
        ("ret_60d", "momentum_rank_60d"),
        ("vol_20d", "vol_rank"),
    ):
        if source_name not in wide.columns:
            ranked = pd.Series(np.nan, index=wide.index, dtype=float)
        else:
            ranked = rank_series_to_unit_interval(wide[source_name])
        frame = ranked.rename("feature_value").reset_index()
        frame["feature_name"] = target_name
        frame["is_filled"] = False
        frame["_raw_missing"] = frame["feature_value"].isna()
        derived.append(frame[["ticker", "trade_date", "feature_name", "feature_value", "is_filled", "_raw_missing"]])

    return pd.concat(derived, ignore_index=True).sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def rank_series_to_unit_interval(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if len(non_null) == 1:
        return pd.Series(np.where(series.notna(), 0.5, np.nan), index=series.index, dtype=float)

    ranked = non_null.rank(method="average")
    normalized = (ranked - 1) / (len(non_null) - 1)
    return normalized.reindex(series.index)


def finalize_fusion_date_feature_matrix(
    *,
    base_features: pd.DataFrame,
    retained_features: list[str],
) -> pd.DataFrame:
    requested_feature_names = expand_requested_feature_names(retained_features)
    global_technical = build_cross_sectional_technical_features_for_date(base_features)
    composite_inputs = pd.concat(
        [
            base_features[["ticker", "trade_date", "feature_name", "feature_value"]],
            global_technical[["ticker", "trade_date", "feature_name", "feature_value"]],
        ],
        ignore_index=True,
    )
    composites = compute_composite_features(composite_inputs)
    composites["is_filled"] = False
    composites["_raw_missing"] = composites["feature_value"].isna()

    raw_features = pd.concat([base_features, global_technical, composites], ignore_index=True)
    raw_features.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    winsorized = preprocessing_module.winsorize_features(raw_features)
    normalized = preprocessing_module.rank_normalize_features(winsorized)
    raw_missing = normalized.pop("_raw_missing").astype(bool)
    finalized = preprocessing_module.add_missing_flags(normalized, raw_missing)
    filtered_long = finalized.loc[
        finalized["feature_name"].astype(str).isin(requested_feature_names),
        ["ticker", "trade_date", "feature_name", "feature_value"],
    ].copy()
    matrix = long_to_feature_matrix(filtered_long, retained_features)
    matrix = matrix.reindex(columns=retained_features)
    return fill_feature_matrix(matrix)


def build_feature_matrix(
    *,
    tickers: list[str],
    retained_features: list[str],
    model_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    install_runtime_optimizations()
    requested_feature_names = expand_requested_feature_names(retained_features)
    features_long = build_features_long_fast(
        tickers=tickers,
        start_date=EVAL_START - timedelta(days=400),
        end_date=EVAL_END,
        as_of=AS_OF,
    )
    if features_long.empty:
        raise RuntimeError("Feature pipeline returned no rows.")

    filtered_long = features_long.loc[
        features_long["feature_name"].astype(str).isin(requested_feature_names),
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


def build_features_long_fast(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    as_of: date,
) -> pd.DataFrame:
    normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")

    as_of_ts = _as_of_datetime(as_of)
    pipeline = FeaturePipeline()
    history_start = start_date - timedelta(days=400)
    prices = get_prices_pit(
        tickers=normalized_tickers,
        start_date=history_start,
        end_date=end_date,
        as_of=as_of_ts,
    )
    if prices.empty:
        logger.warning("feature pipeline found no PIT prices for requested tickers")
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    prices.sort_values(["ticker", "trade_date"], inplace=True)
    output_prices = prices.loc[
        (prices["trade_date"] >= start_date)
        & (prices["trade_date"] <= end_date)
    ].copy()
    if output_prices.empty:
        logger.warning("feature pipeline has no prices inside the requested output window")
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    technical = compute_technical_features(prices)
    technical = technical.loc[
        (technical["trade_date"] >= start_date)
        & (technical["trade_date"] <= end_date)
        & (technical["ticker"].isin(normalized_tickers))
    ].copy()

    fundamental_frames: list[pd.DataFrame] = []
    grouped_prices = {
        str(ticker).upper(): group.copy()
        for ticker, group in output_prices.groupby("ticker", sort=False)
        if not group.empty
    }
    compute_fundamental = feature_pipeline_module.compute_fundamental_features
    worker_count = min(MAX_FUNDAMENTAL_WORKERS, max(len(grouped_prices), 1))
    if worker_count <= 1:
        for ticker, ticker_prices in grouped_prices.items():
            fundamental_frames.append(
                compute_fundamental(
                    ticker=ticker,
                    as_of=as_of_ts,
                    prices_df=ticker_prices,
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    compute_fundamental,
                    ticker=ticker,
                    as_of=as_of_ts,
                    prices_df=ticker_prices,
                ): ticker
                for ticker, ticker_prices in grouped_prices.items()
            }
            for future in as_completed(future_map):
                fundamental_frames.append(future.result())

    fundamentals = (
        pd.concat(fundamental_frames, ignore_index=True)
        if fundamental_frames
        else pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    )

    macro = pipeline._compute_broadcast_macro_features(output_prices, as_of_ts)
    base_features = pd.concat([technical, fundamentals, macro], ignore_index=True)
    composite = compute_composite_features(base_features)
    all_features = pd.concat([base_features, composite], ignore_index=True)
    processed = feature_pipeline_module.preprocess_features(all_features)
    logger.info("feature pipeline completed fast batch with {} rows", len(processed))
    return processed


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


def score_fusion_models_batched(
    *,
    temp_dir: Path,
    trade_dates: list[pd.Timestamp],
    tickers: list[str],
    retained_features: list[str],
    model_specs: dict[str, dict[str, Any]],
    label_series: pd.Series,
    batch_size: int,
) -> tuple[dict[str, pd.Series], dict[str, Any], dict[str, Any], float]:
    prediction_chunks = {model_key: [] for model_key in model_specs}
    feature_rows_total = 0
    aligned_rows_before_filter = 0
    aligned_dates: set[pd.Timestamp] = set()
    observed_tickers: set[str] = set()
    feature_rows_by_date: dict[pd.Timestamp, int] = {}
    peak_rss_gb = current_peak_rss_gb()

    for position, trade_date in enumerate(trade_dates, start=1):
        logger.info(
            "scoring fusion date {}/{} {} from batch partitions",
            position,
            len(trade_dates),
            trade_date.date(),
        )
        date_dir = temp_dir / trade_date.date().isoformat()
        base_features = load_partitioned_base_features_for_date(date_dir)
        date_matrix = finalize_fusion_date_feature_matrix(
            base_features=base_features,
            retained_features=retained_features,
        )
        if date_matrix.empty:
            continue

        feature_rows_total += int(len(date_matrix))
        feature_rows_by_date[trade_date] = int(len(date_matrix))
        observed_tickers.update(date_matrix.index.get_level_values("ticker").astype(str).tolist())
        aligned_index = date_matrix.index.intersection(label_series.index)
        if len(aligned_index) == 0:
            del base_features
            del date_matrix
            gc.collect()
            peak_rss_gb = max(peak_rss_gb, current_peak_rss_gb())
            continue

        aligned_rows_before_filter += int(len(aligned_index))
        aligned_dates.add(trade_date)
        aligned_X = date_matrix.loc[aligned_index].sort_index()

        for model_key, spec in model_specs.items():
            model_input = aligned_X.reindex(columns=spec["feature_names"]).fillna(0.5)
            if model_input.empty:
                continue
            prediction_chunks[model_key].append(
                ChampionChallengerRunner._predict_series(spec["model_object"], model_input),
            )

        del base_features
        del date_matrix
        del aligned_X
        gc.collect()
        peak_rss_gb = max(peak_rss_gb, current_peak_rss_gb())

    predictions = {
        model_key: pd.concat(chunks).sort_index()
        for model_key, chunks in prediction_chunks.items()
        if chunks
    }
    if not feature_rows_by_date:
        raise RuntimeError("Fusion feature batching produced no finalized date slices.")

    feature_counts = pd.Series(feature_rows_by_date).sort_index()
    feature_diagnostics = {
        "rows": int(feature_rows_total),
        "dates": int(len(feature_counts)),
        "tickers": int(len(observed_tickers) if observed_tickers else len(tickers)),
        "min_date": pd.Timestamp(feature_counts.index.min()).date().isoformat(),
        "max_date": pd.Timestamp(feature_counts.index.max()).date().isoformat(),
        "avg_cross_section_size": float(feature_counts.mean()),
        "min_cross_section_size": int(feature_counts.min()),
        "max_cross_section_size": int(feature_counts.max()),
        "batch_mode": "ticker_batched_date_partitioned",
        "fusion_batch_size": int(batch_size),
    }
    sample_size_base = {
        "prediction_rows": int(feature_rows_total),
        "label_rows": int(len(label_series)),
        "aligned_rows_before_filter": int(aligned_rows_before_filter),
        "friday_dates_before_filter": int(len(aligned_dates)),
    }
    return predictions, feature_diagnostics, sample_size_base, peak_rss_gb


def load_live_ic_observation(
    *,
    model_object: Any,
    retained_features: list[str],
    model_features: list[str],
    model_name: str,
    shadow_mode_report_path: Path,
) -> dict[str, Any]:
    shadow_snapshot = load_shadow_mode_live_snapshot(
        shadow_mode_report_path,
        model_name=model_name,
    )
    if shadow_snapshot is not None:
        logger.info(
            "using shadow-mode live IC snapshot from {} instead of rebuilding the full historical PIT panel",
            shadow_mode_report_path,
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
                "report_path": str(shadow_mode_report_path),
                "feature_rows_total": int(shadow_snapshot["feature_rows_total"]),
                "feature_rows_live_date": int(shadow_snapshot["feature_rows_live_date"]),
                "feature_matrix_shape": dict(shadow_snapshot["feature_matrix_shape"]),
                "pit_signal_date": shadow_snapshot["pit_signal_date"],
            },
            "label_diagnostics": {
                "source": "shadow_mode_dry_run",
                "report_path": str(shadow_mode_report_path),
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


def load_shadow_mode_live_snapshot(path: Path, *, model_name: str) -> dict[str, Any] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text())
    snapshot_model_name = payload.get("weekly_signal_pipeline", {}).get("model", {}).get("model_name")
    if snapshot_model_name and str(snapshot_model_name) != model_name:
        logger.warning(
            "shadow-mode live IC snapshot model_name={} does not match validation model_name={}",
            snapshot_model_name,
            model_name,
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
        "model_name": snapshot_model_name,
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


def compute_decay_pct(value: float, reference: float) -> float:
    if pd.isna(value) or pd.isna(reference) or np.isclose(reference, 0.0):
        return float("nan")
    return float((reference - value) / abs(reference))


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


def log_summary(report: dict[str, Any], *, output_path: Path) -> None:
    metrics = dict(report.get("metrics", report.get("fusion", {})))
    comparison = dict(report.get("comparison", report.get("fusion", {})))
    sample_size = dict(report.get("sample_size", {}))
    logger.info(
        "live IC validation {} | live_ic={:.6f} rank_ic={:.6f} icir={:.6f} error_pct={:.4f} dates={} rows={}",
        report["status"],
        float(metrics.get("live_ic", float("nan"))),
        float(metrics["live_rank_ic"]) if metrics["live_rank_ic"] is not None else float("nan"),
        float(metrics["live_icir"]) if metrics["live_icir"] is not None else float("nan"),
        float(
            comparison.get(
                "primary_error_pct",
                comparison.get("live_vs_backtest_error_pct", float("nan")),
            ),
        ),
        int(sample_size.get("dates_with_ic", 0)),
        int(sample_size.get("aligned_rows_after_filter", 0)),
    )
    logger.info("saved live IC validation report to {}", output_path)


def _as_of_datetime(as_of: date) -> datetime:
    return datetime.combine(as_of, time.max, tzinfo=timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
