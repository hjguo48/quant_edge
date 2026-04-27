from __future__ import annotations

"""W12-1: Freeze the W10 60D Ridge champion into an immutable bundle.

Steps:
  1. Load W11 final walk-forward window (last entry in
     walkforward_v9full9y_60d_ridge_13w.json) — the most recent train+val+test split.
  2. Reuse prepare_horizon_artifacts to load aligned features + labels.
  3. Refit RidgeBaselineModel on train+validation with the window's selected alpha
     (deterministic — same fit used to produce the last OOS test predictions).
  4. Pickle the model + dump bundle.json + checksums.json + copy frozen universe.
  5. Output: data/models/bundles/w12_60d_ridge_swbuf_v1/
  6. Refresh legacy pointer data/models/fusion_model_bundle_60d.json with bundle.json.

Bundle contents (per W12_prep_plan_2026-04-27.md):
  - bundle.json: full manifest (identity, runtime policy, model, features, portfolio,
    cost, universe, provenance)
  - model.pkl: single Ridge artifact (final W11 train+val fit)
  - frozen_universe_503.json: copy of training universe snapshot
  - checksums.json: SHA256 of bundle.json + model.pkl + universe + key reports

Pass criteria:
  - All files written
  - Checksums verifiable
  - Model loadable + predicts on test_X reproducing the report's last-window test_ic
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_horizon_fusion import (  # noqa: E402
    extract_ridge_alpha,
    parse_horizon_days,
    prepare_horizon_artifacts,
    select_report_windows,
    slice_all_splits,
)
from scripts.run_walkforward_comparison import (  # noqa: E402
    BENCHMARK_TICKER,
    DEFAULT_ALL_FEATURES_PATH,
    LABEL_BUFFER_DAYS,
    REBALANCE_WEEKDAY,
    parse_date,
)
from src.models.baseline import RidgeBaselineModel  # noqa: E402

DEFAULT_REPORT = "data/reports/walkforward_v9full9y_60d_ridge_13w.json"
DEFAULT_FEATURE_MATRIX = "data/features/walkforward_v9full9y_fm_60d.parquet"
DEFAULT_LABEL = "data/labels/forward_returns_60d_v9full9y.parquet"
DEFAULT_FROZEN_UNIVERSE = "data/features/frozen_universe_503.json"
DEFAULT_BUNDLE_VERSION = "w12_60d_ridge_swbuf_v1"
DEFAULT_BUNDLE_BASE = "data/models/bundles"
LEGACY_BUNDLE_POINTER = "data/models/fusion_model_bundle_60d.json"

CHAMPION_PORTFOLIO = {
    # NB: greyscale runner checks weighting_scheme == "score_weighted" exactly;
    # the buffered-turnover variant is signalled by the buffer/min_trade fields.
    "weighting_scheme": "score_weighted",
    "selection_pct": 0.20,
    "sell_buffer_pct": 0.25,
    "min_trade_weight": 0.01,
    "max_weight": 0.05,
    "min_holdings": 20,
    "weight_shrinkage": 0.0,
    "no_trade_zone": 0.0,
    "turnover_penalty_lambda": 0.0,
}
CHAMPION_COST = {
    "name": "AlmgrenChrissCostModel",
    "eta": 0.426,
    "gamma": 0.942,
    "commission_per_share": 0.005,
    "min_spread_bps": 2.0,
    "gap_penalty_threshold": 0.02,
    "gap_penalty_multiplier": 0.5,
    "low_volume_threshold": 0.30,
    "low_volume_temp_impact_multiplier": 2.0,
}

KEY_REPORTS_FOR_PROVENANCE = [
    "data/reports/walkforward_v9full9y_60d_ridge_13w.json",
    "data/reports/w10_truth_table_60d.json",
    "data/reports/w10_stress_tests.json",
    "data/reports/w10_capacity_sweep.json",
    "data/reports/W10_verdict_2026-04-26.md",
    "data/reports/W10_stress_test_verdict_2026-04-27.md",
    "data/reports/W11_verdict_2026-04-27.md",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    bundle_dir = REPO_ROOT / args.bundle_base / args.bundle_version
    if bundle_dir.exists() and not args.overwrite:
        raise RuntimeError(f"Bundle dir already exists: {bundle_dir} (use --overwrite to replace)")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    logger.info("freezing bundle: {}", bundle_dir)

    report_path = REPO_ROOT / args.report
    payload = json.loads(report_path.read_text())
    horizon_days = parse_horizon_days(payload)
    windows = select_report_windows(payload, limit=None)
    final_window = windows[-1]
    final_window_id = str(final_window["window_id"])
    logger.info("using final window: {} (test {} → {})",
                final_window_id,
                final_window["dates"]["test_start"],
                final_window["dates"]["test_end"])

    as_of = parse_date(str(payload["as_of"]))
    benchmark_ticker = str(payload.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER)).upper()
    rebalance_weekday = int(payload.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY))

    artifacts = prepare_horizon_artifacts(
        label=f"{horizon_days}D",
        horizon_days=horizon_days,
        report_path=report_path,
        report_payload=payload,
        windows=windows,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=REPO_ROOT / args.feature_matrix_path,
        label_cache_path=REPO_ROOT / args.label_path,
        as_of=as_of,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )

    date_keys = ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end")
    final_dates = {key: parse_date(str(final_window["dates"][key])) for key in date_keys}
    split = slice_all_splits(
        X=artifacts.feature_matrix,
        y=artifacts.labels,
        dates=final_dates,
        rebalance_weekday=rebalance_weekday,
    )
    alpha = extract_ridge_alpha(final_window)
    logger.info("final-window Ridge alpha = {}", alpha)

    final_train_X = pd.concat([split["train_X"], split["validation_X"]]).sort_index()
    final_train_y = pd.concat([split["train_y"], split["validation_y"]]).sort_index()
    logger.info("fitting Ridge on train+val: {} rows × {} features", len(final_train_X), final_train_X.shape[1])

    model = RidgeBaselineModel(alpha=alpha)
    model.train(final_train_X, final_train_y)

    test_pred = model.predict(split["test_X"]).rename("test_score")
    test_y = split["test_y"]
    aligned = pd.concat([test_pred.rename("y_hat"), test_y.rename("y")], axis=1).dropna()
    test_ic = float(pearsonr(aligned["y"], aligned["y_hat"]).statistic) if len(aligned) >= 2 else float("nan")
    logger.info("reproduced test_ic on final window: {:.4f} (rows={})", test_ic, len(aligned))

    expected_test_ic = float(final_window.get("results", {}).get("ridge", {}).get("test_ic", float("nan")))
    if not np.isnan(expected_test_ic):
        diff = abs(test_ic - expected_test_ic)
        logger.info("expected test_ic from report: {:.4f}, diff={:.6f}", expected_test_ic, diff)
        if diff > 1e-3:
            logger.warning("test_ic mismatch > 1bp — investigate before proceeding")

    model_path = bundle_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("saved model to {}", model_path)

    universe_src = REPO_ROOT / args.frozen_universe
    universe_dst = bundle_dir / "frozen_universe_503.json"
    shutil.copy2(universe_src, universe_dst)
    universe_payload = json.loads(universe_dst.read_text())
    eligible_count = len(universe_payload.get("tickers", []))
    logger.info("copied frozen universe: {} ({} tickers)", universe_dst, eligible_count)

    feature_names = list(artifacts.retained_features)
    cutoff_date = final_dates["test_end"].isoformat()
    feature_fingerprint = compute_bundle_validator_fingerprint(
        version=args.bundle_version,
        cutoff_date=cutoff_date,
        feature_names=feature_names,
    )

    git_hash = _git_hash()
    bundle_manifest: dict[str, Any] = {
        "version": args.bundle_version,
        "cutoff_date": cutoff_date,
        "window_id": final_window_id,  # top-level for run_greyscale_live consumption
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_git_hash": git_hash,
        "model_type": "ridge_single_horizon",
        "strategy_name": "score_weighted_buffered",  # 描述用; runtime check 看 turnover_controls.weighting_scheme
        "horizon_days": int(horizon_days),
        "rebalance_weekday": int(rebalance_weekday),
        "execution_timing": "weekly Friday signal, T+1 open execution",
        "score_space": "raw",
        "benchmark_ticker": benchmark_ticker,
        "pit_mode": "knowledge_time <= as_of_utc",
        "models": {
            "ridge": {
                "artifact_path": str(model_path.relative_to(REPO_ROOT)),
                "feature_names": feature_names,
                "alpha": float(alpha),
                "window_id": final_window_id,
                "trained_on": "train+validation",
                "train_rows": int(len(final_train_X)),
                "train_dates": [
                    final_dates["train_start"].isoformat(),
                    final_dates["validation_end"].isoformat(),
                ],
                "test_window_dates": [
                    final_dates["test_start"].isoformat(),
                    final_dates["test_end"].isoformat(),
                ],
                "reproduced_test_ic": float(test_ic),
                "report_test_ic": expected_test_ic,
            },
        },
        "seed_weights": {"ridge": 1.0},
        "required_features": feature_names,
        "retained_features": feature_names,
        "feature_fingerprint": feature_fingerprint,
        "turnover_controls": {
            "enabled": True,
            "weighting_scheme": CHAMPION_PORTFOLIO["weighting_scheme"],
            "selection_pct": CHAMPION_PORTFOLIO["selection_pct"],
            "sell_buffer_pct": CHAMPION_PORTFOLIO["sell_buffer_pct"],
            "min_trade_weight": CHAMPION_PORTFOLIO["min_trade_weight"],
            "max_weight": CHAMPION_PORTFOLIO["max_weight"],
            "min_holdings": CHAMPION_PORTFOLIO["min_holdings"],
            "weight_shrinkage": CHAMPION_PORTFOLIO["weight_shrinkage"],
            "no_trade_zone": CHAMPION_PORTFOLIO["no_trade_zone"],
            "turnover_penalty_lambda": CHAMPION_PORTFOLIO["turnover_penalty_lambda"],
        },
        "selection_pct": CHAMPION_PORTFOLIO["selection_pct"],
        "sell_buffer_pct": CHAMPION_PORTFOLIO["sell_buffer_pct"],
        "min_trade_weight": CHAMPION_PORTFOLIO["min_trade_weight"],
        "max_weight": CHAMPION_PORTFOLIO["max_weight"],
        "min_holdings": CHAMPION_PORTFOLIO["min_holdings"],
        "cost_model": CHAMPION_COST,
        "eligible_universe_path": str(universe_dst.relative_to(REPO_ROOT)),
        "eligible_universe_count": int(eligible_count),
        "live_universe_mode": "active_sp500_intersect_bundle_snapshot",
        "source_artifacts": {
            "walkforward_report": str((REPO_ROOT / args.report).relative_to(REPO_ROOT)),
            "feature_matrix_path": str((REPO_ROOT / args.feature_matrix_path).relative_to(REPO_ROOT)),
            "label_cache_path": str((REPO_ROOT / args.label_path).relative_to(REPO_ROOT)),
            "key_provenance_reports": [
                str((REPO_ROOT / p).relative_to(REPO_ROOT))
                for p in KEY_REPORTS_FOR_PROVENANCE
                if (REPO_ROOT / p).exists()
            ],
        },
        "window_dates": {
            key: final_dates[key].isoformat() for key in date_keys
        },
        "fusion_temperature": 0.0,
    }

    bundle_path = bundle_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True))
    logger.info("saved bundle.json to {}", bundle_path)

    checksum_targets = {
        "bundle.json": bundle_path,
        "model.pkl": model_path,
        "frozen_universe_503.json": universe_dst,
    }
    checksums = {name: sha256_of(path) for name, path in checksum_targets.items()}
    provenance_checksums = {}
    for rel_path in KEY_REPORTS_FOR_PROVENANCE:
        full = REPO_ROOT / rel_path
        if full.exists():
            provenance_checksums[rel_path] = sha256_of(full)
    checksums["provenance"] = provenance_checksums

    checksum_path = bundle_dir / "checksums.json"
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True))
    logger.info("saved checksums to {}", checksum_path)

    if args.update_legacy_pointer:
        legacy_path = REPO_ROOT / LEGACY_BUNDLE_POINTER
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_path, legacy_path)
        logger.info("updated legacy bundle pointer at {}", legacy_path)

    logger.info("bundle freeze complete: {}", bundle_dir)
    print()
    print("=" * 78)
    print(f"W12-1 Bundle Freeze: {args.bundle_version}")
    print("=" * 78)
    print(f"Bundle dir: {bundle_dir.relative_to(REPO_ROOT)}")
    print(f"Model: ridge α={alpha} (final window {final_window_id} train+val fit)")
    print(f"Features: {len(feature_names)} ({sum(1 for f in feature_names if not f.startswith('is_missing_'))} real + missing flags)")
    print(f"Universe: {eligible_count} tickers (frozen snapshot)")
    print(f"Test IC reproduction: {test_ic:.4f} (report: {expected_test_ic:.4f})")
    print(f"Strategy: {CHAMPION_PORTFOLIO['weighting_scheme']}")
    print(f"Score space: raw")
    print()
    print("Checksums:")
    for name, sha in checksums.items():
        if isinstance(sha, str):
            print(f"  {name}: {sha[:16]}...")
    if args.update_legacy_pointer:
        print(f"Legacy pointer: {LEGACY_BUNDLE_POINTER}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX)
    p.add_argument("--label-path", default=DEFAULT_LABEL)
    p.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    p.add_argument("--frozen-universe", default=DEFAULT_FROZEN_UNIVERSE)
    p.add_argument("--bundle-base", default=DEFAULT_BUNDLE_BASE)
    p.add_argument("--bundle-version", default=DEFAULT_BUNDLE_VERSION)
    p.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing bundle dir (use with caution)")
    p.add_argument("--update-legacy-pointer", action="store_true", default=True,
                   help="Refresh data/models/fusion_model_bundle_60d.json to current bundle.json")
    return p.parse_args(argv)


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bundle_validator_fingerprint(
    *,
    version: str,
    cutoff_date: str,
    feature_names: list[str],
) -> str:
    """Match src/models/bundle_validator.py:33 BundleValidator.compute_fingerprint."""
    payload = {
        "cutoff_date": cutoff_date,
        "required_features": sorted(str(f) for f in feature_names),
        "version": version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
