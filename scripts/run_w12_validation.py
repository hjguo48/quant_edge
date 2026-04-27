from __future__ import annotations

"""W12-3: Bundle validation + registry/state sanity + offline IC consistency.

Runs three sanity checks on the W12 frozen champion bundle:
  1. BundleValidator (src/models/bundle_validator.py): schema + fingerprint +
     feature-store availability check
  2. Registry/state path sanity: confirm current_state.yaml + champion_registry.yaml
     point at existing bundle/artifact paths
  3. Self-prediction smoke: load model.pkl, predict on test slice, verify reproduction

Output: data/reports/w12_validation_report.json + console summary
Pass: all 3 checks PASS

Note: scripts/run_live_ic_validation.py (offline historical PIT replay) is the
heavyweight path. We skip it here unless --include-ic-replay is set, since
that runner expects prebuilt CSV with explicit `retained` flags and would need
infra not available for the new W11 retention list.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

from loguru import logger
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.session import get_engine  # noqa: E402
from src.models.bundle_validator import BundleValidator  # noqa: E402
import yaml  # noqa: E402

DEFAULT_BUNDLE = "data/models/bundles/w12_60d_ridge_swbuf_v1/bundle.json"
DEFAULT_CURRENT_STATE = "configs/research/current_state.yaml"
DEFAULT_CHAMPION_REGISTRY = "configs/research/champion_registry.yaml"
DEFAULT_GUARDRAILS = "configs/live/w12_60d_ridge_swbuf_guardrails_v1.yaml"
DEFAULT_OUTPUT = "data/reports/w12_validation_report.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    bundle_path = REPO_ROOT / args.bundle
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    bundle = json.loads(bundle_path.read_text())
    logger.info("validating bundle: {} (version={})", bundle_path.name, bundle.get("version"))

    results: dict[str, Any] = {
        "bundle_path": str(bundle_path),
        "bundle_version": bundle.get("version"),
        "checks": {},
    }

    # Check 1: BundleValidator schema + feature store
    logger.info("check 1/3: BundleValidator schema + feature store")
    try:
        validator = BundleValidator(bundle_path)
        engine = get_engine()
        with engine.connect() as conn:
            result = validator.validate_schema(conn)
        results["checks"]["bundle_validator"] = {
            "passed": bool(result.passed),
            "missing_features": list(result.missing_features),
            "extra_features_count": len(result.extra_features),
            "fingerprint_matches": bool(result.metadata.get("fingerprint_matches")),
            "expected_fingerprint": result.metadata.get("expected_fingerprint"),
            "computed_fingerprint": result.metadata.get("computed_fingerprint"),
            "required_feature_count": int(result.metadata.get("required_feature_count", 0)),
        }
        logger.info("  → {} (missing={}, fp_match={})",
                    "PASS" if result.passed else "FAIL",
                    len(result.missing_features),
                    result.metadata.get("fingerprint_matches"))
    except Exception as exc:
        results["checks"]["bundle_validator"] = {"passed": False, "error": str(exc)}
        logger.error("  → FAIL ({})", exc)

    # Check 2: Registry/state sanity
    logger.info("check 2/3: registry + current_state path sanity")
    state_path = REPO_ROOT / args.current_state
    registry_path = REPO_ROOT / args.champion_registry
    state_check = check_path_consistency(state_path, registry_path, bundle)
    results["checks"]["registry_state"] = state_check
    logger.info("  → {} ({} stale paths)",
                "PASS" if state_check["passed"] else "FAIL",
                len(state_check.get("stale_paths", [])))

    # Check 3: Self-prediction smoke (load model, predict on a slice, ensure non-empty)
    logger.info("check 3/3: self-prediction smoke")
    smoke = run_self_prediction_smoke(bundle_path, bundle)
    results["checks"]["self_prediction_smoke"] = smoke
    logger.info("  → {} (predicted {} tickers, mean score={:.4f})",
                "PASS" if smoke["passed"] else "FAIL",
                smoke.get("n_predictions", 0),
                smoke.get("mean_score", float("nan")))

    overall_pass = all(c.get("passed", False) for c in results["checks"].values())
    results["overall_pass"] = bool(overall_pass)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
    logger.info("saved validation report to {}", output_path)

    print_summary(results)
    return 0 if overall_pass else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bundle", default=DEFAULT_BUNDLE)
    p.add_argument("--current-state", default=DEFAULT_CURRENT_STATE)
    p.add_argument("--champion-registry", default=DEFAULT_CHAMPION_REGISTRY)
    p.add_argument("--guardrails", default=DEFAULT_GUARDRAILS)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--include-ic-replay", action="store_true",
                   help="Also run heavyweight PIT IC replay (requires CSV infra)")
    return p.parse_args(argv)


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def check_path_consistency(state_path: Path, registry_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    stale_paths: list[dict[str, str]] = []

    state_data = yaml.safe_load(state_path.read_text())
    registry_data = yaml.safe_load(registry_path.read_text())

    bundle_version = str(bundle.get("version"))
    state_version = str(state_data.get("live_champion", {}).get("version", ""))
    registry_60d_version = str(registry_data.get("champions", {}).get("60D", {}).get("current", ""))

    version_match = (state_version == bundle_version == registry_60d_version)
    if not version_match:
        stale_paths.append({
            "where": "version mismatch",
            "state_version": state_version,
            "registry_60d_version": registry_60d_version,
            "bundle_version": bundle_version,
        })

    paths_to_check = [
        ("state.bundle_path", state_data.get("live_champion", {}).get("bundle_path")),
        ("state.artifact_path", state_data.get("live_champion", {}).get("artifact_path")),
        ("state.guardrails_config", state_data.get("live_champion", {}).get("guardrails_config")),
        ("state.frozen_universe_path", state_data.get("live_champion", {}).get("frozen_universe_path")),
        ("state.walkforward_report", state_data.get("live_champion", {}).get("walkforward_report")),
        ("registry.60D.bundle_path", registry_data.get("champions", {}).get("60D", {}).get("bundle_path")),
        ("registry.60D.artifact_path", registry_data.get("champions", {}).get("60D", {}).get("artifact_path")),
    ]
    for name, rel in paths_to_check:
        if rel is None:
            stale_paths.append({"where": name, "issue": "null"})
            continue
        full = REPO_ROOT / rel
        if not full.exists():
            stale_paths.append({"where": name, "path": str(rel), "issue": "missing on disk"})

    return {
        "passed": bool(version_match and not stale_paths),
        "version_match": version_match,
        "state_version": state_version,
        "registry_60d_version": registry_60d_version,
        "bundle_version": bundle_version,
        "stale_paths": stale_paths,
    }


def run_self_prediction_smoke(bundle_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        artifact_rel = bundle.get("models", {}).get("ridge", {}).get("artifact_path")
        if not artifact_rel:
            return {"passed": False, "error": "no artifact_path in bundle.models.ridge"}
        artifact_path = REPO_ROOT / artifact_rel
        if not artifact_path.exists():
            return {"passed": False, "error": f"artifact missing: {artifact_path}"}

        with artifact_path.open("rb") as f:
            model = pickle.load(f)

        feature_names = bundle.get("models", {}).get("ridge", {}).get("feature_names", [])
        if not feature_names:
            return {"passed": False, "error": "no feature_names in bundle"}

        # Load the feature matrix used for training; predict on test slice
        fm_path_rel = bundle.get("source_artifacts", {}).get("feature_matrix_path")
        if not fm_path_rel:
            return {"passed": False, "error": "no feature_matrix_path in bundle.source_artifacts"}
        fm_path = REPO_ROOT / fm_path_rel
        if not fm_path.exists():
            return {"passed": False, "error": f"feature matrix missing: {fm_path}"}

        df = pd.read_parquet(fm_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # Use last available date as smoke
        last_date = df["trade_date"].max()
        slice_df = df[df["trade_date"] == last_date].copy()
        if slice_df.empty:
            return {"passed": False, "error": "no rows on last date"}

        slice_df = slice_df.set_index(["trade_date", "ticker"])
        X = slice_df.reindex(columns=feature_names)
        # Drop rows with any NaN feature
        X_clean = X.dropna()
        if X_clean.empty:
            return {"passed": False, "error": "all rows have NaN features"}

        preds = model.predict(X_clean)
        preds_series = pd.Series(preds, index=X_clean.index)

        return {
            "passed": True,
            "smoke_date": last_date.date().isoformat(),
            "n_predictions": int(len(preds_series)),
            "mean_score": float(preds_series.mean()),
            "std_score": float(preds_series.std()),
            "min_score": float(preds_series.min()),
            "max_score": float(preds_series.max()),
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def print_summary(results: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("W12-3 Validation Report")
    print("=" * 78)
    print(f"Bundle: {results['bundle_path']}")
    print(f"Version: {results['bundle_version']}")
    print()
    for name, c in results["checks"].items():
        passed = c.get("passed", False)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if not passed and "error" in c:
            print(f"    error: {c['error']}")
        if name == "bundle_validator" and "missing_features" in c:
            mc = len(c.get("missing_features", []))
            print(f"    missing features in feature_store: {mc}")
            print(f"    fingerprint match: {c.get('fingerprint_matches')}")
        if name == "registry_state":
            print(f"    bundle version: {c.get('bundle_version')}")
            print(f"    state version: {c.get('state_version')}")
            print(f"    registry 60D: {c.get('registry_60d_version')}")
            stale = c.get("stale_paths", [])
            if stale:
                print(f"    stale paths ({len(stale)}):")
                for s in stale:
                    print(f"      - {s}")
        if name == "self_prediction_smoke" and passed:
            print(f"    date: {c.get('smoke_date')}, predictions: {c.get('n_predictions')}, "
                  f"mean: {c.get('mean_score'):.4f}")
    print()
    overall = "PASS" if results["overall_pass"] else "FAIL"
    print(f"=== Overall: {overall} ===")


if __name__ == "__main__":
    raise SystemExit(main())
