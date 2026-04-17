#!/usr/bin/env python3
"""Validate frozen research registries against the current repo state."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURE_SOURCES = (
    ("src/features/technical.py", "TECHNICAL_FEATURE_NAMES"),
    ("src/features/fundamental.py", "FUNDAMENTAL_FEATURE_NAMES"),
    ("src/features/macro.py", "MACRO_FEATURE_NAMES"),
    ("src/features/alternative.py", "ALTERNATIVE_FEATURE_NAMES"),
    ("src/features/sector_rotation.py", "SECTOR_ROTATION_FEATURE_NAMES"),
    ("src/features/pipeline.py", "COMPOSITE_FEATURE_NAMES"),
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_feature_names() -> list[str]:
    features: list[str] = []
    for rel_path, var_name in FEATURE_SOURCES:
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)
        found = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        found = list(ast.literal_eval(node.value))
                        break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"Could not locate {var_name} in {rel_path}")
        features.extend(found)
    return features


def validate_paths(paths: list[str]) -> list[str]:
    missing: list[str] = []
    for rel_path in paths:
        if not (REPO_ROOT / rel_path).exists():
            missing.append(rel_path)
    return missing


def main() -> None:
    current_state = load_yaml(REPO_ROOT / "configs/research/current_state.yaml")
    family_registry = load_yaml(REPO_ROOT / "configs/research/family_registry.yaml")
    horizon_registry = load_yaml(REPO_ROOT / "configs/research/horizon_registry.yaml")
    champion_registry = load_yaml(REPO_ROOT / "configs/research/champion_registry.yaml")

    base_features = extract_feature_names()
    expected = set(base_features) | {f"is_missing_{name}" for name in base_features}
    actual = set(family_registry["feature_to_family"].keys())

    missing_features = sorted(expected - actual)
    extra_features = sorted(actual - expected)
    if missing_features:
        raise SystemExit(f"Missing family mappings for {len(missing_features)} features: {missing_features[:10]}")
    if extra_features:
        raise SystemExit(f"Unexpected family mappings for {len(extra_features)} features: {extra_features[:10]}")

    if family_registry.get("feature_count") != len(expected):
        raise SystemExit(
            f"family_registry feature_count={family_registry.get('feature_count')} "
            f"does not match expected={len(expected)}"
        )

    horizon_keys = set(horizon_registry["horizons"].keys())
    if horizon_keys != {"1D", "5D", "20D", "60D"}:
        raise SystemExit(f"Unexpected horizon keys: {sorted(horizon_keys)}")

    current_version = current_state["live_champion"]["version"]
    champion_60d = champion_registry["champions"]["60D"]["current"]
    if champion_60d != current_version:
        raise SystemExit(
            f"Champion mismatch: current_state={current_version} champion_registry={champion_60d}"
        )

    path_fields = [
        current_state["live_champion"]["bundle_path"],
        current_state["live_champion"]["artifact_path"],
        current_state["live_champion"]["walkforward_report"],
        current_state["live_champion"]["g3_gate_report"],
        current_state["live_champion"]["last_greyscale_report"],
    ]
    missing_paths = validate_paths(path_fields)
    if missing_paths:
        raise SystemExit(f"Missing referenced artifacts: {missing_paths}")

    print("Registry validation passed.")
    print(f"Base features: {len(base_features)}")
    print(f"Feature mappings: {len(expected)}")
    print(f"Live champion: {current_version}")


if __name__ == "__main__":
    main()
