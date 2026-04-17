from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.bundle_validator import BundleSchemaError, BundleValidator


class _FakeScalars:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    def all(self) -> list[str]:
        return list(self._values)


class _FakeExecuteResult:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._values)


class _FakeSession:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    def execute(self, _statement) -> _FakeExecuteResult:
        return _FakeExecuteResult(self._values)


def _write_bundle(
    tmp_path: Path,
    *,
    required_features: list[str],
    version: str = "v5_no_analyst",
    cutoff_date: str = "2025-06-30",
) -> Path:
    payload = {
        "version": version,
        "cutoff_date": cutoff_date,
        "required_features": required_features,
        "generator_git_hash": "deadbeef",
        "generated_at": "2026-04-17T09:00:00+00:00",
        "model_type": "ridge_only",
    }
    validator = BundleValidator(tmp_path / "scratch.json")
    payload["feature_fingerprint"] = validator.compute_fingerprint(payload)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return bundle_path


def test_all_features_present_pass(tmp_path: Path) -> None:
    bundle_path = _write_bundle(
        tmp_path,
        required_features=["curve_inverted_x_growth", "vol_60d"],
    )
    validator = BundleValidator(bundle_path)

    result = validator.validate_schema(
        _FakeSession(["curve_inverted_x_growth", "vol_60d", "atr_14"]),
    )

    assert result.passed is True
    assert result.missing_features == []
    assert "atr_14" in result.extra_features
    assert result.metadata["fingerprint_matches"] is True


def test_missing_feature_fail(tmp_path: Path) -> None:
    bundle_path = _write_bundle(
        tmp_path,
        required_features=["curve_inverted_x_growth", "vol_60d"],
    )
    validator = BundleValidator(bundle_path)

    result = validator.validate_schema(_FakeSession(["vol_60d"]))
    assert result.passed is False
    assert result.missing_features == ["curve_inverted_x_growth"]

    with pytest.raises(BundleSchemaError, match="curve_inverted_x_growth"):
        validator.assert_valid(_FakeSession(["vol_60d"]))


def test_fingerprint_stability(tmp_path: Path) -> None:
    validator = BundleValidator(tmp_path / "bundle.json")
    left = {
        "version": "v5_no_analyst",
        "cutoff_date": "2025-06-30",
        "required_features": ["vol_60d", "curve_inverted_x_growth"],
    }
    right = {
        "required_features": ["curve_inverted_x_growth", "vol_60d"],
        "cutoff_date": "2025-06-30",
        "version": "v5_no_analyst",
    }

    assert validator.compute_fingerprint(left) == validator.compute_fingerprint(right)


def test_metadata_roundtrip(tmp_path: Path) -> None:
    bundle_path = _write_bundle(
        tmp_path,
        required_features=["curve_inverted_x_growth", "vol_60d"],
    )
    validator = BundleValidator(bundle_path)
    result = validator.validate_schema(
        _FakeSession(["curve_inverted_x_growth", "vol_60d"]),
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["feature_fingerprint"] == result.metadata["computed_fingerprint"]
    assert result.metadata["generator_git_hash"] == "deadbeef"
    assert result.metadata["version"] == "v5_no_analyst"
