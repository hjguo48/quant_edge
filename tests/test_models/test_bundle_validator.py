from __future__ import annotations

import json
from datetime import date
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


class _FakeMappingResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> "_FakeMappingResult":
        return self

    def all(self) -> list[dict[str, object]]:
        return list(self._rows)

    def first(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None


class _FakeRecencySession:
    def __init__(
        self,
        *,
        latest_by_feature: dict[str, object],
        coverage_by_feature: dict[str, int],
    ) -> None:
        self._latest_by_feature = latest_by_feature
        self._coverage_by_feature = coverage_by_feature

    def execute(self, statement, params=None):  # type: ignore[no-untyped-def]
        sql = str(statement)
        if "MAX(calc_date)" in sql:
            return _FakeMappingResult(
                [
                    {"feature_name": feature_name, "max_calc": calc_date}
                    for feature_name, calc_date in self._latest_by_feature.items()
                ]
            )
        if "COUNT(DISTINCT ticker)" in sql:
            feature_name = (params or {}).get("f")
            return _FakeMappingResult(
                [{"cov": int(self._coverage_by_feature.get(str(feature_name), 0))}]
            )
        raise AssertionError(f"Unexpected SQL in test fake: {sql}")


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


def test_validate_recency_treats_latest_all_null_feature_as_sparse(tmp_path: Path) -> None:
    bundle_path = _write_bundle(
        tmp_path,
        required_features=["curve_inverted_x_growth", "vol_60d"],
    )
    validator = BundleValidator(bundle_path)

    result = validator.validate_recency(
        _FakeRecencySession(
            latest_by_feature={
                "curve_inverted_x_growth": date(2026, 4, 28),
                "vol_60d": date(2026, 4, 28),
            },
            coverage_by_feature={
                "curve_inverted_x_growth": 0,   # latest date rows all NULL
                "vol_60d": 150,
            },
        ),
        max_stale_days=7,
        min_coverage_count=100,
    )

    assert result.passed is False
    assert result.stale_features == []
    assert result.sparse_features == ["curve_inverted_x_growth"]
