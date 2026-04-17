from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text


class BundleSchemaError(RuntimeError):
    """Raised when a live bundle cannot be safely executed."""


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    missing_features: list[str]
    extra_features: list[str]
    metadata: dict[str, Any]


class BundleValidator:
    """Validate that a live bundle can be served from feature_store."""

    def __init__(self, bundle_path: str | Path) -> None:
        self.bundle_path = Path(bundle_path)

    def load_bundle(self) -> dict[str, Any]:
        return json.loads(self.bundle_path.read_text(encoding="utf-8"))

    def compute_fingerprint(self, bundle_dict: dict[str, Any]) -> str:
        version = str(bundle_dict.get("version") or bundle_dict.get("model_bundle_version") or "")
        cutoff_date = str(bundle_dict.get("cutoff_date") or "")
        required_features = sorted(
            str(feature)
            for feature in (
                bundle_dict.get("required_features")
                or bundle_dict.get("retained_features")
                or []
            )
        )
        payload = {
            "cutoff_date": cutoff_date,
            "required_features": required_features,
            "version": version,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()

    def validate_schema(self, feature_store_session: Any) -> ValidationResult:
        bundle = self.load_bundle()
        required_features = sorted(
            str(feature)
            for feature in (
                bundle.get("required_features")
                or bundle.get("retained_features")
                or []
            )
        )
        if not required_features:
            raise BundleSchemaError(
                f"Bundle {self.bundle_path} does not declare required_features or retained_features.",
            )

        rows = feature_store_session.execute(
            text("select distinct feature_name from feature_store order by feature_name"),
        )
        available_features = sorted(str(feature) for feature in rows.scalars().all())
        available_set = set(available_features)
        required_set = set(required_features)

        missing_features = sorted(required_set - available_set)
        extra_features = sorted(available_set - required_set)

        expected_fingerprint = str(bundle.get("feature_fingerprint") or "")
        computed_fingerprint = self.compute_fingerprint(bundle)
        fingerprint_matches = (not expected_fingerprint) or expected_fingerprint == computed_fingerprint

        metadata = {
            "artifact_path": str(self.bundle_path),
            "computed_fingerprint": computed_fingerprint,
            "cutoff_date": bundle.get("cutoff_date"),
            "expected_fingerprint": expected_fingerprint or None,
            "fingerprint_matches": fingerprint_matches,
            "generated_at": bundle.get("generated_at") or bundle.get("generated_at_utc"),
            "generator_git_hash": bundle.get("generator_git_hash"),
            "model_type": bundle.get("model_type"),
            "required_feature_count": len(required_features),
            "version": bundle.get("version") or bundle.get("model_bundle_version"),
        }

        passed = not missing_features and fingerprint_matches
        return ValidationResult(
            passed=passed,
            missing_features=missing_features,
            extra_features=extra_features,
            metadata=metadata,
        )

    def assert_valid(self, feature_store_session: Any) -> ValidationResult:
        result = self.validate_schema(feature_store_session)
        if result.passed:
            return result

        problems: list[str] = []
        if result.missing_features:
            problems.append(
                "missing required features in feature_store: "
                + ", ".join(result.missing_features),
            )
        if not bool(result.metadata.get("fingerprint_matches", True)):
            problems.append(
                "bundle fingerprint mismatch "
                f"(expected={result.metadata.get('expected_fingerprint')}, "
                f"computed={result.metadata.get('computed_fingerprint')})",
            )
        if not problems:
            problems.append("bundle schema validation failed")
        raise BundleSchemaError("; ".join(problems))
