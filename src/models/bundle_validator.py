from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
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


@dataclass(frozen=True)
class RecencyValidationResult:
    """Per-feature recency + coverage check (W12 audit hardening)."""
    passed: bool
    stale_features: list[str]  # features whose latest calc_date is too old
    sparse_features: list[str]  # features with too few rows on latest calc_date
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

    def validate_recency(
        self,
        feature_store_session: Any,
        *,
        max_stale_days: int = 7,
        min_coverage_count: int = 100,
    ) -> RecencyValidationResult:
        """W12 audit hardening: per-feature recency + coverage check.

        For each required feature, verify:
        - latest calc_date is within `max_stale_days` of today
        - on its latest calc_date, at least `min_coverage_count` distinct tickers have values

        This catches silent feature dropout (e.g. shorting backfill ran once and never again),
        which the cheap fingerprint+distinct-name schema check would miss.
        """
        bundle = self.load_bundle()
        required_features = sorted(
            str(feature) for feature in (
                bundle.get("required_features") or bundle.get("retained_features") or []
            )
        )
        if not required_features:
            raise BundleSchemaError(
                f"Bundle {self.bundle_path} does not declare required_features.",
            )

        # Per-feature latest calc_date (cheap)
        latest_rows = feature_store_session.execute(
            text(
                """
                SELECT feature_name, MAX(calc_date) AS max_calc
                FROM feature_store
                WHERE feature_name = ANY(:feats)
                GROUP BY feature_name
                """
            ),
            {"feats": required_features},
        )
        latest_by_feature: dict[str, Any] = {
            row["feature_name"]: row["max_calc"]
            for row in latest_rows.mappings().all()
        }

        # Coverage on each feature's latest calc_date (one query per feature; ~62 fast queries)
        feature_data: dict[str, dict[str, Any]] = {}
        for feat, max_calc in latest_by_feature.items():
            cov_row = feature_store_session.execute(
                text(
                    "SELECT COUNT(DISTINCT ticker) AS cov "
                    "FROM feature_store "
                    "WHERE feature_name = :f AND calc_date = :d AND feature_value IS NOT NULL"
                ),
                {"f": feat, "d": max_calc},
            ).mappings().first()
            feature_data[feat] = {
                "latest_calc_date": max_calc,
                "coverage": int(cov_row["cov"]) if cov_row else 0,
            }

        today = date.today()
        threshold_date = today - timedelta(days=max_stale_days)
        stale: list[str] = []
        sparse: list[str] = []
        seen_features = set(feature_data.keys())

        for feat in required_features:
            data = feature_data.get(feat)
            if data is None:
                # Feature has no rows at all → captured by validate_schema as missing.
                # Skip here (don't double-flag).
                continue
            latest = data["latest_calc_date"]
            if latest < threshold_date:
                stale.append(feat)
            if data["coverage"] < min_coverage_count:
                sparse.append(feat)

        passed = not stale and not sparse
        return RecencyValidationResult(
            passed=passed,
            stale_features=sorted(stale),
            sparse_features=sorted(sparse),
            metadata={
                "today": today.isoformat(),
                "max_stale_days": max_stale_days,
                "min_coverage_count": min_coverage_count,
                "threshold_date": threshold_date.isoformat(),
                "feature_count_checked": len(required_features),
                "feature_count_with_data": len(seen_features),
                "per_feature_stats": {
                    name: {
                        "latest_calc_date": d["latest_calc_date"].isoformat()
                        if hasattr(d["latest_calc_date"], "isoformat") else str(d["latest_calc_date"]),
                        "coverage": d["coverage"],
                    }
                    for name, d in feature_data.items()
                },
            },
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
