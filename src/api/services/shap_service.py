from __future__ import annotations

from pathlib import Path
from typing import Any

from src.api.services.greyscale_reader import GreyscaleReader

_TREE_MODELS = ("xgboost", "lightgbm")


def get_shap_for_ticker(
    ticker: str,
    *,
    report_dir: Path | str,
    top_n: int = 20,
) -> dict[str, Any] | None:
    """Per-ticker feature contribution.

    Returns SHAP-weighted attribution if any tree model produced shap_values
    (multi-model fusion legacy path), otherwise falls back to ridge linear
    attribution (coef × feature) when the wrapper persisted
    ``linear_attribution`` (W12 single-model champion path).

    Response includes ``attribution_type`` ('shap' or 'linear') so the UI
    can label the panel correctly.
    """
    reader = GreyscaleReader(report_dir=report_dir)
    report = reader.get_latest_report()
    if report is None:
        return None

    normalized_ticker = ticker.upper()
    signal_date = report.get("live_outputs", {}).get("signal_date")

    # 1. Tree-model SHAP path (W11 multi-model fusion legacy).
    shap_features = _extract_shap_attribution(
        report=report, ticker=normalized_ticker, top_n=top_n
    )
    if shap_features is not None:
        return {
            "ticker": normalized_ticker,
            "signal_date": signal_date,
            "attribution_type": "shap",
            "features": shap_features,
        }

    # 2. Ridge linear attribution path (W12 single-model champion).
    linear_features = _extract_linear_attribution(
        report=report, ticker=normalized_ticker, top_n=top_n
    )
    if linear_features is not None:
        return {
            "ticker": normalized_ticker,
            "signal_date": signal_date,
            "attribution_type": "linear",
            "features": linear_features,
        }

    return None


def _extract_shap_attribution(
    *, report: dict[str, Any], ticker: str, top_n: int
) -> list[dict[str, Any]] | None:
    shap_values = report.get("shap_values") or {}
    if not shap_values:
        return None

    live_weights = report.get("fusion", {}).get("live_weights", {})
    weighted_features: dict[str, float] = {}

    for model_name in _TREE_MODELS:
        ticker_payload = shap_values.get(model_name, {}).get(ticker)
        if not ticker_payload:
            continue
        model_weight = float(live_weights.get(model_name, 0.0))
        for feature, value in ticker_payload.get("features", {}).items():
            feature_name = str(feature)
            weighted_features[feature_name] = (
                weighted_features.get(feature_name, 0.0) + float(value) * model_weight
            )

    if not weighted_features:
        return None

    ranked = sorted(
        weighted_features.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:top_n]
    return [
        {"feature": feature, "shap_value": round(value, 6)}
        for feature, value in ranked
    ]


def _extract_linear_attribution(
    *, report: dict[str, Any], ticker: str, top_n: int
) -> list[dict[str, Any]] | None:
    payload = report.get("linear_attribution")
    if not isinstance(payload, dict):
        return None
    feature_names = payload.get("feature_names") or []
    coefficients = payload.get("coefficients") or []
    ticker_features = (payload.get("ticker_features") or {}).get(ticker)
    if (
        not feature_names
        or not coefficients
        or not ticker_features
        or len(feature_names) != len(coefficients)
        or len(feature_names) != len(ticker_features)
    ):
        return None

    contributions: list[tuple[str, float]] = []
    for name, coef, value in zip(feature_names, coefficients, ticker_features, strict=False):
        try:
            contrib = float(coef) * float(value)
        except (TypeError, ValueError):
            continue
        if not _is_finite(contrib):
            continue
        contributions.append((str(name), contrib))

    # Drop is_missing_* binary indicator flags — they're noise vs. real
    # economic factors. Drop exact-zero contributions for the same reason.
    contributions = [
        (name, value)
        for name, value in contributions
        if abs(value) > 1e-9 and not name.startswith("is_missing_")
    ]
    if not contributions:
        return None

    ranked = sorted(contributions, key=lambda item: (-abs(item[1]), item[0]))[:top_n]
    return [
        {"feature": feature, "shap_value": round(value, 6)}
        for feature, value in ranked
    ]


def _is_finite(value: float) -> bool:
    try:
        return value == value and value not in (float("inf"), float("-inf"))
    except TypeError:
        return False
