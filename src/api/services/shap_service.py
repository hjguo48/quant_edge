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
    reader = GreyscaleReader(report_dir=report_dir)
    report = reader.get_latest_report()
    if report is None:
        return None

    shap_values = report.get("shap_values", {})
    if not shap_values:
        return None

    normalized_ticker = ticker.upper()
    live_weights = report.get("fusion", {}).get("live_weights", {})
    weighted_features: dict[str, float] = {}

    for model_name in _TREE_MODELS:
        ticker_payload = shap_values.get(model_name, {}).get(normalized_ticker)
        if not ticker_payload:
            continue

        model_weight = float(live_weights.get(model_name, 0.0))
        for feature, value in ticker_payload.get("features", {}).items():
            feature_name = str(feature)
            weighted_features[feature_name] = weighted_features.get(feature_name, 0.0) + float(value) * model_weight

    if not weighted_features:
        return None

    ranked_features = sorted(
        weighted_features.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:top_n]
    return {
        "ticker": normalized_ticker,
        "signal_date": report.get("live_outputs", {}).get("signal_date"),
        "features": [
            {"feature": feature, "shap_value": round(value, 6)}
            for feature, value in ranked_features
        ],
    }
