from __future__ import annotations

from datetime import date
import json
import math
import pickle
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mlflow.tracking import MlflowClient
import pandas as pd

from scripts.run_live_ic_validation import load_champion_model, resolve_source_model_pickle_path
from scripts.run_single_window_validation import align_panel
from src.labels.forward_returns import compute_forward_returns
from src.models.champion_challenger import ChampionChallengerRunner
from src.models.evaluation import information_coefficient_series
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.equal_weight import equal_weight_portfolio
from src.backtest.execution import select_candidate_tickers

DEFAULT_MODEL_NAME = "ridge_60d"
BENCHMARK_TICKER = "SPY"
DEFAULT_SIGNAL_FUSION_REPORT_PATH = "data/reports/signal_fusion_experiment.json"
DEFAULT_HOLDING_PERIOD_REPORT_PATH = "data/reports/holding_period_experiment.json"
DEFAULT_SHORT_HORIZON_MODELS_REPORT_PATH = "data/reports/short_horizon_models.json"
SUPPORTED_FUSION_METHODS = {"equal_weight", "ic_weighted", "recursive_overlay"}


def load_live_strategy_config(
    *,
    repo_root: Path,
    champion: Any,
    registry_tracking_uri: str,
) -> dict[str, Any]:
    signal_fusion_report = _load_json(repo_root / DEFAULT_SIGNAL_FUSION_REPORT_PATH)
    holding_period_report = _load_json(repo_root / DEFAULT_HOLDING_PERIOD_REPORT_PATH)
    short_horizon_models_report = _load_json(repo_root / DEFAULT_SHORT_HORIZON_MODELS_REPORT_PATH)

    recommended_method = str(
        ((signal_fusion_report.get("final_recommendation") or {}).get("recommended_method"))
        or ((signal_fusion_report.get("best_configuration") or {}).get("label"))
        or "equal_weight"
    )
    if recommended_method not in SUPPORTED_FUSION_METHODS:
        raise RuntimeError(f"Unsupported live fusion method {recommended_method!r}.")

    base_scheme = dict(
        (holding_period_report.get("base_scheme") or short_horizon_models_report.get("scheme_config") or {}),
    )
    recommended_configuration = dict(holding_period_report.get("recommended_configuration") or {})
    selection_pct = float(base_scheme.get("selection_pct", 0.20))
    sell_buffer_raw = (
        recommended_configuration.get("sell_buffer_pct")
        if recommended_configuration.get("sell_buffer_pct") is not None
        else base_scheme.get("sell_buffer_pct")
    )
    sell_buffer_pct = float(sell_buffer_raw) if sell_buffer_raw is not None else None

    components: dict[str, dict[str, Any]] = {
        "60D": {
            "source": "registry_champion",
            "model_name": champion.name,
            "version": int(champion.version),
            "stage": champion.stage.value,
            "run_id": champion.run_id,
            "tracking_uri": registry_tracking_uri,
            "horizon": str(champion.metadata.horizon),
            "features": list(champion.metadata.features),
            "selected_feature_count": int(champion.metadata.n_features),
            "aggregate_mean_ic": float(champion.metadata.metrics.get("mean_oos_ic", 0.0)),
        },
    }

    for horizon in ("20D", "10D"):
        model_payload = dict((short_horizon_models_report.get("models") or {}).get(horizon) or {})
        if not model_payload:
            raise RuntimeError(f"Missing short-horizon report payload for {horizon}.")
        walkforward_row = _select_latest_walkforward_row(model_payload.get("walkforward") or [])
        mlflow_payload = dict(walkforward_row.get("mlflow") or {})
        selected_features = list(model_payload.get("selected_features") or [])
        if not selected_features:
            raise RuntimeError(f"Short-horizon live config for {horizon} has no selected features.")
        run_id = str(mlflow_payload.get("run_id") or "")
        if not run_id:
            raise RuntimeError(f"Short-horizon live config for {horizon} is missing an mlflow run_id.")
        components[horizon] = {
            "source": "short_horizon_latest_walkforward_run",
            "model_name": f"ridge_{horizon.lower()}",
            "run_id": run_id,
            "tracking_uri": resolve_runtime_tracking_uri(
                str(mlflow_payload.get("tracking_uri") or registry_tracking_uri),
                repo_root=repo_root,
            ),
            "experiment_name": mlflow_payload.get("experiment_name"),
            "horizon": horizon,
            "features": selected_features,
            "selected_feature_count": int(model_payload.get("selected_feature_count") or len(selected_features)),
            "aggregate_mean_ic": float((model_payload.get("aggregate") or {}).get("mean_ic") or 0.0),
            "test_period": walkforward_row.get("test_period"),
            "best_hyperparams": float(walkforward_row.get("best_hyperparams") or 0.0),
        }

    all_features = _unique_preserve_order(
        feature_name
        for horizon in ("60D", "20D", "10D")
        for feature_name in components[horizon]["features"]
    )
    fusion_method_payload = dict(((signal_fusion_report.get("fusion_methods") or {}).get(recommended_method)) or {})

    return {
        "strategy_name": f"{recommended_method}_fusion",
        "signal_label": _signal_label(recommended_method),
        "method": recommended_method,
        "holding_period": str(recommended_configuration.get("holding_period") or "4W"),
        "rebalance_schedule": "monthly_first_signal",
        "selection_pct": selection_pct,
        "sell_buffer_pct": sell_buffer_pct,
        "min_trade_weight": float(base_scheme.get("min_trade_weight", 0.01)),
        "max_weight": float(base_scheme.get("max_weight", 0.05)),
        "min_holdings": int(base_scheme.get("min_holdings", 20)),
        "components": components,
        "all_features": all_features,
        "method_weights": dict(fusion_method_payload.get("weights") or {}),
        "report_paths": {
            "signal_fusion": DEFAULT_SIGNAL_FUSION_REPORT_PATH,
            "holding_period": DEFAULT_HOLDING_PERIOD_REPORT_PATH,
            "short_horizon_models": DEFAULT_SHORT_HORIZON_MODELS_REPORT_PATH,
        },
    }


def load_live_strategy_models(
    *,
    repo_root: Path,
    strategy_config: dict[str, Any],
    registry: Any,
) -> dict[str, dict[str, Any]]:
    champion = registry.get_champion(DEFAULT_MODEL_NAME)
    if champion is None or champion.metadata is None:
        raise RuntimeError(f"No champion model with metadata is registered for {DEFAULT_MODEL_NAME!r}.")

    base_model, base_audit = load_champion_model(registry=registry, champion=champion)
    models: dict[str, dict[str, Any]] = {
        "60D": {
            "model": base_model,
            "load_audit": base_audit,
        },
    }

    for horizon in ("20D", "10D"):
        spec = strategy_config["components"][horizon]
        model, load_audit = load_pickled_run_model(
            repo_root=repo_root,
            tracking_uri=str(spec["tracking_uri"]),
            run_id=str(spec["run_id"]),
            horizon=horizon,
        )
        models[horizon] = {
            "model": model,
            "load_audit": load_audit,
        }

    return models


def load_pickled_run_model(
    *,
    repo_root: Path,
    tracking_uri: str,
    run_id: str,
    horizon: str,
) -> tuple[Any, dict[str, Any]]:
    resolved_tracking_uri = resolve_runtime_tracking_uri(tracking_uri, repo_root=repo_root)
    source_client = MlflowClient(tracking_uri=resolved_tracking_uri)
    resolution_error: Exception | None = None
    try:
        local_model_path, artifact_resolution = resolve_source_model_pickle_path(
            source_client=source_client,
            source_run_id=run_id,
            source_tracking_uri=resolved_tracking_uri,
        )
    except Exception as exc:
        resolution_error = exc
        run = source_client.get_run(run_id)
        direct_model_path = _resolve_model_path_from_artifact_uri(
            artifact_uri=str(run.info.artifact_uri),
            repo_root=repo_root,
        )
        if direct_model_path is None:
            raise
        local_model_path = str(direct_model_path)
        artifact_resolution = "direct_artifact_uri_lookup"
    with open(local_model_path, "rb") as handle:
        model = pickle.load(handle)
    return model, {
        "method": "source_run_pickle_fallback",
        "tracking_uri": tracking_uri,
        "resolved_tracking_uri": resolved_tracking_uri,
        "run_id": run_id,
        "horizon": horizon,
        "artifact_path": "model/model.pkl",
        "artifact_resolution": artifact_resolution,
        "local_model_path": local_model_path,
        "resolution_error": str(resolution_error) if resolution_error is not None else None,
    }


def predict_component_scores(
    *,
    feature_matrix: pd.DataFrame,
    strategy_config: dict[str, Any],
    models: dict[str, dict[str, Any]],
) -> dict[str, pd.Series]:
    if feature_matrix.empty:
        raise RuntimeError("Feature matrix is empty; cannot compute live strategy scores.")

    component_scores: dict[str, pd.Series] = {}
    for horizon in ("60D", "20D", "10D"):
        features = list(strategy_config["components"][horizon]["features"])
        model_matrix = feature_matrix.reindex(columns=features)
        if model_matrix.empty:
            raise RuntimeError(f"Feature matrix is empty for {horizon} model columns.")
        predictions = ChampionChallengerRunner._predict_series(models[horizon]["model"], model_matrix)
        component_scores[horizon] = pd.Series(predictions, index=model_matrix.index, dtype=float).sort_index()
    return component_scores


def combine_component_scores(
    *,
    component_scores: dict[str, pd.Series],
    strategy_config: dict[str, Any],
) -> pd.Series:
    ranked_60d = cross_sectional_rank(component_scores["60D"]).rename("60D")
    ranked_20d = cross_sectional_rank(component_scores["20D"]).rename("20D")
    ranked_10d = cross_sectional_rank(component_scores["10D"]).rename("10D")
    frame = pd.concat([ranked_60d, ranked_20d, ranked_10d], axis=1).fillna(0.5)

    method = str(strategy_config["method"])
    if method == "equal_weight":
        score = frame.mean(axis=1)
    elif method == "ic_weighted":
        weights = dict(strategy_config.get("method_weights") or {})
        score = (
            float(weights.get("60D", 0.0)) * frame["60D"]
            + float(weights.get("20D", 0.0)) * frame["20D"]
            + float(weights.get("10D", 0.0)) * frame["10D"]
        )
    elif method == "recursive_overlay":
        overlay = 0.5 * (frame["20D"] + frame["10D"])
        score = frame["60D"] + 0.35 * (overlay - 0.5)
    else:
        raise RuntimeError(f"Unsupported fusion method {method!r}.")

    return cross_sectional_rank(score.rename("score")).sort_index()


def build_prediction_snapshot_frame(
    *,
    signal_date: str,
    component_scores: dict[str, pd.Series],
    fused_scores: pd.Series,
) -> pd.DataFrame:
    signal_ts = pd.Timestamp(signal_date)
    current_fused = _select_signal_slice(fused_scores, signal_ts)
    if current_fused.empty:
        raise RuntimeError(f"No fused scores are available for signal date {signal_date}.")

    score_60d = _select_signal_slice(component_scores["60D"], signal_ts)
    score_20d = _select_signal_slice(component_scores["20D"], signal_ts)
    score_10d = _select_signal_slice(component_scores["10D"], signal_ts)
    rank_60d = cross_sectional_rank(score_60d)
    rank_20d = cross_sectional_rank(score_20d)
    rank_10d = cross_sectional_rank(score_10d)
    tickers = current_fused.index.astype(str)

    frame = pd.DataFrame(
        {
            "window_id": [f"LIVE_{signal_date.replace('-', '')}"] * len(current_fused),
            "trade_date": [signal_date] * len(current_fused),
            "ticker": tickers,
            "score": current_fused.to_numpy(dtype=float),
            "score_60d": score_60d.reindex(tickers).to_numpy(dtype=float),
            "score_20d": score_20d.reindex(tickers).to_numpy(dtype=float),
            "score_10d": score_10d.reindex(tickers).to_numpy(dtype=float),
            "rank_60d": rank_60d.reindex(tickers).to_numpy(dtype=float),
            "rank_20d": rank_20d.reindex(tickers).to_numpy(dtype=float),
            "rank_10d": rank_10d.reindex(tickers).to_numpy(dtype=float),
        },
    )
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def build_signal_risk_inputs_from_predictions(
    *,
    prediction_series: pd.Series,
    prices: pd.DataFrame,
    horizon_days: int,
    min_cross_section: int,
) -> dict[str, Any]:
    from scripts.run_live_ic_validation import filter_minimum_cross_sections

    price_frame = prices.copy()
    price_frame["trade_date"] = pd.to_datetime(price_frame["trade_date"])
    labels = compute_forward_returns(
        prices_df=price_frame,
        horizons=(horizon_days,),
        benchmark_ticker=BENCHMARK_TICKER,
    )
    labels = labels.loc[
        (labels["horizon"] == horizon_days)
        & (labels["ticker"].astype(str).str.upper() != BENCHMARK_TICKER)
    ].copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    label_series = (
        labels.set_index(["trade_date", "ticker"])["excess_return"]
        .sort_index()
        .dropna()
    )

    aligned_predictions, aligned_y = align_panel(
        prediction_series.sort_index(),
        label_series,
    )
    filtered_y, filtered_pred, sizes = filter_minimum_cross_sections(
        y=aligned_y,
        y_pred=aligned_predictions,
        min_size=min_cross_section,
    )
    ic_series = information_coefficient_series(y_true=filtered_y, y_pred=filtered_pred)

    return {
        "predicted_scores": filtered_pred,
        "realized_returns": filtered_y,
        "ic_history": [float(value) for value in ic_series.tolist()],
        "champion_ic": float(ic_series.mean()) if len(ic_series) else 0.0,
        "challenger_ic": float(ic_series.mean()) if len(ic_series) else 0.0,
        "challenger_available": False,
        "history_dates": int(len(sizes)),
        "history_rows": int(len(filtered_y)),
    }


def should_rebalance(
    *,
    signal_date: str,
    holding_period: str,
    last_rebalance_signal_date: str | None,
) -> bool:
    if last_rebalance_signal_date is None:
        return True

    current = pd.Timestamp(signal_date)
    previous = pd.Timestamp(last_rebalance_signal_date)
    if holding_period == "1W":
        return True
    if holding_period == "2W":
        return (current - previous).days >= 14
    if holding_period == "4W":
        return (current.year, current.month) != (previous.year, previous.month)
    if holding_period == "8W":
        month_delta = (current.year - previous.year) * 12 + (current.month - previous.month)
        return month_delta >= 2
    raise RuntimeError(f"Unsupported holding period {holding_period!r}.")


def build_buffered_target_weights(
    *,
    scores: pd.Series,
    current_weights: dict[str, float],
    strategy_config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    ranked_scores = pd.Series(scores, dtype=float).dropna().sort_values(ascending=False)
    if ranked_scores.empty:
        return {}, {
            "candidate_count": 0,
            "retained_count": 0,
            "retained_from_buffer": [],
        }

    ranking = ranked_scores.index.astype(str).tolist()
    candidate_tickers = select_candidate_tickers(
        ranking=ranking,
        current_weights=current_weights,
        selection_pct=float(strategy_config["selection_pct"]),
        sell_buffer_pct=strategy_config.get("sell_buffer_pct"),
        min_holdings=int(strategy_config["min_holdings"]),
        max_weight=float(strategy_config["max_weight"]),
    )
    candidate_scores = ranked_scores.reindex(candidate_tickers).dropna()
    constraints = PortfolioConstraints(
        max_weight=float(strategy_config["max_weight"]),
        min_holdings=int(strategy_config["min_holdings"]),
        turnover_buffer=float(strategy_config["min_trade_weight"]),
    )
    target_weights = equal_weight_portfolio(
        candidate_scores,
        n_stocks=len(candidate_scores),
        selection_pct=1.0,
        constraints=constraints,
    )

    min_by_cap = int(math.ceil(1.0 / max(float(strategy_config["max_weight"]), 1e-12)))
    entry_count = max(
        int(strategy_config["min_holdings"]),
        min_by_cap,
        int(math.ceil(len(ranking) * float(strategy_config["selection_pct"]))),
    )
    entry_count = min(len(ranking), entry_count)
    entry_set = set(ranking[:entry_count])
    retained_from_buffer = [
        ticker
        for ticker in candidate_tickers
        if ticker in current_weights and ticker not in entry_set
    ]
    return target_weights, {
        "candidate_count": int(len(candidate_tickers)),
        "retained_count": int(len(retained_from_buffer)),
        "retained_from_buffer": retained_from_buffer,
    }


def cross_sectional_rank(scores: pd.Series) -> pd.Series:
    series = pd.Series(scores, dtype=float).dropna().sort_index()
    if not isinstance(series.index, pd.MultiIndex):
        return series.rank(pct=True, method="average")
    return series.groupby(level="trade_date", sort=True).rank(pct=True, method="average")


def normalize_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    return {
        str(ticker): float(weight)
        for ticker, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    }


def resolve_runtime_tracking_uri(tracking_uri: str, *, repo_root: Path) -> str:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"file", "sqlite"}:
        return tracking_uri

    original_path = _tracking_uri_path(tracking_uri)
    if original_path is None or original_path.exists():
        return tracking_uri

    candidates: list[Path] = []
    repo_index = _find_repo_root_index(original_path.parts, repo_root=repo_root)
    if repo_index is not None:
        suffix = original_path.parts[repo_index + 1 :]
        if suffix:
            candidates.append(repo_root.joinpath(*suffix))
    candidates.append(repo_root / original_path.name)

    for candidate in candidates:
        if candidate.exists():
            if parsed.scheme == "file":
                return candidate.as_uri()
            return f"sqlite:///{candidate.as_posix()}"
    return tracking_uri


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _signal_label(method: str) -> str:
    labels = {
        "equal_weight": "equal_weight_fusion (60D + 10D + 20D Ridge)",
        "ic_weighted": "ic_weighted_fusion (60D + 10D + 20D Ridge)",
        "recursive_overlay": "recursive_overlay_fusion (60D base + 10D/20D timing overlay)",
    }
    return labels.get(method, method)


def _select_latest_walkforward_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Short-horizon walkforward rows are empty.")
    return max(
        rows,
        key=lambda row: (
            _parse_period_end(row.get("test_period")) or date.min,
            str((row.get("mlflow") or {}).get("run_id") or ""),
        ),
    )


def _parse_period_end(period: Any) -> date | None:
    if "->" not in str(period):
        return None
    _, end_raw = [value.strip() for value in str(period).split("->", maxsplit=1)]
    try:
        return date.fromisoformat(end_raw)
    except ValueError:
        return None


def _unique_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _select_signal_slice(series: pd.Series, signal_ts: pd.Timestamp) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        result = series.xs(signal_ts, level="trade_date").astype(float)
        result.index = result.index.astype(str)
        return result.sort_values(ascending=False)
    return pd.Series(series, dtype=float).sort_values(ascending=False)


def _tracking_uri_path(tracking_uri: str) -> Path | None:
    parsed = urlparse(tracking_uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme == "sqlite" and tracking_uri.startswith("sqlite:///"):
        return Path(tracking_uri[len("sqlite:///") :])
    return None


def _resolve_model_path_from_artifact_uri(*, artifact_uri: str, repo_root: Path) -> Path | None:
    resolved_artifact_uri = resolve_runtime_tracking_uri(artifact_uri, repo_root=repo_root)
    artifact_root = _tracking_uri_path(resolved_artifact_uri)
    if artifact_root is None:
        return None
    model_path = artifact_root / "model" / "model.pkl"
    if model_path.exists() and model_path.is_file():
        return model_path
    return None


def _find_repo_root_index(parts: tuple[str, ...], *, repo_root: Path) -> int | None:
    normalized_repo_name = _normalize_repo_name(repo_root.name)
    for index, part in enumerate(parts):
        if _normalize_repo_name(part) == normalized_repo_name:
            return index
    return None


def _normalize_repo_name(value: str) -> str:
    return value.replace("_", "").replace("-", "").lower()
