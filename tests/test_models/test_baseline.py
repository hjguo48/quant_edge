from __future__ import annotations

from datetime import date

from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import DEFAULT_ALPHA_GRID, FactorRankBaselineModel, RidgeBaselineModel
from src.models.experiment import ExperimentTracker, run_single_window_validation


def test_ridge_baseline_selects_alpha_and_predicts_signal() -> None:
    X, y = _build_synthetic_panel()
    train_mask = X.index.get_level_values("date") <= pd.Timestamp(date(2020, 12, 31))
    validation_mask = (
        (X.index.get_level_values("date") >= pd.Timestamp(date(2021, 1, 1)))
        & (X.index.get_level_values("date") <= pd.Timestamp(date(2021, 6, 30)))
    )

    model = RidgeBaselineModel(alpha_grid=DEFAULT_ALPHA_GRID)
    selection = model.select_alpha(
        X_train=X.loc[train_mask],
        y_train=y.loc[train_mask],
        X_val=X.loc[validation_mask],
        y_val=y.loc[validation_mask],
    )
    model.train(X.loc[train_mask], y.loc[train_mask])
    predictions = model.predict(X.loc[validation_mask])
    metrics = model.evaluate(y.loc[validation_mask], predictions)

    assert selection.best_alpha in DEFAULT_ALPHA_GRID
    assert selection.best_ic > 0.10
    assert predictions.index.equals(X.loc[validation_mask].index)
    assert metrics.ic > 0.10


def test_factor_rank_baseline_produces_separated_rankings() -> None:
    X, y = _build_synthetic_panel()
    model = FactorRankBaselineModel(
        domain_features={
            "momentum": ["momentum_20d"],
            "volatility": ["volatility_20d"],
            "value": ["value_pe_inv"],
            "quality": ["quality_roe"],
            "macro": ["macro_yield_spread"],
        },
        feature_signs={"volatility_20d": -1},
    )
    model.train(X, y)
    predictions = model.predict(X)
    metrics = model.evaluate(y, predictions)

    assert predictions.notna().all()
    assert metrics.rank_ic > 0.10


def test_experiment_tracker_logs_model_and_filters_runs(tmp_path) -> None:
    X, y = _build_synthetic_panel()
    train_mask = X.index.get_level_values("date") <= pd.Timestamp(date(2020, 12, 31))
    model = RidgeBaselineModel(alpha=0.1).train(X.loc[train_mask], y.loc[train_mask])

    tracker = ExperimentTracker(tracking_uri=tmp_path.as_uri())
    logged = tracker.log_training_run(
        model=model,
        target_horizon="5D",
        window_id="window_01",
        metrics={"test_ic": 0.123},
        params=model.get_params(),
        timestamp="20260401_120000",
    )

    client = MlflowClient(tracking_uri=tmp_path.as_uri())
    run = client.get_run(logged.run_id)
    artifacts = client.list_artifacts(logged.run_id, path="model")
    filtered_runs = tracker.search_runs(model_type="ridge", horizon="5D")

    assert logged.experiment_name == "ridge_5D_20260401_120000"
    assert run.data.params["alpha"] == "0.1"
    assert run.data.metrics["test_ic"] == pytest.approx(0.123)
    assert run.data.tags["model_type"] == "ridge"
    assert run.data.tags["horizon"] == "5D"
    assert any(item.path == "model/model.pkl" or item.path == "model.pkl" for item in artifacts)
    assert any(item.info.run_id == logged.run_id for item in filtered_runs)


def test_single_window_validation_passes_ic_threshold(tmp_path) -> None:
    X, y = _build_synthetic_panel()
    tracker = ExperimentTracker(tracking_uri=tmp_path.as_uri())
    model = RidgeBaselineModel(alpha_grid=DEFAULT_ALPHA_GRID)

    result = run_single_window_validation(
        model=model,
        X=X,
        y=y,
        tracker=tracker,
        timestamp="20260401_121500",
    )

    assert result.best_alpha in DEFAULT_ALPHA_GRID
    assert result.train_rows > 0
    assert result.validation_rows > 0
    assert result.test_rows > 0
    assert result.validation_metrics.ic > 0.01
    assert result.test_metrics.ic > 0.01
    assert result.passed is True
    assert result.logged_run is not None


def _build_synthetic_panel(
    *,
    seed: int = 42,
    n_tickers: int = 40,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-05", "2021-12-31", freq="W-FRI")
    tickers = [f"TICKER_{i:03d}" for i in range(n_tickers)]

    feature_frames: list[pd.DataFrame] = []
    target_frames: list[pd.Series] = []

    for rebalance_date in dates:
        macro_level = rng.normal()
        frame = pd.DataFrame(
            {
                "momentum_20d": rng.normal(size=n_tickers),
                "volatility_20d": rng.normal(size=n_tickers),
                "value_pe_inv": rng.normal(size=n_tickers),
                "quality_roe": rng.normal(size=n_tickers),
                "macro_yield_spread": macro_level + rng.normal(scale=0.10, size=n_tickers),
            },
            index=pd.Index(tickers, name="ticker"),
        )
        standardized = frame.apply(_cross_sectional_zscore)
        signal = (
            0.12 * standardized["momentum_20d"]
            - 0.08 * standardized["volatility_20d"]
            + 0.10 * standardized["value_pe_inv"]
            + 0.09 * standardized["quality_roe"]
            + 0.05 * standardized["macro_yield_spread"]
        )
        noise = rng.normal(scale=0.05, size=n_tickers)
        target = pd.Series(signal.to_numpy() + noise, index=standardized.index, name="target")

        feature_frames.append(standardized.assign(date=rebalance_date).set_index("date", append=True))
        target_frames.append(target.to_frame().assign(date=rebalance_date).set_index("date", append=True)["target"])

    X = pd.concat(feature_frames).reorder_levels(["date", "ticker"]).sort_index()
    y = pd.concat(target_frames).reorder_levels(["date", "ticker"]).sort_index()
    return X, y


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std
