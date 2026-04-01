from __future__ import annotations

from datetime import date

from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import RidgeBaselineModel
from src.models.experiment import ExperimentTracker, run_single_window_validation
from src.models.tree import (
    LightGBMModel,
    XGBoostModel,
    compare_model_metrics,
    feature_importance_frame,
    zero_contribution_features,
)

SMALL_XGB_SEARCH_SPACE = {
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [20, 40],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [5, 10],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [0.1, 1.0],
}

SMALL_LGBM_SEARCH_SPACE = {
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [20, 40],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_samples": [20, 50],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [0.1, 1.0],
}


@pytest.mark.parametrize(
    ("model_cls", "search_space"),
    [
        (XGBoostModel, SMALL_XGB_SEARCH_SPACE),
        (LightGBMModel, SMALL_LGBM_SEARCH_SPACE),
    ],
)
def test_tree_models_train_predict_and_expose_feature_importances(
    model_cls,
    search_space,
) -> None:
    X, y = _build_tree_panel()
    train_mask = X.index.get_level_values("date") <= pd.Timestamp(date(2020, 12, 31))
    validation_mask = (
        (X.index.get_level_values("date") >= pd.Timestamp(date(2021, 1, 1)))
        & (X.index.get_level_values("date") <= pd.Timestamp(date(2021, 6, 30)))
    )

    model = model_cls(search_n_iter=2, n_jobs=1)
    model.select_hyperparameters(
        X_train=X.loc[train_mask],
        y_train=y.loc[train_mask],
        X_val=X.loc[validation_mask],
        y_val=y.loc[validation_mask],
        n_iter=2,
        search_space=search_space,
    )
    model.train(X.loc[train_mask], y.loc[train_mask])
    predictions = model.predict(X.loc[validation_mask])
    metrics = model.evaluate(y.loc[validation_mask], predictions)
    importance_data = feature_importance_frame(model)

    assert predictions.index.equals(X.loc[validation_mask].index)
    assert metrics.ic > 0.02
    assert model.feature_importances_.index.tolist() == list(X.columns)
    assert importance_data.iloc[0]["importance"] >= importance_data.iloc[-1]["importance"]
    assert "constant_feature" in zero_contribution_features(model)


def test_xgboost_search_logs_trials_and_best_run_to_mlflow(tmp_path) -> None:
    X, y = _build_tree_panel()
    tracker = ExperimentTracker(tracking_uri=tmp_path.as_uri())
    model = XGBoostModel(search_n_iter=2, n_jobs=1)
    model.search_space = SMALL_XGB_SEARCH_SPACE

    result = run_single_window_validation(
        model=model,
        X=X,
        y=y,
        tracker=tracker,
        timestamp="20260401_160000",
    )

    client = MlflowClient(tracking_uri=tmp_path.as_uri())
    runs = tracker.search_runs(model_type="xgboost", horizon="5D", max_results=10)
    search_runs = [run for run in runs if run.data.tags.get("run_kind") == "search_trial"]
    training_runs = [run for run in runs if run.data.tags.get("run_kind") == "training"]

    assert isinstance(result.best_alpha, dict)
    assert len(search_runs) == 2
    assert len(training_runs) == 1
    assert result.logged_run is not None

    training_run = client.get_run(result.logged_run.run_id)
    artifacts = client.list_artifacts(result.logged_run.run_id, path="model")
    assert training_run.data.metrics["validation_ic"] > 0.02
    assert "test_top_decile_return" in training_run.data.metrics
    assert any(item.path == "model/model.pkl" or item.path == "model.pkl" for item in artifacts)


def test_compare_model_metrics_returns_improvement_report() -> None:
    X, y = _build_tree_panel()
    train_mask = X.index.get_level_values("date") <= pd.Timestamp(date(2020, 12, 31))
    test_mask = X.index.get_level_values("date") >= pd.Timestamp(date(2021, 7, 1))

    baseline = RidgeBaselineModel(alpha=0.1).train(X.loc[train_mask], y.loc[train_mask])
    baseline_predictions = baseline.predict(X.loc[test_mask])
    baseline_metrics = baseline.evaluate(y.loc[test_mask], baseline_predictions)

    tree_model = LightGBMModel(n_jobs=1, n_estimators=40, max_depth=4, learning_rate=0.1)
    tree_model.train(X.loc[train_mask], y.loc[train_mask])
    tree_predictions = tree_model.predict(X.loc[test_mask])
    tree_metrics = tree_model.evaluate(y.loc[test_mask], tree_predictions)

    report = compare_model_metrics(baseline_metrics, tree_metrics)

    assert list(report.columns) == ["baseline", "tree", "delta", "pct_change"]
    assert set(report.index) == {"ic", "rank_ic", "icir", "hit_rate", "top_decile_return"}
    assert report.loc["ic", "tree"] == pytest.approx(tree_metrics.ic)


def _build_tree_panel(
    *,
    seed: int = 7,
    n_tickers: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-05", "2021-12-31", freq="W-FRI")
    tickers = [f"TICKER_{i:03d}" for i in range(n_tickers)]

    feature_frames: list[pd.DataFrame] = []
    target_frames: list[pd.Series] = []

    for rebalance_date in dates:
        macro_state = rng.normal()
        base = pd.DataFrame(
            {
                "momentum_20d": rng.normal(size=n_tickers),
                "quality_roe": rng.normal(size=n_tickers),
                "value_pe_inv": rng.normal(size=n_tickers),
                "volatility_20d": rng.normal(size=n_tickers),
                "macro_yield_spread": macro_state + rng.normal(scale=0.10, size=n_tickers),
                "noise_feature": rng.normal(size=n_tickers),
                "constant_feature": np.zeros(n_tickers, dtype=float),
            },
            index=pd.Index(tickers, name="ticker"),
        )
        standardized = base.apply(_cross_sectional_zscore)
        interaction = standardized["momentum_20d"] * standardized["quality_roe"]
        regime = (
            (standardized["value_pe_inv"] > 0).astype(float)
            * (standardized["macro_yield_spread"] > 0).astype(float)
        )
        target = (
            0.10 * standardized["momentum_20d"]
            + 0.08 * standardized["quality_roe"]
            - 0.05 * standardized["volatility_20d"]
            + 0.07 * standardized["value_pe_inv"]
            + 0.04 * standardized["macro_yield_spread"]
            + 0.18 * interaction
            + 0.10 * regime
            + rng.normal(scale=0.04, size=n_tickers)
        )

        feature_frames.append(standardized.assign(date=rebalance_date).set_index("date", append=True))
        target_frames.append(
            pd.Series(target, index=standardized.index, name="target")
            .to_frame()
            .assign(date=rebalance_date)
            .set_index("date", append=True)["target"],
        )

    X = pd.concat(feature_frames).reorder_levels(["date", "ticker"]).sort_index()
    y = pd.concat(target_frames).reorder_levels(["date", "ticker"]).sort_index()
    return X, y


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std
