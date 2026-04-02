from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.execution import PortfolioBacktestResult, simulate_top_decile_portfolio
from src.data.db.models import UniverseMembership
from src.data.db.session import get_session_factory
from src.models.baseline import DEFAULT_ALPHA_GRID, RidgeBaselineModel
from src.models.evaluation import EvaluationSummary
from src.models.experiment import LoggedModelRun


@dataclass(frozen=True)
class WalkForwardWindowConfig:
    window_id: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    rebalance_weekday: int = 4


@dataclass(frozen=True)
class WindowModelResult:
    window_id: str
    target_horizon: str
    best_hyperparams: float
    train_metrics: EvaluationSummary
    validation_metrics: EvaluationSummary
    test_metrics: EvaluationSummary
    train_rows: int
    validation_rows: int
    test_rows: int
    mlflow_run: LoggedModelRun | None
    portfolio: PortfolioBacktestResult | None
    test_predictions: pd.Series


class WalkForwardEngine:
    def __init__(
        self,
        *,
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
        benchmark_ticker: str = "SPY",
        tracking_uri: str | None = None,
    ) -> None:
        self.alpha_grid = tuple(float(value) for value in alpha_grid)
        self.benchmark_ticker = benchmark_ticker.upper()
        self.tracking_uri = tracking_uri

    def run_window(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.DataFrame | None,
        window: WalkForwardWindowConfig,
        target_horizon: str,
        window_id: str | None = None,
        timestamp: str | None = None,
        cost_model: AlmgrenChrissCostModel | None = None,
        universe_by_date: Mapping[pd.Timestamp, set[str]] | None = None,
        simulate_portfolio: bool = True,
    ) -> WindowModelResult:
        train_X, train_y = slice_panel(
            X=X,
            y=y,
            start_date=window.train_start,
            end_date=window.train_end,
            rebalance_weekday=window.rebalance_weekday,
        )
        validation_X, validation_y = slice_panel(
            X=X,
            y=y,
            start_date=window.validation_start,
            end_date=window.validation_end,
            rebalance_weekday=window.rebalance_weekday,
        )
        test_X, test_y = slice_panel(
            X=X,
            y=y,
            start_date=window.test_start,
            end_date=window.test_end,
            rebalance_weekday=window.rebalance_weekday,
        )

        selection_model = RidgeBaselineModel(alpha_grid=self.alpha_grid)
        selection = selection_model.select_alpha(train_X, train_y, validation_X, validation_y)

        train_model = RidgeBaselineModel(alpha=selection.best_hyperparams, alpha_grid=self.alpha_grid)
        train_model.train(train_X, train_y)
        train_predictions = train_model.predict(train_X)
        validation_predictions = train_model.predict(validation_X)
        train_metrics = train_model.evaluate(train_y, train_predictions)
        validation_metrics = train_model.evaluate(validation_y, validation_predictions)

        final_train_X = pd.concat([train_X, validation_X]).sort_index()
        final_train_y = pd.concat([train_y, validation_y]).sort_index()
        final_model = RidgeBaselineModel(alpha=selection.best_hyperparams, alpha_grid=self.alpha_grid)
        final_model.train(final_train_X, final_train_y)
        test_predictions = final_model.predict(test_X)
        test_metrics = final_model.evaluate(test_y, test_predictions)

        effective_window_id = window_id or window.window_id
        logged_run = final_model.log_training_run(
            target_horizon=target_horizon,
            window_id=effective_window_id,
            metrics={
                **prefixed_metrics(train_metrics, "train"),
                **prefixed_metrics(validation_metrics, "validation"),
                **prefixed_metrics(test_metrics, "test"),
            },
            tracking_uri=self.tracking_uri,
            timestamp=timestamp,
            extra_params={
                "best_hyperparams": selection.best_hyperparams,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "rebalance_weekday": window.rebalance_weekday,
                "alpha_grid": ",".join(str(value) for value in self.alpha_grid),
            },
        )

        portfolio: PortfolioBacktestResult | None = None
        if simulate_portfolio:
            if prices is None:
                raise ValueError("prices are required when simulate_portfolio=True")
            portfolio = simulate_top_decile_portfolio(
                predictions=test_predictions,
                prices=prices,
                cost_model=cost_model or AlmgrenChrissCostModel(),
                benchmark_ticker=self.benchmark_ticker,
                universe_by_date=universe_by_date,
            )

        logger.info(
            "completed window {} horizon={} alpha={} validation_ic={:.6f} test_ic={:.6f}",
            effective_window_id,
            target_horizon,
            selection.best_hyperparams,
            validation_metrics.ic,
            test_metrics.ic,
        )
        return WindowModelResult(
            window_id=effective_window_id,
            target_horizon=target_horizon,
            best_hyperparams=float(selection.best_hyperparams),
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            train_rows=int(len(train_X)),
            validation_rows=int(len(validation_X)),
            test_rows=int(len(test_X)),
            mlflow_run=logged_run,
            portfolio=portfolio,
            test_predictions=test_predictions,
        )


def build_universe_by_date(
    *,
    trade_dates: Sequence[pd.Timestamp | date],
    index_name: str = "SP500",
) -> dict[pd.Timestamp, set[str]]:
    normalized_dates = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values().unique()
    if normalized_dates.empty:
        return {}

    min_date = normalized_dates.min().date()
    max_date = normalized_dates.max().date()
    statement = sa.select(
        UniverseMembership.ticker,
        UniverseMembership.effective_date,
        UniverseMembership.end_date,
    ).where(
        UniverseMembership.index_name == index_name,
        UniverseMembership.effective_date <= max_date,
        sa.or_(
            UniverseMembership.end_date.is_(None),
            UniverseMembership.end_date >= min_date,
        ),
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    universe_by_date: dict[pd.Timestamp, set[str]] = {}
    for trade_date in normalized_dates:
        active: set[str] = set()
        trade_day = pd.Timestamp(trade_date).date()
        for row in rows:
            if row["effective_date"] <= trade_day and (row["end_date"] is None or row["end_date"] > trade_day):
                active.add(str(row["ticker"]).upper())
        universe_by_date[pd.Timestamp(trade_date)] = active

    return universe_by_date


def slice_panel(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    start_date: date,
    end_date: date,
    rebalance_weekday: int,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_dates = pd.to_datetime(pd.Index(X.index.get_level_values("trade_date")))
    target_dates = pd.to_datetime(pd.Index(y.index.get_level_values("trade_date")))

    feature_mask = (
        (feature_dates >= pd.Timestamp(start_date))
        & (feature_dates <= pd.Timestamp(end_date))
        & (feature_dates.weekday == rebalance_weekday)
    )
    target_mask = (
        (target_dates >= pd.Timestamp(start_date))
        & (target_dates <= pd.Timestamp(end_date))
        & (target_dates.weekday == rebalance_weekday)
    )

    sliced_X = X.loc[feature_mask].sort_index()
    sliced_y = y.loc[target_mask].sort_index()
    aligned_index = sliced_X.index.intersection(sliced_y.index)
    if aligned_index.empty:
        raise RuntimeError(f"No aligned rows in window {start_date} -> {end_date}.")
    return sliced_X.loc[aligned_index], sliced_y.loc[aligned_index]


def align_panel(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    aligned_index = X.index.intersection(y.index)
    if aligned_index.empty:
        raise RuntimeError("No aligned observations between features and labels.")
    return X.loc[aligned_index].sort_index(), y.loc[aligned_index].sort_index()


def prefixed_metrics(metrics: EvaluationSummary, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_ic": float(metrics.ic),
        f"{prefix}_rank_ic": float(metrics.rank_ic),
        f"{prefix}_icir": float(metrics.icir),
        f"{prefix}_hit_rate": float(metrics.hit_rate),
        f"{prefix}_top_decile_return": float(metrics.top_decile_return),
        f"{prefix}_long_short_return": float(metrics.long_short_return),
        f"{prefix}_turnover": float(metrics.turnover),
    }


def metrics_to_dict(metrics: EvaluationSummary) -> dict[str, float]:
    payload = metrics.to_dict()
    return {key: float(value) if pd.notna(value) else float("nan") for key, value in payload.items()}


def aggregate_window_metrics(results: Sequence[WindowModelResult]) -> dict[str, float]:
    if not results:
        return {
            "mean_test_ic": float("nan"),
            "mean_test_rank_ic": float("nan"),
            "mean_test_icir": float("nan"),
            "mean_test_hit_rate": float("nan"),
        }

    return {
        "mean_test_ic": float(np.mean([result.test_metrics.ic for result in results])),
        "mean_test_rank_ic": float(np.mean([result.test_metrics.rank_ic for result in results])),
        "mean_test_icir": float(np.mean([result.test_metrics.icir for result in results])),
        "mean_test_hit_rate": float(np.mean([result.test_metrics.hit_rate for result in results])),
    }
