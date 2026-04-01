from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.models.evaluation import (
    evaluate_predictions,
    icir,
    information_coefficient,
    rank_information_coefficient,
    turnover,
)


def test_evaluate_predictions_perfect_cross_section() -> None:
    index = pd.MultiIndex.from_product(
        [[date(2021, 1, 1)], [f"TICKER_{i}" for i in range(10)]],
        names=["date", "ticker"],
    )
    y_true = pd.Series([0.01 * (i + 1) for i in range(10)], index=index, dtype=float)
    y_pred = y_true.copy()

    metrics = evaluate_predictions(y_true=y_true, y_pred=y_pred)

    assert metrics.ic == pytest.approx(1.0)
    assert metrics.rank_ic == pytest.approx(1.0)
    assert metrics.hit_rate == pytest.approx(1.0)
    assert metrics.top_decile_return == pytest.approx(0.10)
    assert metrics.long_short_return == pytest.approx(0.09)
    assert metrics.turnover == pytest.approx(0.0)
    assert pd.isna(metrics.icir)


def test_ic_rank_ic_and_icir_on_two_dates() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (date(2021, 1, 1), "AAA"),
            (date(2021, 1, 1), "BBB"),
            (date(2021, 1, 1), "CCC"),
            (date(2021, 1, 8), "AAA"),
            (date(2021, 1, 8), "BBB"),
            (date(2021, 1, 8), "CCC"),
        ],
        names=["date", "ticker"],
    )
    y_true = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=index)
    y_pred = pd.Series([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], index=index)

    assert information_coefficient(y_true=y_true, y_pred=y_pred) == pytest.approx(0.0)
    assert rank_information_coefficient(y_true=y_true, y_pred=y_pred) == pytest.approx(0.0)
    assert icir(y_true=y_true, y_pred=y_pred) == pytest.approx(0.0)


def test_turnover_tracks_top_decile_membership_changes() -> None:
    index = pd.MultiIndex.from_product(
        [[date(2021, 1, 1), date(2021, 1, 8)], [f"TICKER_{i}" for i in range(10)]],
        names=["date", "ticker"],
    )
    y_pred = pd.Series(
        [
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            1.0,
            10.0,
            8.0,
            7.0,
            6.0,
            5.0,
            4.0,
            3.0,
            2.0,
            9.0,
        ],
        index=index,
        dtype=float,
    )

    assert turnover(y_pred=y_pred) == pytest.approx(1.0)
