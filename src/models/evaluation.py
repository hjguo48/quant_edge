from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationSummary:
    ic: float
    rank_ic: float
    icir: float
    hit_rate: float
    top_decile_return: float
    long_short_return: float
    turnover: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    ic_series = information_coefficient_series(y_true=y_true, y_pred=y_pred)
    return float(ic_series.mean()) if not ic_series.empty else float("nan")


def rank_information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    rank_ic_series = rank_information_coefficient_series(y_true=y_true, y_pred=y_pred)
    return float(rank_ic_series.mean()) if not rank_ic_series.empty else float("nan")


def information_coefficient_series(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    grouped = _group_aligned_series(y_true=y_true, y_pred=y_pred)
    values: list[tuple[object, float]] = []

    for date_key, frame in grouped:
        if len(frame) < 2:
            continue
        score = frame["y_true"].corr(frame["y_pred"], method="pearson")
        if pd.notna(score):
            values.append((date_key, float(score)))

    return pd.Series(
        {date_key: value for date_key, value in values},
        dtype=float,
        name="ic",
    )


def rank_information_coefficient_series(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    grouped = _group_aligned_series(y_true=y_true, y_pred=y_pred)
    values: list[tuple[object, float]] = []

    for date_key, frame in grouped:
        if len(frame) < 2:
            continue
        score = frame["y_true"].corr(frame["y_pred"], method="spearman")
        if pd.notna(score):
            values.append((date_key, float(score)))

    return pd.Series(
        {date_key: value for date_key, value in values},
        dtype=float,
        name="rank_ic",
    )


def icir(y_true: pd.Series, y_pred: pd.Series) -> float:
    ic_series = information_coefficient_series(y_true=y_true, y_pred=y_pred)
    if ic_series.empty:
        return float("nan")

    dispersion = float(ic_series.std(ddof=0))
    if math.isclose(dispersion, 0.0):
        return float("nan")
    return float(ic_series.mean() / dispersion)


def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = _align_target_and_prediction(y_true=y_true, y_pred=y_pred)
    if aligned.empty:
        return float("nan")

    signed = np.sign(aligned[["y_true", "y_pred"]])
    correct = signed["y_true"] == signed["y_pred"]
    return float(correct.mean())


def top_decile_return(y_true: pd.Series, y_pred: pd.Series) -> float:
    grouped = _group_aligned_series(y_true=y_true, y_pred=y_pred)
    returns: list[float] = []

    for _, frame in grouped:
        decile_size = _decile_size(len(frame))
        if decile_size is None:
            continue
        selected = frame.nlargest(decile_size, "y_pred")
        returns.append(float(selected["y_true"].mean()))

    return float(np.mean(returns)) if returns else float("nan")


def long_short_return(y_true: pd.Series, y_pred: pd.Series) -> float:
    grouped = _group_aligned_series(y_true=y_true, y_pred=y_pred)
    spreads: list[float] = []

    for _, frame in grouped:
        decile_size = _decile_size(len(frame))
        if decile_size is None:
            continue
        top = frame.nlargest(decile_size, "y_pred")["y_true"].mean()
        bottom = frame.nsmallest(decile_size, "y_pred")["y_true"].mean()
        spreads.append(float(top - bottom))

    return float(np.mean(spreads)) if spreads else float("nan")


def turnover(y_pred: pd.Series) -> float:
    grouped = _group_prediction_cross_sections(y_pred=y_pred)
    if len(grouped) < 2:
        return 0.0

    turnovers: list[float] = []
    previous_selection: set[object] | None = None

    for _, frame in grouped:
        decile_size = _decile_size(len(frame))
        if decile_size is None:
            continue
        selected = set(frame.nlargest(decile_size, "y_pred").index)
        if previous_selection is not None and previous_selection:
            overlap = len(previous_selection & selected)
            turnovers.append(1.0 - (overlap / len(previous_selection)))
        previous_selection = selected

    return float(np.mean(turnovers)) if turnovers else 0.0


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> EvaluationSummary:
    return EvaluationSummary(
        ic=information_coefficient(y_true=y_true, y_pred=y_pred),
        rank_ic=rank_information_coefficient(y_true=y_true, y_pred=y_pred),
        icir=icir(y_true=y_true, y_pred=y_pred),
        hit_rate=hit_rate(y_true=y_true, y_pred=y_pred),
        top_decile_return=top_decile_return(y_true=y_true, y_pred=y_pred),
        long_short_return=long_short_return(y_true=y_true, y_pred=y_pred),
        turnover=turnover(y_pred=y_pred),
    )


def _align_target_and_prediction(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series objects.")

    aligned = pd.concat(
        [y_true.rename("y_true"), y_pred.rename("y_pred")],
        axis=1,
        join="inner",
    ).dropna()
    return aligned.sort_index()


def _group_aligned_series(y_true: pd.Series, y_pred: pd.Series) -> list[tuple[object, pd.DataFrame]]:
    aligned = _align_target_and_prediction(y_true=y_true, y_pred=y_pred)
    if aligned.empty:
        return []

    if isinstance(aligned.index, pd.MultiIndex):
        date_level = _date_level_name(aligned.index)
        groups = []
        for date_key, frame in aligned.groupby(level=date_level, sort=True):
            groups.append((date_key, frame.droplevel(date_level)))
        return groups

    return [(pd.NaT, aligned)]


def _group_prediction_cross_sections(y_pred: pd.Series) -> list[tuple[object, pd.DataFrame]]:
    aligned = pd.DataFrame({"y_pred": y_pred}).dropna().sort_index()
    if aligned.empty:
        return []

    if isinstance(aligned.index, pd.MultiIndex):
        date_level = _date_level_name(aligned.index)
        groups = []
        for date_key, frame in aligned.groupby(level=date_level, sort=True):
            groups.append((date_key, frame.droplevel(date_level)))
        return groups

    return [(pd.NaT, aligned)]


def _date_level_name(index: pd.MultiIndex) -> str | int:
    for candidate in ("date", "trade_date", "signal_date", "calc_date"):
        if candidate in index.names:
            return candidate
    return 0


def _decile_size(cross_section_size: int) -> int | None:
    if cross_section_size < 2:
        return None
    return max(1, int(math.ceil(cross_section_size * 0.10)))
