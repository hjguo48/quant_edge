from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    max_weight: float = 0.05
    min_weight: float = 0.0
    min_holdings: int = 20
    turnover_buffer: float = 0.0


def apply_weight_constraints(
    weights: Mapping[str, float] | pd.Series,
    *,
    ranking: Sequence[str] | pd.Index | None = None,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, float]:
    active = constraints or PortfolioConstraints()
    series = normalize_weights(weights)
    if series.empty:
        return {}

    if active.min_weight > 0.0:
        series = series[series >= active.min_weight]
        series = normalize_weights(series)

    series = cap_weights(series, max_weight=active.max_weight)
    if active.min_holdings > 0:
        series = ensure_min_holdings(
            series,
            ranking=ranking,
            min_holdings=active.min_holdings,
            max_weight=active.max_weight,
        )
        series = cap_weights(series, max_weight=active.max_weight)

    series = normalize_weights(series)
    return {str(ticker): float(weight) for ticker, weight in series.items() if weight > 0.0}


def apply_turnover_buffer(
    target_weights: Mapping[str, float] | pd.Series,
    *,
    current_weights: Mapping[str, float] | pd.Series,
    min_trade_weight: float,
    ranking: Sequence[str] | pd.Index | None = None,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, float]:
    if min_trade_weight <= 0.0:
        return apply_weight_constraints(target_weights, ranking=ranking, constraints=constraints)

    target = normalize_weights(target_weights)
    current = normalize_weights(current_weights)
    union = sorted(set(target.index) | set(current.index))
    fixed: dict[str, float] = {}
    adjustable = []

    for ticker in union:
        delta = float(target.get(ticker, 0.0) - current.get(ticker, 0.0))
        if abs(delta) < min_trade_weight:
            fixed[ticker] = float(current.get(ticker, 0.0))
        else:
            adjustable.append(ticker)

    fixed_total = float(sum(fixed.values()))
    adjustable_target = target.reindex(adjustable).fillna(0.0)
    adjustable_total = float(adjustable_target.sum())
    residual = max(0.0, 1.0 - fixed_total)

    if adjustable and adjustable_total > 0.0:
        scaled = adjustable_target * (residual / adjustable_total)
    elif adjustable and residual > 0.0:
        scaled = pd.Series(residual / len(adjustable), index=adjustable, dtype=float)
    else:
        scaled = pd.Series(dtype=float)

    blended = pd.Series(fixed, dtype=float)
    if not scaled.empty:
        blended = pd.concat([blended, scaled]).groupby(level=0).sum()

    return apply_weight_constraints(blended, ranking=ranking, constraints=constraints)


def normalize_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    series = pd.Series(weights, dtype=float)
    if series.empty:
        return pd.Series(dtype=float)

    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    series = series[series > 0.0]
    if series.empty:
        return pd.Series(dtype=float)

    total = float(series.sum())
    if total <= 0.0:
        return pd.Series(dtype=float)
    return (series / total).sort_values(ascending=False)


def cap_weights(weights: Mapping[str, float] | pd.Series, *, max_weight: float) -> pd.Series:
    series = normalize_weights(weights)
    if series.empty:
        return series
    if max_weight <= 0.0:
        return series

    max_slots_required = int(np.ceil(1.0 / max_weight))
    if len(series) < max_slots_required:
        return pd.Series(1.0 / len(series), index=series.index, dtype=float)

    clipped = series.copy()
    tolerance = 1e-12
    while bool((clipped > max_weight + tolerance).any()):
        over = clipped[clipped > max_weight + tolerance]
        excess = float((over - max_weight).sum())
        clipped.loc[over.index] = max_weight

        under_index = clipped.index[clipped < max_weight - tolerance]
        if len(under_index) == 0 or excess <= tolerance:
            break

        base = clipped.loc[under_index]
        if float(base.sum()) <= tolerance:
            clipped.loc[under_index] += excess / len(under_index)
        else:
            clipped.loc[under_index] += excess * (base / float(base.sum()))

    return normalize_weights(clipped.clip(lower=0.0))


def ensure_min_holdings(
    weights: Mapping[str, float] | pd.Series,
    *,
    ranking: Sequence[str] | pd.Index | None,
    min_holdings: int,
    max_weight: float,
) -> pd.Series:
    series = normalize_weights(weights)
    if min_holdings <= 0 or len(series) >= min_holdings:
        return series
    if ranking is None:
        return series

    ordered = [str(item) for item in ranking]
    target_count = max(min_holdings, int(np.ceil(1.0 / max_weight)) if max_weight > 0.0 else min_holdings)
    target_names = ordered[:target_count]
    if not target_names:
        return series

    broadened = pd.Series(0.0, index=pd.Index(target_names, dtype=object), dtype=float)
    if not series.empty:
        overlap = [ticker for ticker in series.index if ticker in broadened.index]
        broadened.loc[overlap] = series.loc[overlap].astype(float)

    missing = [ticker for ticker in broadened.index if broadened.loc[ticker] <= 0.0]
    if missing:
        seed_weight = min(max_weight, 1.0 / len(broadened))
        broadened.loc[missing] = seed_weight

    return normalize_weights(broadened)
