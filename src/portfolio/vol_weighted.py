from __future__ import annotations

import pandas as pd

from src.portfolio.constraints import PortfolioConstraints, apply_weight_constraints
from src.portfolio.equal_weight import select_top_scores


def vol_inverse_portfolio(
    scores: pd.Series,
    *,
    volatilities: pd.Series,
    n_stocks: int | None = None,
    selection_pct: float = 0.10,
    constraints: PortfolioConstraints | None = None,
    volatility_floor: float = 1e-4,
) -> dict[str, float]:
    selected = select_top_scores(scores, n_stocks=n_stocks, selection_pct=selection_pct)
    if selected.empty:
        return {}

    sigma = (
        pd.Series(volatilities, dtype=float)
        .reindex(selected.index.astype(str))
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(volatility_floor)
        .clip(lower=volatility_floor)
    )
    raw = 1.0 / sigma
    return apply_weight_constraints(raw, ranking=selected.index.astype(str), constraints=constraints)
