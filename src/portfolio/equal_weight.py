from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from src.portfolio.constraints import PortfolioConstraints, apply_weight_constraints


def equal_weight_portfolio(
    scores: pd.Series,
    *,
    n_stocks: int | None = None,
    selection_pct: float = 0.10,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, float]:
    selected = select_top_scores(scores, n_stocks=n_stocks, selection_pct=selection_pct)
    if selected.empty:
        return {}

    raw = pd.Series(1.0, index=selected.index.astype(str), dtype=float)
    return apply_weight_constraints(raw, ranking=selected.index.astype(str), constraints=constraints)


def select_top_scores(
    scores: pd.Series,
    *,
    n_stocks: int | None = None,
    selection_pct: float = 0.10,
) -> pd.Series:
    if scores.empty:
        return pd.Series(dtype=float)

    ranked = scores.dropna().astype(float).sort_values(ascending=False)
    if ranked.empty:
        return ranked

    target_count = int(n_stocks) if n_stocks is not None else max(1, int((len(ranked) * selection_pct) + 0.999999))
    return ranked.head(target_count)
