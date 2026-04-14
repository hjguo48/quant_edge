from __future__ import annotations

"""Confidence-tiered portfolio sizing for high-conviction score buckets.

The default tier schedule follows the Stage 0 design:

- top 10% of scores -> 1.00x multiplier
- 10-20% -> 0.75x multiplier
- 20-30% -> 0.50x multiplier
- below 30% -> no allocation

The function mirrors the existing score-weighted portfolio builder by applying:
candidate selection with hysteresis -> confidence-tier weighting -> optional
shrinkage/no-trade-zone controls -> turnover buffer.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest.execution import select_candidate_tickers
from src.portfolio.constraints import (
    PortfolioConstraints,
    apply_turnover_buffer,
    apply_weight_constraints,
)

ConfidenceTier = tuple[float, float]
DEFAULT_CONFIDENCE_TIERS: tuple[ConfidenceTier, ...] = (
    (0.10, 1.0),
    (0.20, 0.75),
    (0.30, 0.5),
)


def confidence_weighted_portfolio(
    scores: pd.Series,
    previous_weights: Mapping[str, float],
    tiers: Sequence[ConfidenceTier] = DEFAULT_CONFIDENCE_TIERS,
    *,
    selection_pct: float | None = None,
    sell_buffer_pct: float | None = 0.40,
    weight_shrinkage: float = 0.0,
    no_trade_zone: float = 0.0,
    min_trade_weight: float = 0.005,
    max_weight: float = 0.05,
    min_holdings: int = 20,
) -> dict[str, float]:
    """Build a confidence-tier weighted long-only portfolio.

    Args:
        scores: Cross-sectional scores indexed by ticker.
        previous_weights: Current holdings used for hysteresis and turnover control.
        tiers: Ordered `(upper_rank_pct, multiplier)` pairs.
        selection_pct: Optional candidate entry band. Defaults to the widest tier.
    """

    ranked = scores.dropna().astype(float).sort_values(ascending=False)
    if ranked.empty:
        return {}

    normalized_tiers = normalize_confidence_tiers(tiers)
    effective_selection_pct = max(
        float(selection_pct) if selection_pct is not None else 0.0,
        normalized_tiers[-1][0],
    )

    ranking = ranked.index.astype(str).tolist()
    constraints = PortfolioConstraints(
        max_weight=max_weight,
        min_holdings=min_holdings,
        turnover_buffer=min_trade_weight,
    )

    candidate_tickers = select_candidate_tickers(
        ranking=ranking,
        current_weights=previous_weights,
        selection_pct=effective_selection_pct,
        sell_buffer_pct=sell_buffer_pct,
        min_holdings=min_holdings,
        max_weight=max_weight,
    )
    candidate_scores = ranked.reindex(candidate_tickers).dropna()
    if candidate_scores.empty:
        return {}

    target_weights = build_confidence_weights(
        ranked_scores=ranked,
        candidate_scores=candidate_scores,
        tiers=normalized_tiers,
        max_weight=max_weight,
    )
    if not target_weights:
        return {}

    if weight_shrinkage > 0.0 and previous_weights:
        target_weights = apply_weight_shrinkage(
            target_weights=target_weights,
            previous_weights=previous_weights,
            shrinkage=weight_shrinkage,
        )

    if no_trade_zone > 0.0 and previous_weights:
        target_weights = apply_no_trade_zone(
            target_weights=target_weights,
            previous_weights=previous_weights,
            threshold=no_trade_zone,
        )

    if min_trade_weight > 0.0:
        buffer_reference_weights = {
            ticker: float(weight)
            for ticker, weight in previous_weights.items()
            if ticker in set(ranking)
        }
        target_weights = apply_turnover_buffer(
            target_weights,
            current_weights=buffer_reference_weights,
            min_trade_weight=min_trade_weight,
            ranking=ranking,
            constraints=constraints,
        )
    else:
        target_weights = apply_weight_constraints(
            target_weights,
            ranking=ranking,
            constraints=constraints,
        )

    return target_weights


def normalize_confidence_tiers(tiers: Sequence[ConfidenceTier]) -> tuple[ConfidenceTier, ...]:
    if not tiers:
        raise ValueError("confidence tiers must not be empty")

    normalized = sorted((float(bound), float(multiplier)) for bound, multiplier in tiers)
    previous_bound = 0.0
    for bound, multiplier in normalized:
        if not 0.0 < bound <= 1.0:
            raise ValueError(f"tier bound must be in (0, 1], got {bound}")
        if bound <= previous_bound:
            raise ValueError("tier bounds must be strictly increasing")
        if multiplier < 0.0:
            raise ValueError(f"tier multiplier must be non-negative, got {multiplier}")
        previous_bound = bound
    return tuple(normalized)


def build_confidence_weights(
    *,
    ranked_scores: pd.Series,
    candidate_scores: pd.Series,
    tiers: Sequence[ConfidenceTier],
    max_weight: float,
) -> dict[str, float]:
    positive_scores = candidate_scores[candidate_scores > 0.0]
    if positive_scores.empty:
        return {}

    rank_positions = pd.Series(
        np.arange(1, len(ranked_scores) + 1, dtype=float),
        index=ranked_scores.index.astype(str),
        dtype=float,
    )
    candidate_positions = rank_positions.reindex(positive_scores.index.astype(str)).dropna()
    if candidate_positions.empty:
        return {}

    multipliers = candidate_positions.apply(
        lambda rank_position: confidence_multiplier(
            rank_pct=float(rank_position) / float(len(ranked_scores)),
            tiers=tiers,
        ),
    )
    raw = positive_scores.copy()
    raw.index = raw.index.astype(str)
    raw = raw.mul(multipliers.reindex(raw.index).fillna(0.0), fill_value=0.0)
    raw = raw[raw > 0.0]
    if raw.empty:
        return {}

    raw = raw / float(raw.sum())
    raw = raw.clip(upper=max_weight)
    total = float(raw.sum())
    if total <= 0.0:
        return {}
    raw = raw / total
    return {str(ticker): float(weight) for ticker, weight in raw.items() if weight > 0.0}


def confidence_multiplier(*, rank_pct: float, tiers: Sequence[ConfidenceTier]) -> float:
    for upper_bound, multiplier in tiers:
        if rank_pct <= upper_bound:
            return float(multiplier)
    return 0.0


def apply_weight_shrinkage(
    *,
    target_weights: dict[str, float],
    previous_weights: Mapping[str, float],
    shrinkage: float,
) -> dict[str, float]:
    blended: dict[str, float] = {}
    for ticker in set(target_weights) | set(previous_weights):
        target = float(target_weights.get(ticker, 0.0))
        previous = float(previous_weights.get(ticker, 0.0))
        weight = (1.0 - float(shrinkage)) * target + float(shrinkage) * previous
        if weight > 1e-8:
            blended[str(ticker)] = weight

    total = float(sum(blended.values()))
    if total <= 0.0:
        return {}
    return {ticker: weight / total for ticker, weight in blended.items()}


def apply_no_trade_zone(
    *,
    target_weights: dict[str, float],
    previous_weights: Mapping[str, float],
    threshold: float,
) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    for ticker in set(target_weights) | set(previous_weights):
        target = float(target_weights.get(ticker, 0.0))
        previous = float(previous_weights.get(ticker, 0.0))
        if abs(target - previous) < float(threshold) and previous > 0.0:
            adjusted[str(ticker)] = previous
        elif target > 0.0:
            adjusted[str(ticker)] = target

    total = float(sum(adjusted.values()))
    if total <= 0.0:
        return {}
    return {ticker: weight / total for ticker, weight in adjusted.items()}
