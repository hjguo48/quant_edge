from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio.confidence_sizing import (
    DEFAULT_CONFIDENCE_TIERS,
    confidence_weighted_portfolio,
    normalize_confidence_tiers,
)
from src.portfolio.event_overlay import NullOverlay


def test_confidence_weighted_portfolio_respects_tier_cutoffs() -> None:
    scores = pd.Series(
        [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05],
        index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        dtype=float,
    )

    weights = confidence_weighted_portfolio(
        scores,
        previous_weights={},
        tiers=DEFAULT_CONFIDENCE_TIERS,
        min_holdings=1,
        max_weight=1.0,
        min_trade_weight=0.0,
        sell_buffer_pct=0.30,
    )

    assert set(weights) == {"A", "B", "C"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["A"] > weights["B"] > weights["C"]


def test_normalize_confidence_tiers_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        normalize_confidence_tiers(((0.20, 1.0), (0.20, 0.5)))


def test_null_overlay_returns_original_scores() -> None:
    scores = pd.Series([1.0, 0.5], index=["AAA", "BBB"], dtype=float)

    overlay = NullOverlay()
    result = overlay.apply(scores, pd.DataFrame())

    pd.testing.assert_series_equal(result, scores)
