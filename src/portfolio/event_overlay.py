from __future__ import annotations

"""Event overlay interfaces for sparse alpha adjustments on top of base scores."""

from abc import ABC, abstractmethod

import pandas as pd


class EventOverlay(ABC):
    """Sparse score overlay: base_score + event_premium where events are available."""

    @abstractmethod
    def apply(self, base_scores: pd.Series, event_data: pd.DataFrame) -> pd.Series:
        """Return overlay-adjusted scores indexed the same way as `base_scores`."""


class NullOverlay(EventOverlay):
    """Default overlay that leaves the base score unchanged."""

    def apply(self, base_scores: pd.Series, event_data: pd.DataFrame) -> pd.Series:
        del event_data
        return pd.Series(base_scores, copy=True)
