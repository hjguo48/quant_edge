from __future__ import annotations

from src.data.db.models import MinuteBackfillState


def test_minute_backfill_state_model_has_watermark_columns() -> None:
    columns = MinuteBackfillState.__table__.columns

    assert "published_at" in columns
    assert "watermark" in columns
