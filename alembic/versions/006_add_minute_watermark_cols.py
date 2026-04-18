"""Add published_at and watermark columns to minute_backfill_state."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "minute_backfill_state",
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "minute_backfill_state",
        sa.Column("watermark", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("minute_backfill_state", "watermark")
    op.drop_column("minute_backfill_state", "published_at")
