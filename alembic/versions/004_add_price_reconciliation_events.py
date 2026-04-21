"""Add price_reconciliation_events table for Week 3 A-plus gate warnings."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "price_reconciliation_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("field", sa.String(length=20), nullable=False),
        sa.Column("stock_prices_value", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("minute_agg_value", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("delta_bp", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_price_reconciliation_events_trade_date",
        "price_reconciliation_events",
        ["trade_date"],
        unique=False,
    )
    op.create_index(
        "idx_price_reconciliation_events_severity",
        "price_reconciliation_events",
        ["severity"],
        unique=False,
    )
    op.create_index(
        "idx_price_reconciliation_events_batch_id",
        "price_reconciliation_events",
        ["batch_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_price_reconciliation_events_batch_id", table_name="price_reconciliation_events")
    op.drop_index("idx_price_reconciliation_events_severity", table_name="price_reconciliation_events")
    op.drop_index("idx_price_reconciliation_events_trade_date", table_name="price_reconciliation_events")
    op.drop_table("price_reconciliation_events")
