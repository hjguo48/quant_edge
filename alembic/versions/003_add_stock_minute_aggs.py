"""Add stock_minute_aggs hypertable for Week 3 intraday smoke."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stock_minute_aggs",
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("minute_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("high", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("low", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("close", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("vwap", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("transactions", sa.BigInteger(), nullable=True),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=False),
        sa.PrimaryKeyConstraint("ticker", "minute_ts"),
    )
    op.create_index("idx_stock_minute_aggs_trade_date", "stock_minute_aggs", ["trade_date"], unique=False)
    op.create_index(
        "idx_stock_minute_aggs_knowledge_time",
        "stock_minute_aggs",
        ["knowledge_time"],
        unique=False,
    )
    op.create_index("idx_stock_minute_aggs_id", "stock_minute_aggs", ["id"], unique=False)

    op.execute(
        """
        SELECT create_hypertable(
            'stock_minute_aggs',
            'minute_ts',
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '7 days'
        )
        """,
    )


def downgrade() -> None:
    op.drop_index("idx_stock_minute_aggs_id", table_name="stock_minute_aggs")
    op.drop_index("idx_stock_minute_aggs_knowledge_time", table_name="stock_minute_aggs")
    op.drop_index("idx_stock_minute_aggs_trade_date", table_name="stock_minute_aggs")
    op.drop_table("stock_minute_aggs")
