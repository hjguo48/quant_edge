"""Add minute_backfill_state and Timescale compression policy for stock_minute_aggs."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "minute_backfill_state",
        sa.Column("trading_date", sa.Date(), nullable=False),
        sa.Column("source_file", sa.String(length=255), nullable=True),
        sa.Column("rows_raw", sa.Integer(), nullable=True),
        sa.Column("rows_kept", sa.Integer(), nullable=True),
        sa.Column("tickers_loaded", sa.Integer(), nullable=True),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("trading_date"),
    )
    op.create_index(
        "idx_minute_backfill_state_status",
        "minute_backfill_state",
        ["status"],
        unique=False,
    )
    op.create_index(
        "idx_minute_backfill_state_finished_at",
        "minute_backfill_state",
        ["finished_at"],
        unique=False,
    )

    op.execute(
        """
        ALTER TABLE stock_minute_aggs
        SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'ticker',
            timescaledb.compress_orderby = 'minute_ts DESC'
        );
        """,
    )
    op.execute(
        """
        SELECT add_compression_policy(
            'stock_minute_aggs',
            compress_after => INTERVAL '30 days',
            if_not_exists => TRUE
        );
        """,
    )


def downgrade() -> None:
    op.execute("SELECT remove_compression_policy('stock_minute_aggs', if_exists => TRUE);")
    op.execute("ALTER TABLE stock_minute_aggs RESET (timescaledb.compress, timescaledb.compress_segmentby, timescaledb.compress_orderby);")
    op.drop_index("idx_minute_backfill_state_finished_at", table_name="minute_backfill_state")
    op.drop_index("idx_minute_backfill_state_status", table_name="minute_backfill_state")
    op.drop_table("minute_backfill_state")
