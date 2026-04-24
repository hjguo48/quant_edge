"""Add sampled trades hypertable and sampling state table."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stock_trades_sampled",
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("trading_date", sa.Date(), nullable=False),
        sa.Column("sip_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("participant_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trf_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price", sa.Numeric(precision=14, scale=6), nullable=False),
        sa.Column("size", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("decimal_size", sa.Numeric(precision=18, scale=8), nullable=True),
        # exchange can use the -1 sentinel when the upstream payload omits it.
        # sequence_number must normally be populated by stable_sequence_fallback();
        # this server_default is only a final DB safety net and should not be hit
        # by the ingestion path.
        sa.Column("exchange", sa.SmallInteger(), nullable=False, server_default="-1"),
        sa.Column("tape", sa.SmallInteger(), nullable=True),
        sa.Column("conditions", postgresql.ARRAY(sa.SmallInteger()), nullable=True),
        sa.Column("correction", sa.SmallInteger(), nullable=True),
        sa.Column("sequence_number", sa.BigInteger(), nullable=False, server_default="-1"),
        sa.Column("trade_id", sa.String(length=64), nullable=True),
        sa.Column("trf_id", sa.String(length=32), nullable=True),
        sa.Column("sampled_reason", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("ticker", "sip_timestamp", "exchange", "sequence_number"),
    )
    op.execute(
        """
        SELECT create_hypertable(
            'stock_trades_sampled',
            'sip_timestamp',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
        """,
    )
    op.create_index(
        "ix_trades_sampled_knowledge_time",
        "stock_trades_sampled",
        ["ticker", "knowledge_time"],
        unique=False,
    )
    op.create_index(
        "ix_trades_sampled_trading_date",
        "stock_trades_sampled",
        ["ticker", "trading_date"],
        unique=False,
    )
    op.execute(
        """
        ALTER TABLE stock_trades_sampled SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'ticker',
            timescaledb.compress_orderby = 'sip_timestamp DESC, sequence_number DESC'
        );
        """,
    )
    op.execute(
        """
        SELECT add_compression_policy(
            'stock_trades_sampled',
            compress_after => INTERVAL '7 days',
            if_not_exists => TRUE
        );
        """,
    )

    op.create_table(
        "trades_sampling_state",
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("trading_date", sa.Date(), nullable=False),
        sa.Column("sampled_reason", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("rows_ingested", sa.Integer(), nullable=True),
        sa.Column("pages_fetched", sa.Integer(), nullable=True),
        sa.Column("api_calls_used", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "trading_date", "sampled_reason"),
    )
    op.create_index(
        "ix_trades_sampling_state_status",
        "trades_sampling_state",
        ["status", "trading_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_trades_sampling_state_status", table_name="trades_sampling_state")
    op.drop_table("trades_sampling_state")

    op.execute("SELECT remove_compression_policy('stock_trades_sampled', if_exists => TRUE);")
    op.execute(
        """
        ALTER TABLE stock_trades_sampled
        RESET (
            timescaledb.compress,
            timescaledb.compress_segmentby,
            timescaledb.compress_orderby
        );
        """,
    )
    op.drop_index("ix_trades_sampled_trading_date", table_name="stock_trades_sampled")
    op.drop_index("ix_trades_sampled_knowledge_time", table_name="stock_trades_sampled")
    op.drop_table("stock_trades_sampled")
