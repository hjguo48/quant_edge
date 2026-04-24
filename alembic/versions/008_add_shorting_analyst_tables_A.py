"""Add Week 5 Tranche A shorting and analyst tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "short_sale_volume_daily",
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market", sa.String(length=16), nullable=False),
        sa.Column("short_volume", sa.BigInteger(), nullable=False),
        sa.Column("short_exempt_volume", sa.BigInteger(), nullable=True),
        sa.Column("total_volume", sa.BigInteger(), nullable=False),
        sa.Column("file_etag", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "trade_date", "market"),
    )
    op.create_index(
        "ix_short_sale_kt",
        "short_sale_volume_daily",
        ["ticker", "knowledge_time"],
        unique=False,
    )

    op.create_table(
        "grades_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("analyst_firm", sa.String(length=128), nullable=False),
        sa.Column("prior_grade", sa.String(length=16), nullable=True),
        sa.Column("new_grade", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("grade_score_change", sa.SmallInteger(), nullable=False),
        sa.UniqueConstraint("ticker", "event_date", "analyst_firm", name="uq_grade_event"),
    )
    op.create_index(
        "ix_grades_kt",
        "grades_events",
        ["ticker", "knowledge_time"],
        unique=False,
    )

    op.create_table(
        "ratings_events",
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rating_score", sa.SmallInteger(), nullable=False),
        sa.Column("rating_recommendation", sa.String(length=32), nullable=True),
        sa.Column("dcf_rating", sa.Numeric(precision=6, scale=2), nullable=True),
        sa.Column("pe_rating", sa.Numeric(precision=6, scale=2), nullable=True),
        sa.Column("roe_rating", sa.Numeric(precision=6, scale=2), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "event_date"),
    )
    op.create_index(
        "ix_ratings_kt",
        "ratings_events",
        ["ticker", "knowledge_time"],
        unique=False,
    )

    op.create_table(
        "price_target_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("analyst_firm", sa.String(length=128), nullable=True),
        sa.Column("target_price", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("prior_target", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("price_when_published", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("is_consensus", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.UniqueConstraint("ticker", "event_date", "analyst_firm", name="uq_target_event"),
    )
    op.create_index(
        "ix_target_kt",
        "price_target_events",
        ["ticker", "knowledge_time"],
        unique=False,
    )

    op.create_table(
        "earnings_calendar",
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("announce_date", sa.Date(), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timing", sa.String(length=16), nullable=True),
        sa.Column("fiscal_period_end", sa.Date(), nullable=True),
        sa.Column("eps_estimate", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("eps_actual", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("revenue_estimate", sa.BigInteger(), nullable=True),
        sa.Column("revenue_actual", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "announce_date"),
    )
    op.create_index(
        "ix_earnings_cal_kt",
        "earnings_calendar",
        ["ticker", "knowledge_time"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_earnings_cal_kt", table_name="earnings_calendar")
    op.drop_table("earnings_calendar")

    op.drop_index("ix_target_kt", table_name="price_target_events")
    op.drop_table("price_target_events")

    op.drop_index("ix_ratings_kt", table_name="ratings_events")
    op.drop_table("ratings_events")

    op.drop_index("ix_grades_kt", table_name="grades_events")
    op.drop_table("grades_events")

    op.drop_index("ix_short_sale_kt", table_name="short_sale_volume_daily")
    op.drop_table("short_sale_volume_daily")
