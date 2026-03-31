"""Initial QuantEdge schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "stocks",
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("company_name", sa.String(length=200), nullable=False),
        sa.Column("sector", sa.String(length=50), nullable=True),
        sa.Column("industry", sa.String(length=100), nullable=True),
        sa.Column("ipo_date", sa.Date(), nullable=True),
        sa.Column("delist_date", sa.Date(), nullable=True),
        sa.Column("delist_reason", sa.String(length=50), nullable=True),
        sa.Column("shares_outstanding", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.PrimaryKeyConstraint("ticker"),
    )

    op.create_table(
        "stock_prices",
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("high", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("low", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("close", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("adj_close", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "trade_date"),
    )

    op.create_table(
        "fundamentals_pit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("fiscal_period", sa.String(length=10), nullable=False),
        sa.Column("metric_name", sa.String(length=50), nullable=False),
        sa.Column("metric_value", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("event_time", sa.Date(), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_restated", sa.Boolean(), server_default=sa.text("FALSE"), nullable=True),
        sa.Column("source", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "ticker",
            "fiscal_period",
            "metric_name",
            "knowledge_time",
            name="uq_fundamentals_pit_version",
        ),
    )
    op.create_index(
        "idx_fundamentals_pit_lookup",
        "fundamentals_pit",
        ["ticker", "knowledge_time", "metric_name"],
        unique=False,
    )

    op.create_table(
        "universe_membership",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("index_name", sa.String(length=20), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("reason", sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "ticker",
            "index_name",
            "effective_date",
            name="uq_universe_membership_entry",
        ),
    )

    op.create_table(
        "corporate_actions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("action_type", sa.String(length=20), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("ratio", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("old_ticker", sa.String(length=10), nullable=True),
        sa.Column("new_ticker", sa.String(length=10), nullable=True),
        sa.Column("details_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "feature_store",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("calc_date", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.String(length=50), nullable=False),
        sa.Column("feature_value", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("is_filled", sa.Boolean(), server_default=sa.text("FALSE"), nullable=True),
        sa.Column("batch_id", sa.String(length=36), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "ticker",
            "calc_date",
            "feature_name",
            "batch_id",
            name="uq_feature_store_batch",
        ),
    )

    op.create_table(
        "model_registry",
        sa.Column("model_id", sa.String(length=36), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
        sa.Column("model_type", sa.String(length=20), nullable=False),
        sa.Column("train_start", sa.Date(), nullable=False),
        sa.Column("train_end", sa.Date(), nullable=False),
        sa.Column("val_start", sa.Date(), nullable=True),
        sa.Column("val_end", sa.Date(), nullable=True),
        sa.Column("features_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("hyperparams_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=20), server_default=sa.text("'staging'"), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.PrimaryKeyConstraint("model_id"),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("feature_batch_id", sa.String(length=36), nullable=False),
        sa.Column("pred_score", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("pred_rank", sa.Integer(), nullable=False),
        sa.Column("pred_decile", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_registry.model_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "ticker",
            "signal_date",
            "model_version_id",
            name="uq_prediction_model_signal",
        ),
    )

    op.create_table(
        "portfolios",
        sa.Column("portfolio_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("strategy_name", sa.String(length=100), nullable=True),
        sa.Column("weighting_scheme", sa.String(length=20), nullable=True),
        sa.Column("config_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.PrimaryKeyConstraint("portfolio_id"),
    )

    op.create_table(
        "backtest_results",
        sa.Column("backtest_id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("config_snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("equity_curve", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("stat_tests_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_registry.model_id"]),
        sa.PrimaryKeyConstraint("backtest_id"),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("actor", sa.String(length=50), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("feature_batch_id", sa.String(length=36), nullable=True),
        sa.Column("details_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute("CREATE INDEX idx_audit_log_time ON audit_log (timestamp DESC)")

    # Week 1 only needs the market-data hypertable. The feature and prediction
    # tables retain surrogate-key primary keys that are not hypertable-compatible
    # in TimescaleDB until a later schema revision includes the partition key in
    # their unique constraints.
    op.execute("SELECT create_hypertable('stock_prices', 'trade_date', if_not_exists => TRUE)")


def downgrade() -> None:
    op.drop_index("idx_audit_log_time", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_table("backtest_results")
    op.drop_table("portfolios")
    op.drop_table("predictions")
    op.drop_table("model_registry")
    op.drop_table("feature_store")
    op.drop_table("corporate_actions")
    op.drop_table("universe_membership")
    op.drop_index("idx_fundamentals_pit_lookup", table_name="fundamentals_pit")
    op.drop_table("fundamentals_pit")
    op.drop_table("stock_prices")
    op.drop_table("stocks")
