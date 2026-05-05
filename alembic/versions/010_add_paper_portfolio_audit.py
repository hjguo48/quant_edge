"""Add paper portfolio audit persistence table."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paper_portfolio_audit",
        sa.Column("bundle_version", sa.Text(), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("target_weight", sa.Numeric(precision=10, scale=8), nullable=False),
        sa.Column("raw_score", sa.Numeric(precision=12, scale=8), nullable=True),
        sa.Column("generated_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint(
            "bundle_version",
            "signal_date",
            "ticker",
            name="pk_paper_portfolio_audit",
        ),
    )
    op.create_index(
        "idx_paper_portfolio_audit_signal_date",
        "paper_portfolio_audit",
        ["signal_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_paper_portfolio_audit_signal_date", table_name="paper_portfolio_audit")
    op.drop_table("paper_portfolio_audit")
