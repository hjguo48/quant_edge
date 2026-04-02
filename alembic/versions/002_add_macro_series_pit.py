"""Add macro_series_pit under Alembic management."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "macro_series_pit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("series_id", sa.String(length=20), nullable=False),
        sa.Column("observation_date", sa.Date(), nullable=False),
        sa.Column("value", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_revision", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "series_id",
            "observation_date",
            "knowledge_time",
            name="uq_macro_series_pit_version",
        ),
    )
    op.create_index(
        "idx_macro_series_pit_lookup",
        "macro_series_pit",
        ["series_id", "knowledge_time", "observation_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_macro_series_pit_lookup", table_name="macro_series_pit")
    op.drop_table("macro_series_pit")
