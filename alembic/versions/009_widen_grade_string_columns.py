"""Widen grades_events string columns to accommodate full FMP grade labels.

FMP returns grade strings like "Market Outperform" (17 chars) and
"Conviction Buy List" (19 chars) that exceed the original VARCHAR(16).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "grades_events",
        "new_grade",
        existing_type=sa.String(length=16),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "grades_events",
        "prior_grade",
        existing_type=sa.String(length=16),
        type_=sa.String(length=64),
        existing_nullable=True,
    )
    op.alter_column(
        "grades_events",
        "action",
        existing_type=sa.String(length=16),
        type_=sa.String(length=64),
        existing_nullable=False,
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrade unsafe: existing rows may exceed VARCHAR(16) after widening. "
        "Manual data migration required before reverting."
    )
