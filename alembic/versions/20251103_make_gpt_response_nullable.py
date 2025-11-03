from alembic import op
import sqlalchemy as sa

revision = "20251103_make_gpt_response_nullable"
down_revision = "<직전 리비전 ID로 교체>"

def upgrade():
    op.alter_column(
        "emotionstep",
        "gpt_response",
        existing_type=sa.Text(),
        nullable=True,
    )

def downgrade():
    op.alter_column(
        "emotionstep",
        "gpt_response",
        existing_type=sa.Text(),
        nullable=False,
    )
