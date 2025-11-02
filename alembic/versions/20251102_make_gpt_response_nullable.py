# alembic/versions/20251102_make_gpt_response_nullable.py
from alembic import op
import sqlalchemy as sa

# 고유 리비전 ID
revision = "20251102_make_gpt_response_nullable"
# 여기를 직전 리비전ID로 교체
down_revision = "<prev_revision_id>"

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
