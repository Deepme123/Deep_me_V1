from alembic import op
import sqlalchemy as sa

revision = "20251102_make_user_input_nullable"
down_revision = "<이전 리비전 ID로 교체>"  # alembic history로 확인

def upgrade():
    op.alter_column(
        "emotionstep",
        "user_input",
        existing_type=sa.Text(),
        nullable=True,
    )

def downgrade():
    op.alter_column(
        "emotionstep",
        "user_input",
        existing_type=sa.Text(),
        nullable=False,
    )
