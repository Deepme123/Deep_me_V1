from alembic import op
import sqlalchemy as sa

revision = "20251103_make_user_input_nullable"
down_revision = "20251103_make_gpt_response_nullable"  # 위 파일 ID

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
