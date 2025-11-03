from alembic import op
import sqlalchemy as sa

revision = "20251102_add_emotionstep_xor_check"
down_revision = "20251102_make_user_input_nullable"  # 위 파일 ID

CHECK_NAME = "ck_emotionstep_user_or_assistant_only"

def upgrade():
    op.create_check_constraint(
        CHECK_NAME,
        "emotionstep",
        # user/assistant에만 XOR 강제. (그 외 타입은 제약에서 제외)
        "((step_type = 'user' AND user_input IS NOT NULL AND gpt_response IS NULL)"
        " OR (step_type = 'assistant' AND gpt_response IS NOT NULL AND user_input IS NULL)"
        " OR (step_type NOT IN ('user','assistant')))"
    )

def downgrade():
    op.drop_constraint(CHECK_NAME, "emotionstep", type_="check")
