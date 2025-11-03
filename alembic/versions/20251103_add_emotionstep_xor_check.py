from alembic import op

revision = "20251103_add_emotionstep_xor_check"
down_revision = "20251103_make_user_input_nullable"

CHECK_NAME = "ck_emotionstep_user_or_assistant_only"

def upgrade():
    op.create_check_constraint(
        CHECK_NAME,
        "emotionstep",
        # user/assistant일 때만 XOR 강제
        "((step_type = 'user' AND user_input IS NOT NULL AND gpt_response IS NULL)"
        " OR (step_type = 'assistant' AND gpt_response IS NOT NULL AND user_input IS NULL)"
        " OR (step_type NOT IN ('user','assistant')))"
    )

def downgrade():
    op.drop_constraint(CHECK_NAME, "emotionstep", type_="check")
