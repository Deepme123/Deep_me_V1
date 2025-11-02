# app/models/emotion_step.py
from __future__ import annotations

from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text


class EmotionStep(SQLModel, table=True):
    __tablename__ = "emotionstep"

    step_id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    session_id: UUID = Field(index=True)
    step_order: int
    step_type: str  # "user" | "assistant"

    # user/assistant 상호 배타적 사용
    user_input: Optional[str] = Field(default=None)

    # ★ NULL 허용으로 변경
    gpt_response: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    insight_tag: Optional[str] = None
