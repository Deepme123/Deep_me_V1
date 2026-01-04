from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, ForeignKey, Index, UniqueConstraint
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime

# 1. 감정 세션 모델
class EmotionSession(SQLModel, table=True):
    __tablename__ = "emotionsession"

    session_id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.user_id")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    emotion_label: Optional[str] = None
    topic: Optional[str] = None
    trigger_summary: Optional[str] = None
    insight_summary: Optional[str] = None

    steps: List["EmotionStep"] = Relationship(back_populates="session")


# 2. 감정 단계(스텝) 모델
class EmotionStep(SQLModel, table=True):
    __tablename__ = "emotionstep"

    __table_args__ = (
        UniqueConstraint("session_id", "step_order", name="uq_emotionstep_session_order"),
        Index("ix_emotionstep_session_order", "session_id", "step_order"),
    )

    step_id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(
        sa_column=Column(
            ForeignKey("emotionsession.session_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    step_order: int
    step_type: str
    user_input: str
    gpt_response: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    insight_tag: Optional[str] = None

    session: Optional[EmotionSession] = Relationship(back_populates="steps")



