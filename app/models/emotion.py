from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime

# 1. 감정 세션 모델
class EmotionSession(SQLModel, table=True):
    session_id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    emotion_label: Optional[str] = None      # 예: 무시당함
    topic: Optional[str] = None              # 예: 가족, 회사
    trigger_summary: Optional[str] = None
    insight_summary: Optional[str] = None

    steps: List["EmotionStep"] = Relationship(back_populates="session")

# 2. 감정 단계(스텝) 모델
class EmotionStep(SQLModel, table=True):
    step_id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="emotionsession.session_id")
    step_order: int
    step_type: str                 # 예: 감정탐색, 사고탐색 등
    user_input: str
    gpt_response: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    insight_tag: Optional[str] = None

    session: Optional[EmotionSession] = Relationship(back_populates="steps")

