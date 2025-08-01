from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict

# ── 세션 생성 요청 ─────────────────────────────────────────────
class EmotionSessionCreate(BaseModel):
    user_id: UUID
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    emotion_label: Optional[str] = None
    topic: Optional[str] = None
    trigger_summary: Optional[str] = None
    insight_summary: Optional[str] = None

# ── 세션 조회 응답 ────────────────────────────────────────────
class EmotionSessionRead(BaseModel):
    session_id: UUID
    user_id: UUID
    started_at: datetime
    ended_at: Optional[datetime]
    emotion_label: Optional[str]
    topic: Optional[str]
    trigger_summary: Optional[str]
    insight_summary: Optional[str]

    model_config = ConfigDict(from_attributes=True)

# ── 스텝 생성 요청(수동) ───────────────────────────────────────
class EmotionStepCreate(BaseModel):
    session_id: UUID
    step_order: int
    step_type: str
    user_input: str
    gpt_response: str
    created_at: Optional[datetime] = None
    insight_tag: Optional[str] = None

# ── 스텝 조회 응답 ────────────────────────────────────────────
class EmotionStepRead(BaseModel):
    step_id: UUID
    session_id: UUID
    step_order: int
    step_type: str
    user_input: str
    gpt_response: str
    created_at: datetime
    insight_tag: Optional[str]

    model_config = ConfigDict(from_attributes=True)

# ── 스텝 생성 요청(LLM 자동 생성) ────────────────────────────
class EmotionStepGenerateInput(BaseModel):
    session_id: Optional[UUID] = None      # 없으면 자동 생성
    user_id: UUID                          # 새 세션용 필수
    step_type: str
    user_input: str
    temperature: Optional[float] = 0.72
    max_tokens: Optional[int] = 500
    insight_tag: Optional[str] = None
    system_prompt: Optional[str] = None

