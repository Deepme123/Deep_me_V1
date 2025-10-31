# app/schemas/emotion.py
from pydantic import BaseModel
from uuid import UUID
from typing import Optional, List
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
    max_completion_tokens: Optional[int] = 500
    insight_tag: Optional[str] = None
    system_prompt: Optional[str] = None



# ────────── WebSocket 요청 ──────────
class EmotionOpenRequest(BaseModel):
    type: str = "open"
    access_token: Optional[str] = None

class EmotionMessageRequest(BaseModel):
    type: str = "message"
    text: str

class EmotionCloseRequest(BaseModel):
    type: str = "close"
    emotion_label: Optional[str] = None
    topic: Optional[str] = None
    trigger_summary: Optional[str] = None
    insight_summary: Optional[str] = None

class TaskRecommendRequest(BaseModel):
    type: str = "task_recommend"
    max_items: Optional[int] = 5

# ────────── WebSocket 응답 ──────────
class EmotionOpenResponse(BaseModel):
    type: str = "open_ok"
    session_id: UUID
    turns: int

class EmotionMessageResponse(BaseModel):
    type: str
    delta: Optional[str] = None
    message: Optional[str] = None

class EmotionCloseResponse(BaseModel):
    type: str = "close_ok"

class TaskRecommendResponse(BaseModel):
    type: str = "task_recommend_ok"
    items: List[dict]
