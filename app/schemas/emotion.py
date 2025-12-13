# app/schemas/emotion.py
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EmotionSessionCreate(BaseModel):
    user_id: Optional[UUID] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    emotion_label: Optional[str] = None
    topic: Optional[str] = None
    trigger_summary: Optional[str] = None
    insight_summary: Optional[str] = None


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


class EmotionStepCreate(BaseModel):
    session_id: UUID
    step_order: int
    step_type: str
    user_input: str
    gpt_response: str
    created_at: Optional[datetime] = None
    insight_tag: Optional[str] = None


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


class EmotionStepGenerateInput(BaseModel):
    session_id: Optional[UUID] = None      # ?ЖьЬ╝ый??РыПЩ ?ЭьД▒
    user_id: Optional[UUID] = None         # ???╕ьЕШ???ДьИШ
    step_type: str
    user_input: str
    temperature: Optional[float] = 0.72
    max_completion_tokens: Optional[int] = 500
    insight_tag: Optional[str] = None
    system_prompt: Optional[str] = None


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
