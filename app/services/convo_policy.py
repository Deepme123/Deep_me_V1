# app/services/convo_policy.py
from sqlmodel import select, func, Session
from uuid import UUID
import os
from datetime import datetime
from app.models.emotion import EmotionStep

SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "20"))
TURNS_BEFORE_END = int(os.getenv("ACTIVITY_TURNS_BEFORE_END", "2"))  # 예: 종료 2턴 전=18턴
WINDOW = int(os.getenv("ACTIVITY_WINDOW", "1"))  # 허용 오차(폭)
FLAG_TAG = "activity_prompt_fired"

def _turn_count(db: Session, session_id: UUID) -> int:
    return int(db.exec(
        select(func.count(EmotionStep.step_id)).where(
            EmotionStep.session_id == session_id,
            EmotionStep.step_type != "system",
            EmotionStep.insight_tag != FLAG_TAG,
        )
    ).one())

def _already_fired(db: Session, session_id: UUID) -> bool:
    return db.exec(select(EmotionStep.step_id).where(
        EmotionStep.session_id == session_id,
        EmotionStep.insight_tag == FLAG_TAG
    ).limit(1)).first() is not None

def is_activity_turn(session_id: UUID, db: Session) -> bool:
    if _already_fired(db, session_id): return False
    planned = _turn_count(db, session_id) + 1
    target = max(1, SESSION_MAX_TURNS - TURNS_BEFORE_END)   # 기본 18턴
    lo, hi = max(1, target - WINDOW + 1), target
    return lo <= planned <= hi

def mark_activity_injected(session_id: UUID, db: Session) -> None:
    if _already_fired(db, session_id): return
    step_no = _turn_count(db, session_id) + 1
    db.add(EmotionStep(
        session_id=session_id,
        step_order=step_no,
        step_type="system",
        user_input="",
        gpt_response="[activity_prompt_injected]",
        created_at=datetime.utcnow(),
        insight_tag=FLAG_TAG,
    ))
    db.commit()

def is_closing_turn(session_id: UUID, db: Session) -> bool:
    planned = _turn_count(db, session_id) + 1
    return planned == SESSION_MAX_TURNS
