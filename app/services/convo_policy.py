from sqlmodel import select, func, Session
from uuid import UUID
import os
from datetime import datetime
from app.models.emotion import EmotionStep

MIN_TURNS = int(os.getenv("ACTIVITY_MIN_TURNS", "8"))
MAX_TURNS = int(os.getenv("ACTIVITY_MAX_TURNS", "10"))
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "20"))
BEFORE_END_GUARD = int(os.getenv("ACTIVITY_BEFORE_END_GUARD", "3"))
FLAG_TAG = "activity_prompt_fired"

def _already_fired(db: Session, session_id: UUID) -> bool:
    q = select(EmotionStep.step_id).where(
        EmotionStep.session_id == session_id,
        EmotionStep.insight_tag == FLAG_TAG
    ).limit(1)
    return db.exec(q).first() is not None

def _turn_count(db: Session, session_id: UUID) -> int:
    # system/마킹 스텝 제외한 “대화 턴”만 집계
    return int(db.exec(
        select(func.count(EmotionStep.step_id)).where(
            EmotionStep.session_id == session_id,
            EmotionStep.step_type != "system",
            EmotionStep.insight_tag != FLAG_TAG,
        )
    ).one())

def should_inject_activity(session_id: UUID, db: Session) -> bool:
    if _already_fired(db, session_id):
        return False
    existing = _turn_count(db, session_id)
    planned = existing + 1
    latest_allowed = max(1, SESSION_MAX_TURNS - BEFORE_END_GUARD)
    lower = MIN_TURNS
    upper = min(MAX_TURNS, latest_allowed)
    return lower <= planned <= upper

def mark_activity_injected(session_id: UUID, db: Session) -> None:
    if _already_fired(db, session_id):
        return
    step_count = _turn_count(db, session_id)
    db.add(EmotionStep(
        session_id=session_id,
        step_order=step_count + 1,
        step_type="system",
        user_input="",
        gpt_response="[activity_prompt_injected]",
        created_at=datetime.utcnow(),
        insight_tag=FLAG_TAG,
    ))
    db.commit()
