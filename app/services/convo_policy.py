# app/services/convo_policy.py
from sqlmodel import select, func, Session
from uuid import UUID
import os
from datetime import datetime

from app.db.session import get_session
from app.models.emotion import EmotionStep

MIN_TURNS = int(os.getenv("ACTIVITY_MIN_TURNS", "8"))
MAX_TURNS = int(os.getenv("ACTIVITY_MAX_TURNS", "10"))
FLAG_TAG = "activity_prompt_fired"

def _already_fired(db: Session, session_id: UUID) -> bool:
    q = select(EmotionStep.step_id).where(
        EmotionStep.session_id == session_id,
        EmotionStep.insight_tag == FLAG_TAG
    ).limit(1)
    return db.exec(q).first() is not None

def should_inject_activity(session_id: UUID) -> bool:
    with get_session() as db:
        if _already_fired(db, session_id):
            return False
        turns = db.exec(
            select(func.count(EmotionStep.step_id)).where(EmotionStep.session_id == session_id)
        ).one()
        return MIN_TURNS <= turns <= MAX_TURNS

def mark_activity_injected(session_id: UUID) -> None:
    # 중복 삽입 방지용 마킹 스텝 1개 저장
    with get_session() as db:
        if _already_fired(db, session_id):
            return
        step = EmotionStep(
            session_id=session_id,
            step_order=int(db.exec(
                select(func.count(EmotionStep.step_id)).where(EmotionStep.session_id == session_id)
            ).one()) + 1,
            step_type="system",
            user_input="",
            gpt_response="[activity_prompt_injected]",
            created_at=datetime.utcnow(),
            insight_tag=FLAG_TAG,
        )
        db.add(step)
        db.commit()
