from sqlmodel import select, func, Session
from uuid import UUID
import os
from datetime import datetime
from app.models.emotion import EmotionStep

SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "20"))
TURNS_BEFORE_END = int(os.getenv("ACTIVITY_TURNS_BEFORE_END", "2"))
WINDOW = int(os.getenv("ACTIVITY_WINDOW", "1"))
FLAG_TAG = "activity_prompt_fired"

def _already_fired(db: Session, session_id: UUID) -> bool:
    q = select(EmotionStep.step_id).where(
        EmotionStep.session_id == session_id,
        EmotionStep.insight_tag == FLAG_TAG
    ).limit(1)
    return db.exec(q).first() is not None

def _turn_count(db: Session, session_id: UUID) -> int:
    # 사용자-응답 세트만 집계. system/마킹 스텝 제외.
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
    planned = existing + 1  # 이번 요청이 만들 턴
    target = max(1, SESSION_MAX_TURNS - TURNS_BEFORE_END)  # 예: 20-2=18
    lo = max(1, target - WINDOW + 1)
    hi = target
    return lo <= planned <= hi

def mark_activity_injected(session_id: UUID, db: Session) -> None:
    if _already_fired(db, session_id):
        return
    # 마킹은 system 스텝으로 별도 1행
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
