# app/routers/emotion.py
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.core.prompt_loader import get_system_prompt, get_task_prompt
from app.db.session import get_session
from app.dependencies.auth import get_current_user_optional
from app.models.emotion import EmotionSession, EmotionStep
from app.services.step_manager import (
    build_end_session_context,
    build_fixed_farewell,
    build_soft_timeout_hint,
    build_step_context,
    extract_end_session_marker,
    step_for_prompt,
)
from app.schemas.emotion import (
    EmotionSessionCreate,
    EmotionSessionRead,
    EmotionStepCreate,
    EmotionStepGenerateInput,
    EmotionStepRead,
)
from app.services.convo_policy import (
    ACTIVITY_STEP_TYPE,
    _max_step_order,
    is_activity_turn,
)
from app.services.llm_service import generate_noa_response
from app.services.web_test_user import resolve_emotion_user_id

router = APIRouter(prefix="/emotion", tags=["Emotion"])


def _steps_to_conversation(steps: list[EmotionStep]) -> list[tuple[str, str]]:
    convo: list[tuple[str, str]] = []
    for s in steps:
        if s.step_type == "user" and s.user_input:
            convo.append(("user", s.user_input))
        elif s.step_type == "assistant" and s.gpt_response:
            convo.append(("assistant", s.gpt_response))
    return convo


def _emotion_user_id(
    db: Session = Depends(get_session),
    current_user: str | None = Depends(get_current_user_optional),
) -> UUID:
    return resolve_emotion_user_id(db, current_user)


@router.get("/sessions", response_model=list[EmotionSessionRead])
def list_sessions(
    db: Session = Depends(get_session),
    emotion_user_id: UUID = Depends(_emotion_user_id),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    stmt = (
        select(EmotionSession)
        .where(EmotionSession.user_id == emotion_user_id)
        .order_by(EmotionSession.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return db.exec(stmt).all()


@router.get("/steps", response_model=list[EmotionStepRead])
def list_steps(
    session_id: UUID = Query(...),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session),
    emotion_user_id: UUID = Depends(_emotion_user_id),
):
    sess = db.get(EmotionSession, session_id)
    if not sess or sess.user_id != emotion_user_id:
        raise HTTPException(status_code=404, detail="session not found")

    stmt = (
        select(EmotionStep)
        .where(EmotionStep.session_id == session_id)
        .order_by(EmotionStep.step_order)
        .limit(limit)
        .offset(offset)
    )
    return db.exec(stmt).all()


@router.post("/sessions", response_model=EmotionSessionRead)
def create_emotion_session(
    session_data: EmotionSessionCreate,
    db: Session = Depends(get_session),
    emotion_user_id: UUID = Depends(_emotion_user_id),
):
    if session_data.user_id and session_data.user_id != emotion_user_id:
        raise HTTPException(status_code=403, detail="user_id mismatch")

    new_session = EmotionSession(
        **session_data.dict(exclude={"user_id"}),
        user_id=emotion_user_id,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


@router.post("/steps", response_model=EmotionStepRead)
def create_emotion_step(
    step: EmotionStepCreate,
    db: Session = Depends(get_session),
    emotion_user_id: UUID = Depends(_emotion_user_id),
):
    sess = db.get(EmotionSession, step.session_id)
    if not sess or sess.user_id != emotion_user_id:
        raise HTTPException(status_code=404, detail="session not found")

    new_step = EmotionStep(
        session_id=step.session_id,
        step_order=step.step_order,
        step_type=step.step_type,
        user_input=step.user_input,
        gpt_response=step.gpt_response,
        created_at=datetime.utcnow(),
        insight_tag=step.insight_tag,
    )
    db.add(new_step)
    db.commit()
    db.refresh(new_step)
    return new_step


@router.post("/steps/generate", response_model=EmotionStepRead)
def generate_emotion_step(
    input_data: EmotionStepGenerateInput,
    db: Session = Depends(get_session),
    emotion_user_id: UUID = Depends(_emotion_user_id),
):
    if input_data.session_id is None:
        raise HTTPException(status_code=400, detail="session_id is required")

    sess = db.get(EmotionSession, input_data.session_id)
    if not sess or sess.user_id != emotion_user_id:
        raise HTTPException(status_code=404, detail="session not found")

    # ?”’ ?œë„ ì´ˆê³¼ ê°€??(LLM ?¸ì¶œ ?„ì— ì°¨ë‹¨)
    recent_all = db.exec(
        select(EmotionStep)
        .where(EmotionStep.session_id == input_data.session_id)
        .order_by(EmotionStep.step_order)
    ).all()

    # ?œìŠ¤???„ë¡¬?„íŠ¸ ì¡°ë¦½
    current_step = step_for_prompt(recent_all, input_data.user_input)
    system_prompt = get_system_prompt()
    step_context = build_step_context(current_step)
    soft_timeout_hint = build_soft_timeout_hint(recent_all, input_data.user_input)
    system_prompt = f"{system_prompt}\n\n{step_context}"
    if soft_timeout_hint:
        system_prompt = f"{system_prompt}\n\n{soft_timeout_hint}"
    end_session_context = build_end_session_context(current_step)
    if end_session_context:
        system_prompt = f"{system_prompt}\n\n{end_session_context}"
    activity_turn = is_activity_turn(
        user_text=input_data.user_input,
        db=db,
        session_id=input_data.session_id,
        steps=recent_all,
    )
    task_prompt = get_task_prompt() if activity_turn else None

    # LLM ?‘ë‹µ ?ì„±
    convo = _steps_to_conversation(recent_all) + [("user", input_data.user_input)]
    response = generate_noa_response(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        conversation=convo,
        temperature=input_data.temperature,
        max_tokens=input_data.max_completion_tokens,
    )
    response, end_by_token = extract_end_session_marker(response)
    if current_step >= 11 or end_by_token:
        response = build_fixed_farewell()
        end_by_token = True

    # ?¤í… ?€???œë²„?ì„œ step_order ë¶€?? ??WebSocketê³??™ì¼???œì„œ(user?’assistant?’activity)
    current_max_order = _max_step_order(db, input_data.session_id)
    next_order = current_max_order + 1
    user_step = EmotionStep(
        session_id=input_data.session_id,
        step_order=next_order,
        step_type="user",
        user_input=input_data.user_input,
        gpt_response="",
        created_at=datetime.utcnow(),
        insight_tag=input_data.insight_tag,
    )
    assistant_step = EmotionStep(
        session_id=input_data.session_id,
        step_order=next_order + 1,
        step_type="assistant",
        user_input="",
        gpt_response=response,
        created_at=datetime.utcnow(),
        insight_tag=None,
    )
    db.add(user_step)
    db.add(assistant_step)

    if activity_turn:
        marker = EmotionStep(
            session_id=input_data.session_id,
            step_order=next_order + 2,
            step_type=ACTIVITY_STEP_TYPE,
            user_input="",
            gpt_response="",
            created_at=datetime.utcnow(),
            insight_tag=None,
        )
        db.add(marker)

    # ì¢…ë£Œ ?´ì´ë©??¸ì…˜ ì¢…ë£Œ ?€?„ìŠ¤?¬í”„ ?¤ì •
    if (current_step >= 11 or end_by_token) and not sess.ended_at:
        sess.ended_at = datetime.utcnow()
        db.add(sess)

    db.commit()
    db.refresh(assistant_step)
    return assistant_step
