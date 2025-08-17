# app/routers/emotion.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from uuid import UUID
from datetime import datetime

from app.db.session import get_session
from app.models.emotion import EmotionSession, EmotionStep
from app.schemas.emotion import (
    EmotionSessionCreate, EmotionSessionRead,
    EmotionStepCreate, EmotionStepRead,
    EmotionStepGenerateInput,
)
from app.services.llm_service import generate_noa_response
from app.core.prompt_loader import get_system_prompt, get_task_prompt
from app.services.convo_policy import should_inject_activity, mark_activity_injected

router = APIRouter(prefix="/emotion", tags=["Emotion"])


@router.get("/sessions", response_model=list[EmotionSessionRead])
def list_sessions(
    user_id: UUID = Query(...),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session),
):
    stmt = (
        select(EmotionSession)
        .where(EmotionSession.user_id == user_id)
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
):
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
):
    new_session = EmotionSession(**session_data.dict())
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


@router.post("/steps", response_model=EmotionStepRead)
def create_emotion_step(
    step: EmotionStepCreate,
    db: Session = Depends(get_session),
):
    # 일반 수동 저장용 엔드포인트(필요 시 유지)
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
):
    # 세션 존재 검증
    sess = db.get(EmotionSession, input_data.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")

    # 기본 시스템 프롬프트
    system_prompt = get_system_prompt()

    # 활동과제 프롬프트 조건부 주입(8~10턴 구간 1회)
    inject = should_inject_activity(input_data.session_id, db)
    if inject:
        mark_activity_injected(input_data.session_id, db)

    # LLM 응답 생성
    response = generate_noa_response(input_data, system_prompt=system_prompt)

    # 스텝 저장
    new_step = EmotionStep(
        session_id=input_data.session_id,
        step_order=input_data.step_order,
        step_type="gpt_response",
        user_input=input_data.user_input,
        gpt_response=response,
        created_at=datetime.utcnow(),
        insight_tag=None,
    )
    db.add(new_step)
    db.commit()
    db.refresh(new_step)

    # 주입되었으면 마킹 스텝 기록(중복 방지 로직 내부 포함)
    if inject:
        mark_activity_injected(input_data.session_id)

    return new_step
