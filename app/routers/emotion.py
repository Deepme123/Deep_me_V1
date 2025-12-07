# app/routers/emotion.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from uuid import UUID
from datetime import datetime

from app.db.session import get_session
from app.models.emotion import EmotionSession, EmotionStep
from app.schemas.emotion import (
    EmotionSessionCreate,
    EmotionSessionRead,
    EmotionStepCreate,
    EmotionStepRead,
    EmotionStepGenerateInput,
)
from app.services.llm_service import generate_noa_response
from app.core.prompt_loader import get_system_prompt, get_task_prompt
from app.dependencies.auth import get_current_user
from app.services.convo_policy import (
    is_activity_turn,
    is_closing_turn,
    _turn_count,
    SESSION_MAX_TURNS,
    ACTIVITY_STEP_TYPE,
    _max_step_order,
)


router = APIRouter(prefix="/emotion", tags=["Emotion"])

@router.get("/sessions", response_model=list[EmotionSessionRead])
def list_sessions(
    db: Session = Depends(get_session),
    current_user: str = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    stmt = (
        select(EmotionSession)
        .where(EmotionSession.user_id == UUID(current_user))
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
    current_user: str = Depends(get_current_user),
):
    sess = db.get(EmotionSession, session_id)
    if not sess or sess.user_id != UUID(current_user):
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
    current_user: str = Depends(get_current_user),
):
    if session_data.user_id and session_data.user_id != UUID(current_user):
        raise HTTPException(status_code=403, detail="user_id mismatch")

    new_session = EmotionSession(
        **session_data.dict(exclude={"user_id"}),
        user_id=UUID(current_user),
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


@router.post("/steps", response_model=EmotionStepRead)
def create_emotion_step(
    step: EmotionStepCreate,
    db: Session = Depends(get_session),
    current_user: str = Depends(get_current_user),
):
    sess = db.get(EmotionSession, step.session_id)
    if not sess or sess.user_id != UUID(current_user):
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
    current_user: str = Depends(get_current_user),
):
    if input_data.session_id is None:
        raise HTTPException(status_code=400, detail="session_id is required")

    # 세션 존재 및 소유자 검증
    sess = db.get(EmotionSession, input_data.session_id)
    if not sess or sess.user_id != UUID(current_user):
        raise HTTPException(status_code=404, detail="session not found")

    # 🔒 한도 초과 가드 (LLM 호출 전에 차단)
    current_turns = _turn_count(db, input_data.session_id)
    if current_turns >= SESSION_MAX_TURNS:
        if not sess.ended_at:
            sess.ended_at = datetime.utcnow()
            db.add(sess)
            db.commit()
        raise HTTPException(status_code=409, detail="대화 세션이 종료되었어. 새 세션을 시작해줘.")

    # 최근 스텝 조회(역할 보존 전달)
    recent_all = db.exec(
        select(EmotionStep)
        .where(EmotionStep.session_id == input_data.session_id)
        .order_by(EmotionStep.step_order)
    ).all()

    # 시스템 프롬프트 조립
    system_prompt = get_system_prompt()
    activity_turn = is_activity_turn(
        user_text=input_data.user_input,
        db=db,
        session_id=input_data.session_id,
        steps=recent_all,
    )
    closing_turn = is_closing_turn(db, input_data.session_id)

    if activity_turn:
        system_prompt = f"{system_prompt}\n\n{get_task_prompt()}"

    if closing_turn:
        system_prompt = f"""{system_prompt}

[대화 마무리 지침](최우선)
- 아래 지침은 다른 모든 규칙보다 우선한다.
- 질문 금지. 요청하지 않은 과제 제안 금지. 이 메시지로 대화 종료.
- 핵심 요약 2줄
- 오늘 배운 1가지 강조
- 간단한 끝인사 1줄
"""

    # LLM 응답 생성
    response = generate_noa_response(
        input_data=input_data,
        system_prompt=system_prompt,
        recent_steps=recent_all,
    )

    # 스텝 저장(서버에서 step_order 부여) — WebSocket과 동일한 순서(user→assistant→activity)
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

    # 종료 턴이면 세션 종료 타임스탬프 설정
    if closing_turn and not sess.ended_at:
        sess.ended_at = datetime.utcnow()
        db.add(sess)

    db.commit()
    db.refresh(assistant_step)
    return assistant_step

