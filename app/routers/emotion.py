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
from app.services.convo_policy import (
    is_activity_turn,
    is_closing_turn,
    mark_activity_injected,
)

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
    # 수동 저장 엔드포인트
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

    # 시스템 프롬프트 조립
    system_prompt = get_system_prompt()
    activity_turn = is_activity_turn(input_data.session_id, db)
    closing_turn = is_closing_turn(input_data.session_id, db)

    if activity_turn:
        system_prompt = f"{system_prompt}\n\n{get_task_prompt()}"
        mark_activity_injected(input_data.session_id, db)

    if closing_turn:
        system_prompt = f"""{system_prompt}

[대화 마무리 지침](최우선)
- 아래 지침은 다른 모든 규칙보다 우선한다.
- 질문 금지. 요청하지 않은 과제 제안 금지. 이 메시지로 대화 종료.
- 핵심 요약 2줄
- 오늘 배운 1가지 강조
- 간단한 끝인사 1줄
"""

    # 최근 스텝 조회(역할 보존 전달)
    recent_all = db.exec(
        select(EmotionStep)
        .where(EmotionStep.session_id == input_data.session_id)
        .order_by(EmotionStep.step_order)
    ).all()

    # LLM 응답 생성(최근 스텝을 llm_service로 전달)
    response = generate_noa_response(
        input_data=input_data,
        system_prompt=system_prompt,
        recent_steps=recent_all,  # llm_service에서 역할 보존 메시지로 빌드
    )

    # 스텝 저장(클라이언트 step_order 불신, 서버에서 다음 번호 부여)
    next_order = (recent_all[-1].step_order + 1) if recent_all else 1
    new_step = EmotionStep(
        session_id=input_data.session_id,
        step_order=next_order,
        step_type="gpt_response",
        user_input=input_data.user_input,
        gpt_response=response,
        created_at=datetime.utcnow(),
        insight_tag=None,
    )
    db.add(new_step)

    # 종료 턴이면 세션 종료 타임스탬프 설정
    if closing_turn and not sess.ended_at:
        sess.ended_at = datetime.utcnow()
        db.add(sess)

    db.commit()
    db.refresh(new_step)
    return new_step
