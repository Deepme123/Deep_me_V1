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

router = APIRouter(prefix="/emotion", tags=["Emotion"])

# ... (기존 create & generate 엔드포인트 유지) ...

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

@router.post("/steps/generate", response_model=EmotionStepRead)
def generate_emotion_step(
    input_data: EmotionStepGenerateInput,
    db: Session = Depends(get_session),
):
    response = generate_noa_response(input_data)
    new_step = EmotionStep(
        session_id=input_data.session_id,
        step_order=input_data.step_order,
        step_type="gpt_response",
        user_input=input_data.user_input,
        gpt_response=response,
        created_at=datetime.utcnow(),
        insight_tag=None
    )
    db.add(new_step)
    db.commit()
    db.refresh(new_step)
    return new_step
