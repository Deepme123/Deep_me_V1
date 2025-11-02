# app/services/convo_policy.py
from __future__ import annotations

import os
from typing import List
from uuid import UUID
from datetime import datetime

from sqlalchemy import func
from sqlmodel import Session, select

from app.models.emotion import EmotionStep

# 정책 상수
ACTIVITY_STEP_TYPE = os.getenv("ACTIVITY_STEP_TYPE", "activity_suggest")
# 정책 상수 (단일 소스: POLICY_MAX_TURNS, 하위 호환: SESSION_MAX_TURNS)
POLICY_MAX_TURNS = int(
    os.getenv("POLICY_MAX_TURNS") or os.getenv("SESSION_MAX_TURNS", "20")
)
SESSION_MAX_TURNS = POLICY_MAX_TURNS

__all__ = [
    "is_activity_turn",
    "is_closing_turn",
    "mark_activity_injected",
    "_turn_count",
    "POLICY_MAX_TURNS",
    "SESSION_MAX_TURNS",  # ← 내보내기 추가
]


# ──────────────────────────────────────────────────────────────────────────────
# 내부 유틸

def _max_step_order(db: Session, session_id: UUID) -> int:
    """해당 세션의 마지막 step_order를 반환. 없으면 0."""
    last = db.exec(
        select(EmotionStep.step_order)
        .where(EmotionStep.session_id == session_id)
        .order_by(EmotionStep.step_order.desc())
        .limit(1)
    ).first()
    if isinstance(last, tuple):
        last = last[0] if last else 0
    return int(last or 0)

def _already_fired(db: Session, session_id: UUID) -> bool:
    """이미 액티비티 제안 플래그(step_type=ACTIVITY_STEP_TYPE)가 기록됐는지 확인."""
    hit = db.exec(
        select(EmotionStep.step_id).where(
            EmotionStep.session_id == session_id,
            EmotionStep.step_type == ACTIVITY_STEP_TYPE,
        )
    ).first()
    return hit is not None

# ──────────────────────────────────────────────────────────────────────────────
# 공개 API

def _turn_count(db: Session, session_id: UUID) -> int:
    """
    턴 수 계산. 간단히 user 스텝 수를 턴으로 본다.
    (emotion_ws에서 user→assistant 순으로 2개씩 추가하므로 대략적인 세션 길이 판단에 충분)
    """
    c = db.exec(
        select(func.count())
        .select_from(EmotionStep)
        .where(
            EmotionStep.session_id == session_id,
            EmotionStep.step_type == "user",
        )
    ).first()
    if isinstance(c, tuple):
        c = c[0]
    return int(c or 0)

def mark_activity_injected(db: Session, session_id: UUID) -> None:
    """
    액티비티(미션) 제안을 1회 했음을 DB에 남겨 중복 제안 방지.
    별도의 본문 없이 정책 마커 스텝을 남긴다.
    """
    next_order = _max_step_order(db, session_id) + 1
    marker = EmotionStep(
        session_id=session_id,
        step_order=next_order,
        step_type=ACTIVITY_STEP_TYPE,   # "activity_suggest"
        user_input=None,
        gpt_response=None,
        created_at=datetime.utcnow(),
        insight_tag=None,
    )
    db.add(marker)
    db.commit()

def is_activity_turn(
    user_text: str,
    db: Session,
    session_id: UUID,
    steps: List[EmotionStep],
) -> bool:
    """
    이번 턴에 액티비티 제안을 할지 정책 판단.
    규칙(초기 버전, 보수적):
        1) 이미 한 번 제안했다면 False
        2) 스텝이 전혀 없으면 False (웜업 대화 우선)
        3) 마지막 스텝이 분석/요약류면 True (예: 'analysis', 'insight', 'emotion_summary')
        4) 텍스트 트리거(간단 키워드): 우울/힘들/지치/무기력 등 → True
        5) 그 외는 False
    필요에 따라 프로젝트 규칙에 맞춰 확장하면 됨.
    """
    if _already_fired(db, session_id):
        return False

    if not steps:
        return False

    last = steps[-1]
    if getattr(last, "step_type", None) in ("analysis", "insight", "emotion_summary"):
        return True

    # 간단 트리거(한국어 기준 키워드 몇 가지)
    t = (user_text or "").lower()
    hard_triggers = (
        "우울", "힘들", "지치", "무기력", "번아웃", "아무것도 하기 싫", "무기력해", "피곤해 죽", "버티기 힘들"
    )
    if any(k in t for k in hard_triggers):
        return True

    return False

def is_closing_turn(db: Session, session_id: UUID) -> bool:
    """
    세션 종료 안내(권유)를 할지 여부.
    기본 정책: 턴 수가 최대치에 근접하면 True.
    - POLICY_MAX_TURNS - 1 이상이면 종료 권유 시도
    """
    turns = _turn_count(db, session_id)
    return turns >= max(1, POLICY_MAX_TURNS - 1)
