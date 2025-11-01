# app/services/convo_policy.py

from typing import List
from uuid import UUID

from sqlmodel import select, Session

from app.models.emotion import EmotionStep  # 실제 경로에 맞게 조정해

ACTIVITY_STEP_TYPE = "activity_suggest"  # 너네가 쓰는 명칭에 맞춰

def _already_fired(db: Session, session_id: UUID) -> bool:
    """
    이 세션에서 이미 액티비티(미션) 한 번 보냈는지 확인.
    한 번 보냈으면 또 안 보낸다.
    """
    row = db.exec(
        select(EmotionStep.step_id).where(
            EmotionStep.session_id == session_id,
            EmotionStep.step_type == ACTIVITY_STEP_TYPE,
        )
    ).first()
    return row is not None


def is_activity_turn(
    user_text: str,
    db: Session,
    session_id: UUID,
    steps: List[EmotionStep],
) -> bool:
    """
    이번 턴에 액티비티(미션) 제안을 해야 할지 정책적으로 판단한다.

    규칙 예시:
    1) 이미 이 세션에서 한 번 보냈으면 다시 안 보낸다.
    2) 직전 스텝이 GPT 응답이 아니라 유저 입력이고, 감정 분류가 끝났으면 보낼 수 있다.
    3) 트리거가 되는 키워드가 있으면 우선 보낸다.
    실제 규칙은 프로젝트 진행하면서 더 채우면 됨.
    """
    # 1. 중복 방지
    if _already_fired(db, session_id):
        return False

    # 2. 스텝 기반 간단 룰 (예시)
    if not steps:
        # 첫 턴에는 굳이 미션 안 던진다
        return False

    last_step = steps[-1]

    # 예: 마지막 스텝이 "analysis"/"insight" 타입이면 그 다음에 미션 던질 수 있게
    if getattr(last_step, "step_type", None) in ("analysis", "insight", "emotion_summary"):
        return True

    # 3. 텍스트 트리거 예시
    lowered = user_text.lower()
    if "우울" in user_text or "살기" in user_text or "하기 싫" in user_text:
        # 이런 건 조금 더 조심스럽게 True/False 정해야 하는데, 일단 예시
        return True

    return False
