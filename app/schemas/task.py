from pydantic import BaseModel
from uuid import UUID

class TaskRecommendBySessionRequest(BaseModel):
    session_id: UUID
    n: int = 3                # 추천 개수 상한(1~5 정도 권장)
    max_history_chars: int = 1000  # 대화 이력 축약 길이
    recent_steps_limit: int = 10   # 최근 스텝 조회 개수
