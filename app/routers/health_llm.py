# app/routers/health_llm.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.llm_service import generate_noa_response, stream_noa_response

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/llm")
def health_llm(q: Optional[str] = Query(None, description="테스트용 프롬프트(없으면 기본 pong 검사)")):
    """
    비-스트리밍 단발 호출 점검.
    기본 프롬프트: 'pong'을 한 단어로만 답하도록 유도.
    - OK: {"ok": true, "text": "..."}
    - 비정상(내용없음/예상외 응답): 503
    """
    prompt = q or "너는 간단히 한 단어로만 대답해: pong"
    # recent_steps는 헬스체크라 빈 리스트로
    text = generate_noa_response(user_input=prompt, recent_steps=[], system_prompt="(healthcheck)")
    text = (text or "").strip()

    if not text:
        # llm_service에서 백업 모델까지 폴백했는데도 빈 응답
        raise HTTPException(status_code=503, detail="llm_empty_response")

    # 기본 프롬프트일 때 'pong' 포함 여부로 간단 점검
    if q is None and "pong" not in text.lower():
        return {"ok": False, "detail": "unexpected_content", "text": text}

    return {"ok": True, "text": text}


@router.get("/llm/stream")
async def health_llm_stream(q: Optional[str] = Query(None, description="테스트용 프롬프트(없으면 기본 pong 검사)")):
    """
    스트리밍 경로 점검.
    - 내부적으로 async generator를 순회해 토큰을 모았다가 반환
    - OK: {"ok": true, "tokens": N, "text": "..."}
    - 비정상: 503
    """
    prompt = q or "너는 간단히 한 단어로만 대답해: pong"

    tokens: list[str] = []
    try:
        async for piece in stream_noa_response(
            user_input=prompt,
            session=None,           # 헬스체크라 실제 세션 필요 없음
            recent_steps=[],        # 빈 히스토리
            system_prompt="(healthcheck-stream)",
        ):
            if piece:
                tokens.append(piece)
    except RuntimeError as e:
        msg = str(e)
        if msg == "blocked_by_content_filter":
            raise HTTPException(status_code=503, detail="blocked_by_content_filter")
        if msg == "empty_completion_from_llm":
            raise HTTPException(status_code=503, detail="llm_empty_response")
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"llm_stream_error: {e}")

    text = ("".join(tokens)).strip()
    if not text:
        raise HTTPException(status_code=503, detail="llm_stream_empty")

    if q is None and "pong" not in text.lower():
        return {"ok": False, "detail": "unexpected_content", "tokens": len(tokens), "text": text}

    return {"ok": True, "tokens": len(tokens), "text": text}
