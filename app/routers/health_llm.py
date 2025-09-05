from fastapi import APIRouter, HTTPException
from app.services.llm_service import generate_noa_response

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/llm")
def health_llm():
    try:
        txt = generate_noa_response(user_input="ping", recent_steps=[], system_prompt="너는 간단히 한 단어로만 대답해: pong")
        if txt and "pong" in txt.lower():
            return {"ok": True}
        return {"ok": False, "detail": "unexpected_content"}
    except Exception as e:
        raise HTTPException(503, f"llm_error: {e}")
