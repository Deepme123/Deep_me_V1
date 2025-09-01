from typing import Optional, List, Generator
from openai import OpenAI
import os
import logging

from app.models.emotion import EmotionSession, EmotionStep
from app.core.prompt_loader import get_system_prompt

# ─────────────────────────────────────────────
# 환경변수 기반 설정 (없으면 안전한 기본값)
MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "30"))  # SDK에 따라 무시될 수 있음
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.3"))
FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.6"))

# OpenAI 클라이언트 (가능한 경우 timeout 옵션 사용)
try:
    client = OpenAI(timeout=TIMEOUT)
except TypeError:
    # 일부 버전은 timeout 옵션 미지원 → 기본 클라이언트 사용
    client = OpenAI()

logger = logging.getLogger(__name__)
# ※ 프롬프트/히스토리는 로그에 남기지 않는다!


def _condense_history(history: list[str], max_chars: int = 1000) -> str:
    """
    최근 대화(history)를 최대 길이에 맞춰 축약.
    """
    combined = "\n".join(history).strip()
    if len(combined) <= max_chars:
        return combined
    return "...\n" + combined[-max_chars:]


# app/services/llm_service.py
from typing import List, Dict, Optional
from app.core.prompt_loader import get_system_prompt

MAX_TURNS = 8  # user/assistant 쌍 기준 최근 N턴만 유지

def _build_messages(
    user_input: str,
    recent_steps: list,
    system_prompt: Optional[str] = None,
    meta: Optional[str] = None,
) -> List[Dict[str, str]]:
    sys_txt = system_prompt or get_system_prompt()
    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_txt}]
    if meta:
        messages.append({"role": "user", "content": meta})

    # ✅ 역할 보존. 문자열 합치기 금지. 과거 assistant를 user로 넣지 말 것.
    kept = recent_steps[-MAX_TURNS:] if MAX_TURNS else recent_steps
    for s in kept:
        if getattr(s, "user_input", None):
            messages.append({"role": "user", "content": s.user_input})
        if getattr(s, "gpt_response", None):
            messages.append({"role": "assistant", "content": s.gpt_response})

    # 최종 입력
    messages.append({"role": "user", "content": user_input})
    return messages

def _debug_log_messages(messages: List[Dict[str, str]], logger):
    # 첫 메시지가 system인지, ‘GPT:’ 같은 프리픽스가 user에 섞였는지 점검
    try:
        logger.debug("first_role=%s total=%d", messages[0]["role"], len(messages))
        for i, m in enumerate(messages[:6]):  # 앞부분만
            logger.debug("m[%d].role=%s len=%d", i, m["role"], len(m["content"]))
    except Exception:
        pass



# app/services/llm_service.py
def generate_noa_response(input_data, system_prompt=None):
    messages = _build_messages(
        user_input=input_data.user_input,
        recent_steps=input_data.recent_steps,  # 라우터에서 조회해 전달
        system_prompt=system_prompt,
        meta=None,
    )
    _debug_log_messages(messages, logger)

    resp = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=TEMPERATURE, max_completion_tokens=MAX_TOKENS,
        top_p=TOP_P, presence_penalty=PRESENCE_PENALTY, frequency_penalty=PENALTY,
    )
    return resp.choices[0].message.content



# app/services/llm_service.py
async def stream_noa_response(user_input, session, recent_steps, system_prompt=None):
    messages = _build_messages(
        user_input=user_input,
        recent_steps=recent_steps,
        system_prompt=system_prompt,
        meta=None,
    )
    _debug_log_messages(messages, logger)

    stream = client.chat.completions.create(
        model=MODEL, messages=messages, stream=True,
        temperature=TEMPERATURE, max_completion_tokens=MAX_TOKENS,
        top_p=TOP_P, presence_penalty=PRESENCE_PENALTY, frequency_penalty=FREQUENCY_PENALTY,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta
