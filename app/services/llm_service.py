# app/services/llm_service.py
from __future__ import annotations

import os
import logging
from typing import Iterable, Optional, List, Dict, Any, Tuple

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))  # chat.completions 전용 명칭
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))  # seconds
USE_RESPONSES = os.getenv("LLM_USE_RESPONSES", "0") == "1"  # 강제 Responses 경로

# Responses API 계열(접두 기준)
_RESP_PREFIXES = (
    "gpt-5", "gpt-5o", "gpt-5o-mini", "gpt-5-mini",
    "o4", "o4-mini", "o3",
    "omni", "omni-mini", "omni-moderate",
)

__all__ = [
    "generate",
    "stream_noa_response",
    "generate_noa_response",
]


def _use_responses_api(model: str) -> bool:
    if USE_RESPONSES:
        return True
    return any(model.startswith(p) for p in _RESP_PREFIXES)


def _mk_prompt(messages: List[Dict[str, str]]) -> str:
    """Responses API용: role-tagged 텍스트로 합치기"""
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)


def generate(
    messages: List[Dict[str, str]],
    stream: bool = True,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Iterable[str]:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    stream=True면 토큰 단위로 str 델타를 yield, False면 최종 한 번만 yield
    """
    client = OpenAI(timeout=TIMEOUT)

    t = TEMPERATURE if temperature is None else float(temperature)
    tp = TOP_P if top_p is None else float(top_p)
    mt = MAX_TOKENS if max_tokens is None else int(max_tokens)

    if _use_responses_api(MODEL):
        # ===== Responses API 경로 =====
        prompt_text = _mk_prompt(messages)
        try:
            if stream:
                with client.responses.stream(
                    model=MODEL,
                    temperature=t,
                    top_p=tp,
                    # Responses는 max_completion_tokens 사용
                    max_completion_tokens=mt,
                    input={"type": "input_text", "text": prompt_text},
                ) as s:
                    for event in s:
                        # 안전하게 텍스트 델타만 집계
                        if getattr(event, "type", "") == "response.output_text.delta":
                            yield event.delta
                    s.close()
            else:
                res = client.responses.create(
                    model=MODEL,
                    temperature=t,
                    top_p=tp,
                    max_completion_tokens=mt,
                    input={"type": "input_text", "text": prompt_text},
                )
                yield getattr(res, "output_text", "") or ""
        except BadRequestError as e:
            logger.warning("LLM BadRequest (responses): %s", e)
            raise
    else:
        # ===== Chat Completions 경로 =====
        try:
            if stream:
                with client.chat.completions.stream(
                    model=MODEL,
                    temperature=t,
                    top_p=tp,
                    max_tokens=mt,  # chat은 max_tokens
                    messages=messages,
                ) as s:
                    for event in s:
                        if getattr(event, "type", "") == "content.delta":
                            yield event.delta
                    s.close()
            else:
                res = client.chat.completions.create(
                    model=MODEL,
                    temperature=t,
                    top_p=tp,
                    max_tokens=mt,
                    messages=messages,
                )
                text = (res.choices[0].message.content or "") if res.choices else ""
                yield text
        except BadRequestError as e:
            logger.warning("LLM BadRequest (chat): %s", e)
            raise


# ===== 호환 래퍼 (라우터들이 기대하는 인터페이스) =====

def _build_messages(
    system_prompt: Optional[str],
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
) -> List[Dict[str, str]]:
    """
    system → (옵션) task(system으로) → 대화(user/assistant) 순서로 메시지 구성
    """
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if task_prompt:
        # 작업 지시도 system 롤로 붙여 일관성 유지
        msgs.append({"role": "system", "content": task_prompt})

    for role, text in conversation:
        r = role if role in ("user", "assistant", "system") else "user"
        msgs.append({"role": r, "content": text or ""})
    return msgs


def stream_noa_response(
    *,
    system_prompt: Optional[str],
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> Iterable[str]:
    """
    WS에서 쓰는 스트리밍 인터페이스 (str 델타를 yield)
    """
    messages = _build_messages(system_prompt, task_prompt, conversation)
    return generate(
        messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_noa_response(
    *,
    system_prompt: Optional[str],
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> str:
    """
    HTTP 라우터 등에서 쓰는 비스트리밍 인터페이스 (최종 문자열 반환)
    """
    messages = _build_messages(system_prompt, task_prompt, conversation)
    # generate(stream=False)는 한 번만 yield하므로 join으로 안전 수집
    return "".join(
        generate(
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
