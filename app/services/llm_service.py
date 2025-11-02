# app/services/llm_service.py
from __future__ import annotations

import os
import logging
from typing import Iterable, Optional, List, Dict, Any

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))  # chat.completions 전용
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))  # seconds
USE_RESPONSES = os.getenv("LLM_USE_RESPONSES", "0") == "1"  # 강제 Responses 경로

# Responses API 계열(접두 기준)
_RESP_PREFIXES = (
    "gpt-5", "gpt-5o", "gpt-5o-mini", "gpt-5-mini",
    "o4", "o4-mini", "o3",
    "omni", "omni-mini", "omni-moderate"
)


def _use_responses_api(model: str) -> bool:
    if USE_RESPONSES:
        return True
    return any(model.startswith(p) for p in _RESP_PREFIXES)


def _mk_prompt(messages: List[Dict[str, str]]) -> str:
    """Responses API용: role-tagged 텍스트로 합치기"""
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)


def generate(messages: List[Dict[str, str]], stream: bool = True) -> Iterable[str]:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    stream: True면 토큰 단위로 yield
    """
    client = OpenAI(timeout=TIMEOUT)

    if _use_responses_api(MODEL):
        # ===== Responses API 경로 =====
        prompt_text = _mk_prompt(messages)
        try:
            if stream:
                with client.responses.stream(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    # Responses는 max_completion_tokens 사용
                    max_completion_tokens=MAX_TOKENS,
                    input={"type": "input_text", "text": prompt_text},
                ) as s:
                    for event in s:
                        if event.type == "response.output_text.delta":
                            # event.delta: str 조각
                            yield event.delta
                    s.close()
            else:
                res = client.responses.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_completion_tokens=MAX_TOKENS,
                    input={"type": "input_text", "text": prompt_text},
                )
                # SDK 1.x: output_text 편의 접근자
                yield res.output_text or ""
        except BadRequestError as e:
            logger.warning("LLM BadRequest (responses): %s", e)
            raise
    else:
        # ===== Chat Completions 경로 =====
        try:
            if stream:
                with client.chat.completions.stream(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,  # chat은 max_tokens
                    messages=messages,
                ) as s:
                    for event in s:
                        # SDK 이벤트: content.delta
                        if event.type == "content.delta":
                            yield event.delta
                    s.close()
            else:
                res = client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                )
                text = (res.choices[0].message.content or "") if res.choices else ""
                yield text
        except BadRequestError as e:
            logger.warning("LLM BadRequest (chat): %s", e)
            raise
