# app/services/llm_service.py
from __future__ import annotations

import os
import logging
from typing import Optional, List, Tuple, Dict, Any, Iterable

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))  # seconds
USE_RESPONSES = os.getenv("LLM_USE_RESPONSES", "0") == "1"  # reserved

# Responses-스타일(또는 신형) 모델의 관측상 prefix
_RESPONSES_STYLE_MODELS = (
    "gpt-5", "gpt-5o", "gpt-5-mini", "o4", "o4-mini", "o3", "omni", "omni-moderate"
)

_client = OpenAI(timeout=TIMEOUT)
__all__ = ["stream_noa_response", "generate_noa_response"]

# ----- 유틸 -----

def _needs_max_completion_tokens(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _RESPONSES_STYLE_MODELS)

def _chat_create_with_token_fallback(client: OpenAI, **base_kwargs):
    """
    chat.completions 호출 시 max_tokens ↔ max_completion_tokens 자동 전환.
    - 1차: 기존 인자 그대로 시도(없으면 모델 특성에 따라 키 선택)
    - 400에서 'Use "max_completion_tokens"' 문구가 오면 키를 바꿔서 재시도
    """
    kw = dict(base_kwargs)

    # 키가 둘 다 없는 경우, 모델 특성 보고 기본 키 선택
    if "max_tokens" not in kw and "max_completion_tokens" not in kw:
        if _needs_max_completion_tokens(str(kw.get("model", ""))):
            kw["max_completion_tokens"] = kw.pop("max_tokens", None) or MAX_TOKENS
        else:
            kw["max_tokens"] = kw.pop("max_completion_tokens", None) or MAX_TOKENS

    try:
        return client.chat.completions.create(**kw)
    except BadRequestError as e:
        msg = str(e)
        # 서버가 직접 교정 힌트를 준 경우
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            # max_tokens 제거하고 max_completion_tokens로 재시도
            val = kw.pop("max_tokens", None)
            if "max_completion_tokens" not in kw:
                kw["max_completion_tokens"] = val or MAX_TOKENS
            return client.chat.completions.create(**kw)
        raise

def _build_messages(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """
    conversation: [("user"|"assistant", text), ...]
    """
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if task_prompt:
        messages.append({"role": "system", "content": task_prompt})

    for role, text in conversation:
        r = role if role in ("assistant", "system") else "user"
        messages.append({"role": r, "content": text})

    return messages

def stream_noa_response(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Iterable[str]:
    """
    동기 제너레이터: 텍스트 청크를 yield.
    emotion_ws.py의 _iter_chunks가 sync/async 모두 감싸주므로 그대로 사용 가능.
    """
    messages = _build_messages(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        conversation=conversation,
    )

    try:
        # 스트리밍 호출 (토큰 키 자동 전환)
        stream = _chat_create_with_token_fallback(
            _client,
            model=MODEL,
            messages=messages,
            temperature=temperature,
            top_p=TOP_P,
            stream=True,
            max_tokens=max_tokens,  # 필요 시 헬퍼가 max_completion_tokens로 바꿔줌
        )

        for chunk in stream:
            try:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    yield content
            except Exception as e:
                logger.debug("stream chunk skip", extra={"error": str(e)})
                continue

    except BadRequestError as e:
        logger.warning("OpenAI BadRequestError (fallback to non-stream)", extra={"error": str(e)})
        try:
            # 폴백(비스트리밍)도 동일 헬퍼 사용
            resp = _chat_create_with_token_fallback(
                _client,
                model=MODEL,
                messages=messages,
                temperature=temperature,
                top_p=TOP_P,
                stream=False,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            if text:
                yield text
        except Exception as e2:
            logger.error("OpenAI fallback failed", extra={"error": str(e2)})
            raise
    except Exception as e:
        logger.error("OpenAI stream error", extra={"error": str(e)})
        raise

def generate_noa_response(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """스트리밍 제너레이터를 모아 하나의 문자열로 반환."""
    chunks: List[str] = []
    for piece in stream_noa_response(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        conversation=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        chunks.append(piece)
    return "".join(chunks)
