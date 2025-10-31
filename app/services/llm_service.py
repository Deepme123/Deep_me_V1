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
USE_RESPONSES = os.getenv("LLM_USE_RESPONSES", "0") == "1"  # reserved for switching APIs

_client = OpenAI(timeout=TIMEOUT)
__all__ = ["stream_noa_response", "generate_noa_response"]


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
        stream = _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=TOP_P,
            stream=True,
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
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=TOP_P,
                stream=False,
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
