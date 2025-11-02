# app/services/llm_service.py
from __future__ import annotations

import os
import logging
from typing import Iterable, List, Tuple, AsyncGenerator, Dict, Any, Optional

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
USE_RESPONSES = os.getenv("LLM_USE_RESPONSES", "1") == "1"  # Responses 우선 사용 여부

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=TIMEOUT)
    return _client


def _is_reasoning_or_gpt5(model: str) -> bool:
    m = (model or "").lower()
    # GPT-5 계열 + reasoning(o1/o3/o4 reasoning 라인 포함) 가정 분기
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or "reason" in m


def _build_messages(system_prompt: str | None,
                    task_prompt: str | None,
                    conversation: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if task_prompt:
        msgs.append({"role": "system", "content": task_prompt})
    for role, text in conversation:
        r = "user" if role == "user" else "assistant"
        msgs.append({"role": r, "content": text})
    return msgs


def _normalize_for_model(params: Dict[str, Any], model: str, *, endpoint: str) -> Dict[str, Any]:
    """
    endpoint: 'responses' | 'chat'
    - reasoning / gpt-5: temperature/top_p 제거
    - tokens 파라미터 이름 정규화
    """
    p = dict(params)
    if _is_reasoning_or_gpt5(model):
        p.pop("temperature", None)
        p.pop("top_p", None)

    # max tokens 정규화
    max_tokens = p.pop("max_tokens", None)
    if endpoint == "responses":
        if max_tokens is not None:
            p["max_output_tokens"] = max_tokens
    else:  # chat
        if max_tokens is not None:
            p["max_completion_tokens"] = max_tokens
    return p


async def stream_noa_response(
    *,
    system_prompt: str | None,
    task_prompt: str | None,
    conversation: List[Tuple[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    통합 스트리밍 제너레이터:
    - 1차: Responses.stream (권장)
    - 실패 시: Chat Completions(stream) 폴백
    - gpt-5/Reasoning 계열 파라미터 자동 정규화
    """
    client = _client_singleton()
    m = (model or MODEL)

    messages = _build_messages(system_prompt, task_prompt, conversation)

    # ===== 1) Responses API 시도
    if USE_RESPONSES:
        try:
            base_params = {
                "model": m,
                # Responses는 input에 messages 배열을 그대로 줄 수 있음
                "input": messages,
                "temperature": (temperature if temperature is not None else TEMPERATURE),
                "top_p": TOP_P,
                "max_tokens": (max_tokens if max_tokens is not None else MAX_TOKENS),
            }
            params = _normalize_for_model(base_params, m, endpoint="responses")

            with client.responses.stream(**params) as stream:
                for event in stream:
                    # 텍스트 델타만 흘려보냄
                    if event.type == "response.output_text.delta":
                        piece = event.delta or ""
                        if piece:
                            yield piece
                    elif event.type == "response.error":
                        raise BadRequestError(message=str(event.error), request=None, response=None)
                # 완료 이벤트 대기
                _ = stream.get_final_response()
            return
        except BadRequestError as e:
            logger.warning("Responses stream failed; fallback to Chat | %s", _safe_err(e))
        except Exception as e:
            logger.warning("Responses stream exception; fallback to Chat | %s", _safe_err(e))

    # ===== 2) Chat Completions 폴백
    try:
        base_params = {
            "model": m,
            "messages": messages,
            "stream": True,
            "temperature": (temperature if temperature is not None else TEMPERATURE),
            "top_p": TOP_P,
            "max_tokens": (max_tokens if max_tokens is not None else MAX_TOKENS),
        }
        params = _normalize_for_model(base_params, m, endpoint="chat")

        with client.chat.completions.stream(**params) as stream:
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                except Exception:
                    # 방어적 파싱
                    pass
            _ = stream.get_final_completion()
        return
    except BadRequestError as e:
        logger.warning("Chat stream param retry: %s", _safe_err(e))
        # 최후: temperature 제거 + tokens 제거 재시도(매우 보수적)
        try:
            last_params = {
                "model": m,
                "messages": messages,
                "stream": True,
            }
            if not _is_reasoning_or_gpt5(m):
                last_params["temperature"] = 1
            with client.chat.completions.stream(**last_params) as stream:
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
                    except Exception:
                        pass
                _ = stream.get_final_completion()
            return
        except Exception as e2:
            logger.error("Chat fallback failed: %s", _safe_err(e2))
            raise
    except Exception as e:
        logger.error("Chat stream failed: %s", _safe_err(e))
        raise


def _safe_err(e: Exception) -> str:
    try:
        # openai.BadRequestError는 .message/.response.json() 등 다양
        msg = getattr(e, "message", None) or str(e)
        detail = ""
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                j = resp.json()
                detail = f" | {j}"
            except Exception:
                pass
        return f"{msg}{detail}"
    except Exception:
        return str(e)
