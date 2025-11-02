# app/services/llm_service.py
from __future__ import annotations

import os
import logging
from typing import Iterable, List, Tuple, AsyncGenerator, Dict, Any, Optional

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = os.getenv("LLM_TEMPERATURE", None)
# TEMPERATURE는 문자열일 수 있음 → float로 쓸 때만 변환
try:
    _DEFAULT_TEMPERATURE = float(TEMPERATURE) if TEMPERATURE not in (None, "",) else None
except Exception:
    _DEFAULT_TEMPERATURE = None

MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TOP_P = os.getenv("LLM_TOP_P", None)
try:
    _DEFAULT_TOP_P = float(TOP_P) if TOP_P not in (None, "",) else None
except Exception:
    _DEFAULT_TOP_P = None

TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
USE_RESPONSES = os.getenv("LLM_USE_RESPONSES", "1") == "1"  # Responses 우선 사용 여부

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=TIMEOUT)
    return _client


# ===== Model / Param helpers =====

def _is_reasoning_or_gpt5(model: str) -> bool:
    """
    reasoning 계열과 gpt-5 계열은 temperature/top_p 등이 미지원.
    (Azure/OpenAI reasoning 가이드 참조)
    """
    m = (model or "").lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or "reason" in m
    )


def _build_messages(system_prompt: str | None,
                    task_prompt: str | None,
                    conversation: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    OpenAI Responses/Chat 둘 다 호환되는 message 형식으로 구성
    """
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
    - reasoning/gpt-5: temperature/top_p 제거
    - token 파라미터 이름 정규화:
        * responses: max_output_tokens
        * chat:      max_completion_tokens
    """
    p = dict(params)

    # temperature/top_p 정리
    if _is_reasoning_or_gpt5(model):
        p.pop("temperature", None)
        p.pop("top_p", None)
    else:
        # 일반 모델일 때만 유효값 유지
        if p.get("temperature") is None:
            p.pop("temperature", None)
        if p.get("top_p") is None:
            p.pop("top_p", None)

    # tokens 파라미터 이름
    max_tokens = p.pop("max_tokens", None)
    if endpoint == "responses":
        if max_tokens is not None:
            p["max_output_tokens"] = max_tokens
    else:  # chat
        if max_tokens is not None:
            p["max_completion_tokens"] = max_tokens

    return p


def _iter_stream_chat(stream) -> Iterable[str]:
    """
    chat.completions.stream() → 텍스트 델타만 추출
    """
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
        except Exception:
            # 방어적 파싱
            pass


def _iter_stream_responses(stream) -> Iterable[str]:
    """
    responses.stream() → response.output_text.delta 이벤트만 텍스트로 추출
    """
    for event in stream:
        try:
            if event.type == "response.output_text.delta":
                piece = event.delta or ""
                if piece:
                    yield piece
            elif event.type == "response.error":
                raise BadRequestError(message=str(event.error), request=None, response=None)
        except Exception:
            # 이벤트 파싱 실패 시 다음 조각으로
            pass
    # 완료 이벤트 읽기
    try:
        _ = stream.get_final_response()
    except Exception:
        pass


# ===== Public API =====

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
    1) Responses.stream (권장)
    2) 실패 시 Chat Completions(stream) 폴백
    3) 최후: 최소 파라미터로 재시도
    """
    client = _client_singleton()
    m = (model or MODEL)
    messages = _build_messages(system_prompt, task_prompt, conversation)

    # ===== 1) Responses API 시도
    if USE_RESPONSES:
        try:
            base_params = {
                "model": m,
                "input": messages,  # Responses는 input에 messages 배열 허용
                "stream": True,
                "temperature": temperature if temperature is not None else _DEFAULT_TEMPERATURE,
                "top_p": _DEFAULT_TOP_P,
                "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
            }
            params = _normalize_for_model(base_params, m, endpoint="responses")

            with client.responses.stream(**params) as stream:
                for piece in _iter_stream_responses(stream):
                    yield piece
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
            "temperature": temperature if temperature is not None else _DEFAULT_TEMPERATURE,
            "top_p": _DEFAULT_TOP_P,
            "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        }
        params = _normalize_for_model(base_params, m, endpoint="chat")

        with client.chat.completions.stream(**params) as stream:
            for piece in _iter_stream_chat(stream):
                yield piece
        return
    except BadRequestError as e:
        logger.warning("Chat stream param retry: %s", _safe_err(e))
        # 최후: 최소 파라미터로 재시도
        try:
            last_params = {
                "model": m,
                "messages": messages,
                "stream": True,
            }
            # reasoning/gpt-5면 temp/top_p는 붙이지 않음
            if not _is_reasoning_or_gpt5(m):
                last_params["temperature"] = 1
            with client.chat.completions.stream(**last_params) as stream:
                for piece in _iter_stream_chat(stream):
                    yield piece
            return
        except Exception as e2:
            logger.error("Chat fallback failed: %s", _safe_err(e2))
            raise
    except Exception as e:
        logger.error("Chat stream failed: %s", _safe_err(e))
        raise


def generate_noa_response(
    *,
    system_prompt: str | None,
    task_prompt: str | None,
    conversation: List[Tuple[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    """
    단발(비스트림) 응답 생성. emotion.py가 import하는 동기 함수.
    Responses 우선 → Chat 폴백.
    """
    client = _client_singleton()
    m = (model or MODEL)
    messages = _build_messages(system_prompt, task_prompt, conversation)

    # 1) Responses
    if USE_RESPONSES:
        try:
            base_params = {
                "model": m,
                "input": messages,
                "temperature": temperature if temperature is not None else _DEFAULT_TEMPERATURE,
                "top_p": _DEFAULT_TOP_P,
                "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
            }
            params = _normalize_for_model(base_params, m, endpoint="responses")
            resp = client.responses.create(**params)
            # output_text 모아서 반환
            return "".join([o.content[0].text for o in getattr(resp, "output", []) if getattr(o, "content", [])])
        except Exception as e:
            logger.warning("Responses create failed; fallback to Chat | %s", _safe_err(e))

    # 2) Chat
    try:
        base_params = {
            "model": m,
            "messages": messages,
            "temperature": temperature if temperature is not None else _DEFAULT_TEMPERATURE,
            "top_p": _DEFAULT_TOP_P,
            "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        }
        params = _normalize_for_model(base_params, m, endpoint="chat")
        comp = client.chat.completions.create(**params)
        return (comp.choices[0].message.content or "")
    except BadRequestError as e:
        logger.warning("Chat create retry minimal: %s", _safe_err(e))
        try:
            comp = client.chat.completions.create(model=m, messages=messages)
            return (comp.choices[0].message.content or "")
        except Exception as e2:
            logger.error("Chat create failed: %s", _safe_err(e2))
            raise
    except Exception as e:
        logger.error("Chat create failed: %s", _safe_err(e))
        raise


def _safe_err(e: Exception) -> str:
    try:
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
