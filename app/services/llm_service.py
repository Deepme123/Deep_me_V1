# app/services/llm_service.py
from __future__ import annotations

import os
import logging
from typing import Iterable, Generator, Optional
from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "30"))  # 요청 타임아웃(초)
CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", "600"))  # WS로 보낼 조각 크기

# ===== Utils =====
def _chunk_text(s: str, n: int) -> Iterable[str]:
    for i in range(0, len(s), n):
        yield s[i : i + n]


def _log_bad_request(prefix: str, err: BadRequestError) -> None:
    try:
        logger.error("%s | BadRequestError: %s", prefix, str(err))
        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                logger.error("%s | response.status=%s", prefix, resp.status_code)
            except Exception:
                pass
            try:
                logger.error("%s | response.json=%s", prefix, resp.json())
            except Exception:
                try:
                    logger.error("%s | response.text=%s", prefix, resp.text)
                except Exception:
                    pass
    except Exception:
        logger.exception("%s | failed to log BadRequestError", prefix)


def _safe_chat_create(client: OpenAI, *, model: str, messages: list[dict], stream: bool):
    """
    파라미터 호환성 이슈에 대비해 단계적으로 옵션을 줄이며 호출.
    1) temperature + top_p + max_completion_tokens
    2) temperature + max_completion_tokens
    3) max_completion_tokens
    4) (최소) 필수 인자만
    """
    attempts = [
        dict(model=model, messages=messages, stream=stream,
            temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS),
        dict(model=model, messages=messages, stream=stream,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
        dict(model=model, messages=messages, stream=stream,
            max_tokens=MAX_TOKENS),
        dict(model=model, messages=messages, stream=stream),
    ]
    last_err: Optional[Exception] = None
    for i, payload in enumerate(attempts, 1):
        try:
            return client.chat.completions.create(**payload)
        except BadRequestError as e:
            _log_bad_request(f"chat.create attempt#{i}", e)
            last_err = e
            # 스트리밍이 아예 불가(조직 미인증/모델 미지원)면 상위에서 폴백시키도록 즉시 재던짐
            if stream and "param" in str(e).lower() and "stream" in str(e).lower():
                raise
        except Exception as e:
            logger.exception("chat.create attempt#%d unexpected error", i)
            last_err = e
    if last_err:
        raise last_err


def _build_messages(*, system_prompt: str, recent_steps, user_input: str) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for step in recent_steps:
        if getattr(step, "user_input", None):
            msgs.append({"role": "user", "content": step.user_input})
        if getattr(step, "gpt_response", None):
            msgs.append({"role": "assistant", "content": step.gpt_response})
    msgs.append({"role": "user", "content": user_input})
    return msgs

# ===== Public API =====
async def stream_noa_response(*, user_input, session, recent_steps, system_prompt):
    client = OpenAI(timeout=TIMEOUT)
    messages = _build_messages(system_prompt=system_prompt, recent_steps=recent_steps, user_input=user_input)

    try:
        logger.info("LLM: streaming path selected (attempting stream=True)")
        stream = _safe_chat_create(client, model=MODEL, messages=messages, stream=True)

        yielded = False
        for event in stream:
            piece = _extract_text_from_stream_event(event)
            if piece:
                yielded = True
                yield piece

        if not yielded:
            logger.warning("LLM: stream yielded no content; falling back to non-streaming")
            resp = _safe_chat_create(client, model=MODEL, messages=messages, stream=False)
            text = _extract_text_from_chat_completion(resp).strip()
            if not text:
                # 👉 여기서 바로 예외를 던지지 말고, '백업 모델' 시도
                raise RuntimeError("empty_completion_from_llm")
            for chunk in _chunk_text(text, CHUNK_SIZE):
                yield chunk
        return


    except BadRequestError as e:
        emsg = str(e).lower()
        if "must be verified to stream this model" in emsg or ("'param': 'stream'" in emsg and "unsupported_value" in emsg):
            logger.warning("LLM: stream not allowed; falling back to non-streaming (policy)")
        else:
            raise
    except Exception:
        # ✅ 수정: 빈 'logger' 참조 대신 제대로 로그 남기기
        logger.exception("LLM: streaming failed unexpectedly; falling back to non-streaming")

    # (기존) 폴백 경로 유지
    logger.info("LLM: non-streaming fallback path selected (stream=False)")
    resp = _safe_chat_create(client, model=MODEL, messages=messages, stream=False)
    text = _extract_text_from_chat_completion(resp).strip()
    if not text:
        raise RuntimeError("empty_completion_from_llm")
    for chunk in _chunk_text(text, CHUNK_SIZE):
        yield chunk



def generate_noa_response(*, user_input: str, recent_steps, system_prompt: str) -> str:
    """
    하위 호환(동기/단발 응답). 최신 파라미터 규칙으로 단발 응답을 반환.
    """
    client = OpenAI(timeout=TIMEOUT)
    messages = _build_messages(system_prompt=system_prompt, recent_steps=recent_steps, user_input=user_input)
    resp = _safe_chat_create(client, model=MODEL, messages=messages, stream=False)
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        logger.exception("generate_noa_response: failed to extract content")
        return ""

# ✅ 추가: Chat Completions '단발' 응답 텍스트 추출
def _extract_text_from_chat_completion(resp) -> str:
    try:
        choice = getattr(resp, "choices", [None])[0]
        if not choice:
            return ""
        msg = getattr(choice, "message", None)
        finish_reason = getattr(choice, "finish_reason", None)

        # 1) content가 문자열
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        # 2) content가 파츠 배열 (멀티모달 텍스트 파트)
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text") or "")
            if "".join(parts).strip():
                return "".join(parts)

        # 3) 툴콜만 있는 케이스: 우리 서비스는 툴콜 미사용 → 빈본문으로 간주
        tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "function_call", None)
        if tool_calls:
            return ""  # (선택) "[tool_call]" 같은 표식을 넣어도 됨

        # 4) 안전필터 차단
        if finish_reason == "content_filter":
            raise RuntimeError("blocked_by_content_filter")

        return ""
    except Exception:
        logger.exception("extract_text: failed; returning empty")
        return ""


# ✅ 추가: Chat Completions '스트림' 이벤트에서 텍스트 추출
def _extract_text_from_stream_event(event) -> str:
    """
    표준: event.choices[0].delta.content
    일부 SDK/환경: event.delta / event.data 등 변형 가능 → 방어코드
    """
    try:
        # 표준 경로
        choices = getattr(event, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content

        # 방어적 파싱 (dict화 시도)
        if hasattr(event, "model_dump_json"):
            import json
            d = json.loads(event.model_dump_json())
        elif hasattr(event, "dict"):
            d = event.dict()
        else:
            d = getattr(event, "__dict__", {}) or {}

        # 델타 텍스트 위치 탐색
        for key in ("content", "text", "delta"):
            v = d.get(key)
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and isinstance(v.get("content"), str):
                return v["content"]

        return ""
    except Exception:
        logger.exception("extract_text_from_stream_event: failed")
        return ""
