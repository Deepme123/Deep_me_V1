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
        dict(model=model, messages=messages, stream=stream, temperature=TEMPERATURE, top_p=TOP_P, max_completion_tokens=MAX_TOKENS),
        dict(model=model, messages=messages, stream=stream, temperature=TEMPERATURE, max_completion_tokens=MAX_TOKENS),
        dict(model=model, messages=messages, stream=stream, max_completion_tokens=MAX_TOKENS),
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

        yielded = False  # ✅ 추가: 한 토큰이라도 보냈는지 추적
        for event in stream:
            logger.debug("stream event raw: %s", event)
            try:
                choice = getattr(event, "choices", [None])[0]
                if not choice:
                    continue
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    yielded = True
                    yield content
            except Exception:
                logger.exception("stream event parse error")

        # ✅ 추가: 스트리밍에서 아무 토큰도 못 받았을 때 비-스트리밍 폴백
        if not yielded:
            logger.warning("LLM: stream yielded no content; falling back to non-streaming")
            resp = _safe_chat_create(client, model=MODEL, messages=messages, stream=False)
            try:
                text = (resp.choices[0].message.content or "").strip()
            except Exception:
                logger.exception("failed to extract completion content")
                text = ""

            if not text:
                raise RuntimeError("empty_completion_from_llm")

            for piece in _chunk_text(text, CHUNK_SIZE):
                yield piece

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
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        logger.exception("failed to extract completion content")
        text = ""
    text = text.strip()
    if not text:
        raise RuntimeError("empty_completion_from_llm")
    for piece in _chunk_text(text, CHUNK_SIZE):
        yield piece


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
