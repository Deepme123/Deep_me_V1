# app/services/llm_service.py
from __future__ import annotations

from typing import Iterable, Generator, Optional
from openai import OpenAI, BadRequestError
import logging
import os

logger = logging.getLogger(__name__)

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))

CHUNK_SIZE = 200  # 비스트리밍 폴백 시 WS로 쪼개 보내는 크기

def _chunk_text(s: str, n: int) -> Iterable[str]:
    for i in range(0, len(s), n):
        yield s[i:i+n]

def _log_bad_request(prefix: str, err: BadRequestError) -> None:
    # SDK마다 속성 차이가 있어 안전하게 문자열화
    try:
        logger.error("%s | BadRequestError: %s", prefix, str(err))
        if getattr(err, "response", None) is not None:
            try:
                logger.error("%s | response.status=%s", prefix, err.response.status_code)
            except Exception:
                pass
            try:
                logger.error("%s | response.json=%s", prefix, err.response.json())
            except Exception:
                try:
                    logger.error("%s | response.text=%s", prefix, err.response.text)
                except Exception:
                    pass
    except Exception:
        logger.exception("%s | failed to log BadRequestError", prefix)

def _safe_chat_create(client: OpenAI, *, model: str, messages: list[dict], stream: bool) -> any:
    """
    파라미터 호환성 이슈에 대비해 단계적으로 옵션을 줄이며 호출.
    1) temperature + top_p + max_completion_tokens
    2) temperature + max_completion_tokens
    3) max_completion_tokens
    4) (최소) 파라미터 없이
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
            # 스트리밍 미인증/미지원이면 상위에서 폴백 처리하도록 그대로 raise
            if stream and "param" in str(e).lower() and "stream" in str(e).lower():
                raise
        except Exception as e:
            logger.exception("chat.create attempt#%d unexpected error", i)
            last_err = e
    if last_err:
        raise last_err

async def stream_noa_response(*, user_input, session, recent_steps, system_prompt) -> Generator[str, None, None]:
    """
    스트리밍 우선 시도 → 스트리밍 불가(조직 미인증/모델 미지원 등) 시 비스트리밍 폴백.
    단계적 파라미터 축소를 통해 400 원인(unsupported_parameter 등) 회피/진단.
    """
    client = OpenAI()

    # 메시지 조립
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for step in recent_steps:
        if step.user_input:
            messages.append({"role": "user", "content": step.user_input})
        if step.gpt_response:
            messages.append({"role": "assistant", "content": step.gpt_response})
    messages.append({"role": "user", "content": user_input})

    # 1) 스트리밍 시도
    try:
        stream = _safe_chat_create(client, model=MODEL, messages=messages, stream=True)
        for event in stream:
            try:
                choice = getattr(event, "choices", [None])[0]
                if not choice:
                    continue
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    yield content
            except Exception:
                # 혹시 이벤트 파싱 실패해도 스트림은 계속
                logger.exception("stream event parse error")
        return
    except BadRequestError as e:
        # 스트리밍 미지원/미인증 메시지면 폴백 진행
        emsg = str(e).lower()
        if "must be verified to stream this model" in emsg or ("'param': 'stream'" in emsg and "unsupported_value" in emsg):
            logger.warning("stream not allowed; falling back to non-streaming")
        else:
            # 다른 400이면 상위에서 처리
            raise
    except Exception:
        logger.exception("stream attempt failed for non-BadRequestError; falling back")

    # 2) 비스트리밍 폴백
    try:
        resp = _safe_chat_create(client, model=MODEL, messages=messages, stream=False)
        # SDK 응답 객체에서 content 추출
        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            logger.exception("failed to read completion content; defaulting to empty")
            text = ""
        for piece in _chunk_text(text, CHUNK_SIZE):
            yield piece
    except Exception:
        logger.exception("non-streaming fallback failed")
        # 실패 시엔 상위에서 에러 메시지를 전송하도록 예외를 다시 던지지 않고 빈 응답 종료
        return
