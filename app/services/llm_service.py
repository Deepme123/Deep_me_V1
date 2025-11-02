from __future__ import annotations

import os
import logging
from typing import Iterable, List, Tuple, Dict, Any, Generator, Optional

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))  # seconds
USE_RESPONSES = os.getenv("LLM_USE_RESPONSES", "1") == "1"  # Responses API on/off

# 내부적으로 사용할 모델 패턴(대략적인 구분)
_O_SERIES_HINTS = ("o1", "o3", "o4", "omni", "gpt-5", "gpt-5o")

_client: Optional[OpenAI] = None


def _client_sync() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ------------------------------------------------------------------------------
# 공통: 대화/프롬프트 빌더
# ------------------------------------------------------------------------------

def _build_chat_messages(
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
) -> List[Dict[str, str]]:
    sys_text = system_prompt.strip()
    if task_prompt:
        sys_text = f"{sys_text}\n\n{task_prompt.strip()}"

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_text}]
    for role, text in conversation:
        r = "user" if role not in ("user", "assistant", "system") else role
        messages.append({"role": r, "content": text or ""})
    return messages


def _build_responses_input(
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Responses API 입력 포맷: role + content[{type:'input_text', text:...}]"""
    blocks: List[Dict[str, Any]] = []
    sys_text = system_prompt.strip()
    if task_prompt:
        sys_text = f"{sys_text}\n\n{task_prompt.strip()}"
    if sys_text:
        blocks.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": sys_text}],
            }
        )
    for role, text in conversation:
        r = "user" if role not in ("user", "assistant", "system") else role
        blocks.append(
            {"role": r, "content": [{"type": "input_text", "text": text or ""}]}
        )
    return blocks


# ------------------------------------------------------------------------------
# Chat Completions 스트리밍
# ------------------------------------------------------------------------------

def _stream_via_chat(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    client = _client_sync()
    messages = _build_chat_messages(system_prompt, task_prompt, conversation)

    # 최신 모델(o-계열, gpt-5 등)은 max_completion_tokens 권장
    params: Dict[str, Any] = dict(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        top_p=TOP_P,
        stream=True,
        timeout=TIMEOUT,
    )

    # 1차: max_completion_tokens로 시도
    try:
        params["max_completion_tokens"] = max_tokens
        stream = client.chat.completions.create(**params)
    except BadRequestError as e:
        # 일부 구형 계열은 max_completion_tokens 미지원 → max_tokens로 재시도
        logger.warning("Chat stream param retry: %s", getattr(e, "message", str(e)))
        params.pop("max_completion_tokens", None)
        params["max_tokens"] = max_tokens
        stream = client.chat.completions.create(**params)

    for chunk in stream:
        try:
            choice = chunk.choices[0]
            delta = getattr(choice.delta, "content", None)
            if delta:
                yield delta
        except Exception:
            # 스트림 내 빈 청크/툴콜 등은 조용히 스킵
            continue


# ------------------------------------------------------------------------------
# Responses API 스트리밍
# ------------------------------------------------------------------------------

def _stream_via_responses(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """
    NOTE:
    - 올바른 호출은 client.responses.create(..., stream=True)
    - 토큰 파라미터는 max_output_tokens
    """
    client = _client_sync()
    _input = _build_responses_input(system_prompt, task_prompt, conversation)

    stream = client.responses.create(
        model=MODEL,
        input=_input,
        temperature=temperature,
        top_p=TOP_P,
        stream=True,
        max_output_tokens=max_tokens,  # ★ Responses API는 이 이름
        timeout=TIMEOUT,
    )

    for event in stream:
        # 문서/SDK 기준으로 델타 이벤트는 response.output_text.delta
        etype = getattr(event, "type", "") or getattr(event, "event", "")
        if etype.endswith(".delta") and ("output_text" in etype or "text" in etype):
            delta = getattr(event, "delta", None)
            if delta:
                yield delta
        elif etype in ("response.error", "error"):
            # 오류 이벤트면 예외로 끊고 Chat 경로로 폴백하게 함
            emsg = getattr(event, "message", "responses stream error")
            raise RuntimeError(emsg)
        else:
            # 다른 이벤트는 무시 (created/done 등)
            continue


# ------------------------------------------------------------------------------
# 퍼블릭 API
# ------------------------------------------------------------------------------

def stream_noa_response(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Generator[str, None, None]:
    """
    동기 제너레이터. 외부(WS)는 이걸 받아서 async 래핑해 사용.
    우선 Responses를 시도하고 실패 시 Chat으로 폴백.
    """
    # 모델 힌트로 Responses 우선/후순위 가볍게 조정(환경변수 우선)
    use_responses = USE_RESPONSES

    # Responses 우선
    if use_responses:
        try:
            yield from _stream_via_responses(
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                conversation=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return
        except Exception as e:
            logger.warning("Responses stream failed; fallback to Chat | %s", str(e))

    # Chat 폴백
    yield from _stream_via_chat(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        conversation=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_noa_response(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    비스트리밍 동기 생성. REST 라우트 등에서 사용.
    """
    client = _client_sync()

    if USE_RESPONSES:
        try:
            _input = _build_responses_input(system_prompt, task_prompt, conversation)
            resp = client.responses.create(
                model=MODEL,
                input=_input,
                temperature=temperature,
                top_p=TOP_P,
                max_output_tokens=max_tokens,
                timeout=TIMEOUT,
            )
            return getattr(resp, "output_text", "") or ""
        except Exception as e:
            logger.warning("Responses create failed; fallback to Chat | %s", str(e))

    # Chat 경로
    messages = _build_chat_messages(system_prompt, task_prompt, conversation)
    params: Dict[str, Any] = dict(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        top_p=TOP_P,
        timeout=TIMEOUT,
    )
    try:
        params["max_completion_tokens"] = max_tokens
        comp = _client_sync().chat.completions.create(**params)
    except BadRequestError:
        params.pop("max_completion_tokens", None)
        params["max_tokens"] = max_tokens
        comp = _client_sync().chat.completions.create(**params)

    txt = (comp.choices[0].message.content or "") if comp.choices else ""
    return txt
