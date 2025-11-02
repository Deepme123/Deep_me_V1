# app/services/llm_service.py
from __future__ import annotations

import os
import logging
from typing import Iterable, List, Tuple, Optional, Generator

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ========= Config =========
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
DEFAULT_BACKUPS = (os.getenv("LLM_BACKUP_MODELS") or "gpt-4o-mini,gpt-4o").split(",")

# ========= Utils =========
def _is_reasoning(model: str) -> bool:
    """gpt-5*, o4*, o3* 계열은 Responses API를 우선 사용."""
    m = (model or "").lower()
    return m.startswith("gpt-5") or m.startswith("o4") or m.startswith("o3")

def _compose_system(system_prompt: str, task_prompt: Optional[str]) -> str:
    if task_prompt:
        return f"{system_prompt}\n\n---\n[Task Prompt]\n{task_prompt}"
    return system_prompt

def _to_responses_input(
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]]
) -> list:
    """
    Responses API용 blocks.
    content[].type 은 반드시 'input_text' 여야 함.
    """
    blocks = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": _compose_system(system_prompt, task_prompt)}],
        }
    ]
    for role, text in conversation:
        role = "user" if role == "user" else "assistant"
        blocks.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": text or ""}],
            }
        )
    return blocks

def _to_chat_messages(
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]]
) -> list:
    """Chat Completions용 messages 포맷."""
    msgs = [{"role": "system", "content": _compose_system(system_prompt, task_prompt)}]
    for role, text in conversation:
        r = "user" if role == "user" else "assistant"
        msgs.append({"role": r, "content": text or ""})
    return msgs

# ========= Public API =========
def stream_noa_response(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = 800,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    스트리밍 제너레이터.
    - 기본: Responses API (gpt-5*, o4*, o3* 계열)
    - 폴백: Chat Completions (백업 모델들)
    """
    mdl = (model or MODEL).strip()
    client = OpenAI(timeout=TIMEOUT)

    # 1) Responses API 경로 (권장: gpt-5-mini 등)
    if _is_reasoning(mdl):
        try:
            logger.info("LLM: streaming via Responses API (%s)", mdl)
            blocks = _to_responses_input(system_prompt, task_prompt, conversation)

            params = dict(
                model=mdl,
                input=blocks,
            )
            if max_tokens:
                # Responses API는 max_output_tokens
                params["max_output_tokens"] = int(max_tokens)

            # ✅ 파이썬 SDK는 stream 컨텍스트 매니저 사용
            with client.responses.stream(**params) as stream:
                any_delta = False
                for event in stream:
                    et = getattr(event, "type", None)
                    if et == "response.output_text.delta":
                        piece = getattr(event, "delta", None)
                        if piece:
                            any_delta = True
                            yield piece
                    elif et == "response.error":
                        err = getattr(event, "error", None)
                        raise BadRequestError(message=str(err) if err else "responses stream error")

                # 델타가 전혀 없었다면 최종 응답에서 보수적으로 보냄
                final = stream.get_final_response()
                try:
                    final_text = getattr(final, "output_text", None)
                except Exception:
                    final_text = None
                if final_text and not any_delta:
                    yield final_text
            return
        except BadRequestError as e:
            logger.warning("Responses stream failed; fallback to Chat | %s", str(e))
        except Exception as e:
            logger.warning("Responses stream exception; fallback to Chat | %s", str(e))

    # 2) Chat Completions 폴백 경로 (예: gpt-4o-mini)
    last_err: Optional[Exception] = None
    msgs = _to_chat_messages(system_prompt, task_prompt, conversation)

    for bk in [b.strip() for b in DEFAULT_BACKUPS if b.strip()]:
        try:
            logger.info("LLM: fallback via Chat Completions (%s)", bk)

            # 파라미터 호환: 일부 모델은 max_completion_tokens, 일부는 max_tokens
            chat_kwargs = dict(
                model=bk,
                messages=msgs,
                stream=True,
            )
            # temperature는 일반 챗 모델에서만 (gpt-5 계열 제외)
            if temperature is not None and not _is_reasoning(bk):
                chat_kwargs["temperature"] = float(temperature)

            # 1차 시도: 최신 규격
            if max_tokens:
                chat_kwargs["max_completion_tokens"] = int(max_tokens)

            try:
                chat_stream = client.chat.completions.create(**chat_kwargs)
            except BadRequestError as e:
                # 파라미터 호환 재시도 (max_tokens로 변경)
                if "max_completion_tokens" in str(e):
                    chat_kwargs.pop("max_completion_tokens", None)
                    if max_tokens:
                        chat_kwargs["max_tokens"] = int(max_tokens)
                    chat_stream = client.chat.completions.create(**chat_kwargs)
                else:
                    raise

            for chunk in chat_stream:
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    delta = None
                if delta:
                    yield delta
            return
        except Exception as e:
            last_err = e
            logger.error("Chat stream failed on %s: %s", bk, str(e))
            continue

    # 모든 경로 실패
    msg = f"LLM streaming failed for model={mdl}; last={last_err}"
    logger.error(msg)
    raise RuntimeError(msg)

def generate_noa_response(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = 800,
    model: Optional[str] = None,
) -> str:
    """
    레거시 동기식 합성 (호환 래퍼).
    stream_noa_response를 모아서 문자열로 반환.
    """
    buf: List[str] = []
    for piece in stream_noa_response(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        conversation=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    ):
        buf.append(piece)
    return "".join(buf)
