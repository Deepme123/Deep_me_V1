from __future__ import annotations

import os
import logging
from typing import Iterable, List, Tuple, Optional, Generator

from openai import OpenAI, BadRequestError

logger = logging.getLogger(__name__)

# ===== Config =====
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# 아래 두 값은 "일반 챗 모델"에서만 실제로 사용됨 (추론형 모델은 무시)
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))

# 공통적으로 쓰는 논리적 "목표 토큰". API별로 이름이 다르므로 내부에서 매핑.
MAX_TOKENS_DEFAULT = int(os.getenv("LLM_MAX_TOKENS", "800"))

TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "60"))  # seconds
USE_RESPONSES_FIRST = os.getenv("LLM_USE_RESPONSES", "1") == "1"

client = OpenAI()

# 추론형(Reasoning) 모델 식별: gpt-5 계열, o3/o4 계열 등
_REASONING_PREFIXES = ("gpt-5", "o3", "o4", "omni")

def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _REASONING_PREFIXES)

def _merge_system(system_prompt: Optional[str], task_prompt: Optional[str]) -> str:
    # 추론형 모델 권고: system / developer 동시 사용 지양 → system 하나로 합침
    # (Azure 문서 기준 가이드. Responses에서도 일관되게 안전하게 운영) :contentReference[oaicite:4]{index=4}
    system_prompt = (system_prompt or "").strip()
    task_prompt = (task_prompt or "").strip()
    if task_prompt:
        return f"{system_prompt}\n\n[Task Instruction]\n{task_prompt}" if system_prompt else task_prompt
    return system_prompt

def _as_responses_input(system_prompt: str, conversation: List[Tuple[str, str]]):
    """
    Responses API 입력 포맷으로 변환.
    이벤트 스트리밍에서는 response.output_text.delta 를 읽는다. :contentReference[oaicite:5]{index=5}
    """
    blocks = []
    if system_prompt:
        blocks.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    for role, text in conversation:
        if not text:
            continue
        blocks.append({
            "role": role,
            "content": [{"type": "text", "text": text}],
        })
    return blocks

def _as_chat_messages(system_prompt: str, conversation: List[Tuple[str, str]]):
    """
    Chat Completions 입력 포맷으로 변환. :contentReference[oaicite:6]{index=6}
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for role, text in conversation:
        if not text:
            continue
        msgs.append({"role": role, "content": text})
    return msgs

def _yield_from_responses_stream(
    model: str,
    system_prompt: str,
    conversation: List[Tuple[str, str]],
    max_tokens: int,
) -> Generator[str, None, None]:
    """
    Responses API 스트리밍.
    주의: with client.responses.stream(...) 에는 stream=True 넣지 않는다. :contentReference[oaicite:7]{index=7}
    """
    input_blocks = _as_responses_input(system_prompt, conversation)
    # 추론형은 temperature/top_p 미지원 → 전달 안 함. 토큰은 max_output_tokens 사용. :contentReference[oaicite:8]{index=8}
    with client.responses.stream(
        model=model,
        input=input_blocks,
        max_output_tokens=max_tokens,
        timeout=TIMEOUT,  # openai-python에서 kwargs 전달 허용
    ) as stream:
        for event in stream:
            # 표준 이벤트 타입: response.output_text.delta :contentReference[oaicite:9]{index=9}
            if getattr(event, "type", None) == "response.output_text.delta":
                piece = getattr(event, "delta", "")
                if piece:
                    yield piece
        # 필요시 최종 응답 접근 가능
        _ = stream.get_final_response()

def _yield_from_chat_stream_reasoning(
    model: str,
    system_prompt: str,
    conversation: List[Tuple[str, str]],
    max_tokens: int,
) -> Generator[str, None, None]:
    """
    추론형 모델을 Chat Completions로 폴백 호출.
    - temperature/top_p 미지정
    - max_completion_tokens 사용 :contentReference[oaicite:10]{index=10}
    """
    msgs = _as_chat_messages(system_prompt, conversation)
    stream = client.chat.completions.create(
        model=model,
        messages=msgs,
        stream=True,  # Chat은 create(stream=True) 패턴 사용 :contentReference[oaicite:11]{index=11}
        max_completion_tokens=max_tokens,
        timeout=TIMEOUT,
    )
    for chunk in stream:
        delta = (chunk.choices[0].delta.content or "") if chunk.choices and chunk.choices[0].delta else ""
        if delta:
            yield delta

def _yield_from_chat_stream_normal(
    model: str,
    system_prompt: str,
    conversation: List[Tuple[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[str, None, None]:
    """
    일반 챗 모델 스트리밍.
    - temperature/top_p/max_tokens 사용 가능(기본 Chat Completions 규격). :contentReference[oaicite:12]{index=12}
    """
    msgs = _as_chat_messages(system_prompt, conversation)
    stream = client.chat.completions.create(
        model=model,
        messages=msgs,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=TIMEOUT,
    )
    for chunk in stream:
        delta = (chunk.choices[0].delta.content or "") if chunk.choices and chunk.choices[0].delta else ""
        if delta:
            yield delta

def stream_noa_response(
    *,
    system_prompt: str,
    task_prompt: Optional[str],
    conversation: List[Tuple[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS_DEFAULT,
) -> Iterable[str]:
    """
    emotion_ws에서 호출하는 토큰 스트리밍 제너레이터.
    - 추론형 모델 → Responses 우선, 실패 시 Chat 폴백
    - 일반 모델 → Chat 사용
    """
    sys_merged = _merge_system(system_prompt, task_prompt)

    # 추론형 모델 경로
    if _is_reasoning_model(MODEL) or USE_RESPONSES_FIRST:
        try:
            logger.info("LLM: streaming via Responses API (%s)", MODEL)
            return _yield_from_responses_stream(MODEL, sys_merged, conversation, max_tokens)
        except TypeError as e:
            # 흔한 실수: Responses.stream(...) 에 stream=True 전달 → 여기선 제거했지만, 방어적으로 폴백
            logger.warning("Responses stream exception; fallback to Chat | %s", e)
        except BadRequestError as e:
            logger.warning("Responses stream failed; fallback to Chat | %s", e)
        except Exception as e:
            logger.warning("Responses stream error; fallback to Chat | %s", e)

        # Chat 폴백(추론형 파라미터 규칙 적용)
        try:
            logger.info("LLM: fallback via Chat Completions (%s)", MODEL)
            return _yield_from_chat_stream_reasoning(MODEL, sys_merged, conversation, max_tokens)
        except BadRequestError as e:
            logger.error("Chat stream failed (reasoning): %s", e)
            raise
        except Exception as e:
            logger.error("Chat stream error (reasoning): %s", e)
            raise

    # 일반 챗 모델 경로
    try:
        logger.info("LLM: streaming via Chat Completions (%s)", MODEL)
        return _yield_from_chat_stream_normal(MODEL, sys_merged, conversation, max_tokens, temperature, TOP_P)
    except BadRequestError as e:
        logger.error("Chat stream failed: %s", e)
        raise
    except Exception as e:
        logger.error("Chat stream error: %s", e)
        raise

# --- Back-compat wrapper for legacy imports ---
def generate_noa_response(
    system_prompt: str,
    task_prompt: str | None,
    conversation,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    호환성 유지용: 예전 sync 함수 시그니처를 그대로 받아
    내부의 stream 제너레이터를 모두 이어붙여 문자열로 반환한다.
    - GPT-5 계열은 temperature 미지원이므로 내부에서 무시될 수 있음.
    - Responses API는 max_output_tokens, Chat은 max_completion_tokens를 사용 (내부 분기).
    """
    try:
        chunks = stream_noa_response(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(chunks)
    except Exception:
        logger.exception("generate_noa_response failed; returning empty string")
        return ""
