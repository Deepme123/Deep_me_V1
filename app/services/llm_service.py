from typing import Optional, List, Generator
from openai import OpenAI
from app.models.emotion import EmotionSession, EmotionStep
from app.core.prompt_loader import get_system_prompt
import logging
import os

# ─────────────────────────────
# 환경 설정
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

logger = logging.getLogger("noa")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ─────────────────────────────
# LLM 설정
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 800
client = OpenAI()


def _condense_history(history: list[str], max_chars: int = 1000) -> str:
    combined = "\n".join(history).strip()
    if len(combined) <= max_chars:
        return combined
    return "...\n" + combined[-max_chars:]


def _build_messages(
    user_input: str,
    emotion_label: Optional[str],
    topic: Optional[str],
    history_snippet: str,
    system_prompt: Optional[str] = None,
) -> List[dict]:
    sys = system_prompt or get_system_prompt()
    ctx_parts: list[str] = []
    if emotion_label:
        ctx_parts.append(f"감정: {emotion_label}")
    if topic:
        ctx_parts.append(f"주제: {topic}")
    if history_snippet:
        ctx_parts.append(f"최근 대화:\n{history_snippet}")

    context_block = "\n".join(ctx_parts).strip()

    messages = [{"role": "system", "content": sys}]
    if context_block:
        messages.append({"role": "user", "content": context_block})
    messages.append({"role": "user", "content": user_input})

    if DEBUG:
        redacted = _redact_prompt(messages)
        logger.debug("🔍 메시지 (프롬프트 구조): %s", redacted)

    return messages


def _redact_prompt(messages: List[dict]) -> List[dict]:
    """시스템 프롬프트 등 민감한 내용 마스킹"""
    return [
        {"role": m["role"], "content": "[시스템 프롬프트 생략]" if m["role"] == "system" else m["content"]}
        for m in messages
    ]


def generate_noa_response(
    user_input: str,
    session: EmotionSession,
    recent_steps: List[EmotionStep],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> str:
    messages = _build_messages(
        user_input,
        session.emotion_label,
        session.topic,
        _condense_history([
            f"유저: {s.user_input}\nGPT: {s.gpt_response}" for s in recent_steps
        ]),
        system_prompt,
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature or LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def stream_noa_response(
    *,
    user_input: str,
    session: EmotionSession,
    recent_steps: List[EmotionStep],
    system_prompt: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 400,
) -> Generator[str, None, None]:
    """토큰 단위 GPT 응답을 yield"""
    messages = _build_messages(
        user_input,
        session.emotion_label,
        session.topic,
        _condense_history([
            f"유저: {s.user_input}\nGPT: {s.gpt_response}" for s in recent_steps
        ]),
        system_prompt,
    )

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta
