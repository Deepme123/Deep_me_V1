from typing import Optional, List, Generator
from openai import OpenAI
from app.models.emotion import EmotionSession, EmotionStep
from app.core.prompt_loader import get_system_prompt

LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 800
client = OpenAI()


def _condense_history(history: list[str], max_chars: int = 1000) -> str:
    combined = "\n".join(history).strip()
    return combined if len(combined) <= max_chars else "...\n" + combined[-max_chars:]


def _build_messages(
    user_input: str,
    emotion_label: Optional[str],
    topic: Optional[str],
    history_snippet: str,
    system_prompt: Optional[str] = None,
) -> List[dict]:
    sys = system_prompt or get_system_prompt()

    context_parts = []
    if emotion_label:
        context_parts.append(f"감정: {emotion_label}")
    if topic:
        context_parts.append(f"주제: {topic}")
    if history_snippet:
        context_parts.append(f"최근 대화:\n{history_snippet}")
    
    messages = [{"role": "system", "content": sys}]
    if context_parts:
        messages.append({"role": "user", "content": "\n".join(context_parts)})
    messages.append({"role": "user", "content": user_input})

    return messages


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
