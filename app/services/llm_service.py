from typing import Optional, List, Generator
from openai import OpenAI
from app.models.emotion import EmotionSession, EmotionStep
from app.core.prompt_loader import get_system_prompt

LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 800

client = OpenAI()

# ... _condense_history() 동일 ...

def _condense_history(history: list[str], max_chars: int = 1000) -> str:
    """
    대화 기록(history)을 최대 max_chars 길이에 맞춰 축약함.
    
    Args:
        history (list[str]): 이전 대화들의 문자열 리스트.
        max_chars (int): 최대 길이 제한. 기본값은 1000자.

    Returns:
        str: 최근 대화 중심으로 축약된 문자열.
    """
    combined = "\n".join(history).strip()

    # 총 길이가 제한보다 짧으면 그대로 반환
    if len(combined) <= max_chars:
        return combined

    # 길면 최근 내용 중심으로 잘라서 반환
    return "...\n" + combined[-max_chars:]


def _build_messages(
    user_input: str,
    emotion_label: Optional[str],
    topic: Optional[str],
    history_snippet: str,
    system_prompt: Optional[str] = None,
):
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
        _condense_history([  # ✅ 여기!
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
    print("🔍 메시지:", messages)
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
        _condense_history([  # ✅ 여기만 고침
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
    collected = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            collected.append(delta)
            yield delta
    print("🔍 메시지:", messages)