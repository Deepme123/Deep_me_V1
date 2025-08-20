from typing import Optional, List, Generator
from openai import OpenAI
import os
import logging

from app.models.emotion import EmotionSession, EmotionStep
from app.core.prompt_loader import get_system_prompt

# ─────────────────────────────────────────────
# 환경변수 기반 설정 (없으면 안전한 기본값)
MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "30"))  # SDK에 따라 무시될 수 있음
TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.3"))
FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.6"))

# OpenAI 클라이언트 (가능한 경우 timeout 옵션 사용)
try:
    client = OpenAI(timeout=TIMEOUT)
except TypeError:
    # 일부 버전은 timeout 옵션 미지원 → 기본 클라이언트 사용
    client = OpenAI()

logger = logging.getLogger(__name__)
# ※ 프롬프트/히스토리는 로그에 남기지 않는다!


def _condense_history(history: list[str], max_chars: int = 1000) -> str:
    """
    최근 대화(history)를 최대 길이에 맞춰 축약.
    """
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
    """
    LLM에 전달할 메시지 배열 구성.
    ※ 보안: 이 함수 내부에서 메시지를 로그로 남기지 않음.
    """
    sys = system_prompt or get_system_prompt()

    ctx_parts: list[str] = []
    if emotion_label:
        ctx_parts.append(f"감정: {emotion_label}")
    if topic:
        ctx_parts.append(f"주제: {topic}")
    if history_snippet:
        ctx_parts.append(f"최근 대화:\n{history_snippet}")

    messages: List[dict] = [{"role": "system", "content": sys}]
    if ctx_parts:
        messages.append({"role": "user", "content": "\n".join(ctx_parts)})
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
    """
    단건 응답 생성 (스트리밍 아님).
    예외 발생 시 외부로 상세 노출 금지 → 빈 문자열 반환 또는 상위에서 처리.
    """
    messages = _build_messages(
        user_input=user_input,
        emotion_label=session.emotion_label,
        topic=session.topic,
        history_snippet=_condense_history([
            f"유저: {s.user_input}\nGPT: {s.gpt_response}" for s in recent_steps
        ]),
        system_prompt=system_prompt,
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature if temperature is not None else TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
            top_p=TOP_P,
            presence_penalty=PRESENCE_PENALTY,
            frequency_penalty=FREQUENCY_PENALTY,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content
    except Exception:
        # 내부에만 상세 기록 (프롬프트/유저입력 등 민감데이터는 남기지 않음)
        logger.exception("LLM completion failed")
        return ""  # 호출 측에서 빈 응답 처리(재시도/에러 안내 등)


def stream_noa_response(
    *,
    user_input: str,
    session: EmotionSession,
    recent_steps: List[EmotionStep],
    system_prompt: Optional[str] = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = 400,
) -> Generator[str, None, None]:
    """
    토큰 단위 스트리밍 응답 생성.
    예외 발생 시 외부로 상세 노출 금지.
    """
    messages = _build_messages(
        user_input=user_input,
        emotion_label=session.emotion_label,
        topic=session.topic,
        history_snippet=_condense_history([
            f"유저: {s.user_input}\nGPT: {s.gpt_response}" for s in recent_steps
        ]),
        system_prompt=system_prompt,
    )

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            top_p=TOP_P,
            presence_penalty=PRESENCE_PENALTY,
            frequency_penalty=FREQUENCY_PENALTY,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield delta
    except Exception:
        # 스트리밍 실패 → 외부로는 상세 미노출
        logger.exception("LLM streaming failed")
        return  # 조용히 종료 (상위 WS에서 에러 메시지를 일반화해 전송)
