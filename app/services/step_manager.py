from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Callable, Iterable, List

from app.models.emotion import EmotionStep

SOFT_TIMEOUT_TURNS = int(os.getenv("SOFT_TIMEOUT_TURNS", "3"))
END_SESSION_TOKEN = "__END_SESSION__"
END_SESSION_FAREWELL_WORDS = [
    "고마워", "오늘", "함께", "해줘서", "잘", "지냈어", "또", "보자"
]


@dataclass(frozen=True)
class StepMeta:
    step: int
    name: str
    focus: str
    advance_when: Callable[[str, str], bool]


def _norm(text: str | None) -> str:
    return (text or "").strip()


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    return any(k in text for k in keywords)


_GREETING_RE = re.compile(
    r"^(안녕(하세요|하세용|하세여)?|하이|헬로|ㅎㅇ|반가워|좋은 (아침|저녁|밤)|굿(모닝|밤))([!.~ ]*)?$",
    re.I,
)


def _is_greeting_only(text: str) -> bool:
    t = _norm(text)
    if not t:
        return True
    return bool(_GREETING_RE.match(t))


_EMOTION_WORDS = (
    "기쁘", "행복", "즐겁", "신나", "편안", "안도",
    "우울", "슬프", "속상", "허전", "외롭",
    "화나", "짜증", "분노", "답답",
    "불안", "초조", "긴장", "걱정", "무섭", "두렵",
    "무기력", "지치", "피곤",
)
_SITUATION_WORDS = (
    "때문", "때", "상황", "일", "사건", "회사", "학교", "집",
    "가족", "친구", "연인", "동료", "회의", "통화", "메시지", "시험",
    "요즘", "최근", "오늘", "어제", "이번", "지난",
)
_BODY_WORDS = (
    "숨", "심장", "가슴", "몸", "손", "떨", "땀", "두통",
    "머리", "속", "위", "배", "어깨", "목", "긴장", "눈물", "잠", "식욕",
)
_THOUGHT_WORDS = (
    "생각", "머리속", "걱정", "떠올", "상상", "마음속", "판단", "결론", "의문",
)
_BELIEF_WORDS = (
    "해야", "해야만", "당연", "원래", "항상", "절대", "기준",
    "가치", "중요", "옳", "맞", "틀", "책임", "의무",
)
_ACTION_WORDS = (
    "했", "했어", "했다", "하게", "말했", "보냈", "피했", "참았",
    "그만", "돌아", "움직", "행동", "하지", "했다가",
)
_SUMMARY_MARKERS = (
    "정리하면", "요약", "한마디로", "종합하면", "요컨대",
)
_NEED_WORDS = (
    "원해", "바라", "필요", "하고 싶", "했으면", "했으면 좋", "원하는",
)
_REFRAME_MARKERS = (
    "다르게 볼", "다른 해석", "새롭게", "새로운 시각", "다른 관점", "재구성",
)


def _advance_greeting(user_text: str, _: str) -> bool:
    return not _is_greeting_only(user_text)


def _advance_emotion(user_text: str, _: str) -> bool:
    return _has_any(user_text, _EMOTION_WORDS)


def _advance_situation(user_text: str, _: str) -> bool:
    return _has_any(user_text, _SITUATION_WORDS)


def _advance_body(user_text: str, _: str) -> bool:
    return _has_any(user_text, _BODY_WORDS)


def _advance_thought(user_text: str, _: str) -> bool:
    return _has_any(user_text, _THOUGHT_WORDS)


def _advance_belief(user_text: str, _: str) -> bool:
    return _has_any(user_text, _BELIEF_WORDS)


def _advance_action(user_text: str, _: str) -> bool:
    return _has_any(user_text, _ACTION_WORDS)


def _advance_summary(_: str, assistant_text: str) -> bool:
    return _has_any(assistant_text, _SUMMARY_MARKERS)


def _advance_need(user_text: str, _: str) -> bool:
    return _has_any(user_text, _NEED_WORDS)


def _advance_reframe(_: str, assistant_text: str) -> bool:
    return _has_any(assistant_text, _REFRAME_MARKERS)


STEP_METADATA: List[StepMeta] = [
    StepMeta(1, "인사", "관계 신호를 열고 다음 탐색으로 연결", _advance_greeting),
    StepMeta(2, "감정", "느낌을 명확히 드러내는 단계", _advance_emotion),
    StepMeta(3, "상황", "감정을 유발한 상황을 구체화", _advance_situation),
    StepMeta(4, "신체반응", "몸의 반응을 탐색", _advance_body),
    StepMeta(5, "생각", "떠오르는 생각을 탐색", _advance_thought),
    StepMeta(6, "생각 아래 기준", "기준/가치/의무를 드러냄", _advance_belief),
    StepMeta(7, "이후 행동", "그 다음 행동을 확인", _advance_action),
    StepMeta(8, "요약", "흐름을 정리", _advance_summary),
    StepMeta(9, "욕구", "원하는 것/필요를 드러냄", _advance_need),
    StepMeta(10, "부정적인 감정 재구성", "다른 관점 제안", _advance_reframe),
    StepMeta(11, "마무리", "대화를 정리하고 마침", lambda *_: False),
]

MAX_STEP = len(STEP_METADATA)


def _clamp_step(step: int) -> int:
    return max(1, min(MAX_STEP, step))


def advance_step(current_step: int, user_text: str, assistant_text: str) -> int:
    step = _clamp_step(current_step)
    if step >= MAX_STEP:
        return MAX_STEP
    meta = STEP_METADATA[step - 1]
    return step + 1 if meta.advance_when(_norm(user_text), _norm(assistant_text)) else step


def compute_current_step(steps: List[EmotionStep]) -> int:
    step, _stagnant = compute_step_status(steps)
    return _clamp_step(step)


def compute_step_status(steps: List[EmotionStep]) -> tuple[int, int]:
    step = 1
    pending_user = ""
    stagnant_turns = 0
    for s in steps:
        if s.step_type == "user":
            pending_user = s.user_input or ""
        elif s.step_type == "assistant":
            before = step
            after_user = advance_step(step, pending_user, "")
            if after_user != step:
                step = after_user
            else:
                step = advance_step(step, pending_user, s.gpt_response or "")
            if step == before:
                stagnant_turns += 1
            else:
                stagnant_turns = 0
            pending_user = ""
        else:
            continue
    return _clamp_step(step), stagnant_turns


def step_for_prompt(steps: List[EmotionStep], pending_user_text: str) -> int:
    history_step = compute_current_step(steps)
    return advance_step(history_step, pending_user_text, "")


def step_after_turn(
    steps: List[EmotionStep],
    user_text: str,
    assistant_text: str,
) -> int:
    history_step = compute_current_step(steps)
    after_user = advance_step(history_step, user_text, "")
    if after_user != history_step:
        return after_user
    return advance_step(history_step, user_text, assistant_text)


def build_step_context(current_step: int) -> str:
    step = _clamp_step(current_step)
    meta = STEP_METADATA[step - 1]
    return (
        "[CURRENT STEP]\n"
        f"step: {meta.step}/{MAX_STEP}\n"
        f"name: {meta.name}\n"
        f"focus: {meta.focus}"
    )


def get_step_name(current_step: int) -> str:
    step = _clamp_step(current_step)
    return STEP_METADATA[step - 1].name


def build_end_session_context(current_step: int) -> str:
    step = _clamp_step(current_step)
    if step < MAX_STEP:
        return ""
    farewell = build_fixed_farewell()
    return (
        "[SESSION END]\n"
        "rule: send the farewell exactly once and include the token exactly once.\n"
        f"farewell: {farewell}\n"
        f"token: {END_SESSION_TOKEN}"
    )


_SOFT_HINTS = {
    1: "Keep greeting short, then invite a small next share without pressing.",
    2: "Name the feeling softly and mirror the intensity; offer 2 gentle labels.",
    3: "Anchor in a concrete moment; if needed, ask for one specific scene.",
    4: "Link feeling to body cues; prompt one bodily signal if stuck.",
    5: "Surface the thought in simple words; reflect with a short paraphrase.",
    6: "Point to a possible standard/value; offer a light guess, not a claim.",
    7: "Notice what they did or avoided; connect it to the feeling.",
    8: "Give a brief recap; avoid new questions unless needed.",
    9: "Elicit a want/need; suggest two options to choose from.",
    10: "Offer a reframe as a tentative alternative; ask for confirmation only.",
    11: "Close gently; reinforce that it's okay to pause and hold.",
}


def build_soft_timeout_hint(steps: List[EmotionStep], pending_user_text: str) -> str:
    """
    If the conversation stays in the same step too long, keep the step but
    provide a strategy hint to adjust the response style.
    """
    _step, stagnant_turns = compute_step_status(steps)
    if stagnant_turns < max(1, SOFT_TIMEOUT_TURNS):
        return ""
    current_step = step_for_prompt(steps, pending_user_text)
    hint = _SOFT_HINTS.get(current_step, "Vary reflection and keep the pace gentle.")
    return (
        "[SOFT TIMEOUT]\n"
        f"stagnant_turns: {stagnant_turns}\n"
        "rule: keep the current step; change response strategy only.\n"
        f"hint: {hint}"
    )


def extract_end_session_marker(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    if END_SESSION_TOKEN not in text:
        return text, False
    cleaned = text.replace(END_SESSION_TOKEN, "").strip()
    return cleaned, True


def build_fixed_farewell() -> str:
    return " ".join(END_SESSION_FAREWELL_WORDS)
