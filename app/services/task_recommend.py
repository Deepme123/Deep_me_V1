# app/services/task_recommend.py
from typing import List
from uuid import UUID
from sqlmodel import select
from app.db.session import get_session
from app.models.task import Task
from app.models.emotion import EmotionSession, EmotionStep
from app.core.prompt_loader import get_task_prompt
from openai import OpenAI
import os
import json
import re


def _condense_history(lines: list[str], max_chars: int) -> str:
    combined = "\n".join(lines).strip()
    return combined if len(combined) <= max_chars else ("...\n" + combined[-max_chars:])


def recommend_tasks_from_session_core(
    *,
    user_id: UUID,
    session_id: UUID,
    n: int = 3,
    recent_steps_limit: int = 10,
    max_history_chars: int = 1000,
) -> List[Task]:
    """
    최근 감정 대화 세션(context)을 바탕으로 작업(Task) 추천을 생성하고 DB에 저장한다.
    - 모델/샘플링/토큰 한도는 환경변수로 제어:
        LLM_MODEL (기본: gpt-3.5-turbo)
        LLM_TEMPERATURE (기본: 0.7, float)
        LLM_MAX_TOKENS (기본: 800, int)
    - OpenAI 클라이언트는 OPENAI_API_KEY 를 자동 인식.
    선택적으로 OPENAI_BASE_URL / OPENAI_ORG_ID / OPENAI_PROJECT 지원.
    """
    with next(get_session()) as db:
        # 세션 검증
        sess = db.get(EmotionSession, session_id)
        if not sess or sess.user_id != user_id:
            raise ValueError("Emotion session not found or not owned by user")

        # 최신 스텝 수집 (역순 정렬 후 다시 시간순으로 뒤집기)
        stmt = (
            select(EmotionStep)
            .where(EmotionStep.session_id == session_id)
            .order_by(EmotionStep.created_at.desc())
            .limit(recent_steps_limit)
        )
        steps = db.exec(stmt).all()
        steps = list(reversed(steps))

        # 대화 이력 축약
        history_lines = [
            f"유저: {s.user_input or ''}\nGPT: {s.gpt_response or ''}".strip()
            for s in steps
            if (s.user_input or s.gpt_response)
        ]
        history_snippet = _condense_history(history_lines, max_history_chars)

        # 프롬프트 & JSON 정책
        sys_prompt = get_task_prompt().strip()
        json_policy = (
            "출력은 반드시 JSON 배열로만 해. 설명 문장/마크다운/코드블록 없이 "
            '다음 형식으로만 응답해: [{"title": "...", "description": "..."}, ...]'
        )

        # 컨텍스트 블록 구성
        ctx_parts = []
        if sess.emotion_label:
            ctx_parts.append(f"감정: {sess.emotion_label}")
        if sess.topic:
            ctx_parts.append(f"주제: {sess.topic}")
        if history_snippet:
            ctx_parts.append(f"최근 대화:\n{history_snippet}")
        context_block = "\n\n".join(ctx_parts).strip()

        messages = [
            {"role": "system", "content": f"{sys_prompt}\n\n{json_policy}"},
            {"role": "user", "content": f"컨텍스트:\n{context_block}\n\n추천 개수: {n}"},
        ]

        # ---- 환경변수 로딩 (모델/샘플링/토큰 한도) ----
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        except ValueError:
            temperature = 0.7

        try:
            max_completion_tokens = int(os.getenv("LLM_MAX_TOKENS", "800"))
        except ValueError:
            max_completion_tokens = 800

        # ---- OpenAI 클라이언트 구성 (선택적 커스텀 엔드포인트/조직/프로젝트) ----
        client_kwargs = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        org_id = os.getenv("OPENAI_ORG_ID")
        if org_id:
            client_kwargs["organization"] = org_id
        project_id = os.getenv("OPENAI_PROJECT")
        if project_id:
            client_kwargs["project"] = project_id

        client = OpenAI(**client_kwargs)

        # ---- LLM 호출 ----
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # ---- JSON 파싱 (코드블록 방어) ----
        def _strip_codeblock(s: str) -> str:
            s = re.sub(r"^```(?:json)?\s*", "", s.strip())
            s = re.sub(r"\s*```$", "", s.strip())
            return s.strip()

        parsed = None
        for candidate in (raw, _strip_codeblock(raw)):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    break
            except Exception:
                continue
        if not isinstance(parsed, list):
            raise RuntimeError("GPT 응답 파싱 실패")

        # ---- 상한 n 적용 및 DB 저장 ----
        n = max(1, min(5, n))
        tasks_out: list[Task] = []
        for item in parsed[:n]:
            title = (item.get("title") or "").strip()
            description = (item.get("description") or "").strip()
            if not title:
                continue
            t = Task(user_id=user_id, title=title, description=description or None)
            db.add(t)
            tasks_out.append(t)

        if not tasks_out:
            raise RuntimeError("유효한 과제가 없습니다")

        db.commit()
        for t in tasks_out:
            db.refresh(t)

        return tasks_out
