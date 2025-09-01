# app/routers/task.py
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from uuid import UUID
from datetime import datetime
import json
import re
import os

from openai import OpenAI

from app.models.task import Task
from app.models.user import User
from app.models.emotion import EmotionSession, EmotionStep
from app.db.session import get_session
from app.dependencies.auth import get_current_user
from app.core.prompt_loader import get_task_prompt
from app.schemas.task import TaskRecommendBySessionRequest

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def _get_openai_client_and_params():
    """
    환경변수에서 모델/샘플링/토큰 한도 및 선택적 커스텀 엔드포인트를 읽어온다.
    - LLM_MODEL (기본: gpt-3.5-turbo)
    - LLM_TEMPERATURE (기본: 0.7)
    - LLM_MAX_TOKENS (기본: 800)
    - OPENAI_BASE_URL / OPENAI_ORG_ID / OPENAI_PROJECT (선택)
    """
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    except ValueError:
        temperature = 0.7
    try:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "800"))
    except ValueError:
        max_tokens = 800

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
    return client, model, temperature, max_tokens


@router.post("/", response_model=Task)
def create_task(
    task: Task,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task.user_id = user.user_id
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.get("/", response_model=list[Task])
def get_all_tasks(
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    stmt = select(Task).where(Task.user_id == user.user_id)
    return db.exec(stmt).all()


@router.get("/{task_id}", response_model=Task)
def get_task(
    task_id: UUID,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.patch("/{task_id}", response_model=Task)
def update_task(
    task_id: UUID,
    updated: Task,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Task not found")

    task.title = updated.title or task.title
    task.description = updated.description or task.description
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.patch("/{task_id}/complete", response_model=Task)
def complete_task(
    task_id: UUID,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Task not found")

    task.is_completed = True
    task.completed_at = datetime.utcnow()
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.delete("/{task_id}")
def delete_task(
    task_id: UUID,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"ok": True}


@router.post("/gpt", response_model=list[Task])
def recommend_tasks_from_gpt(
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    prompt = get_task_prompt()
    client, model, temperature, max_tokens = _get_openai_client_and_params()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "지금 나에게 추천해줘."},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    result = (response.choices[0].message.content or "").strip()
    pattern = r"\d+\.\s*제목:\s*(.*?)\s*[\r\n]+설명:\s*(.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, result, flags=re.DOTALL)

    if not matches:
        raise HTTPException(status_code=500, detail="GPT 응답 파싱 실패")

    tasks = []
    for title, description in matches:
        title = title.strip()
        description = description.strip()
        if not title:
            continue
        task = Task(user_id=user.user_id, title=title, description=description or None)
        db.add(task)
        tasks.append(task)

    db.commit()
    for task in tasks:
        db.refresh(task)

    return tasks


@router.post("/gpt/by-session", response_model=list[Task], summary="세션 기반 GPT 과제 추천")
def recommend_tasks_from_session(
    payload: TaskRecommendBySessionRequest,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    sess = db.get(EmotionSession, payload.session_id)
    if not sess or sess.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Emotion session not found")

    stmt = (
        select(EmotionStep)
        .where(EmotionStep.session_id == payload.session_id)
        .order_by(EmotionStep.created_at.desc())
        .limit(payload.recent_steps_limit)
    )
    steps = list(reversed(db.exec(stmt).all()))

    history_lines = [
        f"유저: {s.user_input or ''}\nGPT: {s.gpt_response or ''}".strip()
        for s in steps
        if (s.user_input or s.gpt_response)
    ]

    def _condense_history(lines: list[str], max_chars: int) -> str:
        combined = "\n".join(lines).strip()
        return combined if len(combined) <= max_chars else ("...\n" + combined[-max_chars:])

    history_snippet = _condense_history(history_lines, payload.max_history_chars)

    context_parts = []
    if sess.emotion_label:
        context_parts.append(f"감정: {sess.emotion_label}")
    if sess.topic:
        context_parts.append(f"주제: {sess.topic}")
    if history_snippet:
        context_parts.append(f"최근 대화:\n{history_snippet}")
    context_block = "\n\n".join(context_parts).strip()

    sys_prompt = get_task_prompt().strip()
    json_policy = (
        "출력은 반드시 JSON 배열로만 해. 설명 문장/마크다운/코드블록 없이 "
        '다음 형식으로만 응답해: [{"title": "...", "description": "..."}, ...]'
    )

    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\n{json_policy}"},
        {"role": "user", "content": f"컨텍스트:\n{context_block}\n\n추천 개수: {payload.n}"},
    ]

    client, model, temperature, max_tokens = _get_openai_client_and_params()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = (resp.choices[0].message.content or "").strip()

    def _strip_codeblock(s: str) -> str:
        s = re.sub(r"^```(?:json)?\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
        return s.strip()

    parsed = None
    for candidate in [raw, _strip_codeblock(raw)]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                break
        except Exception:
            continue

    if not isinstance(parsed, list):
        raise HTTPException(status_code=500, detail="GPT 응답 파싱 실패")

    n = max(1, min(5, payload.n))
    tasks_out: list[Task] = []

    for item in parsed[:n]:
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        if not title:
            continue
        task = Task(user_id=user.user_id, title=title, description=description or None)
        db.add(task)
        tasks_out.append(task)

    if not tasks_out:
        raise HTTPException(status_code=500, detail="유효한 과제가 없습니다")

    db.commit()
    for task in tasks_out:
        db.refresh(task)

    return tasks_out
