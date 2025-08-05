from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from uuid import UUID

from app.models.task import Task
from app.db.session import get_session
from app.models.user import User
from app.dependencies.auth import get_current_user  # JWT 인증 기반 유저 추출
from app.core.prompt_loader import get_task_prompt
from openai import OpenAI

from app.models.emotion import EmotionSession, EmotionStep
from app.schemas.task import TaskRecommendBySessionRequest
import json
import re

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("/", response_model=Task)
def create_task(
    task: Task,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task.user_id = user.id
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.get("/", response_model=list[Task])
def get_all_tasks(
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    stmt = select(Task).where(Task.user_id == user.id)
    return db.exec(stmt).all()


@router.get("/{task_id}", response_model=Task)
def get_task(
    task_id: UUID,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.id:
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
    if not task or task.user_id != user.id:
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
    if not task or task.user_id != user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    task.is_completed = True
    from datetime import datetime

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
    if not task or task.user_id != user.id:
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
    client = OpenAI()

    # GPT 호출
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "지금 나에게 추천해줘."},
        ],
        temperature=0.7,
        max_tokens=800,
    )

    result = (response.choices[0].message.content or "").strip()

    # 과제 파싱 (멀티 아이템 안전 정규식)
    # 예시 포맷:
    # 1. 제목: XXX
    #    설명: YYY
    # 2. 제목: ...
    pattern = r"\d+\.\s*제목:\s*(.*?)\s*[\r\n]+설명:\s*(.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, result, flags=re.DOTALL)

    if not matches:
        raise HTTPException(status_code=500, detail="GPT 응답 파싱 실패")

    tasks = []
    for title, description in matches:
        title = (title or "").strip()
        description = (description or "").strip()
        if not title:
            continue
        t = Task(
            user_id=user.id,
            title=title,
            description=description or None,
        )
        db.add(t)
        tasks.append(t)

    db.commit()
    for t in tasks:
        db.refresh(t)

    return tasks


# (파일 내 임의 위치에 헬퍼 추가)
def _condense_history(lines: list[str], max_chars: int) -> str:
    combined = "\n".join(lines).strip()
    return combined if len(combined) <= max_chars else ("...\n" + combined[-max_chars:])


# (새 엔드포인트 추가)
@router.post("/gpt/by-session", response_model=list[Task], summary="세션 기반 GPT 과제 추천")
def recommend_tasks_from_session(
    payload: TaskRecommendBySessionRequest,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # 1) 세션 유효성/소유권 검사
    sess = db.get(EmotionSession, payload.session_id)
    if not sess or sess.user_id != user.id:
        raise HTTPException(status_code=404, detail="Emotion session not found")

    # 2) 최근 스텝 조회
    stmt = (
        select(EmotionStep)
        .where(EmotionStep.session_id == payload.session_id)
        .order_by(EmotionStep.created_at.desc())
        .limit(payload.recent_steps_limit)
    )
    steps = db.exec(stmt).all()
    steps = list(reversed(steps))  # 시간순 정렬(오래된→최근)

    # 3) 컨텍스트 빌드(감정/주제/대화이력)
    history_lines = [
        f"유저: {s.user_input or ''}\nGPT: {s.gpt_response or ''}".strip()
        for s in steps
        if (s.user_input or s.gpt_response)
    ]
    history_snippet = _condense_history(history_lines, payload.max_history_chars)

    context_parts = []
    if sess.emotion_label:
        context_parts.append(f"감정: {sess.emotion_label}")
    if sess.topic:
        context_parts.append(f"주제: {sess.topic}")
    if history_snippet:
        context_parts.append(f"최근 대화:\n{history_snippet}")
    context_block = "\n\n".join(context_parts).strip()

    # 4) 프롬프트 구성(전용 프롬프트 + JSON 강제)
    sys_prompt = get_task_prompt().strip()
    json_policy = (
        "출력은 반드시 JSON 배열로만 해. 설명 문장/마크다운/코드블록 없이 "
        '다음 형식으로만 응답해: [{"title": "...", "description": "..."}, ...]'
    )

    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\n{json_policy}"},
        {"role": "user", "content": f"컨텍스트:\n{context_block}\n\n추천 개수: {payload.n}"},
    ]

    # 5) OpenAI 호출
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=800,
    )
    raw = (resp.choices[0].message.content or "").strip()

    # 6) JSON 파싱(코드블록 제거 등 방어)
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

    # 7) 상한 적용 및 DB 저장
    n = max(1, min(5, payload.n))
    tasks_out: list[Task] = []
    for item in parsed[:n]:
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        if not title:
            continue
        t = Task(user_id=user.id, title=title, description=description or None)
        db.add(t)
        tasks_out.append(t)

    if not tasks_out:
        raise HTTPException(status_code=500, detail="유효한 과제가 없습니다")

    db.commit()
    for t in tasks_out:
        db.refresh(t)

    return tasks_out
