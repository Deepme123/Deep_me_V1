from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from uuid import UUID

from app.models.task import Task
from app.db.session import get_session
from app.models.user import User
from app.routers.auth import get_current_user  # JWT 인증 기반 유저 추출
from app.core.prompt_loader import get_task_prompt
from openai import OpenAI

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("/", response_model=Task)
def create_task(task: Task, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    task.user_id = user.user_id
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.get("/", response_model=list[Task])
def get_all_tasks(db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    statement = select(Task).where(Task.user_id == user.user_id)
    return db.exec(statement).all()


@router.get("/{task_id}", response_model=Task)
def get_task(task_id: UUID, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: UUID, updated: Task, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
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
def complete_task(task_id: UUID, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Task not found")

    task.is_completed = True
    from datetime import datetime
    task.completed_at = datetime.utcnow()
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.delete("/{task_id}")
def delete_task(task_id: UUID, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    task = db.get(Task, task_id)
    if not task or task.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"ok": True}

router.post("/gpt", response_model=list[Task])
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
            {"role": "user", "content": "지금 나에게 추천해줘."}
        ],
        temperature=0.7,
        max_tokens=800,
    )

    result = response.choices[0].message.content.strip()

    # 과제 파싱 (간단한 정규식 기반)
    import re
    pattern = r"\d+\.\s*제목:\s*(.*?)\n\s*설명:\s*(.*)"
    matches = re.findall(pattern, result)

    if not matches:
        raise HTTPException(status_code=500, detail="GPT 응답 파싱 실패")

    tasks = []
    for title, description in matches:
        task = Task(
            user_id=user.user_id,
            title=title.strip(),
            description=description.strip()
        )
        db.add(task)
        tasks.append(task)

    db.commit()
    for task in tasks:
        db.refresh(task)

    return tasks