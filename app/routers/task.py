from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from uuid import UUID

from app.models.task import Task
from app.db.session import get_session
from app.models.user import User
from app.core.auth import get_current_user  # JWT 인증 기반 유저 추출

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
