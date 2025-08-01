# app/models/task.py
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class UserTask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    task: str
    is_recommended: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
