from sqlmodel import SQLModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional


class User(SQLModel, table=True):
    user_id: UUID = Field(default_factory=uuid4, primary_key=True)  # 이름 변경됨
    name: str
    email: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    __tablename__ = "user"


