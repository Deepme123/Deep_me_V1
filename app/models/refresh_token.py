from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlmodel import SQLModel, Field


class RefreshToken(SQLModel, table=True):
    """
    회전(rotation)과 재사용 탐지(reuse detection)를 위한 RT 레코드.
    - jti: 토큰 고유 식별자(JWT 'jti' 또는 자체 UUID)
    - token_hash: DB 유출 대비 원문 토큰 해시(sha256)
    - replaced_by: 새 RT의 jti (회전 체인)
    """
    jti: str = Field(primary_key=True, index=True)
    user_id: UUID = Field(index=True, foreign_key="user.user_id")
    token_hash: str = Field(nullable=False)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    revoked_at: Optional[datetime] = None
    replaced_by: Optional[str] = None

    ip: Optional[str] = None
    user_agent: Optional[str] = None
