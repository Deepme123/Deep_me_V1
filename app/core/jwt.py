# app/core/jwt.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, Any

from jose import jwt, JWTError

# ── 설정 값 ─────────────────────────────────────
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "deepme-secret-key")
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
# ───────────────────────────────────────────────

def create_access_token(user_id: str, expires_delta: timedelta | None = None) -> str:
    """
    user_id(문자열)로 JWT Access Token을 발급한다.
    """
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=EXPIRE_MINUTES))
    payload: Dict[str, Any] = {"sub": user_id, "exp": expire, "typ": "access"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_access_token(token: str) -> Dict[str, Any]:
    """
    유효한 Access Token이면 payload(dict)를 반환,
    서명 불일치·만료·type 오류가 나면 JWTError를 던진다.
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    if payload.get("typ") != "access":
        raise JWTError("Invalid token type")
    return payload

def decode_access_token(token: str):
    """
    기존 코드 호환용 래퍼.
    실패 시 None을 반환하도록 감싸줌.
    """
    try:
        return verify_access_token(token)
    except JWTError:
        return None
