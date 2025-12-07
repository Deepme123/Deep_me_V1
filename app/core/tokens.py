from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple
from uuid import UUID, uuid4
from jose import jwt, JWTError
import os

# ← python-jose 사용


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# ---- 설정 ----
ALG = os.getenv("JWT_ALGORITHM", "HS256")

ACCESS_SECRET = os.getenv("JWT_SECRET_KEY", "deepme-secret-key")
ACCESS_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET", "dev_refresh_secret_change_me")
REFRESH_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "21"))

REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "__Host-deepme_rtok")
SECURE_COOKIE = _env_bool("SECURE_COOKIE", True)


# ---- 공통 ----
def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _exp_in(minutes: int = 0, days: int = 0) -> datetime:
    return _utcnow() + timedelta(minutes=minutes, days=days)


def _make_jwt(payload: Dict[str, Any], secret: str, exp: datetime) -> str:
    to_encode = payload.copy()
    to_encode["iat"] = int(_utcnow().timestamp())
    to_encode["exp"] = int(exp.timestamp())
    return jwt.encode(to_encode, secret, algorithm=ALG)


def _decode(token: str, secret: str) -> Dict[str, Any]:
    # jose.jwt.decode는 검증 실패 시 jose.exceptions.JWTError를 던짐
    return jwt.decode(token, secret, algorithms=[ALG])


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---- Access Token ----
def create_access_token(sub: UUID, extra: Dict[str, Any] | None = None) -> str:
    payload = {"sub": str(sub), "typ": "access"}
    if extra:
        payload.update(extra)
    return _make_jwt(payload, ACCESS_SECRET, _exp_in(minutes=ACCESS_MIN))


def verify_access_token(token: str) -> Dict[str, Any]:
    payload = _decode(token, ACCESS_SECRET)
    if payload.get("typ") != "access":
        from jose.exceptions import JWTError
        raise JWTError("Invalid token type")
    return payload


# ---- Refresh Token (회전 전제) ----
def new_refresh_jti() -> str:
    return str(uuid4())


def create_refresh_token(sub: UUID, jti: str) -> Tuple[str, datetime]:
    exp = _exp_in(days=REFRESH_DAYS)
    payload = {"sub": str(sub), "jti": jti, "typ": "refresh"}
    token = _make_jwt(payload, REFRESH_SECRET, exp)
    return token, exp


def verify_refresh_token(token: str) -> Dict[str, Any]:
    payload = _decode(token, REFRESH_SECRET)
    if payload.get("typ") != "refresh":
        from jose.exceptions import JWTError
        raise JWTError("Invalid token type")
    required = ("sub", "jti", "exp")
    for k in required:
        if k not in payload:
            from jose.exceptions import JWTError
            raise JWTError(f"Missing {k}")
    return payload


# ---- 쿠키 ----
def set_refresh_cookie(response, token: str):
    # 개발에서 http라면 .env에서 SECURE_COOKIE=false 설정 필요
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=SECURE_COOKIE,
        path="/",
    )


def clear_refresh_cookie(response):
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        path="/",
    )

def decode_access_token(token: str):
    """verify_access_token 과 동일 기능. 기존 코드 호환용."""
    try:
        return verify_access_token(token)
    except JWTError:
        return None
