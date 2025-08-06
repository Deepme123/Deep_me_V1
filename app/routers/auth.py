from __future__ import annotations

import os
from datetime import datetime
from typing import Optional
from uuid import UUID
from app.core.jwt import create_access_token
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, Request, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.core.tokens import (
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
    new_refresh_jti,
    sha256_hex,
    set_refresh_cookie,
    clear_refresh_cookie,
    REFRESH_COOKIE_NAME,
)
# 프로젝트에 사용자 인증 의존성이 있다면 사용 (예: get_current_user)
try:
    from app.dependencies.auth import get_current_user  # 존재 시 사용
except Exception:
    get_current_user = None  # 미존재 시 /logout에서 대체 처리

auth_router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# 환경변수 & 상수
# ──────────────────────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

# Access Token TTL(분) — app/core/tokens.create_access_token은 .env의 ACCESS_TOKEN_EXPIRE_MINUTES를 사용
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

# AT를 쿠키로도 내려줄지(웹 혼용 환경에서만 권장; 기본 False)
AUTH_SET_COOKIE_ON_POST = os.getenv("AUTH_SET_COOKIE_ON_POST", "false").lower() == "true"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # 배포시 true 권장
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")  # cross-site면 "none"
COOKIE_MAX_AGE = 60 * ACCESS_TOKEN_EXPIRE_MINUTES

# 구글 엔드포인트
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise RuntimeError("❌ GOOGLE_CLIENT_ID 또는 GOOGLE_CLIENT_SECRET 미설정 (.env 확인 필요)")

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ──────────────────────────────────────────────────────────────────────────────
class GoogleIdTokenReq(BaseModel):
    id_token: str

class GoogleAccessTokenReq(BaseModel):
    access_token: str

class AuthTokenModel(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict

# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def _set_access_cookie_if_enabled(response: Response, jwt_token: str) -> None:
    """
    Access Token을 쿠키로도 내려야 하는 환경(웹)에서만 사용.
    기본값은 False이며, 보안상 AT는 메모리 보관 권장.
    """
    if response is not None and AUTH_SET_COOKIE_ON_POST:
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            max_age=COOKIE_MAX_AGE,
            path="/",
        )

def _build_auth_response(user: User, access_token: str) -> AuthTokenModel:
    return AuthTokenModel(
        access_token=access_token,
        expires_in=COOKIE_MAX_AGE,
        user={"user_id": str(user.user_id), "name": user.name, "email": user.email},
    )

def _get_or_create_user(db: Session, *, email: str, name: str | None) -> User:
    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(name=name or "User", email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

async def _verify_id_token_and_extract(http: httpx.AsyncClient, id_token: str) -> tuple[str, str | None]:
    r = await http.get(GOOGLE_TOKENINFO_URL, params={"id_token": id_token}, timeout=10.0)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="유효하지 않은 id_token")
    data = r.json()
    if data.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="aud 불일치")
    if data.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(status_code=400, detail="iss 불일치")
    email = data.get("email")
    name = data.get("name") or data.get("given_name")
    if not email:
        raise HTTPException(status_code=400, detail="email 없음")
    return email, name

async def _fetch_userinfo_with_access_token(http: httpx.AsyncClient, access_token: str) -> tuple[str, str | None]:
    r = await http.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=10.0)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="access_token으로 userinfo 조회 실패")
    u = r.json()
    email = u.get("email")
    name = u.get("name") or u.get("given_name")
    if not email:
        raise HTTPException(status_code=400, detail="email 정보가 없습니다 (scope: openid email profile 필요)")
    return email, name

def issue_tokens_for_user(
    db: Session,
    user: User,
    response: Response,
    user_agent: Optional[str] = None,
    ip: Optional[str] = None,
) -> dict:
    """
    로그인 성공 시 호출: AT 발급 + RT 생성/저장 + RT 쿠키 세팅
    """
    # 1) Access Token (TTL은 .env의 ACCESS_TOKEN_EXPIRE_MINUTES 적용)
    access_token = create_access_token(user.user_id)

    # 2) Refresh Token (회전 전제)
    jti = new_refresh_jti()
    refresh_token, exp = create_refresh_token(user.user_id, jti)

    # 3) DB 저장(원문은 저장하지 않고 해시만 저장)
    db.add(
        RefreshToken(
            jti=jti,
            user_id=user.user_id,
            token_hash=sha256_hex(refresh_token),
            expires_at=exp,
            ip=ip,
            user_agent=user_agent,
        )
    )
    db.commit()

    # 4) RT 쿠키 세팅 (HttpOnly/Secure/SameSite는 app/core/tokens에서 처리)
    set_refresh_cookie(response, refresh_token)

    return {
        "token_type": "bearer",
        "access_token": access_token,
        "expires_in": COOKIE_MAX_AGE,
        "user_id": str(user.user_id),
    }

# ──────────────────────────────────────────────────────────────────────────────
# 1) 리디렉트 기반 (웹)
# ──────────────────────────────────────────────────────────────────────────────
@auth_router.get("/auth/login/google", tags=["auth"])
def login_via_google():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
    }
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}")

@auth_router.get("/auth/callback", response_model=AuthTokenModel, tags=["auth"])
async def google_auth_callback(
    code: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_session),
):
    # code → access_token 교환
    async with httpx.AsyncClient() as http:
        token_res = await http.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=10.0,
        )
        if token_res.status_code != 200:
            raise HTTPException(status_code=400, detail="토큰 요청 실패")
        token_json = token_res.json()

        g_access_token = token_json.get("access_token")
        if not g_access_token:
            raise HTTPException(status_code=400, detail="access_token 누락됨")

        # userinfo 조회
        email, name = await _fetch_userinfo_with_access_token(http, g_access_token)

    user = _get_or_create_user(db, email=email, name=name)

    # AT/RT 발급(+ RT 쿠키 세팅)
    meta = issue_tokens_for_user(
        db,
        user,
        response,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
    )

    # 필요 시 AT를 쿠키로도 내려줌(선택)
    _set_access_cookie_if_enabled(response, meta["access_token"])

    return _build_auth_response(user, meta["access_token"])

# ──────────────────────────────────────────────────────────────────────────────
# 2) POST 기반 (모바일/SPA 권장: id_token)
# ──────────────────────────────────────────────────────────────────────────────
@auth_router.post("/auth/google", response_model=AuthTokenModel, tags=["auth"])
async def auth_with_google(
    body: GoogleIdTokenReq,
    request: Request,
    response: Response,
    db: Session = Depends(get_session),
):
    async with httpx.AsyncClient() as http:
        email, name = await _verify_id_token_and_extract(http, body.id_token)
    user = _get_or_create_user(db, email=email, name=name)

    meta = issue_tokens_for_user(
        db,
        user,
        response,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
    )
    _set_access_cookie_if_enabled(response, meta["access_token"])
    return _build_auth_response(user, meta["access_token"])

# (선택) access_token 경로
@auth_router.post("/auth/google/access", response_model=AuthTokenModel, tags=["auth"])
async def auth_with_google_access(
    body: GoogleAccessTokenReq,
    request: Request,
    response: Response,
    db: Session = Depends(get_session),
):
    async with httpx.AsyncClient() as http:
        email, name = await _fetch_userinfo_with_access_token(http, body.access_token)
    user = _get_or_create_user(db, email=email, name=name)

    meta = issue_tokens_for_user(
        db,
        user,
        response,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
    )
    _set_access_cookie_if_enabled(response, meta["access_token"])
    return _build_auth_response(user, meta["access_token"])

# ──────────────────────────────────────────────────────────────────────────────
# 로그아웃 (쿠키 사용 시)
# ──────────────────────────────────────────────────────────────────────────────
@auth_router.get("/auth/logout", tags=["auth"])
def logout(
    response: Response,
    db: Session = Depends(get_session),
    current_user: Optional[User] = Depends(get_current_user) if get_current_user else None,
):
    """
    현재 사용자 모든 RT 무효화 + 쿠키 제거
    """
    if get_current_user and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    # current_user가 없고 get_current_user가 없다면(개발 편의), 쿠키만 제거
    if current_user:
        for row in db.exec(
            select(RefreshToken).where(
                RefreshToken.user_id == current_user.user_id,
                RefreshToken.revoked_at.is_(None),
            )
        ):
            row.revoked_at = datetime.utcnow()
        db.commit()

    clear_refresh_cookie(response)
    response.delete_cookie("access_token", path="/")
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────────
# (디버그) Swagger 테스트용 비밀번호 로그인
# ──────────────────────────────────────────────────────────────────────────────
FAKE_USERS_DB = {
    "test@example.com": {"user_id": "2d151bd3-cb6c-4837-a575-a796cc3425c5", "password": "1234"}
}

@auth_router.post("/auth/token", tags=["auth"])
def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_session),
):
    user_row = FAKE_USERS_DB.get(form_data.username)
    if not user_row or user_row["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="잘못된 사용자 이름 또는 비밀번호",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 실제 DB 사용자 동기화(없으면 생성)
    user = db.exec(select(User).where(User.email == form_data.username)).first()
    if not user:
        user = User(name="Tester", email=form_data.username)
        db.add(user)
        db.commit()
        db.refresh(user)

    meta = issue_tokens_for_user(
        db,
        user,
        response,
        user_agent=None,
        ip=None,
    )
    _set_access_cookie_if_enabled(response, meta["access_token"])
    return _build_auth_response(user, meta["access_token"])

# ──────────────────────────────────────────────────────────────────────────────
# Refresh (RT 회전 + 재사용 탐지)
# ──────────────────────────────────────────────────────────────────────────────
@auth_router.post("/refresh", tags=["auth"])
async def refresh_token_endpoint(
    request: Request,
    response: Response,
    db: Session = Depends(get_session),
):
    """
    RT 회전:
    1) 쿠키(권장) 또는 바디에서 RT 추출
    2) 서명/만료 검증 → jti/sub 얻기
    3) DB에서 jti 조회 → 이미 무효(revoked/replaced)면 재사용 탐지 → 보안 이벤트 처리
    4) 새 AT/RT 발급, 이전 RT 무효화(replaced_by, revoked_at)
    """
    # 1) 토큰 추출(쿠키 우선)
    rt = request.cookies.get(REFRESH_COOKIE_NAME)
    if not rt:
        try:
            body = await request.json()
            if isinstance(body, dict):
                rt = body.get("refresh_token")
        except Exception:
            rt = None
    if not rt:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing")

    # 2) 서명/만료 검증
    try:
        payload = verify_refresh_token(rt)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    sub = payload.get("sub")
    jti = payload.get("jti")
    if not sub or not jti:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token payload")

    # 3) DB 조회
    rt_row = db.get(RefreshToken, jti)
    if rt_row is None:
        # DB에 기록이 없으면 이미 폐기됐거나 재사용 탐지 케이스 가능
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not recognized")

    # 재사용 탐지: 이미 교체되었거나(replaced_by) 또는 revoked
    if rt_row.revoked_at is not None or rt_row.replaced_by is not None:
        # 간단 대응: 해당 사용자 RT 전부 무효화
        for row in db.exec(select(RefreshToken).where(RefreshToken.user_id == rt_row.user_id)):
            if row.revoked_at is None:
                row.revoked_at = datetime.utcnow()
        db.commit()
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token reused")

    # 토큰 원문 해시 일치 확인(유출/조작 방지)
    if rt_row.token_hash != sha256_hex(rt):
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token tampered")

    # 4) 새 AT/RT 발급(회전)
    user = db.get(User, UUID(sub))
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_access = create_access_token(user.user_id)
    new_jti = new_refresh_jti()
    new_rt, new_exp = create_refresh_token(user.user_id, new_jti)

    # 이전 RT 무효화 + 체인 연결
    rt_row.revoked_at = datetime.utcnow()
    rt_row.replaced_by = new_jti

    # 새 RT 저장
    db.add(
        RefreshToken(
            jti=new_jti,
            user_id=user.user_id,
            token_hash=sha256_hex(new_rt),
            expires_at=new_exp,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    )
    db.commit()

    # 쿠키 교체
    set_refresh_cookie(response, new_rt)

    # 필요 시 AT를 쿠키로도 내려줌(선택)
    _set_access_cookie_if_enabled(response, new_access)

    return _build_auth_response(user, new_access)
