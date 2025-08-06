from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from app.db.session import get_session
from app.models.user import User
from app.core.jwt import create_access_token
from datetime import datetime, timedelta
from uuid import uuid4
import httpx
import os
from urllib.parse import urlencode

auth_router = APIRouter()

# 환경변수 로드
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise RuntimeError("❌ GOOGLE_CLIENT_ID 또는 SECRET이 설정되지 않았습니다 (.env 확인 요망)")

# ✅ 로그인 진입점 - 구글 OAuth URL로 리디렉션
@auth_router.get("/auth/login/google")
def login_via_google():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
    }
    google_oauth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url=google_oauth_url)

# ✅ 콜백 - 토큰 교환 + 사용자 정보 → DB 저장 + JWT 발급 + 쿠키 설정
@auth_router.get("/auth/callback")
async def google_auth_callback(code: str, db: Session = Depends(get_session)):
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    # 1. 토큰 요청
    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=token_data)
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail="❌ 토큰 요청 실패")
        token_json = token_response.json()

    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="❌ access_token 누락됨")

    # 2. 사용자 정보 요청
    user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        user_response = await client.get(user_info_url, headers=headers)
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="❌ 사용자 정보 조회 실패")
        user_data = user_response.json()

    email = user_data.get("email")
    name = user_data.get("name")

    if not email or not name:
        raise HTTPException(status_code=400, detail="❌ 사용자 정보 불완전함 (email 또는 name 없음)")

    # 3. 기존 유저 확인 or 신규 생성
    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(name=name, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)

    # 4. JWT 생성 및 쿠키 설정
    jwt_token = create_access_token(
        str(user.user_id),  # ✅ 실제 UUID 사용
        expires_delta=timedelta(minutes=60)
    )

    response = JSONResponse(content={
        "message": "✅ 로그인 성공",
        "user_id": str(user.user_id),
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat(),
    })

    response.set_cookie(
        key="access_token",
        value=jwt_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=3600,
    )

    return response

# ✅ 로그아웃 엔드포인트 - 쿠키 제거
@auth_router.get("/auth/logout")
def logout():
    response = JSONResponse(content={"message": "👋 로그아웃 완료"})
    response.delete_cookie("access_token")
    return response

# ✅ Swagger 테스트용 로그인 엔드포인트 (/auth/token)
FAKE_USERS_DB = {
    "test@example.com": {
        "user_id": "user-1234",
        "password": "1234"
    }
}

@auth_router.post("/auth/token", tags=["auth"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="잘못된 사용자 이름 또는 비밀번호",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user["user_id"])
    return {"access_token": access_token, "token_type": "bearer"}

# ---- (추가) 요청/응답 모델 & JWT 헬퍼 ----
from pydantic import BaseModel

class GoogleIdTokenReq(BaseModel):
    id_token: str

class AuthTokenModel(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict

def _issue_jwt(user):
    # 서버 자체 JWT 발급 (1시간 유효)
    return create_access_token(str(user.user_id), expires_delta=timedelta(hours=1))


# ---- (추가) POST /auth/google : id_token 기반 인증 ----
@auth_router.post("/auth/google", response_model=AuthTokenModel, tags=["auth"])
async def auth_with_google(body: GoogleIdTokenReq, db: Session = Depends(get_session)):
    # 간편 검증(tokeninfo); 운영에선 google-auth 라이브러리로 서명 검증 권장
    async with httpx.AsyncClient() as client:
        r = await client.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": body.id_token})
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="유효하지 않은 id_token")
    data = r.json()

    # 필수 클레임 검사
    if data.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="aud 불일치")
    if data.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(status_code=400, detail="iss 불일치")

    email = data.get("email")
    name = data.get("name") or data.get("given_name") or (email.split("@")[0] if email else None)
    if not email:
        raise HTTPException(status_code=400, detail="email 없음")

    # 사용자 upsert
    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(name=name or "User", email=email)
        db.add(user)
        db.commit()
        db.refresh(user)

    jwt_ = _issue_jwt(user)
    return {
        "access_token": jwt_,
        "token_type": "bearer",
        "expires_in": 3600,
        "user": {"user_id": str(user.user_id), "name": user.name, "email": user.email},
    }


# ---- (선택 추가) POST /auth/google/access : access_token 기반 인증 ----
# FE가 access_token만 전달하겠다고 하면 아래 엔드포인트도 함께 사용 가능
class GoogleAccessTokenReq(BaseModel):
    access_token: str

@auth_router.post("/auth/google/access", response_model=AuthTokenModel, tags=["auth"])
async def auth_with_google_access(body: GoogleAccessTokenReq, db: Session = Depends(get_session)):
    headers = {"Authorization": f"Bearer {body.access_token}"}
    async with httpx.AsyncClient() as client:
        r = await client.get("https://www.googleapis.com/oauth2/v3/userinfo", headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="access_token으로 userinfo 조회 실패")
    u = r.json()

    email = u.get("email")
    name = u.get("name") or u.get("given_name") or (email.split("@")[0] if email else None)
    if not email:
        raise HTTPException(status_code=400, detail="email 정보가 없습니다 (scope: openid email profile 필요)")

    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(name=name or "User", email=email)
        db.add(user)
        db.commit()
        db.refresh(user)

    jwt_ = _issue_jwt(user)
    return {
        "access_token": jwt_,
        "token_type": "bearer",
        "expires_in": 3600,
        "user": {"user_id": str(user.user_id), "name": user.name, "email": user.email},
    }
