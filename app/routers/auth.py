from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from sqlmodel import Session, select
from app.db.session import get_session
from app.models.user import User
from app.core.jwt import create_access_token
from uuid import uuid4
from datetime import datetime, timedelta
import httpx
import os
from urllib.parse import urlencode
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm


auth_router = APIRouter()

# 환경변수 로드
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise RuntimeError("❌ GOOGLE_CLIENT_ID 또는 SECRET이 설정되지 않았습니다 (.env 확인 요망)")

# ✅ 로그인 진입점 - 구글 OAuth URL로 리디렉션
@auth_router.get("/auth/login")
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
        {str(user.id)},
        expires_delta=timedelta(minutes=60)
    )

    response = JSONResponse(content={
        "message": "✅ 로그인 성공",
        "user_id": str(user.id),
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

# 🔒 예시 사용자 (실제론 DB 사용해야 함)
FAKE_USERS_DB = {
    "test@example.com": {
        "username": "test@example.com",
        "password": "1234",  # 해싱 전 (실제론 암호화 필요)
        "user_id": "user-1234"
    }
}

@auth_router.post("/auth/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="잘못된 사용자 이름 또는 비밀번호",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = create_access_token(data={user["user_id"]})
    return {"access_token": token, "token_type": "bearer"}