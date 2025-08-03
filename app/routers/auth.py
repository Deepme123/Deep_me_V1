from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select
from app.db.session import get_session
from app.models.user import User
from uuid import uuid4
from datetime import datetime
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

# ✅ 콜백 - 토큰 교환 + 사용자 정보 → DB 저장
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

    return {
        "message": "✅ 로그인 성공",
        "user_id": str(user.id),
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat(),
    }
