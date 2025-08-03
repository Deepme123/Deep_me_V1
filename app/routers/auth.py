from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select
from app.db.session import get_session
from app.models.user import User
from uuid import uuid4
from datetime import datetime
import httpx
import os

auth_router = APIRouter()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")


if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise RuntimeError("GOOGLE_CLIENT_ID or SECRET is not set in environment variables")


@auth_router.get("/auth/login")
def login_via_google():
    google_oauth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
    )
    return RedirectResponse(url=google_oauth_url)


@auth_router.get("/auth/callback")
async def google_auth_callback(code: str, db: Session = Depends(get_session)):
    # 1. 토큰 요청
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=data)
        token_json = token_response.json()

    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token not received")

    # 2. 사용자 정보 요청
    user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        user_response = await client.get(user_info_url, headers=headers)
        user_data = user_response.json()

    email = user_data.get("email")
    name = user_data.get("name")

    if not email or not name:
        raise HTTPException(status_code=400, detail="Invalid user info")

    # 3. 기존 유저 확인 or 신규 생성
    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(name=name, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)

    return {
        "message": "로그인 성공",
        "user_id": str(user.id),
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat(),
    }
