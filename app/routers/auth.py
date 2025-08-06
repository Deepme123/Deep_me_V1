from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import Session, select
from app.db.session import get_session
from app.models.user import User
from app.core.jwt import create_access_token
from datetime import timedelta
import httpx
import os
from urllib.parse import urlencode

auth_router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# 환경변수 & 상수
# ──────────────────────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

# 쿠키/응답 전략(환경에 따라 분기)
AUTH_SET_COOKIE_ON_POST = os.getenv("AUTH_SET_COOKIE_ON_POST", "false").lower() == "true"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # 배포시 true 권장
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")  # cross-site면 "none"
COOKIE_MAX_AGE = int(os.getenv("COOKIE_MAX_AGE", "3600"))  # 1시간

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
def _issue_jwt(user: User) -> str:
    return create_access_token(str(user.user_id), expires_delta=timedelta(hours=1))

def _set_cookie_if_enabled(response: Response, jwt_token: str) -> None:
    # POST 방식에서도 쿠키 병행을 원하면 AUTH_SET_COOKIE_ON_POST=true
    if response is not None:
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            max_age=COOKIE_MAX_AGE,
        )

def _build_auth_response(user: User, jwt_token: str, response: Response | None = None) -> AuthTokenModel:
    if response is not None and AUTH_SET_COOKIE_ON_POST:
        _set_cookie_if_enabled(response, jwt_token)
    return AuthTokenModel(
        access_token=jwt_token,
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
        # 필요 시: "prompt": "consent", "access_type": "offline"
    }
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}")

@auth_router.get("/auth/callback", response_model=AuthTokenModel, tags=["auth"])
async def google_auth_callback(code: str, db: Session = Depends(get_session)):
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

        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="access_token 누락됨")

        # userinfo 조회
        email, name = await _fetch_userinfo_with_access_token(http, access_token)

    user = _get_or_create_user(db, email=email, name=name)
    jwt_token = _issue_jwt(user)

    # 콜백은 웹 시나리오가 많으므로 쿠키 설정(항상)
    resp = JSONResponse(
        content={
            "message": "✅ 로그인 성공",
            "user_id": str(user.user_id),
            "name": user.name,
            "email": user.email,
        }
    )
    _set_cookie_if_enabled(resp, jwt_token)
    # 콜백도 통일된 형식을 원하면 아래 모델로 변환해도 됨. 지금은 message 포함 JSON 유지 + 쿠키 세팅.
    return resp

# ──────────────────────────────────────────────────────────────────────────────
# 2) POST 기반 (모바일/SPA 권장: id_token)
# ──────────────────────────────────────────────────────────────────────────────
@auth_router.post("/auth/google", response_model=AuthTokenModel, tags=["auth"])
async def auth_with_google(body: GoogleIdTokenReq, response: Response, db: Session = Depends(get_session)):
    async with httpx.AsyncClient() as http:
        email, name = await _verify_id_token_and_extract(http, body.id_token)
    user = _get_or_create_user(db, email=email, name=name)
    jwt_token = _issue_jwt(user)
    # 환경에 따라 쿠키 병행(웹 SPA 혼용 시 유용)
    return _build_auth_response(user, jwt_token, response)

# (선택) access_token 경로
@auth_router.post("/auth/google/access", response_model=AuthTokenModel, tags=["auth"])
async def auth_with_google_access(body: GoogleAccessTokenReq, response: Response, db: Session = Depends(get_session)):
    async with httpx.AsyncClient() as http:
        email, name = await _fetch_userinfo_with_access_token(http, body.access_token)
    user = _get_or_create_user(db, email=email, name=name)
    jwt_token = _issue_jwt(user)
    return _build_auth_response(user, jwt_token, response)

# ──────────────────────────────────────────────────────────────────────────────
# 로그아웃 (쿠키 사용 시)
# ──────────────────────────────────────────────────────────────────────────────
@auth_router.get("/auth/logout", tags=["auth"])
def logout():
    response = JSONResponse(content={"message": "👋 로그아웃 완료"})
    response.delete_cookie("access_token")
    return response

# ──────────────────────────────────────────────────────────────────────────────
# (디버그) Swagger 테스트용 비밀번호 로그인
# ──────────────────────────────────────────────────────────────────────────────
FAKE_USERS_DB = {
    "test@example.com": {"user_id": "user-1234", "password": "1234"}
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
