from fastapi import APIRouter, Depends
from app.dependencies.auth import get_current_user

user_router = APIRouter()

# ✅ /me: 현재 로그인한 사용자 ID 확인
@user_router.get("/me")
def get_me(user_id: str = Depends(get_current_user)):
    return {"message": "현재 로그인한 사용자 정보입니다", "user_id": user_id}

# ✅ /protected: 단순 JWT 보호 테스트
@user_router.get("/protected")
def protected(user_id: str = Depends(get_current_user)):
    return {"message": f"안녕하세요 {user_id}님, 이 라우터는 인증이 필요합니다."}
