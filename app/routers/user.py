from fastapi import APIRouter, Depends
from app.dependencies.auth import get_current_user
from app.dependencies.auth import get_current_user_from_cookie

user_router = APIRouter()

@user_router.get("/me/cookie")
def get_me_cookie(user_id: str = Depends(get_current_user_from_cookie)):
    return {"message": "✅ 쿠키 인증 성공", "user_id": user_id}

@user_router.get("/me/bearer")
def get_me_bearer(user_id: str = Depends(get_current_user)):
    return {"message": "✅ 헤더(Bearer) 인증 성공", "user_id": user_id}











