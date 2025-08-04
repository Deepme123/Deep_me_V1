from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.core.jwt import decode_access_token
from fastapi import Request

# ✅ 토큰을 헤더에서 추출하는 OAuth2PasswordBearer 스키마
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")  # 로그인 경로 (예시)

# ✅ 토큰 유효성 검사 및 사용자 ID 반환
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 인증 정보입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["sub"]



# ✅ 쿠키 기반 인증 함수
def get_current_user_from_cookie(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="인증되지 않았습니다 (쿠키 없음)")
    
    payload = decode_access_token(token)
    if payload is None or "sub" not in payload:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")
    
    return payload["sub"]
