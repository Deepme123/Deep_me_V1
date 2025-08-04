from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.core.jwt import decode_access_token

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
