from fastapi import Request
from fastapi.security import OAuth2PasswordBearer
from app.core.jwt import decode_access_token
from fastapi import Depends, HTTPException, status

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ✅ 쿠키 또는 헤더에서 토큰을 가져오는 함수
def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    jwt_token = token or request.cookies.get("access_token")
    if not jwt_token:
        raise HTTPException(status_code=401, detail="인증 정보가 없습니다")

    payload = decode_access_token(jwt_token)
    if payload is None or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 인증 정보입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["sub"]
