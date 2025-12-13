from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from app.core.jwt import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
oauth2_optional_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token", auto_error=False
)


def _extract_jwt(request: Request, token: str | None) -> str | None:
    return token or request.cookies.get("access_token")


def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    """Strict auth dependency; raises when no/invalid token."""
    jwt_token = _extract_jwt(request, token)
    if not jwt_token:
        raise HTTPException(status_code=401, detail="?¸ì¦ ?•ë³´ê°€ ?†ìŠµ?ˆë‹¤")

    payload = decode_access_token(jwt_token)
    if payload is None or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="? íš¨?˜ì? ?Šì? ?¸ì¦ ?•ë³´?…ë‹ˆ??",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["sub"]


def get_current_user_optional(
    request: Request, token: str | None = Depends(oauth2_optional_scheme)
):
    """Lenient auth dependency; returns sub when valid, else None."""
    jwt_token = _extract_jwt(request, token)
    if not jwt_token:
        return None

    payload = decode_access_token(jwt_token)
    if payload is None or "sub" not in payload:
        return None
    return payload["sub"]
