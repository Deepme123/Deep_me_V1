from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Request, Response, status
from sqlmodel import Session, select

from app.core.tokens import (
    REFRESH_COOKIE_NAME,
    clear_refresh_cookie,
    create_access_token,
    create_refresh_token,
    new_refresh_jti,
    set_refresh_cookie,
    sha256_hex,
    verify_refresh_token,
)
from app.models.refresh_token import RefreshToken
from app.models.user import User


async def _extract_refresh_token(request: Request) -> Optional[str]:
    """Read refresh token from HttpOnly cookie first, then JSON body fallback."""
    rt = request.cookies.get(REFRESH_COOKIE_NAME)
    if rt:
        return rt
    try:
        body = await request.json()
        if isinstance(body, dict):
            return body.get("refresh_token")
    except Exception:
        return None
    return None


async def refresh_tokens(
    *,
    request: Request,
    response: Response,
    db: Session,
    set_access_cookie: bool = False,
    access_cookie_secure: bool = False,
    access_cookie_samesite: str = "lax",
    access_cookie_max_age: int = 0,
) -> dict:
    """
    Verify incoming refresh token, rotate it, and issue a new short-lived access token.
    - Enforces token reuse detection (revoked/replaced)
    - Stores hashed refresh tokens only
    - Rotates RT and sets HttpOnly cookie
    """
    rt = await _extract_refresh_token(request)
    if not rt:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing")

    try:
        payload = verify_refresh_token(rt)
    except Exception:
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    sub = payload.get("sub")
    jti = payload.get("jti")
    if not sub or not jti:
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token payload")

    rt_row = db.get(RefreshToken, jti)
    if rt_row is None:
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not recognized")

    # Reuse detection: anything already revoked or replaced is invalid and revokes all for this user.
    if rt_row.revoked_at is not None or rt_row.replaced_by is not None:
        for row in db.exec(select(RefreshToken).where(RefreshToken.user_id == rt_row.user_id)):
            if row.revoked_at is None:
                row.revoked_at = datetime.utcnow()
        db.commit()
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token reused")

    # Constant-time-ish hash comparison (sha256 hex)
    if rt_row.token_hash != sha256_hex(rt):
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token tampered")

    user = db.get(User, UUID(sub))
    if user is None:
        clear_refresh_cookie(response)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Rotate tokens
    new_access = create_access_token(user.user_id)
    new_jti = new_refresh_jti()
    new_rt, new_exp = create_refresh_token(user.user_id, new_jti)

    rt_row.revoked_at = datetime.utcnow()
    rt_row.replaced_by = new_jti

    db.add(
        RefreshToken(
            jti=new_jti,
            user_id=user.user_id,
            token_hash=sha256_hex(new_rt),
            expires_at=new_exp,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    )
    db.commit()

    set_refresh_cookie(response, new_rt)

    if set_access_cookie:
        response.set_cookie(
            key="access_token",
            value=new_access,
            httponly=True,
            secure=access_cookie_secure,
            samesite=access_cookie_samesite,
            max_age=access_cookie_max_age,
            path="/",
        )

    return {
        "access_token": new_access,
        "refresh_token": new_rt,
        "token_type": "bearer",
        "expires_in": access_cookie_max_age,
        "user_id": str(user.user_id),
    }
