import os
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.user import User


def ensure_web_test_user(db: Session) -> UUID:
    """
    Upsert a deterministic web-test user and return its UUID.
    """
    email = os.getenv("WEB_TEST_USER_EMAIL", "webtest@local")
    name = os.getenv("WEB_TEST_USER_NAME", "Web Test User")

    existing = db.exec(select(User).where(User.email == email)).first()
    if existing:
        return existing.user_id

    user = User(name=name, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user.user_id


def resolve_emotion_user_id(db: Session, current_user: str | UUID | None) -> UUID:
    """
    Resolve the emotion user id based on auth or web-test mode.
    """
    if current_user:
        try:
            return UUID(str(current_user))
        except Exception:
            raise HTTPException(status_code=401, detail="invalid_token")

    allow_web_test = os.getenv("EMOTION_NO_AUTH_WEB_TEST", "false").lower() == "true"
    if allow_web_test:
        return ensure_web_test_user(db)

    raise HTTPException(status_code=401, detail="Authentication required")
