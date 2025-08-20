# app/main.py
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import text

from app.core.logging_config import setup_logging
from app.db.session import engine, create_all_tables

# 모델 모듈 임포트(테이블 등록 보장용)
from app.models import emotion as _m_emotion  # noqa: F401
from app.models import task as _m_task  # noqa: F401
from app.models import refresh_token as _m_refresh  # noqa: F401

# 라우터
from app.routers import emotion, auth, user, task
from app.routers.emotion_ws import ws_router as emotion_ws_router

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DEEPME Backend",
    version=os.getenv("APP_VERSION", "0.1.0"),
)

# CORS
_default_origins = "https://deep-me-v1.onrender.com,http://localhost:3000,http://localhost:5173"
origins = [
    o.strip()
    for o in os.getenv("CORS_ALLOW_ORIGINS", _default_origins).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(emotion.router)
app.include_router(emotion_ws_router)
app.include_router(auth.auth_router)
app.include_router(user.user_router)
app.include_router(task.router)


@app.on_event("startup")
def _startup() -> None:
    if os.getenv("ENV", "dev") == "dev":
        create_all_tables()
        logger.info("Tables ensured in dev environment")


@app.get("/health")
def health_app():
    return {"ok": True}


@app.get("/health/db")
def health_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception:
        raise HTTPException(status_code=500, detail="Database connection failed")
