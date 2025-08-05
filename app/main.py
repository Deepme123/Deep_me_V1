# app/main.py
from fastapi import FastAPI, HTTPException
from sqlmodel import text
from app.db.session import engine
from app.db.session import create_all_tables
from app.models import emotion, task
from app.routers import emotion
from app.routers.emotion_ws import ws_router as emotion_ws_router
from app.routers import auth 
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, user
from app.routers import task 
import os


app = FastAPI(title="DEEPME Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deep-me-v1.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(emotion.router)
app.include_router(emotion_ws_router)
app.include_router(auth.auth_router)
app.include_router(user.user_router)
app.include_router(task.router)



@app.get("/health/db")
def health_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        # 내부 상세는 로그에 남기고, 외부엔 일반화된 메시지
        raise HTTPException(status_code=500, detail="Database connection failed")

if os.getenv("ENV", "dev") == "dev":
    create_all_tables()

