# app/db/session.py
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

from app.models import emotion, task

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

# SQLAlchemy/SQLModel 엔진 생성
engine = create_engine(
    DATABASE_URL,
    echo=False,             # 개발 중 SQL 로그가 필요하면 True
    pool_pre_ping=True,     # 죽은 커넥션 자동 감지
    pool_size=10,           # 기본 커넥션 풀 크기
    max_overflow=20,        # 풀 초과시 추가 허용 커넥션
    future=True,
)

def get_session() -> Generator[Session, None, None]:
    """
    FastAPI 의존성: 요청 단위 세션 관리.
    정상 종료 시 commit, 예외 시 rollback.
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

# 선택: 초기 개발 단계에서만 사용. Alembic 도입 전 임시 테이블 생성에 활용 가능.
def create_all_tables() -> None:
    from app.models import task  # 모델 메타데이터 로드
    SQLModel.metadata.create_all(engine)
