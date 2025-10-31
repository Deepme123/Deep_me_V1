# app/db/session.py
import os
import logging
from urllib.parse import quote_plus
from sqlmodel import SQLModel, create_engine
from sqlalchemy.engine import url as sa_url  # make_url 사용
from contextlib import contextmanager

log = logging.getLogger(__name__)

def _mask(url: str) -> str:
    """로그 출력용 마스킹 (비밀번호 숨김)"""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" in rest and ":" in rest.split("@", 1)[0]:
        creds, tail = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{tail}"
    return url

def _strip_outer_quotes(s: str) -> str:
    if not s:
        return s
    if (s[0] == s[-1]) and s[0] in ("'", '"', "`"):
        return s[1:-1].strip()
    return s

def _build_db_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    url = _strip_outer_quotes(url)

    # 1) postgres:// → postgresql+psycopg2:// 로 교정
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)

    # 2) Render/Neon 계열이면 sslmode=require 강제
    if any(dom in url for dom in ("render.com", "neon.tech")) and "sslmode=" not in url and url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"

    # 3) DATABASE_URL 비었으면 POSTGRES_*로 구성 (로컬/개발용)
    if not url:
        host = os.getenv("POSTGRES_HOST", "").strip()
        port = os.getenv("POSTGRES_PORT", "5432").strip()
        db   = os.getenv("POSTGRES_DB", "").strip()
        user = os.getenv("POSTGRES_USER", "").strip()
        pwd  = os.getenv("POSTGRES_PASSWORD", "").strip()

        if not (host and db and user):
            raise RuntimeError("DATABASE_URL 비어 있음 + POSTGRES_*도 부족함")

        # 특수문자 있는 비번은 URL 인코딩
        if any(ch in pwd for ch in "@:/?#"):
            pwd = quote_plus(pwd)

        url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

        if any(dom in host for dom in ("render.com", "neon.tech")):
            url += "?sslmode=require"

    # 4) 최종 파싱 검증
    try:
        sa_url.make_url(url)
    except Exception as e:
        raise RuntimeError(f"잘못된 DATABASE_URL 형식: {repr(url)} ({e})")

    log.info("DB URL 적용: %s", _mask(url))
    return url

DATABASE_URL = _build_db_url()
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # 이미 있음
    pool_recycle=1800,         # 30분마다 재활성화
    pool_size=5,               # 기본 5
    max_overflow=5,            # 버스트 허용
)

def get_session():
    """FastAPI Depends(get_session)에서 쓰는 generator."""
    from sqlmodel import Session
    with Session(engine) as s:
        yield s

def create_all_tables():
    SQLModel.metadata.create_all(engine)

@contextmanager
def session_scope():
    """
    WebSocket/백그라운드 작업처럼 Depends(get_session) 못 쓰는 구간에서 쓰는 세션 컨텍스트.
    """
    from sqlmodel import Session
    s = Session(engine)
    try:
        yield s
    finally:
        s.close()
