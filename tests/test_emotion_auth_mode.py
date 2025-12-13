import os
import tempfile
import sys
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlmodel import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

tmp_dir = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(tmp_dir, 'test.db')}")
os.environ["ENV"] = "dev"
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-secret")

from app.main import app  # noqa: E402
from app.db.session import create_all_tables, session_scope  # noqa: E402
from app.models.emotion import EmotionSession  # noqa: E402
from app.models.user import User  # noqa: E402

create_all_tables()
client = TestClient(app)


def test_auth_required_by_default():
    os.environ["EMOTION_NO_AUTH_WEB_TEST"] = "false"
    resp = client.post("/emotion/sessions", json={})
    assert resp.status_code == 401


def test_web_test_mode_allows_anonymous():
    os.environ["EMOTION_NO_AUTH_WEB_TEST"] = "true"
    resp = client.post("/emotion/sessions", json={})
    assert resp.status_code == 200
    data = resp.json()
    web_user_id = UUID(data["user_id"])
    session_id = UUID(data["session_id"])

    with session_scope() as db:
        user = db.exec(
            select(User).where(
                User.email == os.getenv("WEB_TEST_USER_EMAIL", "webtest@local")
            )
        ).first()
        assert user is not None
        assert user.user_id == web_user_id

        session = db.get(EmotionSession, session_id)
        assert session is not None
        assert session.user_id == user.user_id
