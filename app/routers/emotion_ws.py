# app/routers/emotion_ws.py
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import select, Session
from uuid import UUID
from datetime import datetime
import asyncio
import logging
import inspect
import os
import re
import json
import threading
from contextlib import suppress
from typing import AsyncGenerator, Iterable, List, Tuple, Optional, Callable, TypeVar, Any
from urllib.parse import parse_qs

from sqlalchemy.exc import IntegrityError
from fastapi.encoders import jsonable_encoder

from app.db.session import session_scope  # 컨텍스트 매니저 사용統一
from app.models.emotion import EmotionSession, EmotionStep
from app.schemas.emotion import (
    EmotionOpenRequest,
    EmotionOpenResponse,
    EmotionMessageRequest,
    EmotionMessageResponse,
    EmotionCloseRequest,
    EmotionCloseResponse,
    TaskRecommendRequest,
    TaskRecommendResponse,
)
from app.services.llm_service import stream_noa_response
from app.services.task_recommend import recommend_tasks_from_session_core
from app.core.jwt import decode_access_token
from app.core.prompt_loader import get_system_prompt, get_task_prompt
from app.services.convo_policy import (
    is_activity_turn,
    is_closing_turn,
    mark_activity_injected,
    _turn_count,
    ACTIVITY_STEP_TYPE,
)

logger = logging.getLogger(__name__)
router = APIRouter()
ws_router = router
__all__ = ["ws_router", "router"]

# ──────────────────────────────────────────────────────────────────────────────
# 설정/상수

class WSConfig:
    SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", "20"))
    WS_IDLE_TIMEOUT: float = float(os.getenv("WS_IDLE_TIMEOUT", "120"))
    WS_SEND_BUFFER: int = int(os.getenv("WS_SEND_BUFFER", "20"))
    WS_HEARTBEAT_SEC: float = float(os.getenv("WS_HEARTBEAT_SEC", "15"))
    LLM_STREAM_TIMEOUT: float = float(os.getenv("LLM_STREAM_TIMEOUT", "75"))
    RECOMMEND_TIMEOUT: float = float(os.getenv("RECOMMEND_TIMEOUT", "15"))
    WS_HISTORY_TURNS: int = max(5, min(10, int(os.getenv("WS_HISTORY_TURNS", "8"))))
    WS_ALLOWED_ORIGINS: list[str] = [
        o.strip().rstrip("/")
        for o in (os.getenv("WS_ALLOWED_ORIGINS") or "http://localhost:3000,http://localhost:8000,http://127.0.0.1").split(",")
        if o.strip()
    ]
    WS_MAX_USER_TEXT_LEN: int = int(os.getenv("WS_MAX_USER_TEXT_LEN", str(8 * 1024)))  # bytes/ASCII-ish

CFG = WSConfig()

# 메시지 타입 상수
MSG_OPEN = "open"
MSG_MESSAGE = "message"
MSG_CLOSE = "close"
MSG_TASK_RECOMMEND = "task_recommend"
MSG_PING = "ping"
MSG_PONG = "pong"

T = TypeVar("T")

# ──────────────────────────────────────────────────────────────────────────────
# 유틸

def _safe_str(x: object) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def _mask_preview(s: str, k: int = 80) -> str:
    s = s.replace("\n", " ")
    return (s[:k] + "…") if len(s) > k else s
class TurnLimitReached(Exception):
    """Raised when a session has reached the configured turn limit."""
    pass

class IdleTimeout(Exception):
    """Raised when websocket idle timeout hit."""
    pass

def _normalize_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    return origin.strip().rstrip("/").lower()

def _is_origin_allowed(origin: str | None) -> bool:
    norm = _normalize_origin(origin)
    if not norm:
        return False
    return norm in {_normalize_origin(o) for o in CFG.WS_ALLOWED_ORIGINS}

def _extract_bearer_token(websocket: WebSocket) -> str | None:
    auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    if not auth_header:
        return None
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip() or None
    return None

def _extract_token_fallback(websocket: WebSocket) -> str | None:
    # query param fallback
    q = websocket.query_params
    if q:
        for key in ("access_token", "token", "auth_token"):
            if q.get(key):
                return q.get(key)
    return None

def _run_with_session(fn: Callable[[Session], T], *args, **kwargs) -> T:
    with session_scope() as db:
        return fn(db, *args, **kwargs)

async def _with_db(fn: Callable[[Session], T], *args, **kwargs) -> T:
    return await asyncio.to_thread(_run_with_session, fn, *args, **kwargs)

async def _ws_send_safe(websocket: WebSocket, data: dict, *, timeout: float | None = None) -> None:
    payload = jsonable_encoder(data, exclude_none=True)

    async def _send():
        await websocket.send_json(payload)

    try:
        if timeout:
            await asyncio.wait_for(_send(), timeout=timeout)
        else:
            await _send()
    except Exception as e:
        logger.warning("WS send failed | %s | keys=%s", _safe_str(e), list(data.keys()))

async def _ws_recv_safe(
    websocket: WebSocket,
    *,
    timeout: float | None = None,
    raise_on_timeout: bool = False,
    strict_json: bool = True,
) -> dict | None:
    """
    클라이언트 프레임 관용 처리:
    - JSON: dict
    - 단어: ping/open/close
    - 쿼리스트링: type=message&text=...
    - 그 외 문자열: {"type":"message","text": "..."}
    timeout 시 {"type":"ping"} 반환
    """
    try:
        event = await asyncio.wait_for(websocket.receive(), timeout=timeout) if timeout else await websocket.receive()
    except asyncio.TimeoutError:
        if raise_on_timeout:
            raise IdleTimeout()
        return {"type": "ping"}
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.warning("WS recv() failed | %s", _safe_str(e))
        return None

    # 원시 프레임 로깅
    try:
        logger.warning(
            "WS RAW EVENT | keys=%s txt=%r bin=%s",
            list(event.keys()),
            (event.get("text") or "")[:80],
            bool(event.get("bytes")),
        )
    except Exception:
        pass

    if event.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(event.get("code"))

    text = event.get("text")
    data = event.get("bytes")

    if text is not None:
        t = text.strip()

        # 1) JSON 시도 (+ 레거시 정규화)
        if t and (t.startswith("{") or t.startswith("[")):
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    if "type" in obj:
                        return obj
                    if "user_input" in obj or "text" in obj:
                        text_val = obj.get("user_input") or obj.get("text") or ""
                        norm = {"type": "message", "text": text_val}
                        for k in ("step_type", "emotion_label", "topic", "trigger_summary", "insight_summary", "max_items", "access_token"):
                            if k in obj:
                                norm[k] = obj[k]
                        return norm
            except Exception:
                if strict_json:
                    raise
                pass

        # 2) 단어 명령
        tl = t.lower()
        if tl == "ping":
            return {"type": "ping"}
        if tl == "open":
            return {"type": "open"}
        if tl == "close":
            return {"type": "close"}

        # 3) 쿼리스트링
        if "=" in t and "&" in t:
            try:
                q = parse_qs(t, keep_blank_values=True)
                obj = {k: (v[0] if isinstance(v, list) and v else v) for k, v in q.items()}
                if "type" in obj:
                    return obj
            except Exception:
                pass

        # 4) 일반 텍스트 → 사용자 메시지
        return {"type": "message", "text": t}

    if data is not None:
        raise ValueError("binary_frames_not_allowed")

    return None

def _ensure_uuid(x: str | UUID | None) -> UUID | None:
    if x is None:
        return None
    return UUID(str(x))

def _iter_chunks(gen: Iterable[str] | AsyncGenerator[str, None]):
    """
    sync/async 제너레이터를 항상 async 제너레이터로 래핑.
    """
    if inspect.isasyncgen(gen):
        async def _ait():
            async for x in gen:
                yield x
        return _ait()
    else:
        async def _ait2():
            loop = asyncio.get_running_loop()
            q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

            def _worker():
                try:
                    for x in gen:
                        loop.call_soon_threadsafe(q.put_nowait, ("data", x))
                except Exception as exc:
                    loop.call_soon_threadsafe(q.put_nowait, ("err", exc))
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, ("done", None))

            threading.Thread(target=_worker, daemon=True).start()

            while True:
                kind, payload = await q.get()
                if kind == "data":
                    yield payload
                elif kind == "err":
                    raise payload
                else:
                    break
        return _ait2()

def _steps_to_conversation(steps: List[EmotionStep]) -> List[Tuple[str, str]]:
    """DB steps → ('user'|'assistant', text) 시퀀스."""
    conv: List[Tuple[str, str]] = []
    for s in steps:
        if s.step_type == "user" and s.user_input:
            conv.append(("user", s.user_input))
        elif s.step_type == "assistant" and s.gpt_response:
            conv.append(("assistant", s.gpt_response))
    return conv

def _create_emotion_session(db: Session, uid_val: UUID | None) -> EmotionSession:
    s = EmotionSession(
        user_id=uid_val,
        started_at=datetime.utcnow(),
        emotion_label=None,
        topic=None,
        trigger_summary=None,
        insight_summary=None,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s

def _prepare_message_context(db: Session, session_id: UUID, user_text: str) -> dict:
    steps: List[EmotionStep] = list(
        db.exec(
            select(EmotionStep)
            .where(EmotionStep.session_id == session_id)
            .order_by(EmotionStep.created_at.asc())
        )
    )
    # Keep only the most recent 5–10 turns to reduce LLM context size.
    max_entries = CFG.WS_HISTORY_TURNS * 2  # user+assistant per turn
    steps_for_convo = steps[-max_entries:] if len(steps) > max_entries else steps

    if _turn_count(db, session_id) >= CFG.SESSION_MAX_TURNS:
        raise TurnLimitReached()

    want_activity = is_activity_turn(
        user_text=user_text,
        db=db,
        session_id=session_id,
        steps=steps,
    )
    want_close = is_closing_turn(db, session_id)

    last_order = steps[-1].step_order if steps else 0
    # user/assistant orders reserved but not committed until after successful LLM turn
    user_order = last_order + 1
    assistant_order = user_order + 1
    convo = _steps_to_conversation(steps_for_convo) + [("user", user_text)]
    return {
        "want_activity": want_activity,
        "want_close": want_close,
        "user_order": user_order,
        "assistant_order": assistant_order,
        "conversation": convo,
    }

def _commit_full_turn(
    db: Session,
    session_id: UUID,
    user_text: str,
    assistant_text: str,
    user_order: int,
    assistant_order: int,
    *,
    add_activity_marker: bool,
) -> None:
    # Re-check limit just before commit to avoid races between turns.
    if _turn_count(db, session_id) >= CFG.SESSION_MAX_TURNS:
        raise TurnLimitReached()

    step_user = EmotionStep(
        session_id=session_id,
        step_order=user_order,
        step_type="user",
        user_input=user_text,
        gpt_response="",
        created_at=datetime.utcnow(),
        insight_tag=None,
    )
    step_assistant = EmotionStep(
        session_id=session_id,
        step_order=assistant_order,
        step_type="assistant",
        user_input="",
        gpt_response=assistant_text,
        created_at=datetime.utcnow(),
        insight_tag=None,
    )
    db.add(step_user)
    db.add(step_assistant)

    if add_activity_marker:
        marker = EmotionStep(
            session_id=session_id,
            step_order=assistant_order + 1,
            step_type=ACTIVITY_STEP_TYPE,
            user_input="",
            gpt_response="",
            created_at=datetime.utcnow(),
            insight_tag=None,
        )
        db.add(marker)

    db.commit()

def _close_session_record(db: Session, session_id: UUID, payload: EmotionCloseRequest) -> None:
    s = db.get(EmotionSession, session_id)
    if s:
        s.ended_at = datetime.utcnow()
        if payload.emotion_label:
            s.emotion_label = payload.emotion_label
        if payload.topic:
            s.topic = payload.topic
        if payload.trigger_summary:
            s.trigger_summary = payload.trigger_summary
        if payload.insight_summary:
            s.insight_summary = payload.insight_summary
        db.add(s)
        db.commit()

async def _recommend_tasks_async(session_id: UUID, max_items: int) -> List[dict]:
    def _work() -> List[dict]:
        with session_scope() as db:
            sess = db.get(EmotionSession, session_id)
            if not sess or not sess.user_id:
                return []
            uid = sess.user_id

        tasks = recommend_tasks_from_session_core(
            user_id=uid,
            session_id=session_id,
            n=max(1, max_items),
        )
        return jsonable_encoder(tasks, exclude_none=True) if tasks else []

    return await asyncio.to_thread(_work)

# ──────────────────────────────────────────────────────────────────────────────
# Leak guard

class LeakGuard:
    _DEFAULT_MARKERS = [
        r"<<SYS>>",
        r"\bBEGIN SYSTEM PROMPT\b",
        r"\[\s*SYSTEM\s*\]",
        r"\bDO NOT DISCLOSE\b",
        r"\bdeveloper prompt\b",
    ]

    def __init__(self) -> None:
        self.markers: List[str] = list(self._DEFAULT_MARKERS)
        self.ngram: int = int(os.getenv("LEAK_GUARD_NGRAM", "20"))
        self.min_match: int = int(os.getenv("LEAK_GUARD_MIN_MATCH", "3"))
        self.mode: str = os.getenv("LEAK_GUARD_MODE", "mask")  # 'mask' | 'drop'

    def fingerprint(self, text: str, n: Optional[int] = None) -> set[int]:
        n = self.ngram if n is None else n
        if not text:
            return set()
        step = max(3, n // 2)
        return {hash(text[i:i+n]) for i in range(0, max(0, len(text) - n + 1), step)}

    def _might_leak(self, text: str, sys_fp: set[int], n: Optional[int] = None) -> bool:
        n = self.ngram if n is None else n
        if not text or not sys_fp:
            return False
        step = max(3, n // 2)
        fp = {hash(text[i:i+n]) for i in range(0, max(0, len(text) - n + 1), step)}
        return len(sys_fp & fp) >= self.min_match

    def _redact(self, text: str) -> str:
        out = text
        for pat in self.markers:
            out = re.sub(pat, "[redacted]", out, flags=re.I)
        return out

    def sanitize_out(self, piece: str, sys_fp: set[int]) -> str:
        if not isinstance(piece, str) or not piece:
            return ""
        if self._might_leak(piece, sys_fp):
            if self.mode == "drop":
                return ""
            return self._redact(piece)
        return self._redact(piece)

# ──────────────────────────────────────────────────────────────────────────────
# 라우터

@router.websocket("/ws/emotion")
async def ws_emotion(websocket: WebSocket):
    # Pre-accept origin + JWT validation (header/query)
    origin = websocket.headers.get("origin") or websocket.headers.get("Origin")
    if not _is_origin_allowed(origin):
        logger.warning("WS origin rejected | origin=%s", _safe_str(origin))
        await websocket.close(code=1008, reason="origin_not_allowed")
        return

    raw_token = _extract_bearer_token(websocket) or _extract_token_fallback(websocket)
    auth_user_id: UUID | None = None
    if raw_token:
        try:
            auth_payload = decode_access_token(raw_token)
            auth_user_id = _ensure_uuid(auth_payload.user_id)
        except Exception:
            await websocket.close(code=4401, reason="invalid_token")
            return

    subproto = websocket.headers.get("sec-websocket-protocol")
    await websocket.accept(subprotocol=subproto if subproto else None)

    session_id: UUID | None = None
    leak_guard = LeakGuard()
    sys_fp: set[int] = set()
    send_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=CFG.WS_SEND_BUFFER)
    loop = asyncio.get_running_loop()
    last_active = loop.time()
    shutdown = asyncio.Event()
    recommend_fuse_tripped = False

    async def close_ws(code: int = 1000, reason: str = "") -> None:
        if shutdown.is_set():
            return
        shutdown.set()
        with suppress(Exception):
            await websocket.close(code=code, reason=reason)

    async def sender():
        nonlocal last_active
        try:
            while not shutdown.is_set():
                try:
                    item = await asyncio.wait_for(send_queue.get(), timeout=CFG.WS_HEARTBEAT_SEC)
                except asyncio.TimeoutError:
                    await _ws_send_safe(websocket, {"type": MSG_PING})
                    continue
                await _ws_send_safe(websocket, item)
                last_active = loop.time()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WS sender loop error | %s", _safe_str(e))
        finally:
            shutdown.set()

    async def guard_send(data: dict):
        if shutdown.is_set():
            return
        try:
            await asyncio.wait_for(send_queue.put(data), timeout=CFG.WS_HEARTBEAT_SEC)
        except asyncio.TimeoutError:
            shutdown.set()
            raise RuntimeError("send_queue_backpressure")

    send_task = asyncio.create_task(sender())

    # ── 연결 직후 인증된 사용자 기준으로 세션 자동 오픈
    async def _bootstrap_open_if_possible():
        nonlocal session_id, sys_fp, auth_user_id
        try:
            uid = auth_user_id
            if not uid:
                return False

            # user 존재 검증 (없으면 중단)
            try:
                from app.models.user import User  # 지연 import로 순환참조 방지
            except Exception:
                User = None  # type: ignore

            if uid and User:
                user_exists = await _with_db(lambda db: db.get(User, uid) is not None)
                if not user_exists:
                    logger.warning("bootstrap: user not found | user_id=%s", uid)
                    await websocket.close(code=4401, reason="user_not_found")
                    return

            try:
                session = await _with_db(_create_emotion_session, uid)
            except IntegrityError as ie:
                logger.warning("bootstrap commit FK failed; retrying as anonymous | %s", _safe_str(ie))
                session = await _with_db(_create_emotion_session, None)

            session_id = session.session_id  # ← 세션 아이디 보관

            system_prompt = get_system_prompt()
            sys_fp = leak_guard.fingerprint(system_prompt)
            await guard_send(
                EmotionOpenResponse(type="open_ok", session_id=session_id, turns=0).model_dump(),
            )
            logger.info("WS bootstrap open_ok sent | session_id=%s", session_id)
        except Exception:
            logger.exception("WS bootstrap open failed")
            await close_ws(code=1011, reason="bootstrap_failed")
            return False
        return True

    # if we already have a token, bootstrap immediately; otherwise wait for open payload
    if raw_token:
        ok = await _bootstrap_open_if_possible()
        if not ok:
            with suppress(Exception):
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
            return

    try:
        while not shutdown.is_set():
            # 수신을 먼저 기다림
            try:
                msg = await _ws_recv_safe(websocket, timeout=CFG.WS_IDLE_TIMEOUT, raise_on_timeout=True, strict_json=True)
            except IdleTimeout:
                await guard_send({"type": "error", "message": "idle_timeout"})
                break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning("WS recv failed | %s", _safe_str(e))
                await guard_send({"type": "error", "message": "recv_failed"})
                break
            if msg is None or shutdown.is_set():
                continue

            # 파싱된 메시지 타입 로깅
            try:
                logger.warning("WS PARSED | %s", msg.get("type"))
            except Exception:
                pass

            # 활동 시각 갱신
            last_active = loop.time()

            typ = msg.get("type")

            if typ == MSG_PING:
                await guard_send({"type": MSG_PONG})
                continue

            # ── 세션 열기
            if typ == MSG_OPEN:
                # allow token via payload if missing during handshake
                if not auth_user_id:
                    tok = msg.get("access_token")
                    if tok:
                        try:
                            auth_payload = decode_access_token(tok)
                            auth_user_id = _ensure_uuid(auth_payload.user_id)
                        except Exception:
                            await guard_send({"type": "error", "message": "invalid_token"})
                            await close_ws(code=4401, reason="invalid_token")
                            break
                    if not auth_user_id:
                        await guard_send({"type": "error", "message": "auth_required"})
                        await close_ws(code=4401, reason="auth_required")
                        break

                if session_id:
                    await guard_send(EmotionOpenResponse(
                        type="open_ok",
                        session_id=session_id,
                        turns=0,
                    ).model_dump())
                    continue
                try:
                    payload = EmotionOpenRequest(**msg)
                except Exception as e:
                    await guard_send({"type": "error", "message": f"bad open payload: {e}"})
                    continue

                uid = auth_user_id

                try:
                    session = await _with_db(_create_emotion_session, uid)
                except IntegrityError as ie:
                    logger.warning("open commit FK failed; retrying anonymous | %s", _safe_str(ie))
                    session = await _with_db(_create_emotion_session, None)

                session_id = session.session_id

                system_prompt = get_system_prompt()
                sys_fp = leak_guard.fingerprint(system_prompt)

                await guard_send(EmotionOpenResponse(
                    type="open_ok",
                    session_id=session_id,
                    turns=0,
                ).model_dump())

            # ── 사용자 메시지 처리
            elif typ == MSG_MESSAGE:
                if not session_id:
                    await guard_send({"type": "error", "message": "no session"})
                    continue

                try:
                    payload = EmotionMessageRequest(**msg)
                except Exception as e:
                    await guard_send({"type": "error", "message": f"bad message payload: {e}"})
                    continue

                user_text = payload.text or ""
                if len(user_text.encode("utf-8")) > CFG.WS_MAX_USER_TEXT_LEN:
                    await guard_send({"type": "error", "message": "message_too_large"})
                    continue
                logger.info("WS recv user | %s", _mask_preview(user_text, 100))

                # MARK A: DB 조회 직전
                logger.warning("WS MARK A | before DB fetch")

                try:
                    prep = await _with_db(_prepare_message_context, session_id, user_text)
                    want_activity = bool(prep.get("want_activity"))
                    want_close = bool(prep.get("want_close"))
                    user_order = int(prep.get("user_order") or 0)
                    assistant_order = int(prep.get("assistant_order") or 0)
                    convo = prep.get("conversation") or []
                except TurnLimitReached:
                    await guard_send({"type": "limit", "message": "max turns reached"})
                    continue
                except Exception as e:
                    logger.exception("WS DB fetch failed")
                    await guard_send({"type": "error", "message": f"db_failed: {_safe_str(e)}"})
                    continue

                # MARK B: DB 조회 통과
                logger.warning("WS MARK B | after DB fetch, before prompt")

                # 프롬프트 로딩 + MARK C
                try:
                    system_prompt = get_system_prompt()
                    task_prompt = get_task_prompt() if want_activity else None
                except Exception as e:
                    logger.exception("WS prompt load failed")
                    await guard_send({"type": "error", "message": f"prompt_failed: {_safe_str(e)}"})
                    continue

                logger.warning("WS MARK C | after prompt load, before stream")

                # 스트리밍 호출 + 누적 버퍼
                assistant_chunks: List[str] = []

                async def _consume_stream():
                    async for piece in _iter_chunks(
                        stream_noa_response(
                            system_prompt=system_prompt,
                            task_prompt=task_prompt,
                            conversation=convo,
                            temperature=0.7,
                            max_tokens=800,
                        )
                    ):
                        safe_piece = leak_guard.sanitize_out(piece, sys_fp)
                        if not safe_piece:
                            continue
                        assistant_chunks.append(safe_piece)
                        logger.debug("WS delta | %s", _mask_preview(safe_piece))
                        await guard_send(EmotionMessageResponse(type="message_delta", delta=safe_piece).model_dump())

                stream_failed_reason: str | None = None
                await guard_send(EmotionMessageResponse(type="message_start").model_dump())
                try:
                    await asyncio.wait_for(_consume_stream(), timeout=CFG.LLM_STREAM_TIMEOUT)
                except asyncio.TimeoutError:
                    stream_failed_reason = "stream_timeout"
                except Exception as e:
                    stream_failed_reason = f"stream_failed:{_safe_str(e)}"
                finally:
                    await guard_send(EmotionMessageResponse(type="message_end").model_dump())

                if stream_failed_reason:
                    await guard_send({"type": "error", "message": stream_failed_reason, "turn_dropped": True})
                    continue

                assistant_text = "".join(assistant_chunks).strip()
                if not assistant_text:
                    # 내용이 비면 저장하지 않고 종료
                    await guard_send({"type": "error", "message": "empty_assistant_response", "turn_dropped": True})
                    continue

                # ② 사용자/어시스턴트 스텝을 한 트랜잭션으로 커밋
                try:
                    await _with_db(
                        _commit_full_turn,
                        session_id,
                        user_text,
                        assistant_text,
                        user_order,
                        assistant_order,
                        add_activity_marker=want_activity,
                    )
                except TurnLimitReached:
                    await guard_send({"type": "limit", "message": "max turns reached"})
                    continue
                except Exception:
                    logger.exception("WS assistant-step commit failed")
                    await guard_send({"type": "error", "message": "server_error:assistant_step_commit"})
                    continue

                await guard_send(
                    EmotionMessageResponse(
                        type="message",
                        message=assistant_text,
                    ).model_dump()
                )

                if want_close:
                    await guard_send({"type": "suggest_close"})
                
                if want_activity:
                    try:
                        items = await _recommend_tasks_async(session_id, 5)
                    except Exception as e:
                        logger.warning("task recommend failed | %s", _safe_str(e))
                    else:
                        if items:
                            await guard_send(
                                TaskRecommendResponse(
                                    type="task_recommend_ok",
                                    items=items,
                                ).model_dump()
                            )

            # ── 세션 종료
            elif typ == MSG_CLOSE:
                if not session_id:
                    await guard_send({"type": "error", "message": "no session"})
                    continue

                try:
                    payload = EmotionCloseRequest(**msg)
                except Exception as e:
                    await guard_send({"type": "error", "message": f"bad close payload: {e}"})
                    continue

                try:
                    await _with_db(_close_session_record, session_id, payload)
                except Exception as e:
                    logger.warning("WS close update failed | %s", _safe_str(e))
                    await guard_send({"type": "error", "message": "close_failed"})
                    continue

                await guard_send(EmotionCloseResponse(type="close_ok").model_dump())
                break

            # ── 태스크 추천
            elif typ == MSG_TASK_RECOMMEND:
                if not session_id:
                    await guard_send({"type": "error", "message": "no session"})
                    continue

                try:
                    payload = TaskRecommendRequest(**msg)
                except Exception as e:
                    await guard_send({"type": "error", "message": f"bad task payload: {e}"})
                    continue

                if recommend_fuse_tripped:
                    await guard_send({"type": "error", "message": "recommend_unavailable"})
                    continue

                try:
                    recs = await asyncio.wait_for(
                        _recommend_tasks_async(session_id, payload.max_items or 5),
                        timeout=CFG.RECOMMEND_TIMEOUT,
                    )
                except Exception as e:
                    recommend_fuse_tripped = True
                    logger.error("task recommend failed | %s", _safe_str(e))
                    await guard_send({"type": "error", "message": f"recommend failed: {_safe_str(e)}"})
                    continue

                await guard_send(TaskRecommendResponse(
                    type="task_recommend_ok",
                    items=recs,
                ).model_dump())

            else:
                await guard_send({"type": "error", "message": f"unknown type: {typ}"})

    except WebSocketDisconnect:
        shutdown.set()
    except Exception:
        logger.exception("WS fatal error")
        with suppress(Exception):
            await _ws_send_safe(websocket, {"type": "error", "message": "fatal"})
    finally:
        shutdown.set()
        with suppress(Exception):
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
        with suppress(Exception):
            await websocket.close()
