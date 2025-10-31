from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import select
from uuid import UUID
from datetime import datetime
import asyncio
import logging
import inspect
import os
import re
import json
from contextlib import suppress
from typing import AsyncGenerator, Generator, Iterable, List, Tuple, Optional

from sqlalchemy.exc import IntegrityError

from app.db.session import session_scope  # ← 컨텍스트 매니저만 임포트
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

CFG = WSConfig()

# 메시지 타입 상수
MSG_OPEN = "open"
MSG_MESSAGE = "message"
MSG_CLOSE = "close"
MSG_TASK_RECOMMEND = "task_recommend"
MSG_PING = "ping"
MSG_PONG = "pong"

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

async def _ws_send_safe(ws: WebSocket, data: dict, *, timeout: float | None = None) -> None:
    async def _send():
        await ws.send_json(data)
    try:
        if timeout:
            await asyncio.wait_for(_send(), timeout=timeout)
        else:
            await _send()
    except Exception as e:
        logger.warning("WS send failed | %s | keys=%s", _safe_str(e), list(data.keys()))

async def _ws_recv_safe(ws: WebSocket, *, timeout: float | None = None) -> dict | None:
    """원시 프레임 1개 수신 → JSON/dict 또는 텍스트 관용 처리."""
    try:
        event = await asyncio.wait_for(ws.receive(), timeout=timeout) if timeout else await ws.receive()
    except asyncio.TimeoutError:
        return {"type": MSG_PING}
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.warning("WS recv() failed | %s", _safe_str(e))
        return None

    if event.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(event.get("code"))

    text = event.get("text")
    data = event.get("bytes")

    if text is not None:
        t = text.strip()
        # JSON이면 파싱
        try:
            return json.loads(t)
        except Exception:
            tl = t.lower()
            if tl == "ping":
                return {"type": MSG_PING}
            if tl == "open":
                return {"type": MSG_OPEN}
            logger.warning("WS non-JSON text ignored | %s", _mask_preview(t, 60))
            return None

    if data is not None:
        # 현재 바이너리는 미사용
        logger.warning("WS binary frame ignored | len=%s", len(data))
        return None

    return None

def _ensure_uuid(x: str | UUID | None) -> UUID | None:
    if x is None:
        return None
    return UUID(str(x))

def _iter_chunks(gen: Iterable[str] | AsyncGenerator[str, None]) -> AsyncGenerator[str, None] | Generator[str, None, None]:
    """sync/async 제너레이터를 모두 지원."""
    if inspect.isasyncgen(gen):
        async def _ait():
            async for x in gen:
                yield x
        return _ait()
    return gen

def _steps_to_conversation(steps: List[EmotionStep]) -> List[Tuple[str, str]]:
    """DB steps → ('user'|'assistant', text) 시퀀스."""
    conv: List[Tuple[str, str]] = []
    for s in steps:
        if s.step_type == "user" and s.user_input:
            conv.append(("user", s.user_input))
        elif s.step_type == "assistant" and s.gpt_response:
            conv.append(("assistant", s.gpt_response))
    return conv

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
async def ws_emotion(ws: WebSocket):
    await ws.accept()
    token: str | None = None
    session_id: UUID | None = None
    leak_guard = LeakGuard()
    sys_fp: set[int] = set()
    send_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=CFG.WS_SEND_BUFFER)
    loop = asyncio.get_running_loop()
    last_active = loop.time()

    async def sender():
        nonlocal last_active
        try:
            while True:
                try:
                    item = await asyncio.wait_for(send_queue.get(), timeout=CFG.WS_HEARTBEAT_SEC)
                except asyncio.TimeoutError:
                    await _ws_send_safe(ws, {"type": MSG_PING})
                    continue
                await _ws_send_safe(ws, item)
                last_active = loop.time()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WS sender loop error | %s", _safe_str(e))

    async def guard_send(data: dict):
        try:
            send_queue.put_nowait(data)
        except asyncio.QueueFull:
            # ring-buffer: oldest drop
            with suppress(Exception):
                _ = send_queue.get_nowait()
            with suppress(Exception):
                send_queue.put_nowait(data)
            logger.warning("WS send queue overflow; dropped oldest | keys=%s", list(data.keys()))

    send_task = asyncio.create_task(sender())

    # ── 연결 직후 쿼리스트링으로 세션 자동 오픈
    async def _bootstrap_open_if_possible():
        nonlocal token, session_id, sys_fp
        qp = ws.query_params or {}
        q_user_id = qp.get("user_id")
        q_token = qp.get("access_token") or qp.get("token")
        if not q_user_id and not q_token:
            return
        try:
            uid = _ensure_uuid(q_user_id) if q_user_id else None
            if q_token and not uid:
                with suppress(Exception):
                    uid = _ensure_uuid(decode_access_token(q_token).user_id)

            # user 존재 검증 (없으면 None 처리)
            try:
                from app.models.user import User  # 지연 import로 순환참조 방지
            except Exception:
                User = None  # type: ignore

            if uid and User:
                with session_scope() as db:
                    user_obj = db.get(User, uid)
                    if not user_obj:
                        logger.warning("bootstrap: user not found, downgrade to anonymous | user_id=%s", uid)
                        uid = None

            # 세션 생성 (FK 제약 대비)
            def create_session_with_uid(db, uid_val):
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

            with session_scope() as db:
                try:
                    session = create_session_with_uid(db, uid)
                except IntegrityError as ie:
                    logger.warning("bootstrap commit FK failed; retrying as anonymous | %s", _safe_str(ie))
                    db.rollback()
                    session = create_session_with_uid(db, None)

                session_id = session.session_id

            system_prompt = get_system_prompt()
            sys_fp = leak_guard.fingerprint(system_prompt)
            await _ws_send_safe(ws, EmotionOpenResponse(
                type="open_ok",
                session_id=session_id,
                turns=0,
            ).model_dump())
            logger.info("WS bootstrap open_ok sent | session_id=%s", session_id)
        except Exception:
            logger.exception("WS bootstrap open failed")

    await _bootstrap_open_if_possible()

    try:
        while True:
            now = loop.time()
            if now - last_active > CFG.WS_IDLE_TIMEOUT:
                await guard_send({"type": "timeout"})
                break

            msg = await _ws_recv_safe(ws, timeout=CFG.WS_IDLE_TIMEOUT)
            if msg is None:
                continue

            typ = msg.get("type")

            if typ == MSG_PING:
                await guard_send({"type": MSG_PONG})
                continue

            # ── 세션 열기
            if typ == MSG_OPEN:
                # 이미 열린 경우 재응답
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

                token = payload.access_token
                uid = None
                with suppress(Exception):
                    uid = decode_access_token(token).user_id if token else None
                uid = _ensure_uuid(uid)

                # DB: 세션 생성 (FK 제약 대비)
                with session_scope() as db:
                    def create_session_with_uid(db, uid_val):
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

                    try:
                        session = create_session_with_uid(db, uid)
                    except IntegrityError as ie:
                        logger.warning("open commit FK failed; retrying anonymous | %s", _safe_str(ie))
                        db.rollback()
                        session = create_session_with_uid(db, None)

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
                logger.info("WS recv user | %s", _mask_preview(user_text, 100))

                # DB: 최근 스텝들
                with session_scope() as db:
                    steps = list(
                        db.exec(
                            select(EmotionStep)
                            .where(EmotionStep.session_id == session_id)
                            .order_by(EmotionStep.created_at.asc())
                        )
                    )

                    # 회차 제한
                    if _turn_count(steps) >= CFG.SESSION_MAX_TURNS:
                        await guard_send({"type": "limit", "message": "max turns reached"})
                        continue

                # 정책 판단
                want_activity = is_activity_turn(user_text, steps)
                want_close = is_closing_turn(user_text, steps)

                # 프롬프트
                system_prompt = get_system_prompt()
                task_prompt = get_task_prompt() if want_activity else None

                # 스트리밍 호출 + 누적 버퍼
                assistant_chunks: List[str] = []

                async def _gen():
                    try:
                        idx = 0
                        async for piece in _iter_chunks(
                            stream_noa_response(
                                system_prompt=system_prompt,
                                task_prompt=task_prompt,
                                conversation=_steps_to_conversation(steps) + [("user", user_text)],
                                temperature=0.7,
                                max_tokens=800,
                            )
                        ):
                            idx += 1
                            safe_piece = leak_guard.sanitize_out(piece, sys_fp)
                            if not safe_piece:
                                continue
                            assistant_chunks.append(safe_piece)
                            logger.debug("WS delta | %s", _mask_preview(safe_piece))
                            yield safe_piece
                    except Exception as e:
                        logger.warning("stream err | %s", _safe_str(e))
                        raise

                # 전송
                await guard_send(EmotionMessageResponse(type="message_start").model_dump())
                try:
                    async for piece in _gen():
                        await guard_send(EmotionMessageResponse(type="message_delta", delta=piece).model_dump())
                except Exception:
                    await guard_send({"type": "error", "message": "stream failed"})
                finally:
                    await guard_send(EmotionMessageResponse(type="message_end").model_dump())

                # DB: 스텝 기록
                assistant_text = "".join(assistant_chunks)
                with session_scope() as db:
                    existing = list(
                        db.exec(
                            select(EmotionStep.step_order)
                            .where(EmotionStep.session_id == session_id)
                            .order_by(EmotionStep.step_order.asc())
                        )
                    )
                    last_order = existing[-1] if existing else 0
                    user_order = last_order + 1
                    assistant_order = user_order + 1

                    step_user = EmotionStep(
                        session_id=session_id,
                        step_order=user_order,
                        step_type="user",
                        user_input=user_text,
                        gpt_response=None,
                        created_at=datetime.utcnow(),
                        insight_tag=None,
                    )
                    step_assistant = EmotionStep(
                        session_id=session_id,
                        step_order=assistant_order,
                        step_type="assistant",
                        user_input=None,
                        gpt_response=assistant_text,
                        created_at=datetime.utcnow(),
                        insight_tag=None,
                    )
                    db.add(step_user)
                    db.add(step_assistant)
                    db.commit()

                if want_activity:
                    mark_activity_injected(session_id)
                if want_close:
                    await guard_send({"type": "suggest_close"})

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

                with session_scope() as db:
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

                with session_scope() as db:
                    steps = list(
                        db.exec(
                            select(EmotionStep)
                            .where(EmotionStep.session_id == session_id)
                            .order_by(EmotionStep.created_at.asc())
                        )
                    )
                    session = db.get(EmotionSession, session_id)

                try:
                    recs = await recommend_tasks_from_session_core(
                        steps=steps,
                        session=session,
                        max_items=payload.max_items or 5,
                    )
                except Exception as e:
                    await guard_send({"type": "error", "message": f"recommend failed: {e}"})
                    continue

                await guard_send(TaskRecommendResponse(
                    type="task_recommend_ok",
                    items=recs,
                ).model_dump())

            else:
                await guard_send({"type": "error", "message": f"unknown type: {typ}"})

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WS fatal error")
        with suppress(Exception):
            await _ws_send_safe(ws, {"type": "error", "message": "fatal"})
    finally:
        with suppress(Exception):
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
        with suppress(Exception):
            await ws.close()
