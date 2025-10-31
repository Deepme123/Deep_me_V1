# app/routers/emotion_ws.py  (완성본)
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
from contextlib import suppress

from app.db.session import get_session
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

# ──────────────────────────────────────────────────────────────────────────────
# 설정
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "20"))
WS_IDLE_TIMEOUT = float(os.getenv("WS_IDLE_TIMEOUT", "120"))
WS_SEND_BUFFER = int(os.getenv("WS_SEND_BUFFER", "20"))
WS_HEARTBEAT_SEC = float(os.getenv("WS_HEARTBEAT_SEC", "15"))

logger = logging.getLogger(__name__)
router = APIRouter()

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
        logger.warning("WS send failed: %s", e)

async def _ws_recv_safe(ws: WebSocket, *, timeout: float | None = None) -> dict | None:
    try:
        if timeout:
            return await asyncio.wait_for(ws.receive_json(), timeout=timeout)
        return await ws.receive_json()
    except asyncio.TimeoutError:
        return {"type": "ping"}
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.warning("WS recv failed: %s", e)
        return None

def _ensure_uuid(x: str | UUID | None) -> UUID | None:
    if x is None:
        return None
    return UUID(str(x))

def _iter_chunks(gen):
    # generator or async-generator wrapper
    if inspect.isasyncgen(gen):
        async def _ait():
            async for x in gen:
                yield x
        return _ait()
    return gen

async def _async_yield(gen, *, flush_each: bool = False):
    if inspect.isasyncgen(gen):
        async for x in gen:
            yield x
    else:
        for x in gen:
            yield x

def _steps_to_conversation(steps: list[EmotionStep]) -> list[tuple[str, str]]:
    """DB steps를 ('user'|'assistant', text)로 변환."""
    conv: list[tuple[str, str]] = []
    for s in steps:
        if s.step_type == "user" and s.user_input:
            conv.append(("user", s.user_input))
        elif s.step_type == "assistant" and s.gpt_response:
            conv.append(("assistant", s.gpt_response))
    return conv

# ──────────────────────────────────────────────────────────────────────────────
# Leak guard helpers (수정)
_LEAK_MARKERS = [
    r"<<SYS>>",
    r"\bBEGIN SYSTEM PROMPT\b",
    r"\[\s*SYSTEM\s*\]",
    r"\bDO NOT DISCLOSE\b",
    r"\bdeveloper prompt\b",
]
# 환경변수로 민감도/동작 방식 제어
LEAK_GUARD_NGRAM = int(os.getenv("LEAK_GUARD_NGRAM", "20"))        # 기본 20
LEAK_GUARD_MIN_MATCH = int(os.getenv("LEAK_GUARD_MIN_MATCH", "3")) # 기본 3
LEAK_GUARD_MODE = os.getenv("LEAK_GUARD_MODE", "mask")             # 'mask' | 'drop'

def _fingerprint(text: str, n: int | None = None) -> set[int]:
    if n is None:
        n = LEAK_GUARD_NGRAM
    if not text:
        return set()
    step = max(3, n // 2)
    return {hash(text[i:i+n]) for i in range(0, max(0, len(text) - n + 1), step)}

def _might_leak(text: str, sys_fp: set[int], n: int | None = None) -> bool:
    if n is None:
        n = LEAK_GUARD_NGRAM
    if not text or not sys_fp:
        return False
    step = max(3, n // 2)
    fp = {hash(text[i:i+n]) for i in range(0, max(0, len(text) - n + 1), step)}
    return len(sys_fp & fp) >= LEAK_GUARD_MIN_MATCH  # 민감도 환경변수로 제어

def _redact(text: str) -> str:
    out = text
    for pat in _LEAK_MARKERS:
        out = re.sub(pat, "[redacted]", out, flags=re.I)
    return out

def _sanitize_out(piece: str, sys_fp: set[int]) -> str:
    """출력 직전 필터. 누설 징후면 통째로 redacted, 아니면 마커만 치환."""
    if not isinstance(piece, str) or not piece:
        return ""
    if _might_leak(piece, sys_fp):
        # 전면 차단 대신 기본은 '부분 치환(mask)'
        if LEAK_GUARD_MODE == "drop":
            return ""
        return _redact(piece)
    return _redact(piece)

# ──────────────────────────────────────────────────────────────────────────────
# 라우터

@router.websocket("/ws/emotion")
async def ws_emotion(ws: WebSocket):
    await ws.accept()
    token: str | None = None
    session_id: UUID | None = None
    sys_fp: set[int] = set()
    send_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=WS_SEND_BUFFER)
    last_active = asyncio.get_event_loop().time()

    async def sender():
        nonlocal last_active
        try:
            while True:
                try:
                    item = await asyncio.wait_for(send_queue.get(), timeout=WS_HEARTBEAT_SEC)
                except asyncio.TimeoutError:
                    await _ws_send_safe(ws, {"type": "ping"})
                    continue
                await _ws_send_safe(ws, item)
                last_active = asyncio.get_event_loop().time()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WS sender loop error: %s", e)

    async def guard_send(data: dict):
        try:
            await send_queue.put(data)
        except asyncio.QueueFull:
            logger.warning("WS send queue full, dropping item: %s", list(data.keys()))

    send_task = asyncio.create_task(sender())

    try:
        while True:
            now = asyncio.get_event_loop().time()
            if now - last_active > WS_IDLE_TIMEOUT:
                await guard_send({"type": "timeout"})
                break

            msg = await _ws_recv_safe(ws, timeout=WS_IDLE_TIMEOUT)
            if msg is None:
                continue

            typ = msg.get("type")
            if typ == "ping":
                await guard_send({"type": "pong"})
                continue

            # ── 세션 열기
            if typ == "open":
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

                # DB: 세션 생성
                with get_session() as db:
                    session = EmotionSession(
                        user_id=uid,
                        started_at=datetime.utcnow(),
                        emotion_label=None,
                        topic=None,
                        trigger_summary=None,
                        insight_summary=None,
                    )
                    db.add(session)
                    db.commit()
                    db.refresh(session)
                    session_id = session.session_id

                # 시스템 프롬프트 로딩
                system_prompt = get_system_prompt()

                # 누설 방지용 시스템 프롬프트 핑거프린트
                sys_fp = _fingerprint(system_prompt)

                await guard_send(EmotionOpenResponse(
                    type="open_ok",
                    session_id=session_id,
                    turns=0,
                ).model_dump())

            # ── 사용자 메시지 처리
            elif typ == "message":
                if not session_id:
                    await guard_send({"type": "error", "message": "no session"})
                    continue

                try:
                    payload = EmotionMessageRequest(**msg)
                except Exception as e:
                    await guard_send({"type": "error", "message": f"bad message payload: {e}"})
                    continue

                user_text = payload.text or ""
                user_text_preview = _mask_preview(user_text, 100)
                logger.info("WS recv user: %s", user_text_preview)

                # DB: 최근 스텝들
                with get_session() as db:
                    steps = list(
                        db.exec(
                            select(EmotionStep)
                            .where(EmotionStep.session_id == session_id)
                            .order_by(EmotionStep.created_at.asc())
                        )
                    )

                    # 회차 제한
                    if _turn_count(steps) >= SESSION_MAX_TURNS:
                        await guard_send({"type": "limit", "message": "max turns reached"})
                        continue

                # 정책 판단
                want_activity = is_activity_turn(user_text, steps)
                want_close = is_closing_turn(user_text, steps)

                # 프롬프트 구성
                system_prompt = get_system_prompt()
                task_prompt = get_task_prompt() if want_activity else None

                # 스트리밍 호출 + 누적 버퍼
                assistant_chunks: list[str] = []

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
                            # 누설 방지 적용
                            safe_piece = _sanitize_out(piece, sys_fp)
                            if not safe_piece:
                                continue
                            logger.debug("WS send token preview: %s", _mask_preview(safe_piece))
                            assistant_chunks.append(safe_piece)
                            yield safe_piece
                    except Exception as e:
                        logger.warning("stream err: %s", e)
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
                with get_session() as db:
                    step_user = EmotionStep(
                        session_id=session_id,
                        step_order=len(steps) * 2 + 1,
                        step_type="user",
                        user_input=user_text,
                        gpt_response=None,
                        created_at=datetime.utcnow(),
                        insight_tag=None,
                    )
                    step_assistant = EmotionStep(
                        session_id=session_id,
                        step_order=len(steps) * 2 + 2,
                        step_type="assistant",
                        user_input=None,
                        gpt_response=assistant_text,
                        created_at=datetime.utcnow(),
                        insight_tag=None,
                    )
                    db.add(step_user)
                    db.add(step_assistant)
                    db.commit()

                # 액티비티 주입 되었는지 표시
                if want_activity:
                    mark_activity_injected(session_id)

                # 종료 권고 신호
                if want_close:
                    await guard_send({"type": "suggest_close"})

            # ── 세션 종료
            elif typ == "close":
                if not session_id:
                    await guard_send({"type": "error", "message": "no session"})
                    continue

                try:
                    payload = EmotionCloseRequest(**msg)
                except Exception as e:
                    await guard_send({"type": "error", "message": f"bad close payload: {e}"})
                    continue

                # 세션 마감 처리
                with get_session() as db:
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
            elif typ == "task_recommend":
                if not session_id:
                    await guard_send({"type": "error", "message": "no session"})
                    continue

                try:
                    payload = TaskRecommendRequest(**msg)
                except Exception as e:
                    await guard_send({"type": "error", "message": f"bad task payload: {e}"})
                    continue

                with get_session() as db:
                    steps = list(
                        db.exec(
                            select(EmotionStep)
                            .where(EmotionStep.session_id == session_id)
                            .order_by(EmotionStep.created_at.asc())
                        )
                    )
                    session = db.get(EmotionSession, session_id)

                # 추천 로직
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
    except Exception as e:
        logger.exception("WS fatal error: %s", e)
        with suppress(Exception):
            await _ws_send_safe(ws, {"type": "error", "message": "fatal"})
    finally:
        with suppress(Exception):
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
        with suppress(Exception):
            await ws.close()
