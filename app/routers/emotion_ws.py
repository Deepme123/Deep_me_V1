# app/routers/emotion_ws.py
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
from app.models.task import Task
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
WS_IDLE_TIMEOUT_SECS = int(os.getenv("WS_IDLE_TIMEOUT_SECS", "180"))
AUTO_END_AFTER_ACTIVITY = os.getenv("AUTO_END_AFTER_ACTIVITY", "0") == "1"
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "8"))
CLOSE_TOKENS = {"그만", "끝", "종료", "bye", "quit", "exit"}

ws_router = APIRouter()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (기존 + 추가)
def _should_recommend_tasks(user_text: str, sess: EmotionSession) -> bool:
    if not user_text:
        return False
    kw = ("과제", "활동", "뭐 해볼까", "추천해줘", "해볼만한", "미션")
    return any(k in user_text for k in kw)

def _format_tasks_as_chat(tasks: list[Task]) -> str:
    lines = ["", "📝 활동과제 추천"]
    for i, t in enumerate(tasks, 1):
        desc = f"\n   - {getattr(t, 'description', '')}" if getattr(t, "description", None) else ""
        lines.append(f"{i}. {t.title}{desc}")
    lines.append("\n작게 시작하고, 완료하면 체크해줘. ✅")
    return "\n".join(lines)

async def _agen(gen):
    if inspect.isasyncgen(gen):
        async for x in gen:
            yield x
    else:
        for x in gen:
            yield x

# ──────────────────────────────────────────────────────────────────────────────
# Leak guard helpers (추가)
_LEAK_MARKERS = [
    r"<<SYS>>",
    r"BEGIN SYSTEM PROMPT",
    r"\[SYSTEM\]",
    r"DO NOT DISCLOSE",
    r"internal rule",
    r"developer prompt",
]

def _fingerprint(text: str, n: int = 10) -> set[int]:
    if not text:
        return set()
    step = max(3, n // 2)
    return {hash(text[i:i+n]) for i in range(0, max(0, len(text) - n + 1), step)}

def _might_leak(text: str, sys_fp: set[int], n: int = 10) -> bool:
    if not text or not sys_fp:
        return False
    step = max(3, n // 2)
    fp = {hash(text[i:i+n]) for i in range(0, max(0, len(text) - n + 1), step)}
    return len(sys_fp & fp) >= 2  # 두 조각 이상 매칭 시 누설로 간주

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
        return "[redacted]"
    return _redact(piece)

def _mask_preview(s: str, k: int = 80) -> str:
    s = s.replace("\n", " ")
    return (s[:k] + "…") if len(s) > k else s

# ──────────────────────────────────────────────────────────────────────────────
# WebSocket
@ws_router.websocket("/ws/emotion")
async def emotion_chat(websocket: WebSocket):
    logger.info(
        "WS handshake(app-mode): auth_header=%s url=%s",
        bool(websocket.headers.get("authorization")),
        websocket.url,
    )
    await websocket.accept()

    # 인증
    token = None
    auth = websocket.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        await websocket.send_json({"error": "unauthorized", "reason": "missing_authorization_header"})
        await websocket.close(code=4401)
        return
    try:
        payload = decode_access_token(token)
        user_id = UUID(payload["sub"])
    except Exception:
        await websocket.send_json({"error": "invalid_token"})
        await websocket.close(code=4401)
        return

    # user_id 일치 검사
    qp = websocket.query_params
    qp_user_id = qp.get("user_id")
    if qp_user_id and qp_user_id != str(user_id):
        await websocket.send_json({"error": "forbidden_user"})
        await websocket.close(code=4403)
        return

    session_id_param = qp.get("session_id")
    logger.info("WS connected user_id=%s session_id=%s", user_id, session_id_param)

    with next(get_session()) as db:
        # 세션 확보/생성
        sess: EmotionSession | None = None
        if session_id_param:
            with suppress(Exception):
                sess = db.get(EmotionSession, UUID(session_id_param))
        if sess is None:
            logger.info("Create new emotion session for user=%s", user_id)
            sess = EmotionSession(user_id=user_id, started_at=datetime.utcnow())
            db.add(sess)
            db.commit()
            db.refresh(sess)

        await websocket.send_json({"session_id": str(sess.session_id)})

        while True:
            try:
                # 입력 + 타임아웃
                try:
                    req = await asyncio.wait_for(websocket.receive_json(), timeout=WS_IDLE_TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    sess.ended_at = datetime.utcnow()
                    db.add(sess)
                    db.commit()
                    await websocket.send_json({"info": "idle_timeout", "session_closed": True})
                    await websocket.close(code=1001)
                    break

                user_input = (req.get("user_input") or "").strip()
                user_input_norm = user_input.lower()
                if req.get("close") is True or user_input_norm in CLOSE_TOKENS:
                    sess.ended_at = datetime.utcnow()
                    db.add(sess)
                    db.commit()
                    await websocket.send_json({"info": "client_close", "session_closed": True})
                    await websocket.close(code=1000)
                    break
                if not user_input:
                    await websocket.send_json({"error": "empty_input"})
                    continue

                step_type = req.get("step_type", "normal")

                # 이번 턴 성격 결정
                activity_turn = is_activity_turn(sess.session_id, db)
                closing_turn = is_closing_turn(sess.session_id, db)

                # 시스템 프롬프트 조립
                system_prompt = get_system_prompt()
                if activity_turn:
                    system_prompt = f"{system_prompt}\n\n{get_task_prompt()}"
                if closing_turn:
                    system_prompt = f"""{system_prompt}

[대화 마무리 지침](최우선)
- 아래 지침은 다른 모든 규칙보다 우선한다.
- 질문 금지. 요청하지 않은 과제 제안 금지. 이 메시지로 대화 종료.
- 핵심 요약 2줄
- 오늘 배운 1가지 강조
- 간단한 끝인사 1줄
"""

                # 이 턴의 시스템 프롬프트 핑거프린트 생성
                sys_fp = _fingerprint(system_prompt, n=10)

                # 최근 스텝
                recent_all = db.exec(
                    select(EmotionStep)
                    .where(EmotionStep.session_id == sess.session_id)
                    .order_by(EmotionStep.step_order)
                ).all()
                recent = recent_all[-HISTORY_TURNS:] if HISTORY_TURNS > 0 else recent_all

                # === LLM 응답 처리 구간 ===
                collected_tokens: list[str] = []
                first_token = False

                try:
                    async for token_piece in _agen(
                        stream_noa_response(
                            user_input=user_input,
                            session=sess,
                            recent_steps=recent,
                            system_prompt=system_prompt,
                        )
                    ):
                        if not token_piece:
                            continue
                        # 출력 직전 정제
                        safe_piece = _sanitize_out(token_piece, sys_fp)
                        if not safe_piece:
                            continue

                        # 로깅은 마스킹 프리뷰로
                        logger.debug("WS send token preview: %r", _mask_preview(safe_piece))

                        # 프론트에는 텍스트만 전달
                        await websocket.send_json({"token": safe_piece})
                        collected_tokens.append(safe_piece)
                        first_token = True

                except RuntimeError as e:
                    # 사람 친화적 에러 매핑 (세션 유지)
                    msg = str(e)
                    if msg == "blocked_by_content_filter":
                        with suppress(Exception):
                            await websocket.send_json(
                                {"error": "안전필터에 의해 답변이 차단됐어. 질문을 조금 더 중립적으로 바꿔볼래?"}
                            )
                    elif msg == "empty_completion_from_llm":
                        with suppress(Exception):
                            await websocket.send_json(
                                {"error": "모델이 답변을 비워 보냈어. 잠시 후 다시 시도하거나 질문을 조금 더 구체화해줘."}
                            )
                    else:
                        with suppress(Exception):
                            await websocket.send_json({"error": f"LLM 처리 중 오류가 발생했어: {msg}"})
                    continue  # 세션은 닫지 않고 다음 입력 대기

                except Exception:
                    logger.exception("LLM stream failed")
                    with suppress(Exception):
                        await websocket.send_json({"error": "LLM 처리 중 알 수 없는 오류가 발생했어."})
                    continue  # 세션 유지

                if not first_token:
                    with suppress(Exception):
                        await websocket.send_json({"error": "LLM 응답이 비었습니다. 잠시 후 다시 시도해줘."})
                    continue

                # 과제 추천
                try:
                    if not closing_turn and _should_recommend_tasks(user_input, sess):
                        # 짧은 지연으로 토큰 UX 구분
                        await asyncio.sleep(2)
                        tasks = await asyncio.to_thread(
                            recommend_tasks_from_session_core,
                            user_id=user_id,
                            session_id=sess.session_id,
                            n=3,
                            recent_steps_limit=10,
                            max_history_chars=1000,
                        )
                        tasks_text = _format_tasks_as_chat(tasks)
                        # 안전상 동일 정제 적용(옵션)
                        tasks_text = _sanitize_out(tasks_text, sys_fp)
                        await websocket.send_json({"token": tasks_text})
                        collected_tokens.append(tasks_text)
                except Exception:
                    logger.exception("task recommendation failed")

                # 저장
                full_text = "".join(collected_tokens)
                new_step = EmotionStep(
                    session_id=sess.session_id,
                    step_order=(recent_all[-1].step_order + 1) if recent_all else 1,
                    step_type=step_type,
                    user_input=user_input,
                    gpt_response=full_text,
                    created_at=datetime.utcnow(),
                )
                db.add(new_step)
                db.commit()
                db.refresh(new_step)

                # 완료 신호
                await websocket.send_json(
                    {
                        "done": True,
                        "step_id": str(new_step.step_id),
                        "created_at": new_step.created_at.isoformat(),
                    }
                )

                # 활동과제 마킹 및 종료 여부
                if activity_turn:
                    mark_activity_injected(sess.session_id, db)
                    if AUTO_END_AFTER_ACTIVITY:
                        sess.ended_at = datetime.utcnow()
                        db.add(sess)
                        db.commit()
                        await websocket.send_json({"info": "activity_reached", "session_closed": True})
                        await websocket.close(code=1000)
                        break

                # 종료 판단
                turns = _turn_count(db, sess.session_id)
                if closing_turn:
                    sess.ended_at = datetime.utcnow()
                    db.add(sess)
                    db.commit()
                    await websocket.send_json({"info": "session_closed", "turns": turns})
                    await websocket.close(code=1000)
                    break
                if int(turns) > SESSION_MAX_TURNS:
                    logger.warning("turn overflow: %s", turns)
                if int(turns) >= SESSION_MAX_TURNS and not closing_turn:
                    sess.ended_at = datetime.utcnow()
                    db.add(sess)
                    db.commit()
                    await websocket.send_json({"info": "max_turns_guard_close", "session_closed": True})
                    await websocket.close(code=1000)
                    break

            except WebSocketDisconnect:
                logger.info("WS disconnected user_id=%s", user_id)
                break
            except Exception:
                logger.exception("WS error")
                with suppress(Exception):
                    await websocket.send_json({"error": "internal_error"})
                break
