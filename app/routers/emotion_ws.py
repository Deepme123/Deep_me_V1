# app/routers/emotion_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import select
from uuid import UUID
from datetime import datetime
import asyncio
import logging
import inspect
import os

from app.db.session import get_session
from app.models.emotion import EmotionSession, EmotionStep
from app.models.task import Task  # (타입 힌트/포맷용)
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

# 종료·행동 설정
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "20"))
WS_IDLE_TIMEOUT_SECS = int(os.getenv("WS_IDLE_TIMEOUT_SECS", "180"))
AUTO_END_AFTER_ACTIVITY = os.getenv("AUTO_END_AFTER_ACTIVITY", "0") == "1"
CLOSE_TOKENS = {"그만", "끝", "종료", "bye", "quit", "exit"}

ws_router = APIRouter()
logger = logging.getLogger(__name__)


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

    # user_id 일치 검사(옵션)
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
        sess = None
        if session_id_param:
            try:
                sess = db.get(EmotionSession, UUID(session_id_param))
            except Exception:
                sess = None
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
                if req.get("close") is True or user_input in CLOSE_TOKENS:
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

                # 최근 스텝
                recent = db.exec(
                    select(EmotionStep)
                    .where(EmotionStep.session_id == sess.session_id)
                    .order_by(EmotionStep.step_order)
                ).all()

                # 스트리밍
                collected_tokens = []
                async for token_piece in _agen(
                    stream_noa_response(
                        user_input=user_input,
                        session=sess,
                        recent_steps=recent,
                        system_prompt=system_prompt,
                    )
                ):
                    if token_piece:
                        collected_tokens.append(token_piece)
                        await websocket.send_json({"token": token_piece})

                # 과제 추천(키워드) — 마무리 턴에는 비활성화
                try:
                    if not closing_turn and _should_recommend_tasks(user_input, sess):
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
                        await websocket.send_json({"token": tasks_text})
                        collected_tokens.append(tasks_text)
                except Exception:
                    logger.exception("task recommendation failed")

                # 저장
                full_text = "".join(collected_tokens)
                new_step = EmotionStep(
                    session_id=sess.session_id,
                    step_order=(recent[-1].step_order + 1) if recent else 1,
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

                # 활동과제 마킹 및 선택적 즉시 종료
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
                await websocket.send_json({"error": "internal_error"})
