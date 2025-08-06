# app/routers/emotion_ws.py
# app/routers/emotion_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
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
from app.core.jwt import decode_access_token  # JWT 디코드 함수

ws_router = APIRouter()
logger = logging.getLogger(__name__)

# 쿼리스트링으로 토큰 허용(개발/임시용). 운영에선 false 권장
WS_ALLOW_QUERY_TOKEN = os.getenv("WS_ALLOW_QUERY_TOKEN", "false").lower() == "true"


def _should_recommend_tasks(user_text: str, sess: EmotionSession) -> bool:
    """간단 트리거: 키워드 기반 과제 추천 조건."""
    if not user_text:
        return False
    kw = ("과제", "활동", "뭐 해볼까", "추천해줘", "해볼만한", "미션")
    return any(k in user_text for k in kw)


def _format_tasks_as_chat(tasks: list[Task]) -> str:
    """과제 추천을 자연스러운 챗 텍스트로 변환."""
    lines = ["", "📝 활동과제 추천"]
    for i, t in enumerate(tasks, 1):
        desc = f"\n   - {getattr(t, 'description', '')}" if getattr(t, "description", None) else ""
        lines.append(f"{i}. {t.title}{desc}")
    lines.append("\n작게 시작하고, 완료하면 체크해줘. ✅")
    return "\n".join(lines)


async def _agen(gen):
    """sync/async 제너레이터 모두 호환."""
    if inspect.isasyncgen(gen):
        async for x in gen:
            yield x
    else:
        for x in gen:
            yield x


def _extract_token(websocket: WebSocket) -> tuple[str | None, str]:
    """헤더→쿠키→쿼리(옵션) 순으로 토큰 추출."""
    # 1) Authorization 헤더
    auth = websocket.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip(), "header"

    # 2) 쿠키
    cookie_token = websocket.cookies.get("access_token")
    if cookie_token:
        return cookie_token, "cookie"

    # 3) 쿼리(옵션)
    if WS_ALLOW_QUERY_TOKEN:
        q_token = websocket.query_params.get("token")
        if q_token:
            return q_token, "query"

    return None, "none"


@ws_router.websocket("/ws/emotion")
async def emotion_chat(websocket: WebSocket):
    # ── 핸드셰이크 로그(인증 전달 여부만) ──
    has_auth_header = bool(websocket.headers.get("authorization"))
    has_cookie = bool(websocket.cookies.get("access_token"))
    has_q_token = bool(websocket.query_params.get("token"))
    logger.info(
        "WS handshake: auth_header=%s cookie=%s q_token=%s url=%s",
        has_auth_header, has_cookie, has_q_token, websocket.url
    )

    # 디버깅 가시성 위해 우선 업그레이드
    await websocket.accept()

    # 인증 토큰 추출
    token, source = _extract_token(websocket)
    if not token:
        await websocket.send_json({"error": "unauthorized", "reason": "no_token"})
        await websocket.close(code=4401)
        return

    # 토큰 검증
    try:
        payload = decode_access_token(token)
        user_id = UUID(payload["sub"])  # 신뢰 원천은 토큰의 sub
    except Exception:
        await websocket.send_json({"error": "invalid_token"})
        await websocket.close(code=4401)
        return

    # (선택) 쿼리 user_id가 있으면 토큰 sub와 일치 확인
    qp = websocket.query_params
    qp_user_id = qp.get("user_id")
    if qp_user_id and qp_user_id != str(user_id):
        await websocket.send_json({"error": "forbidden_user"})
        await websocket.close(code=4403)
        return

    session_id_param = qp.get("session_id")
    logger.info("WS connected user_id=%s via=%s session_id=%s", user_id, source, session_id_param)

    # ── DB 세션 컨텍스트 ──
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

        # 프런트에 세션ID 최초 1회 알림
        await websocket.send_json({"session_id": str(sess.session_id)})

        while True:
            try:
                req = await websocket.receive_json()
                logger.debug("WS request received (keys=%s)", list(req.keys()))

                user_input = (req.get("user_input") or "").strip()
                if not user_input:
                    await websocket.send_json({"error": "empty_input"})
                    continue

                step_type = req.get("step_type", "normal")
                system_prompt = req.get("system_prompt")

                # 최근 스텝 조회 (시간순 정렬)
                recent = db.exec(
                    select(EmotionStep)
                    .where(EmotionStep.session_id == sess.session_id)
                    .order_by(EmotionStep.step_order)
                ).all()
                logger.debug("Recent steps: %d", len(recent))

                # ── 1) LLM 스트리밍 ──
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

                # ── 2) 과제 추천(조건 충족 시) ──
                try:
                    if _should_recommend_tasks(user_input, sess):
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

                # ── 3) 저장 ──
                full_text = "".join(collected_tokens)
                new_step = EmotionStep(
                    session_id=sess.session_id,
                    step_order=len(recent) + 1,
                    step_type=step_type,
                    user_input=user_input,
                    gpt_response=full_text,
                    created_at=datetime.utcnow(),
                )
                db.add(new_step)
                db.commit()
                db.refresh(new_step)

                # ── 4) 완료 신호 ──
                await websocket.send_json({
                    "done": True,
                    "step_id": str(new_step.step_id),
                    "created_at": new_step.created_at.isoformat(),
                })

            except WebSocketDisconnect:
                logger.info("WS disconnected user_id=%s", user_id)
                break
            except Exception:
                logger.exception("WS error")
                await websocket.send_json({"error": "internal_error"})
