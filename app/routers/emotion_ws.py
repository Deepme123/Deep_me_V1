# app/routers/emotion_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
from uuid import UUID
from datetime import datetime
import asyncio
import logging
import inspect

from app.db.session import get_session
from app.models.emotion import EmotionSession, EmotionStep
from app.models.task import Task  # (타입 힌트/포맷용)
from app.services.llm_service import stream_noa_response
from app.services.task_recommend import recommend_tasks_from_session_core
from app.core.jwt import decode_access_token  # JWT 디코드 함수 (프로젝트 구현에 맞게 import)

ws_router = APIRouter()
logger = logging.getLogger(__name__)


def _should_recommend_tasks(user_text: str, sess: EmotionSession) -> bool:
    """
    간단 트리거: 유저가 과제를 직접 요청하거나(키워드),
    필요 시 감정 레이블 조건을 추가할 수 있음.
    """
    if not user_text:
        return False
    kw = ("과제", "활동", "뭐 해볼까", "추천해줘", "해볼만한", "미션")
    if any(k in user_text for k in kw):
        return True
    # 예) 감정 레이블 기반 자동 제안:
    # if (sess.emotion_label or "").lower() in ("불안", "우울"):
    #     return True
    return False


def _format_tasks_as_chat(tasks: list[Task]) -> str:
    """과제 추천을 자연스러운 챗 형태 텍스트로 변환."""
    lines = ["", "📝 활동과제 추천"]
    for i, t in enumerate(tasks, 1):
        desc = f"\n   - {t.description}" if getattr(t, "description", None) else ""
        lines.append(f"{i}. {t.title}{desc}")
    lines.append("\n작게 시작하고, 완료하면 체크해줘. ✅")
    return "\n".join(lines)


async def _agen(gen):
    """
    stream_noa_response 가 sync generator 또는 async generator 둘 다 가능하도록 래핑.
    """
    if inspect.isasyncgen(gen):
        async for x in gen:
            yield x
    else:
        for x in gen:
            yield x


@ws_router.websocket("/ws/emotion")
async def emotion_chat(websocket: WebSocket):
    # ── 0) 핸드셰이크 직전 로깅(인증 전달 여부만) ──
    has_auth_header = bool(websocket.headers.get("authorization"))
    has_cookie = bool(websocket.cookies.get("access_token"))
    logger.info(
        "WS handshake: auth_header=%s cookie=%s url=%s",
        has_auth_header, has_cookie, websocket.url
    )

    # ── 1) 일단 업그레이드(accept) → 이후 인증 검증(디버깅 가시성 ↑) ──
    await websocket.accept()

    # ── 2) 인증: Authorization 헤더(또는 쿠키)에서 토큰 추출 및 검증 ──
    token = None
    auth = websocket.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        # 브라우저의 경우 쿠키 인증
        token = websocket.cookies.get("access_token")

    if not token:
        await websocket.send_json({"error": "unauthorized"})
        await websocket.close(code=4401)  # Unauthorized
        return

    try:
        payload = decode_access_token(token)  # 프로젝트의 JWT 디코더 사용
        user_id = UUID(payload["sub"])       # 신뢰 원천은 토큰의 sub
    except Exception:
        await websocket.send_json({"error": "invalid_token"})
        await websocket.close(code=4401)
        return

    # (선택) 쿼리스트링 user_id가 왔다면 토큰의 sub와 일치 여부만 체크
    qp = websocket.query_params
    qp_user_id = qp.get("user_id")
    if qp_user_id and qp_user_id != str(user_id):
        await websocket.send_json({"error": "forbidden_user"})
        await websocket.close(code=4403)  # Forbidden
        return

    session_id_param = qp.get("session_id")
    logger.info("WS connected user_id=%s session_id=%s", user_id, session_id_param)

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
                # 민감 데이터 직접 로깅 금지 → 키 정도만
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

                # ── 1) GPT 스트리밍 응답 전송 ──
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
                        await websocket.send_json({"token": token_piece})  # 토큰 스트림

                # ── 2) 2초 지연 후 과제 추천(조건 충족 시) ──
                try:
                    if _should_recommend_tasks(user_input, sess):
                        await asyncio.sleep(2)  # 자연스러운 템포 지연
                        tasks = await asyncio.to_thread(
                            recommend_tasks_from_session_core,
                            user_id=user_id,
                            session_id=sess.session_id,
                            n=3,
                            recent_steps_limit=10,
                            max_history_chars=1000,
                        )
                        tasks_text = _format_tasks_as_chat(tasks)
                        # 과제 추천도 같은 답변의 연장선처럼 추가 토큰으로 전송
                        await websocket.send_json({"token": tasks_text})
                        collected_tokens.append(tasks_text)
                except Exception:
                    # 외부에 스택트레이스 노출 금지
                    logger.exception("task recommendation failed")

                # ── 3) 최종 텍스트(과제 포함) 저장 ──
                full_text = "".join(collected_tokens)
                logger.debug("LLM stream done. Persist step.")

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
                logger.debug("WS send done signal: step_id=%s", new_step.step_id)

            except WebSocketDisconnect:
                logger.info("WS disconnected user_id=%s", user_id)
                break
            except Exception:
                # 내부에만 스택트레이스 기록, 외부는 일반화된 에러만
                logger.exception("WS error")
                await websocket.send_json({"error": "internal_error"})
