# app/routers/emotion_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
from uuid import UUID
from datetime import datetime
from app.db.session import get_session
from app.models.emotion import EmotionSession, EmotionStep
from app.services.llm_service import stream_noa_response

# 과제 추천 코어 (동기 OpenAI → to_thread로 오프로드)
from app.services.task_recommend import recommend_tasks_from_session_core
from app.models.task import Task  # (타입 힌트/포맷용)
import asyncio
import traceback  # 로그 강화를 위한 추가

ws_router = APIRouter()


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
        desc = f"\n   - {t.description}" if t.description else ""
        lines.append(f"{i}. {t.title}{desc}")
    lines.append("\n작게 시작하고, 완료하면 체크해줘. ✅")
    return "\n".join(lines)


@ws_router.websocket("/ws/emotion")
async def emotion_chat(websocket: WebSocket):
    await websocket.accept()
    qp = websocket.query_params
    try:
        user_id = UUID(qp["user_id"])
    except Exception:
        await websocket.close(code=4001)
        return

    session_id_param = qp.get("session_id")
    print(f"🔗 WebSocket 연결: user_id={user_id}, session_id={session_id_param}")

    with next(get_session()) as db:
        # 세션 확보/생성
        sess = db.get(EmotionSession, UUID(session_id_param)) if session_id_param else None
        if sess is None:
            print("🆕 세션 새로 생성")
            sess = EmotionSession(user_id=user_id, started_at=datetime.utcnow())
            db.add(sess)
            db.commit()
            db.refresh(sess)

        # 프런트에 세션ID 최초 1회 알림
        await websocket.send_json({"session_id": str(sess.session_id)})

        while True:
            try:
                req = await websocket.receive_json()
                print("📩 받은 요청:", req)

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
                print(f"📜 최근 스텝 수: {len(recent)}")

                # 1) GPT 스트리밍 응답 전송
                collected_tokens = []
                for token in stream_noa_response(
                    user_input=user_input,
                    session=sess,
                    recent_steps=recent,
                    system_prompt=system_prompt,
                ):
                    if token:
                        collected_tokens.append(token)
                        await websocket.send_json({"token": token})  # 토큰 스트림

                # 2) 2초 지연 후 과제 추천(조건 충족 시)
                tasks_text = ""
                try:
                    if _should_recommend_tasks(user_input, sess):
                        await asyncio.sleep(2)  # ✅ 자연스러운 템포를 위한 지연
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
                    print("⚠️ 과제 추천 중 오류:", traceback.format_exc())

                # 3) 최종 텍스트(과제 포함) 저장
                full_text = "".join(collected_tokens)
                print("✅ 응답 완료, 저장 시작")

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

                # 4) 완료 신호
                await websocket.send_json({
                    "done": True,
                    "step_id": str(new_step.step_id),
                    "created_at": new_step.created_at.isoformat(),
                })
                print("📤 전송 완료")

            except WebSocketDisconnect:
                print("❌ 클라이언트 연결 끊김")
                break
            except Exception:
                print("🚨 예외 발생:", traceback.format_exc())
                await websocket.send_json({
                    "error": "internal_error",
                    "traceback": traceback.format_exc(),
                })
