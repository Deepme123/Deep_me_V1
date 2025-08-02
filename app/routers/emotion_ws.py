from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
from uuid import UUID
from datetime import datetime
from app.db.session import get_session
from app.models.emotion import EmotionSession, EmotionStep
from app.services.llm_service import stream_noa_response  # 반드시 async generator 형태일 것

import traceback

ws_router = APIRouter()


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
        sess = db.get(EmotionSession, UUID(session_id_param)) if session_id_param else None
        if sess is None:
            print("🆕 세션 새로 생성")
            sess = EmotionSession(user_id=user_id, started_at=datetime.utcnow())
            db.add(sess)
            db.commit()
            db.refresh(sess)

        # 클라이언트에 세션 ID 전송
        await websocket.send_json({"session_id": str(sess.session_id)})

        while True:
            try:
                req = await websocket.receive_json()
                print("📩 받은 요청:", req)

                user_input = req.get("user_input", "").strip()
                if not user_input:
                    await websocket.send_json({"error": "empty_input"})
                    continue

                step_type = req.get("step_type", "normal")
                system_prompt = req.get("system_prompt")

                recent = db.exec(
                    select(EmotionStep)
                    .where(EmotionStep.session_id == sess.session_id)
                    .order_by(EmotionStep.step_order)
                ).all()
                print(f"📜 최근 스텝 수: {len(recent)}")

                # GPT 스트리밍 응답 전송
                collected_tokens = []
                print("🤖 GPT 응답 스트리밍 시작")

                async for token in stream_noa_response(
                    user_input=user_input,
                    session=sess,
                    recent_steps=recent,
                    system_prompt=system_prompt,
                ):
                    collected_tokens.append(token)
                    await websocket.send_json({"token": token})  # 실시간 토큰 전송
                    print("📤 토큰 전송:", token)

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

                await websocket.send_json({
                    "done": True,
                    "step_id": str(new_step.step_id),
                    "created_at": new_step.created_at.isoformat(),
                })
                print("📤 저장 완료, 완료 신호 전송")

            except WebSocketDisconnect:
                print("❌ 클라이언트 연결 끊김")
                break

            except Exception as e:
                err_msg = traceback.format_exc()
                print("🚨 예외 발생:", err_msg)
                await websocket.send_json({
                    "error": str(e),
                    "traceback": err_msg
                })
