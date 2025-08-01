# DEEPME Emotion Chat Backend

감정 탐색 및 멘탈 헬스케어 기능을 지원하는 WebSocket 기반 백엔드 API입니다.  
FastAPI, SQLModel, PostgreSQL, OpenAI API를 활용하여 사용자와의 감정 대화를 스트리밍 방식으로 제공합니다.

---

## 📦 프로젝트 구조

```bash
deepme_backend/
├── app/
│   ├── core/               # 시스템 프롬프트 로더 등 핵심 로직
│   ├── db/                 # 데이터베이스 세션 및 초기화
│   ├── models/             # EmotionSession, EmotionStep 모델
│   ├── services/           # LLM 응답 생성 및 스트리밍 로직
│   ├── routes/             # WebSocket 및 API 라우터
│   └── main.py             # FastAPI 애플리케이션 진입점
├── requirements.txt
└── README.md               # 이 문서
```

---

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
uvicorn app.main:app --reload
```

### 3. Swagger 문서 접근

[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧠 시스템 프롬프트

- 기본 프롬프트는 `app/core/prompt_loader.py`를 통해 로딩
- 환경에 따라 외부 텍스트 파일이나 DB에서 로드 가능

---

## 💬 감정 대화 WebSocket API

- 엔드포인트: `ws://localhost:8000/ws/emotion?user_id=<uuid>&session_id=<uuid(optional)>`
- 메시지 형식 (클라이언트 → 서버):

```json
{
  "user_input": "요즘 기분이 별로야.",
  "step_type": "normal",
  "system_prompt": "너는 감정 전문 상담사야"
}
```

- 서버 → 클라이언트 응답:

```json
{ "token": "..." }            # 토큰 스트리밍
{ "done": true, "step_id": "...", "created_at": "..." }  # 완료 응답
```

---

## ✅ 현재 구현된 기능

| 기능 항목 | 설명 |
|-----------|------|
| WebSocket 연결 | 사용자 `UUID` 기반 세션 식별 및 자동 생성 |
| 감정 세션 관리 | EmotionSession 생성 및 연결된 Step 기록 |
| GPT 응답 생성 | `generate_noa_response`, `stream_noa_response`로 스트리밍 처리 |
| 응답 저장 | 사용자 입력 및 GPT 응답을 DB에 저장 |
| 시스템 프롬프트 적용 | 사용자 지정 or 기본 프롬프트 사용 가능 |
| 예외 처리 | WebSocketDisconnect 및 예외 응답 처리 지원 |

---

## 🛠 향후 작업 예정

- [ ] WebSocket Ping/Pong 기반 Keep-Alive
- [ ] GPT 토큰 수 제한 로직 개선
- [ ] 사용자별 프롬프트 설정 관리 기능
- [ ] 대화 흐름 시각화용 API
- [ ] 감정 분류 및 요약 카드 연동

---

## 👤 개발자

- 정 도균 (프로젝트 기획 및 백엔드 개발)

---

## 📝 라이선스

본 프로젝트는 MIT 라이선스를 따릅니다.

