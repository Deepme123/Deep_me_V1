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
## 📌 REST API

### ✅ 사용자 인증 및 계정
- **POST /register**  
  사용자 회원가입 (이름, 이메일, 비밀번호 입력)

- **POST /login**  
  로그인 후 액세스 토큰 반환

- **POST /change-password**  
  현재 비밀번호 확인 후 새 비밀번호로 변경

---

### ✅ 감정 세션 및 단계 관리

- **POST /emotion/sessions**  
  새로운 감정 세션을 생성  
  (입력값: user_id, emotion_label, topic 등)

- **GET /emotion/sessions/{session_id}**  
  특정 감정 세션 상세 조회

- **GET /emotion/sessions/user/{user_id}**  
  특정 사용자(user_id)의 전체 감정 세션 리스트 반환

- **POST /emotion/steps/generate**  
  GPT 기반 감정 응답 생성 (REST 테스트용)  
  입력값: session_id, user_input, step_order 등

- **GET /emotion/steps/{session_id}**  
  특정 감정 세션의 전체 대화 단계 조회

---

## 🔄 WebSocket API

### 🧠 감정 대화 스트리밍
- **WS /ws/emotion**  
  실시간 감정 대화 스트리밍 전용 WebSocket  
  - **쿼리파라미터**:
    - `user_id`: 사용자 UUID (필수)
    - `session_id`: 기존 세션 ID (선택)
  - **클라이언트 → 서버 메시지 예시**
    ```json
    {
      "user_input": "요즘 기분이 별로야.",
      "step_type": "normal",                // 선택값
      "system_prompt": "..."               // 선택값
    }
    ```
  - **서버 → 클라이언트 응답 흐름**
    1. `{ "session_id": "..." }` 최초 세션 정보
    2. `{ "token": "..." }` 스트리밍 응답 토큰 단위
    3. `{ "done": true, "step_id": "...", "created_at": "..." }` 완료 응답

---

## 🚧 예정된 추가 API (예시)
- POST /emotion/sessions/{session_id}/end  
  세션 종료 시각 업데이트

- GET /health  
  헬스체크 API (배포 확인용)
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

