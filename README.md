# DEEPME Emotion Chat Backend

DEEPME 감정 대화 서비스를 위한 FastAPI + SQLModel 백엔드 (HTTP + WebSocket).

## 설정
- 의존성 설치: `pip install -r requirements.txt`
- 개발 서버 실행: `uvicorn app.main:app --reload`
- Swagger: `http://localhost:8000/docs`

## DEV 감정 대화 웹 테스트 모드
- `EMOTION_NO_AUTH_WEB_TEST=true` 설정 시 인증 없이 감정 대화 HTTP/WS 호출 가능 (웹 테스트 전용).
- 웹 테스트 유저는 `WEB_TEST_USER_EMAIL`(기본 `webtest@local`)과 `WEB_TEST_USER_NAME`(기본 `Web Test User`)로 생성/조회되어, 클라이언트가 임의 ID를 지정하지 못하도록 한다.
- 기본값은 false이며, 인증이 필요한 기존 동작을 유지한다.
- 이 모드를 사용할 경우 `/emotion/sessions` 또는 `/ws/emotion`을 익명 호출로 확인하고, 개발 환경 외에서는 반드시 끈다.

## 프로젝트 구조
- `app/main.py`: FastAPI 엔트리포인트
- `app/routers/emotion.py`: 감정 대화 HTTP 라우트
- `app/routers/emotion_ws.py`: 감정 대화 WebSocket
- `app/dependencies/auth.py`: 인증 의존성
- `app/services/web_test_user.py`: 웹 테스트 유저 헬퍼
- `app/db/session.py`: DB 엔진/세션 헬퍼
- `app/models`: SQLModel 모델
- `tests/`: 기본 스모크 테스트

## 프롬프트 콘텐츠 API
- `GET /prompts/system`: 시스템 프롬프트 텍스트 및 메타데이터 반환
- `GET /prompts/task`: 태스크 프롬프트 텍스트 및 메타데이터 반환
- `PUT /prompts/system`: 시스템 프롬프트 업데이트 (원자적 쓰기 + 백업)
- `PUT /prompts/task`: 태스크 프롬프트 업데이트 (원자적 쓰기 + 백업)
- 기본은 인증 필요. 로컬 개발에서만 공개 읽기를 허용하려면 `PROMPT_API_DEV_PUBLIC=true` 설정.
- 업데이트 시 `X-Admin-Key` 헤더가 `PROMPT_ADMIN_KEY`와 일치해야 한다. `PROMPT_ADMIN_KEY`가 비어 있으면 업데이트가 비활성화되며, `ENV=dev`와 `PROMPT_ADMIN_DEV_ALLOW=true`일 때만 예외로 허용된다.
- 요청 본문: `{"content": "new prompt text..."}` (최대 50KB, 비어 있으면 안 됨)
- 백업은 `resources/.backups/` 아래에 타임스탬프 파일로 저장.

### 스모크 체크
1) 유효한 인증 + `X-Admin-Key`로 `PUT /prompts/system` 호출 (새 내용 전달).
2) `GET /prompts/system`이 업데이트된 내용/sha256/updated_at를 반환하는지 확인.
3) 감정 대화 생성 플로우를 실행해 서버 재시작 없이 새 프롬프트가 반영되는지 확인.
