# DEEPME Emotion Chat Backend

FastAPI + SQLModel backend for DEEPME emotion chat (HTTP + WebSocket).

## Setup
- Install deps: `pip install -r requirements.txt`
- Run dev server: `uvicorn app.main:app --reload`
- Swagger: `http://localhost:8000/docs`

## DEV emotion web-test mode
- Set `EMOTION_NO_AUTH_WEB_TEST=true` to allow emotion chat HTTP/WS calls without auth (for web testing only).
- The web-test user is created/read with `WEB_TEST_USER_EMAIL` (default `webtest@local`) and `WEB_TEST_USER_NAME` (default `Web Test User`) so clients cannot pick arbitrary IDs.
- Default (false) keeps current auth-required behavior.
- If using this mode, hit `/emotion/sessions` or `/ws/emotion` anonymously to verify it works; turn it off in any non-dev environment.

## Project layout
- `app/main.py`: FastAPI entrypoint
- `app/routers/emotion.py`: Emotion HTTP routes
- `app/routers/emotion_ws.py`: Emotion WebSocket
- `app/dependencies/auth.py`: Auth dependencies
- `app/services/web_test_user.py`: Web-test user helper
- `app/db/session.py`: DB engine and session helpers
- `app/models`: SQLModel models
- `tests/`: basic smoke tests

## Prompt content API
- `GET /prompts/system`: returns the current system prompt text and metadata.
- `GET /prompts/task`: returns the current task prompt text and metadata.
- `PUT /prompts/system`: update the system prompt text (atomic write + backup).
- `PUT /prompts/task`: update the task prompt text (atomic write + backup).
- Auth is required by default; set `PROMPT_API_DEV_PUBLIC=true` only for local development to allow public read access.
- Updates additionally require header `X-Admin-Key` matching `PROMPT_ADMIN_KEY`. If `PROMPT_ADMIN_KEY` is empty, updates are disabled unless `ENV=dev` and `PROMPT_ADMIN_DEV_ALLOW=true`.
- Request body for updates: `{"content": "new prompt text..."}` (max 50KB, non-empty).
- Backups are stored under `resources/.backups/` with timestamped filenames.

### Smoke check
1) `PUT /prompts/system` with valid auth + `X-Admin-Key` and new content.
2) `GET /prompts/system` returns the updated content/sha256/updated_at.
3) Trigger an emotion generation flow and confirm it reflects the new system prompt without restarting the server.
