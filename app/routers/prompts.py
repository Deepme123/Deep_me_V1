import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from app.core import prompt_loader
from app.dependencies.auth import get_current_user

PROMPT_API_DEV_PUBLIC = os.getenv("PROMPT_API_DEV_PUBLIC", "false").lower() == "true"
PROMPT_ADMIN_KEY = os.getenv("PROMPT_ADMIN_KEY", "") or ""
PROMPT_ADMIN_DEV_ALLOW = os.getenv("PROMPT_ADMIN_DEV_ALLOW", "false").lower() == "true"
ENV = os.getenv("ENV", "dev")

# Public prompts API: no auth required.
router = APIRouter(
    prefix="/prompts",
    tags=["prompts"],
    dependencies=[],
)

MAX_PROMPT_BYTES = 50 * 1024  # 50KB safety cap


class PromptUpdate(BaseModel):
    content: str


def _build_prompt_response(prompt_type: str, content: str, path: Path) -> dict:
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        updated_at = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except FileNotFoundError:
        updated_at = None

    return {
        "type": prompt_type,
        "content": content,
        "sha256": sha256,
        "updated_at": updated_at,
    }


def _ensure_admin(x_admin_key: Optional[str]) -> None:
    # When no admin key is configured, block updates unless explicitly allowed in dev.
    if not PROMPT_ADMIN_KEY:
        if ENV == "dev" and PROMPT_ADMIN_DEV_ALLOW:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="prompt_update_disabled",
        )
    if x_admin_key != PROMPT_ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid_admin_key",
        )


def _validate_content(content: str) -> str:
    if content is None:
        raise HTTPException(status_code=400, detail="content_required")
    trimmed = content.strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail="content_empty")
    byte_len = len(trimmed.encode("utf-8"))
    if byte_len > MAX_PROMPT_BYTES:
        raise HTTPException(status_code=400, detail="content_too_large")
    return trimmed


def _backup_prompt(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / ".backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{path.stem}.{ts}{path.suffix}"
    shutil.copy2(path, backup_path)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _update_prompt(prompt_type: str, path: Path, content: str) -> dict:
    validated = _validate_content(content)
    _backup_prompt(path)
    _write_atomic(path, validated)
    if prompt_type == "system":
        prompt_loader.get_system_prompt.cache_clear()
    return _build_prompt_response(prompt_type, validated, path)


@router.get("/system")
def read_system_prompt():
    content = prompt_loader.get_system_prompt()
    return _build_prompt_response("system", content, prompt_loader.PROMPT_PATH)


@router.get("/task")
def read_task_prompt():
    content = prompt_loader.get_task_prompt()
    return _build_prompt_response("task", content, prompt_loader.TASK_PROMPT_PATH)


@router.put("/system")
def update_system_prompt(
    payload: PromptUpdate,
    x_admin_key: Optional[str] = Header(None, convert_underscores=False),
):
    _ensure_admin(x_admin_key)
    return _update_prompt("system", prompt_loader.PROMPT_PATH, payload.content)


@router.put("/task")
def update_task_prompt(
    payload: PromptUpdate,
    x_admin_key: Optional[str] = Header(None, convert_underscores=False),
):
    _ensure_admin(x_admin_key)
    return _update_prompt("task", prompt_loader.TASK_PROMPT_PATH, payload.content)
