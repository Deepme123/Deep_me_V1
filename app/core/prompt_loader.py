from pathlib import Path
import logging
from functools import lru_cache
import os

# ─────────────────────────────
# 환경 설정 (DEBUG 여부)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ─────────────────────────────
# 프롬프트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = BASE_DIR / "resources" / "system_prompt.txt"

FALLBACK_PROMPT = (
    "너는 감정 기반 챗봇이야. 사용자의 감정을 존중하고, 공감적 질문을 통해 "
    "사용자가 스스로 감정을 탐색하도록 돕는다."
)

def _load(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf-8")
        if DEBUG:
            logging.info(f"[PromptLoader] ✅ system_prompt loaded from '{path}'.")
        return txt.strip()
    except FileNotFoundError:
        logging.warning(
            f"[PromptLoader] ⚠️ '{path}' not found. Using fallback prompt.")
        return FALLBACK_PROMPT

@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return _load(PROMPT_PATH)

# 기존 코드와 호환성을 위해 상수도 제공
SYSTEM_PROMPT = get_system_prompt()
