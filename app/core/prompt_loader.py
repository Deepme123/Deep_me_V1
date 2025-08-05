from pathlib import Path
import logging
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = BASE_DIR / "resources" / "system_prompt.txt"

FALLBACK_PROMPT = (
    "너는 감정 기반 챗봇이야. 사용자의 감정을 존중하고, 공감적 질문을 통해 "
    "사용자가 스스로 감정을 탐색하도록 돕는다."
)

def _load(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf-8")
        # 운영 환경이라면, 굳이 경로 출력할 필요 없음
        return txt.strip()
    except FileNotFoundError:
        logging.warning("[PromptLoader] system_prompt.txt not found. Using fallback.")
        return FALLBACK_PROMPT

@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return _load(PROMPT_PATH)

TASK_PROMPT_PATH = BASE_DIR / "resources" / "task_prompt.txt"

def get_task_prompt() -> str:
    try:
        txt = TASK_PROMPT_PATH.read_text(encoding="utf-8")
        return txt.strip()
    except FileNotFoundError:
        raise RuntimeError("[PromptLoader] task_prompt.txt not found.")


SYSTEM_PROMPT = get_system_prompt()

