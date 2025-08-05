import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    if logger.handlers:
        return  # 중복 설정 방지
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    ))
    logger.addHandler(h)
