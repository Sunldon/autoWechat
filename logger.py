import logging
import logging.handlers
import os
import sys
from pathlib import Path


class _MainNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "__main__":
            main = Path(sys.argv[0]).stem
            record.name = f"<{main}>"
        return True

_LOG_DIR = Path(__file__).parent / "logs"
_INITIALIZED = False


def _auto_init():
    if not _INITIALIZED:
        setup_logger()


def setup_logger(
    log_dir: str | Path = _LOG_DIR,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 7,
):
    """初始化全局日志配置

    - 文件: 所有 DEBUG+ 写入 logs/app.log，按大小轮转，保留 7 个备份
    - 终端: 仅 WARNING+ 输出（重要信息）
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%H:%M:%S",
    ))

    _filter = _MainNameFilter()
    file_handler.addFilter(_filter)
    console_handler.addFilter(_filter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for noisy in ("httpx", "httpcore", "openai", "chromadb", "sentence_transformers", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger（自动初始化日志配置）"""
    _auto_init()
    return logging.getLogger(name)
