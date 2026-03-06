from __future__ import annotations

import logging
import os
from typing import Final

_DEFAULT_LOG_LEVEL: Final[str] = "INFO"
_DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_LOGGER_NAMESPACE: Final[str] = "poc-skills"
_is_configured = False


def _resolve_level(level: str) -> int:
    normalized = level.strip().upper()
    return getattr(logging, normalized, logging.INFO)


def configure_logging(
    level: str | None = None,
    log_format: str = _DEFAULT_LOG_FORMAT,
    datefmt: str = _DEFAULT_LOG_DATEFMT,
) -> None:
    global _is_configured
    if _is_configured:
        return

    resolved_level = _resolve_level(level or os.getenv("LOG_LEVEL", _DEFAULT_LOG_LEVEL))

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
        root_logger.addHandler(stream_handler)

    _is_configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    logger_name = _LOGGER_NAMESPACE if name is None else name
    return logging.getLogger(logger_name)


logger = get_logger(_LOGGER_NAMESPACE)
