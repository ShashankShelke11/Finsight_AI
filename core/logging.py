"""
Structured logging module for FinSight AI.
Supports JSON-formatted output for production observability and readable console output for local development.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any
from core.config import settings


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON strings for log aggregators (Datadog, CloudWatch, etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if hasattr(record, "props") and isinstance(record.props, dict):
            log_obj.update(record.props)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logger(name: str = "finsight") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        if settings.ENVIRONMENT.lower() == "production":
            handler.setFormatter(JSONFormatter())
        else:
            fmt = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
            handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger("finsight")
