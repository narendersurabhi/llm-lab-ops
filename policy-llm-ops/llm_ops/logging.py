from __future__ import annotations

import json
import logging
import random
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from llm_ops.config import settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{7,}\b")
CARD_RE = re.compile(r"\b\d{13,19}\b")


def redact_text(value: str) -> str:
    value = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = PHONE_RE.sub("[REDACTED_PHONE]", value)
    value = CARD_RE.sub("[REDACTED_NUMBER]", value)
    return value


def sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value) if settings.redact_logs else value
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    return value


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "request_id": request_id_var.get(),
    }
    payload.update(kwargs)
    safe_payload = sanitize_value(payload)
    logger.info(json.dumps(safe_payload, ensure_ascii=True))


def should_sample_prompt() -> bool:
    if settings.prompt_sample_rate <= 0:
        return False
    return random.random() < settings.prompt_sample_rate


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "request_id": request_id_var.get(),
            }
        payload.setdefault("level", record.levelname)
        return json.dumps(payload, ensure_ascii=True)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("llm_ops")
    if logger.handlers:
        return logger
    logger.setLevel(settings.log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
