"""Structured logging and Langfuse tracing."""

import json
import logging
import os
import sys
import time

from mono_agent.config import Settings


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to grep locally, ready for any log shipper."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))  # logger.info("...", extra={"fields": {...}})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def tracing_callbacks(settings: Settings) -> list:
    """Langfuse callback handler if keys are configured, else nothing (tracing is optional)."""
    if not settings.tracing_enabled or settings.langfuse_secret_key is None:
        return []
    # The SDK reads its configuration from the environment.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key.get_secret_value())
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def flush_traces(settings: Settings) -> None:
    """Send buffered traces before the process exits (the SDK batches in the background)."""
    if settings.tracing_enabled:
        from langfuse import get_client

        get_client().flush()
