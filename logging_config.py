"""CineBot icin merkezi, makine tarafindan okunabilir log yapilandirmasi."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


_request_id = contextvars.ContextVar("cinebot_request_id", default=None)
_configured = False


def set_request_id(value):
    _request_id.set(value or None)


def get_request_id():
    return _request_id.get()


class JsonLogFormatter(logging.Formatter):
    """Loglari Loki/OpenTelemetry tarafinda kolay sorgulanacak JSON'a cevirir."""

    EXTRA_FIELDS = (
        "event", "channel", "stage", "route", "method", "status",
        "status_code", "duration_ms", "error_type", "session_id",
        "recording_id", "request_id", "remote_addr",
    )

    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname.lower(),
            "service": os.environ.get("OTEL_SERVICE_NAME", "cinebot-api"),
            "environment": os.environ.get(
                "CINEBOT_ENVIRONMENT",
                os.environ.get("RENDER_SERVICE_NAME", "development"),
            ),
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id:
            payload["request_id"] = request_id
        for field in self.EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None and field not in payload:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _LoggingStream:
    """Mevcut print() cagrilarini da ayni JSON log hattina dahil eder."""

    def __init__(self, logger, level, original):
        self.logger = logger
        self.level = level
        self.original = original
        self._buffer = ""

    def write(self, value):
        self._buffer += str(value)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(value)

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip())
        self._buffer = ""

    def isatty(self):
        return False

    def fileno(self):
        return self.original.fileno()

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")


def configure_logging():
    """Root logger'i bir kez yapilandir; stdout her ortamda ana hedeftir."""
    global _configured
    if _configured:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = JsonLogFormatter()

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stream_handler = logging.StreamHandler(original_stdout)
    stream_handler.setFormatter(formatter)
    handlers = [stream_handler]

    log_file = os.environ.get("CINEBOT_LOG_FILE", "").strip()
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=int(os.environ.get("CINEBOT_LOG_MAX_BYTES", "10485760")),
            backupCount=int(os.environ.get("CINEBOT_LOG_BACKUP_COUNT", "3")),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)

    # Kutuphanelerin debug ciktilari hem gurultu hem de hassas header riski tasir.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    if os.environ.get("CINEBOT_CAPTURE_PRINTS", "1").lower() not in {
        "0", "false", "no",
    }:
        sys.stdout = _LoggingStream(
            logging.getLogger("stdout"), logging.INFO, original_stdout
        )
        sys.stderr = _LoggingStream(
            logging.getLogger("stderr"), logging.ERROR, original_stderr
        )
    _configured = True
