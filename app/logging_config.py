"""Structured JSON logging for the inference app.

Every log line is a single JSON object so Cloud Run's log aggregator
can parse fields like duration_ms, predicted_class, and status_code
without regex scraping.
"""

import json
import logging
from datetime import UTC, datetime

# Fields that are internal to LogRecord and should not be forwarded
_SKIP = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.getMessage()  # populates record.message
        payload: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level":     record.levelname,
            "event":     record.message,
        }
        # Merge caller-supplied extra fields
        for key, val in record.__dict__.items():
            if key not in _SKIP:
                payload[key] = val

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]

    # Replace uvicorn's plain-text access log with our middleware
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
