"""
Structured JSON Logger with Correlation ID Tracing.
"""

import logging
import sys
import contextvars
from typing import Optional

# Context variable for distributed request correlation ID
correlation_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        cid = correlation_id_ctx.get()
        record.correlation_id = cid or "N/A"
        return True


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("llm_gateway")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [cid:%(correlation_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(CorrelationFilter())
        logger.addHandler(handler)

    return logger
