"""Resume Screener — Structured logging configuration.

Uses structlog throughout the application. No print() statements allowed.
All log entries are JSON-formatted for machine parsing in production.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "info") -> None:
    """Configure structlog with JSON rendering for production.

    Args:
        log_level: The minimum log level to emit (e.g. "info", "debug").
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Standard library logging → structlog bridge
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)
