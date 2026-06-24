import logging
import sys
from typing import Any

import structlog


def setup_logging(level: int = logging.INFO, json_format: bool = False):
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (default: INFO).
        json_format: Whether to use JSON format (default: False, uses console format).
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Optional name for the logger.

    Returns:
        structlog.stdlib.BoundLogger: A bound logger instance.
    """
    return structlog.get_logger(name)


class RequestContextFilter(logging.Filter):
    """
    Logging filter for adding request context to log records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and enhance log records with request context.

        Args:
            record: The log record to filter.

        Returns:
            bool: True to allow the record to be logged.
        """
        return True
