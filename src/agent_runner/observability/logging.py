import logging
import sys
from typing import Any, cast

import structlog


def _add_trace_context(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add active trace identifiers to a structlog event without copying sensitive payloads.

    Args:
        _: Unused positional value required by the logging processor protocol.
        __: Unused keyword values required by the logging processor protocol.
        event_dict: Structured log event being normalized before emission.

    Returns:
        Added active trace identifiers to a structlog event without copying sensitive payloads.
    """
    from agent_runner.observability.tracing import current_span_id, current_trace_id

    trace_id: str = current_trace_id()
    span_id: str = current_span_id()

    if trace_id:
        event_dict["trace_id"] = trace_id

    if span_id:
        event_dict["span_id"] = span_id

    return event_dict


def setup_logging(level: int = logging.INFO, json_format: bool = False) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (default: INFO).
        json_format: Whether to use JSON format (default: False, uses console format).
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [trace_id=%(trace_id)s span_id=%(span_id)s] %(name)s: %(message)s",
        stream=sys.stdout,
        level=level,
    )

    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestContextFilter())
        handler.addFilter(ExternalMcpCredentialFilter())

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_trace_context,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

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

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


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
        from agent_runner.observability.tracing import current_span_id, current_trace_id

        record.trace_id = current_trace_id() or "-"
        record.span_id = current_span_id() or "-"

        return True


class ExternalMcpCredentialFilter(logging.Filter):
    """Suppress third-party MCP logs that may render resolved URL credentials or Headers.

    Agent Runner emits its own typed, credential-safe diagnostic at every SDK failure boundary.
    The upstream SDK logs exceptions before returning control and may include a resolved URL in
    traceback text, so those duplicate records cannot safely cross the application log boundary.
    """

    _UNSAFE_LOGGER_PREFIXES = (
        "agents.mcp",
        "httpcore",
        "httpx",
        "mcp.client",
        "openai.agents",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Allow application diagnostics while dropping credential-bearing upstream MCP records.

        Args:
            record: Log record emitted by an upstream library or application component.

        Returns:
            ``True`` when the record may be emitted after filtering.
        """

        return not record.name.startswith(self._UNSAFE_LOGGER_PREFIXES)
