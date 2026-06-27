from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass
class Span:
    """
    Tracing span for distributed tracing.

    Represents a single operation within a trace, tracking
    timing, attributes, and parent-child relationships.

    Attributes:
        span_id: Unique identifier for this span.
        trace_id: Trace ID that this span belongs to.
        name: Name of the operation this span represents.
        parent_id: ID of the parent span, if this is a child span.
        attributes: Dictionary of attributes attached to this span.
    """

    span_id: str
    trace_id: str
    name: str
    parent_id: str | None = None
    attributes: dict[str, Any] = None

    def __post_init__(self):
        """
        Initialize attributes dictionary if not provided.
        """
        if self.attributes is None:
            self.attributes = {}

    def set_attribute(self, key: str, value: Any):
        """
        Set an attribute on this span.

        Args:
            key: Attribute key.
            value: Attribute value.
        """
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """
        Add an event to this span.

        Args:
            name: Name of the event.
            attributes: Optional attributes for the event.
        """
        pass


class Tracer:
    """
    Tracer for creating and managing trace spans.

    Provides functionality to create spans, manage span hierarchy,
    and track current span context.

    Attributes:
        service_name: Name of the service for trace identification.
        _current_span: Currently active span in the context.
    """

    def __init__(self, service_name: str = "agent-runner"):
        """
        Initialize the tracer.

        Args:
            service_name: Name of the service being traced.
        """
        self.service_name = service_name
        self._current_span: Span | None = None

    def start_span(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Name of the operation for this span.
            parent: Optional parent span for hierarchical tracing.
            attributes: Optional initial attributes for the span.

        Returns:
            Span: The newly created span.
        """
        trace_id = parent.trace_id if parent else str(uuid4())
        parent_id = parent.span_id if parent else None

        span = Span(
            span_id=str(uuid4()),
            trace_id=trace_id,
            name=name,
            parent_id=parent_id,
            attributes=attributes or {},
        )

        return span

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span]:
        """
        Context manager for scoped span usage.

        Args:
            name: Name of the operation for this span.
            attributes: Optional initial attributes for the span.

        Yields:
            Span: The span for use within the context.
        """
        span = self.start_span(name, self._current_span, attributes)
        previous_span = self._current_span
        self._current_span = span

        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            self._current_span = previous_span


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """
    Get the global tracer instance.

    Returns:
        Tracer: The global tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def init_tracer(service_name: str = "agent-runner") -> Tracer:
    """
    Initialize the global tracer with a service name.

    Args:
        service_name: Name of the service for trace identification.

    Returns:
        Tracer: The initialized tracer instance.
    """
    global _tracer
    _tracer = Tracer(service_name)
    return _tracer
