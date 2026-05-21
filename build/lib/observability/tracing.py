import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator
from uuid import uuid4


@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    parent_id: str | None = None
    attributes: dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        pass


class Tracer:
    def __init__(self, service_name: str = "agent-runner"):
        self.service_name = service_name
        self._current_span: Span | None = None

    def start_span(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
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
    ) -> Generator[Span, None, None]:
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
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def init_tracer(service_name: str = "agent-runner") -> Tracer:
    global _tracer
    _tracer = Tracer(service_name)
    return _tracer
