from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry.trace import SpanKind

from agent_runner.config import Settings
from agent_runner.observability.tracing import Span, Tracer, trace_json
from agent_runner.tools.registry import ToolDefinition


class ToolCallTrace:
    """Typed tracing surface used by one Tool execution."""

    def __init__(self, span: Span, settings: Settings) -> None:
        self._span = span
        self._settings = settings

    def record_result(self, status: str, result: str, error_type: str | None = None) -> None:
        """Attach one terminal Tool outcome and optional captured result."""
        self._span.set_attribute("tool.status", status)
        self._span.set_attribute("error.type", error_type)
        if self._settings.otel_capture_content:
            self._span.set_attribute(
                "gen_ai.tool.call.result",
                trace_json(result, self._settings.otel_content_max_chars),
            )


class ToolTracing:
    """Own span creation and content-capture policy for Tool calls."""

    def __init__(self, tracer: Tracer, settings: Settings) -> None:
        self._tracer = tracer
        self._settings = settings

    @contextmanager
    def trace_call(
        self,
        definition: ToolDefinition,
        tool_call_id: str,
        arguments_json: str,
    ) -> Generator[ToolCallTrace]:
        """Trace one Tool invocation without exposing Span operations to the executor."""
        attributes = {
            "tool.key": definition.tool_key,
            "tool.name": definition.tool_name,
            "tool.source": definition.source_type.value,
            "gen_ai.tool.call.id": tool_call_id,
            "gen_ai.tool.name": definition.tool_name,
        }
        with self._tracer.span("tool.call", attributes, kind=SpanKind.CLIENT) as span:
            if self._settings.otel_capture_content:
                span.set_attribute(
                    "gen_ai.tool.call.arguments",
                    trace_json(arguments_json, self._settings.otel_content_max_chars),
                )
            yield ToolCallTrace(span, self._settings)
