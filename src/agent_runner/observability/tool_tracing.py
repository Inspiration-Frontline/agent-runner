from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry.trace import SpanKind

from agent_runner.config import Settings
from agent_runner.observability.tracing import Span, Tracer, trace_json
from agent_runner.tools.registry import ToolDefinition


class ToolCallTrace:
    """Typed tracing surface used by one Tool execution.

    Attributes:
        _span: OpenTelemetry span receiving bounded Tool-call evidence.
        _settings: Effective application settings retained for this component.
    """

    def __init__(self, span: Span, settings: Settings) -> None:
        """Bind the request's Tool span and content-capture policy.

        Args:
            span: Already-created span owned by :class:`ToolTracing`.
            settings: Runtime settings controlling optional argument/result capture.
        """
        self._span = span
        self._settings = settings

    def record_result(self, status: str, result: str, error_type: str | None = None) -> None:
        """Attach one terminal Tool outcome and optional captured result.

        Args:
            status: Terminal domain status being recorded or persisted.
            result: Operation result to normalize, trace, or persist.
            error_type: Domain error type value used by the operation.
        """
        self._span.set_attribute("tool.status", status)
        self._span.set_attribute("error.type", error_type)

        if self._settings.otel_capture_content:
            self._span.set_attribute(
                "gen_ai.tool.call.result",
                trace_json(result, self._settings.otel_content_max_chars),
            )


class ToolTracing:
    """Own span creation and content-capture policy for Tool calls.

    Attributes:
        _tracer: Application-owned OpenTelemetry facade used by this component.
        _settings: Effective application settings retained for this component.
    """

    def __init__(self, tracer: Tracer, settings: Settings) -> None:
        """Create a Tool tracing boundary with a shared tracer and runtime settings.

        Args:
            tracer: OpenTelemetry facade used to create Tool spans.
            settings: Runtime settings controlling optional content capture.
        """
        self._tracer = tracer
        self._settings = settings

    @contextmanager
    def trace_call(
        self,
        definition: ToolDefinition,
        tool_call_id: str,
        arguments_json: str,
    ) -> Generator[ToolCallTrace]:
        """Trace one Tool invocation without exposing Span operations to the executor.

        Args:
            definition: Frozen Tool definition and source provenance.
            tool_call_id: Provider-generated Tool call identifier.
            arguments_json: Exact JSON Tool arguments emitted by the model.

        Yields:
            Request-scoped Tool trace recorder while its span is active.
        """
        attributes: dict[str, str] = {
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
