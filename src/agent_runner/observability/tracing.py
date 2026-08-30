import json
import logging
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from contextvars import Token
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from opentelemetry import context, propagate, trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

_REDACTED = "[REDACTED]"
_MAX_DEPTH = 12
_SENSITIVE_KEY_SUFFIXES = (
    "accesstoken",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "idtoken",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
)
_SENSITIVE_KEYS = frozenset({"passwd", "token"})


def _hex_trace_id(span_context: trace.SpanContext) -> str:
    """Format a valid OpenTelemetry trace ID as lowercase W3C hexadecimal text.

    Args:
        span_context: OpenTelemetry span context whose trace ID is formatted.

    Returns:
        Formatted a valid OpenTelemetry trace ID as lowercase W3C hexadecimal text.
    """
    return format(span_context.trace_id, "032x") if span_context.is_valid else ""


def _hex_span_id(span_context: trace.SpanContext) -> str:
    """Format a valid OpenTelemetry span ID as lowercase hexadecimal text.

    Args:
        span_context: OpenTelemetry span context whose span ID is formatted.

    Returns:
        Formatted a valid OpenTelemetry span ID as lowercase hexadecimal text.
    """
    return format(span_context.span_id, "016x") if span_context.is_valid else ""


def trace_json(value: Any, max_chars: int) -> str:
    """Serialize trace content with recursive secret redaction and a hard size limit.

    Args:
        value: Candidate value to validate, normalize, or serialize.
        max_chars: Maximum number of characters retained in serialized evidence.

    Returns:
        Serialized trace content with recursive secret redaction and a hard size limit.
    """
    normalized: Any = _sanitize_trace_value(value, depth=0)
    if isinstance(normalized, str):
        serialized: str = normalized
    else:
        serialized = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    if len(serialized) <= max_chars:
        return serialized
    suffix: str = f"...[truncated {len(serialized) - max_chars} chars]"
    prefix_length: int = max(0, max_chars - len(suffix))
    return f"{serialized[:prefix_length]}{suffix}"


def _sanitize_trace_value(value: Any, *, depth: int) -> Any:
    """Recursively produce bounded, credential-redacted values safe for optional span capture.

    Args:
        value: Candidate value to validate, normalize, or serialize.
        depth: Current recursion depth used to enforce the safety limit.

    Returns:
        Bounded, recursively redacted value safe for optional span capture.
    """
    if depth >= _MAX_DEPTH:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return _sanitize_trace_string(value, depth=depth)
    if isinstance(value, bytes):
        return f"[BINARY {len(value)} bytes]"
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _is_sensitive_key(str(key)) else _sanitize_trace_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, Sequence):
        return [_sanitize_trace_value(item, depth=depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        return _sanitize_trace_value(value.model_dump(mode="json"), depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_trace_value(asdict(value), depth=depth + 1)
    return str(value)


def _sanitize_trace_string(value: str, *, depth: int) -> Any:
    """Parse JSON-looking strings so nested sensitive keys receive the same redaction policy.

    Args:
        value: Candidate value to validate, normalize, or serialize.
        depth: Current recursion depth used to enforce the safety limit.

    Returns:
        Parsed JSON-looking strings so nested sensitive keys receive the same redaction policy.
    """
    stripped: str = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed: Any = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return _sanitize_trace_value(parsed, depth=depth + 1)


def _is_sensitive_key(key: str) -> bool:
    """Return whether a normalized key is likely to contain a credential or token.

    Args:
        key: Credential-isolated identity of the target MCP server.

    Returns:
        ``True`` when the normalized key is likely to contain a credential or token.
    """
    normalized: str = "".join(character for character in key.lower() if character.isalnum())
    return normalized in _SENSITIVE_KEYS or any(normalized.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES)


class Span:
    """Small AgentBreaker facade over an OpenTelemetry span.

    Attributes:
        _otel_span: OpenTelemetry span owned by this facade until :meth:`end` is called.
        name: Stable operation name emitted to the tracing backend.
        parent_id: Explicit parent span ID retained for diagnostics, when present.
    """

    def __init__(self, otel_span: trace.Span, name: str, parent_id: str | None = None) -> None:
        """Wrap one OpenTelemetry span with AgentBreaker trace identifiers.

        Args:
            otel_span: OpenTelemetry span implementation owned by the tracer.
            name: Stable operation name shown in tracing backends.
            parent_id: Parent span ID when explicit parent context was provided.
        """
        self._otel_span = otel_span
        self.name = name
        self.parent_id = parent_id

    @property
    def span_id(self) -> str:
        """Return the lowercase hexadecimal ID of this span.

        Returns:
            Lowercase hexadecimal ID of this span.
        """
        return _hex_span_id(self._otel_span.get_span_context())

    @property
    def trace_id(self) -> str:
        """Return the lowercase hexadecimal trace ID shared by this span tree.

        Returns:
            Lowercase hexadecimal trace ID shared by this span tree.
        """
        return _hex_trace_id(self._otel_span.get_span_context())

    def set_attribute(self, key: str, value: Any) -> None:
        """Set one non-null attribute on the wrapped span.

        Args:
            key: Low-cardinality OpenTelemetry attribute key.
            value: Candidate value to validate, normalize, or serialize.
        """
        if value is not None:
            self._otel_span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Append one bounded semantic event to the wrapped span.

        Args:
            name: Stable semantic event name.
            attributes: Optional bounded event attributes; raw secrets must already be removed.
        """
        self._otel_span.add_event(name, attributes=attributes)

    def record_exception(self, error: BaseException) -> None:
        """Mark the span as failed with the exception type without copying its message.

        Args:
            error: Exception or failure being classified or recorded.
        """
        self._otel_span.set_attribute("error.type", type(error).__name__)
        self._otel_span.set_status(Status(StatusCode.ERROR, type(error).__name__))

    @contextmanager
    def activate(self) -> Generator["Span"]:
        """Make this already-created span current in the calling context only.

        Yields:
            This facade while the wrapped span is current in the calling context.
        """
        with trace.use_span(self._otel_span, end_on_exit=False):
            try:
                yield self
            except BaseException as error:
                self.record_exception(error)
                raise

    def end(self) -> None:
        """Finish this span after every context that used it has exited."""
        self._otel_span.end()


class Tracer:
    """Creates real OpenTelemetry spans while keeping the existing local API narrow.

    Attributes:
        service_name: Resource service name assigned to newly created spans.
        _provider: OpenTelemetry provider that owns span processors and lifecycle.
        _otel_tracer: Application-owned OpenTelemetry facade used by this component.
    """

    def __init__(self, service_name: str = "agent-runner", provider: TracerProvider | None = None) -> None:
        """Create a narrow span facade around an OpenTelemetry provider.

        Args:
            service_name: Resource service name used for newly created providers.
            provider: Optional application-owned provider, typically supplied by TracingManager.
        """
        self.service_name = service_name
        self._provider = provider or TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            sampler=ParentBased(TraceIdRatioBased(1.0)),
        )
        self._otel_tracer = self._provider.get_tracer(service_name)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        *,
        parent_context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Generator[Span]:
        """Activate one span, preserve explicit parentage, and record escaping exceptions.

        Args:
            name: OpenTelemetry operation name for the child span.
            attributes: Bounded low-cardinality attributes attached to the span.
            parent_context: Explicit OpenTelemetry parent context, when available.
            kind: OpenTelemetry span kind describing the operation boundary.

        Yields:
            Active span facade that is ended after the context exits.
        """
        wrapped: Span = self.start_span(name, attributes, parent_context=parent_context, kind=kind)
        try:
            with wrapped.activate():
                yield wrapped
        finally:
            wrapped.end()

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        *,
        parent_context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Span:
        """Create a span without binding its context to the current coroutine.

        Args:
            name: OpenTelemetry operation name for the new span.
            attributes: Bounded low-cardinality attributes attached to the span.
            parent_context: Explicit OpenTelemetry parent context, when available.
            kind: OpenTelemetry span kind describing the operation boundary.

        Returns:
            Newly created span facade, not yet activated in the current context.
        """
        parent_span: SpanContext | None = (
            trace.get_current_span(parent_context).get_span_context() if parent_context else None
        )
        parent_id: str | None = _hex_span_id(parent_span) if parent_span is not None and parent_span.is_valid else None
        otel_span: OtelSpan = self._otel_tracer.start_span(
            name,
            context=parent_context,
            kind=kind,
            attributes=attributes,
        )
        return Span(otel_span, name, parent_id)


class TracingManager:
    """Application-scoped owner of the OpenTelemetry provider and exporter.

    Attributes:
        _provider: Application-owned provider flushed during shutdown.
        tracer: Application-owned OpenTelemetry facade.
    """

    def __init__(
        self,
        service_name: str = "agent-runner",
        *,
        endpoint: str | None = None,
        sampling_ratio: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Create the application-owned provider and optional OTLP exporter.

        Args:
            service_name: Resource service name emitted with every span.
            endpoint: Optional OTLP gRPC endpoint; absent means local-only tracing.
            sampling_ratio: Fraction of traces sampled by the provider.
            enabled: Whether exporter setup is attempted when an endpoint is configured.
        """
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            sampler=ParentBased(TraceIdRatioBased(sampling_ratio)),
        )
        if enabled and endpoint:
            self._configure_exporter(endpoint)
        self.tracer = Tracer(service_name, self._provider)

    def _configure_exporter(self, endpoint: str) -> None:
        """Attach the OTLP exporter while keeping exporter failures non-fatal to requests.

        Args:
            endpoint: OTLP endpoint to configure for trace export.
        """
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter: OTLPSpanExporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=endpoint.startswith("http://"),
            )
            self._provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OpenTelemetry OTLP exporter configured for %s", endpoint)
        except Exception:
            logger.exception("OpenTelemetry exporter setup failed; requests will continue without export")

    def shutdown(self) -> None:
        """Flush and stop the provider owned by this application instance."""
        try:
            self._provider.shutdown()
        except Exception:
            logger.exception("OpenTelemetry shutdown failed")


def current_trace_id() -> str:
    """Return the active context trace ID or an empty string outside a span.

    Returns:
        Active context trace ID, or an empty string outside a span.
    """
    return _hex_trace_id(trace.get_current_span().get_span_context())


def current_span_id() -> str:
    """Return the active context span ID or an empty string outside a span.

    Returns:
        Active context span ID, or an empty string outside a span.
    """
    return _hex_span_id(trace.get_current_span().get_span_context())


def extract_trace_context(headers: Mapping[str, str]) -> Context:
    """Extract W3C traceparent metadata from inbound headers.

    Args:
        headers: HTTP headers carrying request metadata or trace context.

    Returns:
        Extracted W3C traceparent metadata from inbound headers.
    """
    return propagate.extract(carrier=headers)


def inject_trace_context(headers: MutableMapping[str, str]) -> None:
    """Inject the active W3C trace context into outbound mutable headers.

    Args:
        headers: HTTP headers carrying request metadata or trace context.
    """
    propagate.inject(carrier=headers)


def attach_trace_context(trace_context: Context) -> Token[Context]:
    """Attach a trace context and return the token required for later detachment.

    Args:
        trace_context: Context to attach to the current coroutine.

    Returns:
        Context token required to detach the attached context.
    """
    return context.attach(trace_context)


def detach_trace_context(token: Token[Context]) -> None:
    """Detach a context previously attached by :func:`attach_trace_context`.

    Args:
        token: Cooperative cancellation token associated with the request.
    """
    context.detach(token)
