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
from opentelemetry.trace import SpanKind, Status, StatusCode

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
    return format(span_context.trace_id, "032x") if span_context.is_valid else ""


def _hex_span_id(span_context: trace.SpanContext) -> str:
    return format(span_context.span_id, "016x") if span_context.is_valid else ""


def trace_json(value: Any, max_chars: int) -> str:
    """Serialize trace content with recursive secret redaction and a hard size limit."""
    normalized = _sanitize_trace_value(value, depth=0)
    if isinstance(normalized, str):
        serialized = normalized
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
    suffix = f"...[truncated {len(serialized) - max_chars} chars]"
    prefix_length = max(0, max_chars - len(suffix))
    return f"{serialized[:prefix_length]}{suffix}"


def _sanitize_trace_value(value: Any, *, depth: int) -> Any:
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
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return _sanitize_trace_value(parsed, depth=depth + 1)


def _is_sensitive_key(key: str) -> bool:
    normalized = "".join(character for character in key.lower() if character.isalnum())
    return normalized in _SENSITIVE_KEYS or any(normalized.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES)


class Span:
    """Small AgentBreaker facade over an OpenTelemetry span."""

    def __init__(self, otel_span: trace.Span, name: str, parent_id: str | None = None) -> None:
        self._otel_span = otel_span
        self.name = name
        self.parent_id = parent_id

    @property
    def span_id(self) -> str:
        return _hex_span_id(self._otel_span.get_span_context())

    @property
    def trace_id(self) -> str:
        return _hex_trace_id(self._otel_span.get_span_context())

    def set_attribute(self, key: str, value: Any) -> None:
        if value is not None:
            self._otel_span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._otel_span.add_event(name, attributes=attributes)

    def record_exception(self, error: BaseException) -> None:
        self._otel_span.set_attribute("error.type", type(error).__name__)
        self._otel_span.set_status(Status(StatusCode.ERROR, type(error).__name__))


class Tracer:
    """Creates real OpenTelemetry spans while keeping the existing local API narrow."""

    def __init__(self, service_name: str = "agent-runner", provider: TracerProvider | None = None) -> None:
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
        parent_span = trace.get_current_span(parent_context).get_span_context() if parent_context else None
        parent_id = _hex_span_id(parent_span) if parent_span is not None and parent_span.is_valid else None
        with self._otel_tracer.start_as_current_span(
            name,
            context=parent_context,
            kind=kind,
            attributes=attributes,
        ) as otel_span:
            wrapped = Span(otel_span, name, parent_id)
            try:
                yield wrapped
            except BaseException as error:
                wrapped.record_exception(error)
                raise


class TracingManager:
    """Application-scoped owner of the OpenTelemetry provider and exporter."""

    def __init__(
        self,
        service_name: str = "agent-runner",
        *,
        endpoint: str | None = None,
        sampling_ratio: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            sampler=ParentBased(TraceIdRatioBased(sampling_ratio)),
        )
        if enabled and endpoint:
            self._configure_exporter(endpoint)
        self.tracer = Tracer(service_name, self._provider)

    def _configure_exporter(self, endpoint: str) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))
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
    return _hex_trace_id(trace.get_current_span().get_span_context())


def current_span_id() -> str:
    return _hex_span_id(trace.get_current_span().get_span_context())


def extract_trace_context(headers: Mapping[str, str]) -> Context:
    return propagate.extract(carrier=headers)


def inject_trace_context(headers: MutableMapping[str, str]) -> None:
    propagate.inject(carrier=headers)


def attach_trace_context(trace_context: Context) -> Token[Context]:
    return context.attach(trace_context)


def detach_trace_context(token: Token[Context]) -> None:
    context.detach(token)
