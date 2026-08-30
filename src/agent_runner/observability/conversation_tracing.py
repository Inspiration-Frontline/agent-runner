from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry.context import Context
from opentelemetry.trace import SpanKind

from agent_runner.api.streaming import DoneEvent, ErrorEvent, PersistedEvent, SavingEvent, TokenDeltaEvent, UsageEvent
from agent_runner.config import ConversationRequest, Settings
from agent_runner.observability.tracing import Span, Tracer, extract_trace_context, trace_json


@dataclass
class ConversationStreamTraceStats:
    """Aggregate stream evidence without creating a span per token delta.

    Attributes:
        saving: Whether a persistence-start event was observed.
        persisted: Whether durable persistence reported success.
        done: Whether the terminal done event was observed.
    """

    event_count: int = 0
    """Number of SSE events emitted for the request."""
    token_event_count: int = 0
    """Number of token-delta events emitted for the request."""
    response_chars: int = 0
    """Number of visible response characters observed in token deltas."""
    input_tokens: int = 0
    """Sum of prompt tokens reported by usage events."""
    output_tokens: int = 0
    """Sum of completion tokens reported by usage events."""
    total_tokens: int = 0
    """Sum of total tokens reported by usage events."""
    saving: bool = False
    """Whether a persistence-start event was emitted."""
    persisted: bool = False
    """Whether durable persistence reported success."""
    done: bool = False
    """Whether the terminal done event was emitted."""
    error_count: int = 0
    """Number of typed error events emitted."""

    def record_event(self, span: Span, event: object) -> None:
        """Record one semantic stream event on the Conversation request span.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
            event: Typed runtime or SDK event to process.
        """
        self.event_count += 1
        if isinstance(event, TokenDeltaEvent):
            self.token_event_count += 1
            self.response_chars += len(event.content or "")
        elif isinstance(event, UsageEvent):
            prompt_tokens: int = event.prompt_tokens or 0
            completion_tokens: int = event.completion_tokens or 0
            total_tokens: int = event.total_tokens or 0
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens
            self.total_tokens += total_tokens
            span.add_event(
                "conversation.usage",
                {
                    "gen_ai.usage.input_tokens": prompt_tokens,
                    "gen_ai.usage.output_tokens": completion_tokens,
                    "gen_ai.usage.total_tokens": total_tokens,
                },
            )
        elif isinstance(event, SavingEvent):
            self.saving = True
            span.add_event("conversation.saving")
        elif isinstance(event, PersistedEvent):
            self.persisted = True
            span.add_event("conversation.persisted")
        elif isinstance(event, DoneEvent):
            self.done = True
            span.add_event("conversation.done")
        elif isinstance(event, ErrorEvent):
            self.error_count += 1
            span.add_event(
                "conversation.error",
                {"error.type": event.error_code or "UNKNOWN", "error.phase": event.phase or "unknown"},
            )

    def finish(self, span: Span) -> None:
        """Attach counters and the terminal stream outcome to the request span.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
        """
        span.set_attribute("conversation.event_count", self.event_count)
        span.set_attribute("conversation.token_event_count", self.token_event_count)
        span.set_attribute("conversation.response_chars", self.response_chars)
        span.set_attribute("gen_ai.usage.input_tokens", self.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", self.output_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", self.total_tokens)
        span.set_attribute("conversation.saving", self.saving)
        span.set_attribute("conversation.persisted", self.persisted)
        span.set_attribute("conversation.done", self.done)
        span.set_attribute("conversation.error_count", self.error_count)
        outcome: str = "completed" if self.done else "failed" if self.error_count else "interrupted"
        span.set_attribute("conversation.outcome", outcome)


class ConversationTrace:
    """Request-scoped recorder exposed to the HTTP streaming boundary.

    Attributes:
        _span: Already-created root request span; activation is explicit and task-local.
        _stats: Bounded aggregate of emitted SSE event and usage evidence.
        _finished: Idempotency guard ensuring aggregate attributes and span end run once.
    """

    def __init__(self, span: Span) -> None:
        """Create a request-scoped trace recorder around an already-created root span.

        Args:
            span: Root span created by :class:`ConversationTracing`.
        """
        self._span = span
        self._stats = ConversationStreamTraceStats()
        self._finished = False

    @property
    def trace_id(self) -> str:
        """Return the lowercase W3C trace identifier for the complete Round request.

        Returns:
            Lowercase W3C trace identifier for the complete Round request.
        """
        return self._span.trace_id

    def record_event(self, event: object) -> None:
        """Record one emitted SSE event.

        Args:
            event: Typed runtime or SDK event to process.
        """
        self._stats.record_event(self._span, event)

    @contextmanager
    def activate(self) -> Generator["ConversationTrace"]:
        """Activate the request span in the coroutine currently doing request work.

        Yields:
            This recorder while its root request span is current in the calling context.
        """
        with self._span.activate():
            yield self

    def finish(self) -> None:
        """Finalize aggregate stream evidence."""
        if self._finished:
            return
        self._finished = True
        try:
            self._stats.finish(self._span)
        finally:
            self._span.end()


class ConversationTracing:
    """Own Conversation HTTP span creation and trace-content policy.

    Attributes:
        _tracer: Application-owned OpenTelemetry facade used by this component.
        _settings: Effective application settings retained for this component.
    """

    def __init__(self, tracer: Tracer, settings: Settings) -> None:
        """Create the Conversation request span policy.

        Args:
            tracer: Application-owned tracer used to create the root span.
            settings: Runtime settings controlling bounded content capture.
        """
        self._tracer = tracer
        self._settings = settings

    def start_request(
        self,
        headers: Mapping[str, str],
        conversation_request: ConversationRequest,
    ) -> ConversationTrace:
        """Create a Round span that can be activated by the route and its SSE generator.

        Args:
            headers: HTTP headers carrying request metadata or trace context.
            conversation_request: Validated public Conversation request.

        Returns:
            Recorder backed by the newly created root request span.
        """
        attributes: dict[str, str | int | bool] = {
            "conversation.id": conversation_request.conversation_id,
            "conversation.file_count": len(conversation_request.file_ids),
            "conversation.reference_count": len(conversation_request.references),
            "conversation.locale": conversation_request.ui_locale,
            "conversation.message_chars": len(conversation_request.message),
            "conversation.attachment_only": not bool(conversation_request.message),
        }
        if self._settings.otel_capture_content:
            attributes["agentbreaker.conversation.request"] = trace_json(
                conversation_request.model_dump(mode="json"),
                self._settings.otel_content_max_chars,
            )
        parent_context: Context = extract_trace_context(headers)
        span: Span = self._tracer.start_span(
            "conversation.request",
            attributes,
            parent_context=parent_context,
            kind=SpanKind.SERVER,
        )
        conversation_trace: ConversationTrace = ConversationTrace(span)
        span.add_event("conversation.accepted")
        return conversation_trace
