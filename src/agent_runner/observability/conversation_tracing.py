from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry.trace import SpanKind

from agent_runner.api.streaming import DoneEvent, ErrorEvent, PersistedEvent, SavingEvent, TokenDeltaEvent, UsageEvent
from agent_runner.config import ConversationRequest, Settings
from agent_runner.observability.tracing import Span, Tracer, extract_trace_context, trace_json


@dataclass
class ConversationStreamTraceStats:
    """Aggregate stream evidence without creating a span per token delta."""

    event_count: int = 0
    token_event_count: int = 0
    response_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    saving: bool = False
    persisted: bool = False
    done: bool = False
    error_count: int = 0

    def record_event(self, span: Span, event: object) -> None:
        """Record one semantic stream event on the Conversation request span."""
        self.event_count += 1
        if isinstance(event, TokenDeltaEvent):
            self.token_event_count += 1
            self.response_chars += len(event.content or "")
        elif isinstance(event, UsageEvent):
            prompt_tokens = event.prompt_tokens or 0
            completion_tokens = event.completion_tokens or 0
            total_tokens = event.total_tokens or 0
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
        """Attach counters and the terminal stream outcome to the request span."""
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
        outcome = "completed" if self.done else "failed" if self.error_count else "interrupted"
        span.set_attribute("conversation.outcome", outcome)


class ConversationTrace:
    """Request-scoped recorder exposed to the HTTP streaming boundary."""

    def __init__(self, span: Span) -> None:
        self._span = span
        self._stats = ConversationStreamTraceStats()

    def record_event(self, event: object) -> None:
        """Record one emitted SSE event."""
        self._stats.record_event(self._span, event)

    def finish(self) -> None:
        """Finalize aggregate stream evidence."""
        self._stats.finish(self._span)


class ConversationTracing:
    """Own Conversation HTTP span creation and trace-content policy."""

    def __init__(self, tracer: Tracer, settings: Settings) -> None:
        self._tracer = tracer
        self._settings = settings

    @contextmanager
    def trace_request(
        self,
        headers: Mapping[str, str],
        conversation_request: ConversationRequest,
    ) -> Generator[ConversationTrace]:
        """Trace the full streaming lifetime from accepted request through terminal SSE."""
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
        parent_context = extract_trace_context(headers)
        with self._tracer.span(
            "conversation.request",
            attributes,
            parent_context=parent_context,
            kind=SpanKind.SERVER,
        ) as span:
            trace = ConversationTrace(span)
            span.add_event("conversation.accepted")
            try:
                yield trace
            finally:
                trace.finish()
