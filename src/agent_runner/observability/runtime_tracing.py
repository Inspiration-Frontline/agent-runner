from collections.abc import Generator
from contextlib import contextmanager

from agent_runner.observability.tracing import Span, Tracer


class RuntimeTracing:
    """Own span names and attribute schemas for Runtime Orchestrator phases."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    @contextmanager
    def trace_file_preparation(self, file_count: int) -> Generator[Span]:
        with self._tracer.span("file.prepare", {"file.count": file_count}) as span:
            yield span

    @contextmanager
    def trace_context_build(
        self,
        conversation_id: str,
        history_message_count: int,
        prepared_file_count: int,
        reference_count: int,
    ) -> Generator[Span]:
        with self._tracer.span(
            "context.build",
            {
                "conversation.id": conversation_id,
                "context.history_message_count": history_message_count,
                "context.prepared_file_count": prepared_file_count,
                "context.reference_count": reference_count,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_agent_run(self, attributes: dict[str, object]) -> Generator[Span]:
        with self._tracer.span("agent.run", attributes) as span:
            yield span

    @contextmanager
    def trace_preflight(self, conversation_id: str, reference_count: int) -> Generator[Span]:
        with self._tracer.span(
            "preflight",
            {"conversation.id": conversation_id, "conversation.reference_count": reference_count},
        ) as span:
            yield span

    @contextmanager
    def trace_reference_preparation(self, reference_count: int) -> Generator[Span]:
        with self._tracer.span("reference.prepare", {"reference.count": reference_count}) as span:
            yield span

    @contextmanager
    def trace_round_persistence(self, attributes: dict[str, object]) -> Generator[Span]:
        with self._tracer.span("round.persist", attributes) as span:
            yield span
