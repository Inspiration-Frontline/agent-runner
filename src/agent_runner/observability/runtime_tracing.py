from collections.abc import Generator, Sequence
from contextlib import contextmanager

from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    AppendConversationRoundProgressRequest,
    AppendConversationRoundProgressResponse,
    CreateConversationRoundCheckpointRequest,
    CreateConversationRoundCheckpointResponse,
    FinalizeConversationRoundRequest,
    FinalizeConversationRoundResponse,
    SaveConversationRoundRequest,
    SaveConversationRoundResponse,
)

from agent_runner.agent_definitions.config_models import AgentDefinition
from agent_runner.context.builder import AgentContext
from agent_runner.observability.tracing import Span, Tracer, current_trace_id, trace_json
from agent_runner.runtime.tool_loop import CapturedModelTurn


class RuntimeTracing:
    """Own span names and attribute schemas for Runtime Orchestrator phases.

    Attributes:
        _tracer: Application-owned OpenTelemetry facade used by this component.
    """

    def __init__(self, tracer: Tracer) -> None:
        """Create the runtime tracing policy around one application tracer.

        Args:
            tracer: OpenTelemetry facade used to create orchestration spans.
        """
        self._tracer = tracer

    @staticmethod
    def get_current_trace_id() -> str:
        """Return the active trace ID used for durable request correlation.

        Returns:
            Active trace ID used for durable request correlation, or an empty string.
        """

        return current_trace_id()

    @contextmanager
    def trace_file_preparation(self, file_count: int) -> Generator[Span]:
        """Trace attachment authorization and readiness preparation.

        Args:
            file_count: Number of file values observed.

        Yields:
            Active span covering this runtime boundary.
        """
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
        """Trace provider-neutral context assembly without recording full prompt content.

        Args:
            conversation_id: Stable public identifier of the Conversation.
            history_message_count: Number of history message values observed.
            prepared_file_count: Number of prepared file values observed.
            reference_count: Number of reference values observed.

        Yields:
            Active span covering this runtime boundary.
        """
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
    def trace_agent_run(self, agent: AgentDefinition, content_max_chars: int) -> Generator[Span]:
        """Trace one resolved Agent run and its bounded configuration summary.

        Args:
            agent: Resolved Agent definition participating in the operation.
            content_max_chars: Maximum characters retained in optional trace content.

        Yields:
            Active span covering this runtime boundary.
        """
        attributes: dict[str, object] = {
            "agent.id": agent.agent_id,
            "agent.name": agent.name,
            "agent.version": agent.version,
            "gen_ai.request.model": agent.model,
            "gen_ai.request.max_tokens": agent.max_output_tokens,
            "gen_ai.request.temperature": agent.temperature,
            "agent.tool_count": len(agent.tools),
            "agent.tools": trace_json(agent.tools, content_max_chars),
        }
        with self._tracer.span("agent.run", attributes) as span:
            yield span

    @staticmethod
    def record_context_failure(span: Span, error_type: str) -> None:
        """Record a failed context-build outcome without exposing tag policy to the orchestrator.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
            error_type: Domain error type value used by the operation.
        """
        span.set_attribute("context.status", "failed")
        span.set_attribute("error.type", error_type)
        span.add_event("context.failed")

    @staticmethod
    def record_context_ready(span: Span, context: AgentContext, agent: AgentDefinition | None) -> None:
        """Record the resolved context shape without its full LLM payload.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
            context: Request or SDK context associated with the operation.
            agent: Resolved Agent definition participating in the operation.
        """
        span.set_attribute("context.status", "ready")
        span.set_attribute("context.system_prompt_chars", len(context.system_prompt))
        span.set_attribute("context.history_message_count", len(context.conversation_history))
        span.set_attribute("context.rag_chunk_count", len(context.rag_chunks))
        span.set_attribute("context.tool_count", len(context.tool_specs))
        span.set_attribute("context.current_message_chars", len(context.current_message.content))

        if agent is not None:
            span.set_attribute("agent.id", agent.agent_id)
            span.set_attribute("agent.name", agent.name)
            span.set_attribute("gen_ai.request.model", agent.model)
        span.add_event("context.ready")

    @staticmethod
    def record_agent_result(
        span: Span,
        response_text: str | None,
        turns: Sequence[CapturedModelTurn],
        terminal_status: str | None,
    ) -> None:
        """Summarize final model and Tool evidence on the owning Agent span.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
            response_text: Bounded final response text used for result metrics.
            turns: Ordered model turns whose usage and Tool counts are summarized.
            terminal_status: Domain terminal status value used by the operation.
        """

        if response_text is None:
            span.set_attribute("agent.status", "missing_result")
            span.add_event("agent.result.missing")

            return
        span.set_attribute("agent.status", terminal_status or "COMPLETED")
        span.set_attribute("agent.response_chars", len(response_text))
        span.set_attribute("agent.turn_count", len(turns))
        span.set_attribute("agent.tool_execution_count", sum(len(turn.tool_executions) for turn in turns))
        span.set_attribute("gen_ai.usage.input_tokens", sum(turn.prompt_tokens for turn in turns))
        span.set_attribute("gen_ai.usage.output_tokens", sum(turn.completion_tokens for turn in turns))
        span.set_attribute("gen_ai.usage.total_tokens", sum(turn.total_tokens for turn in turns))
        span.add_event("agent.completed")

    @contextmanager
    def trace_preflight(self, conversation_id: str, reference_count: int) -> Generator[Span]:
        """Trace Conversation history, file, and reference authorization preflight.

        Args:
            conversation_id: Stable public identifier of the Conversation.
            reference_count: Number of reference values observed.

        Yields:
            Active span covering this runtime boundary.
        """
        with self._tracer.span(
            "preflight",
            {"conversation.id": conversation_id, "conversation.reference_count": reference_count},
        ) as span:
            yield span

    @staticmethod
    def record_preflight_failure(span: Span, error_type: str) -> None:
        """Record a failed authorization/replay preflight.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
            error_type: Domain error type value used by the operation.
        """
        span.set_attribute("preflight.status", "failed")
        span.set_attribute("error.type", error_type)
        span.add_event("preflight.failed")

    @staticmethod
    def record_preflight_ready(span: Span, next_round_number: int, history_message_count: int) -> None:
        """Record the durable boundary selected by a successful preflight.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
            next_round_number: Round number selected by preflight for persistence.
            history_message_count: Number of history message values observed.
        """
        span.set_attribute("preflight.status", "ready")
        span.set_attribute("conversation.next_round_number", next_round_number)
        span.set_attribute("conversation.history_message_count", history_message_count)
        span.add_event("preflight.ready")

    @contextmanager
    def trace_reference_preparation(self, reference_count: int) -> Generator[Span]:
        """Trace preparation of frozen same-Group Conversation references.

        Args:
            reference_count: Number of reference values observed.

        Yields:
            Active span covering this runtime boundary.
        """
        with self._tracer.span("reference.prepare", {"reference.count": reference_count}) as span:
            yield span

    @contextmanager
    def trace_round_persistence(self, request: SaveConversationRoundRequest) -> Generator[Span]:
        """Trace an atomic terminal Round persistence request.

        Args:
            request: Validated Conversation request being traced.

        Yields:
            Active span covering this runtime boundary.
        """
        tool_execution_count: int = sum(len(turn.tool_call_executions) for turn in request.turns)
        attributes: dict[str, object] = {
            "conversation.id": request.conversation_id,
            "conversation.round_number": request.round_number,
            "conversation.round_status": request.status.name,
            "conversation.turn_count": len(request.turns),
            "conversation.tool_execution_count": tool_execution_count,
            "conversation.reference_count": len(request.references),
            "conversation.answer_chars": len(request.final_answer.content) if request.final_answer is not None else 0,
            "conversation.trace_id": request.trace_id,
        }
        with self._tracer.span("round.persist", attributes) as span:
            span.add_event("round.persistence.started")
            yield span

    @contextmanager
    def trace_round_checkpoint(self, request: CreateConversationRoundCheckpointRequest) -> Generator[Span]:
        """Trace creation of an in-progress Round checkpoint.

        Args:
            request: Validated checkpoint request being traced.

        Yields:
            Active span covering this runtime boundary.
        """
        attributes: dict[str, object] = {
            "conversation.id": request.conversation_id,
            "conversation.round_number": request.round_number,
            "conversation.mutation_id": request.mutation_id,
            "conversation.reference_count": len(request.references),
            "conversation.mcp_server_count": len(request.mcp_server_bindings),
            "conversation.trace_id": request.trace_id,
        }
        with self._tracer.span("round.checkpoint", attributes) as span:
            span.add_event("round.checkpoint.started")
            yield span

    @contextmanager
    def trace_round_progress(self, request: AppendConversationRoundProgressRequest) -> Generator[Span]:
        """Trace one append-only Round progress mutation.

        Args:
            request: Validated append-progress request being traced.

        Yields:
            Active span covering this runtime boundary.
        """
        attributes: dict[str, object] = {
            "conversation.id": request.conversation_id,
            "conversation.round_number": request.round_number,
            "conversation.mutation_id": request.mutation_id,
            "conversation.expected_revision": request.expected_revision,
            "conversation.turn_count": len(request.turns),
            "conversation.dispatch_evidence_count": len(request.dispatch_evidence),
        }
        with self._tracer.span("round.append", attributes) as span:
            span.add_event("round.append.started")
            yield span

    @contextmanager
    def trace_round_finalize(self, request: FinalizeConversationRoundRequest) -> Generator[Span]:
        """Trace the optimistic-lock terminal Round transition.

        Args:
            request: Validated terminal-round request being traced.

        Yields:
            Active span covering this runtime boundary.
        """
        attributes: dict[str, object] = {
            "conversation.id": request.conversation_id,
            "conversation.round_number": request.round_number,
            "conversation.mutation_id": request.mutation_id,
            "conversation.expected_revision": request.expected_revision,
            "conversation.round_status": request.status.name,
        }
        with self._tracer.span("round.finalize", attributes) as span:
            span.add_event("round.finalize.started")
            yield span

    @staticmethod
    def record_round_mutation_result(
        span: Span,
        response: (
            CreateConversationRoundCheckpointResponse
            | AppendConversationRoundProgressResponse
            | FinalizeConversationRoundResponse
        ),
        success_event: str,
    ) -> None:
        """Record a typed checkpoint/progress/finalization result on its owning span.

        Args:
            span: Mutation span receiving outcome attributes and an event.
            response: One of the three Round mutation response types.
            success_event: Event name emitted when the response envelope is successful.
        """
        success: bool = response.base is not None and response.base.success
        span.set_attribute("rpc.success", success)
        span.set_attribute("rpc.code", response.base.code if response.base is not None else -1)

        if response.data is not None:
            span.set_attribute("conversation.committed_revision", response.data.committed_revision)
            span.set_attribute("conversation.idempotent_replay", response.data.idempotent_replay)
            span.set_attribute("conversation.round_status", response.data.status.name)
        span.add_event(success_event if success else "round.mutation.rejected")

    @staticmethod
    def record_round_persistence_result(span: Span, response: SaveConversationRoundResponse) -> None:
        """Record the typed Conversation Manager persistence outcome.

        Args:
            span: OpenTelemetry span receiving bounded evidence.
            response: Provider, RPC, or HTTP response to inspect.
        """
        success: bool = response.base is not None and response.base.success
        span.set_attribute("rpc.success", success)
        span.set_attribute("rpc.code", response.base.code if response.base is not None else -1)
        span.add_event("round.persisted" if success else "round.persistence.rejected")
