import asyncio
import json
from dataclasses import dataclass
from time import time_ns
from typing import Any, Protocol
from uuid import uuid4

from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    AppendConversationRoundProgressRequest,
    AppendConversationRoundProgressResponse,
    ToolDispatchEvidence,
    ToolDispatchState,
)
from opentelemetry import trace
from opentelemetry.trace import SpanContext

from agent_runner.conversation.client import ConversationManagerClient
from agent_runner.mcps.sdk_runtime import DispatchEvidenceRecorder
from agent_runner.observability.tracing import current_trace_id


class RevisionState(Protocol):
    """Mutable Round revision exposed by orchestration to serialized dispatch persistence."""

    checkpoint_revision: int
    """Latest durable Round revision observed by the orchestrator."""
    revision_lock: asyncio.Lock
    """Lock serializing dispatch evidence mutations for this Round."""


@dataclass
class PendingDispatch:
    """Request-local copy of the evidence needed to finish one remote delivery attempt."""

    attempt_id: str
    """Stable identifier of the current remote delivery attempt."""
    tool_call_id: str
    """Provider-generated Tool Call identifier."""
    turn_number: int
    """One-based model Turn containing this dispatch."""
    server_id: str
    """Stable MCP Server catalog identifier."""
    tool_name: str
    """MCP Tool name selected by the model."""
    arguments_json: str
    """Exact JSON arguments emitted by the model."""
    dispatch_time: int
    """Epoch milliseconds at which dispatch evidence was created."""
    trace_id: str
    """W3C trace ID correlating this evidence with the request."""
    span_id: str
    """Span ID of the MCP call that owns this dispatch."""


class ConversationDispatchRecorder(DispatchEvidenceRecorder):
    """Persists before/after evidence for every model-selected remote MCP Tool delivery.

    A dispatch is the delivery attempt between the model deciding to call an MCP Tool and the
    Runner receiving a terminal transport result. The recorder commits ``DISPATCHING`` before the
    network call so process loss cannot make a possibly executed side effect disappear from audit
    history. Terminal evidence is serialized through the same Round revision lock afterward.

    Attributes:
        _client: Asynchronous client used for the external boundary call.
        _state: Current durable dispatch state.
        _user_id: Stable identifier of the  user.
        _conversation_id: Stable identifier of the  conversation.
        _round_number: Conversation Round number containing this dispatch.
        _pending: Dispatch evidence awaiting a terminal remote-delivery outcome.
    """

    def __init__(
        self,
        client: ConversationManagerClient,
        state: RevisionState,
        user_id: int,
        conversation_id: str,
        round_number: int,
    ) -> None:
        """Create a recorder bound to one user-owned Conversation Round and revision state.

        Args:
            client: Asynchronous client used for the external boundary call.
            state: Request-scoped mutable orchestration or persistence state.
            user_id: Trusted authenticated user identifier.
            conversation_id: Stable public identifier of the Conversation.
            round_number: Positive one-based Round boundary within the Conversation.
        """
        self._client = client
        self._state = state
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._round_number = round_number
        # Key: remote attempt ID. Value: DISPATCHING evidence awaiting its terminal outcome.
        self._pending: dict[str, PendingDispatch] = {}

    async def before_dispatch(
        self,
        attempt_id: str,
        tool_call_id: str,
        turn_number: int,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Persist redacted DISPATCHING intent before the MCP transport sends request bytes.

        Args:
            attempt_id: Unique identifier of the remote delivery attempt.
            tool_call_id: Provider-generated Tool call identifier.
            turn_number: Positive one-based model Turn number within the Round.
            server_id: Stable MCP Catalog server identifier.
            tool_name: Provider-visible Tool name.
            arguments: Validated Tool arguments supplied by the model.
        """
        span_context: SpanContext = trace.get_current_span().get_span_context()
        pending: PendingDispatch = PendingDispatch(
            attempt_id=attempt_id,
            tool_call_id=tool_call_id,
            turn_number=turn_number,
            server_id=server_id,
            tool_name=tool_name,
            arguments_json=json.dumps(self._redact(arguments), ensure_ascii=False, separators=(",", ":")),
            dispatch_time=time_ns() // 1_000_000,
            trace_id=current_trace_id(),
            span_id=f"{span_context.span_id:016x}" if span_context.is_valid else "0" * 16,
        )
        self._pending[attempt_id] = pending
        await self._append(pending, ToolDispatchState.DISPATCHING)

    async def after_dispatch(self, attempt_id: str, state: str, recovery_reason: str = "") -> None:
        """Complete one pending attempt with its terminal delivery state and recovery context.

        Args:
            attempt_id: Unique identifier of the remote delivery attempt.
            state: Request-scoped mutable orchestration or persistence state.
            recovery_reason: Bounded explanation used for recovery and audit evidence.
        """
        pending: PendingDispatch = self._pending.pop(attempt_id)
        await self._append(pending, ToolDispatchState[state], recovery_reason)

    async def _append(
        self,
        pending: PendingDispatch,
        state: ToolDispatchState,
        recovery_reason: str = "",
    ) -> None:
        """Append one evidence mutation without losing its committed revision to cancellation.

        Args:
            pending: Dispatch evidence awaiting a terminal remote-delivery outcome.
            state: Request-scoped mutable orchestration or persistence state.
            recovery_reason: Bounded explanation used for recovery and audit evidence.
        """
        async with self._state.revision_lock:
            request: AppendConversationRoundProgressRequest = AppendConversationRoundProgressRequest(
                user_id=self._user_id,
                conversation_id=self._conversation_id,
                round_number=self._round_number,
                mutation_id=str(uuid4()),
                expected_revision=self._state.checkpoint_revision,
                dispatch_evidence=[
                    ToolDispatchEvidence(
                        attempt_id=pending.attempt_id,
                        turn_number=pending.turn_number,
                        tool_call_id=pending.tool_call_id,
                        tool_name=pending.tool_name,
                        tool_key=f"mcp.{pending.server_id}.{pending.tool_name}",
                        server_id=pending.server_id,
                        arguments_json=pending.arguments_json,
                        state=state,
                        dispatch_time=pending.dispatch_time,
                        result_time=time_ns() // 1_000_000
                        if state not in {ToolDispatchState.READY, ToolDispatchState.DISPATCHING}
                        else 0,
                        trace_id=pending.trace_id,
                        span_id=pending.span_id,
                        transport_evidence="streamable-http",
                        recovery_reason=recovery_reason[:2000],
                    )
                ],
            )
            append_task: asyncio.Task[AppendConversationRoundProgressResponse] = asyncio.create_task(
                self._client.append_round_progress(request)
            )
            cancellation_received: bool = False
            while not append_task.done():
                try:
                    await asyncio.shield(append_task)
                except asyncio.CancelledError:
                    if append_task.cancelled():
                        raise
                    cancellation_received = True
            response: AppendConversationRoundProgressResponse = await append_task
            if response.base is None or not response.base.success or response.data is None:
                message: str = response.base.message if response.base is not None else "Dispatch checkpoint failed."
                raise RuntimeError(message)
            self._state.checkpoint_revision = response.data.committed_revision
        if cancellation_received:
            raise asyncio.CancelledError

    @classmethod
    def _redact(cls, value: Any) -> Any:
        """Recursively replace values whose object keys indicate authentication material.

        Args:
            value: Candidate value to validate, normalize, or serialize.

        Returns:
            Redacted value with authentication-bearing keys replaced recursively.
        """
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if cls._is_secret_key(str(key)) else cls._redact(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value

    @staticmethod
    def _is_secret_key(key: str) -> bool:
        """Return whether a case-insensitive argument key commonly carries a credential.

        Args:
            key: Credential-isolated identity of the target MCP server.

        Returns:
            ``True`` when the argument key commonly carries a credential.
        """
        lowered: str = key.lower()
        return any(marker in lowered for marker in ("authorization", "password", "secret", "token", "api_key"))
