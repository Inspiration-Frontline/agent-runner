import asyncio
import json
from dataclasses import dataclass
from time import time_ns
from typing import Any, Protocol
from uuid import uuid4

from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    AppendConversationRoundProgressRequest,
    ToolDispatchEvidence,
    ToolDispatchState,
)
from opentelemetry import trace

from agent_runner.conversation.client import ConversationManagerClient
from agent_runner.mcps.sdk_runtime import DispatchEvidenceRecorder
from agent_runner.observability.tracing import current_trace_id


class RevisionState(Protocol):
    """Mutable Round revision exposed by orchestration to serialized dispatch persistence."""

    checkpoint_revision: int


@dataclass
class PendingDispatch:
    """Request-local copy of the evidence needed to finish one remote delivery attempt."""

    attempt_id: str
    tool_call_id: str
    turn_number: int
    server_id: str
    tool_name: str
    arguments_json: str
    dispatch_time: int
    trace_id: str
    span_id: str


class ConversationDispatchRecorder(DispatchEvidenceRecorder):
    """Persists before/after evidence for every model-selected remote MCP Tool delivery.

    A dispatch is the delivery attempt between the model deciding to call an MCP Tool and the
    Runner receiving a terminal transport result. The recorder commits ``DISPATCHING`` before the
    network call so process loss cannot make a possibly executed side effect disappear from audit
    history. Terminal evidence is serialized through the same Round revision lock afterward.
    """

    def __init__(
        self,
        client: ConversationManagerClient,
        state: RevisionState,
        user_id: int,
        conversation_id: str,
        round_number: int,
    ) -> None:
        """Create a recorder bound to one user-owned Conversation Round and revision state."""
        self._client = client
        self._state = state
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._round_number = round_number
        self._lock = asyncio.Lock()
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
        """Persist redacted DISPATCHING intent before the MCP transport sends request bytes."""
        span_context = trace.get_current_span().get_span_context()
        pending = PendingDispatch(
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
        """Complete one pending attempt with its terminal delivery state and recovery context."""
        pending = self._pending.pop(attempt_id)
        await self._append(pending, ToolDispatchState[state], recovery_reason)

    async def _append(
        self,
        pending: PendingDispatch,
        state: ToolDispatchState,
        recovery_reason: str = "",
    ) -> None:
        """Append one evidence mutation while serializing access to the shared Round revision."""
        async with self._lock:
            response = await self._client.append_round_progress(AppendConversationRoundProgressRequest(
                user_id=self._user_id,
                conversation_id=self._conversation_id,
                round_number=self._round_number,
                mutation_id=str(uuid4()),
                expected_revision=self._state.checkpoint_revision,
                dispatch_evidence=[ToolDispatchEvidence(
                    attempt_id=pending.attempt_id,
                    turn_number=pending.turn_number,
                    tool_call_id=pending.tool_call_id,
                    tool_name=pending.tool_name,
                    tool_key=f"mcp.{pending.server_id}.{pending.tool_name}",
                    server_id=pending.server_id,
                    arguments_json=pending.arguments_json,
                    state=state,
                    dispatch_time=pending.dispatch_time,
                    result_time=time_ns() // 1_000_000 if state not in {ToolDispatchState.READY, ToolDispatchState.DISPATCHING} else 0,
                    trace_id=pending.trace_id,
                    span_id=pending.span_id,
                    transport_evidence="streamable-http",
                    recovery_reason=recovery_reason[:2000],
                )],
            ))
            if response.base is None or not response.base.success or response.data is None:
                message = response.base.message if response.base is not None else "Dispatch checkpoint failed."
                raise RuntimeError(message)
            self._state.checkpoint_revision = response.data.committed_revision

    @classmethod
    def _redact(cls, value: Any) -> Any:
        """Recursively replace values whose object keys indicate authentication material."""
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if cls._is_secret_key(str(key)) else cls._redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value

    @staticmethod
    def _is_secret_key(key: str) -> bool:
        """Return whether a case-insensitive argument key commonly carries a credential."""
        lowered = key.lower()
        return any(marker in lowered for marker in ("authorization", "password", "secret", "token", "api_key"))
