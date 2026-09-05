import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.commons import ResponseBase
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    AppendConversationRoundProgressRequest,
    AppendConversationRoundProgressResponse,
    ConversationRoundHistory,
    ConversationRoundMutationResult,
    CreateConversationRoundCheckpointRequest,
    CreateConversationRoundCheckpointResponse,
    FinalizeConversationRoundRequest,
    FinalizeConversationRoundResponse,
    GetConversationRoundHistoryResponse,
    RoundStatus,
    SaveConversationRoundRequest,
    SaveConversationRoundResponse,
)

from agent_runner.agent_definitions.config_models import AgentDefinition, MemoryPolicy
from agent_runner.api.streaming import StreamEvent, StreamEventType
from agent_runner.config import AgentConfig, ConversationRequest, get_settings
from agent_runner.context.builder import AgentContext
from agent_runner.context.models import UserProfile
from agent_runner.observability.runtime_tracing import RuntimeTracing
from agent_runner.observability.tracing import Tracer
from agent_runner.runtime.cancellation import ConversationCancellationRegistry
from agent_runner.runtime.mcp_dispatch import ConversationDispatchRecorder
from agent_runner.runtime.orchestrator import RuntimeOrchestrator, RuntimeRequestState


class FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


class FakeLock:
    def __init__(self) -> None:
        self.released = False

    async def acquire(self, conversation_id: str) -> None:
        assert conversation_id == "conv_persistence"

    async def release(self) -> None:
        self.released = True


class FakeConversationClient:
    def __init__(self, save_success: bool) -> None:
        self.save_success = save_success
        self.saved_request: SaveConversationRoundRequest | None = None

    async def get_round_history(
        self,
        user_id: int,
        conversation_id: str,
    ) -> GetConversationRoundHistoryResponse:
        return GetConversationRoundHistoryResponse(
            base=ResponseBase(code=0, success=True),
            data=ConversationRoundHistory(conversation_id=conversation_id, latest_round_number=0),
        )

    async def save_round(self, request: SaveConversationRoundRequest) -> SaveConversationRoundResponse:
        self.saved_request = request

        return SaveConversationRoundResponse(
            base=ResponseBase(
                code=0 if self.save_success else 500,
                success=self.save_success,
                message="" if self.save_success else "save failed",
            )
        )


class CheckpointConversationClient(FakeConversationClient):
    def __init__(self) -> None:
        super().__init__(save_success=True)
        self.finalize_request: FinalizeConversationRoundRequest | None = None

    async def create_round_checkpoint(
        self, request: CreateConversationRoundCheckpointRequest
    ) -> CreateConversationRoundCheckpointResponse:
        return CreateConversationRoundCheckpointResponse(
            base=ResponseBase(code=0, success=True),
            data=ConversationRoundMutationResult(committed_revision=0),
        )

    async def finalize_round(self, request: FinalizeConversationRoundRequest) -> FinalizeConversationRoundResponse:
        self.finalize_request = request

        return FinalizeConversationRoundResponse(
            base=ResponseBase(code=0, success=True),
            data=ConversationRoundMutationResult(committed_revision=1),
        )


class ConcurrentRevisionConversationClient(CheckpointConversationClient):
    """Hold a dispatch mutation open so cancellation competes for the same revision."""

    def __init__(self) -> None:
        super().__init__()
        self.dispatch_started = asyncio.Event()
        self.release_dispatch = asyncio.Event()

    async def append_round_progress(
        self, request: AppendConversationRoundProgressRequest
    ) -> AppendConversationRoundProgressResponse:
        assert request.expected_revision == 0
        self.dispatch_started.set()
        await self.release_dispatch.wait()

        return AppendConversationRoundProgressResponse(
            base=ResponseBase(code=0, success=True),
            data=ConversationRoundMutationResult(committed_revision=1),
        )

    async def finalize_round(self, request: FinalizeConversationRoundRequest) -> FinalizeConversationRoundResponse:
        self.finalize_request = request
        success = request.expected_revision == 1

        return FinalizeConversationRoundResponse(
            base=ResponseBase(code=0 if success else 409, success=success, message="" if success else "stale revision"),
            data=ConversationRoundMutationResult(committed_revision=2) if success else None,
        )


class FakeConfigLoader:
    async def load(self, agent_id: int) -> AgentConfig:
        return AgentConfig(
            agent_id=agent_id,
            version=1,
            name="general-assistant",
            model="test-model",
            system_prompt="system prompt",
            tools=[],
            mcp_servers=[],
            max_output_tokens=64,
            temperature=0.2,
        )


class FakeContextBuilder:
    async def build(self, **kwargs: Any) -> AgentContext:
        return AgentContext(
            agent_config=kwargs["agent_config"],
            system_prompt="system prompt",
            conversation_history=[],
            user_profile=UserProfile(),
            rag_chunks=[],
            current_message=kwargs["current_message"],
            tool_specs=(),
        )


class FakeAgentFactory:
    async def create(self, config: AgentConfig) -> AgentDefinition:
        return AgentDefinition(
            agent_id=config.agent_id,
            version=config.version,
            name=config.name,
            description="test",
            model=config.model,
            system_prompt=config.system_prompt,
            tools=[],
            mcp_servers=[],
            memory_policy=MemoryPolicy(profile=False, rag=False),
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )


class FakeRuntime:
    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        token: "FakeToken",
    ) -> AsyncGenerator[dict[str, object]]:
        yield {"type": "token_delta", "content": "Persisted "}
        yield {"type": "token_delta", "content": "answer"}
        yield {"type": "usage", "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}


class BlockingRuntime:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        token: "FakeToken",
    ) -> AsyncGenerator[dict[str, object]]:
        self.started.set()
        await asyncio.Event().wait()
        yield {"type": "token_delta", "content": "unreachable"}


class FakeToken:
    def is_cancelled(self) -> bool:
        return False

    def cancel(self) -> None:
        pass


class FakeCancellationManager:
    def create_token(self) -> FakeToken:
        return FakeToken()

    async def cleanup(self, token: FakeToken) -> None:
        pass


async def test_success_reports_done_only_after_persistence() -> None:
    orchestrator, client, lock = _orchestrator(save_success=True)

    events = [
        event
        async for event in orchestrator.run(
            ConversationRequest(conversation_id="conv_persistence", message="Question"), 1, FakeRequest()
        )
    ]

    assert [event.type for event in events] == [
        StreamEventType.TOKEN_DELTA,
        StreamEventType.TOKEN_DELTA,
        StreamEventType.USAGE,
        StreamEventType.SAVING,
        StreamEventType.PERSISTED,
        StreamEventType.DONE,
    ]
    assert client.saved_request is not None
    assert client.saved_request.final_answer is not None
    assert client.saved_request.final_answer.content == "Persisted answer"
    assert client.saved_request.user_id == 1
    assert client.saved_request.conversation_id == "conv_persistence"
    assert lock.released is True


async def test_persistence_failure_never_reports_persisted_or_done() -> None:
    orchestrator, _, lock = _orchestrator(save_success=False)

    events: list[StreamEvent] = [
        event
        async for event in orchestrator.run(
            ConversationRequest(conversation_id="conv_persistence", message="Question"), 1, FakeRequest()
        )
    ]

    assert events[-2].type == StreamEventType.SAVING
    assert events[-1].type == StreamEventType.ERROR
    assert events[-1].error_code == "PERSISTENCE_FAILED"
    assert StreamEventType.PERSISTED not in [event.type for event in events]
    assert StreamEventType.DONE not in [event.type for event in events]
    assert lock.released is True


async def test_task_cancellation_finalizes_an_existing_checkpoint() -> None:
    orchestrator, _, _ = _orchestrator(save_success=True)
    runtime = BlockingRuntime()
    client = CheckpointConversationClient()
    harness = cast(Any, orchestrator)
    harness.openai_runtime = runtime
    harness.conversation_client = client

    async def consume() -> None:
        async for _ in orchestrator.run(
            ConversationRequest(conversation_id="conv_persistence", message="Question"), 1, FakeRequest()
        ):
            pass

    task = asyncio.create_task(consume())
    await runtime.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.finalize_request is not None
    assert client.finalize_request.status.name == "CANCELLED"


async def test_cancellation_waits_for_an_in_flight_dispatch_revision() -> None:
    orchestrator, _, _ = _orchestrator(save_success=True)
    client = ConcurrentRevisionConversationClient()
    harness = cast(Any, orchestrator)
    harness.conversation_client = client
    state = RuntimeRequestState(
        cancellation_token=cast(Any, FakeToken()),
        round_start=1,
        attachment_request_id="attachment-request",
        next_round_number=1,
        preflight_completed=True,
        checkpoint_created=True,
    )
    recorder = ConversationDispatchRecorder(client, state, 1, "conv_persistence", 1)  # type: ignore[arg-type]
    dispatch_task = asyncio.create_task(
        recorder.before_dispatch("attempt-1", "call-1", 1, "fixture", "write", {"value": "x"})
    )
    await client.dispatch_started.wait()
    dispatch_task.cancel()

    cancellation_task = asyncio.create_task(
        orchestrator._persist_terminal_round(
            1,
            ConversationRequest(conversation_id="conv_persistence", message="Question"),
            1,
            1,
            RoundStatus.CANCELLED,
            "Generation cancelled.",
            state=state,
        )
    )
    await asyncio.sleep(0)
    assert client.finalize_request is None

    client.release_dispatch.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch_task
    await cancellation_task

    assert client.finalize_request is not None
    assert client.finalize_request.expected_revision == 1
    assert client.finalize_request.status.name == "CANCELLED"


def test_public_request_forbids_user_and_agent_identity() -> None:
    fields = {"conversation_id": "conv_persistence", "message": "Question", "user_id": 1, "agent_id": 1}
    try:
        ConversationRequest.model_validate(fields)
    except ValueError:
        return

    raise AssertionError("Identity fields must not be accepted from the public request body.")


def test_public_request_rejects_blank_messages() -> None:
    try:
        ConversationRequest(conversation_id="conv_persistence", message="   ")
    except ValueError:
        return

    raise AssertionError("Blank messages must be rejected before model execution.")


def _orchestrator(save_success: bool) -> tuple[RuntimeOrchestrator, FakeConversationClient, FakeLock]:
    orchestrator = object.__new__(RuntimeOrchestrator)
    harness = cast(Any, orchestrator)
    client = FakeConversationClient(save_success)
    lock = FakeLock()
    harness.config_loader = FakeConfigLoader()
    harness.context_builder = FakeContextBuilder()
    harness.agent_factory = FakeAgentFactory()
    harness.openai_runtime = FakeRuntime()
    harness.cancellation_manager = FakeCancellationManager()
    harness.conversation_client = client
    harness.execution_lock = lock
    harness.settings = get_settings()
    harness.runtime_tracing = RuntimeTracing(Tracer())
    harness.cancellation_registry = ConversationCancellationRegistry()
    harness._lock_acquired = False
    harness._terminal_round_persisted = False

    return orchestrator, client, lock
