from collections.abc import AsyncGenerator

from agent_breaker_conversation_manager_protos.ifl.agentbreaker.commons import ResponseBase
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    ConversationRoundHistory,
    GetConversationRoundHistoryResponse,
    SaveConversationRoundResponse,
)

from agent_runner.agent_definitions.config_models import AgentDefinition, MemoryPolicy
from agent_runner.api.streaming import StreamEvent, StreamEventType
from agent_runner.config import AgentConfig, ChatRequest
from agent_runner.context.builder import AgentContext, Message
from agent_runner.runtime.orchestrator import RuntimeOrchestrator


class FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


class FakeLock:
    def __init__(self) -> None:
        self.released = False

    async def acquire(self, conversation_id: str) -> None:
        assert conversation_id == "conv_phase3"

    async def release(self) -> None:
        self.released = True


class FakeConversationClient:
    def __init__(self, save_success: bool) -> None:
        self.save_success = save_success
        self.saved_request = None

    async def get_round_history(self, user_id: int, conversation_id: str):
        return GetConversationRoundHistoryResponse(
            base=ResponseBase(code=0, success=True),
            data=ConversationRoundHistory(conversation_id=conversation_id, latest_round_number=0),
        )

    async def save_round(self, request):
        self.saved_request = request
        return SaveConversationRoundResponse(
            base=ResponseBase(
                code=0 if self.save_success else 500,
                success=self.save_success,
                message="" if self.save_success else "save failed",
            )
        )


class FakeConfigLoader:
    async def load(self, agent_id: int):
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
    async def build(self, **kwargs):
        return AgentContext(
            agent_config=kwargs["agent_config"],
            system_prompt="system prompt",
            conversation_history=[],
            user_profile={},
            rag_chunks=[],
            current_message=Message(role="user", content=kwargs["current_message"]),
            tool_specs=[],
        )


class FakeAgentFactory:
    async def create(self, config: AgentConfig):
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
    async def run_streamed(self, agent, context, token) -> AsyncGenerator[dict]:
        yield {"type": "token_delta", "content": "Persisted "}
        yield {"type": "token_delta", "content": "answer"}
        yield {"type": "usage", "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}


class FakeToken:
    def is_cancelled(self) -> bool:
        return False

    def cancel(self) -> None:
        pass


class FakeCancellationManager:
    def create_token(self):
        return FakeToken()

    async def cleanup(self, token) -> None:
        pass


async def test_success_reports_done_only_after_persistence():
    orchestrator, client, lock = _orchestrator(save_success=True)

    events = [event async for event in orchestrator.run(
        ChatRequest(conversation_id="conv_phase3", message="Question"), 1, FakeRequest()
    )]

    assert [event.type for event in events] == [
        StreamEventType.TOKEN_DELTA,
        StreamEventType.TOKEN_DELTA,
        StreamEventType.USAGE,
        StreamEventType.SAVING,
        StreamEventType.PERSISTED,
        StreamEventType.DONE,
    ]
    assert client.saved_request.final_answer.content == "Persisted answer"
    assert client.saved_request.user_id == 1
    assert client.saved_request.conversation_id == "conv_phase3"
    assert lock.released is True


async def test_persistence_failure_never_reports_persisted_or_done():
    orchestrator, _, lock = _orchestrator(save_success=False)

    events: list[StreamEvent] = [event async for event in orchestrator.run(
        ChatRequest(conversation_id="conv_phase3", message="Question"), 1, FakeRequest()
    )]

    assert events[-2].type == StreamEventType.SAVING
    assert events[-1].type == StreamEventType.ERROR
    assert events[-1].error_code == "PERSISTENCE_FAILED"
    assert StreamEventType.PERSISTED not in [event.type for event in events]
    assert StreamEventType.DONE not in [event.type for event in events]
    assert lock.released is True


def test_public_request_forbids_user_and_agent_identity():
    fields = {"conversation_id": "conv_phase3", "message": "Question", "user_id": 1, "agent_id": 1}
    try:
        ChatRequest.model_validate(fields)
    except ValueError:
        return
    raise AssertionError("Identity fields must not be accepted from the public request body.")


def test_public_request_rejects_blank_messages():
    try:
        ChatRequest(conversation_id="conv_phase3", message="   ")
    except ValueError:
        return
    raise AssertionError("Blank messages must be rejected before model execution.")


def _orchestrator(save_success: bool):
    orchestrator = object.__new__(RuntimeOrchestrator)
    client = FakeConversationClient(save_success)
    lock = FakeLock()
    orchestrator.config_loader = FakeConfigLoader()
    orchestrator.context_builder = FakeContextBuilder()
    orchestrator.agent_factory = FakeAgentFactory()
    orchestrator.openai_runtime = FakeRuntime()
    orchestrator.cancellation_manager = FakeCancellationManager()
    orchestrator.conversation_client = client
    orchestrator.execution_lock = lock
    return orchestrator, client, lock
