from types import SimpleNamespace
from typing import Any

import pytest
from agents import Runner, function_tool

from agent_runner.agent_definitions.config_models import AgentDefinition, MemoryPolicy
from agent_runner.api.streaming import UsageEvent
from agent_runner.config import AgentConfig
from agent_runner.context.builder import AgentContext, Message, RuntimeToolCall
from agent_runner.context.models import UserProfile
from agent_runner.runtime.openai_agents_sdk_adapter import OpenAIAgentsSdkAdapter
from agent_runner.runtime.orchestrator import RuntimeOrchestrator
from agent_runner.tools.registry import ToolDefinition, ToolRegistry


class DummyModelFactory:
    def __init__(self) -> None:
        self.created_models: list[str] = []

    def create_model(self, model: str) -> str:
        self.created_models.append(model)
        return f"model:{model}"

    async def close(self) -> None:
        pass


def test_build_input_preserves_history_and_current_message() -> None:
    runtime = OpenAIAgentsSdkAdapter(model_factory=DummyModelFactory())
    context = AgentContext(
        agent_config=_agent_config(),
        system_prompt="system",
        conversation_history=[
            Message(role="user", content="previous user"),
            Message(role="assistant", content="previous assistant"),
        ],
        user_profile=UserProfile(),
        rag_chunks=[],
        current_message=Message(role="user", content="current user"),
        tool_specs=(),
    )

    assert runtime._build_input(context) == [
        {"role": "user", "content": "previous user"},
        {"role": "assistant", "content": "previous assistant"},
        {"role": "user", "content": "current user"},
    ]


def test_build_input_preserves_plain_text_when_multipart_content_is_empty() -> None:
    runtime = OpenAIAgentsSdkAdapter(model_factory=DummyModelFactory())
    context = AgentContext(
        agent_config=_agent_config(),
        system_prompt="system",
        conversation_history=[Message(role="assistant", content="Previous answer")],
        user_profile=UserProfile(),
        rag_chunks=[],
        current_message=Message(role="user", content="Expand on that answer"),
        tool_specs=(),
    )

    assert runtime._build_input(context)[-1] == {
        "role": "user",
        "content": "Expand on that answer",
    }


def test_build_input_converts_provider_neutral_tool_history_to_responses_items() -> None:
    runtime = OpenAIAgentsSdkAdapter(model_factory=DummyModelFactory())
    context = AgentContext(
        agent_config=_agent_config(),
        system_prompt="system",
        conversation_history=[
            Message(
                role="assistant",
                content="",
                tool_calls=(
                    RuntimeToolCall(
                        call_id="call-1",
                        call_type="function",
                        function_name="calculate_expression",
                        arguments='{"expression":"6*7"}',
                    ),
                ),
            ),
            Message(role="tool", content='{"result":42}', tool_call_id="call-1"),
            Message(role="assistant", content="The result was 42."),
        ],
        user_profile=UserProfile(),
        rag_chunks=[],
        current_message=Message(role="user", content="What was it?"),
        tool_specs=(),
    )

    assert runtime._build_input(context) == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "calculate_expression",
            "arguments": '{"expression":"6*7"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": '{"result":42}'},
        {"role": "assistant", "content": "The result was 42."},
        {"role": "user", "content": "What was it?"},
    ]


def test_build_input_rejects_tool_result_without_call_id() -> None:
    runtime = OpenAIAgentsSdkAdapter(model_factory=DummyModelFactory())
    context = AgentContext(
        agent_config=_agent_config(),
        system_prompt="system",
        conversation_history=[Message(role="tool", content='{"result":42}')],
        user_profile=UserProfile(),
        rag_chunks=[],
        current_message=Message(role="user", content="continue"),
        tool_specs=(),
    )

    with pytest.raises(ValueError, match="non-empty tool_call_id"):
        runtime._build_input(context)


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
def test_sdk_message_role_accepts_supported_roles(role: str) -> None:
    assert OpenAIAgentsSdkAdapter._get_sdk_message_role(role) == role


def test_sdk_message_role_rejects_unsupported_role() -> None:
    with pytest.raises(ValueError, match="Unsupported OpenAI input message role"):
        OpenAIAgentsSdkAdapter._get_sdk_message_role("tool")


@pytest.mark.parametrize(
    ("detail", "expected"),
    [("low", "low"), ("high", "high"), ("auto", "auto"), ("original", "original"), ("", "auto")],
)
def test_orchestrator_normalizes_image_detail(detail: str, expected: str) -> None:
    assert RuntimeOrchestrator._get_image_detail(detail) == expected


def test_build_sdk_agent_uses_agents_sdk_model() -> None:
    model_factory = DummyModelFactory()
    runtime = OpenAIAgentsSdkAdapter(model_factory=model_factory)

    sdk_agent = runtime._build_sdk_agent(_agent(), "system prompt")

    assert sdk_agent.name == "Smoke"
    assert sdk_agent.instructions == "system prompt"
    assert sdk_agent.model == "model:Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert sdk_agent.model_settings.temperature == 0.3
    assert sdk_agent.model_settings.max_tokens == 256
    assert sdk_agent.model_settings.include_usage is True
    assert sdk_agent.model_settings.extra_args is not None
    assert sdk_agent.model_settings.extra_args["timeout"] == 120.0
    assert model_factory.created_models == ["Qwen/Qwen3-235B-A22B-Instruct-2507"]


def test_response_completed_event_usage_is_passed_through() -> None:
    runtime = OpenAIAgentsSdkAdapter(model_factory=DummyModelFactory())
    event = SimpleNamespace(
        data=SimpleNamespace(
            response=SimpleNamespace(usage=SimpleNamespace(input_tokens=12, output_tokens=5, total_tokens=17))
        )
    )

    usage = runtime._convert_response_completed_usage(event.data)
    assert usage is not None
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 17


def test_response_completed_event_without_usage_is_ignored() -> None:
    runtime = OpenAIAgentsSdkAdapter(model_factory=DummyModelFactory())
    event = SimpleNamespace(data=SimpleNamespace(response=SimpleNamespace(usage=None)))

    assert runtime._convert_response_completed_usage(event.data) is None


def test_orchestrator_converts_usage_event_without_estimation() -> None:
    orchestrator = object.__new__(RuntimeOrchestrator)
    event = orchestrator._convert_event(
        {
            "type": "usage",
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
        }
    )

    assert isinstance(event, UsageEvent)
    assert event.prompt_tokens == 12
    assert event.completion_tokens == 5
    assert event.total_tokens == 17


def _agent() -> AgentDefinition:
    return AgentDefinition(
        agent_id=1,
        version=1,
        name="Smoke",
        description="Smoke test agent",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        system_prompt="system",
        tools=[],
        mcp_servers=[],
        memory_policy=MemoryPolicy(profile=False, rag=False),
        max_output_tokens=256,
        temperature=0.3,
    )


def _agent_config() -> AgentConfig:
    return AgentConfig(
        agent_id=1,
        version=1,
        name="Smoke",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        system_prompt="system",
        tools=[],
        mcp_servers=[],
        max_output_tokens=256,
        temperature=0.3,
    )


@pytest.mark.asyncio
async def test_run_registers_configured_tools_on_sdk_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    @function_tool
    def echo_value(value: str) -> str:
        """Return the supplied value."""
        return value

    registry = ToolRegistry()
    registry.register(ToolDefinition.from_function_tool("test.echo", echo_value))
    agent = _agent()
    agent.tools = ["test.echo"]
    context = AgentContext(
        agent_config=_agent_config(),
        system_prompt="system",
        conversation_history=[],
        user_profile=UserProfile(),
        rag_chunks=[],
        current_message=Message(role="user", content="echo this"),
        tool_specs=(),
    )
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(final_output="done")

    monkeypatch.setattr(Runner, "run", fake_run)

    response = await OpenAIAgentsSdkAdapter(model_factory=DummyModelFactory()).run(
        agent,
        context,
        tool_registry=registry,
    )

    sdk_agent = captured["starting_agent"]
    assert [tool.name for tool in sdk_agent.tools] == ["echo_value"]
    assert captured["input"] == [{"role": "user", "content": "echo this"}]
    assert response.content == "done"
