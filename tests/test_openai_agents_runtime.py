from types import SimpleNamespace

from agent_definitions.config_models import AgentDefinition, MemoryPolicy
from api.streaming import UsageEvent
from context.builder import AgentContext, Message
from runtime.openai_agents_runtime import OpenAIAgentsRuntime
from runtime.orchestrator import RuntimeOrchestrator


class DummyModelFactory:
    def __init__(self):
        self.created_models: list[str] = []

    def create_model(self, model: str) -> str:
        self.created_models.append(model)
        return f"model:{model}"


def test_build_input_preserves_history_and_current_message():
    runtime = OpenAIAgentsRuntime(model_factory=DummyModelFactory())
    context = AgentContext(
        agent_config=_agent(),
        system_prompt="system",
        conversation_history=[
            Message(role="user", content="previous user"),
            Message(role="assistant", content="previous assistant"),
        ],
        user_profile={},
        rag_chunks=[],
        current_message=Message(role="user", content="current user"),
        tool_specs=[],
    )

    assert runtime._build_input(context) == [
        {"role": "user", "content": "previous user"},
        {"role": "assistant", "content": "previous assistant"},
        {"role": "user", "content": "current user"},
    ]


def test_build_sdk_agent_uses_agents_sdk_model():
    model_factory = DummyModelFactory()
    runtime = OpenAIAgentsRuntime(model_factory=model_factory)

    sdk_agent = runtime._build_sdk_agent(_agent(), "system prompt")

    assert sdk_agent.name == "Smoke"
    assert sdk_agent.instructions == "system prompt"
    assert sdk_agent.model == "model:Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert sdk_agent.model_settings.temperature == 0.3
    assert sdk_agent.model_settings.max_tokens == 256
    assert sdk_agent.model_settings.include_usage is True
    assert sdk_agent.model_settings.extra_args["timeout"] == 120.0
    assert model_factory.created_models == ["Qwen/Qwen3-235B-A22B-Instruct-2507"]


def test_response_completed_event_usage_is_passed_through():
    runtime = OpenAIAgentsRuntime(model_factory=DummyModelFactory())
    event = SimpleNamespace(
        data=SimpleNamespace(
            response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=12, output_tokens=5, total_tokens=17)
            )
        )
    )

    assert runtime._convert_response_completed_usage(event.data) == {
        "type": "usage",
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }


def test_response_completed_event_without_usage_is_ignored():
    runtime = OpenAIAgentsRuntime(model_factory=DummyModelFactory())
    event = SimpleNamespace(data=SimpleNamespace(response=SimpleNamespace(usage=None)))

    assert runtime._convert_response_completed_usage(event.data) is None


def test_orchestrator_converts_usage_event_without_estimation():
    orchestrator = object.__new__(RuntimeOrchestrator)
    event = orchestrator._convert_event({
        "type": "usage",
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    })

    assert isinstance(event, UsageEvent)
    assert event.prompt_tokens == 12
    assert event.completion_tokens == 5
    assert event.total_tokens == 17


def _agent() -> AgentDefinition:
    return AgentDefinition(
        agent_id="smoke-test",
        version="local",
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
