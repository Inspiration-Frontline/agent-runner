from typing import Any

import pytest
from agents import Model, ModelSettings

from agent_runner.gateway.litellm_client import LiteLLMModelFactory
from agent_runner.observability.litellm_tracing import TracedModel


def test_bare_proxy_model_uses_openai_provider_prefix() -> None:
    factory = LiteLLMModelFactory(
        base_url="http://localhost:4000",
        api_key="sk-test",
        request_timeout_seconds=3,
    )

    assert factory._normalize_model("Qwen/Qwen3-4B") == "openai/Qwen/Qwen3-4B"


def test_known_provider_model_is_left_unchanged() -> None:
    factory = LiteLLMModelFactory(
        base_url="http://localhost:4000",
        api_key="sk-test",
        request_timeout_seconds=3,
    )

    assert factory._normalize_model("anthropic/claude-sonnet-4-5") == "anthropic/claude-sonnet-4-5"


def test_created_model_targets_external_proxy() -> None:
    factory = LiteLLMModelFactory(
        base_url="http://localhost:4000",
        api_key="sk-test",
        request_timeout_seconds=3,
    )

    model = factory.create_model("Qwen/Qwen3-4B")

    assert isinstance(model, Model)
    assert model.model == "openai/Qwen/Qwen3-4B"
    assert model.base_url == "http://localhost:4000"
    assert model.api_key == "sk-test"
    assert factory.request_timeout_seconds == 3


def test_token_limit_takes_precedence_over_provider_stop_reason() -> None:
    response = type("Response", (), {"usage": type("Usage", (), {"output_tokens": 512})(), "output": []})()

    assert TracedModel._get_finish_reason(response, ModelSettings(max_tokens=512)) == "length"


class HeaderCapturingModel:
    def __init__(self) -> None:
        self.model_settings: ModelSettings | None = None

    async def get_response(self, *args: Any, **kwargs: Any) -> object:
        self.model_settings = kwargs.get("model_settings") or args[2]
        return object()


@pytest.mark.asyncio
async def test_traced_model_injects_current_llm_span_context() -> None:
    delegate = HeaderCapturingModel()
    model = TracedModel(delegate, "openai/test-model")  # type: ignore[arg-type]

    with model._tracer.span("agent.run") as agent_span:
        await model.get_response(None, [], ModelSettings(), [], None, [], None)

    assert delegate.model_settings is not None
    traceparent = dict(delegate.model_settings.extra_headers or {})["traceparent"]
    assert traceparent.split("-")[1] == agent_span.trace_id


def test_traced_model_fulfils_both_sdk_model_operations() -> None:
    assert not TracedModel.__abstractmethods__
    assert callable(TracedModel.get_response)
    assert callable(TracedModel.stream_response)
