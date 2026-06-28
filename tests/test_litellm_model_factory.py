from agent_runner.gateway.litellm_client import LiteLLMModelFactory


def test_bare_proxy_model_uses_openai_provider_prefix():
    factory = LiteLLMModelFactory(
        base_url="http://localhost:4000",
        api_key="sk-test",
        request_timeout_seconds=3,
    )

    assert factory._normalize_model("Qwen/Qwen3-4B") == "openai/Qwen/Qwen3-4B"


def test_known_provider_model_is_left_unchanged():
    factory = LiteLLMModelFactory(
        base_url="http://localhost:4000",
        api_key="sk-test",
        request_timeout_seconds=3,
    )

    assert factory._normalize_model("anthropic/claude-sonnet-4-5") == "anthropic/claude-sonnet-4-5"


def test_created_model_targets_external_proxy():
    factory = LiteLLMModelFactory(
        base_url="http://localhost:4000",
        api_key="sk-test",
        request_timeout_seconds=3,
    )

    model = factory.create_model("Qwen/Qwen3-4B")

    assert model.model == "openai/Qwen/Qwen3-4B"
    assert model.base_url == "http://localhost:4000"
    assert model.api_key == "sk-test"
    assert factory.request_timeout_seconds == 3
