import json

import pytest

from agent_runner.agent_definitions import loader as loader_module
from agent_runner.config import CONFIG_DIR, ConfigurationManager, Settings, get_env_file, get_settings
from agent_runner.context import profile_adapter, rag_adapter
from agent_runner.context.builder import ContextBuilder


def test_context_builder_reads_current_context_budget() -> None:
    builder = ContextBuilder(settings=Settings().model_copy(update={"max_context_tokens": 2048}))

    assert builder.token_budget_manager.max_tokens == 2048


def test_profile_adapter_reads_current_service_url() -> None:
    adapter = profile_adapter.ProfileAdapter(Settings().model_copy(update={"user_profiler_url": "http://profile-test"}))

    assert adapter.base_url == "http://profile-test"


def test_rag_adapter_reads_current_service_url() -> None:
    adapter = rag_adapter.RAGAdapter(Settings().model_copy(update={"knowledge_service_url": "http://rag-test"}))

    assert adapter.base_url == "http://rag-test"


def test_agent_config_loader_reads_current_service_defaults() -> None:
    fake_settings = Settings().model_copy(
        update={
            "agent_config_center_url": "http://config-test",
            "agent_config_cache_enabled": False,
            "redis_host": "redis-test",
            "redis_port": 6380,
            "redis_password": "secret",
            "redis_db": 2,
            "redis_socket_connect_timeout_seconds": 3.5,
            "redis_socket_timeout_seconds": 4.5,
            "agent_config_cache_ttl_seconds": 600,
            "local_agent_config_enabled": False,
            "local_agent_config_path": "./config/custom.json",
            "max_output_tokens": 2048,
        }
    )

    loader = loader_module.AgentConfigLoader(fake_settings)

    assert loader.base_url == "http://config-test"
    assert loader.redis_client is None
    assert loader.cache_ttl_seconds == 600
    assert loader.local_agent_config_enabled is False
    assert loader.max_output_tokens == 2048
    assert str(loader.local_config_path).endswith("config\\custom.json")


def test_local_env_file_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_RUNNER_ENV_FILE", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    assert get_env_file() == CONFIG_DIR / "agent-runner.env"


def test_local_settings_do_not_require_manual_env_vars() -> None:
    settings = get_settings()

    assert settings.nacos_enabled is False
    assert settings.local_agent_config_enabled is True
    assert settings.lite_llm_base_url == "http://localhost:4000"
    assert settings.lite_llm_api_key


def test_otel_priority_is_nacos_then_file_then_code_default(tmp_path) -> None:
    env_file = tmp_path / "agent-runner.env"
    env_file.write_text(
        "OTEL_SERVICE_NAME=file-runner\n"
        "OTEL_EXPORTER_OTLP_ENDPOINT=http://file-collector:4317\n"
        "OTEL_TRACES_SAMPLER_ARG=0.5\n"
        "NACOS_ENABLED=true\n",
        encoding="utf-8",
    )
    file_settings = Settings(_env_file=env_file)
    manager = ConfigurationManager(file_settings)
    manager._nacos_loader = type(
        "NacosSnapshot",
        (),
        {"cached_config": {"observability": {"otel_service_name": "nacos-runner"}}},
    )()

    merged = manager.get_settings()

    assert merged.otel_service_name == "nacos-runner"
    assert merged.otel_exporter_otlp_endpoint == "http://file-collector:4317"
    assert merged.otel_sampling_ratio == 0.5
    assert file_settings.otel_enabled is True


def test_local_general_agent_uses_the_service_output_budget() -> None:
    config = json.loads((CONFIG_DIR / "agents.json").read_text(encoding="utf-8"))
    general_agent = next(agent for agent in config["agents"] if agent["agent_id"] == 1)

    assert len(config["agents"]) == 1
    assert general_agent["max_output_tokens"] == get_settings().max_output_tokens == 4096
    assert [binding["server_id"] for binding in general_agent["mcp_servers"]] == [
        "deepwiki",
        "microsoft-learn",
        "context7",
        "openai-docs",
    ]
