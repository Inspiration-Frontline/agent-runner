import json

import pytest
from v2.nacos.common.client_config import ClientConfig

from agent_runner.agent_definitions import loader as loader_module
from agent_runner.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    ConfigurationManager,
    Settings,
    get_env_file,
    get_settings,
    resolve_project_path,
)
from agent_runner.context import profile_adapter, rag_adapter
from agent_runner.context.builder import ContextBuilder
from agent_runner.nacos_config import NacosConfigLoader


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


def test_local_settings_enable_nacos_without_manual_env_vars() -> None:
    settings = get_settings()

    assert settings.nacos_enabled is True
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


def test_mcp_pool_priority_is_nacos_then_file_then_code_default(tmp_path) -> None:
    env_file = tmp_path / "agent-runner.env"
    env_file.write_text(
        "MCP_POOL_MAX_CONNECTIONS_PER_SERVER=2\n"
        "MCP_POOL_IDLE_TIMEOUT_SECONDS=45\n"
        "MCP_POOL_BORROW_TIMEOUT_SECONDS=6\n"
        "NACOS_ENABLED=true\n",
        encoding="utf-8",
    )
    file_settings = Settings(_env_file=env_file)
    manager = ConfigurationManager(file_settings)
    manager._nacos_loader = type(
        "NacosSnapshot",
        (),
        {"cached_config": {"mcp": {"pool_max_connections_per_server": 4}}},
    )()

    merged = manager.get_settings()

    assert Settings(_env_file=None).mcp_pool_max_connections_per_server == 4
    assert merged.mcp_pool_max_connections_per_server == 4
    assert merged.mcp_pool_idle_timeout_seconds == 45
    assert merged.mcp_pool_borrow_timeout_seconds == 6


def test_project_relative_configuration_paths_ignore_process_working_directory() -> None:
    manager = ConfigurationManager(Settings(_env_file=None))
    manager._nacos_loader = type(
        "NacosSnapshot",
        (),
        {"cached_config": {"mcp": {"catalog_path": "./config/mcp-servers.json"}}},
    )()

    merged = manager.get_settings()

    assert resolve_project_path(merged.mcp_catalog_path) == CONFIG_DIR / "mcp-servers.json"
    assert resolve_project_path("./config/agents.json") == CONFIG_DIR / "agents.json"
    assert resolve_project_path(str(PROJECT_ROOT / "external.json")) == PROJECT_ROOT / "external.json"


def test_mcp_secrets_use_latest_nacos_snapshot_with_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV_ONLY_MCP_KEY", "environment-secret")
    manager = ConfigurationManager(Settings(_env_file=None))
    nacos_snapshot = type(
        "NacosSnapshot",
        (),
        {
            "cached_config": {"mcp": {"secrets": {"NACOS_MCP_KEY": "nacos-secret"}}},
            "configuration_revision": 3,
        },
    )()
    manager._nacos_loader = nacos_snapshot

    initial_snapshot = manager.get_mcp_secret_snapshot()
    nacos_snapshot.cached_config = {"mcp": {"secrets": {"NACOS_MCP_KEY": "rotated-secret"}}}
    nacos_snapshot.configuration_revision = 4
    rotated_snapshot = manager.get_mcp_secret_snapshot()

    assert initial_snapshot.resolve("${secret:NACOS_MCP_KEY}") == "nacos-secret"
    assert initial_snapshot.resolve("${secret:ENV_ONLY_MCP_KEY}") == "environment-secret"
    assert initial_snapshot.configuration_revision == 3
    assert rotated_snapshot.resolve("${secret:NACOS_MCP_KEY}") == "rotated-secret"
    assert rotated_snapshot.configuration_revision == 4


def test_nacos_secret_replacement_is_atomic_and_empty_content_revokes() -> None:
    loader = NacosConfigLoader(enabled=False)
    loader._parse_and_replace_config("mcp:\n  secrets:\n    FIXTURE_KEY: first\n")

    with pytest.raises(ValueError):
        loader._parse_and_replace_config("- not\n- an\n- object\n")

    assert loader.cached_config == {"mcp": {"secrets": {"FIXTURE_KEY": "first"}}}
    assert loader.configuration_revision == 1

    loader._parse_and_replace_config("")

    assert loader.cached_config == {}
    assert loader.configuration_revision == 2


async def test_nacos_sdk_logging_is_disabled_at_the_application_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_configs: list[ClientConfig] = []

    class FakeNacosClient:
        async def get_config(self, config_param: object) -> str:
            return ""

        async def add_listener(self, data_id: str, group: str, listener: object) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    async def create_config_service(client_config: ClientConfig) -> FakeNacosClient:
        captured_configs.append(client_config)

        return FakeNacosClient()

    monkeypatch.setattr("agent_runner.nacos_config.NacosConfigService.create_config_service", create_config_service)
    loader = NacosConfigLoader(enabled=True)

    await loader.initialize()
    await loader.close()

    assert len(captured_configs) == 1
    assert captured_configs[0].log_level == "CRITICAL"


def test_local_general_agent_uses_the_service_output_budget() -> None:
    config = json.loads((CONFIG_DIR / "agents.json").read_text(encoding="utf-8"))
    general_agent = next(agent for agent in config["agents"] if agent["agent_id"] == 1)

    assert len(config["agents"]) == 1
    assert general_agent["max_output_tokens"] == get_settings().max_output_tokens == 4096
    assert [binding["server_id"] for binding in general_agent["mcp_servers"]] == [
        "deepwiki",
        "microsoft-learn",
        "context7",
        "tavily",
        "exa",
        "openai-docs",
    ]
