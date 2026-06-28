from agent_runner.agent_definitions import loader as loader_module
from agent_runner.config import CONFIG_DIR, get_env_file, get_settings
from agent_runner.context import profile_adapter, rag_adapter
from agent_runner.context.builder import ContextBuilder


def test_context_builder_reads_current_context_budget(monkeypatch):
    monkeypatch.setattr("agent_runner.context.builder.get_settings", lambda: type("S", (), {"max_context_tokens": 2048})())

    builder = ContextBuilder()

    assert builder.token_budget_manager.max_tokens == 2048


def test_profile_adapter_reads_current_service_url(monkeypatch):
    monkeypatch.setattr(
        "agent_runner.context.profile_adapter.get_settings",
        lambda: type("S", (), {"user_profiler_url": "http://profile-test"})(),
    )

    adapter = profile_adapter.ProfileAdapter()

    assert adapter.base_url == "http://profile-test"


def test_rag_adapter_reads_current_service_url(monkeypatch):
    monkeypatch.setattr(
        "agent_runner.context.rag_adapter.get_settings",
        lambda: type("S", (), {"knowledge_service_url": "http://rag-test"})(),
    )

    adapter = rag_adapter.RAGAdapter()

    assert adapter.base_url == "http://rag-test"


def test_agent_config_loader_reads_current_service_defaults(monkeypatch):
    fake_settings = type(
        "S",
        (),
        {
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
        },
    )()
    monkeypatch.setattr("agent_runner.agent_definitions.loader.get_settings", lambda: fake_settings)

    loader = loader_module.AgentConfigLoader()

    assert loader.base_url == "http://config-test"
    assert loader.redis_client is None
    assert loader.cache_ttl_seconds == 600
    assert loader.local_agent_config_enabled is False
    assert loader.max_output_tokens == 2048
    assert str(loader.local_config_path).endswith("config\\custom.json")


def test_local_env_file_is_default(monkeypatch):
    monkeypatch.delenv("AGENT_RUNNER_ENV_FILE", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    assert get_env_file() == CONFIG_DIR / "agent-runner.env"


def test_local_settings_do_not_require_manual_env_vars():
    settings = get_settings()

    assert settings.nacos_enabled is False
    assert settings.local_agent_config_enabled is True
    assert settings.lite_llm_base_url == "http://localhost:4000"
    assert settings.lite_llm_api_key
