from config import CONFIG_DIR, get_env_file, get_settings


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
