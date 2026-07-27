from agent_runner.api.debug_routes import debug_config
from agent_runner.api.responses import DebugConfigResponse, HealthResponse
from agent_runner.main import app, health_check, root


def test_api_version_matches_project_version() -> None:
    assert app.version == "0.0.1"


async def test_health_check_returns_typed_response() -> None:
    response = await health_check()

    assert response == HealthResponse(status="healthy")


async def test_root_redirects_to_swagger_docs() -> None:
    response = await root()

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


async def test_debug_config_returns_typed_response() -> None:
    response = await debug_config()

    assert isinstance(response, DebugConfigResponse)
    assert response.lite_llm_base_url == "http://localhost:4000"
    assert response.nacos_enabled is False
