"""Debug endpoints for development and testing environments.

These endpoints are conditionally registered based on configuration:
- Enabled by default in local/dev environments
- Disabled in production environments

Configuration:
- environment: Controls overall environment mode (local, dev, prod)
- debug_endpoints_enabled: Explicit toggle for debug endpoints
"""

from fastapi import APIRouter, Request

from agent_runner.api.responses import DebugConfigResponse
from agent_runner.application_services import ApplicationServices
from agent_runner.config import Settings


async def get_debug_config(request: Request) -> DebugConfigResponse:
    """Debug endpoint to view current configuration values.

    Returns the merged configuration from Nacos and local files.
    Useful for testing configuration priority and dynamic refresh.

    Args:
        request: FastAPI request carrying the application service container.

    Returns:
        DebugConfigResponse: Current configuration values including:
            - lite_llm_base_url
            - agent_config_center_url
            - redis_host
            - agent_config_cache_ttl_seconds
            - nacos_enabled
            - environment
            - debug_endpoints_enabled
    """
    services: ApplicationServices = request.app.state.services
    settings: Settings = services.get_settings()

    return DebugConfigResponse(
        lite_llm_base_url=settings.lite_llm_base_url,
        agent_config_center_url=settings.agent_config_center_url,
        conversation_service_url=settings.conversation_service_url,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        agent_config_cache_ttl_seconds=settings.agent_config_cache_ttl_seconds,
        nacos_enabled=settings.nacos_enabled,
        nacos_namespace=settings.nacos_namespace,
        environment=settings.environment,
        debug_endpoints_enabled=settings.debug_endpoints_enabled,
    )


def create_debug_router() -> APIRouter:
    """Create development-only routes for one application instance.

    Returns:
        Development-only routing table for one application instance.
    """
    router: APIRouter = APIRouter(tags=["debug"])
    router.add_api_route(
        "/debug/config",
        get_debug_config,
        methods=["GET"],
        response_model=DebugConfigResponse,
    )

    return router
