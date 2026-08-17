from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from agent_runner.api.debug_routes import create_debug_router
from agent_runner.api.responses import HealthResponse
from agent_runner.api.routes import create_agent_router
from agent_runner.application_services import ApplicationServices
from agent_runner.config import ConfigurationManager, Settings
from agent_runner.mcps.connection_pool import McpConnectionPool
from agent_runner.mcps.sdk_runtime import McpSchemaCache
from agent_runner.observability.logging import setup_logging
from agent_runner.observability.metrics import MetricsCollector
from agent_runner.observability.tracing import TracingManager
from agent_runner.runtime.cancellation import ConversationCancellationRegistry


def create_app() -> FastAPI:
    """Build one Agent Runner application with explicitly owned mutable services."""
    configuration = ConfigurationManager(Settings())
    metrics = MetricsCollector()
    cancellations = ConversationCancellationRegistry()
    mcp_connection_pool = McpConnectionPool()
    mcp_schema_cache = McpSchemaCache()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        settings = await configuration.initialize()
        tracing = TracingManager(
            settings.otel_service_name,
            endpoint=settings.otel_exporter_otlp_endpoint,
            sampling_ratio=settings.otel_sampling_ratio,
            enabled=settings.otel_enabled,
        )
        application.state.services = ApplicationServices(
            configuration=configuration,
            tracing=tracing,
            metrics=metrics,
            cancellations=cancellations,
            mcp_connection_pool=mcp_connection_pool,
            mcp_schema_cache=mcp_schema_cache,
        )
        if settings.debug_endpoints_enabled or settings.environment in {"local", "dev"}:
            application.include_router(create_debug_router(), prefix="/v1/agent")
        try:
            yield
        finally:
            await mcp_connection_pool.close()
            await configuration.close()
            tracing.shutdown()

    application = FastAPI(
        title="Agent Runner",
        description="Core runtime component for AI Runtime Platform",
        version="0.0.1",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.middleware("http")(metrics.collect_http_metrics)
    application.include_router(create_agent_router(), prefix="/v1/agent")

    @application.get("/", include_in_schema=False)
    async def redirect_to_docs() -> RedirectResponse:
        """Redirect the service root to the interactive API documentation."""
        return RedirectResponse(url="/docs")

    @application.get("/health", response_model=HealthResponse)
    async def get_health_status() -> HealthResponse:
        """Return the current service health status."""
        return HealthResponse(status="healthy")

    return application
