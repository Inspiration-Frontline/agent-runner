from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial

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
from agent_runner.mcps.secrets import ConfigurationSecretProvider
from agent_runner.observability.logging import setup_logging
from agent_runner.observability.metrics import MetricsCollector
from agent_runner.observability.tracing import TracingManager
from agent_runner.runtime.cancellation import ConversationCancellationRegistry


@asynccontextmanager
async def manage_application_lifecycle(
    application: FastAPI,
    configuration: ConfigurationManager,
    metrics: MetricsCollector,
    cancellations: ConversationCancellationRegistry,
    mcp_connection_pool: McpConnectionPool,
    mcp_schema_cache: McpSchemaCache,
    mcp_secret_provider: ConfigurationSecretProvider,
) -> AsyncIterator[None]:
    """Initialize shared services and deterministically close application-owned resources.

    Args:
        application: FastAPI instance receiving lifecycle-owned services and optional debug routes.
        configuration: Configuration manager whose Nacos listener belongs to this application.
        metrics: Process-local HTTP metrics collector.
        cancellations: Registry of request cancellation tokens.
        mcp_connection_pool: Shared pool whose transport managers must close during shutdown.
        mcp_schema_cache: Shared MCP schema cache owned by the application lifecycle.
        mcp_secret_provider: Provider exposing the latest immutable Secret snapshot.

    Yields:
        Control while the application is accepting requests.
    """
    setup_logging()
    settings: Settings = await configuration.initialize()
    tracing: TracingManager = TracingManager(
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
        mcp_secret_provider=mcp_secret_provider,
    )

    if settings.debug_endpoints_enabled or settings.environment in {"local", "dev"}:
        application.include_router(create_debug_router(), prefix="/v1/agent")

    try:
        yield
    finally:
        await mcp_connection_pool.close()
        await configuration.close()
        tracing.shutdown()


async def redirect_to_docs() -> RedirectResponse:
    """Redirect the service root to the interactive API documentation.

    Returns:
        Redirect response targeting the interactive API documentation.
    """

    return RedirectResponse(url="/docs")


async def get_health_status() -> HealthResponse:
    """Return the current service health status.

    Returns:
        Current service health status.
    """

    return HealthResponse(status="healthy")


def create_app() -> FastAPI:
    """Build one Agent Runner application with explicitly owned mutable services.

    Returns:
        build one Agent Runner application with explicitly owned mutable services.
    """
    configuration: ConfigurationManager = ConfigurationManager(Settings())
    metrics: MetricsCollector = MetricsCollector()
    cancellations: ConversationCancellationRegistry = ConversationCancellationRegistry()
    mcp_connection_pool: McpConnectionPool = McpConnectionPool()
    mcp_schema_cache: McpSchemaCache = McpSchemaCache()
    mcp_secret_provider: ConfigurationSecretProvider = ConfigurationSecretProvider(
        configuration.get_mcp_secret_snapshot
    )
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] = partial(
        manage_application_lifecycle,
        configuration=configuration,
        metrics=metrics,
        cancellations=cancellations,
        mcp_connection_pool=mcp_connection_pool,
        mcp_schema_cache=mcp_schema_cache,
        mcp_secret_provider=mcp_secret_provider,
    )
    application: FastAPI = FastAPI(
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

    application.add_api_route("/", redirect_to_docs, include_in_schema=False)
    application.add_api_route("/health", get_health_status, response_model=HealthResponse)

    return application
