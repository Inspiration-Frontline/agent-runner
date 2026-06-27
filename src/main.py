from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.debug_routes import router as debug_router
from api.routes import router as agent_router
from config import initialize_settings, settings
from nacos_config import close_nacos_loader
from observability.logging import setup_logging
from observability.metrics import metrics_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application:
    - On startup: Initialize logging and merge Nacos configuration with local config
    - On shutdown: Close Nacos client and cleanup resources

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the application for normal operation.
    """
    setup_logging()

    # Initialize configuration from Nacos (merges with local config)
    await initialize_settings()

    yield

    # Cleanup Nacos resources on shutdown
    await close_nacos_loader()


app = FastAPI(
    title="Agent Runner",
    description="Core runtime component for AI Runtime Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(metrics_middleware)

app.include_router(agent_router, prefix="/v1/agent")

# Conditionally register debug endpoints based on environment
if settings.debug_endpoints_enabled or settings.environment in ["local", "dev"]:
    app.include_router(debug_router, prefix="/v1/agent")
    # Note: debug_router endpoints already have /debug prefix, so full path is /v1/agent/debug/*


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: A dictionary with status "healthy" indicating the service is running.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import asyncio

    import uvicorn

    server_config = uvicorn.Config(
        app,
        host=settings.server_host,
        port=settings.server_port,
    )
    server = uvicorn.Server(server_config)
    asyncio.run(server.serve())
