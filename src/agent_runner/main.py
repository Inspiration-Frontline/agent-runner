from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_runner.api.debug_routes import router as debug_router
from agent_runner.api.routes import router as agent_router
from agent_runner.config import get_settings, initialize_settings
from agent_runner.nacos_config import close_nacos_loader
from agent_runner.observability.logging import setup_logging
from agent_runner.observability.metrics import metrics_middleware


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

    await initialize_settings()

    current_settings = get_settings()
    if current_settings.debug_endpoints_enabled or current_settings.environment in ["local", "dev"]:
        app.include_router(debug_router, prefix="/v1/agent")

    yield

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


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: A dictionary with status "healthy" indicating the service is running.
    """
    return {"status": "healthy"}
