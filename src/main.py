from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as agent_router
from config import settings
from observability.logging import setup_logging
from observability.metrics import metrics_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.
    Currently only sets up logging on startup.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the application for normal operation.
    """
    setup_logging()
    yield


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
