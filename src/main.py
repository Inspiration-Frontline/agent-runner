from agent_runner.config import settings
from agent_runner.local_startup import open_docs_when_ready
from agent_runner.main import app

if __name__ == "__main__":
    import asyncio
    import threading

    import uvicorn

    if settings.open_browser_on_startup and settings.environment in {"local", "dev"}:
        threading.Thread(
            target=open_docs_when_ready,
            args=(settings.server_port,),
            daemon=True,
            name="open-agent-runner-docs",
        ).start()

    server_config = uvicorn.Config(
        app,
        host=settings.server_host,
        port=settings.server_port,
    )
    server = uvicorn.Server(server_config)
    asyncio.run(server.serve())
