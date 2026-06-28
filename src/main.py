from agent_runner.config import settings
from agent_runner.main import app

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
