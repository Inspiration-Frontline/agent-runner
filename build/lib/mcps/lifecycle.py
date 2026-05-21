import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class MCPSession:
    session_id: str
    server_id: str
    connection: any
    active: bool = True


class MCPLifecycle:
    def __init__(self, mcp_manager):
        self._mcp_manager = mcp_manager
        self._sessions: dict[str, MCPSession] = {}

    async def create_session(self, session_id: str, server_id: str) -> MCPSession:
        connection = await self._mcp_manager.get_connection(server_id)
        session = MCPSession(
            session_id=session_id,
            server_id=server_id,
            connection=connection,
        )
        self._sessions[session_id] = session
        logger.info(f"Created MCP session: {session_id} for server: {server_id}")
        return session

    async def close_session(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            await self._mcp_manager.release_connection(session.server_id, session.connection)
            session.active = False
            del self._sessions[session_id]
            logger.info(f"Closed MCP session: {session_id}")

    async def get_session(self, session_id: str) -> MCPSession | None:
        return self._sessions.get(session_id)

    @asynccontextmanager
    async def session_scope(self, session_id: str, server_id: str) -> AsyncGenerator[MCPSession, None]:
        session = await self.create_session(session_id, server_id)
        try:
            yield session
        finally:
            await self.close_session(session_id)

    async def close_all_sessions(self):
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)
        logger.info("All MCP sessions closed")
