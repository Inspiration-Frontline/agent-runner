import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class MCPSession:
    """
    Session for MCP server connection management.

    Represents an active session with an MCP server, tracking
    the session ID, server ID, connection, and active status.

    Attributes:
        session_id: Unique identifier for this session.
        server_id: ID of the MCP server this session connects to.
        connection: The underlying connection object.
        active: Whether this session is currently active.
    """

    session_id: str
    server_id: str
    connection: any
    active: bool = True


class MCPLifecycle:
    """
    Lifecycle manager for MCP sessions.

    Manages the creation, tracking, and cleanup of MCP sessions,
    providing scoped session management with automatic cleanup.

    Attributes:
        _mcp_manager: MCP manager instance for connection operations.
        _sessions: Dictionary mapping session IDs to MCPSession instances.
    """

    def __init__(self, mcp_manager):
        """
        Initialize the MCP lifecycle manager.

        Args:
            mcp_manager: The MCP manager for connection operations.
        """
        self._mcp_manager = mcp_manager
        self._sessions: dict[str, MCPSession] = {}

    async def create_session(self, session_id: str, server_id: str) -> MCPSession:
        """
        Create a new MCP session.

        Args:
            session_id: Unique identifier for the session.
            server_id: ID of the MCP server to connect to.

        Returns:
            MCPSession: The created session instance.
        """
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
        """
        Close and cleanup an MCP session.

        Args:
            session_id: ID of the session to close.
        """
        session = self._sessions.get(session_id)
        if session:
            await self._mcp_manager.release_connection(session.server_id, session.connection)
            session.active = False
            del self._sessions[session_id]
            logger.info(f"Closed MCP session: {session_id}")

    async def get_session(self, session_id: str) -> MCPSession | None:
        """
        Retrieve an active MCP session.

        Args:
            session_id: ID of the session to retrieve.

        Returns:
            MCPSession | None: The session if found, None otherwise.
        """
        return self._sessions.get(session_id)

    @asynccontextmanager
    async def session_scope(self, session_id: str, server_id: str) -> AsyncGenerator[MCPSession, None]:
        """
        Context manager for scoped MCP session usage.

        Args:
            session_id: Unique identifier for the session.
            server_id: ID of the MCP server to connect to.

        Yields:
            MCPSession: The session instance for use within the scope.
        """
        session = await self.create_session(session_id, server_id)
        try:
            yield session
        finally:
            await self.close_session(session_id)

    async def close_all_sessions(self):
        """
        Close all active MCP sessions.
        """
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)
        logger.info("All MCP sessions closed")
