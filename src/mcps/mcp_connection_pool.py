import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """
    Connection wrapper for connection pool management.

    Tracks the connection ID, underlying connection object,
    and usage status for pool management.

    Attributes:
        connection_id: Unique identifier for this pooled connection.
        connection: The underlying connection object.
        in_use: Whether this connection is currently in use.
    """

    connection_id: str
    connection: Any
    in_use: bool = False


class MCPConnectionPool:
    """
    Connection pool for MCP server connections.

    Manages a pool of connections to a specific MCP server,
    providing connection reuse and concurrency control.

    Attributes:
        server_id: ID of the MCP server for this pool.
        max_connections: Maximum number of connections allowed.
        _connections: Dictionary mapping connection IDs to pooled connections.
        _available: Queue of available connection IDs.
        _lock: Lock for thread-safe pool operations.
        _semaphore: Semaphore for connection limit enforcement.
    """

    def __init__(self, server_id: str, max_connections: int = 5):
        """
        Initialize the connection pool.

        Args:
            server_id: ID of the MCP server this pool manages.
            max_connections: Maximum number of connections in the pool.
        """
        self.server_id = server_id
        self.max_connections = max_connections
        self._connections: dict[str, PooledConnection] = {}
        self._available: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_connections)

    async def acquire(self) -> Any:
        """
        Acquire a connection from the pool.

        Returns an available connection or creates a new one if needed.

        Returns:
            Any: A connection object for use.
        """
        await self._semaphore.acquire()

        async with self._lock:
            if self._available:
                conn_id = self._available.popleft()
                pooled_conn = self._connections[conn_id]
                pooled_conn.in_use = True
                return pooled_conn.connection

            conn_id = f"{self.server_id}_{len(self._connections)}"
            connection = await self._create_connection()
            pooled_conn = PooledConnection(
                connection_id=conn_id,
                connection=connection,
                in_use=True,
            )
            self._connections[conn_id] = pooled_conn
            return connection

    async def release(self, connection: Any):
        """
        Release a connection back to the pool.

        Args:
            connection: The connection to release.
        """
        async with self._lock:
            for conn_id, pooled_conn in self._connections.items():
                if pooled_conn.connection == connection:
                    pooled_conn.in_use = False
                    self._available.append(conn_id)
                    break

        self._semaphore.release()

    async def _create_connection(self) -> Any:
        """
        Create a new connection to the MCP server.

        Returns:
            Any: A new connection object.
        """
        logger.info(f"Creating new MCP connection for server: {self.server_id}")
        return {"server_id": self.server_id, "connected": True}

    async def close_all(self):
        """
        Close all connections in the pool.
        """
        async with self._lock:
            for conn_id, pooled_conn in self._connections.items():
                try:
                    await self._close_connection(pooled_conn.connection)
                except Exception as e:
                    logger.warning(f"Error closing connection {conn_id}: {e}")

            self._connections.clear()
            self._available.clear()
            logger.info(f"Closed all connections for server: {self.server_id}")

    async def _close_connection(self, connection: Any):
        """
        Close a single connection.

        Args:
            connection: The connection to close.
        """
        logger.info(f"Closing MCP connection for server: {self.server_id}")

    @property
    def active_connections(self) -> int:
        """
        Get the number of currently active connections.

        Returns:
            int: Number of connections currently in use.
        """
        return sum(1 for c in self._connections.values() if c.in_use)

    @property
    def total_connections(self) -> int:
        """
        Get the total number of connections in the pool.

        Returns:
            int: Total number of connections (active and available).
        """
        return len(self._connections)
