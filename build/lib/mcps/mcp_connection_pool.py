import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    connection_id: str
    connection: Any
    in_use: bool = False


class MCPConnectionPool:
    def __init__(self, server_id: str, max_connections: int = 5):
        self.server_id = server_id
        self.max_connections = max_connections
        self._connections: dict[str, PooledConnection] = {}
        self._available: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_connections)

    async def acquire(self) -> Any:
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
        async with self._lock:
            for conn_id, pooled_conn in self._connections.items():
                if pooled_conn.connection == connection:
                    pooled_conn.in_use = False
                    self._available.append(conn_id)
                    break

        self._semaphore.release()

    async def _create_connection(self) -> Any:
        logger.info(f"Creating new MCP connection for server: {self.server_id}")
        return {"server_id": self.server_id, "connected": True}

    async def close_all(self):
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
        logger.info(f"Closing MCP connection for server: {self.server_id}")

    @property
    def active_connections(self) -> int:
        return sum(1 for c in self._connections.values() if c.in_use)

    @property
    def total_connections(self) -> int:
        return len(self._connections)
