import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from mcps.mcp_connection_pool import MCPConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    server_id: str
    name: str
    transport: str
    config: dict[str, Any]


class MCPManager:
    def __init__(self, max_connections_per_server: int = 5):
        self._pools: dict[str, MCPConnectionPool] = {}
        self._server_configs: dict[str, MCPServerConfig] = {}
        self._max_connections_per_server = max_connections_per_server
        self._lock = asyncio.Lock()

    async def register_server(self, config: MCPServerConfig):
        async with self._lock:
            if config.server_id in self._server_configs:
                logger.warning(f"MCP server {config.server_id} already registered")
                return

            self._server_configs[config.server_id] = config
            self._pools[config.server_id] = MCPConnectionPool(
                server_id=config.server_id,
                max_connections=self._max_connections_per_server,
            )
            logger.info(f"Registered MCP server: {config.server_id}")

    async def unregister_server(self, server_id: str):
        async with self._lock:
            if server_id in self._pools:
                await self._pools[server_id].close_all()
                del self._pools[server_id]

            if server_id in self._server_configs:
                del self._server_configs[server_id]

            logger.info(f"Unregistered MCP server: {server_id}")

    async def get_connection(self, server_id: str) -> Any:
        pool = self._pools.get(server_id)
        if not pool:
            raise ValueError(f"MCP server not found: {server_id}")

        return await pool.acquire()

    async def release_connection(self, server_id: str, connection: Any):
        pool = self._pools.get(server_id)
        if pool:
            await pool.release(connection)

    async def discover_tools(self, server_id: str) -> list[dict[str, Any]]:
        config = self._server_configs.get(server_id)
        if not config:
            raise ValueError(f"MCP server not found: {server_id}")

        return []

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        connection = None
        try:
            connection = await self.get_connection(server_id)
            return await self._execute_tool_call(connection, tool_name, arguments)
        finally:
            if connection:
                await self.release_connection(server_id, connection)

    async def _execute_tool_call(
        self,
        connection: Any,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        return {"status": "not_implemented", "tool": tool_name}

    async def close_all(self):
        async with self._lock:
            for pool in self._pools.values():
                await pool.close_all()
            self._pools.clear()
            logger.info("All MCP connections closed")

    def get_registered_servers(self) -> list[str]:
        return list(self._server_configs.keys())
