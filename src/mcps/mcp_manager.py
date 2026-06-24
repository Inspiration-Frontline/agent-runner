import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from mcps.mcp_connection_pool import MCPConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server.

    Defines the server's identity, transport mechanism, and
    connection configuration parameters.

    Attributes:
        server_id: Unique identifier for this MCP server.
        name: Human-readable name for this MCP server.
        transport: Transport mechanism: 'stdio', 'sse', or 'websocket'.
        config: Server-specific configuration parameters.
    """

    server_id: str
    name: str
    transport: str
    config: dict[str, Any]


class MCPManager:
    """
    Manager for MCP (Model Context Protocol) servers.

    Handles registration, connection pooling, and tool discovery
    for multiple MCP servers, providing centralized management.

    Attributes:
        _pools: Dictionary mapping server IDs to connection pools.
        _server_configs: Dictionary mapping server IDs to server configurations.
        _max_connections_per_server: Maximum connections allowed per server.
        _lock: Lock for thread-safe manager operations.
    """

    def __init__(self, max_connections_per_server: int = 5):
        """
        Initialize the MCP manager.

        Args:
            max_connections_per_server: Maximum connections per server pool.
        """
        self._pools: dict[str, MCPConnectionPool] = {}
        self._server_configs: dict[str, MCPServerConfig] = {}
        self._max_connections_per_server = max_connections_per_server
        self._lock = asyncio.Lock()

    async def register_server(self, config: MCPServerConfig):
        """
        Register a new MCP server.

        Args:
            config: Configuration for the MCP server to register.
        """
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
        """
        Unregister an MCP server and close its connections.

        Args:
            server_id: ID of the server to unregister.
        """
        async with self._lock:
            if server_id in self._pools:
                await self._pools[server_id].close_all()
                del self._pools[server_id]

            if server_id in self._server_configs:
                del self._server_configs[server_id]

            logger.info(f"Unregistered MCP server: {server_id}")

    async def get_connection(self, server_id: str) -> Any:
        """
        Get a connection to an MCP server.

        Args:
            server_id: ID of the server to connect to.

        Returns:
            Any: A connection object from the server's pool.

        Raises:
            ValueError: If the server is not registered.
        """
        pool = self._pools.get(server_id)
        if not pool:
            raise ValueError(f"MCP server not found: {server_id}")

        return await pool.acquire()

    async def release_connection(self, server_id: str, connection: Any):
        """
        Release a connection back to the server's pool.

        Args:
            server_id: ID of the server the connection belongs to.
            connection: The connection to release.
        """
        pool = self._pools.get(server_id)
        if pool:
            await pool.release(connection)

    async def discover_tools(self, server_id: str) -> list[dict[str, Any]]:
        """
        Discover tools available on an MCP server.

        Args:
            server_id: ID of the server to discover tools from.

        Returns:
            list[dict[str, Any]]: List of tool specifications.

        Raises:
            ValueError: If the server is not registered.
        """
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
        """
        Call a tool on an MCP server.

        Args:
            server_id: ID of the server hosting the tool.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Any: The result of the tool execution.
        """
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
        """
        Execute a tool call on a connection.

        Args:
            connection: The connection to use for execution.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Any: The result of the tool execution.
        """
        return {"status": "not_implemented", "tool": tool_name}

    async def close_all(self):
        """
        Close all connections for all registered servers.
        """
        async with self._lock:
            for pool in self._pools.values():
                await pool.close_all()
            self._pools.clear()
            logger.info("All MCP connections closed")

    def get_registered_servers(self) -> list[str]:
        """
        Get list of registered server IDs.

        Returns:
            list[str]: List of registered server IDs.
        """
        return list(self._server_configs.keys())
