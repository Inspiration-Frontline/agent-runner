"""Application-owned, bounded Streamable HTTP MCP connection pooling.

Connections are pooled only after catalog secrets have been resolved.  A borrower gets exclusive
ownership of one SDK connection until its Agent request ends, which prevents request metadata,
cancellation, and durable dispatch evidence from crossing request boundaries.
"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic

from agents.mcp import MCPServerManager, MCPServerStreamableHttp

from agent_runner.mcps.catalog import ResolvedMcpServer


@dataclass(frozen=True)
class McpConnectionPoolSettings:
    """Effective per-server pool limits after Nacos, file, and default resolution."""

    max_connections_per_server: int
    idle_timeout_seconds: float
    borrow_timeout_seconds: float


@dataclass(frozen=True)
class McpConnectionKey:
    """Non-secret identity used to isolate pooled connections by effective credentials."""

    server_id: str
    url: str
    credential_fingerprint: str

    @classmethod
    def from_resolved(cls, server: ResolvedMcpServer) -> "McpConnectionKey":
        """Create a key without retaining resolved credential values outside the SDK client."""
        canonical_headers = json.dumps(sorted(server.headers.items()), separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical_headers.encode("utf-8")).hexdigest()
        return cls(server.server_id, server.url, fingerprint)

    @property
    def cache_key(self) -> str:
        """Return a stable cache identifier that never exposes credentials in logs or memory keys."""
        return f"{self.server_id}:{hashlib.sha256(f'{self.url}|{self.credential_fingerprint}'.encode()).hexdigest()}"


@dataclass
class PooledMcpConnection:
    """One live SDK manager/server pair that is either leased or available for borrowing."""

    key: McpConnectionKey
    server: MCPServerStreamableHttp
    manager: MCPServerManager
    last_returned_at: float
    borrowed: bool = False
    invalid: bool = False

    async def close(self) -> None:
        """Release the SDK-managed connection and its task-affine cleanup worker."""
        await self.manager.cleanup_all()


@dataclass
class _ConnectionBucket:
    """Mutable state for one credential-isolated connection pool."""

    connections: list[PooledMcpConnection] = field(default_factory=list)
    creating: int = 0
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class McpConnectionPool:
    """Hikari-inspired bounded borrower pool for SDK Streamable HTTP connections.

    The pool is application-owned.  It has no knowledge of a particular Agent request; request
    adapters are created above this layer and delegate to an exclusively borrowed connection.
    """

    def __init__(self) -> None:
        """Create an empty pool whose connections are opened lazily on first use."""
        self._buckets: dict[McpConnectionKey, _ConnectionBucket] = {}
        self._closed = False

    async def borrow(
        self,
        key: McpConnectionKey,
        settings: McpConnectionPoolSettings,
        creator: Callable[[], Awaitable[PooledMcpConnection]],
    ) -> PooledMcpConnection:
        """Borrow an exclusive live connection, waiting only up to the configured timeout."""
        deadline = monotonic() + settings.borrow_timeout_seconds
        bucket = self._buckets.setdefault(key, _ConnectionBucket())

        while True:
            expired = await self._take_expired(bucket, settings.idle_timeout_seconds)
            await self._close_all(expired)

            async with bucket.condition:
                if self._closed:
                    raise RuntimeError("MCP connection pool is closed.")
                reusable = next((item for item in bucket.connections if not item.borrowed and not item.invalid), None)
                if reusable is not None:
                    reusable.borrowed = True
                    return reusable
                if len(bucket.connections) + bucket.creating < settings.max_connections_per_server:
                    bucket.creating += 1
                    break
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for an MCP connection to {key.server_id}.")
                try:
                    await asyncio.wait_for(bucket.condition.wait(), timeout=remaining)
                except TimeoutError as error:
                    raise TimeoutError(f"Timed out waiting for an MCP connection to {key.server_id}.") from error

        try:
            connection = await creator()
        except BaseException:
            async with bucket.condition:
                bucket.creating -= 1
                bucket.condition.notify_all()
            raise

        async with bucket.condition:
            bucket.creating -= 1
            if self._closed:
                connection.invalid = True
            else:
                connection.borrowed = True
                bucket.connections.append(connection)
                bucket.condition.notify_all()
                return connection
            bucket.condition.notify_all()
        await connection.close()
        raise RuntimeError("MCP connection pool closed while creating a connection.")

    async def release(self, connection: PooledMcpConnection, invalidate: bool = False) -> None:
        """Return a lease or evict it after a transport-level failure."""
        bucket = self._buckets.get(connection.key)
        if bucket is None:
            await connection.close()
            return
        should_close = False
        async with bucket.condition:
            connection.borrowed = False
            connection.last_returned_at = monotonic()
            connection.invalid = connection.invalid or invalidate or self._closed
            if connection.invalid and connection in bucket.connections:
                bucket.connections.remove(connection)
                should_close = True
            bucket.condition.notify_all()
        if should_close:
            await connection.close()

    async def close(self) -> None:
        """Close idle and borrowed connections during FastAPI application shutdown."""
        self._closed = True
        connections: list[PooledMcpConnection] = []
        for bucket in self._buckets.values():
            async with bucket.condition:
                connections.extend(bucket.connections)
                bucket.connections.clear()
                bucket.condition.notify_all()
        await self._close_all(connections)

    async def _take_expired(self, bucket: _ConnectionBucket, idle_timeout_seconds: float) -> list[PooledMcpConnection]:
        """Remove idle, returned connections before a new borrow attempt."""
        now = monotonic()
        async with bucket.condition:
            expired = [
                item
                for item in bucket.connections
                if not item.borrowed and now - item.last_returned_at >= idle_timeout_seconds
            ]
            for item in expired:
                bucket.connections.remove(item)
            return expired

    @staticmethod
    async def _close_all(connections: list[PooledMcpConnection]) -> None:
        """Close independent SDK managers without allowing one cleanup failure to leak the rest."""
        results = await asyncio.gather(*(connection.close() for connection in connections), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                continue
