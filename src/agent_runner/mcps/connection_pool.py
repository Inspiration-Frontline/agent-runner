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
from typing import cast

from agents.mcp import MCPServerStreamableHttp

from agent_runner.mcps.catalog import ResolvedMcpServer


@dataclass(frozen=True)
class McpConnectionPoolSettings:
    """Effective per-server pool limits after Nacos, file, and default resolution."""

    max_connections_per_server: int
    """Maximum number of live connections allowed for one server identity."""
    idle_timeout_seconds: float
    """Monotonic idle duration after which an available connection is evicted."""
    borrow_timeout_seconds: float
    """Maximum duration a borrower waits for capacity or a returned connection."""


@dataclass(frozen=True)
class McpConnectionKey:
    """Non-secret identity used to isolate pooled connections by effective credentials."""

    server_id: str
    """Stable Catalog identity of the remote MCP Server."""
    endpoint_fingerprint: str
    """SHA-256 fingerprint of the configured endpoint URL."""
    credential_fingerprint: str
    """SHA-256 fingerprint of resolved URL/header credential material."""
    configuration_revision: int = field(compare=False)
    """Monotonic Nacos snapshot revision used to order credential rotation events."""

    @classmethod
    def from_resolved(cls, server: ResolvedMcpServer) -> "McpConnectionKey":
        """Create a key without retaining resolved credential values outside the SDK client.

        Args:
            server: Resolved or connected MCP server participating in the operation.

        Returns:
            Credential-isolated key containing only stable identity fingerprints.
        """
        endpoint_fingerprint: str = hashlib.sha256(server.profile.url.encode("utf-8")).hexdigest()
        canonical_headers: str = json.dumps(sorted(server.headers.items()), separators=(",", ":"))
        resolved_credentials: str = f"{server.url}|{canonical_headers}"
        credential_fingerprint: str = hashlib.sha256(resolved_credentials.encode("utf-8")).hexdigest()

        return cls(
            server.server_id,
            endpoint_fingerprint,
            credential_fingerprint,
            server.configuration_revision,
        )

    @property
    def cache_key(self) -> str:
        """Return a stable cache identifier that never exposes credentials in logs or memory keys.

        Returns:
            Stable cache identifier that never exposes credentials in logs or memory keys.
        """

        return f"{self.server_id}:{self.endpoint_fingerprint}:{self.credential_fingerprint}"


class TaskAffineMcpConnectionManager:
    """Own one Streamable HTTP lifecycle in the same task from connect through cleanup.

    AnyIO cancel scopes used by the MCP transport are task-affine. A transport background failure
    may cancel its owning task, so cleanup must run in that task's ``finally`` block instead of
    being delegated after the worker has already exited.

    Attributes:
        _server: SDK Streamable HTTP server whose transport lifecycle belongs to the owner task.
        _connect_timeout_seconds: Maximum initialize-handshake duration before entry fails.
        _close_requested: One-way signal asking the owner task to leave its hosting wait state.
        _cleanup_lock: Lock making repeated/concurrent cleanup requests idempotent.
        _ready: Future completed by the owner after connect succeeds or fails.
        _owner_task: Sole task allowed to connect, host, and clean up the AnyIO transport.
        _active: Whether connect completed and the server is available to an exclusive borrower.
        _cleanup_error: Transport cleanup failure re-raised to the lifecycle caller, if any.
    """

    def __init__(self, server: MCPServerStreamableHttp, connect_timeout_seconds: float) -> None:
        """Prepare an owner task that starts lazily when the manager is entered.

        Args:
            server: Resolved or connected MCP server participating in the operation.
            connect_timeout_seconds: Maximum duration in seconds for the MCP initialize handshake.
        """
        self._server = server
        self._connect_timeout_seconds = connect_timeout_seconds
        self._close_requested = asyncio.Event()
        self._cleanup_lock = asyncio.Lock()
        self._ready: asyncio.Future[None] | None = None
        self._owner_task: asyncio.Task[None] | None = None
        self._active = False
        self._cleanup_error: BaseException | None = None

    @property
    def active_servers(self) -> list[MCPServerStreamableHttp]:
        """Expose the connected server using the SDK Manager's public shape.

        Returns:
            Connected server when active; otherwise an empty list.
        """

        return [self._server] if self._active else []

    async def __aenter__(self) -> "TaskAffineMcpConnectionManager":
        """Start the owner and wait for a bounded initialize handshake.

        Returns:
            The initialized MCP server owned by this asyncio task.
        """

        if self._owner_task is not None:
            raise RuntimeError("MCP connection manager cannot be entered more than once.")
        self._ready = asyncio.get_running_loop().create_future()
        self._owner_task = asyncio.create_task(self._run(), name=f"mcp-owner-{self._server.name}")

        try:
            async with asyncio.timeout(self._connect_timeout_seconds):
                await asyncio.shield(self._ready)
        except BaseException:
            if not self._ready.done():
                self._owner_task.cancel()
            await self._wait_for_owner()

            if self._ready.done() and not self._ready.cancelled():
                self._ready.exception()

            raise

        return self

    async def cleanup_all(self) -> None:
        """Ask the owner to close and wait without allowing caller cancellation to interrupt it."""
        async with self._cleanup_lock:
            self._close_requested.set()
            await self._wait_for_owner()

            if self._cleanup_error is not None:
                raise self._cleanup_error

    async def _run(self) -> None:
        """Connect, remain the transport host, and always clean up in this same task."""
        ready: asyncio.Future[None] | None = self._ready

        if ready is None:
            raise RuntimeError("MCP connection owner started without a readiness future.")

        try:
            connect: Callable[[], Awaitable[None]] = cast(Callable[[], Awaitable[None]], self._server.connect)
            await connect()
            self._active = True

            if not ready.done():
                ready.set_result(None)

            await self._close_requested.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
        finally:
            try:
                cleanup: Callable[[], Awaitable[None]] = cast(Callable[[], Awaitable[None]], self._server.cleanup)
                await cleanup()
            except BaseException as error:
                self._cleanup_error = error

            self._active = False

    async def _wait_for_owner(self) -> None:
        """Wait through repeated caller cancellation, then preserve cancellation for the caller."""
        owner_task: asyncio.Task[None] | None = self._owner_task

        if owner_task is None:
            return

        cancelled: bool = False
        while not owner_task.done():
            try:
                await asyncio.shield(owner_task)
            except asyncio.CancelledError:
                cancelled = True
        owner_task.exception()

        if cancelled:
            raise asyncio.CancelledError


@dataclass
class PooledMcpConnection:
    """One live SDK manager/server pair that is either leased or available for borrowing."""

    key: McpConnectionKey
    """Credential-isolated identity used to locate this connection bucket."""
    server: MCPServerStreamableHttp
    """SDK server facade backed by the live transport."""
    manager: TaskAffineMcpConnectionManager
    """Task-affine lifecycle owner responsible for transport cleanup."""
    last_returned_at: float
    """Monotonic timestamp recorded when the connection last became idle."""
    borrowed: bool = False
    """Whether a request currently owns this connection lease."""
    invalid: bool = False
    """Whether the connection must be closed instead of reused."""

    async def close(self) -> None:
        """Release the SDK-managed connection and its task-affine cleanup worker."""
        await self.manager.cleanup_all()


@dataclass
class _ConnectionBucket:
    """Mutable state for one credential-isolated connection pool."""

    connections: list[PooledMcpConnection] = field(default_factory=list)
    """Live connections associated with one immutable connection key."""
    creating: int = 0
    """Number of connection slots reserved while handshakes are in flight."""
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    """Condition coordinating borrowers, creators, and releasers."""


class McpConnectionPool:
    """Hikari-inspired bounded borrower pool for SDK Streamable HTTP connections.

    The pool is application-owned.  It has no knowledge of a particular Agent request; request
    adapters are created above this layer and delegate to an exclusively borrowed connection.

    Attributes:
        _buckets: Collection of buckets consumed in deterministic order.
        _active_keys: Collection of active keys consumed in deterministic order.
        _activation_lock: Lock serializing access to activation state.
        _closed: Whether the pool has stopped accepting borrows and releases.
    """

    def __init__(self) -> None:
        """Create an empty pool whose connections are opened lazily on first use."""
        # Key: server, URL, and credential fingerprint identity. Value: isolated connection bucket.
        self._buckets: dict[McpConnectionKey, _ConnectionBucket] = {}
        # Key: stable MCP server ID. Value: newest credential-isolated key allowed to populate its pool.
        self._active_keys: dict[str, McpConnectionKey] = {}
        self._activation_lock = asyncio.Lock()
        self._closed = False

    async def borrow(
        self,
        key: McpConnectionKey,
        settings: McpConnectionPoolSettings,
        creator: Callable[[], Awaitable[PooledMcpConnection]],
    ) -> PooledMcpConnection:
        """Borrow one exclusive live connection through a bounded, cancellation-safe state machine.

        The method first evicts expired idle entries, then atomically chooses one of three paths:
        reuse an idle connection, reserve capacity for one new connection, or wait for a return.
        ``bucket.creating`` reserves capacity before network I/O starts, so concurrent borrowers
        cannot all observe free capacity and exceed the configured maximum. Connection creation and
        cleanup happen outside the Condition lock so slow network operations never block releases.

        Args:
            key: Credential-isolated identity of the target MCP server.
            settings: Effective capacity, idle-expiry, and wait-time limits.
            creator: Async factory that opens one initialized SDK connection.

        Returns:
            An exclusively leased connection that must later be passed to :meth:`release`.

        Raises:
            TimeoutError: No connection becomes available before the shared borrow deadline.
            RuntimeError: The application pool is closed before the lease can be returned.
        """
        active: bool = await self._activate_key(key)

        if not active:
            connection: PooledMcpConnection = await creator()
            connection.borrowed = True
            connection.invalid = True

            return connection

        # One absolute deadline covers every wake-up, preventing spurious notifications from
        # restarting the configured borrow timeout.
        deadline: float = monotonic() + settings.borrow_timeout_seconds
        bucket: _ConnectionBucket = self._buckets.setdefault(key, _ConnectionBucket())

        while True:
            # Remove expired entries under the bucket lock, but close their network resources after
            # releasing it. SDK cleanup can block and must not stall borrowers returning leases.
            expired: list[PooledMcpConnection] = await self._take_expired(bucket, settings.idle_timeout_seconds)
            await self._close_all(expired)

            async with bucket.condition:
                if self._closed:
                    raise RuntimeError("MCP connection pool is closed.")
                reusable: PooledMcpConnection | None = next(
                    (item for item in bucket.connections if not item.borrowed and not item.invalid), None
                )

                if reusable is not None:
                    reusable.borrowed = True

                    return reusable

                if len(bucket.connections) + bucket.creating < settings.max_connections_per_server:
                    # Reserve the slot before leaving the lock. Other borrowers include this count
                    # in their capacity check while the actual Streamable HTTP handshake is running.
                    bucket.creating += 1
                    break
                remaining: float = deadline - monotonic()

                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for an MCP connection to {key.server_id}.")

                try:
                    await asyncio.wait_for(bucket.condition.wait(), timeout=remaining)
                except TimeoutError as error:
                    raise TimeoutError(f"Timed out waiting for an MCP connection to {key.server_id}.") from error

        # The remote initialize handshake intentionally runs without holding bucket.condition.
        try:
            connection = await creator()
        except BaseException:
            # Cancellation and ordinary failures both release the capacity reservation and wake a
            # waiter that may now create a replacement connection.
            async with bucket.condition:
                bucket.creating -= 1
                bucket.condition.notify_all()

            raise

        # Publish the completed connection atomically. If shutdown won the race, close this late
        # connection instead of allowing a lease to escape a closed application pool.
        async with bucket.condition:
            bucket.creating -= 1
            active_key: McpConnectionKey | None = self._active_keys.get(key.server_id)

            if self._closed:
                connection.invalid = True
            elif active_key != key:
                # A newer Nacos revision replaced these credentials during the remote handshake.
                # The already-started request may finish with this unpooled lease, but no later
                # request can borrow it and release() will close it immediately.
                connection.borrowed = True
                connection.invalid = True
                bucket.condition.notify_all()

                return connection
            else:
                connection.borrowed = True
                bucket.connections.append(connection)
                bucket.condition.notify_all()

                return connection

            bucket.condition.notify_all()

        await connection.close()

        raise RuntimeError("MCP connection pool closed while creating a connection.")

    async def release(self, connection: PooledMcpConnection, invalidate: bool = False) -> None:
        """Return a lease or atomically evict it after a transport-level failure.

        Invalid connections are detached while holding the bucket lock, then closed outside the
        lock. Every return notifies waiters because it either exposes reusable capacity or removes a
        broken entry so a waiter can reserve replacement capacity.

        Args:
            connection: Exclusively borrowed MCP connection.
            invalidate: Whether a transport failure requires immediate pool eviction.
        """
        bucket: _ConnectionBucket | None = self._buckets.get(connection.key)

        if bucket is None:
            await connection.close()
            return

        should_close: bool = False
        async with bucket.condition:
            connection.borrowed = False
            connection.last_returned_at = monotonic()
            connection.invalid = connection.invalid or invalidate or self._closed

            if connection.invalid:
                if connection in bucket.connections:
                    bucket.connections.remove(connection)

                should_close = True

            bucket.condition.notify_all()

        if should_close:
            await connection.close()

    async def close(self) -> None:
        """Detach and close all connections during FastAPI application shutdown.

        Buckets are drained before SDK cleanup begins. Borrowers waking during this process observe
        ``_closed`` and fail instead of receiving a connection whose manager is being torn down.
        """
        self._closed = True
        self._active_keys.clear()
        connections: list[PooledMcpConnection] = []

        for bucket in self._buckets.values():
            async with bucket.condition:
                connections.extend(bucket.connections)
                bucket.connections.clear()
                bucket.condition.notify_all()

        await self._close_all(connections)

    async def _activate_key(self, key: McpConnectionKey) -> bool:
        """Activate a credential revision and retire connections for older credentials.

        A newer configuration revision wins permanently over delayed work carrying an older
        snapshot. Idle stale connections close immediately. Borrowed stale connections are marked
        invalid and close when their owning request returns them, so active Tool calls are never
        interrupted during credential rotation.

        Args:
            key: Resolved endpoint and credential identity for the requesting connection.

        Returns:
            ``True`` when the key may use the shared pool, or ``False`` when an older in-flight
            request must receive an unpooled connection.
        """
        connections_to_close: list[PooledMcpConnection] = []
        async with self._activation_lock:
            active_key: McpConnectionKey | None = self._active_keys.get(key.server_id)

            if active_key is not None and key.configuration_revision < active_key.configuration_revision:
                return False

            if active_key == key:
                if key.configuration_revision > active_key.configuration_revision:
                    self._active_keys[key.server_id] = key

                return True

            self._active_keys[key.server_id] = key
            stale_keys: list[McpConnectionKey] = [
                candidate for candidate in self._buckets if candidate.server_id == key.server_id
            ]

            for stale_key in stale_keys:
                stale_bucket: _ConnectionBucket = self._buckets[stale_key]
                async with stale_bucket.condition:
                    idle_connections: list[PooledMcpConnection] = []

                    for connection in stale_bucket.connections:
                        connection.invalid = True

                        if not connection.borrowed:
                            idle_connections.append(connection)
                            connections_to_close.append(connection)

                    for connection in idle_connections:
                        stale_bucket.connections.remove(connection)

                    stale_bucket.condition.notify_all()

        await self._close_all(connections_to_close)

        return True

    async def _take_expired(self, bucket: _ConnectionBucket, idle_timeout_seconds: float) -> list[PooledMcpConnection]:
        """Atomically detach returned connections whose monotonic idle age reached the limit.

        Args:
            bucket: Mutable connection bucket protected by its condition lock.
            idle_timeout_seconds: Maximum monotonic idle age before eviction.

        Returns:
            Connections detached from the bucket because their idle age exceeded the limit.
        """
        now: float = monotonic()
        async with bucket.condition:
            expired: list[PooledMcpConnection] = [
                item
                for item in bucket.connections
                if not item.borrowed and now - item.last_returned_at >= idle_timeout_seconds
            ]

            for item in expired:
                bucket.connections.remove(item)

            return expired

    @staticmethod
    async def _close_all(connections: list[PooledMcpConnection]) -> None:
        """Close independent SDK managers without allowing one cleanup failure to skip the rest.

        Args:
            connections: Connections detached from the bucket and closed outside its lock.
        """
        results: list[None | BaseException] = await asyncio.gather(
            *(connection.close() for connection in connections), return_exceptions=True
        )

        for result in results:
            if isinstance(result, BaseException):
                continue
