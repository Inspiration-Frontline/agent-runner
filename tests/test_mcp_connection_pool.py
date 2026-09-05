import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from agents.mcp import MCPServerStreamableHttp

from agent_runner.mcps.connection_pool import (
    McpConnectionKey,
    McpConnectionPool,
    McpConnectionPoolSettings,
    PooledMcpConnection,
    TaskAffineMcpConnectionManager,
)
from agent_runner.mcps.sdk_runtime import _await_cancellation_safe_cleanup


class FakeManager:
    def __init__(self) -> None:
        self.close_count = 0

    async def cleanup_all(self) -> None:
        self.close_count += 1


def pool_settings(max_connections: int = 4, idle_timeout: float = 300.0) -> McpConnectionPoolSettings:
    return McpConnectionPoolSettings(max_connections, idle_timeout, 1.0)


def connection_key(credential_fingerprint: str = "fingerprint", configuration_revision: int = 1) -> McpConnectionKey:
    return McpConnectionKey("fixture", "endpoint-fingerprint", credential_fingerprint, configuration_revision)


async def test_pool_reuses_a_returned_exclusive_connection() -> None:
    pool = McpConnectionPool()
    key = connection_key()
    created: list[PooledMcpConnection] = []

    async def create() -> PooledMcpConnection:
        connection = PooledMcpConnection(key, SimpleNamespace(), FakeManager(), 0)  # type: ignore[arg-type]
        created.append(connection)

        return connection

    first = await pool.borrow(key, pool_settings(), create)
    await pool.release(first)
    second = await pool.borrow(key, pool_settings(), create)

    assert second is first
    assert len(created) == 1
    await pool.release(second)
    await pool.close()


async def test_pool_waits_for_a_returned_connection_at_capacity() -> None:
    pool = McpConnectionPool()
    key = connection_key()

    async def create() -> PooledMcpConnection:
        return PooledMcpConnection(key, SimpleNamespace(), FakeManager(), 0)  # type: ignore[arg-type]

    first = await pool.borrow(key, pool_settings(max_connections=1), create)
    waiting = asyncio.create_task(pool.borrow(key, pool_settings(max_connections=1), create))
    await asyncio.sleep(0)
    assert not waiting.done()

    await pool.release(first)
    second = await waiting

    assert second is first
    await pool.release(second)
    await pool.close()


async def test_pool_evicts_idle_connections_before_creating_a_replacement() -> None:
    pool = McpConnectionPool()
    key = connection_key()
    managers: list[FakeManager] = []

    async def create() -> PooledMcpConnection:
        manager = FakeManager()
        managers.append(manager)

        return PooledMcpConnection(key, SimpleNamespace(), manager, 0)  # type: ignore[arg-type]

    first = await pool.borrow(key, pool_settings(idle_timeout=0.001), create)
    await pool.release(first)
    await asyncio.sleep(0.01)
    second = await pool.borrow(key, pool_settings(idle_timeout=0.001), create)

    assert second is not first
    assert managers[0].close_count == 1
    await pool.release(second)
    await pool.close()


async def test_pool_evicts_a_transport_failed_connection_on_return() -> None:
    pool = McpConnectionPool()
    key = connection_key()
    managers: list[FakeManager] = []

    async def create() -> PooledMcpConnection:
        manager = FakeManager()
        managers.append(manager)

        return PooledMcpConnection(key, SimpleNamespace(), manager, 0)  # type: ignore[arg-type]

    first = await pool.borrow(key, pool_settings(), create)
    await pool.release(first, invalidate=True)
    second = await pool.borrow(key, pool_settings(), create)

    assert second is not first
    assert managers[0].close_count == 1
    await pool.release(second)
    await pool.close()


async def test_pool_closes_idle_connections_when_credentials_rotate() -> None:
    pool = McpConnectionPool()
    old_key = connection_key("old-fingerprint", configuration_revision=1)
    new_key = connection_key("new-fingerprint", configuration_revision=2)
    old_manager = FakeManager()

    async def create_old() -> PooledMcpConnection:
        return PooledMcpConnection(old_key, SimpleNamespace(), old_manager, 0)  # type: ignore[arg-type]

    async def create_new() -> PooledMcpConnection:
        return PooledMcpConnection(new_key, SimpleNamespace(), FakeManager(), 0)  # type: ignore[arg-type]

    old_connection = await pool.borrow(old_key, pool_settings(), create_old)
    await pool.release(old_connection)
    new_connection = await pool.borrow(new_key, pool_settings(), create_new)

    assert old_manager.close_count == 1
    await pool.release(new_connection)
    await pool.close()


async def test_pool_closes_borrowed_old_credentials_only_after_return() -> None:
    pool = McpConnectionPool()
    old_key = connection_key("old-fingerprint", configuration_revision=1)
    new_key = connection_key("new-fingerprint", configuration_revision=2)
    old_manager = FakeManager()

    async def create_old() -> PooledMcpConnection:
        return PooledMcpConnection(old_key, SimpleNamespace(), old_manager, 0)  # type: ignore[arg-type]

    async def create_new() -> PooledMcpConnection:
        return PooledMcpConnection(new_key, SimpleNamespace(), FakeManager(), 0)  # type: ignore[arg-type]

    old_connection = await pool.borrow(old_key, pool_settings(), create_old)
    new_connection = await pool.borrow(new_key, pool_settings(), create_new)

    assert old_manager.close_count == 0
    await pool.release(old_connection)
    assert old_manager.close_count == 1

    await pool.release(new_connection)
    await pool.close()


async def test_transport_cleanup_survives_repeated_request_cancellation() -> None:
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_cancelled = False

    async def cleanup() -> None:
        nonlocal cleanup_cancelled
        cleanup_started.set()

        try:
            await allow_cleanup.wait()
        except asyncio.CancelledError:
            cleanup_cancelled = True

            raise

    finalizer = asyncio.create_task(_await_cancellation_safe_cleanup(cleanup()))
    await cleanup_started.wait()
    finalizer.cancel()
    await asyncio.sleep(0)
    finalizer.cancel()
    await asyncio.sleep(0)

    assert not finalizer.done()
    assert not cleanup_cancelled

    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await finalizer
    assert not cleanup_cancelled


async def test_transport_owner_cleans_up_in_its_connect_task_after_background_cancellation() -> None:
    class TaskAffineServer:
        name = "fixture"

        def __init__(self) -> None:
            self.owner_task: asyncio.Task[object] | None = None
            self.cleanup_count = 0

        async def connect(self) -> None:
            self.owner_task = asyncio.current_task()

        async def cleanup(self) -> None:
            assert asyncio.current_task() is self.owner_task
            self.cleanup_count += 1

    fake_server = TaskAffineServer()
    manager = TaskAffineMcpConnectionManager(cast(MCPServerStreamableHttp, fake_server), 1.0)
    await manager.__aenter__()
    owner_task = manager._owner_task

    if owner_task is None:
        raise AssertionError("MCP owner task was not created")

    owner_task.cancel()
    await manager.cleanup_all()

    assert fake_server.cleanup_count == 1
    assert manager.active_servers == []


async def test_transport_owner_finishes_cleanup_after_connection_failure() -> None:
    class FailingServer:
        name = "fixture"

        def __init__(self) -> None:
            self.owner_task: asyncio.Task[object] | None = None
            self.cleanup_completed = False

        async def connect(self) -> None:
            self.owner_task = asyncio.current_task()

            raise ConnectionError("fixture unavailable")

        async def cleanup(self) -> None:
            assert asyncio.current_task() is self.owner_task
            await asyncio.sleep(0)
            self.cleanup_completed = True

    fake_server = FailingServer()
    manager = TaskAffineMcpConnectionManager(cast(MCPServerStreamableHttp, fake_server), 1.0)

    with pytest.raises(ConnectionError, match="fixture unavailable"):
        await manager.__aenter__()

    assert fake_server.cleanup_completed
    assert manager.active_servers == []
