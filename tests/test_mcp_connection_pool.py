import asyncio
from types import SimpleNamespace

from agent_runner.mcps.connection_pool import (
    McpConnectionKey,
    McpConnectionPool,
    McpConnectionPoolSettings,
    PooledMcpConnection,
)


class FakeManager:
    def __init__(self) -> None:
        self.close_count = 0

    async def cleanup_all(self) -> None:
        self.close_count += 1


def pool_settings(max_connections: int = 4, idle_timeout: float = 300.0) -> McpConnectionPoolSettings:
    return McpConnectionPoolSettings(max_connections, idle_timeout, 1.0)


async def test_pool_reuses_a_returned_exclusive_connection() -> None:
    pool = McpConnectionPool()
    key = McpConnectionKey("fixture", "https://example.test/mcp", "fingerprint")
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
    key = McpConnectionKey("fixture", "https://example.test/mcp", "fingerprint")

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
    key = McpConnectionKey("fixture", "https://example.test/mcp", "fingerprint")
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
    key = McpConnectionKey("fixture", "https://example.test/mcp", "fingerprint")
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
