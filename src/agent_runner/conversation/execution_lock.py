import asyncio
from contextlib import suppress
from uuid import uuid4

import redis.asyncio as aioredis

from agent_runner.config import get_settings


class ConversationBusyError(RuntimeError):
    pass


class ConversationExecutionLock:
    _LEASE_MS = 180_000
    _RENEW_INTERVAL_SECONDS = 30
    _RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""
    _RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""

    def __init__(self) -> None:
        settings = get_settings()
        self._redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
        )
        self._key: str | None = None
        self._token: str | None = None
        self._renewal_task: asyncio.Task[None] | None = None

    async def acquire(self, conversation_id: str) -> None:
        key = f"agent-runner:execution:{conversation_id}"
        token = str(uuid4())
        acquired = await self._redis.set(key, token, nx=True, px=self._LEASE_MS)
        if not acquired:
            raise ConversationBusyError("Another request is already running for this conversation.")
        self._key = key
        self._token = token
        self._renewal_task = asyncio.create_task(self._renew())

    async def _renew(self) -> None:
        while True:
            await asyncio.sleep(self._RENEW_INTERVAL_SECONDS)
            if self._key is None or self._token is None:
                return
            renewed = await self._redis.eval(
                self._RENEW_SCRIPT, 1, self._key, self._token, self._LEASE_MS
            )
            if renewed != 1:
                return

    async def release(self) -> None:
        if self._renewal_task is not None:
            self._renewal_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._renewal_task
            self._renewal_task = None
        if self._key is not None and self._token is not None:
            await self._redis.eval(self._RELEASE_SCRIPT, 1, self._key, self._token)
        self._key = None
        self._token = None

    async def close(self) -> None:
        await self.release()
        await self._redis.aclose()
