import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class RequestLifecycle:
    request_id: str
    agent_id: str
    user_id: str


class LifecycleManager:
    def __init__(self):
        self._active_requests: dict[str, RequestLifecycle] = {}

    def register(self, lifecycle: RequestLifecycle):
        self._active_requests[lifecycle.request_id] = lifecycle
        logger.info(f"Request registered: {lifecycle.request_id}")

    def unregister(self, request_id: str):
        if request_id in self._active_requests:
            del self._active_requests[request_id]
            logger.info(f"Request unregistered: {request_id}")

    def get_active_requests(self) -> list[RequestLifecycle]:
        return list(self._active_requests.values())

    def get_request(self, request_id: str) -> RequestLifecycle | None:
        return self._active_requests.get(request_id)

    @asynccontextmanager
    async def request_scope(self, lifecycle: RequestLifecycle) -> AsyncGenerator[RequestLifecycle, None]:
        self.register(lifecycle)
        try:
            yield lifecycle
        except asyncio.CancelledError:
            logger.info(f"Request cancelled: {lifecycle.request_id}")
            raise
        except Exception as e:
            logger.exception(f"Request failed: {lifecycle.request_id}, error: {e}")
            raise
        finally:
            self.unregister(lifecycle.request_id)
