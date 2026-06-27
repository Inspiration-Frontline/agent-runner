import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RequestLifecycle:
    """
    Lifecycle tracking for a single request.

    Contains metadata about a request including IDs for tracking
    and logging purposes.

    Attributes:
        request_id: Unique identifier for this request.
        agent_id: ID of the agent handling this request.
        user_id: ID of the user making this request.
    """

    request_id: str
    agent_id: str
    user_id: str


class LifecycleManager:
    """
    Manager for request lifecycle tracking.

    Tracks active requests, provides request scope management,
    and handles request registration and cleanup.

    Attributes:
        _active_requests: Dictionary mapping request IDs to lifecycle instances.
    """

    def __init__(self):
        """
        Initialize the lifecycle manager with empty request tracking.
        """
        self._active_requests: dict[str, RequestLifecycle] = {}

    def register(self, lifecycle: RequestLifecycle):
        """
        Register a new request lifecycle.

        Args:
            lifecycle: The lifecycle instance to register.
        """
        self._active_requests[lifecycle.request_id] = lifecycle
        logger.info(f"Request registered: {lifecycle.request_id}")

    def unregister(self, request_id: str):
        """
        Unregister a request lifecycle.

        Args:
            request_id: ID of the request to unregister.
        """
        if request_id in self._active_requests:
            del self._active_requests[request_id]
            logger.info(f"Request unregistered: {request_id}")

    def get_active_requests(self) -> list[RequestLifecycle]:
        """
        Get all active request lifecycles.

        Returns:
            list[RequestLifecycle]: List of all active requests.
        """
        return list(self._active_requests.values())

    def get_request(self, request_id: str) -> RequestLifecycle | None:
        """
        Get a specific request lifecycle by ID.

        Args:
            request_id: ID of the request to retrieve.

        Returns:
            RequestLifecycle | None: The lifecycle if found, None otherwise.
        """
        return self._active_requests.get(request_id)

    @asynccontextmanager
    async def request_scope(self, lifecycle: RequestLifecycle) -> AsyncGenerator[RequestLifecycle]:
        """
        Context manager for scoped request lifecycle management.

        Automatically registers the request on entry and unregisters
        on exit, handling cancellation and errors appropriately.

        Args:
            lifecycle: The lifecycle instance to manage.

        Yields:
            RequestLifecycle: The lifecycle for use within the scope.
        """
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
