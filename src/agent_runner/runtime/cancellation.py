import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CancellationToken:
    """
    Token for request cancellation management.

    Tracks cancellation status and manages callbacks that should
    be executed when cancellation is triggered.

    Attributes:
        _cancelled: Whether this token has been cancelled.
        _callbacks: List of callbacks to execute on cancellation.
    """

    _cancelled: bool = field(default=False, init=False)
    _callbacks: list[Any] = field(default_factory=list, init=False)

    def cancel(self):
        """
        Trigger cancellation and execute all registered callbacks.
        """
        self._cancelled = True
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback())
            else:
                callback()

    def is_cancelled(self) -> bool:
        """
        Check if this token has been cancelled.

        Returns:
            bool: True if cancelled, False otherwise.
        """
        return self._cancelled

    def add_callback(self, callback: Any):
        """
        Add a callback to be executed on cancellation.

        Args:
            callback: The callback function to add.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Any):
        """
        Remove a previously registered callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)


class CancellationManager:
    """
    Manager for cancellation tokens.

    Provides centralized management of cancellation tokens,
    supporting creation, retrieval, cancellation, and cleanup.

    Attributes:
        _tokens: Dictionary mapping token IDs to cancellation tokens.
    """

    def __init__(self):
        """
        Initialize the cancellation manager.
        """
        self._tokens: dict[str, CancellationToken] = {}

    def create_token(self, token_id: str | None = None) -> CancellationToken:
        """
        Create a new cancellation token.

        Args:
            token_id: Optional ID for the token (auto-generated if None).

        Returns:
            CancellationToken: The created token.
        """
        import uuid

        token_id = token_id or str(uuid.uuid4())
        token = CancellationToken()
        self._tokens[token_id] = token
        return token

    def get_token(self, token_id: str) -> CancellationToken | None:
        """
        Retrieve a cancellation token by ID.

        Args:
            token_id: ID of the token to retrieve.

        Returns:
            CancellationToken | None: The token if found, None otherwise.
        """
        return self._tokens.get(token_id)

    def cancel_token(self, token_id: str):
        """
        Cancel a token by ID.

        Args:
            token_id: ID of the token to cancel.
        """
        token = self._tokens.get(token_id)
        if token:
            token.cancel()

    async def cleanup(self, token: CancellationToken):
        """
        Cleanup a cancellation token.

        Args:
            token: The token to cleanup.
        """
        token.cancel()
        await asyncio.sleep(0)

    def remove_token(self, token_id: str):
        """
        Remove a token from management.

        Args:
            token_id: ID of the token to remove.
        """
        if token_id in self._tokens:
            del self._tokens[token_id]
