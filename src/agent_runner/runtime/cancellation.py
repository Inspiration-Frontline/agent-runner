import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

CancellationCallback = Callable[[], None | Awaitable[None]]


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
    _callbacks: list[CancellationCallback] = field(default_factory=list, init=False)

    def cancel(self) -> None:
        """Publish the stop signal used by every long-running operation in this request.

        The model adapter, tool executor, and stream loop all retain the same token.  Setting the
        flag before invoking callbacks makes cancellation checkpoints observe the stop request even
        when a callback itself takes time.  Callbacks are invoked only on the first transition;
        this is important because the HTTP Stop endpoint retries briefly and must remain idempotent.
        Synchronous callbacks run inline, while coroutine callbacks are scheduled on the current
        event loop so closing a provider client cannot block the response to the Stop endpoint.

        Returns:
            ``None``.  Call :meth:`is_cancelled` when the caller needs to inspect the state.
        """
        if self._cancelled:
            return
        self._cancelled = True
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback())
            else:
                callback()

    def is_cancelled(self) -> bool:
        """Report whether model/tool execution should stop at its next cancellation checkpoint.

        Returns:
            ``True`` after :meth:`cancel` has been signalled; otherwise ``False``.
        """
        return self._cancelled

    def add_callback(self, callback: CancellationCallback) -> None:
        """Attach cleanup for an SDK or tool resource that must stop with this request.

        A callback added after cancellation is executed immediately (or scheduled when async),
        which closes the race between provider startup and a user's Stop click.  This avoids
        retaining a callback that can never receive the signal.

        Args:
            callback: Synchronous or async zero-argument cleanup function.

        Returns:
            ``None``.  The callback is retained until cancellation or explicit removal.
        """
        if self._cancelled:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback())
            else:
                callback()
            return
        self._callbacks.append(callback)

    def remove_callback(self, callback: CancellationCallback) -> None:
        """Release one callback after its owning provider or tool has finished.

        Removing callbacks is part of request teardown: retaining closures here would keep model
        clients and request state alive until the token itself is garbage-collected.

        Args:
            callback: Previously registered callback object; unknown callbacks are ignored.

        Returns:
            ``None``.
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

    def __init__(self) -> None:
        """Create the compatibility token manager used by isolated runtime callers.

        Conversation cancellation uses :class:`ConversationCancellationRegistry`; this manager
        remains for request-local token lookup and cleanup in older integrations.
        """
        # Key: caller-supplied token ID. Value: request-local cooperative cancellation token.
        self._tokens: dict[str, CancellationToken] = {}

    def create_token(self, token_id: str | None = None) -> CancellationToken:
        """Create and retain a token under a stable lookup ID.

        Args:
            token_id: Optional integration-supplied ID; a UUID is generated when omitted.

        Returns:
            Newly created token.
        """
        import uuid

        effective_token_id: str = token_id or str(uuid.uuid4())
        token: CancellationToken = CancellationToken()
        self._tokens[effective_token_id] = token
        return token

    def get_token(self, token_id: str) -> CancellationToken | None:
        """Look up a token for legacy callers that address requests by token ID.

        Args:
            token_id: Previously assigned token ID.

        Returns:
            Token when still active, otherwise ``None``.
        """
        return self._tokens.get(token_id)

    def cancel_token(self, token_id: str) -> None:
        """Signal cancellation for a token addressed by its compatibility ID.

        Args:
            token_id: Previously assigned token ID.
        """
        token: CancellationToken | None = self._tokens.get(token_id)
        if token:
            token.cancel()

    async def cleanup(self, token: CancellationToken) -> None:
        """Cancel and yield once so scheduled callbacks can finish before request teardown.

        Args:
            token: Request token being removed from active execution.
        """
        token.cancel()
        await asyncio.sleep(0)

    def remove_token(self, token_id: str) -> None:
        """Forget a compatibility token after its request has terminated.

        Args:
            token_id: Previously assigned token ID.
        """
        if token_id in self._tokens:
            del self._tokens[token_id]


class ConversationCancellationRegistry:
    """Tracks active generations by authenticated user and Conversation.

    The HTTP cancel endpoint does not own the task object that is executing the model stream. This
    registry is the narrow bridge between that endpoint and the request-scoped orchestrator. The
    user ID is part of the key so a caller cannot cancel another user's Conversation by guessing
    its identifier, and identity-checked unregistering prevents an old request from deleting a
    newer request's token after a retry.

    Attributes:
        _tokens: Collection of tokens consumed in deterministic order.
    """

    def __init__(self) -> None:
        """Create an empty registry for active request cancellation tokens.

        Tokens are intentionally in-memory: cancellation is a best-effort control signal for the
        current Runner process, while durable Round status is persisted by the orchestrator.
        """
        # Key: authenticated user ID and Conversation ID. Value: active generation cancellation token.
        self._tokens: dict[tuple[int, str], CancellationToken] = {}

    def register(self, user_id: int, conversation_id: str, token: CancellationToken) -> None:
        """Make an in-flight generation discoverable by the HTTP Stop endpoint.

        The cancel route runs in a different coroutine from the model stream and cannot access the
        orchestrator's local token directly.  This registry is therefore the process-local control
        plane between those two requests; it does not persist conversation state or replace the
        durable Round status written by the orchestrator.  Registration is performed after the
        per-conversation execution lease is acquired, so a key represents the one generation that
        is allowed to run.  Assignment intentionally replaces an older token because a retry may
        start immediately after the prior request releases its lease.

        Args:
            user_id: Trusted gateway identity.  Including it in the key prevents one user from
                cancelling another user's conversation by guessing its ID.
            conversation_id: Stable conversation identifier used by the public cancel API.
            token: Token shared with the stream, provider adapter, and tool executor.  Its
                callbacks perform the actual best-effort interruption.

        Returns:
            ``None``.  The token can subsequently be addressed through :meth:`cancel`.
        """
        self._tokens[(user_id, conversation_id)] = token

    def cancel(self, user_id: int, conversation_id: str) -> bool:
        """Signal the active generation without touching durable conversation data.

        A missing token means the generation has not registered yet or has already entered
        teardown.  Returning ``False`` lets the HTTP route remain idempotent when a user presses
        Stop after completion; the route may then abort its client connection as a fallback.

        Args:
            user_id: Trusted gateway identity used to isolate cancellation by owner.
            conversation_id: Stable conversation identifier from the Stop request.

        Returns:
            ``True`` when a token was found and signalled, otherwise ``False`` when no generation
            is currently registered.
        """
        token: CancellationToken | None = self._tokens.get((user_id, conversation_id))
        if token is None:
            return False
        token.cancel()
        return True

    def unregister(self, user_id: int, conversation_id: str, token: CancellationToken) -> None:
        """Remove this request's registration after stream teardown, preserving replacements.

        Cleanup runs from a ``finally`` block and can race a fast retry.  Comparing object identity
        rather than only the ``(user_id, conversation_id)`` key prevents the old request from
        deleting the replacement token and making the new Stop request ineffective.  The registry
        deliberately forgets completed tokens because cancellation is only meaningful while work
        is executing; the persisted Round status remains the source of truth afterwards.

        Args:
            user_id: Trusted gateway identity used to locate the isolated entry.
            conversation_id: Stable conversation identifier used by the cancel route.
            token: Exact token previously registered by this request; a different token indicates
                that a newer generation now owns the key.

        Returns:
            ``None``.
        """
        key: tuple[int, str] = (user_id, conversation_id)
        if self._tokens.get(key) is token:
            del self._tokens[key]
