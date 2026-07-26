import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

logger: logging.Logger = logging.getLogger(__name__)


class RuntimeEventType(StrEnum):
    """
    Enumeration of runtime event types.

    Defines the different types of events that occur during
    agent request processing and execution.

    Attributes:
        REQUEST_START: Event indicating a request has started.
        CONTEXT_BUILT: Event indicating context building has completed.
        AGENT_LOADED: Event indicating agent configuration has been loaded.
        TOOL_CALLED: Event indicating a tool has been called.
        MODEL_RESPONSE: Event indicating a model response has been received.
        STREAM_COMPLETE: Event indicating streaming has completed.
        ERROR: Event indicating an error has occurred.
        REQUEST_END: Event indicating a request has ended.
    """

    REQUEST_START = "request_start"
    CONTEXT_BUILT = "context_built"
    AGENT_LOADED = "agent_loaded"
    TOOL_CALLED = "tool_called"
    MODEL_RESPONSE = "model_response"
    STREAM_COMPLETE = "stream_complete"
    ERROR = "error"
    REQUEST_END = "request_end"


@dataclass
class RuntimeEvent:
    """
    Event representing a runtime operation.

    Contains event metadata including ID, type, timestamp, and associated request ID. Event-specific
    payloads must use dedicated typed event classes instead of adding an arbitrary data dictionary.

    Attributes:
        event_id: Unique identifier for this event.
        event_type: Type of this runtime event.
        timestamp: Timestamp when this event occurred.
        request_id: ID of the request this event belongs to.
    """

    event_id: str
    event_type: RuntimeEventType
    timestamp: datetime
    request_id: str

    @classmethod
    def create(cls, event_type: RuntimeEventType, request_id: str) -> "RuntimeEvent":
        """
        Create a new runtime event with auto-generated ID and timestamp.

        Args:
            event_type: Type of the event to create.
            request_id: ID of the request for this event.

        Returns:
            RuntimeEvent: The newly created event.
        """
        return cls(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.now(UTC),
            request_id=request_id,
        )


type RuntimeEventHandler = Callable[[RuntimeEvent], None | Awaitable[None]]


class RuntimeEventBus:
    """
    Event bus for publishing and subscribing to runtime events.

    Provides a pub/sub mechanism for runtime events, allowing
    components to subscribe to specific event types and react
    to events as they occur.

    Attributes:
        _handlers: Dictionary mapping event types to handler lists.
    """

    def __init__(self) -> None:
        """
        Initialize the event bus with empty handler lists.
        """
        self._handlers: dict[RuntimeEventType, list[RuntimeEventHandler]] = {
            event_type: [] for event_type in RuntimeEventType
        }

    def subscribe(self, event_type: RuntimeEventType, handler: RuntimeEventHandler) -> None:
        """
        Subscribe a handler to a specific event type.

        Args:
            event_type: Type of events to subscribe to.
            handler: Handler function to execute on events.
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: RuntimeEventType, handler: RuntimeEventHandler) -> None:
        """
        Unsubscribe a handler from a specific event type.

        Args:
            event_type: Type of events to unsubscribe from.
            handler: Handler function to remove.
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: RuntimeEvent) -> None:
        """
        Publish an event to all subscribed handlers.

        Args:
            event: The event to publish.
        """
        handlers: list[RuntimeEventHandler] = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Runtime event handler failed: event_type=%s handler=%r",
                    event.event_type,
                    handler,
                )
