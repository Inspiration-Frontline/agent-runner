from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class RuntimeEventType(str, Enum):
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
    event_id: str
    event_type: RuntimeEventType
    timestamp: datetime
    data: dict[str, Any]
    request_id: str

    @classmethod
    def create(cls, event_type: RuntimeEventType, data: dict[str, Any], request_id: str) -> "RuntimeEvent":
        return cls(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            data=data,
            request_id=request_id,
        )


class RuntimeEventBus:
    def __init__(self):
        self._handlers: dict[RuntimeEventType, list] = {
            event_type: [] for event_type in RuntimeEventType
        }

    def subscribe(self, event_type: RuntimeEventType, handler):
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: RuntimeEventType, handler):
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: RuntimeEvent):
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                pass


import asyncio
