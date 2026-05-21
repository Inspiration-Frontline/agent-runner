from enum import Enum
from typing import Any

from pydantic import BaseModel


class StreamEventType(str, Enum):
    TOKEN_DELTA = "token_delta"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    type: StreamEventType
    content: str | None = None
    tool: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    error_message: str | None = None


class TokenDeltaEvent(StreamEvent):
    def __init__(self, content: str):
        super().__init__(type=StreamEventType.TOKEN_DELTA, content=content)


class ToolStartEvent(StreamEvent):
    def __init__(self, tool: str, tool_args: dict[str, Any] | None = None):
        super().__init__(type=StreamEventType.TOOL_START, tool=tool, tool_args=tool_args)


class ToolResultEvent(StreamEvent):
    def __init__(self, tool: str, tool_result: Any):
        super().__init__(type=StreamEventType.TOOL_RESULT, tool=tool, tool_result=tool_result)


class ErrorEvent(StreamEvent):
    def __init__(self, error_message: str):
        super().__init__(type=StreamEventType.ERROR, error_message=error_message)


class DoneEvent(StreamEvent):
    def __init__(self):
        super().__init__(type=StreamEventType.DONE)
