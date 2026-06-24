from enum import Enum
from typing import Any

from pydantic import BaseModel


class StreamEventType(str, Enum):
    """
    Enumeration of stream event types for SSE responses.

    Defines the different types of events that can be sent through
    the Server-Sent Events (SSE) stream during agent execution.

    Attributes:
        TOKEN_DELTA: Event containing a text token delta from the model.
        TOOL_START: Event indicating a tool execution has started.
        TOOL_RESULT: Event containing the result of a tool execution.
        ERROR: Event indicating an error occurred during execution.
        DONE: Event indicating the stream has completed.
    """

    TOKEN_DELTA = "token_delta"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    """
    Base model for all stream events.

    Provides a common structure for all events sent through the SSE stream,
    with optional fields for different event types.

    Attributes:
        type: The type of this stream event.
        content: Text content for TOKEN_DELTA events.
        tool: Tool identifier for TOOL_START and TOOL_RESULT events.
        tool_args: Tool arguments for TOOL_START events.
        tool_result: Tool execution result for TOOL_RESULT events.
        error_message: Error description for ERROR events.
    """

    type: StreamEventType
    content: str | None = None
    tool: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    error_message: str | None = None


class TokenDeltaEvent(StreamEvent):
    """
    Event containing a text token delta from the model.

    Sent during streaming to provide incremental text output from the agent.
    """

    def __init__(self, content: str):
        """
        Initialize a token delta event.

        Args:
            content: The text token delta to send.
        """
        super().__init__(type=StreamEventType.TOKEN_DELTA, content=content)


class ToolStartEvent(StreamEvent):
    """
    Event indicating a tool execution has started.

    Sent when the agent begins executing a tool, providing the tool name
    and arguments being used.
    """

    def __init__(self, tool: str, tool_args: dict[str, Any] | None = None):
        """
        Initialize a tool start event.

        Args:
            tool: The identifier of the tool being executed.
            tool_args: Optional arguments passed to the tool.
        """
        super().__init__(type=StreamEventType.TOOL_START, tool=tool, tool_args=tool_args)


class ToolResultEvent(StreamEvent):
    """
    Event containing the result of a tool execution.

    Sent after a tool completes execution, providing the result data
    that can be used by the agent.
    """

    def __init__(self, tool: str, tool_result: Any):
        """
        Initialize a tool result event.

        Args:
            tool: The identifier of the tool that was executed.
            tool_result: The result returned by the tool execution.
        """
        super().__init__(type=StreamEventType.TOOL_RESULT, tool=tool, tool_result=tool_result)


class ErrorEvent(StreamEvent):
    """
    Event indicating an error occurred during execution.

    Sent when an error occurs during agent execution, providing
    a descriptive error message.
    """

    def __init__(self, error_message: str):
        """
        Initialize an error event.

        Args:
            error_message: Description of the error that occurred.
        """
        super().__init__(type=StreamEventType.ERROR, error_message=error_message)


class DoneEvent(StreamEvent):
    """
    Event indicating the stream has completed.

    Sent as the final event in the stream to signal that all
    processing has finished.
    """

    def __init__(self):
        """
        Initialize a done event.
        """
        super().__init__(type=StreamEventType.DONE)
