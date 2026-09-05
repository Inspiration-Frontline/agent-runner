"""Tracing adapters for MCP discovery and tool dispatch.

This module deliberately owns OpenTelemetry span construction so the transport and durable
dispatch code can concentrate on connection and persistence behavior.
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from agent_runner.observability.tracing import Tracer

_Result = TypeVar("_Result")


class McpTracedOperations:
    """Decorates MCP operations with tracing without leaking span mechanics into runtime logic.

    Attributes:
        _tracer: Application-owned OpenTelemetry facade used by this component.
    """

    def __init__(self, tracer: Tracer) -> None:
        """Create an adapter over the application's configured tracer.

        Args:
            tracer: Application-owned OpenTelemetry facade.
        """
        self._tracer = tracer

    async def discover(self, server_id: str, operation: Callable[[], Awaitable[_Result]]) -> _Result:
        """Execute one ``tools/list`` round trip inside its discovery span.

        Args:
            server_id: Stable MCP Catalog server identifier.
            operation: Asynchronous boundary operation to trace and invoke.

        Returns:
            The provider result returned by the traced ``tools/list`` operation.
        """
        with self._tracer.span("mcp.discovery") as span:
            span.set_attribute("mcp.server.id", server_id)

            return await operation()

    async def dispatch(
        self,
        server_id: str,
        tool_name: str,
        tool_call_id: str,
        attempt_id: str,
        operation: Callable[[], Awaitable[_Result]],
    ) -> _Result:
        """Execute remote delivery of one model-selected MCP tool call inside its span.

        Args:
            server_id: Stable MCP Catalog server identifier.
            tool_name: Provider-visible Tool name.
            tool_call_id: Provider-generated Tool call identifier.
            attempt_id: Unique identifier of the remote delivery attempt.
            operation: Asynchronous boundary operation to trace and invoke.

        Returns:
            The remote Tool result returned by the traced dispatch operation.
        """
        with self._tracer.span("mcp.dispatch") as span:
            span.set_attribute("gen_ai.tool.call.id", tool_call_id)
            span.set_attribute("mcp.server.id", server_id)
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.dispatch.attempt_id", attempt_id)

            return await operation()

    async def record_result(
        self,
        server_id: str,
        tool_name: str,
        tool_call_id: str,
        attempt_id: str,
        state: str,
        operation: Callable[[], Awaitable[_Result]],
        error_type: str = "",
    ) -> _Result:
        """Execute durable terminal-result recording inside its result span.

        Args:
            server_id: Stable MCP Catalog server identifier.
            tool_name: Provider-visible Tool name.
            tool_call_id: Provider-generated Tool call identifier.
            attempt_id: Unique identifier of the remote delivery attempt.
            state: Request-scoped mutable orchestration or persistence state.
            operation: Asynchronous boundary operation to trace and invoke.
            error_type: Domain error type value used by the operation.

        Returns:
            The persistence result returned by the traced terminal-recording operation.
        """
        with self._tracer.span("mcp.result") as span:
            span.set_attribute("gen_ai.tool.call.id", tool_call_id)
            span.set_attribute("mcp.server.id", server_id)
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.dispatch.attempt_id", attempt_id)
            span.set_attribute("mcp.result.state", state)

            if error_type:
                span.set_attribute("error.type", error_type)

            return await operation()
