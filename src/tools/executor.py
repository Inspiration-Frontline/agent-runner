import asyncio
import logging
from functools import partial
from typing import Any

from runtime.cancellation import CancellationToken
from tools.registry import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executor for running tool implementations.

    Handles tool execution with support for both async and sync handlers,
    cancellation tokens, and batch execution of multiple tool calls.

    Attributes:
        registry: Tool registry containing available tools.
    """

    def __init__(self, registry: ToolRegistry | None = None):
        """
        Initialize the tool executor.

        Args:
            registry: Optional tool registry (default: creates new registry).
        """
        self.registry = registry or ToolRegistry()

    async def execute(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Any:
        """
        Execute a single tool with given arguments.

        Args:
            tool_id: ID of the tool to execute.
            arguments: Arguments to pass to the tool.
            cancellation_token: Optional token for execution cancellation.

        Returns:
            Any: The result of tool execution.

        Raises:
            ValueError: If tool is not found.
            asyncio.CancelledError: If execution is cancelled.
        """
        tool = self.registry.get(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError("Tool execution cancelled")

        logger.info(f"Executing tool: {tool_id} with arguments: {arguments}")

        try:
            if tool.handler:
                if asyncio.iscoroutinefunction(tool.handler):
                    result = await tool.handler(**arguments)
                else:
                    handler = partial(tool.handler, **arguments)
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, handler
                    )
            else:
                result = await self._execute_tool(tool, arguments)

            logger.info(f"Tool {tool_id} executed successfully")
            return result

        except asyncio.CancelledError:
            logger.info(f"Tool {tool_id} execution cancelled")
            raise
        except Exception:
            logger.exception(f"Error executing tool {tool_id}")
            raise

    async def _execute_tool(self, tool: ToolDefinition, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool without a handler (placeholder implementation).

        Args:
            tool: The tool definition to execute.
            arguments: Arguments to pass to the tool.

        Returns:
            Any: Placeholder result indicating tool is not implemented.
        """
        return {"status": "not_implemented", "tool": tool.tool_id}

    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        cancellation_token: CancellationToken | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute multiple tool calls in batch.

        Args:
            tool_calls: List of tool call specifications.
            cancellation_token: Optional token for batch cancellation.

        Returns:
            list[dict[str, Any]]: List of execution results with status.
        """
        results = []
        for call in tool_calls:
            if cancellation_token and cancellation_token.is_cancelled():
                break

            tool_id = call.get("tool_id") or call.get("name")
            arguments = call.get("arguments", {})

            try:
                result = await self.execute(tool_id, arguments, cancellation_token)
                results.append({
                    "tool_id": tool_id,
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "tool_id": tool_id,
                    "status": "error",
                    "error": str(e),
                })

        return results
