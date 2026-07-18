import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from agents.tool_context import ToolContext

from agent_runner.runtime.cancellation import CancellationToken
from agent_runner.tools.registry import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executor for running tool implementations.

    Invokes SDK-decorated FunctionTools outside an Agent run, with cancellation
    tokens and concurrent batch execution for tests and non-SDK callers.

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
        tool_key: str,
        arguments: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Any:
        """
        Execute a single tool with given arguments.

        Args:
            tool_key: Globally unique key of the Tool to execute.
            arguments: Arguments to pass to the tool.
            cancellation_token: Optional token for execution cancellation.

        Returns:
            Any: The result of tool execution.

        Raises:
            ValueError: If tool is not found.
            asyncio.CancelledError: If execution is cancelled.
        """
        tool = self.registry.get(tool_key)
        if not tool:
            raise ValueError(f"Tool not found: {tool_key}")

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError("Tool execution cancelled")

        logger.info("Executing Tool %s with arguments: %s", tool_key, arguments)

        try:
            if tool.function_tool is None:
                result = await self._execute_tool(tool, arguments)
            else:
                arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                context = ToolContext(
                    context=None,
                    tool_name=tool.tool_name,
                    tool_call_id=str(uuid4()),
                    tool_arguments=arguments_json,
                )
                result = await tool.function_tool.on_invoke_tool(context, arguments_json)

            logger.info("Tool %s executed successfully", tool_key)
            return result

        except asyncio.CancelledError:
            logger.info("Tool %s execution cancelled", tool_key)
            raise
        except Exception:
            logger.exception("Error executing Tool %s", tool_key)
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
        return {"status": "not_implemented", "tool_key": tool.tool_key}

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
        async def execute_call(call: dict[str, Any]) -> dict[str, Any]:
            tool_key = call.get("tool_key") or call.get("name")
            arguments = call.get("arguments", {})
            try:
                result = await self.execute(tool_key, arguments, cancellation_token)
                return {
                    "tool_key": tool_key,
                    "status": "success",
                    "result": result,
                }
            except Exception as e:
                return {
                    "tool_key": tool_key,
                    "status": "error",
                    "error": str(e),
                }

        tasks = [asyncio.create_task(execute_call(call)) for call in tool_calls]

        def cancel_tasks() -> None:
            for task in tasks:
                task.cancel()

        if cancellation_token is not None:
            cancellation_token.add_callback(cancel_tasks)
        try:
            # Results retain model-emitted order while execution happens concurrently.
            return await asyncio.gather(*tasks)
        finally:
            if cancellation_token is not None:
                cancellation_token.remove_callback(cancel_tasks)
