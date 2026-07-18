import asyncio
import json
from dataclasses import dataclass, field
from time import time_ns
from typing import Any

from agents.tool_context import ToolContext

from agent_runner.runtime.cancellation import CancellationToken
from agent_runner.tools.registry import ToolDefinition


def epoch_millis() -> int:
    return time_ns() // 1_000_000


@dataclass(frozen=True)
class CapturedToolCall:
    tool_call_id: str
    tool_name: str
    arguments: str


@dataclass
class CapturedToolExecution:
    tool_call_id: str
    tool_key: str
    tool_name: str
    arguments: str
    status: str
    result_content: str
    raw_result: str
    error_message: str
    start_time: int
    end_time: int


@dataclass
class CapturedModelTurn:
    request_messages: list[dict[str, Any]]
    message_storage_mode: str
    tools: list[ToolDefinition]
    response_content: str
    response_tool_calls: list[CapturedToolCall]
    tool_executions: list[CapturedToolExecution]
    request_id: str
    trace_id: str
    start_time: int
    llm_end_time: int
    end_time: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_request: str
    raw_response: str


@dataclass
class AgentRunCapture:
    turns: list[CapturedModelTurn] = field(default_factory=list)
    final_output: str = ""


class ToolExecutionCollector:
    """Request-scoped audit collector used by concurrently executing SDK Tool handlers."""

    def __init__(self) -> None:
        self._executions: dict[str, CapturedToolExecution] = {}
        self._calls: dict[str, CapturedToolCall] = {}

    def record_call(self, tool_call: CapturedToolCall) -> None:
        """Retain model-emitted calls even if SDK cancellation later clears raw_responses output."""
        self._calls.setdefault(tool_call.tool_call_id, tool_call)

    async def execute(
        self,
        *,
        tool_call_id: str,
        definition: ToolDefinition,
        arguments_json: str,
        tool_context: ToolContext[Any],
        cancellation_token: CancellationToken | None,
    ) -> str:
        start_time = epoch_millis()
        try:
            if cancellation_token is not None and cancellation_token.is_cancelled():
                raise asyncio.CancelledError("Tool execution cancelled")
            if definition.function_tool is None:
                raise ValueError(f"Tool has no SDK FunctionTool: {definition.tool_key}")
            result = await definition.function_tool.on_invoke_tool(tool_context, arguments_json)
            result_content = json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            self._executions[tool_call_id] = CapturedToolExecution(
                tool_call_id=tool_call_id,
                tool_key=definition.tool_key,
                tool_name=definition.tool_name,
                arguments=arguments_json,
                status="COMPLETED",
                result_content=result_content,
                raw_result=result_content,
                error_message="",
                start_time=start_time,
                end_time=max(start_time, epoch_millis()),
            )
            return result_content
        except asyncio.CancelledError:
            self._executions[tool_call_id] = CapturedToolExecution(
                tool_call_id=tool_call_id,
                tool_key=definition.tool_key,
                tool_name=definition.tool_name,
                arguments=arguments_json,
                status="CANCELLED",
                result_content="",
                raw_result="",
                error_message="Generation cancelled.",
                start_time=start_time,
                end_time=max(start_time, epoch_millis()),
            )
            raise
        except Exception as error:
            # A single Tool failure is returned to the model so sibling calls and the Agent loop
            # can continue. The persisted execution still records the failure independently.
            error_message = str(error) or type(error).__name__
            result_content = json.dumps(
                {"status": "error", "error": error_message},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._executions[tool_call_id] = CapturedToolExecution(
                tool_call_id=tool_call_id,
                tool_key=definition.tool_key,
                tool_name=definition.tool_name,
                arguments=arguments_json,
                status="FAILED",
                result_content=result_content,
                raw_result=result_content,
                error_message=error_message,
                start_time=start_time,
                end_time=max(start_time, epoch_millis()),
            )
            return result_content

    def get(self, tool_call_id: str) -> CapturedToolExecution | None:
        return self._executions.get(tool_call_id)

    def values(self) -> list[CapturedToolExecution]:
        return list(self._executions.values())

    def calls(self) -> list[CapturedToolCall]:
        return list(self._calls.values())
