import asyncio
import json
from dataclasses import dataclass, field
from time import time_ns

from agents.tool_context import ToolContext

from agent_runner.config import Settings
from agent_runner.context.builder import CapturedMessage
from agent_runner.observability.tool_tracing import ToolTracing
from agent_runner.observability.tracing import Tracer
from agent_runner.runtime.cancellation import CancellationToken
from agent_runner.tools.registry import ToolDefinition


def get_epoch_millis() -> int:
    """Return epoch milliseconds used for LLM/Tool audit boundaries.

    Returns:
        Current wall-clock epoch timestamp in milliseconds.
    """
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
    request_messages: list[CapturedMessage]
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

    def __init__(self, settings: Settings | None = None, tracer: Tracer | None = None) -> None:
        """Create a request-scoped collector for Tool calls and terminal outcomes.

        The SDK may invoke sibling Tools concurrently and may discard raw response objects during
        cancellation. Keeping calls and executions here lets persistence reconstruct a complete,
        one-to-one audit trail independently of SDK object lifetime.
        """
        # Key: model Tool call ID. Value: terminal execution result captured for persistence.
        self._executions: dict[str, CapturedToolExecution] = {}
        # Key: model Tool call ID. Value: original model-requested Tool call and arguments.
        self._calls: dict[str, CapturedToolCall] = {}
        current_settings = settings or Settings()
        self._tracing = ToolTracing(tracer or Tracer(current_settings.otel_service_name), current_settings)

    def record_call(self, tool_call: CapturedToolCall) -> None:
        """Retain a model-emitted call before its handler starts or SDK output is discarded.

        Args:
            tool_call: Stable call ID, Tool name, and model-emitted JSON arguments.
        """
        self._calls.setdefault(tool_call.tool_call_id, tool_call)

    async def execute(
        self,
        *,
        tool_call_id: str,
        definition: ToolDefinition,
        arguments_json: str,
        tool_context: ToolContext[object],
        cancellation_token: CancellationToken | None,
    ) -> str:
        """Execute one configured Tool and retain normalized terminal evidence.

        Args:
            tool_call_id: SDK call ID used to join result evidence to the model response.
            definition: Frozen Tool definition and decorated invocation hook.
            arguments_json: Exact JSON arguments emitted by the model.
            tool_context: SDK context passed to the Tool handler.
            cancellation_token: Request token checked before invocation.

        Returns:
            JSON text returned to the model for success or a normalized error object.

        Raises:
            asyncio.CancelledError: When cancellation interrupts the Tool invocation.
        """
        with self._tracing.trace_call(definition, tool_call_id, arguments_json) as trace:
            start_time = get_epoch_millis()
            try:
                if cancellation_token is not None and cancellation_token.is_cancelled():
                    raise asyncio.CancelledError("Tool execution cancelled")
                if definition.function_tool is None:
                    raise ValueError(f"Tool has no SDK FunctionTool: {definition.tool_key}")
                result = await definition.function_tool.on_invoke_tool(tool_context, arguments_json)
                result_content = json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                self._record_execution(
                    tool_call_id, definition, arguments_json, "COMPLETED", result_content, "", start_time
                )
                trace.record_result("COMPLETED", result_content)
                return result_content
            except asyncio.CancelledError:
                self._record_execution(
                    tool_call_id, definition, arguments_json, "CANCELLED", "", "Generation cancelled.", start_time
                )
                trace.record_result("CANCELLED", "Generation cancelled.")
                raise
            except Exception as error:
                # A Tool failure becomes a model-visible result so sibling calls can continue.
                error_message = str(error) or type(error).__name__
                result_content = json.dumps(
                    {"status": "error", "error": error_message},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                self._record_execution(
                    tool_call_id, definition, arguments_json, "FAILED", result_content, error_message, start_time
                )
                trace.record_result("FAILED", result_content, type(error).__name__)
                return result_content

    def _record_execution(
        self,
        tool_call_id: str,
        definition: ToolDefinition,
        arguments_json: str,
        status: str,
        result_content: str,
        error_message: str,
        start_time: int,
    ) -> None:
        """Store one normalized Tool outcome without duplicating persistence mapping."""
        self._executions[tool_call_id] = CapturedToolExecution(
            tool_call_id=tool_call_id,
            tool_key=definition.tool_key,
            tool_name=definition.tool_name,
            arguments=arguments_json,
            status=status,
            result_content=result_content,
            raw_result=result_content,
            error_message=error_message,
            start_time=start_time,
            end_time=max(start_time, get_epoch_millis()),
        )

    def record_external_execution(
        self,
        tool_call_id: str,
        definition: ToolDefinition,
        arguments_json: str,
        status: str,
        result: object,
        error_message: str,
    ) -> None:
        """Record the terminal result of an SDK-owned MCP Tool invocation."""
        result_content = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, separators=(",", ":"), default=str
        )
        self._record_execution(
            tool_call_id,
            definition,
            arguments_json,
            status,
            result_content,
            error_message,
            get_epoch_millis(),
        )

    def get(self, tool_call_id: str) -> CapturedToolExecution | None:
        """Return terminal evidence for one Tool call when a handler recorded it.

        Args:
            tool_call_id: SDK/model call ID.

        Returns:
            Captured execution or ``None`` while a handler has not produced an outcome.
        """
        return self._executions.get(tool_call_id)

    def list_executions(self) -> list[CapturedToolExecution]:
        """Return recorded executions in observation order for persistence mapping.

        Returns:
            Snapshot list of terminal Tool executions.
        """
        return list(self._executions.values())

    def list_calls(self) -> list[CapturedToolCall]:
        """Return model-emitted calls in observation order.

        Returns:
            Snapshot list used to synthesize missing cancelled/failed outcomes.
        """
        return list(self._calls.values())
