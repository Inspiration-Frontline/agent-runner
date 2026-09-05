import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal, Protocol, TypeIs
from uuid import uuid4

from agents import Agent, FunctionTool, Model, ModelSettings, Runner
from agents.items import ModelResponse, TResponseInputItem
from agents.mcp import MCPServer
from agents.result import RunResult, RunResultStreaming
from agents.retry import ModelRetrySettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent, StreamEvent
from agents.tool_context import ToolContext
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_function_tool_call_param import ResponseFunctionToolCallParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_input_message_content_list_param import ResponseInputMessageContentListParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_usage import ResponseUsage

from agent_runner.agent_definitions.config_models import AgentDefinition
from agent_runner.config import Settings, resolve_project_path
from agent_runner.context.builder import (
    AgentContext,
    CapturedMessage,
    Message,
    ModelImagePart,
    ModelTextPart,
    RuntimeToolCall,
    captured_message_to_dict,
    message_to_capture,
)
from agent_runner.gateway.litellm_client import LiteLLMModelFactory
from agent_runner.mcps.catalog import McpServerCatalog
from agent_runner.mcps.sdk_runtime import DispatchEvidenceRecorder, DurableMcpServer, SdkMcpRuntime
from agent_runner.observability.tracing import Tracer, current_trace_id
from agent_runner.runtime.cancellation import CancellationToken
from agent_runner.runtime.model_events import (
    ModelError,
    ModelStreamEvent,
    ModelTokenDelta,
    ModelToolCompleted,
    ModelToolStarted,
    ModelUsage,
)
from agent_runner.runtime.tool_loop import (
    AgentRunCapture,
    CapturedModelTurn,
    CapturedToolCall,
    CapturedToolExecution,
    ToolExecutionCollector,
    get_epoch_millis,
)
from agent_runner.tools.registry import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)

SdkMessageRole = Literal["user", "assistant", "system", "developer"]
_SDK_MESSAGE_ROLES: frozenset[SdkMessageRole] = frozenset({"user", "assistant", "system", "developer"})


def _is_sdk_message_role(role: str) -> TypeIs[SdkMessageRole]:
    """Return whether a provider role belongs to the SDK-supported message-role set.

    Args:
        role: Provider-neutral message role to validate or convert.

    Returns:
        ``True`` when a provider role belongs to the SDK-supported message-role set.
    """

    return role in _SDK_MESSAGE_ROLES


class AgentModelFactory(Protocol):
    """Small model-factory boundary used by the SDK adapter and its tests."""

    async def ensure_reachable(self) -> None:
        """Verify that the configured model gateway can accept a connection."""

    def create_model(self, model: str) -> str | Model:
        """Create an SDK-compatible model for one Agent definition.

        Args:
            model: Provider model identifier.

        Returns:
            SDK-compatible model for one Agent definition.
        """

    async def close(self) -> None:
        """Release resources owned by the model factory."""


@dataclass(frozen=True)
class AssistantResponse:
    """Compatibility response returned by the non-streaming adapter path."""

    content: str
    """Assistant response text returned by the compatibility path."""
    role: str = "assistant"
    """Provider-neutral role associated with the response."""


@dataclass(frozen=True)
class NormalizedModelOutput:
    """Assistant text and Tool Calls extracted from one SDK response."""

    content: str
    """Visible assistant content extracted from the SDK response."""
    tool_calls: tuple[CapturedToolCall, ...]
    """Tool Calls extracted from the SDK response in provider order."""


@dataclass(frozen=True)
class PreparedSdkExecution:
    """One request-scoped SDK Agent and the Tool audit state attached to it."""

    agent: Agent[Any]
    """SDK Agent instance configured for this request."""
    definitions: list[ToolDefinition]
    """Resolved Tool definitions wrapped for SDK execution."""
    collector: ToolExecutionCollector
    """Request-scoped collector for durable Tool evidence."""


@dataclass
class StreamingSdkExecution:
    """Mutable evidence collected while one SDK stream is active."""

    prepared: PreparedSdkExecution
    """Agent and Tool state used by the active SDK run."""
    result: RunResultStreaming
    """OpenAI Agents SDK streaming result handle."""
    run_start: int
    """Epoch milliseconds at which the SDK run started."""
    trace_id: str
    """W3C trace ID associated with the SDK run."""
    model_completed_times: list[int] = field(default_factory=list)
    """Completion timestamps for each provider model response."""
    model_completed_usages: list[tuple[int, int, int]] = field(default_factory=list)
    """Prompt/completion/total usage triples by provider response."""
    partial_response_text: str = ""
    """Text deltas from the current provider response before a completed event is observed."""


@dataclass(frozen=True)
class CapturedTurnEvidence:
    """Normalized output, Tool audit, and timing for one model response."""

    response_content: str
    """Visible assistant text captured for the model Turn."""
    tool_calls: list[CapturedToolCall]
    """Model-emitted Tool Calls in response order."""
    executions: list[CapturedToolExecution]
    """Terminal Tool execution evidence associated with the Turn."""
    llm_end: int
    """Epoch milliseconds immediately after the provider response."""
    turn_end: int
    """Epoch milliseconds at which the Turn capture completed."""


@dataclass(frozen=True)
class CaptureBuildOptions:
    """Shared immutable inputs used while reconstructing persisted model Turns."""

    definitions: list[ToolDefinition]
    """Resolved Tool definitions included in the capture."""
    collector: ToolExecutionCollector
    """Collector containing terminal Tool execution evidence."""
    trace_id: str
    """W3C trace ID persisted with the captured Turn."""
    model_completed_times: list[int]
    """Provider response completion timestamps used for Turn timing."""
    cancelled: bool
    """Whether request cancellation should mark the final Turn partial."""


@dataclass(frozen=True)
class ObservedPartialCaptureOptions:
    """Inputs needed to preserve a partial SDK capture."""

    initial_messages: list[CapturedMessage]
    """Initial provider-neutral messages supplied to the model."""
    definitions: list[ToolDefinition]
    """Resolved Tool definitions offered during the request."""
    collector: ToolExecutionCollector
    """Request-scoped collector containing observed Tool evidence."""
    run_start: int
    """Epoch-millisecond timestamp captured before SDK execution."""
    trace_id: str
    """W3C trace identifier associated with the request."""
    model_completed_times: list[int]
    """Provider completion timestamps observed before cancellation."""
    model_completed_usages: list[tuple[int, int, int]]
    """Provider prompt/completion/total usage triples."""
    cancelled: bool
    """Whether request cancellation was observed."""
    response_content: str
    """Visible assistant text observed before cancellation or failure."""


async def invoke_sdk_tool(
    tool_context: ToolContext[Any],
    arguments_json: str,
    *,
    captured: ToolDefinition,
    collector: ToolExecutionCollector,
    cancellation_token: CancellationToken | None,
) -> str:
    """Invoke one SDK Tool through the AgentBreaker execution-audit boundary.

    Args:
        tool_context: SDK context containing the model-generated Tool call ID.
        arguments_json: Exact serialized arguments emitted by the model.
        captured: Frozen Tool definition bound to this SDK wrapper.
        collector: Request-scoped collector recording execution evidence.
        cancellation_token: Request cancellation token propagated to the collector.

    Returns:
        Normalized result JSON returned to the OpenAI Agents SDK.
    """

    return await collector.execute(
        tool_call_id=str(tool_context.tool_call_id),
        definition=captured,
        arguments_json=arguments_json,
        tool_context=tool_context,
        cancellation_token=cancellation_token,
    )


class OpenAIAgentsSdkAdapter:
    """Adapter between AgentBreaker's runtime contracts and openai-agents-python.

    This class does not implement an Agent or Tool loop. It converts the local
    AgentBreaker definition and context into an Agents SDK Agent, delegates the
    complete model/Tool loop to ``Runner``, then maps SDK stream events and run
    evidence back to AgentBreaker's typed runtime and persistence contracts.

    Attributes:
        settings: Effective application settings for the operation.
        tracer: Application-owned OpenTelemetry facade.
        model_factory: Factory that creates the configured SDK model adapter.
        mcp_runtime: MCP runtime that prepares request-scoped SDK server adapters.
        last_capture: Most recent typed capture retained for persistence.
    """

    def __init__(
        self,
        model_factory: AgentModelFactory | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        mcp_runtime: SdkMcpRuntime | None = None,
    ) -> None:
        """Create the SDK adapter and an empty durable capture snapshot.

        Args:
            model_factory: Provider-neutral model factory. The default uses the configured LiteLLM
                gateway, allowing ModelScope or another OpenAI-compatible provider behind it.
            settings: Effective application settings for the operation.
            tracer: Application-owned OpenTelemetry facade.
            mcp_runtime: MCP runtime that prepares request-scoped SDK server adapters.
        """
        self.settings = settings or Settings()
        self.tracer = tracer or Tracer(self.settings.otel_service_name)
        self.model_factory = model_factory or LiteLLMModelFactory(settings=self.settings, tracer=self.tracer)
        catalog: McpServerCatalog = McpServerCatalog.from_file(resolve_project_path(self.settings.mcp_catalog_path))
        self.mcp_runtime = mcp_runtime or SdkMcpRuntime(catalog, self.tracer, settings=self.settings)
        self.last_capture = AgentRunCapture()

    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
        tool_registry: ToolRegistry | None = None,
        dispatch_recorder: DispatchEvidenceRecorder | None = None,
    ) -> AsyncGenerator[ModelStreamEvent]:
        """Stream one SDK Agent run and always retain its durable capture evidence.

        Args:
            agent: Resolved Agent definition participating in the operation.
            context: Request or SDK context associated with the operation.
            cancellation_token: Request-scoped cooperative cancellation token.
            tool_registry: Registry used to resolve the Agent's configured Tools.
            dispatch_recorder: Optional recorder for durable MCP delivery evidence.

        Yields:
            Normalized SDK events while retaining durable capture evidence.
        """
        self.last_capture = AgentRunCapture()

        await self.model_factory.ensure_reachable()

        async with self.mcp_runtime.session(agent.mcp_servers, dispatch_recorder) as mcp_session:
            execution: StreamingSdkExecution = self._start_streaming_execution(
                agent,
                context,
                cancellation_token,
                tool_registry,
                mcp_session.servers,
                list(mcp_session.definitions),
                mcp_session.dispatch_hooks,
            )

            try:
                async for event in self._stream_sdk_events(execution, cancellation_token):
                    yield event
            finally:
                self._finalize_stream_capture(execution, context, cancellation_token)

    def _start_streaming_execution(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None,
        tool_registry: ToolRegistry | None,
        mcp_servers: tuple[MCPServer, ...],
        mcp_definitions: list[ToolDefinition],
        dispatch_hooks: Any,
    ) -> StreamingSdkExecution:
        """Create the SDK run and its request-scoped evidence accumulator.

        Args:
            agent: Resolved Agent definition participating in the operation.
            context: Request or SDK context associated with the operation.
            cancellation_token: Request-scoped cooperative cancellation token.
            tool_registry: Registry used to resolve the Agent's configured Tools.
            mcp_servers: Collection of mcp servers consumed in deterministic order.
            mcp_definitions: Frozen MCP Tool definitions offered to the model.
            dispatch_hooks: Request-scoped SDK hooks that join MCP calls to durable evidence.

        Returns:
            create the SDK run and its request-scoped evidence accumulator.
        """
        prepared: PreparedSdkExecution = self._prepare_sdk_execution(
            agent=agent,
            context=context,
            cancellation_token=cancellation_token,
            tool_registry=tool_registry,
            mcp_servers=mcp_servers,
            mcp_definitions=mcp_definitions,
            dispatch_hooks=dispatch_hooks,
        )
        trace_id: str = current_trace_id()

        if not trace_id:
            raise RuntimeError("Agent execution requires an active OpenTelemetry trace.")
        run_start: int = get_epoch_millis()
        result: RunResultStreaming = Runner.run_streamed(
            starting_agent=prepared.agent,
            input=self._build_input(context),
            max_turns=10,
        )

        if cancellation_token is not None:
            cancellation_token.add_callback(result.cancel)

        return StreamingSdkExecution(prepared, result, run_start, trace_id)

    async def _stream_sdk_events(
        self,
        execution: StreamingSdkExecution,
        cancellation_token: CancellationToken | None,
    ) -> AsyncGenerator[ModelStreamEvent]:
        """Consume SDK events with normalized cancellation and error behavior.

        Args:
            execution: Active SDK execution and its request-scoped capture state.
            cancellation_token: Request-scoped cooperative cancellation token.

        Yields:
            Normalized runtime events converted from SDK events.
        """

        try:
            async for event in execution.result.stream_events():
                if cancellation_token and cancellation_token.is_cancelled():
                    logger.info("Stream cancelled by token")
                    execution.result.cancel()
                    break
                self._record_stream_completion(execution, event)
                converted: ModelTokenDelta | ModelToolStarted | ModelToolCompleted | ModelUsage | ModelError | None = (
                    self._convert_stream_event(event, execution.prepared.collector)
                )

                if isinstance(converted, ModelTokenDelta):
                    execution.partial_response_text += converted.content

                if converted is not None:
                    yield converted
        except asyncio.CancelledError:
            logger.info("Stream cancelled")
            execution.result.cancel()

            raise
        except TimeoutError as error:
            execution.result.cancel()
            logger.exception("SDK streaming timed out")
            yield ModelError(str(error))
        except Exception as error:
            logger.exception("Error during SDK streaming")
            yield ModelError(str(error))

    @staticmethod
    def _record_stream_completion(execution: StreamingSdkExecution, event: StreamEvent) -> None:
        """Retain completion time and usage for each raw model response.

        Args:
            execution: Active SDK execution and its request-scoped capture state.
            event: Typed runtime or SDK event to process.
        """

        if not isinstance(event, RawResponsesStreamEvent) or not isinstance(event.data, ResponseCompletedEvent):
            return
        usage: ResponseUsage | None = event.data.response.usage
        execution.partial_response_text = ""
        execution.model_completed_times.append(get_epoch_millis())
        execution.model_completed_usages.append(
            (
                usage.input_tokens if usage is not None else 0,
                usage.output_tokens if usage is not None else 0,
                usage.total_tokens if usage is not None else 0,
            )
        )

    def _finalize_stream_capture(
        self,
        execution: StreamingSdkExecution,
        context: AgentContext,
        cancellation_token: CancellationToken | None,
    ) -> None:
        """Build durable evidence after success, failure, or cancellation.

        Args:
            execution: Active SDK execution and its request-scoped capture state.
            context: Request or SDK context associated with the operation.
            cancellation_token: Request-scoped cooperative cancellation token.
        """
        self.last_capture = self._build_capture(
            result=execution.result,
            context=context,
            definitions=execution.prepared.definitions,
            collector=execution.prepared.collector,
            run_start=execution.run_start,
            trace_id=execution.trace_id,
            model_completed_times=execution.model_completed_times,
            model_completed_usages=execution.model_completed_usages,
            cancellation_token=cancellation_token,
            partial_response_text=execution.partial_response_text,
        )

    async def run(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
        tool_registry: ToolRegistry | None = None,
        dispatch_recorder: DispatchEvidenceRecorder | None = None,
    ) -> AssistantResponse:
        """Execute one request through the SDK's non-streaming Agent and Tool loop.

        Args:
            agent: Resolved AgentBreaker definition.
            context: Provider-neutral execution context.
            cancellation_token: Optional token checked before invoking the model.
            tool_registry: Registry containing the configured decorated Tools.

            dispatch_recorder: Optional recorder for durable MCP delivery evidence.
        Returns:
            Typed assistant response for legacy callers.

        Raises:
            asyncio.CancelledError: When cancellation was already signalled.
        """

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError("Execution cancelled")

        await self.model_factory.ensure_reachable()

        async with self.mcp_runtime.session(agent.mcp_servers, dispatch_recorder) as mcp_session:
            prepared: PreparedSdkExecution = self._prepare_sdk_execution(
                agent=agent,
                context=context,
                cancellation_token=cancellation_token,
                tool_registry=tool_registry,
                mcp_servers=mcp_session.servers,
                mcp_definitions=list(mcp_session.definitions),
                dispatch_hooks=mcp_session.dispatch_hooks,
            )
            result: RunResult = await Runner.run(
                starting_agent=prepared.agent,
                input=self._build_input(context),
                max_turns=10,
            )

            return AssistantResponse(content=str(result.final_output), role="assistant")

    def _prepare_sdk_execution(
        self,
        *,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None,
        tool_registry: ToolRegistry | None,
        mcp_servers: tuple[MCPServer, ...] = (),
        mcp_definitions: list[ToolDefinition] | None = None,
        dispatch_hooks: Any = None,
    ) -> PreparedSdkExecution:
        """Resolve and attach configured Tools identically for both SDK Runner entry points.

        Args:
            agent: Resolved Agent definition participating in the operation.
            context: Request or SDK context associated with the operation.
            cancellation_token: Request-scoped cooperative cancellation token.
            tool_registry: Registry used to resolve the Agent's configured Tools.
            mcp_servers: Collection of mcp servers consumed in deterministic order.
            mcp_definitions: Frozen MCP Tool definitions offered to the model.
            dispatch_hooks: Request-scoped SDK hooks that join MCP calls to durable evidence.

        Returns:
            resolve and attach configured Tools identically for both SDK Runner entry points.
        """
        collector: ToolExecutionCollector = ToolExecutionCollector(self.settings, self.tracer)
        registry: ToolRegistry = tool_registry or ToolRegistry()
        local_definitions: list[ToolDefinition] = self._resolve_tool_definitions(agent, registry)
        definitions: list[ToolDefinition] = [*local_definitions, *(mcp_definitions or [])]

        self._bind_mcp_execution(dispatch_hooks, mcp_servers, collector, definitions)
        sdk_tools: list[FunctionTool] = self._build_sdk_tools(
            local_definitions,
            collector,
            cancellation_token,
        )

        return PreparedSdkExecution(
            agent=self._build_sdk_agent(
                agent,
                context.system_prompt,
                sdk_tools,
                mcp_servers,
                dispatch_hooks,
            ),
            definitions=definitions,
            collector=collector,
        )

    @staticmethod
    def _bind_mcp_execution(
        dispatch_hooks: Any,
        mcp_servers: tuple[MCPServer, ...],
        collector: ToolExecutionCollector,
        definitions: list[ToolDefinition],
    ) -> None:
        """Attach request-scoped collectors to MCP hooks and durable servers.

        Args:
            dispatch_hooks: Optional hooks recording dispatch evidence.
            mcp_servers: Servers participating in this execution.
            collector: Collector receiving Tool execution evidence.
            definitions: Complete local and MCP Tool definitions.
        """
        if dispatch_hooks is not None:
            dispatch_hooks.bind_collector(collector, definitions)

        for mcp_server in mcp_servers:
            if isinstance(mcp_server, DurableMcpServer):
                mcp_server.bind_execution_collector(collector, definitions)

    def _build_sdk_agent(
        self,
        agent: AgentDefinition,
        system_prompt: str,
        tools: Sequence[FunctionTool] | None = None,
        mcp_servers: tuple[MCPServer, ...] = (),
        dispatch_hooks: Any = None,
    ) -> Agent[Any]:
        """Translate an AgentBreaker definition into one request-scoped SDK Agent.

        Args:
            agent: Stable name/model/tool configuration.
            system_prompt: Request-specific prompt assembled by ``ContextBuilder``.
            tools: Audited SDK FunctionTools available to the model.

            mcp_servers: Collection of mcp servers consumed in deterministic order.
            dispatch_hooks: Request-scoped SDK hooks that join MCP calls to durable evidence.
        Returns:
            SDK Agent configured with provider timeout, retry, and parallel Tool policy.
        """

        return Agent(
            name=agent.name,
            instructions=system_prompt,
            model=self.model_factory.create_model(agent.model),
            model_settings=ModelSettings(
                temperature=agent.temperature,
                max_tokens=agent.max_output_tokens,
                include_usage=True,
                extra_args={
                    # LiteLLM accepts a numeric timeout and forwards it to the OpenAI client.
                    # Passing an httpx.Timeout instance causes LiteLLM's request metadata
                    # serializer to fail before the network call starts.
                    "timeout": self.settings.lite_llm_request_timeout_seconds,
                    "num_retries": self.settings.lite_llm_max_retries,
                    "max_retries": self.settings.lite_llm_max_retries,
                },
                retry=ModelRetrySettings(max_retries=self.settings.lite_llm_max_retries),
                parallel_tool_calls=True,
            ),
            tools=[*tools] if tools else [],
            mcp_servers=list(mcp_servers),
            mcp_config={"include_server_in_tool_names": True},
            hooks=dispatch_hooks,
        )

    @staticmethod
    def _resolve_tool_definitions(agent: AgentDefinition, registry: ToolRegistry) -> list[ToolDefinition]:
        """Resolve configured Tool keys before model invocation.

        Args:
            agent: Definition containing stable Tool keys.
            registry: Runtime registry of decorated Tool definitions.

        Returns:
            Definitions in the agent's configured order.

        Raises:
            ValueError: When configuration references a Tool not registered in this process.
        """
        definitions: list[ToolDefinition] = []

        for tool_key in agent.tools:
            definition: ToolDefinition | None = registry.get(tool_key)

            if definition is None:
                raise ValueError(f"Configured Tool is not registered: {tool_key}")

            definitions.append(definition)

        return definitions

    @staticmethod
    def _build_sdk_tools(
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        cancellation_token: CancellationToken | None,
    ) -> list[FunctionTool]:
        """Wrap SDK Tools with auditing while preserving generated metadata.

        Args:
            definitions: Frozen Tool definitions participating in the operation.
            collector: Request-scoped collector for durable execution evidence.
            cancellation_token: Request-scoped cooperative cancellation token.

        Returns:
            SDK Tool wrappers with AgentBreaker auditing and provenance.
        """
        tools: list[FunctionTool] = []

        for definition in definitions:
            if definition.function_tool is None:
                raise ValueError(f"Tool has no SDK FunctionTool: {definition.tool_key}")

            # Preserve every option generated/configured by @function_tool and replace only the
            # invocation hook needed for AgentBreaker execution auditing.
            invoke: Callable[[ToolContext[Any], str], Awaitable[str]] = partial(
                invoke_sdk_tool,
                captured=definition,
                collector=collector,
                cancellation_token=cancellation_token,
            )
            tools.append(replace(definition.function_tool, on_invoke_tool=invoke))

        return tools

    def _build_input(self, context: AgentContext) -> list[TResponseInputItem]:
        """Convert neutral messages into Responses API input items.

        Args:
            context: Request or SDK context associated with the operation.

        Returns:
            convert neutral messages into Responses API input items.
        """
        input_items: list[TResponseInputItem] = []

        for message in context.conversation_history:
            input_items.extend(self._build_message_input_items(message))

        input_items.extend(self._build_message_input_items(context.current_message))
        return input_items

    def _build_message_input_items(self, message: Message) -> list[TResponseInputItem]:
        """Convert one neutral message, including Tool linkage, into SDK items.

        Args:
            message: Provider-neutral message to convert or persist.

        Returns:
            convert one neutral message, including Tool linkage, into SDK items.
        """

        if message.role == "assistant" and message.tool_calls:
            items: list[TResponseInputItem] = []

            if message.content:
                items.append({"role": "assistant", "content": message.content})

            items.extend(
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id=call.call_id,
                    name=call.function_name,
                    arguments=call.arguments,
                )
                for call in message.tool_calls
            )

            return items

        if message.role == "tool":
            if not message.tool_call_id:
                raise ValueError("A Tool result message requires a non-empty tool_call_id.")

            output: FunctionCallOutput = FunctionCallOutput(
                type="function_call_output",
                call_id=message.tool_call_id,
                output=message.content,
            )

            return [output]

        sdk_message: EasyInputMessageParam = EasyInputMessageParam(
            role=self._get_sdk_message_role(message.role),
            content=self._convert_provider_content(message),
        )

        return [sdk_message]

    @staticmethod
    def _convert_provider_content(message: Message) -> str | ResponseInputMessageContentListParam:
        """Translate neutral content into provider-bound text/image parts.

        Args:
            message: Provider-neutral message to convert or persist.

        Returns:
            Translated neutral content into provider-bound text/image parts.
        """

        if not message.model_content:
            return message.content

        converted: ResponseInputMessageContentListParam = []

        for part in message.model_content:
            if isinstance(part, ModelTextPart):
                converted.append(ResponseInputTextParam(type="input_text", text=part.text))
            elif isinstance(part, ModelImagePart):
                converted.append(ResponseInputImageParam(type="input_image", image_url=part.url, detail=part.detail))

        return converted

    @staticmethod
    def _get_sdk_message_role(role: str) -> SdkMessageRole:
        """Validate a provider-neutral role before crossing the OpenAI SDK boundary.

        Args:
            role: Provider-neutral message role to validate or convert.

        Returns:
            Validated a provider-neutral role before crossing the OpenAI SDK boundary.
        """

        if _is_sdk_message_role(role):
            return role

        raise ValueError(f"Unsupported OpenAI input message role: {role}")

    @staticmethod
    def _to_capture_message(message: Message) -> CapturedMessage:
        """Keep stable provider-neutral content while excluding transient signed SDK URLs.

        Args:
            message: Runtime message with typed model and capture representations.

        Returns:
            Typed durable role/content value using scalar text or non-empty stable parts.
        """

        return message_to_capture(message)

    def _convert_stream_event(
        self,
        event: StreamEvent,
        collector: ToolExecutionCollector | None = None,
    ) -> ModelStreamEvent | None:
        """Map SDK events to the small event vocabulary exposed over SSE.

        Args:
            event: Typed runtime or SDK event to process.
            collector: Request-scoped collector for durable execution evidence.

        Returns:
            Typed AgentBreaker event, or ``None`` for non-public SDK events.
        """

        if isinstance(event, RawResponsesStreamEvent):
            return self._convert_raw_response_event(event)

        if isinstance(event, RunItemStreamEvent):
            return self._convert_run_item_event(event, collector)

        return None

    def _convert_raw_response_event(self, event: RawResponsesStreamEvent) -> ModelStreamEvent | None:
        """Convert raw token and usage events while ignoring SDK bookkeeping.

        Args:
            event: Typed runtime or SDK event to process.

        Returns:
            convert raw token and usage events while ignoring SDK bookkeeping.
        """

        if isinstance(event.data, ResponseTextDeltaEvent):
            return ModelTokenDelta(event.data.delta) if event.data.delta else None

        if isinstance(event.data, ResponseCompletedEvent):
            return self._convert_response_completed_usage(event.data)

        return None

    @staticmethod
    def _convert_run_item_event(
        event: RunItemStreamEvent,
        collector: ToolExecutionCollector | None,
    ) -> ModelStreamEvent | None:
        """Convert SDK Tool events and update request-scoped audit evidence.

        Args:
            event: Typed runtime or SDK event to process.
            collector: Request-scoped collector for durable execution evidence.

        Returns:
            convert SDK Tool events and update request-scoped audit evidence.
        """

        if event.name == "tool_called":
            raw_item: Any | None = getattr(event.item, "raw_item", None)
            captured_call: CapturedToolCall = CapturedToolCall(
                tool_call_id=str(getattr(event.item, "call_id", None) or ""),
                tool_name=str(getattr(raw_item, "name", "")) if raw_item is not None else "",
                arguments=str(getattr(raw_item, "arguments", "{}")) if raw_item is not None else "{}",
            )

            if collector is not None:
                collector.record_call(captured_call)

            return ModelToolStarted(
                tool_name=captured_call.tool_name,
                tool_call_id=captured_call.tool_call_id,
                arguments_json=captured_call.arguments,
            )

        if event.name == "tool_output":
            tool_call_id: str = str(getattr(event.item, "call_id", None) or "")
            execution: CapturedToolExecution | None = collector.get(tool_call_id) if collector is not None else None

            return ModelToolCompleted(
                tool_name=execution.tool_name if execution is not None else "",
                tool_call_id=tool_call_id,
                result=getattr(event.item, "output", None),
                status=execution.status if execution is not None else "COMPLETED",
            )

        return None

    @staticmethod
    def _convert_response_completed_usage(event_data: ResponseCompletedEvent) -> ModelUsage | None:
        """Extract provider usage when a completed response reports it.

        Args:
            event_data: SDK response-completed event.

        Returns:
            Typed usage event, or ``None`` when usage was omitted.
        """
        usage: Any | None = getattr(getattr(event_data, "response", None), "usage", None)

        if usage is None:
            return None

        return ModelUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    def _build_capture(
        self,
        *,
        result: RunResultStreaming,
        context: AgentContext,
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        run_start: int,
        trace_id: str,
        model_completed_times: list[int],
        model_completed_usages: list[tuple[int, int, int]],
        cancellation_token: CancellationToken | None,
        partial_response_text: str,
    ) -> AgentRunCapture:
        """Reconstruct durable provider-neutral Turns from one SDK run.

        Args:
            result: SDK result; context: request context; definitions: offered Tools; collector:
                Tool evidence collector; run_start: execution start; trace_id: W3C correlation ID.
            model_completed_times: Provider completion timestamps; model_completed_usages: usage triples.
            cancellation_token: Request cancellation state; partial_response_text: observed partial text.

        Returns:
            Reconstructed durable provider-neutral Turns from one SDK run.
        """
        initial_messages: list[CapturedMessage] = self._build_initial_capture_messages(context)
        raw_responses: list[ModelResponse] = list(result.raw_responses)
        cancelled: bool = self._is_cancelled(cancellation_token)

        if not raw_responses and (collector.list_calls() or partial_response_text):
            return self._build_observed_partial_capture(
                initial_messages=initial_messages,
                definitions=definitions,
                collector=collector,
                run_start=run_start,
                trace_id=trace_id,
                model_completed_times=model_completed_times,
                model_completed_usages=model_completed_usages,
                cancelled=cancelled,
                response_content=partial_response_text,
            )
        options: CaptureBuildOptions = self._build_capture_options(
            definitions, collector, trace_id, model_completed_times, cancelled
        )
        turns: list[CapturedModelTurn] = self._build_captured_turns(raw_responses, initial_messages, run_start, options)

        return AgentRunCapture(turns=turns, final_output=str(result.final_output or ""))

    @staticmethod
    def _build_capture_options(
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        trace_id: str,
        model_completed_times: list[int],
        cancelled: bool,
    ) -> CaptureBuildOptions:
        """Create immutable options shared by captured Turn reconstruction."""

        return CaptureBuildOptions(definitions, collector, trace_id, model_completed_times, cancelled)

    def _build_initial_capture_messages(self, context: AgentContext) -> list[CapturedMessage]:
        """Build the complete stable message snapshot supplied to the first model call.

        Args:
            context: Request or SDK context associated with the operation.

        Returns:
            build the complete stable message snapshot supplied to the first model call.
        """
        messages: list[CapturedMessage] = [CapturedMessage(role="system", content=context.system_prompt)]
        messages.extend(self._to_capture_message(item) for item in context.conversation_history)
        messages.append(self._to_capture_message(context.current_message))

        return messages

    @staticmethod
    def _is_cancelled(cancellation_token: CancellationToken | None) -> bool:
        """Return whether the request token has been cancelled."""

        return cancellation_token is not None and cancellation_token.is_cancelled()

    def _build_captured_turns(
        self,
        responses: list[ModelResponse],
        initial_messages: list[CapturedMessage],
        run_start: int,
        options: CaptureBuildOptions,
    ) -> list[CapturedModelTurn]:
        """Build one persisted Turn per actual SDK model response.

        Args:
            responses: Collection of responses consumed in deterministic order.
            initial_messages: Collection of initial messages consumed in deterministic order.
            run_start: Epoch-millisecond boundary captured before SDK execution.
            options: Immutable capture options shared across Turn reconstruction.

        Returns:
            build one persisted Turn per actual SDK model response.
        """
        turns: list[CapturedModelTurn] = []
        request_messages: list[CapturedMessage] = initial_messages
        full_model_input: list[CapturedMessage] = list(initial_messages)
        previous_turn_end: int = run_start
        assigned_ids: set[str] = set()

        for index, response in enumerate(responses):
            evidence: CapturedTurnEvidence = self._build_turn_evidence(
                response, index, len(responses), previous_turn_end, assigned_ids, options
            )
            turns.append(
                self._build_captured_turn(
                    response,
                    evidence,
                    request_messages,
                    full_model_input,
                    previous_turn_end,
                    index,
                    options,
                )
            )
            request_messages = self._next_turn_delta(
                evidence.response_content, evidence.tool_calls, evidence.executions
            )
            full_model_input.extend(request_messages)
            previous_turn_end = evidence.turn_end

        return turns

    def _build_turn_evidence(
        self,
        response: ModelResponse,
        index: int,
        response_count: int,
        previous_turn_end: int,
        assigned_ids: set[str],
        options: CaptureBuildOptions,
    ) -> CapturedTurnEvidence:
        """Normalize response output, fill cancelled calls, and derive Turn timing.

        Args:
            response: Provider, RPC, or HTTP response to inspect.
            index: Zero-based position within the containing ordered sequence.
            response_count: Number of response values observed.
            previous_turn_end: Epoch-millisecond end boundary of the preceding Turn.
            assigned_ids: Stable identifiers of the selected assigned values.
            options: Immutable capture options shared across Turn reconstruction.

        Returns:
            Normalized response output, fill cancelled calls, and derive Turn timing.
        """
        event_end: int = (
            options.model_completed_times[index]
            if index < len(options.model_completed_times)
            else max(previous_turn_end, get_epoch_millis())
        )
        normalized: NormalizedModelOutput = self._normalize_response_output(response.output)
        tool_calls: list[CapturedToolCall] = list(normalized.tool_calls)

        if not tool_calls and index == response_count - 1:
            tool_calls = [call for call in options.collector.list_calls() if call.tool_call_id not in assigned_ids]
        assigned_ids.update(call.tool_call_id for call in tool_calls)
        executions: list[CapturedToolExecution] = self._complete_execution_audit(
            tool_calls=tool_calls,
            definitions=options.definitions,
            collector=options.collector,
            fallback_time=event_end,
            cancelled=options.cancelled,
        )
        llm_end: int = max(previous_turn_end, min([event_end, *(item.start_time for item in executions)]))
        turn_end: int = max([llm_end, *(item.end_time for item in executions)])

        return CapturedTurnEvidence(normalized.content, tool_calls, executions, llm_end, turn_end)

    def _build_captured_turn(
        self,
        response: ModelResponse,
        evidence: CapturedTurnEvidence,
        request_messages: list[CapturedMessage],
        full_model_input: list[CapturedMessage],
        start_time: int,
        index: int,
        options: CaptureBuildOptions,
    ) -> CapturedModelTurn:
        """Create one durable model Turn from normalized response evidence.

        Args:
            response: Provider, RPC, or HTTP response to inspect.
            evidence: Captured timing, usage, and response evidence for one Turn.
            request_messages: Collection of request messages consumed in deterministic order.
            full_model_input: Complete provider-neutral model input accumulated through the Turn.
            start_time: Timestamp representing start time.
            index: Zero-based position within the containing ordered sequence.
            options: Immutable capture options shared across Turn reconstruction.

        Returns:
            Durable model Turn assembled from normalized response evidence.
        """
        request_id: Any | str = (
            getattr(response, "request_id", None) or getattr(response, "response_id", None) or str(uuid4())
        )

        return CapturedModelTurn(
            request_messages=request_messages,
            message_storage_mode="FULL_SNAPSHOT" if index == 0 else "APPEND_DELTA",
            tools=options.definitions,
            response_content=evidence.response_content,
            response_tool_calls=evidence.tool_calls,
            tool_executions=evidence.executions,
            request_id=request_id,
            trace_id=options.trace_id,
            start_time=start_time,
            llm_end_time=evidence.llm_end,
            end_time=evidence.turn_end,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            raw_request=self._build_raw_request(full_model_input, options.definitions),
            raw_response=self._build_raw_response(response, evidence.tool_calls),
        )

    def _build_raw_request(
        self,
        messages: list[CapturedMessage],
        definitions: list[ToolDefinition],
    ) -> str:
        """Serialize deterministic request audit data.

        Args:
            messages: Messages retained in their original model order.
            definitions: Frozen Tool definitions participating in the operation.

        Returns:
            Serialized deterministic request audit data.
        """

        return json.dumps(
            {
                "messages": [captured_message_to_dict(message) for message in messages],
                "tools": [self._convert_definition_to_dict(item) for item in definitions],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _build_raw_response(
        self,
        response: ModelResponse,
        tool_calls: list[CapturedToolCall],
    ) -> str:
        """Serialize provider response data and normalized Tool Calls for audit.

        Args:
            response: Provider, RPC, or HTTP response to inspect.
            tool_calls: Collection of tool calls consumed in deterministic order.

        Returns:
            Serialized provider response data and normalized Tool Calls for audit.
        """

        return json.dumps(
            {
                "output": [self._dump_model(item) for item in response.output],
                "observed_tool_calls": [self._captured_tool_call_to_dict(call) for call in tool_calls],
                "response_id": getattr(response, "response_id", None),
                "request_id": getattr(response, "request_id", None),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _captured_tool_call_to_dict(call: CapturedToolCall) -> dict[str, str]:
        """Serialize one captured Tool Call for raw audit JSON.

        Args:
            call: Captured provider Tool call being converted or audited.

        Returns:
            Serialized one captured Tool Call for raw audit JSON.
        """

        return {"id": call.tool_call_id, "name": call.tool_name, "arguments": call.arguments}

    def _build_observed_partial_capture(
        self,
        *,
        initial_messages: list[CapturedMessage],
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        run_start: int,
        trace_id: str,
        model_completed_times: list[int],
        model_completed_usages: list[tuple[int, int, int]],
        cancelled: bool,
        response_content: str = "",
    ) -> AgentRunCapture:
        """Preserve observed Tool evidence when cancellation removes SDK raw output.

        Args:
            initial_messages: Input snapshot; definitions: offered Tools; collector: Tool evidence;
                run_start: execution start; trace_id: W3C ID; model_completed_times: completion times.
            model_completed_usages: Provider usage triples; cancelled: cancellation state;
                response_content: text observed before cancellation or failure.

        Returns:
            Capture containing the observed partial Turn.
        """
        options: ObservedPartialCaptureOptions = ObservedPartialCaptureOptions(
            initial_messages=initial_messages,
            definitions=definitions,
            collector=collector,
            run_start=run_start,
            trace_id=trace_id,
            model_completed_times=model_completed_times,
            model_completed_usages=model_completed_usages,
            cancelled=cancelled,
            response_content=response_content,
        )
        return self._build_observed_partial_capture_from_options(options)

    def _build_observed_partial_capture_from_options(self, options: ObservedPartialCaptureOptions) -> AgentRunCapture:
        """Build a partial capture from normalized request-scoped options.

        Args:
            options: Immutable input snapshot and timing/evidence values for the partial capture.

        Returns:
            Capture containing the observed partial Turn.
        """
        normalized_initial_messages: list[CapturedMessage] = [
            self._coerce_capture_message(message) for message in options.initial_messages
        ]
        tool_calls: list[CapturedToolCall] = options.collector.list_calls()
        event_llm_end: int = options.model_completed_times[-1] if options.model_completed_times else get_epoch_millis()
        executions: list[CapturedToolExecution] = self._complete_execution_audit(
            tool_calls=tool_calls,
            definitions=options.definitions,
            collector=options.collector,
            fallback_time=event_llm_end,
            cancelled=options.cancelled,
        )
        llm_end: int = max(options.run_start, min([event_llm_end, *(item.start_time for item in executions)]))
        turn_end: int = max([llm_end, *(item.end_time for item in executions)])
        prompt_tokens, completion_tokens, total_tokens = (
            options.model_completed_usages[-1] if options.model_completed_usages else (0, 0, 0)
        )
        turn: CapturedModelTurn = self._build_partial_turn(
            normalized_initial_messages,
            options.definitions,
            tool_calls,
            executions,
            options.trace_id,
            options.run_start,
            llm_end,
            turn_end,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            options.cancelled,
            options.response_content,
        )

        return AgentRunCapture(turns=[turn], final_output=options.response_content)

    def _build_partial_turn(
        self,
        messages: list[CapturedMessage],
        definitions: list[ToolDefinition],
        tool_calls: list[CapturedToolCall],
        executions: list[CapturedToolExecution],
        trace_id: str,
        run_start: int,
        llm_end: int,
        turn_end: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cancelled: bool,
        response_content: str,
    ) -> CapturedModelTurn:
        """Create the single failed or cancelled Turn for partial SDK evidence.

        Args:
            messages: Ordered input; definitions: offered Tools; tool_calls: model Tool calls;
                executions: terminal Tool evidence; trace_id: W3C ID; run_start/llm_end/turn_end:
                timing boundaries; prompt_tokens/completion_tokens/total_tokens: provider usage.
            cancelled: Whether cancellation was observed; response_content: visible partial text.

        Returns:
            The single failed or cancelled Turn for partial SDK evidence.
        """

        return CapturedModelTurn(
            request_messages=messages,
            message_storage_mode="FULL_SNAPSHOT",
            tools=definitions,
            response_content=response_content,
            response_tool_calls=tool_calls,
            tool_executions=executions,
            request_id=str(uuid4()),
            trace_id=trace_id,
            start_time=run_start,
            llm_end_time=llm_end,
            end_time=turn_end,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_request=self._build_raw_request(messages, definitions),
            raw_response=self._build_partial_raw_response(tool_calls, cancelled),
        )

    def _build_partial_raw_response(
        self,
        tool_calls: list[CapturedToolCall],
        cancelled: bool,
    ) -> str:
        """Serialize observed Tool Calls when no provider response survives.

        Args:
            tool_calls: Collection of tool calls consumed in deterministic order.
            cancelled: Whether request cancellation was observed.

        Returns:
            Serialized observed Tool Calls when no provider response survives.
        """

        return json.dumps(
            {
                "output": [],
                "observed_tool_calls": [self._captured_tool_call_to_dict(call) for call in tool_calls],
                "raw_response_removed_by_sdk_cancellation": cancelled,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _complete_execution_audit(
        self,
        *,
        tool_calls: list[CapturedToolCall],
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        fallback_time: int,
        cancelled: bool,
    ) -> list[CapturedToolExecution]:
        """Guarantee one terminal execution record per model-emitted Tool Call.

        Args:
            tool_calls: Collection of tool calls consumed in deterministic order.
            definitions: Frozen Tool definitions participating in the operation.
            collector: Request-scoped collector for durable execution evidence.
            fallback_time: Timestamp representing fallback time.
            cancelled: Whether request cancellation was observed.

        Returns:
            Capture with one terminal execution record per model-emitted Tool Call.
        """
        definitions_by_name: dict[str, ToolDefinition] = {
            definition.tool_name: definition for definition in definitions
        }
        executions: list[CapturedToolExecution] = []

        for tool_call in tool_calls:
            execution: CapturedToolExecution | None = collector.get(tool_call.tool_call_id)

            if execution is not None:
                executions.append(execution)
                continue

            executions.append(
                self._build_missing_tool_execution(
                    tool_call,
                    definitions_by_name.get(tool_call.tool_name),
                    fallback_time,
                    cancelled,
                )
            )

        return executions

    @staticmethod
    def _build_missing_tool_execution(
        tool_call: CapturedToolCall,
        definition: ToolDefinition | None,
        fallback_time: int,
        cancelled: bool,
    ) -> CapturedToolExecution:
        """Build explicit failed/cancelled evidence when no Tool result exists.

        Args:
            tool_call: Captured model Tool call to convert or audit.
            definition: Frozen Tool definition and source provenance.
            fallback_time: Timestamp representing fallback time.
            cancelled: Whether request cancellation was observed.

        Returns:
            build explicit failed/cancelled evidence when no Tool result exists.
        """
        error_message: str = (
            "Generation cancelled before the Tool produced a result."
            if cancelled
            else "Tool execution did not produce an auditable result."
        )
        result_content: str = (
            ""
            if cancelled
            else json.dumps(
                {"status": "error", "error": error_message},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

        return CapturedToolExecution(
            tool_call_id=tool_call.tool_call_id,
            tool_key=definition.tool_key if definition is not None else "unknown",
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments,
            status="CANCELLED" if cancelled else "FAILED",
            result_content=result_content,
            raw_result=result_content,
            error_message=error_message,
            start_time=fallback_time,
            end_time=fallback_time,
        )

    @staticmethod
    def _normalize_response_output(outputs: list[Any]) -> NormalizedModelOutput:
        """Extract assistant text and Tool calls from heterogeneous SDK response items.

        Args:
            outputs: SDK response output items.

        Returns:
            One typed value containing assistant text and Tool Calls in model order.
        """
        text_parts: list[str] = []
        tool_calls: list[CapturedToolCall] = []

        for item in outputs:
            item_type: Any | str = getattr(item, "type", "")

            if item_type == "message":
                for part in getattr(item, "content", []) or []:
                    text: Any | None = getattr(part, "text", None)

                    if text is not None:
                        text_parts.append(str(text))
            elif item_type == "function_call":
                tool_calls.append(
                    CapturedToolCall(
                        tool_call_id=str(getattr(item, "call_id", "")),
                        tool_name=str(getattr(item, "name", "")),
                        arguments=str(getattr(item, "arguments", "{}")),
                    )
                )

        return NormalizedModelOutput(content="".join(text_parts), tool_calls=tuple(tool_calls))

    @staticmethod
    def _coerce_capture_message(message: CapturedMessage | dict[str, Any]) -> CapturedMessage:
        """Accept legacy test/provider snapshots at the JSON boundary before internal capture.

        Args:
            message: Provider-neutral message to convert or persist.

        Returns:
            Typed capture message accepted from a legacy provider snapshot.
        """

        if isinstance(message, CapturedMessage):
            return message

        tool_calls: tuple[RuntimeToolCall, ...] = tuple(
            RuntimeToolCall(
                call_id=str(call.get("id", "")),
                call_type=str(call.get("type", "function")),
                function_name=str((call.get("function") or {}).get("name", "")),
                arguments=str((call.get("function") or {}).get("arguments", "{}")),
            )
            for call in message.get("tool_calls", [])
        )

        return CapturedMessage(
            role=str(message.get("role", "user")),
            content=str(message.get("content", "")) if isinstance(message.get("content"), str) else "",
            tool_calls=tool_calls,
            tool_call_id=str(message.get("tool_call_id", "")) or None,
        )

    @staticmethod
    def _next_turn_delta(
        response_content: str,
        tool_calls: list[CapturedToolCall],
        executions: list[CapturedToolExecution],
    ) -> list[CapturedMessage]:
        """Create neutral assistant/tool continuation messages for the next model Turn.

        Args:
            response_content: Assistant content from the preceding model response.
            tool_calls: Tool calls emitted by that response.
            executions: Normalized Tool execution results.

        Returns:
            APPEND_DELTA messages required to continue the model loop.
        """

        if not tool_calls:
            return []

        messages: list[CapturedMessage] = [
            CapturedMessage(
                role="assistant",
                content=response_content,
                tool_calls=tuple(
                    RuntimeToolCall(
                        call_id=call.tool_call_id,
                        call_type="function",
                        function_name=call.tool_name,
                        arguments=call.arguments,
                    )
                    for call in tool_calls
                ),
            )
        ]

        executions_by_id: dict[str, CapturedToolExecution] = {
            execution.tool_call_id: execution for execution in executions
        }

        for call in tool_calls:
            execution: CapturedToolExecution | None = executions_by_id.get(call.tool_call_id)
            messages.append(
                CapturedMessage(
                    role="tool",
                    content=execution.result_content if execution is not None else "",
                    tool_call_id=call.tool_call_id,
                )
            )

        return messages

    @staticmethod
    def _convert_definition_to_dict(definition: ToolDefinition) -> dict[str, Any]:
        """Serialize a Tool definition into deterministic raw-request audit JSON.

        Args:
            definition: Frozen Tool schema and provenance.

        Returns:
            JSON-compatible dictionary stored in the raw request audit payload.
        """

        return {
            "tool_key": definition.tool_key,
            "tool_name": definition.tool_name,
            "description": definition.description,
            "parameters": definition.parameters,
            "strict": definition.strict,
            "source_type": definition.source_type.value,
            "definition_hash": definition.definition_hash,
        }

    @staticmethod
    def _dump_model(value: Any) -> Any:
        """Convert an SDK response item, not an LLM model, into JSON-compatible audit data.

        SDK output items are usually Pydantic models, while provider adapters may already return
        dictionaries or primitives. Calling the public ``model_dump`` API when present preserves
        structured fields; returning other values unchanged lets ``json.dumps(default=str)`` handle
        adapter-specific fallbacks without coupling this runtime to every SDK item subclass.

        Args:
            value: SDK/Pydantic object, mapping, sequence, or scalar.

        Returns:
            JSON-compatible data or the original scalar fallback.
        """

        model_dump: Any | None = getattr(value, "model_dump", None)
        return model_dump(mode="json") if callable(model_dump) else value

    async def close(self) -> None:
        """Release HTTP resources owned by the LiteLLM model factory."""
        await self.model_factory.close()
