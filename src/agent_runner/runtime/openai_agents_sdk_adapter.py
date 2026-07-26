import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from agents import Agent, FunctionTool, ModelSettings, Runner
from agents.retry import ModelRetrySettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent, StreamEvent
from agents.tool_context import ToolContext
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from agent_runner.agent_definitions.config_models import AgentDefinition
from agent_runner.config import get_settings
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
    epoch_millis,
)
from agent_runner.tools.registry import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssistantResponse:
    """Compatibility response returned by the non-streaming adapter path."""

    content: str
    role: str = "assistant"


@dataclass(frozen=True)
class NormalizedModelOutput:
    """Assistant text and Tool Calls extracted from one SDK response."""

    content: str
    tool_calls: tuple[CapturedToolCall, ...]


class OpenAIAgentsSdkAdapter:
    """
    Adapter between AgentBreaker's runtime contracts and openai-agents-python.

    This class does not implement an Agent or Tool loop. It converts the local
    AgentBreaker definition and context into an Agents SDK Agent, delegates the
    complete model/Tool loop to ``Runner``, then maps SDK stream events and run
    evidence back to AgentBreaker's typed runtime and persistence contracts.
    """

    def __init__(self, model_factory: LiteLLMModelFactory | None = None):
        """Create the SDK adapter and an empty durable capture snapshot.

        Args:
            model_factory: Provider-neutral model factory. The default uses the configured LiteLLM
                gateway, allowing ModelScope or another OpenAI-compatible provider behind it.
        """
        self.model_factory = model_factory or LiteLLMModelFactory()
        self.last_capture = AgentRunCapture()

    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> AsyncGenerator[ModelStreamEvent]:
        """Run one request through the SDK Tool loop and yield AgentBreaker semantic events.

        The SDK owns repeated model/Tool scheduling. AgentBreaker wraps the configured decorated
        FunctionTools only to collect persistence evidence and translate SDK events to SSE.
        ``last_capture`` is finalized even when streaming fails or is cancelled.

        Args:
            agent: Resolved AgentBreaker definition.
            context: Prompt, replay history, attachment metadata, and Tool context for this run.
            cancellation_token: Optional request token that cancels SDK streaming and Tool work.
            tool_registry: Registry containing the configured decorated Tools.

        Yields:
            Typed AgentBreaker runtime events consumed by the orchestrator.
        """
        # Agents SDK agents are declarative; Runner owns execution and streaming.
        self.last_capture = AgentRunCapture()
        collector = ToolExecutionCollector()
        registry = tool_registry or ToolRegistry()
        definitions = self._resolve_tool_definitions(agent, registry)
        sdk_agent = self._build_sdk_agent(
            agent,
            context.system_prompt,
            self._build_sdk_tools(definitions, collector, cancellation_token),
        )
        sdk_input = self._build_input(context)
        run_start = epoch_millis()
        trace_id = str(uuid4())
        model_completed_times: list[int] = []
        model_completed_usages: list[tuple[int, int, int]] = []
        # Runner recognizes model Tool Calls, invokes the FunctionTools attached to sdk_agent,
        # appends their outputs to the next model input, and repeats until a final output.
        result = Runner.run_streamed(
            starting_agent=sdk_agent,
            input=sdk_input,
            max_turns=10,
        )
        if cancellation_token is not None:
            cancellation_token.add_callback(result.cancel)

        try:
            async for event in result.stream_events():
                if cancellation_token and cancellation_token.is_cancelled():
                    logger.info("Stream cancelled by token")
                    result.cancel()
                    break

                converted = self._convert_stream_event(event, collector)
                if isinstance(event, RawResponsesStreamEvent) and isinstance(event.data, ResponseCompletedEvent):
                    model_completed_times.append(epoch_millis())
                    usage = getattr(event.data.response, "usage", None)
                    model_completed_usages.append(
                        (
                            getattr(usage, "input_tokens", 0) if usage is not None else 0,
                            getattr(usage, "output_tokens", 0) if usage is not None else 0,
                            getattr(usage, "total_tokens", 0) if usage is not None else 0,
                        )
                    )
                if converted is not None:
                    yield converted

        except asyncio.CancelledError:
            logger.info("Stream cancelled")
            result.cancel()
            raise
        except TimeoutError as exc:
            result.cancel()
            logger.exception("SDK streaming timed out")
            yield ModelError(str(exc))
        except Exception as exc:
            logger.exception("Error during SDK streaming")
            yield ModelError(str(exc))
        finally:
            self.last_capture = self._build_capture(
                result=result,
                context=context,
                definitions=definitions,
                collector=collector,
                run_start=run_start,
                trace_id=trace_id,
                model_completed_times=model_completed_times,
                model_completed_usages=model_completed_usages,
                cancellation_token=cancellation_token,
            )

    async def run(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
    ) -> AssistantResponse:
        """Execute the non-streaming compatibility path without configured Tools.

        Args:
            agent: Resolved AgentBreaker definition.
            context: Provider-neutral execution context.
            cancellation_token: Optional token checked before invoking the model.

        Returns:
            Typed assistant response for legacy callers.

        Raises:
            asyncio.CancelledError: When cancellation was already signalled.
        """
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError("Execution cancelled")

        sdk_agent = self._build_sdk_agent(agent, context.system_prompt, [])
        result = await Runner.run(
            starting_agent=sdk_agent,
            input=self._build_input(context),
            max_turns=10,
        )
        return AssistantResponse(content=str(result.final_output), role="assistant")

    def _build_sdk_agent(
        self, agent: AgentDefinition, system_prompt: str, tools: list[FunctionTool] | None = None
    ) -> Agent:
        """Translate an AgentBreaker definition into one request-scoped SDK Agent.

        Args:
            agent: Stable name/model/tool configuration.
            system_prompt: Request-specific prompt assembled by ``ContextBuilder``.
            tools: Audited SDK FunctionTools available to the model.

        Returns:
            SDK Agent configured with provider timeout, retry, and parallel Tool policy.
        """
        settings = get_settings()
        return Agent(
            name=agent.name,
            instructions=system_prompt,
            model=self.model_factory.create_model(agent.model),
            model_settings=ModelSettings(
                temperature=agent.temperature,
                max_tokens=agent.max_output_tokens,
                include_usage=True,
                extra_args={"timeout": settings.lite_llm_request_timeout_seconds},
                retry=ModelRetrySettings(max_retries=settings.lite_llm_max_retries),
                parallel_tool_calls=True,
            ),
            tools=tools or [],
        )

    def _resolve_tool_definitions(self, agent: AgentDefinition, registry: ToolRegistry) -> list[ToolDefinition]:
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
            definition = registry.get(tool_key)
            if definition is None:
                raise ValueError(f"Configured Tool is not registered: {tool_key}")
            definitions.append(definition)
        return definitions

    def _build_sdk_tools(
        self,
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        cancellation_token: CancellationToken | None,
    ) -> list[FunctionTool]:
        """Wrap decorated SDK Tools with auditing while preserving their generated metadata.

        ``dataclasses.replace`` keeps schema, strict mode, timeout, approval, guardrail, and future
        FunctionTool options produced by ``@function_tool``. Only the invocation hook changes.

        Args:
            definitions: Frozen Tool definitions selected by the Agent.
            collector: Request-scoped audit collector.
            cancellation_token: Token propagated into each Tool invocation.

        Returns:
            SDK FunctionTools with auditing hooks attached.
        """
        tools: list[FunctionTool] = []
        for definition in definitions:
            if definition.function_tool is None:
                raise ValueError(f"Tool has no SDK FunctionTool: {definition.tool_key}")

            async def invoke(
                tool_context: ToolContext[Any],
                arguments_json: str,
                captured: ToolDefinition = definition,
            ) -> str:
                """Invoke one SDK Tool through the audit collector.

                Args:
                    tool_context: SDK context containing the model call ID.
                    arguments_json: Exact serialized arguments emitted by the model.
                    captured: Frozen Tool definition bound by the loop closure.

                Returns:
                    Normalized result JSON returned to the SDK.
                """
                return await collector.execute(
                    tool_call_id=str(tool_context.tool_call_id),
                    definition=captured,
                    arguments_json=arguments_json,
                    tool_context=tool_context,
                    cancellation_token=cancellation_token,
                )

            # Preserve every option generated/configured by @function_tool and replace only the
            # invocation hook needed for AgentBreaker execution auditing.
            tools.append(replace(definition.function_tool, on_invoke_tool=invoke))
        return tools

    def _build_input(self, context: AgentContext) -> list[dict[str, Any]]:
        """Convert provider-neutral messages into this SDK's Responses API input shapes.

        Conversation Manager stores assistant Tool Calls and Tool results as neutral messages.
        Responses input represents them as separate ``function_call`` and
        ``function_call_output`` items, so this conversion remains at the SDK boundary.

        Args:
            context: Neutral typed history/current message including Tool Calls and model-only parts.

        Returns:
            Responses API input items with transient signed URLs confined to the current run.
        """

        def provider_content(message: Message) -> Any:
            """Translate one neutral content list into provider-bound Responses parts.

            Args:
                message: Neutral runtime message with optional typed model-only content.

            Returns:
                Scalar text or Responses-compatible text/image parts.
            """
            if not message.model_content:
                return message.content

            converted: list[dict[str, Any]] = []
            for part in message.model_content:
                if isinstance(part, ModelTextPart):
                    converted.append({"type": "input_text", "text": part.text})
                elif isinstance(part, ModelImagePart):
                    converted.append(
                        {
                            "type": "input_image",
                            "image_url": part.url,
                            "detail": part.detail,
                        }
                    )
            return converted

        input_items: list[dict[str, Any]] = []
        for message in context.conversation_history:
            if message.role == "assistant" and message.tool_calls:
                if message.content:
                    input_items.append({"role": "assistant", "content": message.content})
                for tool_call in message.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.call_id,
                            "name": tool_call.function_name,
                            "arguments": tool_call.arguments,
                        }
                    )
            elif message.role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id or "",
                        "output": message.content,
                    }
                )
            else:
                input_items.append(
                    {
                        "role": message.role,
                        "content": provider_content(message),
                    }
                )

        input_items.append(
            {
                "role": "user",
                "content": provider_content(context.current_message),
            }
        )
        return input_items

    def _to_capture_message(self, message: Message) -> CapturedMessage:
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
        """Map SDK raw/run-item events to the small event vocabulary exposed over SSE.

        Args:
            event: SDK stream event.
            collector: Collector used to enrich Tool result events with audit status.

        Returns:
            Typed runtime event, or ``None`` for SDK bookkeeping events.
        """
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent):
                if not event.data.delta:
                    return None
                return ModelTokenDelta(event.data.delta)

            if isinstance(event.data, ResponseCompletedEvent):
                return self._convert_response_completed_usage(event.data)

            return None

        if isinstance(event, RunItemStreamEvent):
            if event.name == "tool_called":
                raw_item = getattr(event.item, "raw_item", None)
                captured_call = CapturedToolCall(
                    tool_call_id=str(getattr(event.item, "call_id", None) or ""),
                    tool_name=getattr(raw_item, "name", "") if raw_item is not None else "",
                    arguments=getattr(raw_item, "arguments", None) if raw_item is not None else "{}",
                )
                if collector is not None:
                    collector.record_call(captured_call)
                return ModelToolStarted(
                    tool_name=captured_call.tool_name,
                    tool_call_id=captured_call.tool_call_id,
                    arguments_json=captured_call.arguments,
                )

            if event.name == "tool_output":
                tool_call_id = str(getattr(event.item, "call_id", None) or "")
                execution = collector.get(tool_call_id) if collector is not None else None
                return ModelToolCompleted(
                    tool_name=execution.tool_name if execution is not None else "",
                    tool_call_id=tool_call_id,
                    result=getattr(event.item, "output", None),
                    status=execution.status if execution is not None else "COMPLETED",
                )

        return None

    def _convert_response_completed_usage(self, event_data: Any) -> ModelUsage | None:
        """Extract provider usage when a completed response reports it.

        Args:
            event_data: SDK response-completed event.

        Returns:
            Typed usage event, or ``None`` when usage was omitted.
        """
        usage = getattr(getattr(event_data, "response", None), "usage", None)
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
        result,
        context: AgentContext,
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        run_start: int,
        trace_id: str,
        model_completed_times: list[int],
        model_completed_usages: list[tuple[int, int, int]],
        cancellation_token: CancellationToken | None,
    ) -> AgentRunCapture:
        """Reconstruct durable provider-neutral Turns from one completed or interrupted SDK run.

        One SDK raw response corresponds to one LLM Call/Turn. The first request stores a complete
        snapshot; later requests store only the assistant Tool Calls and Tool results appended by
        the preceding Turn.

        Args:
            result: SDK run result containing raw responses.
            context: Initial provider-neutral execution context.
            definitions: Frozen Tool definitions.
            collector: Observed Tool calls and executions.
            run_start: Epoch-millisecond request start.
            trace_id: Request trace identifier.
            model_completed_times: Completion timestamps observed from SDK events.
            model_completed_usages: Provider usage tuples in response order.
            cancellation_token: Optional token indicating interrupted capture.

        Returns:
            Durable provider-neutral capture for Conversation Manager persistence.
        """
        # Rebuild the normalized context supplied to the first model call. System instructions are
        # stored for audit even though the SDK receives them through Agent.instructions.
        initial_messages = [CapturedMessage(role="system", content=context.system_prompt)]
        initial_messages.extend(self._to_capture_message(message) for message in context.conversation_history)
        initial_messages.append(self._to_capture_message(context.current_message))

        turns: list[CapturedModelTurn] = []
        request_messages = initial_messages
        full_model_input = list(initial_messages)
        previous_turn_end = run_start
        raw_responses = list(result.raw_responses)
        # Cancellation can remove SDK raw_responses after tool_called was already emitted. Preserve
        # that observed evidence as a partial Turn instead of losing the Tool Call entirely.
        if not raw_responses and collector.calls():
            return self._build_observed_partial_capture(
                initial_messages=initial_messages,
                definitions=definitions,
                collector=collector,
                run_start=run_start,
                trace_id=trace_id,
                model_completed_times=model_completed_times,
                model_completed_usages=model_completed_usages,
                cancelled=cancellation_token is not None and cancellation_token.is_cancelled(),
            )
        assigned_tool_call_ids: set[str] = set()
        # Normalize every actual model response independently so one Tool loop becomes multiple
        # persisted Turns rather than one flattened request/answer pair.
        for index, response in enumerate(raw_responses):
            event_llm_end = (
                model_completed_times[index]
                if index < len(model_completed_times)
                else max(previous_turn_end, epoch_millis())
            )
            normalized_output = self._normalize_response_output(response.output)
            response_content = normalized_output.content
            tool_calls = list(normalized_output.tool_calls)
            if not tool_calls and index == len(raw_responses) - 1:
                # The SDK may clear a cancelled response's output after already publishing
                # tool_called events. Those events remain authoritative audit evidence.
                tool_calls = [call for call in collector.calls() if call.tool_call_id not in assigned_tool_call_ids]
            assigned_tool_call_ids.update(call.tool_call_id for call in tool_calls)
            executions = self._complete_execution_audit(
                tool_calls=tool_calls,
                definitions=definitions,
                collector=collector,
                fallback_time=event_llm_end,
                cancelled=cancellation_token is not None and cancellation_token.is_cancelled(),
            )
            # Some providers begin scheduling Tool handlers before the SDK publishes the response
            # completed event. Persist the LLM boundary no later than the first observed Tool start.
            llm_end = min([event_llm_end, *(execution.start_time for execution in executions)])
            llm_end = max(previous_turn_end, llm_end)
            turn_end = max([llm_end, *(execution.end_time for execution in executions)])
            # Raw payloads are audit snapshots assembled from SDK objects. Logical replay uses the
            # normalized messages above and never depends on these provider-oriented JSON strings.
            raw_request = json.dumps(
                {
                    "messages": [captured_message_to_dict(message) for message in full_model_input],
                    "tools": [self._definition_dict(definition) for definition in definitions],
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            raw_response = json.dumps(
                {
                    "output": [self._model_dump(item) for item in response.output],
                    "observed_tool_calls": [
                        {
                            "id": call.tool_call_id,
                            "name": call.tool_name,
                            "arguments": call.arguments,
                        }
                        for call in tool_calls
                    ],
                    "response_id": getattr(response, "response_id", None),
                    "request_id": getattr(response, "request_id", None),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
            turns.append(
                CapturedModelTurn(
                    request_messages=request_messages,
                    message_storage_mode="FULL_SNAPSHOT" if index == 0 else "APPEND_DELTA",
                    tools=definitions,
                    response_content=response_content,
                    response_tool_calls=tool_calls,
                    tool_executions=executions,
                    request_id=(
                        getattr(response, "request_id", None) or getattr(response, "response_id", None) or str(uuid4())
                    ),
                    trace_id=trace_id,
                    start_time=previous_turn_end,
                    llm_end_time=llm_end,
                    end_time=turn_end,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                    raw_request=raw_request,
                    raw_response=raw_response,
                )
            )
            # Only Tool-producing responses have a following LLM request. Persist precisely the
            # assistant call plus ordered execution outputs required for that next request.
            request_messages = self._next_turn_delta(response_content, tool_calls, executions)
            full_model_input.extend(request_messages)
            previous_turn_end = turn_end
        return AgentRunCapture(turns=turns, final_output=str(result.final_output or ""))

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
    ) -> AgentRunCapture:
        """Build a cancellable audit Turn when SDK raw output is no longer available.

        This path uses model-emitted ``tool_called`` events and collected executions. It does not
        invent a successful response; the owning Round/Turn remains FAILED or CANCELLED.

        Args:
            initial_messages: Messages supplied to the first model call.
            definitions: Frozen Tool definitions.
            collector: Calls observed before cancellation.
            run_start: Epoch-millisecond request start.
            trace_id: Request trace identifier.
            model_completed_times: Observed model completion timestamps.
            model_completed_usages: Observed provider usage tuples.
            cancelled: Whether cancellation caused the missing raw response.

        Returns:
            Partial capture preserving observed Tool evidence.
        """
        initial_messages = [self._coerce_capture_message(message) for message in initial_messages]
        tool_calls = collector.calls()
        event_llm_end = model_completed_times[-1] if model_completed_times else epoch_millis()
        executions = self._complete_execution_audit(
            tool_calls=tool_calls,
            definitions=definitions,
            collector=collector,
            fallback_time=event_llm_end,
            cancelled=cancelled,
        )
        llm_end = max(run_start, min([event_llm_end, *(item.start_time for item in executions)]))
        turn_end = max([llm_end, *(item.end_time for item in executions)])
        prompt_tokens, completion_tokens, total_tokens = (
            model_completed_usages[-1] if model_completed_usages else (0, 0, 0)
        )
        raw_request = json.dumps(
            {
                "messages": [captured_message_to_dict(message) for message in initial_messages],
                "tools": [self._definition_dict(definition) for definition in definitions],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        raw_response = json.dumps(
            {
                "output": [],
                "observed_tool_calls": [
                    {"id": call.tool_call_id, "name": call.tool_name, "arguments": call.arguments}
                    for call in tool_calls
                ],
                "raw_response_removed_by_sdk_cancellation": cancelled,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        turn = CapturedModelTurn(
            request_messages=initial_messages,
            message_storage_mode="FULL_SNAPSHOT",
            tools=definitions,
            response_content="",
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
            raw_request=raw_request,
            raw_response=raw_response,
        )
        return AgentRunCapture(turns=[turn], final_output="")

    def _complete_execution_audit(
        self,
        *,
        tool_calls: list[CapturedToolCall],
        definitions: list[ToolDefinition],
        collector: ToolExecutionCollector,
        fallback_time: int,
        cancelled: bool,
    ) -> list[CapturedToolExecution]:
        """Guarantee one terminal execution record for every model-emitted Tool Call.

        Missing outcomes can occur when cancellation wins before a Tool coroutine starts. The
        synthesized FAILED/CANCELLED record makes the one-to-one persistence invariant explicit.

        Args:
            tool_calls: Model-emitted calls observed by the collector.
            definitions: Frozen definitions used to resolve Tool keys.
            collector: Existing successful/failed executions.
            fallback_time: Timestamp used for synthesized records.
            cancelled: Whether missing executions should be marked CANCELLED.

        Returns:
            Complete execution list aligned with model-emitted calls.
        """
        definitions_by_name = {definition.tool_name: definition for definition in definitions}
        executions: list[CapturedToolExecution] = []
        for tool_call in tool_calls:
            execution = collector.get(tool_call.tool_call_id)
            if execution is not None:
                executions.append(execution)
                continue

            definition = definitions_by_name.get(tool_call.tool_name)
            status = "CANCELLED" if cancelled else "FAILED"
            error_message = (
                "Generation cancelled before the Tool produced a result."
                if cancelled
                else "Tool execution did not produce an auditable result."
            )
            result_content = (
                ""
                if cancelled
                else json.dumps(
                    {"status": "error", "error": error_message},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            executions.append(
                CapturedToolExecution(
                    tool_call_id=tool_call.tool_call_id,
                    tool_key=definition.tool_key if definition is not None else "unknown",
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    status=status,
                    result_content=result_content,
                    raw_result=result_content,
                    error_message=error_message,
                    start_time=fallback_time,
                    end_time=fallback_time,
                )
            )
        return executions

    def _normalize_response_output(self, outputs: list[Any]) -> NormalizedModelOutput:
        """Extract assistant text and Tool calls from heterogeneous SDK response items.

        Args:
            outputs: SDK response output items.

        Returns:
            One typed value containing assistant text and Tool Calls in model order.
        """
        text_parts: list[str] = []
        tool_calls: list[CapturedToolCall] = []
        for item in outputs:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for part in getattr(item, "content", []) or []:
                    text = getattr(part, "text", None)
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

    def _coerce_capture_message(self, message: CapturedMessage | dict[str, Any]) -> CapturedMessage:
        """Accept legacy test/provider snapshots at the JSON boundary before internal capture."""
        if isinstance(message, CapturedMessage):
            return message
        tool_calls = tuple(
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

    def _next_turn_delta(
        self,
        response_content: str,
        tool_calls: list[CapturedToolCall],
        executions,
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
        executions_by_id = {execution.tool_call_id: execution for execution in executions}
        for call in tool_calls:
            execution = executions_by_id.get(call.tool_call_id)
            messages.append(
                CapturedMessage(
                    role="tool",
                    content=execution.result_content if execution is not None else "",
                    tool_call_id=call.tool_call_id,
                )
            )
        return messages

    def _definition_dict(self, definition: ToolDefinition) -> dict[str, Any]:
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

    def _model_dump(self, value: Any) -> Any:
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
        model_dump = getattr(value, "model_dump", None)
        return model_dump(mode="json") if callable(model_dump) else value

    async def close(self):
        """Release HTTP resources owned by the LiteLLM model factory."""
        await self.model_factory.close()
