import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import replace
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
from agent_runner.context.builder import AgentContext
from agent_runner.gateway.litellm_client import LiteLLMModelFactory
from agent_runner.runtime.cancellation import CancellationToken
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


class OpenAIAgentsRuntime:
    """
    Runtime for executing AgentBreaker agents through openai-agents-python.

    This class converts the local AgentBreaker agent definition and context into
    an Agents SDK Agent, delegates the agent loop and model stream handling to the
    SDK, then maps semantic SDK stream events back to AgentBreaker's SSE event
    dictionary contract.
    """

    def __init__(self, model_factory: LiteLLMModelFactory | None = None):
        """Create a runtime and initialize the request result snapshot exposed to the orchestrator."""
        self.model_factory = model_factory or LiteLLMModelFactory()
        self.last_capture = AgentRunCapture()

    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Run one request through the SDK Tool loop and yield AgentBreaker semantic events.

        The SDK owns repeated model/Tool scheduling. AgentBreaker wraps the configured decorated
        FunctionTools only to collect persistence evidence and translate SDK events to SSE.
        ``last_capture`` is finalized even when streaming fails or is cancelled.
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
                    model_completed_usages.append((
                        getattr(usage, "input_tokens", 0) if usage is not None else 0,
                        getattr(usage, "output_tokens", 0) if usage is not None else 0,
                        getattr(usage, "total_tokens", 0) if usage is not None else 0,
                    ))
                if converted is not None:
                    yield converted

        except asyncio.CancelledError:
            logger.info("Stream cancelled")
            result.cancel()
            raise
        except TimeoutError as exc:
            result.cancel()
            logger.exception("SDK streaming timed out")
            yield {"type": "error", "content": str(exc)}
        except Exception as exc:
            logger.exception("Error during SDK streaming")
            yield {"type": "error", "content": str(exc)}
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
    ) -> dict[str, Any]:
        """Execute the non-streaming compatibility path without configured Tools."""
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError("Execution cancelled")

        sdk_agent = self._build_sdk_agent(agent, context.system_prompt, [])
        result = await Runner.run(
            starting_agent=sdk_agent,
            input=self._build_input(context),
            max_turns=10,
        )
        return {
            "content": result.final_output,
            "role": "assistant",
        }

    def _build_sdk_agent(
        self, agent: AgentDefinition, system_prompt: str, tools: list[FunctionTool] | None = None
    ) -> Agent:
        """Translate an AgentBreaker definition into one request-scoped SDK Agent."""
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

    def _resolve_tool_definitions(
        self, agent: AgentDefinition, registry: ToolRegistry
    ) -> list[ToolDefinition]:
        """Resolve stable configured Tool keys and fail before model invocation when one is missing."""
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
        """Convert provider-neutral replay messages into Responses API input item shapes.

        Conversation Manager stores assistant Tool Calls and Tool results as neutral messages.
        Responses input represents them as separate ``function_call`` and
        ``function_call_output`` items, so this conversion remains at the SDK boundary.
        """
        input_items: list[dict[str, Any]] = []
        for message in context.conversation_history:
            tool_calls = message.metadata.get("tool_calls") or []
            if message.role == "assistant" and tool_calls:
                if message.content:
                    input_items.append({"role": "assistant", "content": message.content})
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id") or "",
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "{}",
                    })
            elif message.role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": message.metadata.get("tool_call_id") or "",
                    "output": message.content,
                })
            else:
                input_items.append({"role": message.role, "content": message.content})

        input_items.append({
            "role": "user",
            "content": context.current_message.content,
        })
        return input_items

    def _convert_stream_event(
        self,
        event: StreamEvent,
        collector: ToolExecutionCollector | None = None,
    ) -> dict[str, Any] | None:
        """Map SDK raw/run-item events to the small event vocabulary exposed over SSE."""
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent):
                if not event.data.delta:
                    return None
                return {
                    "type": "token_delta",
                    "content": event.data.delta,
                }

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
                return {
                    "type": "tool_start",
                    "tool": captured_call.tool_name,
                    "tool_call_id": captured_call.tool_call_id,
                    "args": captured_call.arguments,
                }

            if event.name == "tool_output":
                tool_call_id = str(getattr(event.item, "call_id", None) or "")
                execution = collector.get(tool_call_id) if collector is not None else None
                return {
                    "type": "tool_result",
                    "tool": execution.tool_name if execution is not None else "",
                    "tool_call_id": tool_call_id,
                    "tool_result": getattr(event.item, "output", None),
                    "tool_status": execution.status if execution is not None else "COMPLETED",
                }

        return None

    def _convert_response_completed_usage(self, event_data: Any) -> dict[str, Any] | None:
        """Extract provider usage when a completed response reports it."""
        usage = getattr(getattr(event_data, "response", None), "usage", None)
        if usage is None:
            return None

        return {
            "type": "usage",
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }

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
        """
        # Rebuild the normalized context supplied to the first model call. System instructions are
        # stored for audit even though the SDK receives them through Agent.instructions.
        initial_messages = [{"role": "system", "content": context.system_prompt}]
        initial_messages.extend(
            {"role": message.role, "content": message.content, **message.metadata}
            for message in context.conversation_history
        )
        initial_messages.append({"role": "user", "content": context.current_message.content})

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
            response_content, tool_calls = self._normalize_response_output(response.output)
            if not tool_calls and index == len(raw_responses) - 1:
                # The SDK may clear a cancelled response's output after already publishing
                # tool_called events. Those events remain authoritative audit evidence.
                tool_calls = [
                    call for call in collector.calls()
                    if call.tool_call_id not in assigned_tool_call_ids
                ]
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
                    "messages": full_model_input,
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
            turns.append(CapturedModelTurn(
                request_messages=request_messages,
                message_storage_mode="FULL_SNAPSHOT" if index == 0 else "APPEND_DELTA",
                tools=definitions,
                response_content=response_content,
                response_tool_calls=tool_calls,
                tool_executions=executions,
                request_id=(
                    getattr(response, "request_id", None)
                    or getattr(response, "response_id", None)
                    or str(uuid4())
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
            ))
            # Only Tool-producing responses have a following LLM request. Persist precisely the
            # assistant call plus ordered execution outputs required for that next request.
            request_messages = self._next_turn_delta(response_content, tool_calls, executions)
            full_model_input.extend(request_messages)
            previous_turn_end = turn_end
        return AgentRunCapture(turns=turns, final_output=str(result.final_output or ""))

    def _build_observed_partial_capture(
        self,
        *,
        initial_messages: list[dict[str, Any]],
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
        """
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
                "messages": initial_messages,
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
            result_content = "" if cancelled else json.dumps(
                {"status": "error", "error": error_message},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            executions.append(CapturedToolExecution(
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
            ))
        return executions

    def _normalize_response_output(self, outputs: list[Any]) -> tuple[str, list[CapturedToolCall]]:
        """Extract assistant text and function calls from heterogeneous SDK response items."""
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
                tool_calls.append(CapturedToolCall(
                    tool_call_id=str(getattr(item, "call_id", "")),
                    tool_name=str(getattr(item, "name", "")),
                    arguments=str(getattr(item, "arguments", "{}")),
                ))
        return "".join(text_parts), tool_calls

    def _next_turn_delta(
        self,
        response_content: str,
        tool_calls: list[CapturedToolCall],
        executions,
    ) -> list[dict[str, Any]]:
        """Create the provider-neutral APPEND_DELTA consumed by the next model Turn."""
        if not tool_calls:
            return []
        messages: list[dict[str, Any]] = [{
            "role": "assistant",
            "content": response_content,
            "tool_calls": [
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {"name": call.tool_name, "arguments": call.arguments},
                }
                for call in tool_calls
            ],
        }]
        executions_by_id = {execution.tool_call_id: execution for execution in executions}
        for call in tool_calls:
            execution = executions_by_id.get(call.tool_call_id)
            messages.append({
                "role": "tool",
                "content": execution.result_content if execution is not None else "",
                "tool_call_id": call.tool_call_id,
            })
        return messages

    def _definition_dict(self, definition: ToolDefinition) -> dict[str, Any]:
        """Serialize the frozen AgentBreaker Tool definition included in raw request audit JSON."""
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
        """
        model_dump = getattr(value, "model_dump", None)
        return model_dump(mode="json") if callable(model_dump) else value

    async def close(self):
        """Release HTTP resources owned by the LiteLLM model factory."""
        await self.model_factory.close()
