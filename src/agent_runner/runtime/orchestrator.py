"""
Runtime orchestrator module.

This module provides the core orchestration logic for agent execution,
coordinating configuration loading, context building, tool execution,
and model invocation through the OpenAI Agents SDK runtime.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from time import time_ns
from uuid import uuid4

from agent_breaker_conversation_manager_protos.ifl.agentbreaker.commons import AgentIdentity
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    AssistantAnswer,
    AssistantMessage,
    ConversationTurn,
    LlmCall,
    LlmConversationMessage,
    LlmMessageStorageMode,
    LlmRequest,
    LlmResponse,
    MessageRole,
    RoundStatus,
    SaveConversationRoundRequest,
    TokenUsage,
    TurnStatus,
    UserRequest,
)
from fastapi import Request

from agent_runner.agent_definitions.factory import AgentFactory
from agent_runner.agent_definitions.loader import AgentConfigLoader
from agent_runner.api.streaming import (
    DoneEvent,
    ErrorEvent,
    PersistedEvent,
    SavingEvent,
    StreamEvent,
    TokenDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
    UsageEvent,
)
from agent_runner.config import ChatRequest, get_settings
from agent_runner.context.builder import ContextBuilder, Message
from agent_runner.conversation import (
    ConversationBusyError,
    ConversationExecutionLock,
    ConversationManagerClient,
)
from agent_runner.runtime.cancellation import CancellationManager, conversation_cancellation_registry
from agent_runner.runtime.openai_agents_runtime import OpenAIAgentsRuntime
from agent_runner.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class RuntimeOrchestrator:
    """
    Core runtime orchestrator for agent execution.

    This class coordinates all components needed to execute an agent request:
    - Configuration loading from cache/files/remote service
    - Context building from conversation history, profile, and RAG
    - Agent instantiation with loaded configuration
    - Tool execution and MCP management
    - Model invocation through OpenAI Agents SDK
    - Streaming response generation
    - Request cancellation handling

    The orchestrator follows a request-scoped lifecycle, creating fresh instances
    for each request rather than maintaining long-term state.

    Attributes:
        config_loader: Loader for agent configurations.
        context_builder: Builder for agent execution context.
        agent_factory: Factory for creating agent instances.
        tool_executor: Executor for tool invocations.
        cancellation_manager: Manager for request cancellation tokens.
        openai_runtime: Runtime wrapper for OpenAI Agents SDK.
    """

    def __init__(self):
        """
        Initialize the runtime orchestrator with all required components.

        Creates instances of all sub-components needed for agent execution,
        including configuration loader, context builder, agent factory,
        tool executor, cancellation manager, and OpenAI runtime wrapper.
        """
        self.config_loader = AgentConfigLoader()
        self.context_builder = ContextBuilder()
        self.agent_factory = AgentFactory()
        self.tool_executor = ToolExecutor()
        self.cancellation_manager = CancellationManager()
        self.openai_runtime = OpenAIAgentsRuntime()
        self.conversation_client = ConversationManagerClient()
        self.execution_lock = ConversationExecutionLock()
        self._lock_acquired = False
        self._terminal_round_persisted = False

    async def acquire_conversation(self, conversation_id: str) -> None:
        """Acquire the conversation execution lease before an SSE response is opened."""
        await self.execution_lock.acquire(conversation_id)
        self._lock_acquired = True

    async def run(
        self, chat_request: ChatRequest, user_id: int, http_request: Request
    ) -> AsyncGenerator[StreamEvent]:
        """
        Execute an agent chat request and stream responses.

        This method orchestrates the complete agent execution flow:
        1. Load agent configuration
        2. Build execution context from conversation history, profile, and RAG
        3. Create agent instance with configuration
        4. Stream responses through OpenAI Agents SDK runtime
        5. Handle client disconnect and cancellation
        6. Clean up resources on completion or error

        Args:
            chat_request: The chat request containing agent ID, user info, and message.
            http_request: The HTTP request object for disconnect detection.

        Returns:
            AsyncGenerator[StreamEvent, None]: Stream of events including:
                - TokenDeltaEvent: Text tokens from model response
                - ToolStartEvent: Tool invocation start
                - ToolResultEvent: Tool execution result
                - UsageEvent: Token usage reported by the upstream model response
                - ErrorEvent: Error messages
                - DoneEvent: Completion marker

        Raises:
            asyncio.CancelledError: If the request is cancelled by client disconnect.
        """
        cancellation_token = self.cancellation_manager.create_token()
        round_start = _epoch_millis()
        next_round_number: int | None = None
        user_request_validated = False

        try:
            if not getattr(self, "_lock_acquired", False):
                await self.acquire_conversation(chat_request.conversation_id)
            conversation_cancellation_registry.register(
                user_id, chat_request.conversation_id, cancellation_token
            )

            history = await self.conversation_client.get_round_history(user_id, chat_request.conversation_id)
            if history.base is None or not history.base.success:
                message = history.base.message if history.base is not None else "Conversation history RPC failed."
                yield ErrorEvent(message, error_code="CONVERSATION_ACCESS_DENIED", phase="preflight")
                return
            if history.data is None:
                yield ErrorEvent(
                    "Conversation history RPC returned no data.",
                    error_code="INVALID_HISTORY_RESPONSE",
                    phase="preflight",
                )
                return
            next_round_number = history.data.latest_round_number + 1
            user_request_validated = True

            conversation_history: list[Message] = []
            if history.data.latest_round_number > 0:
                replay = await self.conversation_client.get_model_context(
                    user_id, chat_request.conversation_id, history.data.latest_round_number
                )
                if replay.base is None or not replay.base.success:
                    message = replay.base.message if replay.base is not None else "Conversation replay RPC failed."
                    yield ErrorEvent(message, error_code="REPLAY_FAILED", phase="preflight")
                    return
                if replay.data is None:
                    yield ErrorEvent(
                        "Conversation replay RPC returned no data.",
                        error_code="INVALID_REPLAY_RESPONSE",
                        phase="preflight",
                    )
                    return
                conversation_history = self._to_context_messages(replay.data.context_messages)

            settings = get_settings()
            agent_config = await self.config_loader.load(settings.default_agent_id)

            context = await self.context_builder.build(
                agent_config=agent_config,
                conversation_id=chat_request.conversation_id,
                user_id=user_id,
                current_message=chat_request.message,
                conversation_history=conversation_history,
            )

            agent = await self.agent_factory.create(agent_config)
            response_text = ""
            usage: UsageEvent | None = None
            call_start = _epoch_millis()

            async for event in self.openai_runtime.run_streamed(agent, context, cancellation_token):
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, cancelling request")
                    cancellation_token.cancel()
                    await self._persist_terminal_round(
                        user_id, chat_request, next_round_number, round_start,
                        RoundStatus.CANCELLED, "Generation cancelled."
                    )
                    return

                if cancellation_token.is_cancelled():
                    await self._persist_terminal_round(
                        user_id, chat_request, next_round_number, round_start,
                        RoundStatus.CANCELLED, "Generation cancelled."
                    )
                    return

                converted = self._convert_event(event)
                if isinstance(converted, ErrorEvent):
                    await self._persist_terminal_round(
                        user_id, chat_request, next_round_number, round_start,
                        RoundStatus.FAILED, converted.error_message or "Model execution failed."
                    )
                    yield converted
                    return
                if isinstance(converted, TokenDeltaEvent):
                    response_text += converted.content or ""
                elif isinstance(converted, UsageEvent):
                    usage = converted
                yield converted

                if await http_request.is_disconnected():
                    cancellation_token.cancel()

            call_end = _epoch_millis()
            if cancellation_token.is_cancelled() or await http_request.is_disconnected():
                await self._persist_terminal_round(
                    user_id, chat_request, next_round_number, round_start,
                    RoundStatus.CANCELLED, "Generation cancelled."
                )
                return
            if not response_text.strip():
                await self._persist_terminal_round(
                    user_id, chat_request, next_round_number, round_start,
                    RoundStatus.FAILED, "The model returned an empty response."
                )
                yield ErrorEvent(
                    "The model returned an empty response.",
                    error_code="EMPTY_MODEL_RESPONSE",
                    phase="model",
                )
                return

            save_request = self._build_save_request(
                user_id=user_id,
                conversation_id=chat_request.conversation_id,
                user_message=chat_request.message,
                response_text=response_text,
                agent=agent,
                system_prompt=context.system_prompt,
                usage=usage,
                round_start=round_start,
                call_start=call_start,
                call_end=call_end,
                round_number=next_round_number,
                conversation_history=conversation_history,
            )
            yield SavingEvent()
            try:
                saved = await self.conversation_client.save_round(save_request)
            except Exception as error:
                logger.exception("Failed to persist completed round")
                yield ErrorEvent(str(error), error_code="PERSISTENCE_FAILED", phase="persistence")
                return
            if saved.base is None or not saved.base.success:
                message = saved.base.message if saved.base is not None else "Round persistence RPC failed."
                yield ErrorEvent(message, error_code="PERSISTENCE_FAILED", phase="persistence")
                return

            yield PersistedEvent()
            yield DoneEvent()

        except asyncio.CancelledError:
            logger.info("Request cancelled")
            if user_request_validated and next_round_number is not None:
                await self._persist_terminal_round(
                    user_id, chat_request, next_round_number, round_start,
                    RoundStatus.CANCELLED, "Generation cancelled."
                )
            await self._cleanup(cancellation_token)
            raise

        except GeneratorExit:
            if user_request_validated and next_round_number is not None:
                await self._persist_terminal_round(
                    user_id, chat_request, next_round_number, round_start,
                    RoundStatus.CANCELLED, "Generation cancelled."
                )
            raise

        except ConversationBusyError as error:
            yield ErrorEvent(str(error), error_code="CONVERSATION_BUSY", phase="preflight")

        except Exception as e:
            logger.exception("Error during agent execution")
            if user_request_validated and next_round_number is not None:
                await self._persist_terminal_round(
                    user_id, chat_request, next_round_number, round_start,
                    RoundStatus.FAILED, str(e) or "Agent execution failed."
                )
            yield ErrorEvent(error_message=str(e), error_code="EXECUTION_FAILED", phase="execution")
        finally:
            conversation_cancellation_registry.unregister(
                user_id, chat_request.conversation_id, cancellation_token
            )
            await self._cleanup(cancellation_token)
            if getattr(self, "_lock_acquired", False):
                await self.execution_lock.release()
                self._lock_acquired = False

    def _build_save_request(
        self,
        *,
        user_id: int,
        conversation_id: str,
        user_message: str,
        response_text: str,
        agent,
        system_prompt: str,
        usage: UsageEvent | None,
        round_start: int,
        call_start: int,
        call_end: int,
        round_number: int,
        conversation_history: list[Message],
    ) -> SaveConversationRoundRequest:
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            )

        llm_request = LlmRequest(
            provider="litellm",
            model=agent.model,
            messages=[LlmConversationMessage(role=MessageRole.SYSTEM, content=system_prompt)]
            + [self._to_llm_message(message) for message in conversation_history]
            + [LlmConversationMessage(role=MessageRole.USER, content=user_message)],
            temperature=agent.temperature,
            max_output_tokens=agent.max_output_tokens,
            message_storage_mode=LlmMessageStorageMode.FULL_SNAPSHOT,
        )
        llm_response = LlmResponse(
            message=AssistantMessage(content=response_text),
            finish_reason="stop",
            usage=token_usage,
        )
        turn = ConversationTurn(
            turn_number=1,
            llm_call=LlmCall(
                request=llm_request,
                response=llm_response,
                request_id=str(uuid4()),
                trace_id=str(uuid4()),
                start_time=call_start,
                end_time=call_end,
            ),
            status=TurnStatus.COMPLETED,
            start_time=call_start,
            end_time=call_end,
            agent_identity=AgentIdentity(
                agent_id=agent.agent_id,
                name=agent.name,
                version=agent.version,
            ),
        )
        return SaveConversationRoundRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            round_number=round_number,
            user_request=UserRequest(content=user_message),
            turns=[turn],
            final_answer=AssistantAnswer(content=response_text, source_turn_number=1),
            status=RoundStatus.COMPLETED,
            start_time=round_start,
            end_time=call_end,
        )

    async def _persist_terminal_round(
        self,
        user_id: int,
        chat_request: ChatRequest,
        round_number: int,
        round_start: int,
        status: RoundStatus,
        error_message: str,
    ) -> None:
        if getattr(self, "_terminal_round_persisted", False):
            return
        request = SaveConversationRoundRequest(
            user_id=user_id,
            conversation_id=chat_request.conversation_id,
            round_number=round_number,
            user_request=UserRequest(content=chat_request.message),
            status=status,
            error_message=error_message,
            start_time=round_start,
            end_time=max(round_start, _epoch_millis()),
        )
        try:
            response = await self.conversation_client.save_round(request)
            if response.base is None or not response.base.success:
                logger.error("Failed to persist terminal round: %s", response.base)
            else:
                self._terminal_round_persisted = True
        except Exception:
            logger.exception("Failed to persist terminal round")

    def _to_context_messages(self, messages) -> list[Message]:
        role_names = {
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.TOOL: "tool",
        }
        return [
            Message(role=role_names[message.role], content=message.content)
            for message in messages
            if message.role in role_names
        ]

    def _to_llm_message(self, message: Message) -> LlmConversationMessage:
        roles = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "tool": MessageRole.TOOL,
        }
        return LlmConversationMessage(role=roles[message.role], content=message.content)

    def _convert_event(self, event: dict) -> StreamEvent:
        """
        Convert raw runtime event dictionary to structured StreamEvent object.

        Args:
            event: Raw event dictionary from OpenAI Agents SDK runtime.

        Returns:
            StreamEvent: Structured event object based on event type:
                - TokenDeltaEvent for "token_delta" events
                - ToolStartEvent for "tool_start" events
                - ToolResultEvent for "tool_result" events
                - UsageEvent for "usage" events
                - ErrorEvent for "error" events
                - TokenDeltaEvent (fallback) for unknown event types
        """
        event_type = event.get("type")

        if event_type == "token_delta":
            return TokenDeltaEvent(content=event.get("content", ""))
        elif event_type == "tool_start":
            return ToolStartEvent(tool=event.get("tool", ""), tool_args=event.get("args"))
        elif event_type == "tool_result":
            return ToolResultEvent(tool=event.get("tool", ""), tool_result=event.get("tool_result") or event.get("result"))
        elif event_type == "usage":
            return UsageEvent(
                prompt_tokens=event["prompt_tokens"],
                completion_tokens=event["completion_tokens"],
                total_tokens=event["total_tokens"],
            )
        elif event_type == "error":
            return ErrorEvent(error_message=event.get("error_message") or event.get("content", ""))
        else:
            return TokenDeltaEvent(content=event.get("content", ""))

    async def _cleanup(self, cancellation_token):
        """
        Clean up resources after request cancellation or completion.

        Args:
            cancellation_token: The cancellation token to clean up.
        """
        await self.cancellation_manager.cleanup(cancellation_token)

    async def close(self):
        await self.config_loader.close()
        await self.openai_runtime.close()
        await self.conversation_client.close()
        await self.execution_lock.close()


def _epoch_millis() -> int:
    return time_ns() // 1_000_000
