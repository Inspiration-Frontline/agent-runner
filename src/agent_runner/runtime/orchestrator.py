"""
Runtime orchestrator module.

This module provides the core orchestration logic for agent execution,
coordinating configuration loading, context building, tool execution,
and model invocation through the OpenAI Agents SDK runtime.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from fastapi import Request

from agent_runner.agent_definitions.factory import AgentFactory
from agent_runner.agent_definitions.loader import AgentConfigLoader
from agent_runner.api.streaming import (
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TokenDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
    UsageEvent,
)
from agent_runner.config import ChatRequest
from agent_runner.context.builder import ContextBuilder
from agent_runner.runtime.cancellation import CancellationManager
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

    async def run(self, chat_request: ChatRequest, http_request: Request) -> AsyncGenerator[StreamEvent]:
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

        try:
            agent_config = await self.config_loader.load(chat_request.agent_id, chat_request.version)

            context = await self.context_builder.build(
                agent_config=agent_config,
                conversation_id=chat_request.conversation_id,
                user_id=chat_request.user_id,
                current_message=chat_request.message,
            )

            agent = await self.agent_factory.create(agent_config)

            async for event in self.openai_runtime.run_streamed(agent, context, cancellation_token):
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, cancelling request")
                    cancellation_token.cancel()
                    break

                if cancellation_token.is_cancelled():
                    break

                yield self._convert_event(event)

                if await http_request.is_disconnected():
                    cancellation_token.cancel()

            yield DoneEvent()

        except asyncio.CancelledError:
            logger.info("Request cancelled")
            await self._cleanup(cancellation_token)
            raise

        except Exception as e:
            logger.exception("Error during agent execution")
            yield ErrorEvent(error_message=str(e))

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
