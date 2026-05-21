import asyncio
import logging
from typing import AsyncGenerator

from fastapi import Request

from agents.factory import AgentFactory
from agents.loader import AgentConfigLoader
from api.streaming import DoneEvent, ErrorEvent, StreamEvent, TokenDeltaEvent, ToolResultEvent, ToolStartEvent
from config import ChatRequest
from context.builder import ContextBuilder
from runtime.cancellation import CancellationManager
from sdk.openai_runtime import OpenAIAgentsRuntime
from tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class RuntimeOrchestrator:
    def __init__(self):
        self.config_loader = AgentConfigLoader()
        self.context_builder = ContextBuilder()
        self.agent_factory = AgentFactory()
        self.tool_executor = ToolExecutor()
        self.cancellation_manager = CancellationManager()
        self.openai_runtime = OpenAIAgentsRuntime()

    async def run(self, chat_request: ChatRequest, http_request: Request) -> AsyncGenerator[StreamEvent, None]:
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
        event_type = event.get("type")

        if event_type == "token_delta":
            return TokenDeltaEvent(content=event.get("content", ""))
        elif event_type == "tool_start":
            return ToolStartEvent(tool=event.get("tool", ""), tool_args=event.get("args"))
        elif event_type == "tool_result":
            return ToolResultEvent(tool=event.get("tool", ""), tool_result=event.get("result"))
        elif event_type == "error":
            return ErrorEvent(error_message=event.get("error_message") or event.get("content", ""))
        else:
            return TokenDeltaEvent(content=event.get("content", ""))

    async def _cleanup(self, cancellation_token):
        await self.cancellation_manager.cleanup(cancellation_token)
