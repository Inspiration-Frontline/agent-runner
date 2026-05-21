import asyncio
import logging
from typing import Any, AsyncGenerator

from agents.config_models import AgentDefinition
from context.builder import AgentContext
from gateway.litellm_client import LiteLLMClient
from runtime.cancellation import CancellationToken

logger = logging.getLogger(__name__)


class OpenAIAgentsRuntime:
    def __init__(self):
        self.litellm_client = LiteLLMClient()

    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        messages = self._build_messages(agent, context)
        tools = self._build_tools(context)

        try:
            async for event in self.litellm_client.chat_completion_stream(
                model=agent.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=agent.temperature,
                max_tokens=agent.max_output_tokens,
            ):
                if cancellation_token and cancellation_token.is_cancelled():
                    logger.info("Stream cancelled by token")
                    break

                yield event

        except asyncio.CancelledError:
            logger.info("Stream cancelled")
            raise
        except Exception as e:
            if agent.mock_response:
                logger.warning(
                    "Error during streaming for agent %s, falling back to mock response: %s",
                    agent.agent_id,
                    e,
                )
                async for event in self._stream_mock_response(agent.mock_response):
                    yield event
                return
            logger.exception("Error during streaming")
            yield {"type": "error", "content": str(e)}

    async def run(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        messages = self._build_messages(agent, context)
        tools = self._build_tools(context)

        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError("Execution cancelled")

        try:
            response = await self.litellm_client.chat_completion(
                model=agent.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=agent.temperature,
                max_tokens=agent.max_output_tokens,
            )

            return self._parse_response(response)

        except Exception as e:
            logger.exception("Error during completion")
            raise

    def _build_messages(self, agent: AgentDefinition, context: AgentContext) -> list[dict[str, Any]]:
        messages = []

        messages.append({
            "role": "system",
            "content": context.system_prompt,
        })

        for msg in context.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        messages.append({
            "role": "user",
            "content": context.current_message.content,
        })

        return messages

    def _build_tools(self, context: AgentContext) -> list[dict[str, Any]]:
        return context.tool_specs

    def _parse_response(self, response: Any) -> dict[str, Any]:
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content,
            "role": message.role,
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        return result

    async def _stream_mock_response(self, mock_response: str) -> AsyncGenerator[dict[str, Any], None]:
        for token in mock_response.split():
            yield {"type": "token_delta", "content": token + " "}
            await asyncio.sleep(0)

    async def close(self):
        await self.litellm_client.close()
