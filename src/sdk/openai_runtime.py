import asyncio
import logging
from typing import Any, AsyncGenerator

from agents.config_models import AgentDefinition
from context.builder import AgentContext
from gateway.litellm_client import LiteLLMClient
from runtime.cancellation import CancellationToken

logger = logging.getLogger(__name__)


class OpenAIAgentsRuntime:
    """
    Runtime for executing agents using OpenAI Agents SDK patterns.

    Provides methods to execute agent requests through LiteLLM,
    supporting both streaming and non-streaming responses,
    with cancellation support and mock response fallback.

    Attributes:
        litellm_client: LiteLLM client for model interactions.
    """

    def __init__(self):
        """
        Initialize the OpenAI Agents runtime with LiteLLM client.
        """
        self.litellm_client = LiteLLMClient()

    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Execute an agent request with streaming response.

        Args:
            agent: The agent definition to execute.
            context: The execution context for the agent.
            cancellation_token: Optional token for request cancellation.

        Yields:
            dict[str, Any]: Stream events containing token deltas, tool calls, or errors.
        """
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
        """
        Execute an agent request with non-streaming response.

        Args:
            agent: The agent definition to execute.
            context: The execution context for the agent.
            cancellation_token: Optional token for request cancellation.

        Returns:
            dict[str, Any: The complete response from the agent.
        """
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
        """
        Build the message list for model invocation.

        Args:
            agent: The agent definition.
            context: The execution context.

        Returns:
            list[dict[str, Any]]: List of messages in OpenAI format.
        """
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
        """
        Build the tool specifications for model invocation.

        Args:
            context: The execution context containing tool specs.

        Returns:
            list[dict[str, Any]]: List of tool specifications in OpenAI format.
        """
        return context.tool_specs

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """
        Parse the model response into a structured dictionary.

        Args:
            response: The raw response from LiteLLM.

        Returns:
            dict[str, Any]: Parsed response with content, role, and optional tool calls.
        """
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
        """
        Stream a mock response for testing purposes.

        Args:
            mock_response: The mock response text to stream.

        Yields:
            dict[str, Any]: Token delta events for the mock response.
        """
        for token in mock_response.split():
            yield {"type": "token_delta", "content": token + " "}
            await asyncio.sleep(0)

    async def close(self):
        """
        Close the LiteLLM client connection.
        """
        await self.litellm_client.close()
