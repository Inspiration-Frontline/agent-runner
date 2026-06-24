import logging
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


class LiteLLMClient:
    """
    Client for LiteLLM model gateway interactions.

    Provides methods to interact with the LiteLLM proxy server for
    chat completions and streaming responses, supporting tool calls.

    Attributes:
        base_url: Base URL for the LiteLLM proxy server.
        api_key: API key for LiteLLM authentication.
        client: Async OpenAI client configured for LiteLLM.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the LiteLLM client.

        Args:
            base_url: Optional override for LiteLLM base URL.
            api_key: Optional override for LiteLLM API key.
        """
        self.base_url = base_url or settings.lite_llm_base_url
        self.api_key = api_key or settings.lite_llm_api_key or "sk-agent-breaker-local"
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Any:
        """
        Execute a non-streaming chat completion request.

        Args:
            model: Model identifier to use for completion.
            messages: List of conversation messages.
            tools: Optional list of tool specifications.
            tool_choice: Optional tool choice strategy.
            temperature: Sampling temperature parameter.
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional completion parameters.

        Returns:
            Any: The completion response from LiteLLM.
        """
        logger.info(f"Calling LiteLLM with model: {model}")

        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if tools:
            completion_kwargs["tools"] = tools
        if tool_choice:
            completion_kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**completion_kwargs)
        return response

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Execute a streaming chat completion request.

        Args:
            model: Model identifier to use for completion.
            messages: List of conversation messages.
            tools: Optional list of tool specifications.
            tool_choice: Optional tool choice strategy.
            temperature: Sampling temperature parameter.
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional completion parameters.

        Yields:
            dict[str, Any]: Stream chunks parsed into event dictionaries.
        """
        logger.info(f"Calling LiteLLM stream with model: {model}")

        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        if tools:
            completion_kwargs["tools"] = tools
        if tool_choice:
            completion_kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**completion_kwargs)

        async for chunk in response:
            yield self._parse_chunk(chunk)

    def _parse_chunk(self, chunk: Any) -> dict[str, Any]:
        """
        Parse a stream chunk into an event dictionary.

        Args:
            chunk: Raw chunk from the streaming response.

        Returns:
            dict[str, Any]: Parsed event with type and content/tool data.
        """
        if not chunk.choices:
            return {"type": "unknown", "content": ""}

        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        tool_calls = getattr(delta, "tool_calls", None)

        if content:
            return {
                "type": "token_delta",
                "content": content,
            }
        elif tool_calls:
            return {
                "type": "tool_start",
                "tool": tool_calls[0].function.name,
                "args": tool_calls[0].function.arguments,
            }

        return {"type": "unknown", "content": ""}

    async def close(self):
        """
        Close the OpenAI client connection.
        """
        await self.client.close()
