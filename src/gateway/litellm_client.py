import logging
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


class LiteLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
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
        await self.client.close()
