import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.retry import ModelRetrySettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent, StreamEvent
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from agent_runner.agent_definitions.config_models import AgentDefinition
from agent_runner.config import get_settings
from agent_runner.context.builder import AgentContext
from agent_runner.gateway.litellm_client import LiteLLMModelFactory
from agent_runner.runtime.cancellation import CancellationToken

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
        self.model_factory = model_factory or LiteLLMModelFactory()

    async def run_streamed(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Execute an agent request with an SDK-managed streaming response.
        """
        sdk_agent = self._build_sdk_agent(agent, context.system_prompt)
        sdk_input = self._build_input(context)
        result = Runner.run_streamed(
            starting_agent=sdk_agent,
            input=sdk_input,
            max_turns=10,
        )

        try:
            async for event in result.stream_events():
                if cancellation_token and cancellation_token.is_cancelled():
                    logger.info("Stream cancelled by token")
                    result.cancel()
                    break

                converted = self._convert_stream_event(event)
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

    async def run(
        self,
        agent: AgentDefinition,
        context: AgentContext,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """
        Execute an agent request and return the final output.
        """
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError("Execution cancelled")

        sdk_agent = self._build_sdk_agent(agent, context.system_prompt)
        result = await Runner.run(
            starting_agent=sdk_agent,
            input=self._build_input(context),
            max_turns=10,
        )
        return {
            "content": result.final_output,
            "role": "assistant",
        }

    def _build_sdk_agent(self, agent: AgentDefinition, system_prompt: str) -> Agent:
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
            ),
            tools=[],
        )

    def _build_input(self, context: AgentContext) -> list[dict[str, str]]:
        input_items: list[dict[str, str]] = []
        for message in context.conversation_history:
            input_items.append({
                "role": message.role,
                "content": message.content,
            })

        input_items.append({
            "role": "user",
            "content": context.current_message.content,
        })
        return input_items

    def _convert_stream_event(self, event: StreamEvent) -> dict[str, Any] | None:
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent):
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
                return {
                    "type": "tool_start",
                    "tool": getattr(raw_item, "name", "") if raw_item is not None else "",
                    "args": getattr(raw_item, "arguments", None) if raw_item is not None else None,
                }

            if event.name == "tool_output":
                return {
                    "type": "tool_result",
                    "tool": getattr(event.item, "tool_name", ""),
                    "tool_result": getattr(event.item, "output", None),
                }

        return None

    def _convert_response_completed_usage(self, event_data: Any) -> dict[str, Any] | None:
        usage = getattr(getattr(event_data, "response", None), "usage", None)
        if usage is None:
            return None

        return {
            "type": "usage",
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }

    async def close(self):
        await self.model_factory.close()
