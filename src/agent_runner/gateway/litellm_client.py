import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import Model
from opentelemetry.trace import SpanKind

from agent_runner.config import Settings, get_settings
from agent_runner.observability.tracing import Span, get_tracer, inject_trace_context, trace_json

logger: logging.Logger = logging.getLogger(__name__)


class TracedModel(Model):
    """Adds one authoritative OpenTelemetry span around each SDK model invocation."""

    def __init__(self, delegate: Model, model_name: str) -> None:
        self._delegate = delegate
        self.model = model_name
        self.base_url = getattr(delegate, "base_url", None)
        self.api_key = getattr(delegate, "api_key", None)

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke a non-streaming model request with W3C propagation."""
        with get_tracer().span("llm.call", self._span_attributes(), kind=SpanKind.CLIENT) as span:
            traced_args, traced_kwargs = self._inject_model_headers(args, kwargs)
            self._record_request(span, traced_args, traced_kwargs)
            response = await self._delegate.get_response(*traced_args, **traced_kwargs)
            self._record_response(
                span,
                response,
                self._argument(traced_args, traced_kwargs, "model_settings", 2),
            )
            return response

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Return a stream whose complete network lifetime is covered by ``llm.call``."""
        return self._stream_response(args, kwargs)

    async def _stream_response(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> AsyncIterator[Any]:
        with get_tracer().span("llm.call", self._span_attributes(), kind=SpanKind.CLIENT) as span:
            traced_args, traced_kwargs = self._inject_model_headers(args, kwargs)
            self._record_request(span, traced_args, traced_kwargs)
            model_settings = self._argument(traced_args, traced_kwargs, "model_settings", 2)
            async for event in self._delegate.stream_response(*traced_args, **traced_kwargs):
                if getattr(event, "type", "") == "response.completed":
                    self._record_response(span, getattr(event, "response", None), model_settings)
                yield event

    def get_retry_advice(self, request: Any) -> Any:
        """Preserve provider-specific retry guidance from the wrapped SDK model."""
        return self._delegate.get_retry_advice(request)

    async def close(self) -> None:
        """Release resources owned by the wrapped SDK model."""
        await self._delegate.close()

    def _span_attributes(self) -> dict[str, str]:
        return {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": self.model,
            "server.address": "litellm",
        }

    def _record_request(self, span: Span, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        model_settings = self._argument(args, kwargs, "model_settings", 2)
        model_input = self._argument(args, kwargs, "input", 1)
        system_instructions = self._argument(args, kwargs, "system_instructions", 0)
        tools = self._argument(args, kwargs, "tools", 3)
        if model_settings is not None:
            span.set_attribute("gen_ai.request.max_tokens", getattr(model_settings, "max_tokens", None))
            span.set_attribute("gen_ai.request.temperature", getattr(model_settings, "temperature", None))
        if isinstance(tools, Sequence):
            span.set_attribute("gen_ai.request.tool_count", len(tools))

        settings = get_settings()
        if not settings.otel_capture_content:
            return
        span.set_attribute(
            "gen_ai.system_instructions",
            trace_json(system_instructions, settings.otel_content_max_chars),
        )
        span.set_attribute("gen_ai.input.messages", trace_json(model_input, settings.otel_content_max_chars))

    def _record_response(self, span: Span, response: Any, model_settings: Any) -> None:
        if response is None:
            return
        span.set_attribute("gen_ai.response.id", getattr(response, "response_id", None))
        span.set_attribute("gen_ai.response.model", getattr(response, "model", None))
        span.set_attribute("gen_ai.response.status", getattr(response, "status", None))
        usage = getattr(response, "usage", None)
        if usage is not None:
            span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "input_tokens", None))
            span.set_attribute("gen_ai.usage.output_tokens", getattr(usage, "output_tokens", None))
            span.set_attribute("gen_ai.usage.total_tokens", getattr(usage, "total_tokens", None))
        span.set_attribute("gen_ai.response.finish_reason", self._finish_reason(response, model_settings))

        settings = get_settings()
        if settings.otel_capture_content:
            span.set_attribute(
                "gen_ai.output.messages",
                trace_json(getattr(response, "output", response), settings.otel_content_max_chars),
            )

    @staticmethod
    def _finish_reason(response: Any, model_settings: Any) -> str:
        usage = getattr(response, "usage", None)
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        max_tokens = getattr(model_settings, "max_tokens", None)
        if max_tokens is not None and output_tokens >= max_tokens:
            return "length"
        output = getattr(response, "output", ()) or ()
        if any(getattr(item, "type", "") in {"function_call", "tool_call"} for item in output):
            return "tool_calls"
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None)
        return str(incomplete_reason or "stop")

    @staticmethod
    def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int) -> Any:
        if name in kwargs:
            return kwargs[name]
        return args[index] if len(args) > index else None

    @staticmethod
    def _inject_model_headers(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        headers: dict[str, str] = {}
        inject_trace_context(headers)
        traced_kwargs = dict(kwargs)
        model_settings = traced_kwargs.get("model_settings")
        if model_settings is not None:
            traced_kwargs["model_settings"] = TracedModel._merge_headers(model_settings, headers)
            return args, traced_kwargs
        if len(args) <= 2:
            return args, traced_kwargs
        traced_args = list(args)
        traced_args[2] = TracedModel._merge_headers(traced_args[2], headers)
        return tuple(traced_args), traced_kwargs

    @staticmethod
    def _merge_headers(model_settings: Any, trace_headers: dict[str, str]) -> Any:
        existing_headers = dict(model_settings.extra_headers or {})
        existing_headers.update(trace_headers)
        return replace(model_settings, extra_headers=existing_headers)


class LiteLLMModelFactory:
    """
    Factory for OpenAI Agents SDK models backed by the LiteLLM proxy.

    The Agents SDK owns the agent loop and stream parsing. LiteLLM is used only as
    the model integration layer, and the configured external LiteLLM proxy remains
    the gateway that forwards provider requests.
    """

    DEFAULT_PROVIDER_PREFIX: str = "openai/"
    KNOWN_PROVIDER_PREFIXES: set[str] = {
        "ai21",
        "aleph_alpha",
        "anthropic",
        "azure",
        "bedrock",
        "cohere",
        "deepseek",
        "gemini",
        "groq",
        "huggingface",
        "mistral",
        "ollama",
        "openai",
        "openrouter",
        "perplexity",
        "replicate",
        "vertex_ai",
        "vllm",
        "watsonx",
    }

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> None:
        """Configure the LiteLLM proxy endpoint, credentials, and request timeout."""
        current_settings: Settings = get_settings()
        self.base_url: str = base_url or current_settings.lite_llm_base_url
        self.api_key: str = api_key or current_settings.lite_llm_api_key or "sk-agent-breaker-local"
        self.request_timeout_seconds: float = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else current_settings.lite_llm_request_timeout_seconds
        )

    def create_model(self, model: str) -> Model:
        """
        Create an Agents SDK LiteLLM model for the configured proxy.

        Args:
            model: Agent model identifier. Bare model names are treated as
                OpenAI-compatible model names served by the external LiteLLM proxy.

        Returns:
            LitellmModel: A model implementation consumable by Agents SDK Agent.
        """
        normalized_model: str = self._normalize_model(model)
        logger.info("Using LiteLLM proxy model: %s", normalized_model)
        delegate = LitellmModel(
            model=normalized_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return TracedModel(delegate, normalized_model)

    def _normalize_model(self, model: str) -> str:
        """
        Ensure LiteLLM can resolve a provider for proxy-routed model names.
        """
        provider_prefix: str = model.split("/", 1)[0].lower()
        if provider_prefix in self.KNOWN_PROVIDER_PREFIXES:
            return model

        logger.debug("Treating bare model %s as OpenAI-compatible LiteLLM proxy model.", model)
        return f"{self.DEFAULT_PROVIDER_PREFIX}{model}"

    async def close(self) -> None:
        """
        Kept for lifecycle symmetry with other gateway clients.
        """
        pass
