from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from agents.models.interface import Model
from opentelemetry.trace import SpanKind

from agent_runner.config import Settings
from agent_runner.observability.tracing import Span, Tracer, inject_trace_context, trace_json


@dataclass(frozen=True)
class _TracedModelArguments:
    """Effective positional and keyword arguments after W3C trace-header injection."""

    args: tuple[Any, ...]
    # Key: provider argument name. Value: original or trace-enriched provider argument value.
    kwargs: dict[str, Any]


class TracedModel(Model):
    """Decorate one SDK Model with AgentBreaker OpenTelemetry instrumentation."""

    def __init__(
        self,
        delegate: Model,
        model_name: str,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Wrap a provider model while preserving its public identity and close lifecycle."""
        current_settings = settings or Settings()
        self._delegate = delegate
        self._settings = current_settings
        self._tracer = tracer or Tracer(current_settings.otel_service_name)
        self.model = model_name
        self.base_url = getattr(delegate, "base_url", None)
        self.api_key = getattr(delegate, "api_key", None)

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke a non-streaming model request with W3C propagation."""
        with self._tracer.span("llm.call", self._get_span_attributes(), kind=SpanKind.CLIENT) as span:
            traced_arguments = self._inject_model_headers(args, kwargs)
            self._record_request(span, traced_arguments.args, traced_arguments.kwargs)
            response = await self._delegate.get_response(*traced_arguments.args, **traced_arguments.kwargs)
            self._record_response(
                span,
                response,
                self._get_argument(traced_arguments.args, traced_arguments.kwargs, "model_settings", 2),
            )
            return response

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Return a stream whose complete network lifetime is covered by ``llm.call``."""
        return self._stream_response(args, kwargs)

    async def _stream_response(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> AsyncIterator[Any]:
        """Keep the client span open until the provider's asynchronous stream terminates."""
        with self._tracer.span("llm.call", self._get_span_attributes(), kind=SpanKind.CLIENT) as span:
            traced_arguments = self._inject_model_headers(args, kwargs)
            self._record_request(span, traced_arguments.args, traced_arguments.kwargs)
            model_settings = self._get_argument(
                traced_arguments.args,
                traced_arguments.kwargs,
                "model_settings",
                2,
            )
            async for event in self._delegate.stream_response(*traced_arguments.args, **traced_arguments.kwargs):
                if getattr(event, "type", "") == "response.completed":
                    self._record_response(span, getattr(event, "response", None), model_settings)
                yield event

    def get_retry_advice(self, request: Any) -> Any:
        """Preserve provider-specific retry guidance from the wrapped SDK model."""
        return self._delegate.get_retry_advice(request)

    async def close(self) -> None:
        """Release resources owned by the wrapped SDK model."""
        await self._delegate.close()

    def _get_span_attributes(self) -> dict[str, str]:
        """Build stable low-cardinality attributes shared by streaming and non-streaming calls."""
        return {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": self.model,
            "server.address": "litellm",
        }

    def _record_request(self, span: Span, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Record request metadata and optionally bounded content on the active LLM span."""
        model_settings = self._get_argument(args, kwargs, "model_settings", 2)
        model_input = self._get_argument(args, kwargs, "input", 1)
        system_instructions = self._get_argument(args, kwargs, "system_instructions", 0)
        tools = self._get_argument(args, kwargs, "tools", 3)
        if model_settings is not None:
            span.set_attribute("gen_ai.request.max_tokens", getattr(model_settings, "max_tokens", None))
            span.set_attribute("gen_ai.request.temperature", getattr(model_settings, "temperature", None))
        if isinstance(tools, Sequence):
            span.set_attribute("gen_ai.request.tool_count", len(tools))

        if not self._settings.otel_capture_content:
            return
        span.set_attribute(
            "gen_ai.system_instructions",
            trace_json(system_instructions, self._settings.otel_content_max_chars),
        )
        span.set_attribute("gen_ai.input.messages", trace_json(model_input, self._settings.otel_content_max_chars))

    def _record_response(self, span: Span, response: Any, model_settings: Any) -> None:
        """Record provider identity, token usage, finish reason, and optional response content."""
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
        span.set_attribute("gen_ai.response.finish_reason", self._get_finish_reason(response, model_settings))

        if self._settings.otel_capture_content:
            span.set_attribute(
                "gen_ai.output.messages",
                trace_json(getattr(response, "output", response), self._settings.otel_content_max_chars),
            )

    @staticmethod
    def _get_finish_reason(response: Any, model_settings: Any) -> str:
        """Normalize length, Tool-call, incomplete, and ordinary completion outcomes."""
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
    def _get_argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int) -> Any:
        """Resolve one provider argument from keyword form first and positional form second."""
        if name in kwargs:
            return kwargs[name]
        return args[index] if len(args) > index else None

    @staticmethod
    def _inject_model_headers(args: tuple[Any, ...], kwargs: dict[str, Any]) -> _TracedModelArguments:
        """Return one named argument bundle with trace headers merged into model settings."""
        headers: dict[str, str] = {}
        inject_trace_context(headers)
        traced_kwargs = dict(kwargs)
        model_settings = traced_kwargs.get("model_settings")
        if model_settings is not None:
            traced_kwargs["model_settings"] = TracedModel._merge_headers(model_settings, headers)
            return _TracedModelArguments(args, traced_kwargs)
        if len(args) <= 2:
            return _TracedModelArguments(args, traced_kwargs)
        traced_args = list(args)
        traced_args[2] = TracedModel._merge_headers(traced_args[2], headers)
        return _TracedModelArguments(tuple(traced_args), traced_kwargs)

    @staticmethod
    def _merge_headers(model_settings: Any, trace_headers: dict[str, str]) -> Any:
        """Copy immutable model settings while preserving caller headers and adding trace context."""
        existing_headers = dict(model_settings.extra_headers or {})
        existing_headers.update(trace_headers)
        return replace(model_settings, extra_headers=existing_headers)
