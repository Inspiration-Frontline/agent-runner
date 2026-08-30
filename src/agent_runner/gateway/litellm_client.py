import logging

from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import Model

from agent_runner.config import Settings
from agent_runner.observability.litellm_tracing import TracedModel
from agent_runner.observability.tracing import Tracer

logger: logging.Logger = logging.getLogger(__name__)


class LiteLLMModelFactory:
    """Factory for OpenAI Agents SDK models backed by the LiteLLM proxy.

    The Agents SDK owns the agent loop and stream parsing. LiteLLM is used only as
    the model integration layer, and the configured external LiteLLM proxy remains
    the gateway that forwards provider requests.

    Attributes:
        _settings: Effective application settings retained for this component.
        _tracer: Application-owned OpenTelemetry facade used by this component.
        base_url: Absolute URL for base.
        api_key: Credential passed only to the configured provider boundary.
        request_timeout_seconds: Maximum duration in seconds for one provider request.
    """

    DEFAULT_PROVIDER_PREFIX: str = "openai/"
    """Provider prefix applied when a configured model omits one."""
    KNOWN_PROVIDER_PREFIXES: frozenset[str] = frozenset(
        {
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
    )
    """Provider prefixes recognized as already qualified model identifiers."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout_seconds: float | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Configure the LiteLLM proxy endpoint, credentials, and request timeout.

        Args:
            base_url: Absolute URL for base.
            api_key: Credential passed only to the configured provider boundary.
            request_timeout_seconds: Maximum duration in seconds for one provider request.
            settings: Effective application settings for the operation.
            tracer: Application-owned OpenTelemetry facade.
        """
        current_settings: Settings = settings or Settings()
        self._settings = current_settings
        self._tracer = tracer or Tracer(current_settings.otel_service_name)
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
        delegate: LitellmModel = LitellmModel(
            model=normalized_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return TracedModel(delegate, normalized_model, self._settings, self._tracer)

    def _normalize_model(self, model: str) -> str:
        """Ensure LiteLLM can resolve a provider for proxy-routed model names.

        Args:
            model: Provider model identifier.

        Returns:
            Model identifier with an explicit provider prefix when one was not supplied.
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
