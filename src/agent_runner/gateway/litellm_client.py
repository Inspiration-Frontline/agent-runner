import logging

from agents.extensions.models.litellm_model import LitellmModel

from agent_runner.config import Settings, get_settings

logger: logging.Logger = logging.getLogger(__name__)


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

    def create_model(self, model: str) -> LitellmModel:
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
        return LitellmModel(
            model=normalized_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

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
