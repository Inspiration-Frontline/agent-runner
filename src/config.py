from pathlib import Path
import os
import logging

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def get_env_file() -> Path:
    """
    Determine which environment file to load based on ENVIRONMENT variable.

    Returns:
        Path to the appropriate .env file for the current environment.

    Supported environments:
    - local (default): .env.local
    - dev: .env.dev
    - stg: .env.stg
    - prod: .env.prod
    """
    environment = os.getenv("ENVIRONMENT", "local").lower()
    env_file_map = {
        "local": ".env.local",
        "dev": ".env.dev",
        "stg": ".env.stg",
        "prod": ".env.prod",
    }
    env_filename = env_file_map.get(environment, ".env.local")
    return PROJECT_ROOT / env_filename


class Settings(BaseSettings):
    """
    Application settings loaded from environment files.

    This class manages all configuration for the agent-runner service,
    including server settings, service URLs, Redis configuration, and
    agent-specific parameters.

    Environment files are loaded based on the ENVIRONMENT variable:
    - .env.local (default for local development)
    - .env.dev (development environment)
    - .env.stg (staging environment)
    - .env.prod (production environment)
    """

    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_name: str = "agent-runner"
    debug: bool = Field(default=False, validation_alias="AGENT_RUNNER_DEBUG")

    # Server configuration
    server_host: str = Field(default="0.0.0.0", validation_alias="SERVER_HOST")
    server_port: int = Field(default=8000, validation_alias="SERVER_PORT")

    # LiteLLM gateway configuration
    lite_llm_base_url: str = "http://localhost:4000"
    lite_llm_api_key: str = "sk-agent-breaker-local"

    # Service URLs for downstream dependencies
    agent_config_center_url: str = "http://localhost:8081"
    conversation_service_url: str = "http://localhost:8082"
    user_profiler_url: str = "http://localhost:8083"
    knowledge_service_url: str = "http://localhost:8084"

    # Local agent configuration settings
    local_agent_config_enabled: bool = True
    local_agent_config_path: str = str(CONFIG_DIR / "agents.json")

    # Context and output token limits
    max_context_tokens: int = 128000
    max_output_tokens: int = 4096

    # Redis configuration for caching
    redis_host: str = Field(default="localhost", validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT")
    redis_password: str = Field(default="", validation_alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    agent_config_cache_ttl_seconds: int = Field(
        default=300, validation_alias="AGENT_CONFIG_CACHE_TTL_SECONDS"
    )

    # Nacos configuration center settings
    nacos_enabled: bool = Field(default=True, validation_alias="NACOS_ENABLED")
    nacos_server_address: str = Field(default="127.0.0.1:8848", validation_alias="NACOS_SERVER_ADDRESS")
    nacos_namespace: str = Field(default="agent-breaker-local", validation_alias="NACOS_NAMESPACE")
    nacos_data_id: str = Field(default="agent-runner.yaml", validation_alias="NACOS_DATA_ID")
    nacos_group: str = Field(default="DEFAULT_GROUP", validation_alias="NACOS_GROUP")
    nacos_username: str = Field(default="nacos", validation_alias="NACOS_USERNAME")
    nacos_password: str = Field(default="nacos", validation_alias="NACOS_PASSWORD")


# Base settings loaded from local environment files (before Nacos merge)
_base_settings = Settings()


class ConfigurationManager:
    """
    Configuration manager that merges local and Nacos configurations.

    Nacos configuration priority:
    - If Nacos is enabled and has a value, use Nacos value
    - Otherwise, use the value from local configuration file

    Dynamic refresh is handled by nacos_config.py listener updating the cache.
    This manager reads from the cache on each get_settings() call.
    """

    def __init__(self, base_settings: Settings):
        self._base_settings = base_settings

    async def initialize(self) -> Settings:
        """
        Initialize by starting Nacos listener (if enabled) and loading initial config.

        Returns:
            Settings: The merged settings instance.
        """
        if not self._base_settings.nacos_enabled:
            logger.info("Nacos is disabled, using local configuration only")
            return self._base_settings

        try:
            from nacos_config import get_nacos_loader

            loader = await get_nacos_loader()
            logger.info(f"Nacos config client initialized: data_id={loader.data_id}, group={loader.group}")
        except Exception as e:
            logger.warning(f"Failed to initialize Nacos: {e}, using local configuration only")

        return self.get_settings()

    def _merge_settings(self, base: Settings, nacos_config: dict) -> Settings:
        """Merge local settings with Nacos configuration (Nacos values override)."""
        field_mapping = {
            "server": {"host": "server_host", "port": "server_port"},
            "lite_llm": {"base_url": "lite_llm_base_url", "api_key": "lite_llm_api_key"},
            "services": {
                "agent_config_center_url": "agent_config_center_url",
                "conversation_service_url": "conversation_service_url",
                "user_profiler_url": "user_profiler_url",
                "knowledge_service_url": "knowledge_service_url",
            },
            "local_agent_config": {"enabled": "local_agent_config_enabled", "path": "local_agent_config_path"},
            "context": {"max_context_tokens": "max_context_tokens", "max_output_tokens": "max_output_tokens"},
            "redis": {"host": "redis_host", "port": "redis_port", "password": "redis_password", "db": "redis_db"},
            "cache": {"agent_config_ttl_seconds": "agent_config_cache_ttl_seconds"},
        }

        updates: dict[str, any] = {}
        for nacos_section, field_map in field_mapping.items():
            section_config = nacos_config.get(nacos_section, {})
            if isinstance(section_config, dict):
                for nacos_key, field_name in field_map.items():
                    if nacos_key in section_config:
                        updates[field_name] = section_config[nacos_key]

        for key in ["app_name", "debug"]:
            if key in nacos_config:
                updates[key] = nacos_config[key]

        # Use model_copy to avoid environment variable override
        return base.model_copy(update=updates)

    def get_settings(self) -> Settings:
        """
        Get current settings, merging from Nacos cache if available.

        The nacos_config.py listener keeps the cache updated in background.
        This method reads the latest cached values on each call.

        Returns:
            Settings: The current merged settings.
        """
        if not self._base_settings.nacos_enabled:
            return self._base_settings

        try:
            from nacos_config import _nacos_loader

            if _nacos_loader and _nacos_loader._cached_config:
                return self._merge_settings(self._base_settings, _nacos_loader._cached_config)
        except Exception:
            pass

        return self._base_settings


# Global configuration manager instance
_config_manager: ConfigurationManager | None = None


def get_settings() -> Settings:
    """
    Get the current application settings.

    Returns the merged settings if configuration manager is initialized,
    otherwise returns the base settings from local configuration files.

    Returns:
        Settings: The current application settings.
    """
    if _config_manager:
        return _config_manager.get_settings()
    return _base_settings


async def initialize_settings() -> Settings:
    """
    Initialize application settings with Nacos configuration merge.

    This function should be called during application startup to ensure
    Nacos configuration is loaded and merged with local configuration.

    Returns:
        Settings: The initialized settings instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(_base_settings)
    return await _config_manager.initialize()


# For backward compatibility, expose settings as the base settings initially
# It will be updated after initialize_settings() is called
settings = _base_settings


class MemoryPolicy(BaseModel):
    """
    Memory policy configuration for agent context building.

    Determines which external context sources should be included when
    building the agent's conversation context.

    Attributes:
        profile: Whether to include user profile information in context.
        rag: Whether to include RAG (Retrieval-Augmented Generation) results in context.
    """

    profile: bool = True
    rag: bool = True


class ChatRequest(BaseModel):
    """
    Chat request model for agent interactions.

    Represents a single chat request to an agent, including the agent identity,
    conversation context, user information, and the message to process.

    Attributes:
        agent_id: Unique identifier of the agent to invoke.
        version: Optional version of the agent configuration to use.
                If None, the latest version will be loaded.
        conversation_id: Optional conversation ID for continuing an existing conversation.
                        If None, a new conversation will be started.
        user_id: Unique identifier of the user making the request.
        message: The user's message content to process.
    """

    agent_id: str
    version: str | None = None
    conversation_id: str | None = None
    user_id: str
    message: str


class AgentConfig(BaseModel):
    """
    Complete agent configuration model.

    Contains all configuration parameters needed to instantiate and run an agent,
    including model settings, prompts, tools, MCP servers, and runtime parameters.

    Attributes:
        agent_id: Unique identifier of the agent.
        version: Version string of this configuration.
        model: The LLM model identifier to use (e.g., "Qwen/Qwen3-4B").
        system_prompt: The system prompt that defines the agent's behavior and personality.
        tools: List of tool identifiers available to this agent.
        mcp_servers: List of MCP server identifiers this agent can connect to.
        memory_policy: Policy determining which context sources to include.
        max_output_tokens: Maximum number of tokens in the agent's response.
        temperature: Sampling temperature for response generation (0.0 to 2.0).
        mock_response: Optional mock response for testing purposes.
    """

    agent_id: str
    version: str
    model: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    max_output_tokens: int = 4096
    temperature: float = 0.7
    mock_response: str | None = None
