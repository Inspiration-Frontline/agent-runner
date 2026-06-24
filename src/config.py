from pathlib import Path
import os

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


settings = Settings()


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
