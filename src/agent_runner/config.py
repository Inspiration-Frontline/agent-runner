import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from agent_runner.mcps.secrets import McpSecretSnapshot

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path independently of the process working directory.

    Args:
        configured_path: Absolute filesystem path or a path relative to the Agent Runner project root.

    Returns:
        Path: The absolute configured path, or the project-root-relative path.
    """
    path: Path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_env_file() -> Path:
    """
    Determine which environment file to load based on ENVIRONMENT variable.

    Local development defaults to config/agent-runner.env so PyCharm can run
    src/main.py directly without requiring manual environment variables.

    Returns:
        Path to the appropriate .env file for the current environment.

    Supported environments:
    - local (default): config/agent-runner.env
    - dev: config/agent-runner.dev.env
    - stg: config/agent-runner.stg.env
    - prod: config/agent-runner.prod.env
    """
    env_file_override: str | None = os.getenv("AGENT_RUNNER_ENV_FILE")
    if env_file_override:
        return Path(env_file_override).expanduser().resolve()

    environment: str = os.getenv("ENVIRONMENT", "local").lower()
    env_file_map: dict[str, Path] = {
        "local": CONFIG_DIR / "agent-runner.env",
        "dev": CONFIG_DIR / "agent-runner.dev.env",
        "stg": CONFIG_DIR / "agent-runner.stg.env",
        "prod": CONFIG_DIR / "agent-runner.prod.env",
    }
    return env_file_map.get(environment, CONFIG_DIR / "agent-runner.env")


class Settings(BaseSettings):
    """
    Application settings loaded from environment files.

    This class manages all configuration for the agent-runner service,
    including server settings, service URLs, Redis configuration, and
    agent-specific parameters.

    Environment files are loaded based on the ENVIRONMENT variable:
    - config/agent-runner.env (default for local development)
    - config/agent-runner.dev.env (development environment)
    - config/agent-runner.stg.env (staging environment)
    - config/agent-runner.prod.env (production environment)
    """

    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_name: str = "agent-runner"
    """Application name used in logs, health metadata, and tracing resources."""
    debug: bool = Field(default=False, validation_alias="AGENT_RUNNER_DEBUG")
    """Enables development diagnostics and verbose local behavior."""

    # Server configuration
    server_host: str = Field(default="0.0.0.0", validation_alias="SERVER_HOST")
    """Host address on which the HTTP server listens."""
    server_port: int = Field(default=8001, validation_alias="SERVER_PORT")
    """TCP port on which the HTTP server listens."""
    open_browser_on_startup: bool = Field(default=True, validation_alias="OPEN_BROWSER_ON_STARTUP")
    """Whether local startup opens the interactive API documentation."""

    # LiteLLM gateway configuration
    lite_llm_base_url: str = "http://localhost:4000"
    """Base URL of the LiteLLM gateway used for model calls."""
    lite_llm_api_key: str = "sk-agent-breaker-local"
    """API key sent to the local LiteLLM gateway; never returned by debug endpoints."""
    lite_llm_request_timeout_seconds: float = Field(default=120.0, validation_alias="LITE_LLM_REQUEST_TIMEOUT_SECONDS")
    """Maximum duration of one LiteLLM request."""
    lite_llm_max_retries: int = Field(default=0, validation_alias="LITE_LLM_MAX_RETRIES")
    """Number of provider retries delegated to the LiteLLM client."""

    # Service URLs for downstream dependencies
    agent_config_center_url: str = "http://localhost:8081"
    """HTTP base URL of the Agent Configuration Center."""
    conversation_service_url: str = "http://localhost:8082"
    """HTTP base URL of Conversation Manager."""
    conversation_rpc_url: str = Field(default="tri://127.0.0.1:20880", validation_alias="CONVERSATION_RPC_URL")
    """Dubbo-compatible RPC endpoint used for durable Conversation operations."""
    default_agent_id: int = Field(default=1, validation_alias="DEFAULT_AGENT_ID")
    """Agent definition ID selected when a request omits an explicit internal choice."""
    user_profiler_url: str = "http://localhost:8083"
    """HTTP base URL of the User Profiler service."""
    knowledge_service_url: str = "http://localhost:8084"
    """HTTP base URL of the Knowledge Manager service."""

    # Local agent configuration settings
    local_agent_config_enabled: bool = True
    """Whether local Agent definitions may be used as a development fallback."""
    local_agent_config_path: str = str(CONFIG_DIR / "agents.json")
    """Path to the local Agent definition document."""
    mcp_catalog_path: str = Field(default=str(CONFIG_DIR / "mcp-servers.json"), validation_alias="MCP_CATALOG_PATH")
    """Path to the local MCP Server catalog document."""
    mcp_catalog_json: str = Field(default="", validation_alias="MCP_CATALOG_JSON")
    """Inline MCP catalog JSON, when supplied instead of a file."""
    mcp_pool_max_connections_per_server: int = Field(
        default=4,
        validation_alias="MCP_POOL_MAX_CONNECTIONS_PER_SERVER",
        ge=1,
        le=32,
    )
    """Maximum live MCP connections retained for one credential identity."""
    mcp_pool_idle_timeout_seconds: float = Field(
        default=300.0,
        validation_alias="MCP_POOL_IDLE_TIMEOUT_SECONDS",
        gt=0,
        le=3600,
    )
    """Idle duration after which an available MCP connection is evicted."""
    mcp_pool_borrow_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="MCP_POOL_BORROW_TIMEOUT_SECONDS",
        gt=0,
        le=120,
    )
    """Maximum duration a request waits to borrow an MCP connection."""

    # Context and output token limits
    max_context_tokens: int = 128000
    """Maximum estimated input context size accepted by the Runner."""
    max_output_tokens: int = 4096
    """Maximum output token budget passed to the model adapter."""
    file_preparation_timeout_seconds: float = Field(default=120.0, validation_alias="FILE_PREPARATION_TIMEOUT_SECONDS")
    """Maximum time spent waiting for selected files to become ready."""

    # Redis configuration for caching
    redis_host: str = Field(default="localhost", validation_alias="REDIS_HOST")
    """Redis host used for Conversation leases and optional Agent caching."""
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT")
    """Redis TCP port used for Conversation leases and optional Agent caching."""
    redis_password: str = Field(default="", validation_alias="REDIS_PASSWORD")
    """Redis authentication secret, kept out of diagnostics and traces."""
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    """Redis logical database index."""
    redis_socket_connect_timeout_seconds: float = Field(
        default=1.0, validation_alias="REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS"
    )
    """Timeout for establishing a Redis socket connection."""
    redis_socket_timeout_seconds: float = Field(default=1.0, validation_alias="REDIS_SOCKET_TIMEOUT_SECONDS")
    """Timeout for Redis socket operations."""
    agent_config_cache_enabled: bool = Field(default=False, validation_alias="AGENT_CONFIG_CACHE_ENABLED")
    """Whether Agent definitions are cached in Redis."""
    agent_config_cache_ttl_seconds: int = Field(default=300, validation_alias="AGENT_CONFIG_CACHE_TTL_SECONDS")
    """Lifetime of an Agent definition in the Redis cache."""

    # Nacos configuration center settings
    nacos_enabled: bool = Field(default=False, validation_alias="NACOS_ENABLED")
    """Whether Nacos is consulted for configuration and dynamic refresh."""
    nacos_server_address: str = Field(default="127.0.0.1:8848", validation_alias="NACOS_SERVER_ADDRESS")
    """Nacos client host and port."""
    nacos_namespace: str = Field(default="agent-breaker-local", validation_alias="NACOS_NAMESPACE")
    """Nacos namespace containing this environment's configuration."""
    nacos_data_id: str = Field(default="agent-runner.yaml", validation_alias="NACOS_DATA_ID")
    """Nacos data ID loaded by the Runner."""
    nacos_group: str = Field(default="DEFAULT_GROUP", validation_alias="NACOS_GROUP")
    """Nacos group containing the Runner data ID."""
    nacos_username: str = Field(default="nacos", validation_alias="NACOS_USERNAME")
    """Nacos username used by the local configuration client."""
    nacos_password: str = Field(default="nacos", validation_alias="NACOS_PASSWORD")
    """Nacos password, kept out of logs and public configuration projections."""

    # Environment and debug settings
    environment: str = Field(default="local", validation_alias="ENVIRONMENT")
    """Deployment environment name used to select local configuration files."""
    debug_endpoints_enabled: bool = Field(default=True, validation_alias="DEBUG_ENDPOINTS_ENABLED")
    """Whether development-only configuration and metrics endpoints are registered."""

    # OpenTelemetry tracing. Every environment defaults to 100% sampling; deployments can lower
    # the ratio without changing code when volume or retention requirements change.
    otel_enabled: bool = Field(default=True, validation_alias="OTEL_ENABLED")
    """Whether OpenTelemetry spans and metrics integrations are enabled."""
    otel_service_name: str = Field(default="agent-runner", validation_alias="OTEL_SERVICE_NAME")
    """Service name attached to exported OpenTelemetry resources."""
    otel_exporter_otlp_endpoint: str = Field(
        default="http://127.0.0.1:4317", validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    """OTLP gRPC endpoint receiving exported spans."""
    otel_sampling_ratio: float = Field(default=1.0, validation_alias="OTEL_TRACES_SAMPLER_ARG", ge=0.0, le=1.0)
    """Trace sampling ratio applied by the application tracer provider."""
    # Content capture is opt-in because prompts and Tool payloads can contain business data.
    otel_capture_content: bool = Field(default=False, validation_alias="OTEL_CAPTURE_CONTENT")
    """Whether bounded request content may be captured after redaction."""
    otel_content_max_chars: int = Field(
        default=16_384,
        validation_alias="OTEL_CONTENT_MAX_CHARS",
        ge=256,
        le=131_072,
    )
    """Maximum characters retained for opt-in trace content capture."""

    # Built-in Tool network safety settings. Persisted Tool results are not semantically trimmed;
    # the byte limit protects the HTTP client from unbounded remote responses.
    tool_http_timeout_seconds: float = Field(default=15.0, validation_alias="TOOL_HTTP_TIMEOUT_SECONDS")
    """Timeout for built-in Tool HTTP calls."""
    web_search_max_results: int = Field(default=5, validation_alias="WEB_SEARCH_MAX_RESULTS")
    """Maximum search results returned by the built-in Web Search Tool."""
    web_fetch_max_bytes: int = Field(default=2_000_000, validation_alias="WEB_FETCH_MAX_BYTES")
    """Maximum response bytes read by the built-in page fetcher."""
    web_fetch_max_redirects: int = Field(default=3, validation_alias="WEB_FETCH_MAX_REDIRECTS")
    """Maximum redirects followed by the built-in page fetcher."""


class ConfigurationManager:
    """Configuration manager that merges local and Nacos configurations.

    Nacos configuration priority:
    - If Nacos is enabled and has a value, use Nacos value
    - Otherwise, use the value from local configuration file

    Dynamic refresh is handled by nacos_config.py listener updating the cache.
    This manager reads from the cache on each get_settings() call.

    Attributes:
        _base_settings: File-backed settings snapshot used as the lower-priority merge base.
        _nacos_loader: Loader responsible for resolving nacos state.
    """

    def __init__(self, base_settings: Settings) -> None:
        """Create a settings manager with the local settings snapshot as its merge base.

        Args:
            base_settings: File-backed settings snapshot used as the lower-priority merge base.
        """
        self._base_settings = base_settings
        self._nacos_loader: Any | None = None

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
            from agent_runner.nacos_config import NacosConfigLoader

            self._nacos_loader = NacosConfigLoader.from_settings(self._base_settings)
            await self._nacos_loader.initialize()
            logger.info(
                "Nacos config client initialized: data_id=%s, group=%s",
                self._nacos_loader.data_id,
                self._nacos_loader.group,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Nacos: {e}, using local configuration only")

        return self.get_settings()

    @staticmethod
    def _merge_settings(base: Settings, nacos_config: dict[str, Any]) -> Settings:
        """Merge local settings with Nacos configuration (Nacos values override).

        Args:
            base: Local settings snapshot used as the merge base.
            nacos_config: Higher-priority Nacos configuration values.

        Returns:
            Merged local settings with Nacos configuration (Nacos values override).
        """
        field_mapping: dict[str, dict[str, str]] = {
            "server": {"host": "server_host", "port": "server_port"},
            "lite_llm": {
                "base_url": "lite_llm_base_url",
                "api_key": "lite_llm_api_key",
                "request_timeout_seconds": "lite_llm_request_timeout_seconds",
                "max_retries": "lite_llm_max_retries",
            },
            "services": {
                "agent_config_center_url": "agent_config_center_url",
                "conversation_service_url": "conversation_service_url",
                "conversation_rpc_url": "conversation_rpc_url",
                "user_profiler_url": "user_profiler_url",
                "knowledge_service_url": "knowledge_service_url",
            },
            "agent": {"default_agent_id": "default_agent_id"},
            "local_agent_config": {"enabled": "local_agent_config_enabled", "path": "local_agent_config_path"},
            "mcp": {
                "catalog_path": "mcp_catalog_path",
                "catalog_json": "mcp_catalog_json",
                "pool_max_connections_per_server": "mcp_pool_max_connections_per_server",
                "pool_idle_timeout_seconds": "mcp_pool_idle_timeout_seconds",
                "pool_borrow_timeout_seconds": "mcp_pool_borrow_timeout_seconds",
            },
            "context": {"max_context_tokens": "max_context_tokens", "max_output_tokens": "max_output_tokens"},
            "redis": {
                "host": "redis_host",
                "port": "redis_port",
                "password": "redis_password",
                "db": "redis_db",
                "socket_connect_timeout_seconds": "redis_socket_connect_timeout_seconds",
                "socket_timeout_seconds": "redis_socket_timeout_seconds",
            },
            "cache": {"agent_config_ttl_seconds": "agent_config_cache_ttl_seconds"},
            "agent_config_cache": {
                "enabled": "agent_config_cache_enabled",
                "ttl_seconds": "agent_config_cache_ttl_seconds",
            },
            "debug": {"endpoints_enabled": "debug_endpoints_enabled"},
            "observability": {
                "otel_enabled": "otel_enabled",
                "otel_service_name": "otel_service_name",
                "otel_exporter_otlp_endpoint": "otel_exporter_otlp_endpoint",
                "otel_sampling_ratio": "otel_sampling_ratio",
                "otel_capture_content": "otel_capture_content",
                "otel_content_max_chars": "otel_content_max_chars",
            },
        }

        updates: dict[str, Any] = {}
        for nacos_section, field_map in field_mapping.items():
            section_config: Any = nacos_config.get(nacos_section, {})
            if isinstance(section_config, dict):
                for nacos_key, field_name in field_map.items():
                    if nacos_key in section_config:
                        updates[field_name] = section_config[nacos_key]

        for key in ["app_name", "debug", "environment"]:
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
            if self._nacos_loader is not None and self._nacos_loader.cached_config:
                return self._merge_settings(self._base_settings, self._nacos_loader.cached_config)
        except Exception:
            logger.exception("Failed to merge the current Nacos configuration")

        return self._base_settings

    def get_mcp_secret_snapshot(self) -> "McpSecretSnapshot":
        """Return the latest typed MCP Secret snapshot without merging it into Settings.

        Secret values remain in the outbound credential boundary and therefore cannot appear in
        Settings serialization or the debug configuration response.

        Returns:
            Latest typed MCP Secret snapshot without merging it into Settings.
        """
        from agent_runner.mcps.secrets import McpSecretSnapshot

        if self._nacos_loader is None:
            return McpSecretSnapshot.create({}, configuration_revision=0)

        cached_config: Any = self._nacos_loader.cached_config
        mcp_config: Any = cached_config.get("mcp", {})
        raw_secrets: Any | dict[Any, Any] = mcp_config.get("secrets", {}) if isinstance(mcp_config, dict) else {}
        if not isinstance(raw_secrets, dict):
            raise ValueError("Nacos mcp.secrets must be a YAML object")

        configuration_revision: Any | int = getattr(self._nacos_loader, "configuration_revision", 0)
        return McpSecretSnapshot.create(raw_secrets, configuration_revision)

    async def close(self) -> None:
        """Release the loader owned by this application-scoped manager."""
        if self._nacos_loader is not None:
            await self._nacos_loader.close()
            self._nacos_loader = None


def get_settings() -> Settings:
    """Load a fresh file-backed settings snapshot for standalone component use.

    Returns:
        load a fresh file-backed settings snapshot for standalone component use.
    """
    return Settings()


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
    """Whether user profile context is included in model input."""
    rag: bool = True
    """Whether retrieved knowledge context is included in model input."""


class ConversationReferenceRequest(BaseModel):
    """Frozen source Conversation boundary selected by the browser."""

    model_config = ConfigDict(extra="forbid")

    source_conversation_id: str = Field(min_length=1, max_length=64, pattern=r"^conv_[A-Za-z0-9_-]+$")
    """Owned source Conversation whose history is referenced."""
    source_end_round_number: int = Field(gt=0)
    """Inclusive completed Round boundary frozen for the reference."""


class ConversationRequest(BaseModel):
    """
    Conversation request model for agent interactions.

    Represents one Conversation request to an agent, including context and the message to
    process. ``file_ids`` are stable references selected by the browser; Agent Runner sends them to
    Conversation Manager for authorization/preparation and never interprets them as file bytes.

    Attributes:
        agent_id: Unique identifier of the agent to invoke.
        version: Optional version of the agent configuration to use.
                If None, the latest version will be loaded.
        conversation_id: Optional conversation ID for continuing an existing conversation.
                        If None, a new conversation will be started.
        file_ids: Optional immutable attachment IDs to freeze and prepare before model execution.
        message: The user's message content to process.
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=64, pattern=r"^conv_[A-Za-z0-9_-]+$")
    """Destination Conversation receiving the new request."""
    message: str = Field(default="", max_length=100_000)
    """Visible user text; may be empty when attachments are supplied."""
    file_ids: list[str] = Field(default_factory=list, max_length=5)
    """Stable uploaded file IDs selected for this request."""
    # TODO: Move this limit to Nacos and expose the same effective value to Conversation Manager,
    # Agent Runner, and UI instead of maintaining independent validation constants.
    references: list[ConversationReferenceRequest] = Field(default_factory=list, max_length=10)
    """Frozen same-Group source Conversation references used as context."""
    ui_locale: Literal["zh-CN", "en-US"] = "zh-CN"
    """Locale used for attachment-only system instructions."""

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        """Normalize message whitespace before request-level validation runs.

        Args:
            value: Candidate value to validate, normalize, or serialize.

        Returns:
            Normalized message whitespace before request-level validation runs.
        """
        return value.strip()

    @field_validator("file_ids")
    @classmethod
    def validate_file_ids(cls, value: list[str]) -> list[str]:
        """Reject duplicate or blank stable attachment IDs supplied by the caller.

        Args:
            value: Candidate value to validate, normalize, or serialize.

        Returns:
            Validated duplicate or blank stable attachment IDs supplied by the caller.
        """
        normalized: list[str] = [file_id.strip() for file_id in value]
        if any(not file_id for file_id in normalized):
            raise ValueError("file_ids cannot contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("file_ids must be unique")
        return normalized

    @field_validator("references")
    @classmethod
    def validate_references(cls, value: list[ConversationReferenceRequest]) -> list[ConversationReferenceRequest]:
        """Reject duplicate source Conversations before any downstream RPC.

        Args:
            value: Candidate value to validate, normalize, or serialize.

        Returns:
            Validated duplicate source Conversations before any downstream RPC.
        """
        source_ids: list[str] = [reference.source_conversation_id for reference in value]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("referenced Conversations must be unique")
        return value

    @model_validator(mode="after")
    def require_message_or_files(self) -> "ConversationRequest":
        """Require visible text unless the request contains at least one attachment.

        Returns:
            The validated request after enforcing text-or-attachment presence.
        """
        if not self.message and not self.file_ids:
            raise ValueError("message or file_ids is required")
        if any(reference.source_conversation_id == self.conversation_id for reference in self.references):
            raise ValueError("a Conversation cannot reference itself")
        return self


class CancelConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=64, pattern=r"^conv_[A-Za-z0-9_-]+$")
    """Conversation whose active generation should be cancelled."""


class AgentConfig(BaseModel):
    """
    Complete agent configuration model.

    Contains all configuration parameters needed to instantiate and run an agent,
    including model settings, prompts, tools, MCP servers, and runtime parameters.

    Attributes:
        agent_id: Unique identifier of the agent.
        version: Version string of this configuration.
        model: The LLM model identifier to use (e.g., "Qwen/Qwen3-235B-A22B-Instruct-2507").
        system_prompt: The system prompt that defines the agent's behavior and personality.
        tools: List of tool identifiers available to this agent.
        mcp_servers: List of MCP server identifiers this agent can connect to.
        memory_policy: Policy determining which context sources to include.
        max_output_tokens: Maximum number of tokens in the agent's response.
        temperature: Sampling temperature for response generation (0.0 to 2.0).
    """

    agent_id: int
    """Stable Agent definition identifier."""
    version: int
    """Version of the Agent definition."""
    name: str
    """Display and handoff name of the Agent."""
    model: str
    """Provider model identifier selected for execution."""
    system_prompt: str
    """Base system instruction supplied to the model."""
    tools: list[str] = Field(default_factory=list)
    """Globally unique built-in Tool keys enabled for the Agent."""
    mcp_servers: list["MCPServerBindingConfig"] = Field(default_factory=list)
    """MCP Server bindings available to the Agent."""
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    """Profile and retrieval context policy for this Agent."""
    max_output_tokens: int = 4096
    """Maximum output tokens requested from the provider."""
    temperature: float = 0.7
    """Provider sampling temperature."""


class MCPServerBindingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    """Stable MCP Server catalog identifier."""
    required: bool = True
    """Whether this Server must connect for the Agent request to proceed."""
